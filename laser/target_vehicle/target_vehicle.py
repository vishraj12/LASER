from typing import TypedDict, Literal, Type

import carla
from laser.sensor import CollisionSensor
import numpy as np
from laser.carla_agents.vehicle_decision_interpreter import VehicleDecisionInterpreter
from laser.laser_agents import Agent, spawn_actor_by_script

import os

from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.envs.sensor_interface import SensorInterface
from leaderboard.utils.route_manipulation import interpolate_trajectory

def convert_transform_to_location(transform_vec):
    """
    Convert a vector of transforms to a vector of locations
    """
    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


def _yaw_delta_deg(yaw_a, yaw_b):
    return abs((yaw_a - yaw_b + 180.0) % 360.0 - 180.0)


def _signed_yaw_delta_deg(from_yaw, to_yaw):
    return (to_yaw - from_yaw + 180.0) % 360.0 - 180.0


def ego_route_end_location(world, start_wp, distance, maneuver="straight"):
    """End pose ~distance m along the intended ego maneuver.

    Same role as stock LASER ``wp.next(d)[0]`` (automatic mission goal, not
    scripted controls). At junction forks, ``maneuver`` selects among
    successors: ``straight`` / ``left`` / ``right``.

    CARLA is left-handed: yaw grows clockwise seen from above, so a successor
    whose yaw is GREATER than the start's is to the RIGHT (the global route
    planner labels that turn ``RoadOption.RIGHT``). The branches below used to
    test the opposite sign, so on Town10HD junction 189 ``right`` picked the
    left-turn connector and ``left`` found no candidate, fell back to the
    lateral probe and routed a 525 m loop that began with a right turn.
    """
    maneuver = (maneuver or "straight").lower()
    yaw0 = start_wp.transform.rotation.yaw
    candidates = list(start_wp.next(distance) or [])

    def _probe(lateral_sign: float):
        """lateral_sign: -1 = left of travel, +1 = right."""
        loc = start_wp.transform.location
        fwd = start_wp.transform.get_forward_vector()
        right = start_wp.transform.get_right_vector()
        probe = carla.Location(
            loc.x + fwd.x * distance * 0.45 + right.x * lateral_sign * distance * 0.55,
            loc.y + fwd.y * distance * 0.45 + right.y * lateral_sign * distance * 0.55,
            loc.z,
        )
        end_wp = world.get_map().get_waypoint(
            probe, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        return end_wp.transform.location if end_wp is not None else probe

    if maneuver == "left":
        if candidates:
            best = min(
                candidates,
                key=lambda w: _signed_yaw_delta_deg(yaw0, w.transform.rotation.yaw),
            )
            if _signed_yaw_delta_deg(yaw0, best.transform.rotation.yaw) < -15.0:
                return best.transform.location
        return _probe(-1.0)

    if maneuver == "right":
        if candidates:
            best = max(
                candidates,
                key=lambda w: _signed_yaw_delta_deg(yaw0, w.transform.rotation.yaw),
            )
            if _signed_yaw_delta_deg(yaw0, best.transform.rotation.yaw) > 15.0:
                return best.transform.location
        return _probe(1.0)

    # straight (default): smallest heading change, else forward probe
    if candidates:
        best = min(
            candidates,
            key=lambda w: _yaw_delta_deg(w.transform.rotation.yaw, yaw0),
        )
        if _yaw_delta_deg(best.transform.rotation.yaw, yaw0) <= 45.0:
            return best.transform.location

    loc = start_wp.transform.location
    fwd = start_wp.transform.get_forward_vector()
    probe = carla.Location(
        loc.x + fwd.x * distance,
        loc.y + fwd.y * distance,
        loc.z,
    )
    end_wp = world.get_map().get_waypoint(
        probe, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if end_wp is not None:
        return end_wp.transform.location
    if candidates:
        return best.transform.location
    return probe


#: The route is extended until it could not run out within the horizon at this
#: speed. Stock LASER plans only 50 m past the scene anchor, which a 10 m/s ego
#: passes mid-episode; from then on a route-conditioned policy is either handed a
#: short route or points LASER invented, which no other method gives it.
ROUTE_COVER_SPEED_MPS = 20.0


def extend_route(carla_map, route, extra_m, step_m=1.0):
    """Continue ``route`` along the road past its end by ``extra_m`` metres,
    taking the straight-most successor at each fork."""
    if not route or extra_m <= 0:
        return list(route)
    wp = carla_map.get_waypoint(
        route[-1][0].location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if wp is None:
        return list(route)
    out = list(route)
    walked = 0.0
    while walked < extra_m:
        successors = wp.next(step_m)
        if not successors:
            break
        yaw = wp.transform.rotation.yaw
        wp = min(successors, key=lambda w: _yaw_delta_deg(w.transform.rotation.yaw, yaw))
        out.append((wp.transform, RoadOption.LANEFOLLOW))
        walked += step_m
    return out


def route_length(route):
    return sum(
        a[0].location.distance(b[0].location) for a, b in zip(route[:-1], route[1:])
    )


class TargetVehicle(Agent):
    def __init__(self, world, agent_name, agent_script, lane_wps, agent_manager, queue) -> None:
        self.name = agent_name
        self.carla_actor, self.init_wp, self.init_velocity = spawn_actor_by_script(world, agent_script, lane_wps, role_name='hero')
        CarlaDataProvider.get_world().tick()
        self.type = 'vehicle'
        self._script = agent_script
        self.lane_wps = lane_wps
        self.lane_id2lane_num = {}
        for i, lane_wp in enumerate(lane_wps):
            self.lane_id2lane_num[lane_wp.lane_id] = i + 1

        self.agent_manager = agent_manager

        chased_actor = self.carla_actor
        spectator = CarlaDataProvider.get_world().get_spectator()
        trans = chased_actor.get_transform()
        spectator.set_transform(carla.Transform(trans.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

        # Set up the user's agent, and the timer to avoid freezing the simulation
        ego = os.environ.get("LASER_EGO", "interfuser").lower()
        self._ego_mode = ego
        self._idm_controller = None
        self._plant2_controller = None
        self._tfv6_controller = None
        self._simlingo_controller = None
        self._agent = None
        self.agent_instance = None

        init_state = agent_script['init_state']
        l = init_state[0] - 1

        # Plan from the actual spawn waypoint (same pose as the ego actor).
        # Recomputing previous(x) can jump onto another road/lane_id and flip yaw.
        wp = self.init_wp
        print(wp)

        # Stock LASER derives the mission from the road anchor, not script.json:
        # spawn at lane_wps[l].previous(abs(init_state.x)), then route to
        # lane_wps[l].next(50)[0]. Keep that exact rule for straight scenarios.
        # The added junction scenarios extend it only by selecting the requested
        # left/right branch at the same distance. This is policy-independent:
        # every ego receives the same gps_route and world-coordinate route.
        maneuver = getattr(agent_manager, "ego_maneuver", "straight") or "straight"
        # Stock default 50 m; T04HardBrake sets LASER_ROUTE_LOOKAHEAD_M=160 to
        # match scenario_hard_brake.yaml corridor length.
        look_ahead = float(os.environ.get("LASER_ROUTE_LOOKAHEAD_M", "50"))
        if maneuver == "straight":
            nxt = lane_wps[l].next(look_ahead)
            if not nxt:
                raise RuntimeError(
                    f"stock ego route: lane_wps[{l}].next({look_ahead}) empty"
                )
            end_loc = nxt[0].transform.location
            route_mode = "stock_lane_next"
        else:
            end_loc = ego_route_end_location(
                world, lane_wps[l], look_ahead, maneuver=maneuver
            )
            route_mode = "stock_distance_maneuver_branch"
        print(
            f"VUT route end=({end_loc.x:.1f},{end_loc.y:.1f},{end_loc.z:.1f}) "
            f"maneuver={maneuver} look_ahead={look_ahead:.0f}m mode={route_mode} ego={ego}"
        )
        route_cfg = agent_script.get("route") or {}
        dest = route_cfg.get("destination")
        if dest is not None and len(dest) >= 2:
            end_loc = carla.Location(
                float(dest[0]), float(dest[1]), float(dest[2]) if len(dest) > 2 else 0.0
            )
            end_wp = world.get_map().get_waypoint(
                end_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if end_wp is not None:
                end_loc = end_wp.transform.location
            print(
                f"VUT route maneuver={route_cfg.get('maneuver', 'custom')} "
                f"destination=({end_loc.x:.1f},{end_loc.y:.1f},{end_loc.z:.1f})"
            )

        gps_route, self.route = interpolate_trajectory(
            world, [wp.transform.location, end_loc], hop_resolution=1.0
        )
        horizon_s = float(getattr(agent_manager, "simulation_time", 0) or 0)
        cover_m = horizon_s * float(
            os.environ.get("LASER_ROUTE_COVER_SPEED_MPS", ROUTE_COVER_SPEED_MPS)
        )
        planned_m = route_length(self.route)
        if cover_m > planned_m:
            self.route = extend_route(world.get_map(), self.route, cover_m - planned_m)
        print(
            f"VUT route {planned_m:.0f} m planned, {route_length(self.route):.0f} m "
            f"after extension to cover {horizon_s:g} s"
        )
        CarlaDataProvider.set_ego_route(convert_transform_to_location(self.route))

        CarlaDataProvider.get_world().tick()

        debug_mode = False
        if debug_mode:
            self._draw_waypoints(world, self.route, vertical_shift=1.0, persistency=50000.0)

        self._harness_ego = None
        policy_request = os.environ.get("LASER_POLICY_REQUEST")
        if policy_request:
            # The harness's own policy request: the same policy repository and the
            # same method-side driver the other methods use, instead of LASER's
            # native re-implementations below (see harness_policy_ego.py).
            from laser.target_vehicle.harness_policy_ego import HarnessPolicyEgo

            self._harness_ego = HarnessPolicyEgo(
                self.carla_actor, self.route, policy_request, agent_manager
            )
            self._ego_mode = "harness"
        elif ego == "idm":
            from laser.target_vehicle.idm_ego import IDMEgoController, IDMParams
            mobil_lane_ids = {
                wp.lane_id for wp in lane_wps[: max(1, agent_manager.driving_lane_num)]
            }
            self._idm_controller = IDMEgoController(
                self.carla_actor,
                self.route,
                params=IDMParams.from_env(),
                mobil_lane_ids=mobil_lane_ids,
            )
        elif ego in ("plant2", "plant"):
            from laser.target_vehicle.plant2_ego import PlanT2EgoController
            self._ego_mode = "plant2"
            self._plant2_controller = PlanT2EgoController(
                self.carla_actor, self.route, seed=int(os.environ.get("PLANT2_SEED", "0"))
            )
        elif ego in ("tfv6", "transfuser_v6", "transfuser-v6"):
            from laser.target_vehicle.tfv6_ego import TFv6EgoController
            self._ego_mode = "tfv6"
            self._tfv6_controller = TFv6EgoController(
                self.carla_actor,
                self.route,
                seed=int(os.environ.get("TFV6_SEED", "0")),
                maneuver=maneuver,
            )
        elif ego in ("simlingo", "laser_simlingo"):
            from laser.target_vehicle.simlingo_ego import SimLingoEgoController
            self._ego_mode = "simlingo"
            self._simlingo_controller = SimLingoEgoController(
                self.carla_actor,
                self.route,
                seed=int(os.environ.get("SIMLINGO_SEED", "0")),
            )
        elif ego == "interfuser":
            from team_code.interfuser_agent import InterfuserAgent
            self.agent_instance = InterfuserAgent('leaderboard/team_code/interfuser_config.py')
            self.agent_instance.set_global_plan(gps_route, self.route)
            self.agent_instance._init()
            self.agent_instance.sensor_interface = SensorInterface()
            self._agent = AgentWrapper(self.agent_instance)
            self._agent.setup_sensors(self.carla_actor, False)
        elif ego == "transfuser":
            from submission_agent import HybridAgent
            ckpt = os.environ.get(
                "TRANSFUSER_CKPT",
                os.path.expanduser("~/scratch/transfuser/model_ckpt/models_2022/transfuser"),
            )
            self.agent_instance = HybridAgent(ckpt)
            self.agent_instance.set_global_plan(gps_route, self.route)
            self.agent_instance._init()
            self.agent_instance.sensor_interface = SensorInterface()
            self._agent = AgentWrapper(self.agent_instance)
            self._agent.setup_sensors(self.carla_actor, False)
        else:
            raise ValueError(f"Unknown LASER_EGO={ego}")

        self.collision_sensor = CollisionSensor(queue, self)

        self._llm_agent = None

    def init_after_carla_tick(self):
        pass

    def on_tick(self, dt):
        if self._ego_mode == "harness":
            ego_action = self._harness_ego.run_step(dt)
        elif self._ego_mode == "idm":
            ego_action = self._idm_controller.run_step(dt)
        elif self._ego_mode == "plant2":
            ego_action = self._plant2_controller.run_step(dt)
        elif self._ego_mode == "tfv6":
            ego_action = self._tfv6_controller.run_step(dt)
        elif self._ego_mode == "simlingo":
            ego_action = self._simlingo_controller.run_step(dt)
        else:
            ego_action = self._agent()
            print(ego_action)

        self.carla_actor.apply_control(ego_action)

    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            size = 0.2
            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0) # Green
                size = 0.1

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(255, 0, 0), life_time=persistency)

    def get_self_obs_info(self):
        transform = self.carla_actor.get_transform()
        wp = CarlaDataProvider.get_map().get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_id = wp.lane_id
        speed = self.carla_actor.get_velocity()
        acceleration = self.carla_actor.get_acceleration()


        # lane_id, transform, speed, acceleration = self._decision_interpreter_inst.get_self_obs_info()
        # print(f"{self.name}: {transform}")
        # Scripted corridor is only the spawn lane_wps (e.g. -1/-2). If ego
        # leaves that (junction, opposite lanes, off-road project), CARLA
        # lane_id is unknown — don't crash the LLM tick.
        if lane_id not in self.lane_id2lane_num:
            print(
                f"warning: VUT unknown carla lane_id={lane_id} "
                f"known={self.lane_id2lane_num} loc=({transform.location.x:.1f},"
                f"{transform.location.y:.1f})"
            )
            lane_num = next(iter(self.lane_id2lane_num.values()), 1)
        else:
            lane_num = self.lane_id2lane_num[lane_id]
        
        lane_wp = self.lane_wps[0]
        transform_wp = lane_wp.transform
        transform_self = transform
        m_wp = np.array(transform_wp.get_matrix())
        m_wp_inv = np.array(transform_wp.get_inverse_matrix())
        m_self = np.array(transform_self.get_matrix())
        t_in_wp_view = np.dot(m_wp_inv, (m_self - m_wp)[:, 3]) # x axis points to front, y axis points to right
        location = t_in_wp_view[:2]
        # print(lane_wp.transform, transform, t_rel_transformed)

        direction_angle = transform_self.rotation.yaw - transform_wp.rotation.yaw
        while direction_angle > 180:
            direction_angle -= 360
        while direction_angle < -180:
            direction_angle += 360

        location = [150 + location[0], location[1]]

        speed = speed.length()
        acceleration = acceleration.length()
        return lane_num, location, speed, acceleration, direction_angle

    async def get_decisions(self, sensor_data):
        decisions = {
                    'current_step_number': 1,
                    'lane_change_direction': 'FOLLOW LANE',
                    'lane_change_delay': 0,
                    'target_speed': self.init_velocity.length()
                    }
        return decisions

    def handle_decisions(self, decisions, dt):
        pass

    def destroy(self):
        if self._harness_ego is not None:
            try:
                self._harness_ego.close()
            except Exception as exc:  # noqa: BLE001
                print(f"harness ego close failed: {exc}")
            self._harness_ego = None
        if self._tfv6_controller is not None:
            try:
                self._tfv6_controller.close()
            except Exception as exc:  # noqa: BLE001
                print(f"tfv6 close failed: {exc}")
            self._tfv6_controller = None
        if self._simlingo_controller is not None:
            try:
                self._simlingo_controller.close()
            except Exception as exc:  # noqa: BLE001
                print(f"simlingo close failed: {exc}")
            self._simlingo_controller = None
        if self._plant2_controller is not None:
            try:
                self._plant2_controller.close()
            except Exception as exc:  # noqa: BLE001
                print(f"plant2 close failed: {exc}")
            self._plant2_controller = None
        super().destroy()

