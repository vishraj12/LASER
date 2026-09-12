"""Analytic IDM (+ optional MOBIL) ego for LASER.

Longitudinal: classic Treiber Intelligent Driver Model → throttle/brake.
Lateral: lane-center P-steer; MOBIL commits via BasicAgent.lane_change when
         enabled (lane_change / overtake).

Defaults and MOBIL extras aligned with friedeggs/idm (highway_ego / drivev2
reference): blocked-lead standoff, LC below v_min when blocked, oncoming pass
gate, contraflow bias. CARLA map/waypoints stay authoritative for neighbors.

Env overrides (optional):
  IDM_DESIRED_SPEED_MPS   default 12.0
  IDM_TIME_HEADWAY_S      default 1.5
  IDM_MIN_GAP_M           default 2.0
  IDM_MAX_ACCEL_MPS2      default 3.0
  IDM_COMFORT_DECEL_MPS2  default 3.0
  IDM_DELTA               default 4.0
  IDM_LEAD_MAX_M          default 80.0
  IDM_LANE_HALF_WIDTH_M   default 1.8
  IDM_ENABLE_MOBIL        default 0  (set 1 for lane_change / overtake)
  MOBIL_P, MOBIL_A_THR, MOBIL_B_SAFE, MOBIL_BIAS_RIGHT, MOBIL_BIAS_ONCOMING,
  MOBIL_MIN_INTERVAL, MOBIL_V_MIN
  (MOBIL_MIN_INTERVAL also applies as post-LC settle before the next decision)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import carla

from laser.carla_agents.navigation.basic_agent import BasicAgent
from laser.carla_agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# Neighbor record: (along_m, signed_speed_mps, length_m)
Neighbor = Tuple[float, float, float]

# friedeggs/idm :: laws.py (highway_ego / drivev2 reference)
BLOCKED_LEAD_MPS = 1.0
BLOCKED_GAP_M = 12.0
BLOCKED_STANDOFF_M = 8.0
ONCOMING_SAFETY = 2.0
MIN_PASS_CLOSING = 2.0
LC_FRONT_GAP = 8.0
LC_REAR_GAP = 6.0
STEER_RATE = 4.0  # normalized steer / s


@dataclass
class IDMParams:
    # Reference defaults (friedeggs/idm IDM_V0/A/B/S0/T).
    desired_speed_mps: float = 12.0
    time_headway_s: float = 1.5
    min_gap_m: float = 2.0
    max_accel_mps2: float = 3.0
    comfort_decel_mps2: float = 3.0
    delta: float = 4.0
    lead_max_m: float = 80.0
    lane_half_width_m: float = 1.8

    @classmethod
    def from_env(cls) -> "IDMParams":
        def _f(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            return float(raw)

        return cls(
            desired_speed_mps=_f("IDM_DESIRED_SPEED_MPS", 12.0),
            time_headway_s=_f("IDM_TIME_HEADWAY_S", 1.5),
            min_gap_m=_f("IDM_MIN_GAP_M", 2.0),
            max_accel_mps2=_f("IDM_MAX_ACCEL_MPS2", 3.0),
            comfort_decel_mps2=_f("IDM_COMFORT_DECEL_MPS2", 3.0),
            delta=_f("IDM_DELTA", 4.0),
            lead_max_m=_f("IDM_LEAD_MAX_M", 80.0),
            lane_half_width_m=_f("IDM_LANE_HALF_WIDTH_M", 1.8),
        )


@dataclass
class MOBILParams:
    politeness: float = 0.5
    a_thr: float = 0.15
    b_safe: float = 4.0
    bias_right: float = 0.15
    bias_oncoming: float = 0.5
    min_interval_s: float = 2.0
    v_min_mps: float = 3.0
    behind_max_m: float = 40.0

    @classmethod
    def from_env(cls) -> "MOBILParams":
        def _f(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            return float(raw)

        return cls(
            politeness=_f("MOBIL_P", 0.5),
            a_thr=_f("MOBIL_A_THR", 0.15),
            b_safe=_f("MOBIL_B_SAFE", 4.0),
            bias_right=_f("MOBIL_BIAS_RIGHT", 0.15),
            bias_oncoming=_f("MOBIL_BIAS_ONCOMING", 0.5),
            min_interval_s=_f("MOBIL_MIN_INTERVAL", 2.0),
            v_min_mps=_f("MOBIL_V_MIN", 3.0),
            behind_max_m=_f("MOBIL_BEHIND_MAX_M", 40.0),
        )


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def idm_acceleration(
    v: float,
    gap: Optional[float],
    v_lead: Optional[float],
    params: IDMParams,
    v0: Optional[float] = None,
    s0: Optional[float] = None,
) -> float:
    """Treiber IDM longitudinal acceleration (m/s^2).

    ``gap is None`` => free road (no interaction term).
    ``v0`` / ``s0`` overrides used for MOBIL hypotheticals and blocked standoff.
    """
    v0 = max((params.desired_speed_mps if v0 is None else v0), 1.0)
    a = params.max_accel_mps2
    b = max(params.comfort_decel_mps2, 0.1)
    s0_use = params.min_gap_m if s0 is None else float(s0)
    T = params.time_headway_s
    delta = params.delta

    free = a * (1.0 - (max(v, 0.0) / v0) ** delta)
    if gap is None or v_lead is None:
        return free

    dv = v - v_lead
    s_star = s0_use + max(0.0, v * T + (v * dv) / (2.0 * math.sqrt(a * b)))
    s = max(gap, 0.5)
    return free - a * (s_star / s) ** 2


def _same_dir_lead(lead: Optional[Neighbor]) -> bool:
    """True for a same-direction leader (not contraflow ahead)."""
    return lead is not None and lead[1] >= -0.5


def _is_blocked(lead: Optional[Neighbor], gap: Optional[float]) -> bool:
    """Stopped/slow same-dir lead close enough that a low-speed LC is allowed."""
    return (
        _same_dir_lead(lead)
        and gap is not None
        and lead[1] <= BLOCKED_LEAD_MPS
        and gap <= BLOCKED_GAP_M
    )


def _ego_s0(
    params: IDMParams, lead: Optional[Neighbor], gap: Optional[float]
) -> float:
    """Jam distance: standoff while a stoppable blocker is ahead, else s0.

    Matches friedeggs/idm ``ego_s0`` — without this, IDM parks at 2 m and MOBIL
    never beats the threshold around a stopped lead.
    """
    if _is_blocked(lead, gap):
        return BLOCKED_STANDOFF_M
    return params.min_gap_m


def _lane_slot_clear(lead: Optional[Neighbor], fol: Optional[Neighbor]) -> bool:
    """Room to sit in the target lane right now (friedeggs ``lane_clear``).

    Oncoming traffic is handled by ``_oncoming_pass_clear``, not this slot test.
    """
    if lead is not None and lead[1] >= -0.5 and 0.0 < lead[0] < LC_FRONT_GAP:
        return False
    if fol is not None and -LC_REAR_GAP < fol[0] < 0.0:
        return False
    return True


def _oncoming_pass_clear(
    lead_t: Optional[Neighbor],
    v_ego: float,
    pass_distance: float,
    v_blocker: float,
) -> bool:
    """Contraflow safe for a pass of ``pass_distance`` (friedeggs oncoming_clear)."""
    if lead_t is None or lead_t[0] <= 0.0 or lead_t[1] >= -0.5:
        return True
    v = max(v_ego, 1.0)
    v_rel = max(v - max(v_blocker, 0.0), MIN_PASS_CLOSING)
    t_pass = max(pass_distance, 1.0) / v_rel
    closing = v + abs(lead_t[1])
    if closing <= 0.1:
        return True
    return (lead_t[0] / closing) >= ONCOMING_SAFETY * t_pass


def _forward_speed(actor: carla.Actor) -> float:
    vel = actor.get_velocity()
    forward = actor.get_transform().get_forward_vector()
    return vel.x * forward.x + vel.y * forward.y + vel.z * forward.z


def _extent_x(actor: carla.Actor) -> float:
    try:
        return float(actor.bounding_box.extent.x)
    except Exception:
        return 2.3


def _extent_y(actor: carla.Actor) -> float:
    try:
        return float(actor.bounding_box.extent.y)
    except Exception:
        return 1.0


def _signed_speed_along(actor: carla.Actor, forward: carla.Vector3D) -> float:
    vel = actor.get_velocity()
    return vel.x * forward.x + vel.y * forward.y + vel.z * forward.z


def find_lead_vehicle(
    ego: carla.Actor,
    params: IDMParams,
) -> Tuple[Optional[carla.Actor], Optional[float], float]:
    """Closest same-lane (or cutting-into-lane) vehicle ahead of ego.

    Returns ``(lead, gap_m, lead_speed_mps)``.
    ``gap_m is None`` when there is no lead (free road).
    Gap is along-track bumper-to-bumper.
    """
    try:
        ego_tf = ego.get_transform()
    except RuntimeError:
        return None, None, 0.0

    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()
    ego_ext = _extent_x(ego)
    half_w = params.lane_half_width_m
    max_d = params.lead_max_m

    carla_map = CarlaDataProvider.get_map()
    ego_wp = carla_map.get_waypoint(
        ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
    )

    best = None
    best_gap = float("inf")
    best_speed = 0.0

    world = CarlaDataProvider.get_world()
    for other in world.get_actors().filter("vehicle.*"):
        if other.id == ego.id:
            continue
        try:
            other_loc = other.get_location()
        except RuntimeError:
            continue

        dx = other_loc.x - ego_loc.x
        dy = other_loc.y - ego_loc.y
        along = dx * forward.x + dy * forward.y
        if along <= 0.0:
            continue
        lat = abs(dx * right.x + dy * right.y)
        if along > max_d + 10.0:
            continue

        other_wp = carla_map.get_waypoint(
            other_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        same_lane = False
        if ego_wp is not None and other_wp is not None:
            same_lane = (
                ego_wp.road_id == other_wp.road_id
                and ego_wp.lane_id == other_wp.lane_id
            )
        cutting_in = lat <= half_w

        if not (same_lane or cutting_in):
            continue

        gap = along - ego_ext - _extent_x(other)
        if gap < best_gap:
            best = other
            best_gap = gap
            best_speed = max(0.0, _forward_speed(other))

    if best is None:
        return None, None, 0.0
    return best, max(best_gap, 0.1), best_speed


def accel_to_control(accel: float, params: IDMParams) -> Tuple[float, float]:
    """Map IDM accel (m/s^2) -> (throttle, brake) in [0, 1]."""
    a_max = max(params.max_accel_mps2, 0.1)
    b_scale = max(params.comfort_decel_mps2, 0.1) * 2.0
    if accel >= 0.0:
        return min(1.0, accel / a_max), 0.0
    return 0.0, min(1.0, -accel / b_scale)


def _vehicle_overlaps_lane(
    other: carla.Actor,
    lane_wp: carla.Waypoint,
    half_lane: float,
) -> bool:
    """True if other body overlaps the lane strip (not only nearest-center)."""
    try:
        other_loc = other.get_location()
    except RuntimeError:
        return False
    lane_tf = lane_wp.transform
    right = lane_tf.get_right_vector()
    dx = other_loc.x - lane_tf.location.x
    dy = other_loc.y - lane_tf.location.y
    lat = abs(dx * right.x + dy * right.y)
    return lat <= half_lane + _extent_y(other)


def _lane_neighbors(
    ego: carla.Actor,
    lane_wp: carla.Waypoint,
    params: IDMParams,
    mobil: MOBILParams,
) -> Tuple[Optional[Neighbor], Optional[Neighbor]]:
    """(leader, follower) on ``lane_wp`` relative to ego.

    Leader may be oncoming (negative signed speed). Follower is same-direction
    only (receding contraflow behind us is ignored), matching drivev2 MOBIL.
    """
    try:
        ego_tf = ego.get_transform()
    except RuntimeError:
        return None, None

    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    half_lane = max(lane_wp.lane_width * 0.5, params.lane_half_width_m)
    max_ahead = params.lead_max_m
    max_behind = mobil.behind_max_m

    lead: Optional[Neighbor] = None
    fol: Optional[Neighbor] = None

    world = CarlaDataProvider.get_world()
    carla_map = CarlaDataProvider.get_map()
    for other in world.get_actors().filter("vehicle.*"):
        if other.id == ego.id:
            continue
        try:
            other_loc = other.get_location()
        except RuntimeError:
            continue

        other_wp = carla_map.get_waypoint(
            other_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        same_lane = (
            other_wp is not None
            and other_wp.road_id == lane_wp.road_id
            and other_wp.lane_id == lane_wp.lane_id
        )
        if not (same_lane or _vehicle_overlaps_lane(other, lane_wp, half_lane)):
            continue

        dx = other_loc.x - ego_loc.x
        dy = other_loc.y - ego_loc.y
        along = dx * forward.x + dy * forward.y
        if along > max_ahead or along < -max_behind:
            continue

        speed = _signed_speed_along(other, forward)
        length = 2.0 * _extent_x(other)
        opposing = speed < -0.5

        if along > 0.0:
            if lead is None or along < lead[0]:
                lead = (along, speed, length)
        elif not opposing:
            if fol is None or along > fol[0]:
                fol = (along, speed, length)

    return lead, fol


def _gap_to_lead(lead: Optional[Neighbor], ego_len: float) -> Optional[float]:
    if lead is None:
        return None
    return max(lead[0] - lead[2] / 2.0 - ego_len / 2.0, 0.5)


def _gap_from_fol(fol: Optional[Neighbor], ego_len: float) -> Optional[float]:
    if fol is None:
        return None
    return max(-fol[0] - fol[2] / 2.0 - ego_len / 2.0, 0.5)


def _gap_between(rear: Optional[Neighbor], front: Optional[Neighbor]) -> Optional[float]:
    if rear is None or front is None:
        return None
    return max(front[0] - rear[0] - front[2] / 2.0 - rear[2] / 2.0, 0.5)


class IDMEgoController:
    """Lane-follow (steer) + IDM longitudinal; optional MOBIL lane changes."""

    def __init__(
        self,
        vehicle: carla.Vehicle,
        route: Sequence[Tuple[carla.Transform, object]],
        params: Optional[IDMParams] = None,
        enable_mobil: Optional[bool] = None,
        mobil_params: Optional[MOBILParams] = None,
        mobil_lane_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.vehicle = vehicle
        self.params = params or IDMParams.from_env()
        if enable_mobil is None:
            enable_mobil = _env_flag("IDM_ENABLE_MOBIL", False)
        self.enable_mobil = bool(enable_mobil)
        self.mobil = mobil_params or MOBILParams.from_env()
        self._mobil_cooldown = 0.0
        self._mobil_lane_ids = set(mobil_lane_ids) if mobil_lane_ids else None
        # Generic LC commit state (any MOBIL-enabled road — not scenario-specific).
        self._lc_active = False
        self._lc_target_lane_id: Optional[int] = None
        self._lc_target_road_id: Optional[int] = None
        self._lc_settle_hold_s = 0.0
        self._prev_lat_err: Optional[float] = None
        self._prev_steer = 0.0
        # Home lane for contraflow bias (friedeggs lanes.is_contraflow).
        try:
            home_wp = CarlaDataProvider.get_map().get_waypoint(
                vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            self._home_lane_id: Optional[int] = (
                int(home_wp.lane_id) if home_wp is not None else None
            )
        except Exception:
            self._home_lane_id = None

        init_kmh = self.params.desired_speed_mps * 3.6
        self._agent = BasicAgent(
            vehicle,
            direction=1,
            target_speed=init_kmh,
            opt_dict={
                "ignore_traffic_lights": True,
                "ignore_stop_signs": True,
                "ignore_vehicles": True,
            },
            map_inst=CarlaDataProvider.get_map(),
        )
        plan: List[Tuple[carla.Waypoint, object]] = []
        carla_map = CarlaDataProvider.get_map()
        for transform, road_option in route:
            wp = carla_map.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is not None:
                plan.append((wp, road_option))
        if plan:
            self._agent.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)

        print(
            f"IDM ego: v0={self.params.desired_speed_mps:.1f} m/s "
            f"T={self.params.time_headway_s:.1f}s "
            f"s0={self.params.min_gap_m:.1f}m "
            f"a={self.params.max_accel_mps2:.1f} "
            f"b={self.params.comfort_decel_mps2:.1f} "
            f"mobil={'ON' if self.enable_mobil else 'off'} "
            f"mobil_lanes={sorted(self._mobil_lane_ids) if self._mobil_lane_ids else 'any'} "
            f"route_wps={len(plan)}"
        )

    def _lane_change_in_progress(self) -> bool:
        lp = self._agent._local_planner
        if getattr(lp, "preparing_lane_change", False):
            return True
        opt = getattr(lp, "target_road_option", None)
        if opt in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            return True
        queue = getattr(lp, "_waypoints_queue", None)
        if not queue:
            return False
        for _, qopt in list(queue)[:24]:
            if qopt in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                return True
        return False

    @staticmethod
    def _same_direction(a: carla.Waypoint, b: carla.Waypoint) -> bool:
        # On 2-way roads get_left_lane() flips between +/− lane_id forever.
        return a.lane_id * b.lane_id > 0

    @staticmethod
    def _driving_lane_row(seed: carla.Waypoint) -> List[carla.Waypoint]:
        """All same-direction Driving lanes at this cross-section."""
        probe = seed
        while True:
            left_p = probe.get_left_lane()
            if (
                left_p is None
                or left_p.lane_type != carla.LaneType.Driving
                or not IDMEgoController._same_direction(probe, left_p)
            ):
                break
            probe = left_p
        row: List[carla.Waypoint] = []
        walk: Optional[carla.Waypoint] = probe
        while walk is not None:
            if walk.lane_type == carla.LaneType.Driving:
                row.append(walk)
            nxt = walk.get_right_lane()
            walk = (
                nxt
                if (
                    nxt is not None
                    and nxt.lane_type == carla.LaneType.Driving
                    and IDMEgoController._same_direction(walk, nxt)
                )
                else None
            )
        return row

    def _find_lane_wp(
        self, seed: carla.Waypoint, lane_id: int
    ) -> Optional[carla.Waypoint]:
        for wp in self._driving_lane_row(seed):
            if wp.lane_id == lane_id:
                return wp
        # Opposite-direction neighbor (e.g. overtake into oncoming).
        for neigh in (seed.get_left_lane(), seed.get_right_lane()):
            if (
                neigh is not None
                and neigh.lane_type == carla.LaneType.Driving
                and neigh.lane_id == lane_id
            ):
                return neigh
        return None

    def _resolve_lc_target_wp(
        self, cur_wp: carla.Waypoint
    ) -> Optional[carla.Waypoint]:
        """Waypoint on the committed target lane near the ego."""
        if self._lc_target_lane_id is None or cur_wp is None:
            return None
        return self._find_lane_wp(cur_wp, self._lc_target_lane_id)

    def _clear_lc_commit(self) -> None:
        self._lc_active = False
        self._lc_target_lane_id = None
        self._lc_target_road_id = None
        self._lc_settle_hold_s = 0.0
        self._prev_lat_err = None

    def _lat_yaw_err(
        self, ego_tf: carla.Transform, wp: carla.Waypoint
    ) -> Tuple[float, float]:
        """Lateral / yaw error to ``wp``.

        Use the home-lane right axis when the target is contraflow so a
        northbound pass into the southbound lane still steers toward +x
        instead of flipping with the opposite lane's yaw.
        """
        ego_loc = ego_tf.location
        frame_wp = wp
        if (
            self._home_lane_id is not None
            and wp.lane_id * self._home_lane_id < 0
        ):
            seed = CarlaDataProvider.get_map().get_waypoint(
                ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            home = (
                self._find_lane_wp(seed, self._home_lane_id)
                if seed is not None
                else None
            )
            if home is not None:
                frame_wp = home
        right = frame_wp.transform.get_right_vector()
        lat_err = (ego_loc.x - wp.transform.location.x) * right.x + (
            ego_loc.y - wp.transform.location.y
        ) * right.y
        lane_yaw = math.radians(frame_wp.transform.rotation.yaw)
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        yaw_err = (lane_yaw - ego_yaw + math.pi) % (2.0 * math.pi) - math.pi
        return float(lat_err), float(yaw_err)

    def _update_lc_settle(self, dt: float) -> bool:
        """Accumulate time spent near the target lane; True when settled."""
        if self._lc_target_lane_id is None:
            return True
        try:
            ego_tf = self.vehicle.get_transform()
            ego_loc = ego_tf.location
        except RuntimeError:
            self._lc_settle_hold_s = 0.0
            return False
        carla_map = CarlaDataProvider.get_map()
        seed = carla_map.get_waypoint(
            ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if seed is None:
            self._lc_settle_hold_s = 0.0
            return False
        tgt = self._resolve_lc_target_wp(seed)
        if tgt is None:
            self._lc_settle_hold_s = 0.0
            return False
        lat_err, yaw_err = self._lat_yaw_err(ego_tf, tgt)
        near = abs(lat_err) < 0.70 and abs(yaw_err) < math.radians(15.0)
        on_lane = seed.lane_id == self._lc_target_lane_id and abs(lat_err) < 0.90
        if near or on_lane:
            self._lc_settle_hold_s += max(dt, 0.0)
        else:
            self._lc_settle_hold_s = 0.0
        return self._lc_settle_hold_s >= 0.40

    def _steer_target_wp(
        self, seed: carla.Waypoint, ego_tf: carla.Transform
    ) -> Optional[carla.Waypoint]:
        """Lane center to track: committed LC target, else allowed corridor."""
        ego_loc = ego_tf.location
        if self._lc_target_lane_id is not None:
            tgt = self._resolve_lc_target_wp(seed)
            if tgt is not None:
                return tgt
        row = self._driving_lane_row(seed)
        if self._mobil_lane_ids is not None:
            allowed = [wp for wp in row if wp.lane_id in self._mobil_lane_ids]
            if allowed:
                right = ego_tf.get_right_vector()

                def _lat(wp: carla.Waypoint) -> float:
                    return abs(
                        (ego_loc.x - wp.transform.location.x) * right.x
                        + (ego_loc.y - wp.transform.location.y) * right.y
                    )

                return min(allowed, key=_lat)
        return seed

    def _lane_center_steer(self, max_steer: float = 0.35, dt: float = 0.05) -> float:
        """Damped P-control onto LC target / allowed corridor center."""
        try:
            ego_tf = self.vehicle.get_transform()
            ego_loc = ego_tf.location
        except RuntimeError:
            return 0.0
        carla_map = CarlaDataProvider.get_map()
        seed = carla_map.get_waypoint(
            ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if seed is None:
            return 0.0
        wp = self._steer_target_wp(seed, ego_tf)
        if wp is None:
            return 0.0
        lat_err, yaw_err = self._lat_yaw_err(ego_tf, wp)
        lat_dot = 0.0
        if self._prev_lat_err is not None and dt > 1e-3:
            lat_dot = (lat_err - self._prev_lat_err) / dt
        self._prev_lat_err = lat_err
        # Deadband near center to stop hunting across the lane.
        lat_term = 0.0 if abs(lat_err) < 0.25 else lat_err
        # Positive lat_err => ego right of center => steer left.
        steer = -0.32 * lat_term - 0.45 * yaw_err - 0.12 * lat_dot
        # Soften authority when already close.
        auth = max_steer
        if abs(lat_err) < 1.2:
            auth = min(auth, 0.12 + 0.18 * abs(lat_err))
        return max(-auth, min(auth, float(steer)))

    def _mobil_evaluate(
        self,
        cur_wp: carla.Waypoint,
        tgt_wp: carla.Waypoint,
        to_right: bool,
        v_e: float,
        s0: float,
        contraflow: bool,
    ) -> Tuple[bool, float, str]:
        """Safety + incentive tests for cur_wp → tgt_wp."""
        ego_len = 2.0 * _extent_x(self.vehicle)
        lead_c, fol_c = _lane_neighbors(self.vehicle, cur_wp, self.params, self.mobil)
        lead_t, fol_t = _lane_neighbors(self.vehicle, tgt_wp, self.params, self.mobil)

        a_e_cur = idm_acceleration(
            v_e, _gap_to_lead(lead_c, ego_len),
            None if lead_c is None else lead_c[1], self.params, s0=s0,
        )
        a_e_new = idm_acceleration(
            v_e, _gap_to_lead(lead_t, ego_len),
            None if lead_t is None else lead_t[1], self.params, s0=s0,
        )

        if fol_t is None:
            a_nf_cur = a_nf_new = 0.0
        else:
            a_nf_cur = idm_acceleration(
                fol_t[1],
                _gap_between(fol_t, lead_t),
                None if lead_t is None else lead_t[1],
                self.params,
                v0=max(abs(fol_t[1]), 1.0),
            )
            a_nf_new = idm_acceleration(
                fol_t[1],
                _gap_from_fol(fol_t, ego_len),
                v_e,
                self.params,
                v0=max(abs(fol_t[1]), 1.0),
            )
            if a_nf_new < -self.mobil.b_safe:
                return False, 0.0, f"unsafe for follower ({a_nf_new:.1f} m/s^2)"

        if fol_c is None:
            a_of_cur = a_of_new = 0.0
        else:
            a_of_cur = idm_acceleration(
                fol_c[1],
                _gap_from_fol(fol_c, ego_len),
                v_e,
                self.params,
                v0=max(abs(fol_c[1]), 1.0),
            )
            a_of_new = idm_acceleration(
                fol_c[1],
                _gap_between(fol_c, lead_c),
                None if lead_c is None else lead_c[1],
                self.params,
                v0=max(abs(fol_c[1]), 1.0),
            )

        gain = (a_e_new - a_e_cur) + self.mobil.politeness * (
            (a_nf_new - a_nf_cur) + (a_of_new - a_of_cur)
        )
        thr = self.mobil.a_thr + (
            -self.mobil.bias_right if to_right else self.mobil.bias_right
        )
        # Dear to enter contraflow; cheap to leave (friedeggs MOBIL_BIAS_ONCOMING).
        if contraflow:
            thr += self.mobil.bias_oncoming
        if self._home_lane_id is not None and cur_wp.lane_id * self._home_lane_id < 0:
            thr -= self.mobil.bias_oncoming

        if gain <= thr:
            return False, gain, f"gain {gain:.2f} <= threshold {thr:.2f}"
        return True, gain, f"gain {gain:.2f} > threshold {thr:.2f}"

    def _mobil_step(self, dt: float, v: float) -> None:
        if not self.enable_mobil:
            return
        self._mobil_cooldown = max(0.0, self._mobil_cooldown - max(dt, 0.0))

        # Geometric settle: keep the LC commit until centered on the target lane
        # (planner CHANGELANE* clearing early was letting us overshoot).
        if self._lc_target_lane_id is not None:
            if self._update_lc_settle(dt):
                self._clear_lc_commit()
                self._mobil_cooldown = max(
                    self._mobil_cooldown, self.mobil.min_interval_s
                )
                print(
                    f"MOBIL: LC settle (geometry) cooldown {self._mobil_cooldown:.2f}s "
                    f"(min_interval={self.mobil.min_interval_s:.2f}s)"
                )
            else:
                self._lc_active = True
                return

        if self._mobil_cooldown > 0.0:
            return

        carla_map = CarlaDataProvider.get_map()
        try:
            ego_loc = self.vehicle.get_location()
        except RuntimeError:
            return
        cur_wp = carla_map.get_waypoint(
            ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if cur_wp is None:
            return

        ego_len = 2.0 * _extent_x(self.vehicle)
        lead_c, _fol_c = _lane_neighbors(
            self.vehicle, cur_wp, self.params, self.mobil
        )
        gap_c = _gap_to_lead(lead_c, ego_len)
        blocked = _is_blocked(lead_c, gap_c)
        # Crawling with no blocker: hold. Exception = pass a stopped lead
        # (friedeggs is_blocked), otherwise MOBIL deadlocks below v_min.
        if v < self.mobil.v_min_mps and not blocked:
            return

        try:
            ego_tf = self.vehicle.get_transform()
        except RuntimeError:
            return
        right = ego_tf.get_right_vector()

        def _lat_to(wp: carla.Waypoint) -> float:
            return abs(
                (ego_loc.x - wp.transform.location.x) * right.x
                + (ego_loc.y - wp.transform.location.y) * right.y
            )

        # Distance to allowed corridor centers (not the projected wp — that is
        # always ~0 after project_to_road and hides off-road / wrong-lane).
        if self._mobil_lane_ids is not None:
            allowed_lats = []
            for lid in self._mobil_lane_ids:
                wp = self._find_lane_wp(cur_wp, lid)
                if wp is not None:
                    allowed_lats.append(_lat_to(wp))
            if not allowed_lats or min(allowed_lats) > 1.25:
                return
            if cur_wp.lane_id not in self._mobil_lane_ids:
                return

        candidates = []
        left = cur_wp.get_left_lane()
        right_ln = cur_wp.get_right_lane()
        if left is not None and left.lane_type == carla.LaneType.Driving:
            if self._mobil_lane_ids is None or left.lane_id in self._mobil_lane_ids:
                candidates.append(("left", left, False))
        if right_ln is not None and right_ln.lane_type == carla.LaneType.Driving:
            if self._mobil_lane_ids is None or right_ln.lane_id in self._mobil_lane_ids:
                candidates.append(("right", right_ln, True))
        if not candidates:
            return

        s0 = _ego_s0(self.params, lead_c, gap_c)
        best = None
        best_tgt = None
        for side, tgt_wp, to_right in candidates:
            contraflow = (
                self._home_lane_id is not None
                and tgt_wp.lane_id * self._home_lane_id < 0
            )
            lead_t, fol_t = _lane_neighbors(
                self.vehicle, tgt_wp, self.params, self.mobil
            )
            if contraflow:
                if not _lane_slot_clear(lead_t, fol_t):
                    continue
                pass_distance = (gap_c or 0.0) + 3.0 * ego_len
                v_blocker = max(lead_c[1], 0.0) if _same_dir_lead(lead_c) else 0.0
                if not _oncoming_pass_clear(lead_t, v, pass_distance, v_blocker):
                    if _env_flag("MOBIL_DEBUG", False):
                        print(
                            f"MOBIL: skip {side} contraflow not clear "
                            f"pass_d={pass_distance:.1f}"
                        )
                    continue
            ok, gain, why = self._mobil_evaluate(
                cur_wp, tgt_wp, to_right, v, s0=s0, contraflow=contraflow
            )
            # IDM singularities near a stopped lead produce huge gains; real
            # MOBIL incentives are O(1). When blocked, a large *positive* gain
            # is the escape signal (friedeggs holds ~8 m standoff so gains stay
            # finite — if we are already nose-in, still allow the pass).
            if ok and abs(gain) > 12.0:
                if blocked and gain > 12.0:
                    why = f"blocked escape gain={gain:.1f}"
                else:
                    ok = False
                    why = f"gain {gain:.2f} unphysical (>12)"
            if ok and (best is None or gain > best[1]):
                best = (side, gain, why)
                best_tgt = tgt_wp

        if best is None or best_tgt is None:
            if _env_flag("MOBIL_DEBUG", False) and candidates:
                bits = []
                for side, tgt_wp, to_right in candidates:
                    contraflow = (
                        self._home_lane_id is not None
                        and tgt_wp.lane_id * self._home_lane_id < 0
                    )
                    ok, gain, why = self._mobil_evaluate(
                        cur_wp, tgt_wp, to_right, v, s0=s0, contraflow=contraflow
                    )
                    bits.append(f"{side}:{'ok' if ok else 'no'}({why})")
                print(
                    f"MOBIL: no commit v={v:.2f} blocked={blocked} "
                    + "; ".join(bits)
                )
            return

        side, gain, why = best
        self._agent.lane_change(
            side,
            same_lane_time=0.05,
            other_lane_time=2.5,
            lane_change_time=1.5,
        )
        self._lc_active = True
        self._lc_target_lane_id = int(best_tgt.lane_id)
        self._lc_target_road_id = int(best_tgt.road_id)
        self._lc_settle_hold_s = 0.0
        self._prev_lat_err = None
        self._mobil_cooldown = self.mobil.min_interval_s
        print(
            f"MOBIL: change {side} ({why}) "
            f"target_lane={self._lc_target_lane_id}"
        )

    def run_step(self, dt: float) -> carla.VehicleControl:
        v = max(0.0, _forward_speed(self.vehicle))
        self._mobil_step(dt, v)

        # Geometric LC commit (not planner queue) drives lead + lateral mode.
        in_lc = self._lc_target_lane_id is not None
        lead_actor = None
        gap: Optional[float] = None
        v_lead = 0.0
        lead_src = "same"
        ego_len = 2.0 * _extent_x(self.vehicle)
        s0 = self.params.min_gap_m

        carla_map = CarlaDataProvider.get_map()
        try:
            ego_loc = self.vehicle.get_location()
        except RuntimeError:
            ego_loc = None
        cur_wp = (
            carla_map.get_waypoint(
                ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if ego_loc is not None
            else None
        )
        lead_lane: Optional[Neighbor] = None
        if cur_wp is not None:
            lead_lane, _ = _lane_neighbors(
                self.vehicle, cur_wp, self.params, self.mobil
            )
            gap_lane = _gap_to_lead(lead_lane, ego_len)
            s0 = _ego_s0(self.params, lead_lane, gap_lane)

        if in_lc:
            # During LC, follow the *target* lane lead so IDM does not full-stop
            # on the old-lane slow vehicle mid-swap.
            tgt_wp = self._resolve_lc_target_wp(cur_wp) if cur_wp is not None else None
            if tgt_wp is not None:
                lead_n, _fol_n = _lane_neighbors(
                    self.vehicle, tgt_wp, self.params, self.mobil
                )
                gap = _gap_to_lead(lead_n, ego_len)
                v_lead = 0.0 if lead_n is None else float(lead_n[1])
                lead_src = f"tgt_lane={self._lc_target_lane_id}"
            else:
                lead_actor, gap, v_lead = find_lead_vehicle(self.vehicle, self.params)
                lead_src = "same_fallback"
        elif lead_lane is not None:
            gap = _gap_to_lead(lead_lane, ego_len)
            v_lead = float(lead_lane[1])
            lead_src = "lane"
        else:
            lead_actor, gap, v_lead = find_lead_vehicle(self.vehicle, self.params)

        accel = idm_acceleration(
            v,
            gap,
            v_lead if gap is not None else None,
            self.params,
            s0=s0,
        )
        throttle, brake = accel_to_control(accel, self.params)
        # Traditional IDM+MOBIL: no reverse. Hold standoff via s0 only.
        if in_lc:
            brake = min(float(brake), 0.45)
            if brake < 1e-3 and throttle < 0.25:
                throttle = max(float(throttle), 0.35 if v < 2.0 else 0.15)
        else:
            brake = min(
                float(brake),
                1.0 if s0 >= BLOCKED_STANDOFF_M - 1e-6 else 0.85,
            )

        control = self._agent.run_step()
        control.throttle = throttle
        control.brake = brake
        control.reverse = False
        max_steer = 0.55 if in_lc else 0.22
        want = self._lane_center_steer(max_steer=max_steer, dt=max(dt, 1e-3))
        dt_u = max(dt, 1e-3)
        max_step = STEER_RATE * dt_u
        steer = max(
            self._prev_steer - max_step, min(self._prev_steer + max_step, want)
        )
        self._prev_steer = float(steer)
        control.steer = steer
        control.hand_brake = False
        control.manual_gear_shift = False

        lead_id = lead_actor.id if lead_actor is not None else None
        gap_s = f"{gap:.1f}" if gap is not None else "inf"
        print(
            f"IDM: v={v:.2f} gap={gap_s} v_lead={v_lead:.2f} "
            f"a={accel:.2f} s0={s0:.1f} lead={lead_id} src={lead_src} "
            f"throttle={control.throttle:.2f} brake={control.brake:.2f} "
            f"steer={control.steer:.3f} mobil={'on' if self.enable_mobil else 'off'}"
            f"{' lc' if in_lc else ''}"
        )
        return control
