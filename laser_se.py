import os
import json
import carla
import logging
import argparse

from langchain_openai import ChatOpenAI
from laser.agent_manager import AgentManager


args = None
client = None
carla_world = None
carla_map = None
agent_manager = None

lane_wps = None
driving_lane_num = None

def get_wp(x, y, z):
    global carla_map
    return carla_map.get_waypoint(carla.Location(x, y, z), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))


def find_opposite_driving_wp(wp):
    """Walk left/right until a Driving lane with opposite lane_id sign."""
    for starter in (wp.get_left_lane(), wp.get_right_lane()):
        cur = starter
        for _ in range(6):
            if cur is None:
                break
            if cur.lane_type == carla.LaneType.Driving and cur.lane_id * wp.lane_id < 0:
                return cur
            # keep walking across same-direction lanes
            nxt = cur.get_left_lane() if starter == wp.get_left_lane() else cur.get_right_lane()
            if nxt is None:
                nxt = cur.get_right_lane() if starter == wp.get_left_lane() else cur.get_left_lane()
            cur = nxt
    return None


def hold_light_states(world, states, tries=5):
    """Freeze every light and set `states` ({light id: (light, state)}) so they take.

    This runs straight after load_world, while the world is asynchronous and the
    town's own light cycle still runs: a state set before the new episode's first
    tick can be overwritten by that cycle and then frozen. With
    setup_junction189_lights' old order on CARLA 0.9.16 the ego's green was lost
    in 4 of 8 loads of Town10HD_Opt (every light of junction 189 red for the whole
    run; 3 of 7 replays and one red-light run), and held in all 6 that waited one
    tick first. So: wait for a tick, freeze, set, wait for the next tick, read
    the states back, and set them again if any did not take. Staged this way the
    ego's green held in 6 of 6 loads.
    """
    def next_tick():
        if world.get_settings().synchronous_mode:
            world.tick()
        else:
            world.wait_for_tick()

    next_tick()
    world.freeze_all_traffic_lights(True)
    for _ in range(tries):
        for tl, state in states.values():
            tl.set_state(state)
        next_tick()
        missed = [tid for tid, (tl, state) in states.items() if tl.get_state() != state]
        if not missed:
            return True
        print(f"WARN lights: {missed} did not take; setting them again")
    print(f"WARN lights: {missed} still not as set after {tries} tries")
    return False


def setup_junction189_lights(world, ego_wp, cross_wp):
    """Freeze lights: ego approach green, crossing approach red (SafeBench red-light)."""

    def best_light_for_approach(approach_wp, max_dist=40.0):
        best = None
        best_d = max_dist
        ax = approach_wp.transform.location.x
        ay = approach_wp.transform.location.y
        for tl in world.get_actors().filter('traffic.traffic_light*'):
            try:
                stop_wps = tl.get_stop_waypoints()
            except Exception:
                continue
            for sw in stop_wps or []:
                # Same travel direction and near the approach.
                yaw_d = abs((sw.transform.rotation.yaw - approach_wp.transform.rotation.yaw + 180) % 360 - 180)
                if yaw_d > 45.0:
                    continue
                d = ((sw.transform.location.x - ax) ** 2 + (sw.transform.location.y - ay) ** 2) ** 0.5
                if d < best_d:
                    best_d = d
                    best = tl
        return best

    ego_light = best_light_for_approach(ego_wp)
    cross_light = best_light_for_approach(cross_wp)

    states = {tl.id: (tl, carla.TrafficLightState.Red)
              for tl in world.get_actors().filter('traffic.traffic_light*')}
    if ego_light is not None:
        states[ego_light.id] = (ego_light, carla.TrafficLightState.Green)
        print(f"T10J189 lights: ego approach TL {ego_light.id} → Green")
    else:
        print("WARN T10J189: could not find ego approach traffic light")

    if cross_light is not None:
        states[cross_light.id] = (cross_light, carla.TrafficLightState.Red)
        print(f"T10J189 lights: cross approach TL {cross_light.id} → Red")
    else:
        print("WARN T10J189: could not find cross approach traffic light")
    hold_light_states(world, states)


def setup_junction189_right_lights(world, ego_wp, cross_wp):
    """Right-turn lights: ego approach GREEN, crossing approach RED.

    The intersection method stages every junction family under one signal plan
    (orchestration/carla_port/scenarios.py: the ego's N/S arms green, E/W red),
    so its right-turning ego has the green and the crossing actor runs a red.
    This mode used to give the ego the red, which a light-obeying ego policy
    (SimLingo, TFv6, PlanT2) answers by stopping at the line and never turning.
    """
    def best_light_for_approach(approach_wp):
        best = None
        best_d = 1e9
        ax = approach_wp.transform.location.x
        ay = approach_wp.transform.location.y
        for tl in world.get_actors().filter('traffic.traffic_light*'):
            try:
                stop_wps = tl.get_stop_waypoints()
            except Exception:
                continue
            for sw in stop_wps or []:
                yaw_d = abs((sw.transform.rotation.yaw - approach_wp.transform.rotation.yaw + 180) % 360 - 180)
                if yaw_d > 45.0:
                    continue
                d = ((sw.transform.location.x - ax) ** 2 + (sw.transform.location.y - ay) ** 2) ** 0.5
                if d < best_d:
                    best_d = d
                    best = tl
        return best

    ego_light = best_light_for_approach(ego_wp)
    cross_light = best_light_for_approach(cross_wp)

    states = {tl.id: (tl, carla.TrafficLightState.Red)
              for tl in world.get_actors().filter('traffic.traffic_light*')}
    if ego_light is not None:
        states[ego_light.id] = (ego_light, carla.TrafficLightState.Green)
        print(f"T10J189Right lights: ego approach TL {ego_light.id} → Green")
    else:
        print("WARN T10J189Right: could not find ego approach traffic light")

    if cross_light is not None:
        states[cross_light.id] = (cross_light, carla.TrafficLightState.Red)
        print(f"T10J189Right lights: cross approach TL {cross_light.id} → Red")
    else:
        print("WARN T10J189Right: could not find cross approach traffic light")
    hold_light_states(world, states)


def init_world():
    global client, carla_world, carla_map, lane_wps, driving_lane_num, ego_maneuver

    # Automatic VUT mission (straight / left / right). Not from script.json.
    ego_maneuver = "straight"
    client.reload_world()
    carla_world = client.get_world()
    if args.road == 'T04Highway':
        carla_world = client.load_world("Town04")
        carla_map = carla_world.get_map()
        # T04Highway
        lane_wps = [get_wp(4.671636, -68, 0), get_wp(8.171656, -68, 0), get_wp(11.671675, -68, 0), get_wp(15.171694, -68, 0)]
        driving_lane_num = 4
    elif args.road == 'T05Highway':
        carla_world = client.load_world("Town05")
        carla_map = carla_world.get_map()
        # T05Highway
        lane_wps = [get_wp(105.0, 194.0, 0), get_wp(105.0, 191.0, 0), get_wp(105.0, 188.0, 0)]
        driving_lane_num = 3
    elif args.road == 'T06Highway':
        carla_world = client.load_world("Town06")
        carla_map = carla_world.get_map()
        # T06Highway
        lane_wps = [get_wp(x=544.0, y=38.2, z=0), get_wp(x=544.0, y=41.7, z=0), get_wp(x=544.0, y=45.2, z=0), get_wp(x=544.0, y=48.7, z=0), get_wp(x=544.0, y=52.1, z=0)]
        driving_lane_num = 5
    elif args.road == 'T10Urban1':
        carla_world = client.load_world("Town10HD")
        carla_map = carla_world.get_map()
        # Town10Urban1
        lane_wps = [get_wp(106, 35, 0), get_wp(109, 35, 0), get_wp(117, 35, 0)] # bus stop at (109, 55.5)
        driving_lane_num = 2
        # bus_stop_pos = 20.5 # 150 - (20.5 + 3.5) = 126
    elif args.road == 'T04CutIn':
        # Town04 OpenDRIVE road 47 — orchestration highway cut-in corridor
        # (exp006 discover: 3 same-dir lanes @ 3.5 m, travel yaw≈90°).
        # Road 47 has 4 driving lanes (ids +3..+6); use the inner three that
        # match the highway port's 3-lane cut-in fit (ids +3/+4/+5).
        # Cross-section y≈-38 sits south of the road-47 overpass band.
        carla_world = client.load_world("Town04")
        carla_map = carla_world.get_map()
        lane_wps = [
            get_wp(-5.712, -38.414, 0),   # left  (carla lane_id +3)
            get_wp(-9.212, -38.400, 0),   # center (carla lane_id +4) — VUT
            get_wp(-12.712, -38.386, 0),  # right (carla lane_id +5) — cut-in
        ]
        driving_lane_num = 3
        for i, wp in enumerate(lane_wps):
            loc = wp.transform.location
            print(
                f"T04CutIn lane{i+1} road={wp.road_id} lane_id={wp.lane_id} "
                f"loc=({loc.x:.2f},{loc.y:.2f}) yaw={wp.transform.rotation.yaw:.1f}"
            )
            if wp.road_id != 47:
                print(f"WARN T04CutIn: expected road 47, got {wp.road_id}")
    elif args.road == 'T04HardBrake':
        # CARLA stand-in for scenario_hard_brake.yaml:
        #   map: {kind: straight, num_lanes: 2, lane_width: 3.5, length: 160}
        # YAML script frame (heading north): ego (1.75,-60)@12, slow (1.75,-30)@4,
        # adjacent (-1.75,-48)@10.5. We do NOT place at those absolute world
        # coords; we match lane count + relative gaps/speeds on Town04 road 47
        # (2 same-dir lanes: id +4 left / +5 ego-right).
        # Anchor at the SOUTH end (~y=-45.9). previous() from the cut-in
        # y≈-38 cross-section jumps onto road 775/48 — keep everyone on road 47
        # and place ahead actors with negative init_state (next()).
        # Route look-ahead: road 47 offers ~112 m forward from these anchors
        # before joining another road. Use 100 m (not 160) so the plan stays
        # on-road; YAML's 160 m length is the abstract corridor, not a single
        # OpenDRIVE road_id.
        carla_world = client.load_world("Town04")
        carla_map = carla_world.get_map()
        lane_wps = [
            get_wp(-9.242, -45.87, 0),   # left / adjacent (carla lane_id +4)
            get_wp(-12.742, -45.86, 0),  # right / ego     (carla lane_id +5)
        ]
        driving_lane_num = 2
        os.environ["LASER_ROUTE_LOOKAHEAD_M"] = os.environ.get(
            "LASER_ROUTE_LOOKAHEAD_M", "100"
        )
        for i, wp in enumerate(lane_wps):
            loc = wp.transform.location
            print(
                f"T04HardBrake lane{i+1} road={wp.road_id} lane_id={wp.lane_id} "
                f"loc=({loc.x:.2f},{loc.y:.2f}) yaw={wp.transform.rotation.yaw:.1f}"
            )
            if wp.road_id != 47:
                print(f"WARN T04HardBrake: expected road 47, got {wp.road_id}")
    elif args.road == 'T10Urban2':
        carla_world = client.load_world("Town10HD")
        carla_map = carla_world.get_map()
        # Town10Urban2
        lane_wps = [get_wp(10.0, -64.7, 0), get_wp(10.0, -68.2, 0), get_wp(10.0, -75.6, 0)] # bus stop at (54.1, -68.2)
        driving_lane_num = 2
        # bus_stop_pos = 44.1 # 150 - (44.1 + 3.5) = 102.4
    elif args.road == 'T01Overtake':
        # Table pin: Town01 road 8, 2-way, lane width 4.0 m.
        # lane +1 northbound (ego / blocker), lane -1 southbound (oncoming).
        # Left of +1 is -1 (true opposing Driving lane).
        carla_world = client.load_world("Town01")
        carla_map = carla_world.get_map()
        inbound = get_wp(392.29, 100.54, 0)   # road 8, lane +1, yaw≈90
        opposite = find_opposite_driving_wp(inbound)
        if opposite is None:
            raise RuntimeError("T01Overtake: no opposite Driving lane near (392.29,100.54)")
        if inbound.road_id != 8 or abs(inbound.lane_width - 4.0) > 0.05:
            print(
                f"WARN T01Overtake: expected road 8 width 4.0, "
                f"got road={inbound.road_id} width={inbound.lane_width:.2f}"
            )
        print(
            f"T01Overtake inbound road={inbound.road_id} lane_id={inbound.lane_id} "
            f"width={inbound.lane_width:.2f} yaw={inbound.transform.rotation.yaw:.1f} "
            f"loc=({inbound.transform.location.x:.2f},{inbound.transform.location.y:.2f})"
        )
        print(
            f"T01Overtake opposite road={opposite.road_id} lane_id={opposite.lane_id} "
            f"width={opposite.lane_width:.2f} yaw={opposite.transform.rotation.yaw:.1f} "
            f"loc=({opposite.transform.location.x:.2f},{opposite.transform.location.y:.2f})"
        )
        lane_wps = [inbound, opposite]
        driving_lane_num = 2
        # Enough corridor to approach blocker + start the pass (~308 m road).
        os.environ["LASER_ROUTE_LOOKAHEAD_M"] = os.environ.get(
            "LASER_ROUTE_LOOKAHEAD_M", "150"
        )
    elif args.road == 'T04VehiclePassing':
        carla_world = client.load_world("Town04")
        carla_map = carla_world.get_map()
        # Legacy Town04 ManeuverOppositeDirection_2 pin (superseded by T01Overtake).
        inbound = get_wp(63.3, -190.3, 0)
        opposite = find_opposite_driving_wp(inbound)
        if opposite is None:
            raise RuntimeError("T04VehiclePassing: no opposite Driving lane found near (63.3,-190.3)")
        print(f"T04VehiclePassing inbound lane_id={inbound.lane_id} yaw={inbound.transform.rotation.yaw:.1f}")
        print(f"T04VehiclePassing opposite lane_id={opposite.lane_id} yaw={opposite.transform.rotation.yaw:.1f}")
        lane_wps = [inbound, opposite]
        driving_lane_num = 2
    elif args.road == 'T10VehiclePassing':
        carla_world = client.load_world("Town10HD")
        carla_map = carla_world.get_map()
        # Ego inbound near T10Urban1 corridor; lane 2 = true opposite carriageway.
        inbound = get_wp(106, 35, 0)
        opposite = find_opposite_driving_wp(inbound)
        if opposite is None:
            raise RuntimeError("T10VehiclePassing: no opposite Driving lane found near (106,35)")
        print(f"T10VehiclePassing inbound lane_id={inbound.lane_id} yaw={inbound.transform.rotation.yaw:.1f}")
        print(f"T10VehiclePassing opposite lane_id={opposite.lane_id} yaw={opposite.transform.rotation.yaw:.1f}")
        lane_wps = [inbound, opposite]
        driving_lane_num = 2
    elif args.road == 'T10J189':
        # Town10HD_Opt junction 189 — conflict ≈ (-48.7, 24.5).
        # lane 1: ego south→north (straight through)
        # lane 2: crossing west→east (red-light runner)
        ego_maneuver = "straight"
        carla_world = client.load_world("Town10HD_Opt")
        carla_map = carla_world.get_map()
        ego_approach = get_wp(-48.8, -17.5, 0)   # ~40 m south of conflict, yaw≈90
        cross_approach = get_wp(-68.7, 24.5, 0)  # ~20 m west of conflict, yaw≈0
        if ego_approach.is_junction or cross_approach.is_junction:
            print(f"WARN T10J189: approach wp in junction "
                  f"(ego_junc={ego_approach.is_junction}, cross_junc={cross_approach.is_junction})")
        print(f"T10J189 ego   lane_id={ego_approach.lane_id} yaw={ego_approach.transform.rotation.yaw:.1f} "
              f"loc=({ego_approach.transform.location.x:.1f},{ego_approach.transform.location.y:.1f})")
        print(f"T10J189 cross lane_id={cross_approach.lane_id} yaw={cross_approach.transform.rotation.yaw:.1f} "
              f"loc=({cross_approach.transform.location.x:.1f},{cross_approach.transform.location.y:.1f})")
        lane_wps = [ego_approach, cross_approach]
        driving_lane_num = 2
        setup_junction189_lights(carla_world, ego_approach, cross_approach)
    elif args.road == 'T10J189Right':
        # Right turn at junction 189. CARLA is left-handed: the ego heads +y
        # (yaw 90), so its RIGHT is -x, and the right-turn connector (onto road
        # 19, yaw 180) leaves only from the OUTER lane at x=-52.3; the inner lane
        # at x=-48.8 offers straight and LEFT (road 375 -> road 20, eastbound),
        # which is what this mode used to turn into. A right turn joins the
        # near-side westbound lane (y~13), so the stream it conflicts with is
        # westbound through traffic arriving from the ego's left (+x); it starts
        # east of the box, 20 m before the anchor below (script init_state -20).
        # Maneuver comes from this road flag (not VUT.route in script.json).
        ego_maneuver = "right"
        carla_world = client.load_world("Town10HD_Opt")
        carla_map = carla_world.get_map()
        # Ego closer to the stop line than red-light (figure: at the line).
        ego_approach = get_wp(-52.3, -5.0, 0)
        cross_approach = get_wp(0.0, 13.1, 0)
        if ego_approach.is_junction or cross_approach.is_junction:
            print(f"WARN T10J189Right: approach wp in junction "
                  f"(ego_junc={ego_approach.is_junction}, cross_junc={cross_approach.is_junction})")
        print(f"T10J189Right ego   lane_id={ego_approach.lane_id} yaw={ego_approach.transform.rotation.yaw:.1f} "
              f"loc=({ego_approach.transform.location.x:.1f},{ego_approach.transform.location.y:.1f})")
        print(f"T10J189Right cross lane_id={cross_approach.lane_id} yaw={cross_approach.transform.rotation.yaw:.1f} "
              f"loc=({cross_approach.transform.location.x:.1f},{cross_approach.transform.location.y:.1f})")
        lane_wps = [ego_approach, cross_approach]
        driving_lane_num = 2
        setup_junction189_right_lights(carla_world, ego_approach, cross_approach)
    elif args.road == 'T10J189Left':
        # SafeBench unprotected left-turn: ego south→north then LEFT onto westbound.
        ego_maneuver = "left"
        carla_world = client.load_world("Town10HD_Opt")
        carla_map = carla_world.get_map()
        ego_approach = get_wp(-48.8, -17.5, 0)
        north_ref = get_wp(-48.8, 45, 0)
        oncoming_approach = find_opposite_driving_wp(north_ref)
        if oncoming_approach is None:
            oncoming_approach = north_ref
            print("WARN T10J189Left: find_opposite_driving_wp failed; using north_ref")
        if ego_approach.is_junction or oncoming_approach.is_junction:
            print(f"WARN T10J189Left: approach wp in junction "
                  f"(ego_junc={ego_approach.is_junction}, oncoming_junc={oncoming_approach.is_junction})")
        print(f"T10J189Left ego   lane_id={ego_approach.lane_id} yaw={ego_approach.transform.rotation.yaw:.1f} "
              f"loc=({ego_approach.transform.location.x:.1f},{ego_approach.transform.location.y:.1f})")
        print(f"T10J189Left oncoming lane_id={oncoming_approach.lane_id} yaw={oncoming_approach.transform.rotation.yaw:.1f} "
              f"loc=({oncoming_approach.transform.location.x:.1f},{oncoming_approach.transform.location.y:.1f})")
        lane_wps = [ego_approach, oncoming_approach]
        driving_lane_num = 2
        hold_light_states(carla_world, {
            tl.id: (tl, carla.TrafficLightState.Green)
            for tl in carla_world.get_actors().filter('traffic.traffic_light*')})
        print("T10J189Left lights: all frozen Green (unprotected left + oncoming through)")
    elif args.road == 'T05Urban':
        carla_world = client.load_world("Town05")
        carla_map = carla_world.get_map()
        # T05Urban
        lane_wps = [get_wp(-66.0, -87.7, 0), get_wp(-66.0, -84.7, 0), get_wp(-66.0, -81.7, 0)] # bus stop at (-84.6, -84.7)
        driving_lane_num = 2
        # bus_stop_pos = 18.6 # 150 - (18.6 + 3.5) = 128

    print(f"init_world ego_maneuver={ego_maneuver} road={args.road}")


def init_agents():
    global client, carla_world, carla_map, lane_wps, driving_lane_num, agent_manager, ego_maneuver

    openai_llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

    script = json.load(open(args.script, 'r'))
    print("Load script: ")
    print(script)
    
    simulation_time = args.time
    agent_manager = AgentManager(
        client,
        carla_world,
        simulation_time,
        script,
        lane_wps,
        driving_lane_num,
        openai_llm,
        ego_maneuver=ego_maneuver,
    )

def destroy():
    global agent_manager
    
    agent_manager.destroy(None)

if __name__ == "__main__":
    logging.basicConfig(filename='run.log', level=logging.DEBUG, filemode='w')

    parser = argparse.ArgumentParser(
        description='Script Execution')
    parser.add_argument("--host", 
                        default='127.0.0.1',
                        help="carla host ip (default: 127.0.0.1)")
    parser.add_argument("-p", "--port", 
                        default=2000,
                        type=int,
                        help="carla host port (default: 2000)")
    parser.add_argument("-r", "--road", 
                        type=str,
                        help="select road segment: T04Highway, T04CutIn, T04HardBrake, T01Overtake, T05Highway, T06Highway, T10Urban1, T10Urban2, T05Urban, T04VehiclePassing, T10VehiclePassing, T10J189, T10J189Right, T10J189Left", required=True)
    parser.add_argument("-s", "--script", 
                        type=str,
                        help="path/to/script.json")
    parser.add_argument("-t", "--time", 
                        default=10,
                        type=int,
                        help="simulation time")

    args = parser.parse_args()

    # IDM+MOBIL only for Merge/hard-brake and Overtake roads.
    # Cut-in / turns / red-light stay IDM-only — force MOBIL off even if the
    # shell still has IDM_ENABLE_MOBIL=1 from a previous run.
    _mobil_roads = {
        "T04HardBrake", "T04Highway", "T01Overtake", "T04VehiclePassing", "T10VehiclePassing",
    }
    if os.environ.get("LASER_EGO", "").lower() == "idm":
        if args.road in _mobil_roads:
            os.environ.setdefault("IDM_ENABLE_MOBIL", "1")
        else:
            os.environ["IDM_ENABLE_MOBIL"] = "0"

    client = carla.Client(args.host, args.port)
    client.set_timeout(20)

    init_world()
    print('carla init finished')
    init_agents()
    print('agent init finished')

    try:
        agent_manager.run_scenario()
    except KeyboardInterrupt:
        print('\nCancelled')
    except RuntimeError as e:
        print(f'\nCARLA runtime error: {e}')
    finally:
        try:
            destroy()
        except Exception as e:
            print(f'teardown failed: {e}')
    
