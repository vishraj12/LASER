import logging
import os
import json
import asyncio
import time
import carla
from queue import Queue
from laser.laser_agents import Agent

import numpy as np
from laser.target_vehicle.dummy_target_vehicle import DummyTargetVehicle
from laser.target_vehicle.target_vehicle import TargetVehicle

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from laser.target_vehicle.target_vehicle_recorder import TargetVehicleRecorder

time_delta = 0.05 # time per frame
time_query = 0.5 # time per llm query

class AgentManager:
    def __init__(
        self,
        client,
        world,
        simulation_time,
        scene_script,
        lane_wps,
        driving_lane_num,
        llm,
        ego_maneuver="straight",
    ) -> None:
        self._timeout = 20
        self.client = client
        self.world = world
        self.simulation_time = simulation_time
        self.ego_maneuver = (ego_maneuver or "straight").lower()
        self.driving_lane_num = driving_lane_num
        print(f"AgentManager ego_maneuver={self.ego_maneuver}")
        
        settings = world.get_settings()
        # https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#possible-configurations
        # Synchronous mode + fixed time-step
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = time_delta
        # Headless smokes: skip UE rendering when no bird's-eye camera is attached.
        # (RGB video needs rendering — leave LASER_VIDEO=1 + LASER_NO_RENDERING=0.)
        _video = os.environ.get("LASER_VIDEO", "0").strip().lower() not in ("0", "false", "no", "off", "")
        _no_render_env = os.environ.get("LASER_NO_RENDERING")
        if _no_render_env is None:
            settings.no_rendering_mode = not _video
        else:
            settings.no_rendering_mode = _no_render_env.strip().lower() not in (
                "0", "false", "no", "off", ""
            )
        print(
            f"AgentManager sync dt={time_delta}s no_rendering={settings.no_rendering_mode} "
            f"video={'on' if _video else 'off'}"
        )
        world.apply_settings(settings)

        world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))
        time.sleep(1)
        self._stride = int(time_query / time_delta)
        self._stride_cnt = self._stride - 1

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)

        self.target_recorder = TargetVehicleRecorder(client)
        
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()
        
        # Init LASER agents
        self.sensor_queue = Queue()
        self._vehicles = []
        self._pedestrians = []
        self._target_vehicle = []
        # Stock LASER: spawn in script key order (VUT first when listed first).
        for agent_name, agent_script in scene_script.items():
            if agent_script["type"] == 'VUT':
                self._target_vehicle.append(TargetVehicle(world, agent_name, agent_script, lane_wps, self, self.sensor_queue))
            elif agent_script["type"] == 'dummy':
                self._vehicles.append(DummyTargetVehicle(world, agent_name, agent_script, lane_wps, driving_lane_num, self, self.sensor_queue))
            elif agent_script["type"] == 'agent':
                agent = Agent(world, agent_name, agent_script, lane_wps, driving_lane_num, llm, self, self.sensor_queue)
                if agent.type == 'vehicle':
                    self._vehicles.append(agent)
                elif agent.type == 'pedestrian':
                    self._pedestrians.append(agent)

        world.tick() # for actors to spawn, and then their locations are correct
        self.agents = self._target_vehicle + self._vehicles + self._pedestrians
        for agent in self.agents:
            agent.init_after_carla_tick()
        for _ in range(9): 
            world.tick() # for setting velocity
        
        self.spectator_chase()

        self.target_recorder.register_actor(self.agents[0].carla_actor)

        self.logger = logging.getLogger(__name__)

        # TEMP: dense pose/control trace → se_records/<run>/trajectories.json
        # Disable with LASER_TRAJ_JSON=0
        self._traj_enabled = os.environ.get("LASER_TRAJ_JSON", "1") != "0"
        self._traj = {agent.name: [] for agent in self.agents} if self._traj_enabled else {}
        if self._traj_enabled:
            print(f"TEMP traj JSON enabled → {self.target_recorder.directory}/trajectories.json")

    def parse_sensor_data(self, dt):
        self.logger.debug(self.sensor_queue.qsize())
        self.handle_sensor_data(dt)
        while (self.sensor_queue.qsize() > 0):
            sensor_data, agent = self.sensor_queue.get()
            if type(sensor_data) == carla.CollisionEvent:
                self.target_recorder.parse_CollisionEvent(sensor_data, agent)

    async def get_decisions(self, dt):
        decisions = [agent.get_decisions(None) for agent in self.agents]
        results = await asyncio.gather(*decisions)
        return results
        
    def handle_sensor_data(self, dt):
        results = asyncio.run(self.get_decisions(dt))
        # print(results)
        for agent, decisions in zip(self.agents, results):
            agent.handle_decisions(decisions, dt)

    def track(self, dt):
        for agent in self.agents:
            agent.on_tick(dt)

    def destroy(self, SW_token):
        # token usage
        
        with open(os.path.join(self.target_recorder.directory, 'token.txt'), 'w') as file:
            token_usage = {
                "SW": SW_token,
            }
            for agent in self.agents:
                if agent._llm_agent is not None:
                    token_usage[agent.name] = agent._llm_agent_inst.usage_metadata
            json_object = json.dumps(token_usage, indent=4)
            file.write(json_object)

        with open(os.path.join(self.target_recorder.directory, 'time.txt'), 'w') as file:
            time_usage = {
                "simulation_time": self._timestamp_last_run - self._timestamp_start,
                "real_world_time": time.time() - self.start_system_time,
            }
            json_object = json.dumps(time_usage, indent=4)
            file.write(json_object)

        if self._traj_enabled:
            try:
                traj_path = os.path.join(self.target_recorder.directory, "trajectories.json")
                payload = {
                    "temp": True,
                    "dt": time_delta,
                    "simulation_time": self._timestamp_last_run - self._timestamp_start,
                    "actors": self._traj,
                }
                with open(traj_path, "w") as file:
                    json.dump(payload, file)
                print(f"TEMP wrote {traj_path} ({sum(len(v) for v in self._traj.values())} samples)")
            except Exception as e:
                print(f"TEMP traj JSON write failed: {e}")

        # destroy
        print(f"Finish time: {self._timestamp_last_run}")
        try:
            self.target_recorder.destroy()
        except Exception as e:
            print(f"recorder destroy failed: {e}")
        for agent in self.agents:
            try:
                agent.destroy()
            except Exception as e:
                print(f"agent {getattr(agent, 'name', '?')} destroy failed: {e}")



    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()

        self._running = True

        world = CarlaDataProvider.get_world()
        snapshot = world.get_snapshot()
        self._timestamp_start = snapshot.timestamp.elapsed_seconds
        self._timestamp_last_run = self._timestamp_start
        print(f"Start time: {self._timestamp_start}")

        while self._timestamp_last_run - self._timestamp_start <= self.simulation_time - time_delta: 
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self._stride_cnt += 1
            if self._stride_cnt == self._stride:
                self._stride_cnt = 0
                self.parse_sensor_data(time_delta)
            self.track(time_delta)
            if self._traj_enabled:
                self._record_traj_frame(timestamp.elapsed_seconds)

            self.spectator_chase()


        if self._running:
            CarlaDataProvider.get_world().tick(self._timeout)

    def _record_traj_frame(self, sim_t):
        """TEMP: append one pose/control sample per agent."""
        t = float(sim_t - self._timestamp_start)
        for agent in self.agents:
            actor = getattr(agent, "carla_actor", None)
            if actor is None:
                continue
            try:
                tf = actor.get_transform()
                loc = tf.location
                vel = actor.get_velocity()
                speed = float((vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5)
                ctrl = actor.get_control()
                sample = {
                    "t": round(t, 4),
                    "x": round(loc.x, 3),
                    "y": round(loc.y, 3),
                    "z": round(loc.z, 3),
                    "yaw": round(tf.rotation.yaw, 2),
                    "speed": round(speed, 3),
                    "throttle": round(float(ctrl.throttle), 3),
                    "brake": round(float(ctrl.brake), 3),
                    "steer": round(float(ctrl.steer), 3),
                }
            except Exception:
                continue
            self._traj.setdefault(agent.name, []).append(sample)

    def spectator_chase(self):
        chased_actor = self.agents[0].carla_actor
        spectator = CarlaDataProvider.get_world().get_spectator()
        trans = chased_actor.get_transform()
        spectator.set_transform(carla.Transform(trans.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

