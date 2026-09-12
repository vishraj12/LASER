"""TFv6 ego for LASER via a Python 3.10 inference bridge.

LASER (3.8) owns CARLA + sensor callbacks. A long-lived worker in
``third_party/tfv6/.venv`` runs ``build_policy`` / ``act``.

Environment
-----------
  LASER_EGO=tfv6
  TFV6_ROOT / TFV6_PYTHON / TFV6_CHECKPOINT
  TFV6_DUMMY_SENSORS=1   skip CARLA cameras/lidar/radar; send zeros (plumbing)
  TFV6_DEVICE            default cuda:0 if available else cpu (worker-side)
"""

from __future__ import annotations

import math
import os
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import carla
import numpy as np

from laser.target_vehicle.tfv6_bridge import TFv6BridgeClient, _resolve_tfv6_root
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def _default_checkpoint() -> str:
    return str(
        Path.home()
        / "scratch"
        / "scenario_orchestration"
        / "third_party"
        / "checkpoints"
        / "tfv6_cvpr2026"
        / "tfv6_resnet34"
    )


def _world_to_ego_2d(
    point_xy: Sequence[float], ego_xy: Sequence[float], yaw_rad: float
) -> List[float]:
    dx = float(point_xy[0]) - float(ego_xy[0])
    dy = float(point_xy[1]) - float(ego_xy[1])
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return [c * dx + s * dy, -s * dx + c * dy]


def _forward_speed(actor: carla.Actor) -> float:
    vel = actor.get_velocity()
    forward = actor.get_transform().get_forward_vector()
    return float(vel.x * forward.x + vel.y * forward.y + vel.z * forward.z)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


class TFv6EgoController:
    """Closed-loop TFv6 ego: CARLA sensors in 3.8, inference in 3.10."""

    def __init__(
        self,
        vehicle: carla.Vehicle,
        route: Sequence[Tuple[carla.Transform, object]],
        seed: int = 0,
        maneuver: str = "straight",
    ) -> None:
        self.vehicle = vehicle
        self.route = list(route)
        self.seed = seed
        self._maneuver = (maneuver or "straight").lower()
        self._step = 0
        self._dummy = _env_flag("TFV6_DUMMY_SENSORS", default=False)
        self._sensors: List[carla.Sensor] = []
        self._latest: Dict[str, Any] = {}
        self._bridge = TFv6BridgeClient(_resolve_tfv6_root())

        ckpt = os.environ.get("TFV6_CHECKPOINT") or _default_checkpoint()
        if not os.path.isdir(ckpt):
            raise FileNotFoundError(f"TFv6 checkpoint dir missing: {ckpt}")

        device = os.environ.get("TFV6_DEVICE") or "cuda:0"

        town = "Town10HD"
        try:
            town = CarlaDataProvider.get_world().get_map().name.split("/")[-1]
            if town.endswith("_Opt"):
                town = town[: -len("_Opt")]
        except Exception:  # noqa: BLE001
            pass

        request = {
            "schema_version": "1.0.0",
            "name": "tfv6",
            "interface": "ego_policy_v1",
            "implementation": "scenario_orchestration.policy.build_policy",
            "observation_space": "sensor",
            "action_space": "control",
            "repository": str(self._bridge.root),
            "entry_point": "scenario_orchestration/policy.py",
            "checkpoint": ckpt,
            "parameters": {"checkpoint": ckpt, "device": device, "town": town},
            "seed": self.seed,
        }
        info = self._bridge.start(request)
        self._sensor_specs = list(info.get("sensors") or [])
        self._img_h = int(info.get("final_image_height") or 384)
        self._img_w_total = int(info.get("final_image_width") or 1152)
        self._cam_w = max(1, self._img_w_total // 3)
        # Checkpoint trains with radars; policy.act includes them when enabled.
        self._use_radars = bool(info.get("use_radars", True))

        if not self._dummy:
            # Cameras need UE rendering.
            if os.environ.get("LASER_NO_RENDERING") is None:
                print(
                    "TFv6 ego: tip — set LASER_NO_RENDERING=0 for real RGB "
                    "(default headless no_rendering blanks cameras)"
                )
            self._attach_sensors()
            # One sync tick so the first listen callbacks can fire.
            try:
                world = CarlaDataProvider.get_world()
                if world is not None:
                    world.tick()
            except Exception:  # noqa: BLE001
                pass
        else:
            print("TFv6 ego: TFV6_DUMMY_SENSORS=1 — zeroed cameras/lidar/radar")

        print(
            f"TFv6 ego: root={self._bridge.root} ckpt={ckpt} dummy={self._dummy} "
            f"sensors={len(self._sensors)} route_wps={len(self.route)}"
        )

    # -- sensors -----------------------------------------------------------

    def _attach_sensors(self) -> None:
        world = CarlaDataProvider.get_world()
        bp_lib = world.get_blueprint_library()
        weak_self = weakref.ref(self)

        for spec in self._sensor_specs:
            name = str(spec["name"])
            kind = str(spec.get("kind") or "")
            if name.startswith("PCAM") or (not kind and "fov" in spec):
                bp = bp_lib.find("sensor.camera.rgb")
                bp.set_attribute("image_size_x", str(int(spec.get("width", self._cam_w))))
                bp.set_attribute("image_size_y", str(int(spec.get("height", self._img_h))))
                bp.set_attribute("fov", str(float(spec.get("fov", 60.0))))
                transform = carla.Transform(
                    carla.Location(
                        x=float(spec.get("x", 0.0)),
                        y=float(spec.get("y", 0.0)),
                        z=float(spec.get("z", 2.25)),
                    ),
                    carla.Rotation(yaw=float(spec.get("yaw", 0.0))),
                )

                def _on_image(image, name=name, weak=weak_self):
                    self_ref = weak()
                    if self_ref is None:
                        return
                    array = np.frombuffer(image.raw_data, dtype=np.uint8)
                    array = array.reshape((image.height, image.width, 4))[:, :, :3][
                        :, :, ::-1
                    ].copy()
                    self_ref._latest[name] = array

                sensor = world.spawn_actor(bp, transform, attach_to=self.vehicle)
                sensor.listen(_on_image)
                self._sensors.append(sensor)
            elif "lidar" in kind or name == "lidar":
                bp = bp_lib.find("sensor.lidar.ray_cast")
                bp.set_attribute("channels", str(int(spec.get("channels", 64))))
                bp.set_attribute("range", str(float(spec.get("range_m", 100.0))))
                bp.set_attribute(
                    "rotation_frequency",
                    str(float(spec.get("rotation_frequency", 20.0))),
                )
                bp.set_attribute("points_per_second", "256000")
                transform = carla.Transform(
                    carla.Location(
                        x=float(spec.get("x", 1.0)),
                        y=float(spec.get("y", 0.0)),
                        z=float(spec.get("z", 2.5)),
                    )
                )

                def _on_lidar(measurement, name=name, weak=weak_self):
                    self_ref = weak()
                    if self_ref is None:
                        return
                    pts = np.frombuffer(measurement.raw_data, dtype=np.float32)
                    pts = np.reshape(pts, (-1, 4)).copy()
                    self_ref._latest[name] = pts

                sensor = world.spawn_actor(bp, transform, attach_to=self.vehicle)
                sensor.listen(_on_lidar)
                self._sensors.append(sensor)
            elif "radar" in kind or name.startswith("radar"):
                if not self._use_radars:
                    continue
                bp = bp_lib.find("sensor.other.radar")
                bp.set_attribute(
                    "horizontal_fov", str(float(spec.get("horizontal_fov", 90.0)))
                )
                bp.set_attribute(
                    "vertical_fov", str(float(spec.get("vertical_fov", 0.1)))
                )
                bp.set_attribute("range", str(float(spec.get("range_m", 100.0))))
                bp.set_attribute(
                    "points_per_second",
                    str(int(spec.get("points_per_second", 1500))),
                )
                transform = carla.Transform(
                    carla.Location(
                        x=float(spec.get("x", 0.0)),
                        y=float(spec.get("y", 0.0)),
                        z=float(spec.get("z", 0.6)),
                    ),
                    carla.Rotation(yaw=float(spec.get("yaw", 0.0))),
                )

                def _on_radar(measurement, name=name, weak=weak_self):
                    self_ref = weak()
                    if self_ref is None:
                        return
                    # Same as leaderboard SensorInterface._parse_radar_cb:
                    # raw float32 buffer, shape (N, 4). Do NOT convert to xyz —
                    # lead.common.common_utils.radar_points_to_ego expects raw.
                    pts = np.frombuffer(measurement.raw_data, dtype=np.dtype("f4"))
                    pts = np.reshape(pts, (len(measurement), 4)).copy()
                    self_ref._latest[name] = pts

                sensor = world.spawn_actor(bp, transform, attach_to=self.vehicle)
                sensor.listen(_on_radar)
                self._sensors.append(sensor)

    # -- observation -------------------------------------------------------

    def _build_route_ego_frame(self) -> List[List[float]]:
        ego_xy = [
            self.vehicle.get_location().x,
            self.vehicle.get_location().y,
        ]
        yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        world_pts = [
            [float(transform.location.x), float(transform.location.y)]
            for transform, _opt in self.route
        ]
        if not world_pts:
            return [[2.5 + i, 0.0] for i in range(20)]

        start = min(
            range(len(world_pts)),
            key=lambda i: (world_pts[i][0] - ego_xy[0]) ** 2
            + (world_pts[i][1] - ego_xy[1]) ** 2,
        )
        sampled: List[List[float]] = []
        distance = 0.0
        next_sample = 2.5
        previous = ego_xy
        for point in world_pts[start:]:
            distance += math.hypot(point[0] - previous[0], point[1] - previous[1])
            previous = point
            if distance >= next_sample:
                sampled.append(_world_to_ego_2d(point, ego_xy, yaw))
                next_sample += 1.0
            if len(sampled) >= 20:
                break
        while len(sampled) < 20:
            last = sampled[-1] if sampled else [2.5, 0.0]
            if len(sampled) >= 2:
                dx = sampled[-1][0] - sampled[-2][0]
                dy = sampled[-1][1] - sampled[-2][1]
                norm = max(math.hypot(dx, dy), 1e-6)
                step = [dx / norm, dy / norm]
            else:
                step = [1.0, 0.0]
            sampled.append([last[0] + step[0], last[1] + step[1]])
        return sampled[:20]

    def _route_commands(self) -> Tuple[str, str]:
        """Near/far RoadOption names from the shared LASER route (leaderboard-style)."""
        if not self.route:
            return "LANEFOLLOW", "LANEFOLLOW"
        ego_xy = [
            self.vehicle.get_location().x,
            self.vehicle.get_location().y,
        ]
        start = min(
            range(len(self.route)),
            key=lambda i: (
                (self.route[i][0].location.x - ego_xy[0]) ** 2
                + (self.route[i][0].location.y - ego_xy[1]) ** 2
            ),
        )
        ahead = self.route[start : start + 40]
        if not ahead:
            ahead = self.route[-1:]

        def name_of(opt) -> str:
            raw = getattr(opt, "name", None) or str(opt)
            raw = str(raw).upper()
            if "CHANGELANE" in raw:
                return "LANEFOLLOW"
            for label in ("LEFT", "RIGHT", "STRAIGHT", "LANEFOLLOW"):
                if label in raw:
                    return label
            return "LANEFOLLOW"

        near = name_of(ahead[0][1])
        far = near
        for _tf, opt in ahead[1:]:
            label = name_of(opt)
            if label in ("LEFT", "RIGHT", "STRAIGHT"):
                far = label
                break
        else:
            far = name_of(ahead[-1][1])
        # Junction scenarios: planner options are often LANEFOLLOW-only on the
        # dense route; the road flag / ego_maneuver is the intended manoeuvre.
        if self._maneuver == "right":
            far = "RIGHT"
            # After clearing the stop line, the active command is the turn.
            if self.vehicle.get_location().y > -1.0:
                near = "RIGHT"
        elif self._maneuver == "left":
            far = "LEFT"
            if self.vehicle.get_location().y > -1.0:
                near = "LEFT"
        return near, far

    def build_observation(self) -> Dict[str, Any]:
        cameras: Dict[str, Any] = {}
        for spec in self._sensor_specs:
            name = str(spec["name"])
            kind = str(spec.get("kind") or "")
            if name.startswith("PCAM") or (not kind and "fov" in spec):
                h = int(spec.get("height", self._img_h))
                w = int(spec.get("width", self._cam_w))
                img = self._latest.get(name)
                if img is None or self._dummy:
                    cameras[name] = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    cameras[name] = img
            elif name == "lidar" or "lidar" in kind:
                pts = self._latest.get(name)
                if pts is None or self._dummy:
                    cameras[name] = np.zeros((0, 4), dtype=np.float32)
                else:
                    cameras[name] = pts
            elif name.startswith("radar") or "radar" in kind:
                if not self._use_radars:
                    continue
                pts = self._latest.get(name)
                if pts is None or self._dummy:
                    cameras[name] = np.zeros((0, 4), dtype=np.float32)
                else:
                    cameras[name] = pts

        loc = self.vehicle.get_location()
        yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        cmd, next_cmd = self._route_commands()
        return {
            "ego": {
                "speed_mps": max(0.0, _forward_speed(self.vehicle)),
                "x": float(loc.x),
                "y": float(loc.y),
                "yaw_rad": float(yaw),
            },
            "route": self._build_route_ego_frame(),
            "command": cmd,
            "next_command": next_cmd,
            "sensor": {"cameras": cameras},
        }

    # -- control -----------------------------------------------------------

    def run_step(self, dt: float) -> carla.VehicleControl:
        del dt
        self._step += 1
        observation = self.build_observation()
        action = self._bridge.act(observation)
        raw = action.get("control") if isinstance(action, dict) else None
        if not isinstance(raw, dict):
            raise ValueError(f"TFv6 returned no control: {action!r}")
        steer = float(raw["steer"])
        throttle = float(raw["throttle"])
        brake = float(raw["brake"])
        if not all(math.isfinite(v) for v in (steer, throttle, brake)):
            raise ValueError(f"TFv6 non-finite control: {raw!r}")
        control = carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            steer=float(np.clip(steer, -1.0, 1.0)),
            brake=float(np.clip(brake, 0.0, 1.0)),
        )
        if self._step % 10 == 1:
            meta = (action.get("meta") or {}) if isinstance(action, dict) else {}
            print(
                f"TFv6 step={self._step} v={observation['ego']['speed_mps']:.2f} "
                f"cmd={meta.get('command')} "
                f"throttle={control.throttle:.2f} brake={control.brake:.2f} "
                f"steer={control.steer:.2f}"
            )
        return control

    def close(self) -> None:
        for sensor in self._sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:  # noqa: BLE001
                pass
        self._sensors = []
        self._bridge.close()

