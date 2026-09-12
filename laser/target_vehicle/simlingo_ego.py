"""SimLingo ego for LASER via a torch-2.x inference bridge.

Attaches the Leaderboard forward camera (1024x512 fov-110), builds an
``ego_policy_v1`` sensor observation, and calls SimLingo through
``simlingo_bridge`` (separate venv — LASER stays on torch 1.13).

Environment
-----------
  LASER_EGO=simlingo
  SIMLINGO_ROOT / SIMLINGO_PYTHON / SIMLINGO_CHECKPOINT
  SIMLINGO_WORKDIR       cwd for pretrained/InternVL2-1B cache
  AV_CKPT                parent of simlingo/ + optional pretrained/
  CARLA_ROOT             CARLA tree with PythonAPI/carla/agents
  LASER_NO_RENDERING=0   required for real RGB (default headless blanks cameras)
  SIMLINGO_DUMMY_SENSORS=1  plumbing-only zeros
"""

from __future__ import annotations

import math
import os
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import carla
import numpy as np

from laser.target_vehicle.simlingo_bridge import (
    SimLingoBridgeClient,
    _default_checkpoint,
    _resolve_simlingo_root,
)
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


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


class SimLingoEgoController:
    """Closed-loop SimLingo ego: CARLA camera in LASER, inference in worker."""

    def __init__(
        self,
        vehicle: carla.Actor,
        route: Sequence,
        seed: int = 0,
    ) -> None:
        self.vehicle = vehicle
        self.route = list(route)
        self.seed = int(seed)
        self._dummy = _env_flag("SIMLINGO_DUMMY_SENSORS", default=False)
        self._step = 0
        self._sensors: List[carla.Sensor] = []
        self._latest: Dict[str, Any] = {}
        self._bridge = SimLingoBridgeClient(_resolve_simlingo_root())

        ckpt = os.environ.get("SIMLINGO_CHECKPOINT") or _default_checkpoint()
        weights = os.environ.get("SIMLINGO_WEIGHTS") or str(
            Path(ckpt) / "checkpoints" / "epoch=013.ckpt" / "pytorch_model.pt"
        )
        request: Dict[str, Any] = {
            "name": "simlingo",
            "interface": "ego_policy_v1",
            "observation_space": "sensor",
            "action_space": "control",
            "checkpoint": ckpt,
            "seed": self.seed,
            "parameters": {
                "checkpoint": ckpt,
                "weights": weights,
                "device": os.environ.get("SIMLINGO_DEVICE", "cuda:0"),
                "mode_token": os.environ.get("SIMLINGO_MODE_TOKEN", "<SAFETY>"),
                "instruction": os.environ.get("SIMLINGO_INSTRUCTION", ""),
            },
            "repository": str(self._bridge.root),
        }
        info = self._bridge.start(request)
        self._sensor_specs = list(info.get("sensors") or [])
        if not self._sensor_specs:
            self._sensor_specs = [
                {
                    "name": "rgb_front",
                    "width": 1024,
                    "height": 512,
                    "fov": 110.0,
                    "x": -1.5,
                    "y": 0.0,
                    "z": 2.0,
                }
            ]

        if not self._dummy:
            self._attach_sensors()
            if _env_flag("LASER_NO_RENDERING", default=True):
                print(
                    "WARN SimLingo: LASER_NO_RENDERING is on — cameras may be black. "
                    "Set LASER_NO_RENDERING=0 for real RGB."
                )
        else:
            print("SimLingo ego: SIMLINGO_DUMMY_SENSORS=1 — zeroed camera")

        print(
            f"SimLingo ego: root={self._bridge.root} ckpt={ckpt} dummy={self._dummy}",
            flush=True,
        )

    def _attach_sensors(self) -> None:
        world = CarlaDataProvider.get_world()
        bp_lib = world.get_blueprint_library()
        weak_self = weakref.ref(self)
        for spec in self._sensor_specs:
            name = str(spec.get("name") or "rgb_front")
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(int(spec.get("width", 1024))))
            bp.set_attribute("image_size_y", str(int(spec.get("height", 512))))
            bp.set_attribute("fov", str(float(spec.get("fov", 110.0))))
            transform = carla.Transform(
                carla.Location(
                    x=float(spec.get("x", -1.5)),
                    y=float(spec.get("y", 0.0)),
                    z=float(spec.get("z", 2.0)),
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

    def _target_points_ego(self) -> List[List[float]]:
        """Near/far route targets in ego frame (SimLingo expects two points)."""
        ego_xy = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        world_pts = [
            [float(t.location.x), float(t.location.y)] for t, _opt in self.route
        ]
        if not world_pts:
            return [[10.0, 0.0], [20.0, 0.0]]
        # pick points ~10 m and ~20 m ahead along remaining route
        best_near, best_far = None, None
        for pt in world_pts:
            local = _world_to_ego_2d(pt, ego_xy, yaw)
            if local[0] < 2.0:
                continue
            if best_near is None or abs(local[0] - 10.0) < abs(best_near[0] - 10.0):
                best_near = local
            if best_far is None or abs(local[0] - 20.0) < abs(best_far[0] - 20.0):
                best_far = local
        if best_near is None:
            best_near = [10.0, 0.0]
        if best_far is None:
            best_far = [best_near[0] + 10.0, best_near[1]]
        return [best_near, best_far]

    def _observation(self) -> Dict[str, Any]:
        cameras: Dict[str, Any] = {}
        for spec in self._sensor_specs:
            name = str(spec.get("name") or "rgb_front")
            h = int(spec.get("height", 512))
            w = int(spec.get("width", 1024))
            if self._dummy:
                cameras[name] = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                frame = self._latest.get(name)
                if frame is None:
                    cameras[name] = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    cameras[name] = frame
        return {
            "ego": {"speed_mps": _forward_speed(self.vehicle)},
            "sensor": {"cameras": cameras},
            "route": self._target_points_ego(),
        }

    def run_step(self, dt: float) -> carla.VehicleControl:
        del dt
        # Ensure at least one tick of sensor data when not dummy.
        if not self._dummy and self._step == 0:
            CarlaDataProvider.get_world().tick()
        action = self._bridge.act(self._observation())
        raw = action.get("control") or action
        control = carla.VehicleControl(
            throttle=float(np.clip(float(raw.get("throttle", 0.0)), 0.0, 1.0)),
            steer=float(np.clip(float(raw.get("steer", 0.0)), -1.0, 1.0)),
            brake=float(np.clip(float(raw.get("brake", 0.0)), 0.0, 1.0)),
        )
        self._step += 1
        if self._step <= 3 or self._step % 20 == 0:
            meta = action.get("meta") or {}
            lang = meta.get("language")
            print(
                f"SimLingo step={self._step} v={_forward_speed(self.vehicle):.2f} "
                f"throttle={control.throttle:.2f} brake={control.brake:.2f} "
                f"steer={control.steer:.2f}"
                + (f" lang={lang!r}" if lang else ""),
                flush=True,
            )
        return control

    def close(self) -> None:
        for sensor in self._sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:
                pass
        self._sensors.clear()
        self._bridge.close()
