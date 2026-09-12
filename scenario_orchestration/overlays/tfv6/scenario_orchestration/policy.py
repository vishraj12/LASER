"""This checkpoint behind the `ego_policy_v1` interface, command and all.

A harness that owns the CARLA world can drive this policy without the CARLA
Leaderboard: `build_policy(request)` returns an object whose `sensors()`
declares the rig to attach and whose `act(observation)` returns a control.

Preprocess is aligned with ``lead.inference.sensor_agent.SensorAgent.tick``:
JPEG round-trip, FOV crop, lidar history accumulate (+ radar-as-lidar), and
``radar_points_to_ego`` before ``preprocess_radar_input``.

Navigation: prefer harness ``observation["command"]`` / ``next_command``
(RoadOption names). Else fall back to heading change on ``observation["route"]``.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

SENSORS: List[Dict[str, Any]] = [
    {"name": "PCAM_L0", "width": 384, "height": 384, "fov": 60.0, "yaw": -57.5,
     "x": 0.0, "y": -0.3, "z": 2.25},
    {"name": "PCAM_F0", "width": 384, "height": 384, "fov": 60.0, "yaw": 0.0,
     "x": 0.25, "y": 0.0, "z": 2.25},
    {"name": "PCAM_R0", "width": 384, "height": 384, "fov": 60.0, "yaw": 57.5,
     "x": 0.0, "y": 0.3, "z": 2.25},
    {"name": "lidar", "kind": "sensor.lidar.ray_cast", "channels": 64,
     "range_m": 100.0, "rotation_frequency": 20.0,
     "x": 1.0, "y": 0.0, "z": 2.5},
]
RADARS: List[Dict[str, Any]] = [
    {"name": "radar1", "kind": "sensor.other.radar", "x": 2.6, "z": 0.60,
     "yaw": -45.0, "horizontal_fov": 90.0, "vertical_fov": 0.1,
     "range_m": 100.0, "points_per_second": 1500},
    {"name": "radar2", "kind": "sensor.other.radar", "x": 2.6, "z": 0.60,
     "yaw": 45.0, "horizontal_fov": 90.0, "vertical_fov": 0.1,
     "range_m": 100.0, "points_per_second": 1500},
    {"name": "radar3", "kind": "sensor.other.radar", "x": -2.6, "z": 0.60,
     "yaw": 135.0, "horizontal_fov": 90.0, "vertical_fov": 0.1,
     "range_m": 100.0, "points_per_second": 1500},
    {"name": "radar4", "kind": "sensor.other.radar", "x": -2.6, "z": 0.60,
     "yaw": 225.0, "horizontal_fov": 90.0, "vertical_fov": 0.1,
     "range_m": 100.0, "points_per_second": 1500},
]
CAMERA_ORDER = ("PCAM_L0", "PCAM_F0", "PCAM_R0")

DEFAULT_CHECKPOINT = os.environ.get(
    "TFV6_CHECKPOINT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "outputs", "checkpoints", "tfv6_resnet34"))

LEFT, RIGHT, STRAIGHT, LANEFOLLOW = 0, 1, 2, 3
COMMAND_DIM = 6
COMMAND_NAMES = ("LEFT", "RIGHT", "STRAIGHT", "LANEFOLLOW")
TURN_DEGREES = 25.0


def _one_hot(index: int) -> np.ndarray:
    vector = np.zeros(COMMAND_DIM, dtype=np.float32)
    vector[int(index) % COMMAND_DIM] = 1.0
    return vector


def _heading_change(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0

    def bearing(a, b):
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

    start = bearing(points[0], points[min(2, len(points) - 1)])
    end = bearing(points[-3], points[-1])
    return (end - start + 180.0) % 360.0 - 180.0


def _classify(change: float) -> int:
    if abs(change) < TURN_DEGREES:
        return LANEFOLLOW
    return RIGHT if change > 0 else LEFT


def _parse_command(value: Any, default: int = LANEFOLLOW) -> int:
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    name = str(value).upper()
    if "CHANGELANE" in name:
        return LANEFOLLOW
    for idx, label in enumerate(COMMAND_NAMES):
        if label in name:
            return idx
    return default


class TransfuserV6CvprPolicy:
    """TFv6 from the CVPR branch, behind `ego_policy_v1`."""

    def __init__(self, request: Optional[Dict[str, Any]] = None):
        self.request = dict(request or {})
        params = dict(self.request.get("parameters") or {})
        self.checkpoint = str(self.request.get("checkpoint")
                              or params.get("checkpoint")
                              or DEFAULT_CHECKPOINT)
        self.town = str(params.get("town", "Town10HD"))
        self.device_name = str(params.get("device", "cuda:0"))
        self.inference = None
        self.training_config = None
        self.config_expert = None
        self.config_closed_loop = None
        self._torch = None
        self._steps = 0
        self._commands: List[str] = []
        self._lidar_queue: Deque[np.ndarray] = deque()
        self._radar_queue: Deque[np.ndarray] = deque()
        self._pose_queue: Deque[Tuple[float, float, float]] = deque()
        self._stuck_detector = 0
        self._creep_counter = 0

    def sensors(self) -> List[Dict[str, Any]]:
        return [dict(s) for s in SENSORS] + [dict(s) for s in RADARS]

    def load(self) -> None:
        """Build the model, so a bad checkpoint fails before the simulation."""
        if self.inference is not None:
            return
        import torch
        from lead.expert.config_expert import ExpertConfig
        from lead.inference.closed_loop_inference import ClosedLoopInference
        from lead.inference.config_closed_loop import ClosedLoopConfig
        from lead.training.config_training import TrainingConfig

        self._torch = torch
        config_path = os.path.join(self.checkpoint, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"no config.json in checkpoint dir {self.checkpoint!r}; the "
                "cvpr2026 branch stores its training config as JSON, not the "
                "config.yaml the main stack writes")
        with open(config_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        self.training_config = TrainingConfig(stored)
        if not getattr(self.training_config, "use_discrete_command", False):
            raise RuntimeError(
                f"checkpoint {self.checkpoint!r} was trained without a discrete "
                "command; this adapter exists to supply one, so it is the wrong "
                "checkpoint for it")
        self.config_expert = ExpertConfig()
        self.config_closed_loop = ClosedLoopConfig()
        stack = int(getattr(self.config_expert, "lidar_stack_size", 5) or 5)
        self._lidar_queue = deque(maxlen=stack)
        self._radar_queue = deque(maxlen=2 * stack)
        self._pose_queue = deque(maxlen=stack)
        self.inference = ClosedLoopInference(
            config_training=self.training_config,
            config_closed_loop=self.config_closed_loop,
            config_expert=self.config_expert,
            model_path=self.checkpoint,
            device=torch.device(self.device_name),
            prefix="model",
        )

    def reset(self) -> None:
        self._steps = 0
        self._commands = []
        self._lidar_queue.clear()
        self._radar_queue.clear()
        self._pose_queue.clear()
        self._stuck_detector = 0
        self._creep_counter = 0

    def close(self) -> None:
        self.inference = None

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.inference is None:
            self.load()
        batch, commands = self._batch(observation)
        prediction = self.inference.forward(data=batch)
        throttle = float(prediction.throttle)
        steer = float(prediction.steer)
        brake = float(prediction.brake)
        speed = float(((observation.get("ego") or {}).get("speed_mps")) or 0.0)

        # SensorAgent creeping heuristic — without it a stop at red never recovers
        # through ClosedLoopInference alone.
        cfg = self.config_closed_loop
        if cfg is not None and getattr(cfg, "sensor_agent_creeping", False):
            if speed < 0.25 and brake >= 0.3:
                self._stuck_detector += 1
            else:
                self._stuck_detector = 0
            threshold = int(getattr(cfg, "sensor_agent_stuck_threshold", 40))
            duration = int(getattr(cfg, "sensor_agent_stuck_move_duration", 40))
            creep_thr = float(getattr(cfg, "sensor_agent_stuck_throttle", 0.55))
            if self._creep_counter > 0:
                throttle, brake = creep_thr, 0.0
                self._creep_counter -= 1
            elif self._stuck_detector > threshold:
                self._creep_counter = duration
                self._stuck_detector = 0
                throttle, brake = creep_thr, 0.0
                print(f"TFv6 creeping: throttle={creep_thr} for {duration} frames")

        self._steps += 1
        if not self._commands or self._commands[-1] != commands[0]:
            self._commands.append(commands[0])
        return {
            "control": {"throttle": throttle, "steer": steer, "brake": brake},
            "meta": {"policy": "tfv6", "step": self._steps,
                     "command": commands[0], "next_command": commands[1],
                     "stuck": self._stuck_detector},
        }

    def _batch(self, observation: Dict[str, Any]):
        torch = self._torch
        device = self.inference.device
        sensors = dict((observation.get("sensor") or {}).get("cameras") or {})
        ego = dict(observation.get("ego") or {})

        previous, target, following, commands = self._navigation(observation)
        speed = float(ego.get("speed_mps") or 0.0)
        pose = (
            float(ego.get("x") or 0.0),
            float(ego.get("y") or 0.0),
            float(ego.get("yaw_rad") or 0.0),
        )

        radar_ego = self._radar_ego_clouds(sensors)
        if radar_ego is not None:
            self._radar_queue.append(radar_ego)

        def tensor(array, dtype=torch.float32):
            return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)

        batch = {
            "rgb": tensor(self._stitch(sensors))[None],
            "rasterized_lidar": tensor(self._lidar(sensors, pose)),
            "target_point_previous": tensor(previous).view(1, 2),
            "target_point": tensor(target).view(1, 2),
            "target_point_next": tensor(following).view(1, 2),
            "speed": tensor([speed]).view(1),
            "command": tensor(_one_hot(commands[2])).view(1, COMMAND_DIM),
            "next_command": tensor(_one_hot(commands[3])).view(1, COMMAND_DIM),
            "town": np.array([self.town]),
        }
        if getattr(self.training_config, "use_radars", False):
            batch["radar"] = tensor(self._radar_model_input(sensors))[None]
        return batch, commands

    def _navigation(self, observation: Dict[str, Any]):
        route = observation.get("route") or []
        points = [(float(p[0]), float(p[1])) for p in route
                  if isinstance(p, (list, tuple)) and len(p) >= 2]
        if not points:
            points = [(2.5 + i, 0.0) for i in range(20)]

        def at(index):
            return points[min(index, len(points) - 1)]

        near = _parse_command(observation.get("command"), default=-1)
        far = _parse_command(observation.get("next_command"), default=-1)
        if near < 0 or far < 0:
            half = points[: max(3, len(points) // 2)]
            if near < 0:
                near = _classify(_heading_change(half))
            if far < 0:
                far = _classify(_heading_change(points))
        return at(0), at(7), at(15), (
            COMMAND_NAMES[near], COMMAND_NAMES[far], near, far)

    def _stitch(self, sensors: Dict[str, Any]) -> np.ndarray:
        """CHW strip with JPEG + FOV crop like SensorAgent.tick."""
        import cv2
        from lead.common import common_utils

        config = self.training_config
        height = config.final_image_height
        width = config.final_image_width
        per = width // len(CAMERA_ORDER)
        views = []
        for name in CAMERA_ORDER:
            image = sensors.get(name)
            if image is None:
                views.append(np.zeros((height, per, 3), dtype=np.uint8))
                continue
            array = np.asarray(image)[..., :3]
            if array.shape[0] != height or array.shape[1] != per:
                array = cv2.resize(array, (per, height), interpolation=cv2.INTER_AREA)
            views.append(array.astype(np.uint8))
        strip = np.concatenate(views, axis=1)  # RGB HWC, same as post BGR2RGB

        quality = 90
        if self.config_closed_loop is not None:
            quality = int(getattr(self.config_closed_loop, "jpeg_quality", 90))
        _, encoded = cv2.imencode(
            ".jpg", strip, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        strip = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        rgb = np.transpose(strip, (2, 0, 1))

        reduction = float(getattr(config, "horizontal_fov_reduction", 0) or 0)
        if reduction > 0:
            rgb = common_utils.fov_crop(rgb, reduction, chw=True)
        return np.asarray(rgb, dtype=np.uint8)

    def _radar_ego_clouds(self, sensors: Dict[str, Any]) -> Optional[np.ndarray]:
        """All radars in ego frame via official radar_points_to_ego."""
        if not getattr(self.training_config, "use_radars", False):
            return None
        from lead.common import common_utils

        clouds = []
        for index, spec in enumerate(RADARS, start=1):
            points = sensors.get(spec["name"])
            if points is None:
                continue
            arr = np.asarray(points, dtype=np.float32)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape((-1, 4))
            cal = self.config_expert.radar_calibration[str(index)]
            clouds.append(common_utils.radar_points_to_ego(
                arr, sensor_pos=cal["pos"], sensor_rot=cal["rot"])[:, :3])
        if not clouds:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(clouds, axis=0).astype(np.float32)

    def _radar_model_input(self, sensors: Dict[str, Any]) -> np.ndarray:
        from lead.common import common_utils
        from lead.data_loader import carla_dataset_utils

        raw = {}
        for index, spec in enumerate(RADARS, start=1):
            points = sensors.get(spec["name"])
            if points is None:
                raw[f"radar{index}"] = np.zeros((0, 4), dtype=np.float32)
                continue
            arr = np.asarray(points, dtype=np.float32)
            if arr.size == 0:
                raw[f"radar{index}"] = np.zeros((0, 4), dtype=np.float32)
                continue
            if arr.ndim == 1:
                arr = arr.reshape((-1, 4))
            cal = self.config_expert.radar_calibration[str(index)]
            raw[f"radar{index}"] = common_utils.radar_points_to_ego(
                arr, sensor_pos=cal["pos"], sensor_rot=cal["rot"])
        blocks = carla_dataset_utils.preprocess_radar_input(
            self.training_config, raw)
        if not blocks:
            n = (self.training_config.num_radar_points_per_sensor
                 * self.training_config.num_radar_sensors)
            return np.zeros((n, 5), dtype=np.float32)
        return np.concatenate(blocks, axis=0).astype(np.float32)

    def _lidar(self, sensors: Dict[str, Any], pose: Tuple[float, float, float]) -> np.ndarray:
        """Accumulate + rasterize like SensorAgent.tick / BaseAgent.accumulate_lidar."""
        from lead.common import common_utils
        from lead.data_loader import training_cache
        from lead.data_loader.carla_dataset_utils import rasterize_lidar

        config = self.training_config
        points = sensors.get("lidar")
        if points is not None and np.asarray(points).size:
            cloud = np.asarray(points)[:, :3].astype(np.float32)
            self._lidar_queue.append(cloud)
            self._pose_queue.append(pose)
        elif not self._lidar_queue:
            return np.zeros((1, 1, config.lidar_height_pixel,
                             config.lidar_width_pixel), dtype=np.float32)

        # Current-frame ego coords of past poses (same idea as BaseAgent).
        cx, cy, cyaw = self._pose_queue[-1]
        past_xy = []
        past_yaw = []
        for x, y, yaw in self._pose_queue:
            dx, dy = x - cx, y - cy
            c, s = math.cos(-cyaw), math.sin(-cyaw)
            past_xy.append((c * dx - s * dy, s * dx + c * dy))
            dyaw = (yaw - cyaw + math.pi) % (2 * math.pi) - math.pi
            past_yaw.append(dyaw)

        lidar_queue = list(self._lidar_queue)[::-1]
        past_xy = past_xy[::-1]
        past_yaw = past_yaw[::-1]
        accumulate = bool(getattr(self.config_expert, "lidar_accumulation", True))

        lidar_accumulated = []
        for i, lidar_pc in enumerate(lidar_queue):
            if i > 0 and not accumulate:
                break
            dx, dy = past_xy[i]
            dyaw = past_yaw[i]
            lidar_pc = common_utils.align_lidar(
                lidar_pc, np.array([-dx, -dy, 0.0]), -dyaw)
            stamp = np.ones((lidar_pc.shape[0], 1), dtype=np.float32) * i
            lidar_accumulated.append(np.concatenate([lidar_pc, stamp], axis=1))
        lidar = (np.concatenate(lidar_accumulated, axis=0)
                 if lidar_accumulated else np.zeros((0, 4), dtype=np.float32))

        # Radar-as-lidar densify (ExpertConfig.save_radar_pc_as_lidar).
        if (getattr(self.config_expert, "use_radars", True)
                and getattr(self.config_expert, "save_radar_pc_as_lidar", True)
                and self._radar_queue):
            radar_queue = list(self._radar_queue)[::-1]
            for i, radar_pc in enumerate(radar_queue):
                if i > 0 and not accumulate:
                    break
                if i >= len(past_xy):
                    break
                dx, dy = past_xy[min(i, len(past_xy) - 1)]
                dyaw = past_yaw[min(i, len(past_yaw) - 1)]
                radar_pc = common_utils.align_lidar(
                    radar_pc, np.array([-dx, -dy, 0.0]), -dyaw)
                if getattr(self.config_expert, "duplicate_radar_near_ego", False):
                    radius = float(getattr(
                        self.config_expert, "duplicate_radar_radius", 10.0))
                    factor = int(getattr(
                        self.config_expert, "duplicate_radar_factor", 1))
                    near = radar_pc[np.linalg.norm(radar_pc[:, :2], axis=1) < radius]
                    if near.size and factor > 1:
                        radar_pc = np.concatenate(
                            [radar_pc] + [near] * (factor - 1), axis=0)
                stamp = np.ones((radar_pc.shape[0], 1), dtype=np.float32) * i
                lidar = np.concatenate(
                    [lidar, np.concatenate([radar_pc, stamp], axis=1)], axis=0)

        used = int(getattr(config, "training_used_lidar_steps", 1) or 1)
        lidar = lidar[lidar[:, -1] < used]
        cloud = lidar[:, :3].astype(np.float64)
        for axis, precision in enumerate((self.config_expert.point_precision_x,
                                          self.config_expert.point_precision_y,
                                          self.config_expert.point_precision_z)):
            cloud[:, axis] = np.round(cloud[:, axis] / precision) * precision
        if cloud.shape[0] == 0:
            return np.zeros((1, 1, config.lidar_height_pixel,
                             config.lidar_width_pixel), dtype=np.float32)
        grid = rasterize_lidar(config=config, lidar=cloud)[..., None]
        grid = training_cache.compress_float_image(grid, config)
        grid = training_cache.decompress_float_image(grid).squeeze()[None, None]
        return np.asarray(grid, dtype=np.float32)

    def metadata(self) -> Dict[str, Any]:
        return {"policy": "tfv6", "checkpoint": self.checkpoint,
                "commands_issued": self._commands,
                "note": "preprocess aligned with SensorAgent.tick; command from "
                        "observation['command'] or route heading fallback"}


def build_policy(request: Dict[str, Any]) -> TransfuserV6CvprPolicy:
    return TransfuserV6CvprPolicy(request)
