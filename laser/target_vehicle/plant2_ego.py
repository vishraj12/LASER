"""PlanT 2.0 ego for LASER (privileged state -> waypoints/control).

Builds a ``scenario_orchestration``-compatible state observation each tick
(objects, ego-frame route, BEV semantic raster) and calls
``third_party/plant2/scenario_orchestration/policy.py``.

Environment
-----------
  LASER_EGO=plant2
  PLANT2_ROOT           optional; default = <laser_repo>/../plant2 sibling under
                        scenario_orchestration/third_party/plant2, or
                        PLANT2_CHECKPOINT's parent tree
  PLANT2_CHECKPOINT / PLANT_CHECKPOINT
                        path to a ``.ckpt`` (required for real inference)
  PLANT2_BLANK_BEV=1    plumbing only: blank BEV if ObsManager cannot run
  PLANT2_SEED           seed for checkpoint selection (default 0)

BEV maps: ``carla_garage/birds_eye_view/maps_2ppm_cv/<Town>.h5`` (Town10HD
included). Town names ending in ``_Opt`` fall back to the base town file.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import carla
import cv2
import numpy as np

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def _resolve_plant2_root() -> Path:
    env = os.environ.get("PLANT2_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    # laser_se lives in .../third_party/laser → sibling plant2
    here = Path(__file__).resolve()
    # .../laser/laser/target_vehicle/plant2_ego.py
    laser_repo = here.parents[2]
    sibling = laser_repo.parent / "plant2"
    if sibling.is_dir() and (sibling / "scenario_orchestration" / "policy.py").is_file():
        return sibling

    # scratch/LASER clone with plant2 checked out elsewhere
    scratch = Path.home() / "scratch" / "scenario_orchestration" / "third_party" / "plant2"
    if scratch.is_dir() and (scratch / "scenario_orchestration" / "policy.py").is_file():
        return scratch

    raise RuntimeError(
        "Cannot find plant2 repository. Set PLANT2_ROOT to the plant2 checkout "
        f"(expected policy at <root>/scenario_orchestration/policy.py). Tried "
        f"{sibling} and {scratch}."
    )


def _world_to_ego_2d(
    point_xy: Sequence[float], ego_xy: Sequence[float], yaw_rad: float
) -> List[float]:
    """Same transform as carla_garage.transfuser_utils.inverse_conversion_2d."""
    dx = float(point_xy[0]) - float(ego_xy[0])
    dy = float(point_xy[1]) - float(ego_xy[1])
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    # rotation_matrix.T @ (point - translation)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return [local_x, local_y]


def _forward_speed(actor: carla.Actor) -> float:
    vel = actor.get_velocity()
    forward = actor.get_transform().get_forward_vector()
    return float(vel.x * forward.x + vel.y * forward.y + vel.z * forward.z)


class _StopCriteriaStub:
    """Minimal stand-in for RunStopSign so ObsManager can attach."""

    target_stop_sign = None
    stop_completed = True


class _BEVConfig:
    lidar_resolution_width = 256
    lidar_resolution_height = 256
    pixels_per_meter_collection = 2.0


class PlanT2EgoController:
    """Closed-loop PlanT ego driven from CARLA privileged state."""

    def __init__(
        self,
        vehicle: carla.Vehicle,
        route: Sequence[Tuple[carla.Transform, object]],
        seed: int = 0,
    ) -> None:
        self.vehicle = vehicle
        self.route = list(route)
        self.seed = int(os.environ.get("PLANT2_SEED", seed))
        self._plant2_root = _resolve_plant2_root()
        self._policy = None
        self._bev_manager = None
        self._stop_criteria = _StopCriteriaStub()
        self._blank_bev_warned = False
        self._last_bev: Optional[Dict[str, Any]] = None
        self._bev_fail_count = 0
        self._step = 0
        self._bev_debug = os.environ.get("PLANT2_BEV_DEBUG") in ("1", "true", "True")
        self._bev_debug_every = max(
            1, int(os.environ.get("PLANT2_BEV_DEBUG_EVERY", "20"))
        )

        # ObsManager uses many RPCs; default 20s client timeout can fire mid-episode
        # when the LLM blocks the sync client for a long wall-clock stretch.
        try:
            client = CarlaDataProvider.get_client()
            if client is not None:
                client.set_timeout(60.0)
        except Exception:  # noqa: BLE001
            pass

        self._ensure_plant2_on_path()
        self._load_policy()
        self._maybe_attach_bev()

        print(
            f"PlanT2 ego: root={self._plant2_root} seed={self.seed} "
            f"bev_manager={'yes' if self._bev_manager else 'no'} "
            f"route_wps={len(self.route)}"
        )

    # -- setup -------------------------------------------------------------

    def _ensure_plant2_on_path(self) -> None:
        root = self._plant2_root
        # PlanT + BEV only. Do not prepend plant2's scenario_runner_autopilot —
        # that shadows LASER's CarlaDataProvider (missing set_ego_route).
        for sub in (
            root / "scenario_orchestration",
            root / "PlanT",
            root / "carla_garage",
        ):
            s = str(sub)
            if sub.is_dir() and s not in sys.path:
                sys.path.insert(0, s)
        # Prefer LASER's policy overlay (torch/device/index fixes) when present so
        # a stock third_party/plant2 pin stays runnable without upstream write access.
        # Insert last so it wins over plant2's own scenario_orchestration/policy.py.
        laser_repo = Path(__file__).resolve().parents[2]
        overlay_policy = (
            laser_repo
            / "scenario_orchestration"
            / "overlays"
            / "plant2"
            / "scenario_orchestration"
        )
        if (overlay_policy / "policy.py").is_file():
            s = str(overlay_policy)
            if s in sys.path:
                sys.path.remove(s)
            sys.path.insert(0, s)
            print(f"PlanT2 policy overlay: {overlay_policy}")

    def _load_policy(self) -> None:
        import policy as plant2_policy  # noqa: WPS433 — plant2 entry point

        ckpt_dir = self._plant2_root / "checkpoints"
        # Prefer an explicit file; a bare directory makes policy.resolve_checkpoint
        # return the dir itself and then fail the ".ckpt" suffix check.
        env_ckpt = os.environ.get("PLANT2_CHECKPOINT") or os.environ.get(
            "PLANT_CHECKPOINT"
        )
        if env_ckpt:
            declared = Path(env_ckpt).expanduser().resolve()
            if declared.is_dir():
                ckpts = sorted(declared.glob("*.ckpt"))
                if not ckpts:
                    raise FileNotFoundError(
                        f"No PlanT2 .ckpt under checkpoint directory {declared}"
                    )
                checkpoint = str(ckpts[self.seed % len(ckpts)])
                print(
                    f"PlanT2 checkpoint directory={declared}; selected "
                    f"{Path(checkpoint).name} for seed={self.seed}"
                )
            else:
                checkpoint = str(declared)
        else:
            ckpts = sorted(ckpt_dir.glob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(
                    f"No PlanT2 .ckpt under {ckpt_dir}. Download from "
                    "https://huggingface.co/SimonGer/PlanT2"
                )
            checkpoint = str(ckpts[self.seed % len(ckpts)])

        request = {
            "schema_version": "1.0.0",
            "name": "plant2",
            "interface": "ego_policy_v1",
            "implementation": "plant2.agent.PlanTAgent",
            "observation_space": "state",
            "action_space": "waypoints",
            "repository": str(self._plant2_root),
            "entry_point": "scenario_orchestration/policy.py",
            "checkpoint": checkpoint,
            "parameters": {"num_waypoints": 4, "checkpoint": checkpoint},
            "seed": self.seed,
        }
        print(f"PlanT2 loading checkpoint={checkpoint}")
        self._policy = plant2_policy.build_policy(request).load()
        self._policy.reset()

    def _maybe_attach_bev(self) -> None:
        try:
            from birds_eye_view.chauffeurnet import ObsManager
            from birds_eye_view.run_stop_sign import RunStopSign
        except ImportError as exc:
            print(
                f"PlanT2: BEV ObsManager unavailable ({exc}); "
                "set PLANT2_BLANK_BEV=1 for plumbing or fix PYTHONPATH"
            )
            return

        world = CarlaDataProvider.get_world()
        try:
            self._stop_criteria = RunStopSign(world)
        except Exception as exc:  # noqa: BLE001
            print(f"PlanT2: RunStopSign failed ({exc}); using stub")
            self._stop_criteria = _StopCriteriaStub()

        # Ensure maps_2ppm_cv/<Town>.h5 exists (CARLA often reports Town10HD_Opt).
        map_dir = (
            self._plant2_root / "carla_garage" / "birds_eye_view" / "maps_2ppm_cv"
        )
        town = world.get_map().name.split("/")[-1]
        h5 = map_dir / f"{town}.h5"
        if not h5.is_file() and town.endswith("_Opt"):
            base = town[: -len("_Opt")]
            alt = map_dir / f"{base}.h5"
            if alt.is_file():
                try:
                    h5.symlink_to(alt.name)
                    print(f"PlanT2: linked {h5.name} -> {alt.name}")
                except OSError as exc:
                    print(f"PlanT2: could not link {h5.name}: {exc}")

        obs_config = {
            "width_in_pixels": _BEVConfig.lidar_resolution_width,
            "pixels_ev_to_bottom": _BEVConfig.lidar_resolution_height / 2.0,
            "pixels_per_meter": _BEVConfig.pixels_per_meter_collection,
            "history_idx": [-1],
            "scale_bbox": True,
            "scale_mask_col": 1.0,
            "map_folder": "maps_2ppm_cv",
        }
        try:
            manager = ObsManager(obs_config, _BEVConfig())
            manager.attach_ego_vehicle(self.vehicle, criteria_stop=self._stop_criteria)
            self._bev_manager = manager
        except Exception as exc:  # noqa: BLE001
            print(f"PlanT2: failed to attach BEV ObsManager ({exc})")
            self._bev_manager = None

    # -- observation -------------------------------------------------------

    def _ego_yaw_rad(self) -> float:
        return math.radians(self.vehicle.get_transform().rotation.yaw)

    def _ego_xy(self) -> List[float]:
        loc = self.vehicle.get_location()
        return [loc.x, loc.y]

    def _build_route_ego_frame(self) -> List[List[float]]:
        """Sample the shared route by arc length in PlanT's ego frame.

        The route is ordered at roughly 1 m resolution. Sampling by route arc
        length preserves curves and turns; sampling only by increasing ego-x
        would discard the lateral part of a turn and then pad it as straight.
        """
        ego_xy = self._ego_xy()
        yaw = self._ego_yaw_rad()
        world_pts = [
            [float(transform.location.x), float(transform.location.y)]
            for transform, _opt in self.route
        ]
        if not world_pts:
            return [[2.5 + i, 0.0] for i in range(20)]

        # Continue from the closest point on the ordered global route.
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

        # Near the mission end, continue in the direction of the final segment.
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

    def _build_objects(self) -> List[Dict[str, Any]]:
        ego_xy = self._ego_xy()
        yaw = self._ego_yaw_rad()
        ego_tf = self.vehicle.get_transform()
        objects: List[Dict[str, Any]] = []
        world = CarlaDataProvider.get_world()

        for other in world.get_actors().filter("vehicle.*"):
            if other.id == self.vehicle.id:
                continue
            loc = other.get_location()
            pos = _world_to_ego_2d([loc.x, loc.y], ego_xy, yaw)
            if pos[0] ** 2 + pos[1] ** 2 > 80.0**2:
                continue
            extent = other.bounding_box.extent
            other_yaw = math.radians(other.get_transform().rotation.yaw)
            objects.append(
                {
                    "type": "car",
                    "position": [pos[0], pos[1], loc.z - ego_tf.location.z],
                    "yaw_rad": other_yaw - yaw,
                    "speed_mps": max(0.0, _forward_speed(other)),
                    "extent": [extent.x, extent.y, extent.z],
                    "id": other.id,
                    "type_id": other.type_id,
                }
            )

        for walker in world.get_actors().filter("walker.pedestrian.*"):
            loc = walker.get_location()
            pos = _world_to_ego_2d([loc.x, loc.y], ego_xy, yaw)
            if pos[0] ** 2 + pos[1] ** 2 > 50.0**2:
                continue
            extent = walker.bounding_box.extent
            w_yaw = math.radians(walker.get_transform().rotation.yaw)
            objects.append(
                {
                    "type": "walker",
                    "position": [pos[0], pos[1], loc.z - ego_tf.location.z],
                    "yaw_rad": w_yaw - yaw,
                    "speed_mps": max(0.0, _forward_speed(walker)),
                    "extent": [extent.x, extent.y, extent.z],
                    "id": walker.id,
                }
            )

        for tl in world.get_actors().filter("traffic.traffic_light*"):
            loc = tl.get_location()
            pos = _world_to_ego_2d([loc.x, loc.y], ego_xy, yaw)
            if pos[0] ** 2 + pos[1] ** 2 > 60.0**2:
                continue
            try:
                state = str(tl.get_state()).split(".")[-1]
            except Exception:  # noqa: BLE001
                state = "Unknown"
            objects.append(
                {
                    "type": "traffic_light",
                    "position": [pos[0], pos[1], 0.0],
                    "yaw_rad": 0.0,
                    "state": state,
                }
            )

        return objects

    def _build_bev(self) -> Optional[Dict[str, Any]]:
        if self._bev_manager is None:
            return self._last_bev
        try:
            if hasattr(self._stop_criteria, "tick"):
                self._stop_criteria.tick(self.vehicle)
            raw = self._bev_manager.get_observation(None)["bev_semantic_classes"]
            # Policy indexes a color palette — must be integer class ids.
            bev = {"semantic_classes": np.asarray(raw, dtype=np.int64)}
            self._last_bev = bev
            self._bev_fail_count = 0
            return bev
        except Exception as exc:  # noqa: BLE001
            self._bev_fail_count += 1
            # Keep a real raster: reuse last good frame (not blank zeros).
            if self._last_bev is not None:
                if self._bev_fail_count <= 3 or self._bev_fail_count % 20 == 0:
                    print(
                        f"PlanT2: BEV get_observation failed ({exc}); "
                        f"reusing last raster (fail#{self._bev_fail_count})"
                    )
                return self._last_bev
            if not self._blank_bev_warned:
                print(f"PlanT2: BEV get_observation failed ({exc}); no cached raster yet")
                self._blank_bev_warned = True
            return None

    def build_observation(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {
            "ego": {"speed_mps": max(0.0, _forward_speed(self.vehicle))},
            "objects": self._build_objects(),
            "route": self._build_route_ego_frame(),
            "speed_limit_kph": float(os.environ.get("PLANT2_SPEED_LIMIT_KPH", "50")),
        }
        bev = self._build_bev()
        if bev is not None:
            obs["bev"] = bev
        elif os.environ.get("PLANT2_BLANK_BEV") in ("1", "true", "True"):
            # Plumbing only — prefer real / cached raster above.
            obs["bev"] = {"semantic_classes": np.zeros((256, 256), dtype=np.int64)}
        return obs

    def _write_bev_debug(
        self, observation: Dict[str, Any], action: Dict[str, Any]
    ) -> None:
        """Render the exact model-facing BEV with ego-frame inputs overlaid."""
        if not self._bev_debug:
            return
        if self._step != 1 and self._step % self._bev_debug_every != 0:
            return

        bev = observation.get("bev")
        if not isinstance(bev, dict) or "semantic_classes" not in bev:
            return

        from plant_variables import PlanTVariables

        classes = np.asarray(bev["semantic_classes"], dtype=np.int64)
        if classes.ndim != 2 or min(classes.shape) <= 128:
            return

        # This is exactly the orientation and crop used by policy._encode_bev.
        model_classes = np.rot90(classes)[64:-64, 64:-64]
        palette = np.clip(
            np.asarray(PlanTVariables.bev_colors, dtype=np.float32) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        model_classes = np.clip(model_classes, 0, len(palette) - 1)
        image = palette[model_classes]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        ppm = float(_BEVConfig.pixels_per_meter_collection)
        center_col = image.shape[1] // 2
        # Native warp places ego at row 127 before the [64:-64] crop.
        center_row = classes.shape[0] // 2 - 1 - 64

        def to_pixel(point: Sequence[float]) -> Tuple[int, int]:
            # PlanT ego frame: x forward, y right. Image forward points upward.
            return (
                int(round(center_col + ppm * float(point[1]))),
                int(round(center_row - ppm * float(point[0]))),
            )

        route = observation.get("route") or []
        route_pixels = np.asarray([to_pixel(point) for point in route], np.int32)
        if len(route_pixels) >= 2:
            cv2.polylines(image, [route_pixels], False, (0, 255, 255), 1, cv2.LINE_AA)
        for pixel in route_pixels:
            cv2.circle(image, tuple(pixel), 1, (0, 255, 255), -1)

        for obj in observation.get("objects") or []:
            position = obj.get("position")
            if position is None:
                continue
            pixel = to_pixel(position)
            cv2.circle(image, pixel, 3, (0, 0, 255), -1)
            yaw = float(obj.get("yaw_rad", 0.0))
            tip = to_pixel(
                [
                    float(position[0]) + 3.0 * math.cos(yaw),
                    float(position[1]) + 3.0 * math.sin(yaw),
                ]
            )
            cv2.arrowedLine(image, pixel, tip, (0, 0, 255), 1, cv2.LINE_AA)

        predicted = action.get("path") or action.get("waypoints") or []
        predicted_pixels = np.asarray(
            [to_pixel(point) for point in predicted], np.int32
        )
        if len(predicted_pixels) >= 2:
            cv2.polylines(
                image, [predicted_pixels], False, (255, 255, 0), 1, cv2.LINE_AA
            )

        cv2.circle(image, (center_col, center_row), 3, (255, 255, 255), -1)
        cv2.arrowedLine(
            image,
            (center_col, center_row),
            (center_col, center_row - 12),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        scale = 4
        enlarged = cv2.resize(
            image,
            (image.shape[1] * scale, image.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        header = np.zeros((72, enlarged.shape[1], 3), dtype=np.uint8)
        labels = [
            "WHITE ego  YELLOW shared route  RED objects  CYAN predicted path",
            f"step={self._step} ego-frame model input: 128x128, 2 px/m",
        ]
        for index, label in enumerate(labels):
            cv2.putText(
                header,
                label,
                (8, 25 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        output = np.vstack((header, enlarged))
        debug_dir = Path(os.environ.get("SAVE_PATH", ".")) / "plant2_bev_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        output_path = debug_dir / f"bev_overlay_{self._step:04d}.png"
        cv2.imwrite(str(output_path), output)
        print(f"PlanT2 BEV debug: {output_path}")

    # -- control -----------------------------------------------------------

    def run_step(self, dt: float) -> carla.VehicleControl:
        del dt  # PlanT is not explicitly dt-integrated here
        self._step += 1
        if hasattr(self._stop_criteria, "tick"):
            try:
                self._stop_criteria.tick(self.vehicle)
            except Exception:  # noqa: BLE001
                pass

        observation = self.build_observation()
        action = self._policy.act(observation)
        self._write_bev_debug(observation, action)

        control = carla.VehicleControl()
        raw = action.get("control") if isinstance(action, dict) else None
        if not isinstance(raw, dict) or not raw:
            raise ValueError(
                "PlanT2 returned no control. LASER will not substitute fixed "
                "throttle for a missing policy action."
            )
        try:
            steer = float(raw["steer"])
            throttle = float(raw["throttle"])
            brake = float(raw["brake"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"PlanT2 returned invalid control: {raw!r}") from exc
        if not all(math.isfinite(value) for value in (steer, throttle, brake)):
            raise ValueError(f"PlanT2 returned non-finite control: {raw!r}")
        if not (-1.0 <= steer <= 1.0 and 0.0 <= throttle <= 1.0 and 0.0 <= brake <= 1.0):
            raise ValueError(f"PlanT2 returned out-of-range control: {raw!r}")
        control.steer = steer
        control.throttle = throttle
        control.brake = brake

        if self._step % 10 == 1:
            n_obj = len(observation.get("objects") or [])
            has_bev = "bev" in observation
            print(
                f"PlanT2 step={self._step} objs={n_obj} bev={has_bev} "
                f"v={observation['ego']['speed_mps']:.2f} "
                f"throttle={control.throttle:.2f} brake={control.brake:.2f} "
                f"steer={control.steer:.2f}"
            )
        return control

    def close(self) -> None:
        if self._policy is not None:
            try:
                self._policy.close()
            except Exception:  # noqa: BLE001
                pass
