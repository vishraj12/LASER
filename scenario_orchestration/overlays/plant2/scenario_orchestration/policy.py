"""PlanT 2.0 exposed as a ``scenario_orchestration`` ego policy.

This file is the whole integration boundary between the ``scenario_orchestration``
experiment harness and this repository, as specified by
``scenario_orchestration/third_party/README.md`` and ``DESIGN.md`` section 6:

    <repository>/
    └── scenario_orchestration/
        └── policy.py

    ``policy.py`` must construct the policy from a ``PolicyRequest`` and expose
    the declared ``ego_policy_v1`` interface: given an observation in the
    declared observation space, return an action in the declared action space.

The harness declares this policy in ``configs/policy/plant2.yaml`` as

    interface:          ego_policy_v1
    observation_space:  state
    action_space:       waypoints

so the surface below is ``state -> waypoints``.

Nothing here imports the harness. A ``PolicyRequest`` is re-declared locally and
parsed from the serialized document, exactly as the process-level contract
intends, so this repository stays free to pin its own Python, CUDA, PyTorch and
CARLA versions.


Using it
--------

From a method repository's own runner::

    import json, sys
    sys.path.insert(0, "third_party/plant2/scenario_orchestration")
    import policy as plant2_policy

    ego = plant2_policy.build_policy(json.load(open("policy.json")))
    ego.reset()
    while not done:
        action = ego.act(observation)      # -> {"waypoints": [[x, y], ...], ...}
    ego.close()

From the command line (useful for validation and for driving the policy over a
pipe from a runner in a different environment)::

    python scenario_orchestration/policy.py --describe
    python scenario_orchestration/policy.py --policy-request policy.json \
        --observation obs.json --output action.json
    python scenario_orchestration/policy.py --self-test


The observation (observation space ``state``)
---------------------------------------------

A plain JSON-compatible mapping. All geometry is in the **ego frame that PlanT
itself uses**: metres, ``+x`` forward, ``+y`` right (CARLA's left-handed
convention, i.e. what ``carla_garage.transfuser_utils.get_relative_transform``
returns), yaw in radians relative to the ego heading.

    {
      "ego": {"speed_mps": 5.4},

      "objects": [
        {"type": "car",           "position": [12.0, 0.2], "yaw_rad": 0.01,
         "speed_mps": 8.0, "extent": [2.45, 1.06, 0.75], "id": 137},
        {"type": "walker",        "position": [8.0, 4.0],  "yaw_rad": -1.57,
         "speed_mps": 1.2, "extent": [0.18, 0.18, 0.93], "id": 240},
        {"type": "traffic_light", "position": [20.0, 0.0], "yaw_rad": 0.0,
         "state": "Red"},
        {"type": "stop_sign",     "position": [15.0, 0.0], "yaw_rad": 0.0}
      ],

      "route": [[2.5, 0.0], [3.5, 0.0], ...],
      "speed_limit_kph": 50,

      "bev": {"semantic_classes": [[...256x256 ints 0..4...]]}
    }

``ego.speed_mps``
    Forward speed of the ego vehicle, m/s. Also accepted as ``observation["speed"]``.

``objects``
    Object-centric scene description. ``type`` is one of ``car``, ``walker``,
    ``static``, ``static_car``, ``stop_sign``, ``traffic_light``, ``emergency``;
    the method-agnostic aliases ``vehicle``, ``pedestrian``, ``obstacle`` and
    ``parked_car`` are accepted too. ``extent`` is the CARLA **half**-extent
    ``[x, y, z]`` in metres (length, width, height), which is what
    ``carla.BoundingBox.extent`` reports. ``position`` may be 2D or 3D.
    ``traffic_light`` objects need ``state`` (``Red``/``Yellow``/``Green``; only
    red and yellow are fed to the model, matching ``PlanT/PlanT_agent.py``).
    Optional keys that reproduce the reference agent's own filtering:
    ``id`` (used to keep track of walkers that have started moving),
    ``type_id`` (CARLA blueprint id -- selects emergency vehicles and relevant
    static props), ``mesh_path`` and ``scale`` (static car extents), and
    ``scenario`` (the ``VehicleOpensDoor`` width bump).

``route``
    The upcoming route in the ego frame. PlanT's route embedding is a fixed
    ``Linear(20 * 2, ...)``, so exactly 20 points are fed to the model: a longer
    route is truncated and a shorter one is padded by repeating its last point.
    The reference agent samples one point per metre starting 2.5 m ahead
    (``config.tf_first_checkpoint_distance``, ``config.points_per_meter``);
    matching that sampling is what makes the numbers comparable to the paper.

``speed_limit_kph``
    Snapped to the nearest speed limit PlanT was trained with
    (50 / 80 / 100 / 120 km/h). Defaults to 50.

``bev``
    Required by every released PlanT 2.0 checkpoint (they all carry
    ``model.training.input_bev = True``). Either

      * ``{"semantic_classes": HxW}`` -- the raw class-index raster produced by
        ``carla_garage.birds_eye_view.chauffeurnet.ObsManager`` under
        ``bev_semantic_classes`` (256x256 at 2 px/m, ego 128 px from the
        bottom). It is rotated, centre-cropped and colourised here exactly as
        ``PlanT/PlanT_agent.py`` does, or
      * ``{"image": 3xHxW}`` -- an already-prepared float tensor, used as is
        (3x128x128 for the released checkpoints).

    A bare array is accepted for either form and dispatched on its rank. If a
    checkpoint needs a BEV and the observation carries none, ``act`` raises
    ``ObservationError``; set ``PLANT2_BLANK_BEV=1`` to substitute a blank
    raster instead, which is meant for plumbing tests, not for results.


The action (action space ``waypoints``)
---------------------------------------

    {
      "waypoints":        [[x, y], ...],   # the declared action, ego frame, m
      "path":             [[x, y], ...],   # predicted route-aligned path
      "target_speed_mps": 6.7,
      "control":          {"steer": .., "throttle": .., "brake": ..},
      "meta":             {"step": 12, ...}
    }

``waypoints`` is the contract. ``control`` is a convenience for methods that
prefer to hand the simulator a control directly: it is produced by PlanT's own
lateral and longitudinal controllers, which is how the published closed-loop
numbers are obtained, so a method that consumes it reproduces PlanT's behaviour
rather than re-deriving a controller. It is omitted if the controllers cannot be
constructed in this environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# -- contract identity ------------------------------------------------------

#: Must match ``configs/policy/plant2.yaml`` in the harness.
POLICY_NAME = "plant2"
INTERFACE = "ego_policy_v1"
OBSERVATION_SPACE = "state"
ACTION_SPACE = "waypoints"
SCHEMA_VERSION = "1.0.0"

#: This repository's root, i.e. the parent of ``scenario_orchestration/``.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where the released checkpoints live once they have been pulled from
#: https://huggingface.co/SimonGer/PlanT2
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

#: Speed limits PlanT was trained with, in km/h (``PlanT/plant_variables.py``).
SPEED_LIMITS_KPH = (50, 80, 100, 120)

#: PlanT's route embedding is a fixed ``Linear(20 * 2, n_embd)``.
ROUTE_POINTS = 20

#: The reference agent brakes for the first 40 ticks to let the sim settle
#: (``PlanT/PlanT_agent.py``). Methods that want that behaviour can read it back
#: from ``metadata()``; it is not imposed on the returned waypoints.
INITIAL_FRAMES_DELAY = 40


class PolicyError(Exception):
    """This policy cannot be constructed or stepped as requested."""


class ObservationError(PolicyError):
    """An observation did not satisfy the ``state`` observation space."""


class CheckpointError(PolicyError):
    """No usable PlanT 2.0 checkpoint could be resolved."""


# ---------------------------------------------------------------------------
# The serialized request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyRequest:
    """A local mirror of the harness's ``PolicyRequest`` (``policy.json``).

    Deliberately re-declared rather than imported: the contract is the JSON
    document, not a shared Python class, which is what lets this repository keep
    its own environment. Unknown keys are ignored so the harness can grow the
    document without breaking this policy.
    """

    name: str = POLICY_NAME
    interface: str = INTERFACE
    implementation: str = "plant2.agent.PlanTAgent"
    observation_space: str = OBSERVATION_SPACE
    action_space: str = ACTION_SPACE
    seed: int = 0
    schema_version: str = SCHEMA_VERSION
    experiment_id: str = ""
    repository: str | None = None
    entry_point: str | None = None
    checkpoint: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    requires: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "PolicyRequest":
        payload = dict(payload or {})
        known = set(cls.__dataclass_fields__)
        data = {k: v for k, v in payload.items() if k in known}
        data["seed"] = int(data.get("seed") or 0)
        data["parameters"] = dict(data.get("parameters") or {})
        data["requires"] = list(data.get("requires") or [])
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "PolicyRequest":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def check(self) -> None:
        """Reject a request this repository cannot honour.

        The harness validates the declaration too, but a method repository may
        construct us directly, so the check lives on both sides of the boundary.
        """
        if self.name and self.name != POLICY_NAME:
            raise PolicyError(
                f"policy.py of {POLICY_NAME!r} was handed a request for "
                f"{self.name!r}"
            )
        if self.interface != INTERFACE:
            raise PolicyError(
                f"{POLICY_NAME!r} implements {INTERFACE!r}, not "
                f"{self.interface!r}"
            )
        if self.observation_space != OBSERVATION_SPACE:
            raise PolicyError(
                f"{POLICY_NAME!r} consumes {OBSERVATION_SPACE!r} observations, "
                f"not {self.observation_space!r}"
            )
        if self.action_space != ACTION_SPACE:
            raise PolicyError(
                f"{POLICY_NAME!r} emits {ACTION_SPACE!r}, not "
                f"{self.action_space!r}"
            )


# ---------------------------------------------------------------------------
# Importing this repository's own modules
# ---------------------------------------------------------------------------


def _prepare_sys_path() -> None:
    """Make ``PlanT/`` and ``carla_garage/`` importable, in that order.

    This mirrors the ``PYTHONPATH`` the README asks for. Both directories are
    imported by bare module name (``lit_module`` does ``from model import
    HFLM``), so they have to be on the path rather than loaded from a file.

    The order is not cosmetic: both directories contain a ``model.py``, and
    ``carla_garage``'s pulls in the TransFuser video backbone and its
    dependencies. ``PlanT`` has to win, so the two entries are placed
    deliberately rather than merely appended.
    """
    wanted = [str(REPO_ROOT / relative) for relative in ("PlanT", "carla_garage")]
    for path in wanted:
        while path in sys.path:
            sys.path.remove(path)
    for index, path in enumerate(wanted):
        sys.path.insert(index, path)


def _import_global_config() -> Any:
    """``carla_garage.config.GlobalConfig``, the controllers' parameter block.

    ``carla_garage/config.py`` touches ``carla`` only to build a handful of
    debug-drawing ``carla.Color`` constants at class-definition time, so when
    the CARLA egg is absent -- which is the normal case for a policy consumed by
    a runner that owns the simulator elsewhere -- a stub is enough to read the
    controller gains out of it.
    """
    _prepare_sys_path()
    try:
        import config  # noqa: PLC0415 - repository-local module
    except ImportError as exc:
        if "carla" not in str(exc):
            raise
        stub = types.ModuleType("carla")
        stub.Color = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        sys.modules.setdefault("carla", stub)
        import config  # noqa: PLC0415
    return config.GlobalConfig()


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def _candidate_paths(value: str) -> list[Path]:
    """Every place a declared checkpoint path could reasonably resolve to.

    The harness declares the checkpoint from its own root
    (``third_party/plant2/checkpoints/plant2.ckpt``) while this repository sees
    that same file as ``checkpoints/plant2.ckpt``, so both readings are tried.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return [path]
    candidates = [Path.cwd() / path, REPO_ROOT / path]
    # Strip a leading ``third_party/<repo>/`` written from the harness's root.
    parts = path.parts
    if len(parts) > 2 and parts[0] == "third_party":
        candidates.append(REPO_ROOT / Path(*parts[2:]))
    candidates.append(CHECKPOINT_DIR / path.name)
    seen: set[Path] = set()
    unique = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def _available_checkpoints(directory: Path) -> list[Path]:
    """Checkpoints in ``directory``, in a stable order.

    PlanT 2.0 publishes one checkpoint per training seed, so the order has to be
    deterministic for a seed to select reproducibly.
    """
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.ckpt") if p.is_file())


def resolve_checkpoint(request: PolicyRequest) -> tuple[Path, str]:
    """Pick the checkpoint for ``request`` and say why.

    In order of precedence:

    1. ``PLANT2_CHECKPOINT``, then ``PLANT_CHECKPOINT`` -- the environment wins,
       so a sweep can pin a checkpoint without touching the harness configs.
       ``PLANT_CHECKPOINT`` is the variable ``PlanT/PlanT_agent.py`` already
       reads, which keeps one knob for both entry points.
    2. ``parameters.checkpoint`` in the policy request, when it names a *file*.
    3. ``checkpoint`` in the policy request, when it names a *file*.
    4. A seed-indexed pick inside a declared *directory* -- the normal case, since
       the harness declares ``third_party/plant2/checkpoints``.
    5. A seed-indexed pick among ``checkpoints/*.ckpt``.

    Where a declared path names a file that is not there, its directory is
    searched for the released checkpoints and the seed selects among them. That
    is deliberate: the harness config declares the generic
    ``checkpoints/plant2.ckpt`` while the published artefacts are named
    ``epoch=029_final_{1,2,3}.ckpt``, one per training seed. The substitution is
    always reported in the returned reason and in ``metadata()`` so it can never
    silently change what a paper table means.
    """
    seed = int(request.seed)

    for variable in ("PLANT2_CHECKPOINT", "PLANT_CHECKPOINT"):
        declared = os.environ.get(variable)
        if declared:
            path = Path(declared).expanduser()
            if not path.exists():
                raise CheckpointError(
                    f"{variable}={declared} does not exist"
                )
            return path, f"{variable}"

    declared = request.parameters.get("checkpoint") or request.checkpoint
    if declared:
        candidates = _candidate_paths(str(declared))
        for candidate in candidates:
            if candidate.is_file():
                return candidate, "policy request"
        # A declared *directory* is the normal case, not an error: PlanT 2.0
        # publishes one checkpoint per training seed, so the harness declares the
        # directory and the seed picks within it. Tested with is_file() above
        # rather than exists(), because a directory that "exists" would otherwise
        # be returned as the checkpoint and rejected later for not being a .ckpt.
        for candidate in candidates:
            if candidate.is_dir():
                available = _available_checkpoints(candidate)
                if available:
                    chosen = available[seed % len(available)]
                    return chosen, (
                        f"declared checkpoint directory {declared!r}; selected "
                        f"{chosen.name} by seed {seed} from "
                        f"{len(available)} checkpoint(s) in {candidate}"
                    )
        # Declared but absent: fall back within the directory it pointed at.
        for candidate in candidates:
            available = _available_checkpoints(candidate.parent)
            if available:
                chosen = available[seed % len(available)]
                return chosen, (
                    f"declared checkpoint {declared!r} is missing; selected "
                    f"{chosen.name} by seed {seed} from "
                    f"{len(available)} checkpoint(s) in {candidate.parent}"
                )

    available = _available_checkpoints(CHECKPOINT_DIR)
    if available:
        chosen = available[seed % len(available)]
        return chosen, (
            f"selected {chosen.name} by seed {seed} from {len(available)} "
            f"checkpoint(s) in {CHECKPOINT_DIR}"
        )

    raise CheckpointError(
        f"no PlanT 2.0 checkpoint found. Declared: {declared!r}. Searched "
        f"{CHECKPOINT_DIR}. Download one from "
        "https://huggingface.co/SimonGer/PlanT2, or point PLANT2_CHECKPOINT at it."
    )


# ---------------------------------------------------------------------------
# Translating a ``state`` observation into PlanT's object tokens
# ---------------------------------------------------------------------------

#: Method-agnostic object names accepted in an observation, mapped onto the class
#: names ``PlanT/plant_variables.py`` scores.
CLASS_ALIASES = {
    "car": "car",
    "vehicle": "car",
    "walker": "walker",
    "pedestrian": "walker",
    "static": "static",
    "obstacle": "static",
    "static_car": "static_car",
    "parked_car": "static_car",
    "parked_vehicle": "static_car",
    "stop_sign": "stop_sign",
    "traffic_light": "traffic_light",
    "emergency": "emergency",
    "emergency_vehicle": "emergency",
}

#: Blueprints the reference agent promotes to the ``emergency`` class.
EMERGENCY_TYPE_IDS = frozenset(
    {
        "vehicle.dodge.charger_police",
        "vehicle.dodge.charger_police_2020",
        "vehicle.carlamotors.firetruck",
        "vehicle.ford.ambulance",
    }
)

#: The only static props PlanT is trained to attend to; everything else static is
#: dropped (``PlanT/PlanT_agent.py``).
RELEVANT_STATIC_TYPE_IDS = frozenset(
    {"static.prop.constructioncone", "static.prop.trafficwarning"}
)

#: Traffic-light states that produce a token at all.
BLOCKING_LIGHT_STATES = frozenset({"Red", "Yellow"})


def _normalize_angle_degree(x: float) -> float:
    """``carla_garage.transfuser_utils.normalize_angle_degree``, inlined.

    Copied rather than imported because ``transfuser_utils`` imports ``carla``,
    and a policy consumed by a runner that owns the simulator elsewhere should
    not need the CARLA egg just to convert an angle.
    """
    x = float(x) % 360.0
    if x > 180.0:
        x -= 360.0
    return x


def _rad2deg(theta: float) -> float:
    import math  # noqa: PLC0415 - stdlib, kept local for symmetry

    return _normalize_angle_degree(math.degrees(float(theta)))


def _as_xyz(value: Any, what: str) -> tuple[float, float, float]:
    try:
        coords = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ObservationError(f"{what} is not a sequence of numbers: {value!r}") from exc
    if len(coords) == 2:
        return coords[0], coords[1], 0.0
    if len(coords) == 3:
        return coords[0], coords[1], coords[2]
    raise ObservationError(f"{what} must have 2 or 3 components, got {len(coords)}")


def _object_class(obj: Mapping[str, Any]) -> str:
    declared = obj.get("type") or obj.get("class")
    if declared is None:
        raise ObservationError(f"object has neither 'type' nor 'class': {obj!r}")
    key = str(declared).strip().lower()
    if key not in CLASS_ALIASES:
        raise ObservationError(
            f"unknown object type {declared!r}; expected one of "
            f"{sorted(set(CLASS_ALIASES))}"
        )
    return CLASS_ALIASES[key]


def snap_speed_limit_kph(value: Any) -> int:
    """Nearest speed limit PlanT has an embedding for.

    ``PlanT/PlanT_agent.py`` indexes ``speed_cats`` directly and raises on
    anything else; a policy that a foreign runner feeds should not fall over
    because a scenario declared 60 km/h.
    """
    if value is None:
        return SPEED_LIMITS_KPH[0]
    try:
        speed = float(value)
    except (TypeError, ValueError) as exc:
        raise ObservationError(f"speed_limit_kph is not numeric: {value!r}") from exc
    return min(SPEED_LIMITS_KPH, key=lambda limit: abs(limit - speed))


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


class PlanT2Policy:
    """PlanT 2.0 behind the ``ego_policy_v1`` interface.

    ``state`` observation in, ``waypoints`` action out. The model, its
    checkpoint and PlanT's controllers are loaded lazily on the first ``act``
    (or an explicit ``load()``), so constructing the policy stays cheap enough
    for a harness to do it while only validating a matrix.
    """

    def __init__(self, request: PolicyRequest | Mapping[str, Any] | None = None) -> None:
        if not isinstance(request, PolicyRequest):
            request = PolicyRequest.from_dict(request)
        request.check()
        # PlanT's own modules are imported by bare name throughout this file.
        _prepare_sys_path()

        self.request = request
        self.seed = int(request.seed)
        self.parameters = dict(request.parameters)

        #: How many waypoints to return. ``configs/policy/plant2.yaml`` declares
        #: ``num_waypoints``, while a checkpoint predicts its own trained number;
        #: the request truncates, it never pads.
        self.num_waypoints = self.parameters.get("num_waypoints")

        #: ``PlanT/PlanT_agent.py`` drops the ``static`` class on longest6, where
        #: static props never appear. Off by default; a method that evaluates on
        #: longest6 can switch it on through the request.
        self.drop_static_class = bool(self.parameters.get("drop_static_class", False))

        self._model = None
        self._config = None
        self._controllers: dict[str, Any] | None = None
        self._checkpoint: Path | None = None
        self._checkpoint_reason = ""
        self._model_flags: dict[str, Any] = {}
        self._moving_walkers: set[Any] = set()
        self._step = 0

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> "PlanT2Policy":
        """Resolve the checkpoint, seed the RNGs and build the network."""
        if self._model is not None:
            return self

        _prepare_sys_path()
        import numpy as np  # noqa: PLC0415
        import random  # noqa: PLC0415
        import torch  # noqa: PLC0415

        # The network is deterministic at inference, but a policy that declares a
        # seed should honour it: a method may perturb the observation, and the
        # controllers keep integral state.
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        checkpoint, reason = resolve_checkpoint(self.request)
        self._checkpoint, self._checkpoint_reason = checkpoint, reason
        if reason != "policy request":
            print(
                f"[{POLICY_NAME}] checkpoint {checkpoint} ({reason})",
                file=sys.stderr,
            )

        if checkpoint.suffix != ".ckpt":
            raise CheckpointError(
                f"PlanT 2.0 loads Lightning '.ckpt' checkpoints; got {checkpoint}"
            )

        from lit_module import LitHFLM  # noqa: PLC0415 - repository-local

        self.device = self.parameters.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        try:
            raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            # torch<2.0 has no weights_only=
            raw = torch.load(checkpoint, map_location="cpu")
        training = raw["hyper_parameters"]["cfg"]["model"]["training"]
        waypoints = raw["hyper_parameters"]["cfg"]["model"]["waypoints"]
        self._model_flags = {
            "input_bev": bool(training.get("input_bev", False)),
            "input_static_cars": bool(training.get("input_static_cars", False)),
            "input_ego_speed": bool(training.get("input_ego_speed", False)),
            "range": training.get("range", False),
            "range_factor_front": training.get("range_factor_front", 1),
            "waypoint_representation": waypoints.get("representation"),
            "wps_len": waypoints.get("wps_len"),
            "path_len": waypoints.get("path_len"),
        }
        del raw

        # strict=False: released ckpts are Lightning 2.5 / newer HF and omit
        # buffers like embeddings.position_ids that older transformers still
        # register. Missing buffers keep their default init (safe for inference).
        model = LitHFLM.load_from_checkpoint(
            checkpoint, map_location=self.device, strict=False
        )
        # HF / timm submodules (and default buffers for missing keys) can remain
        # on CPU after load_from_checkpoint — force one device before act().
        model = model.to(self.device)
        model.eval()
        self._model = model
        return self

    def reset(self, seed: int | None = None) -> None:
        """Start a new episode.

        Clears the per-episode state PlanT carries between ticks: which walkers
        have been seen moving, the step counter, and the controllers' integral
        windows.
        """
        if seed is not None:
            self.seed = int(seed)
        self._moving_walkers.clear()
        self._step = 0
        self._controllers = None

    def close(self) -> None:
        """Release the network and its device memory."""
        self._model = None
        self._controllers = None
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __enter__(self) -> "PlanT2Policy":
        return self.load()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- the ego_policy_v1 step -------------------------------------------

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """One step of ``ego_policy_v1``: a ``state`` observation to ``waypoints``."""
        import torch  # noqa: PLC0415

        self.load()
        if not isinstance(observation, Mapping):
            raise ObservationError(
                f"observation must be a mapping, got {type(observation).__name__}"
            )

        batch = self._build_batch(observation)
        with torch.no_grad():
            _, _, (pred_path, pred_wps, pred_speed), _ = self._model(batch)

        self._step += 1
        return self._build_action(
            pred_path=pred_path,
            pred_wps=pred_wps,
            pred_speed=pred_speed,
            ego_speed=self._ego_speed(observation),
        )

    #: ``step`` and ``__call__`` are aliases, so a runner can use whichever name
    #: its own policy plumbing expects without an adapter.
    step = act

    def __call__(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return self.act(observation)

    # -- observation -> model input ---------------------------------------

    def _ego_speed(self, observation: Mapping[str, Any]) -> float:
        ego = observation.get("ego")
        if isinstance(ego, Mapping):
            for key in ("speed_mps", "speed", "forward_speed"):
                if key in ego:
                    return float(ego[key])
        for key in ("speed_mps", "speed"):
            if key in observation:
                return float(observation[key])
        raise ObservationError(
            "observation carries no ego speed; expected ego.speed_mps (m/s)"
        )

    def _build_batch(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Assemble the batch ``PlanT/model.py`` consumes.

        Mirrors ``PlanTAgent.get_input_batch``, and reuses this repository's own
        ``dataset.generate_batch`` so the tokenisation cannot drift from training.
        """
        from dataset import generate_batch  # noqa: PLC0415 - repository-local

        sample: dict[str, Any] = {
            "input": self._encode_objects(observation.get("objects") or []),
            "output": [],
            "route": [],
            "waypoints": 0,
            "target_point": [],
            "route_original": self._encode_route(observation.get("route")),
            "speed_limit": self._encode_speed_limit(observation),
            "ego_speed": self._ego_speed(observation),
            "target_speed": 0,
        }
        if self._model_flags.get("input_ego_speed"):
            # The reference agent never populates this because the released
            # checkpoints are trained without it; the model reads it under this
            # name, so a checkpoint that wants it gets it.
            sample["input_ego_speed"] = self._ego_speed(observation)
        if self._model_flags.get("input_bev"):
            sample["BEV"] = self._encode_bev(observation.get("bev"))

        batch = generate_batch([sample])
        import torch  # noqa: PLC0415

        for key, value in list(batch.items()):
            if not torch.is_tensor(value):
                continue
            # Embedding / advanced indexing need int64; generate_batch uses int32.
            if value.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.uint8,
            ):
                value = value.long()
            batch[key] = value.to(self.device)
        # No object forecasting at inference: this is what switches the model's
        # auxiliary heads off (``PlanT/model.py``).
        batch["y_objs"] = None
        return batch

    def _encode_speed_limit(self, observation: Mapping[str, Any]) -> int:
        from plant_variables import PlanTVariables  # noqa: PLC0415

        declared = observation.get("speed_limit_kph")
        if declared is None and "speed_limit_mps" in observation:
            declared = float(observation["speed_limit_mps"]) * 3.6
        return PlanTVariables.speed_cats[snap_speed_limit_kph(declared)]

    def _encode_route(self, route: Any) -> Any:
        """The route as the fixed-width ``20 x 2`` block the model embeds."""
        import numpy as np  # noqa: PLC0415

        if route is None:
            raise ObservationError(
                "observation carries no 'route'; PlanT is route-conditioned and "
                f"embeds exactly {ROUTE_POINTS} ego-frame points"
            )
        points = [_as_xyz(point, "route point")[:2] for point in route]
        if not points:
            raise ObservationError("'route' is empty")
        if len(points) > ROUTE_POINTS:
            points = points[:ROUTE_POINTS]
        while len(points) < ROUTE_POINTS:
            # Repeating the last point is the benign padding: it reads as "the
            # route ends here", which is what a short route means.
            points.append(points[-1])
        return np.asarray(points, dtype=np.float32)

    def _encode_bev(self, bev: Any) -> Any:
        """The BEV token input, colourised exactly as ``PlanT_agent.tick`` does."""
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        from plant_variables import PlanTVariables  # noqa: PLC0415

        if bev is None:
            if os.environ.get("PLANT2_BLANK_BEV") in ("1", "true", "True"):
                print(
                    f"[{POLICY_NAME}] PLANT2_BLANK_BEV is set: feeding a blank BEV "
                    "raster. Driving behaviour from this run is not meaningful.",
                    file=sys.stderr,
                )
                bev = np.zeros((256, 256), dtype=np.int64)
            else:
                raise ObservationError(
                    "this checkpoint was trained with input_bev=True and the "
                    "observation carries no 'bev'. Provide either "
                    "{'semantic_classes': HxW} (the 'bev_semantic_classes' raster "
                    "from carla_garage.birds_eye_view.chauffeurnet.ObsManager, "
                    "256x256 at 2 px/m) or {'image': 3xHxW}. Set "
                    "PLANT2_BLANK_BEV=1 to substitute a blank raster for plumbing "
                    "tests only."
                )

        if isinstance(bev, Mapping):
            if "image" in bev:
                return torch.as_tensor(np.asarray(bev["image"]), dtype=torch.float32)
            if "semantic_classes" not in bev:
                raise ObservationError(
                    "'bev' mapping needs 'semantic_classes' or 'image'; got keys "
                    f"{sorted(bev)}"
                )
            classes = np.asarray(bev["semantic_classes"])
        else:
            classes = np.asarray(bev)
            if classes.ndim == 3:
                # Already a CHW image.
                return torch.as_tensor(classes, dtype=torch.float32)

        if classes.ndim != 2:
            raise ObservationError(
                f"'bev' semantic raster must be 2-D (HxW), got shape {classes.shape}"
            )
        if min(classes.shape) <= 128:
            raise ObservationError(
                "'bev' semantic raster is centre-cropped by 64 px per side, so it "
                f"must exceed 128x128; got {classes.shape}"
            )
        palette = torch.tensor(PlanTVariables.bev_colors)
        rotated = np.rot90(classes)
        # Indexing requires Long/Byte/Bool — torch.int (int32) raises on modern PyTorch.
        cropped = torch.as_tensor(
            rotated[64:-64, 64:-64].copy(), dtype=torch.long
        ).clamp(0, max(palette.shape[0] - 1, 0))
        return palette[cropped].permute(2, 0, 1)

    def _encode_objects(self, objects: Iterable[Mapping[str, Any]]) -> list[list[float]]:
        """Object tokens ``[class, x, y, yaw_deg, speed_kph, width_m, length_m]``.

        A faithful port of the labelling in ``PlanTAgent.get_input_batch``: the
        class remapping, the range gate, the walker-has-moved rule and the
        published static extents all matter for behaviour, so they are applied
        here rather than left to each calling method.
        """
        from plant_variables import PlanTVariables  # noqa: PLC0415
        from util.static_extents import CAR_EXTENTS, STATIC_EXTENTS  # noqa: PLC0415

        # A copy: the reference agent mutates the shared class attribute.
        type_nums = dict(PlanTVariables.class_nums)
        if self.drop_static_class:
            type_nums.pop("static", None)
        if not self._model_flags.get("input_static_cars"):
            type_nums.pop("static_car", None)

        car_types = set(PlanTVariables.car_types)
        input_range = self._model_flags.get("range")
        front_factor = self._model_flags.get("range_factor_front") or 1

        cars: list[list[float]] = []
        others: list[list[float]] = []

        for index, obj in enumerate(objects):
            if not isinstance(obj, Mapping):
                raise ObservationError(
                    f"objects[{index}] must be a mapping, got {type(obj).__name__}"
                )
            name = _object_class(obj)
            pos_x, pos_y, pos_z = _as_xyz(
                obj.get("position", obj.get("location")), f"objects[{index}].position"
            )

            # Range gate, as in ``PlanTAgent.run_step``.
            if input_range:
                divisor = front_factor**2 if pos_x > 0 else 1
                if (
                    pos_x**2 / divisor + pos_y**2 > float(input_range) ** 2
                    or abs(pos_z) > 30
                ):
                    continue

            speed = float(obj.get("speed_mps", obj.get("speed", 0.0)) or 0.0)
            extent = list(_as_xyz(obj.get("extent") or [1.5, 1.5, 0.5], f"objects[{index}].extent"))
            type_id = obj.get("type_id")

            if name == "walker":
                identity = obj.get("id", ("walker", index))
                if speed < 0.1 and identity not in self._moving_walkers:
                    # A walker that has never moved is not an obstacle yet.
                    continue
                self._moving_walkers.add(identity)
            elif name == "car" and type_id in EMERGENCY_TYPE_IDS:
                name = "emergency"
            elif name == "static":
                if type_id is not None and type_id not in RELEVANT_STATIC_TYPE_IDS:
                    continue
                if type_id in STATIC_EXTENTS:
                    extent = list(STATIC_EXTENTS[type_id])
            elif name == "static_car":
                mesh_path = obj.get("mesh_path")
                if mesh_path in CAR_EXTENTS:
                    extent = list(CAR_EXTENTS[mesh_path])
                    scale = obj.get("scale")
                    if scale is not None:
                        extent = [component * float(scale) for component in extent]
            elif name == "traffic_light":
                if str(obj.get("state", "Red")) not in BLOCKING_LIGHT_STATES:
                    continue

            if name not in type_nums:
                continue

            width = extent[1] * 2
            if "Door" in str(obj.get("scenario") or ""):
                # ``VehicleOpensDoor``: the open door widens the obstacle.
                width += 1

            if name in car_types:
                cars.append(
                    [
                        type_nums[name],
                        pos_x,
                        pos_y,
                        _rad2deg(obj.get("yaw_rad", obj.get("yaw", 0.0)) or 0.0),
                        speed * 3.6,
                        width,
                        extent[0] * 2,
                    ]
                )
            else:
                others.append(
                    [
                        type_nums[name],
                        pos_x,
                        pos_y,
                        _rad2deg(obj.get("yaw_rad", obj.get("yaw", 0.0)) or 0.0),
                        0.0,
                        width,
                        extent[0] * 2,
                    ]
                )

        # Cars first, then everything else: the reference agent's token order.
        return cars + others

    # -- model output -> action -------------------------------------------

    @staticmethod
    def interpolate_waypoints(waypoints: Any) -> Any:
        """Resample a polyline to one point every 0.1 m.

        Ported from ``PlanTAgent.interpolate_waypoints``; the lateral controller
        is tuned for this spacing.
        """
        import numpy as np  # noqa: PLC0415
        from scipy.interpolate import PchipInterpolator  # noqa: PLC0415

        waypoints = np.asarray(waypoints, dtype=np.float64).copy()
        waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
        shift = np.roll(waypoints, 1, axis=0)
        shift[0] = shift[1]

        dists = np.linalg.norm(waypoints - shift, axis=1)
        dists = np.cumsum(dists)
        # Keeps the interpolation knots strictly increasing when the prediction
        # is stationary.
        dists += np.arange(0, len(dists)) * 1e-4

        interp = PchipInterpolator(dists, waypoints, axis=0)
        interp_points = interp(np.arange(0.1, dists[-1], 0.1))
        if interp_points.shape[0] == 0:
            # Every point sits at the origin: fall back to the furthest one.
            interp_points = waypoints[None, -1]
        return interp_points

    def _build_action(
        self,
        pred_path: Any,
        pred_wps: Any,
        pred_speed: Any,
        ego_speed: float,
    ) -> dict[str, Any]:
        import numpy as np  # noqa: PLC0415
        from torch.nn import functional as F  # noqa: PLC0415

        from plant_variables import PlanTVariables  # noqa: PLC0415

        path = None if pred_path is None else pred_path.detach().squeeze().cpu().numpy()
        wps = None if pred_wps is None else pred_wps.detach().squeeze().cpu().numpy()

        # Target speed, exactly as ``PlanTAgent._get_control`` derives it.
        if pred_speed is not None:
            distribution = F.softmax(pred_speed.detach().squeeze().cpu(), dim=0).numpy()
            target_speed = float(
                np.sum(np.asarray(PlanTVariables.target_speeds) * distribution)
            )
        elif wps is not None:
            target_speed = float(np.linalg.norm(wps[2] - wps[3]) * 4.0)
            mean_speed = float(
                np.linalg.norm(wps[:-1] - wps[1:], axis=-1).mean() * 4.0
            )
            if ego_speed < 0.01:
                # Creep heuristic: never claim a standstill is a target speed.
                target_speed = min(mean_speed, 0.1)
        else:
            raise PolicyError(
                "checkpoint predicts neither waypoints nor a speed distribution"
            )

        steering_reference = path if path is not None else wps
        returned = self._trim(wps)
        action: dict[str, Any] = {
            "waypoints": returned,
            "path": None if path is None else [[float(x), float(y)] for x, y in path],
            "target_speed_mps": target_speed,
            "meta": {
                "policy": POLICY_NAME,
                "interface": INTERFACE,
                "step": self._step,
                "frame_ego": "x_forward_y_right_metres",
                "waypoints_predicted": None if wps is None else int(len(wps)),
                "waypoints_returned": None if returned is None else len(returned),
                "num_waypoints_requested": self.num_waypoints,
                "settling_ticks": INITIAL_FRAMES_DELAY,
            },
        }

        control = self._control(steering_reference, target_speed, ego_speed)
        if control is not None:
            if self._step < INITIAL_FRAMES_DELAY:
                # ``PlanTAgent.run_step`` holds the brake while the simulator
                # settles. The prediction is still reported as-is; only the
                # convenience control reflects it, so a method that drives from
                # 'control' reproduces PlanT's behaviour tick for tick.
                control = {"steer": 0.0, "throttle": 0.0, "brake": 1.0}
                action["meta"]["settling"] = True
            action["control"] = control
        return action

    def _trim(self, wps: Any) -> list[list[float]] | None:
        """Return at most ``num_waypoints`` predicted waypoints."""
        if wps is None:
            return None
        points = [[float(x), float(y)] for x, y in wps]
        if self.num_waypoints:
            requested = int(self.num_waypoints)
            if requested > len(points):
                # Never invent waypoints: report what the checkpoint predicts and
                # let the caller see the shortfall in ``meta``.
                return points
            return points[:requested]
        return points

    def _control(
        self,
        reference: Any,
        target_speed: float,
        ego_speed: float,
    ) -> dict[str, float] | None:
        """PlanT's own controllers, so a method can drive the sim directly.

        Returns ``None`` when the controllers cannot be built here; the declared
        action space is ``waypoints``, so this is strictly an extra.
        """
        import numpy as np  # noqa: PLC0415

        if reference is None:
            return None
        if self._controllers is None:
            try:
                from lateral_controller import LateralPIDController  # noqa: PLC0415
                from longitudinal_controller import (  # noqa: PLC0415
                    LongitudinalLinearRegressionController,
                )

                config = self._config or _import_global_config()
                self._config = config
                self._controllers = {
                    "lateral": LateralPIDController(config),
                    "longitudinal": LongitudinalLinearRegressionController(config),
                }
            except Exception as exc:  # noqa: BLE001 - an extra must never break act
                print(
                    f"[{POLICY_NAME}] controllers unavailable, returning waypoints "
                    f"only: {exc}",
                    file=sys.stderr,
                )
                self._controllers = {}
        if not self._controllers:
            return None

        throttle, brake = self._controllers["longitudinal"].get_throttle_and_brake(
            target_speed < 0.05, target_speed, ego_speed
        )
        if ego_speed < 0.05 and brake:
            # Held at a stop: steer straight so the PID's integral term does not
            # wind up against a stationary reference.
            straight = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
            steer = self._controllers["lateral"].step(
                straight, ego_speed, np.array([0.0, 0.0]), 0.0, False
            )
        else:
            steer = self._controllers["lateral"].step(
                self.interpolate_waypoints(reference),
                ego_speed,
                np.array([0.0, 0.0]),
                0.0,
                False,
            )
        return {
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
        }

    # -- introspection -----------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        """What this policy is, for a runner's provenance record."""
        return {
            "name": POLICY_NAME,
            "interface": INTERFACE,
            "observation_space": OBSERVATION_SPACE,
            "action_space": ACTION_SPACE,
            "schema_version": SCHEMA_VERSION,
            "repository": str(REPO_ROOT),
            "experiment_id": self.request.experiment_id,
            "seed": self.seed,
            "parameters": dict(self.parameters),
            "checkpoint": None if self._checkpoint is None else str(self._checkpoint),
            "checkpoint_reason": self._checkpoint_reason,
            "device": getattr(self, "device", None),
            "loaded": self._model is not None,
            "model": dict(self._model_flags),
            "route_points": ROUTE_POINTS,
            "speed_limits_kph": list(SPEED_LIMITS_KPH),
        }


# ---------------------------------------------------------------------------
# Entry points the contract asks for
# ---------------------------------------------------------------------------


def build_policy(
    request: PolicyRequest | Mapping[str, Any] | str | Path | None = None,
) -> PlanT2Policy:
    """Construct the policy from a ``PolicyRequest``.

    This is the function the contract names: a method repository's runner calls
    it with the parsed ``policy.json`` (or its path) and gets back an
    ``ego_policy_v1`` object. The network is not loaded yet -- call ``load()``,
    or just ``act()``.
    """
    if isinstance(request, (str, Path)):
        request = PolicyRequest.from_json(request)
    return PlanT2Policy(request)


#: Alias for runners that look for a conventional factory name.
make_policy = build_policy


def load_policy(policy_request_path: str | Path) -> PlanT2Policy:
    """``build_policy`` from a serialized ``policy.json``, network loaded."""
    return build_policy(PolicyRequest.from_json(policy_request_path)).load()


def leaderboard_agent(
    request: PolicyRequest | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """How to run PlanT as a CARLA Leaderboard 2.0 agent, for methods that own CARLA.

    A method that already drives the CARLA leaderboard does not need the
    step-wise interface above: it needs an ``--agent`` path and the environment
    that agent reads. Handing that back from the same policy request is what
    keeps such a method at one integration instead of one per policy.
    """
    if not isinstance(request, PolicyRequest):
        request = PolicyRequest.from_dict(request)
    checkpoint, reason = resolve_checkpoint(request)
    bench2drive = bool(request.parameters.get("bench2drive", False))
    return {
        "agent": str(REPO_ROOT / "PlanT" / "PlanT_agent.py"),
        "entry_point": "PlanTAgent",
        "agent_config": "",
        # PlanT consumes the privileged map track; Bench2Drive runs on SENSORS.
        "track": "SENSORS" if bench2drive else "MAP",
        "checkpoint": str(checkpoint),
        "checkpoint_reason": reason,
        "leaderboard_root": str(REPO_ROOT / "leaderboard_autopilot"),
        "scenario_runner_root": str(REPO_ROOT / "scenario_runner_autopilot"),
        "evaluator": str(
            REPO_ROOT
            / "leaderboard_autopilot"
            / "leaderboard"
            / "leaderboard_evaluator_local.py"
        ),
        "env": {
            "PLANT_CHECKPOINT": str(checkpoint),
            # The agent reads this unconditionally; empty disables visualisation.
            "PLANT_VIZ": os.environ.get("PLANT_VIZ", ""),
            "WORK_DIR": str(REPO_ROOT),
            "LEADERBOARD_ROOT": str(REPO_ROOT / "leaderboard_autopilot"),
            "SCENARIO_RUNNER_ROOT": str(REPO_ROOT / "scenario_runner_autopilot"),
            "IS_BENCH2DRIVE": "1" if bench2drive else "0",
            "PYTHONPATH": os.pathsep.join(
                # PlanT before carla_garage: both ship a ``model.py``.
                ([str(Path(carla_root) / "PythonAPI" / "carla")] if (carla_root := os.environ.get("CARLA_ROOT")) else [])
                + [
                    str(REPO_ROOT / "leaderboard_autopilot"),
                    str(REPO_ROOT / "scenario_runner_autopilot"),
                    str(REPO_ROOT / "PlanT"),
                    str(REPO_ROOT / "carla_garage"),
                ]
            ),
        },
        "traffic_manager_seed": int(request.seed),
    }


def describe() -> dict[str, Any]:
    """The policy's declaration, for cross-checking against the harness config."""
    return {
        "name": POLICY_NAME,
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "observation_space": OBSERVATION_SPACE,
        "action_space": ACTION_SPACE,
        "implementation": "plant2.agent.PlanTAgent",
        "entry_point": "scenario_orchestration/policy.py",
        "repository": str(REPO_ROOT),
        "checkpoints": [p.name for p in _available_checkpoints(CHECKPOINT_DIR)],
        "route_points": ROUTE_POINTS,
        "speed_limits_kph": list(SPEED_LIMITS_KPH),
        "notes": [
            "Every released PlanT 2.0 checkpoint is trained with input_bev=True, "
            "so observations must carry a BEV raster.",
            "PlanT is route-conditioned: exactly 20 ego-frame route points are "
            "embedded, sampled one per metre from 2.5 m ahead.",
            "'control' is returned alongside 'waypoints' as a convenience, from "
            "PlanT's own lateral/longitudinal controllers.",
        ],
    }


def example_observation(with_bev: bool = True) -> dict[str, Any]:
    """A representative ``state`` observation, for validation and documentation.

    A red-light interaction: the ego runs straight towards a signalised
    intersection it has a red light for, while another vehicle crosses from the
    left and a pedestrian waits on the kerb.
    """
    observation: dict[str, Any] = {
        "ego": {"speed_mps": 6.0},
        "objects": [
            {
                "type": "car",
                "position": [18.0, -9.0, 0.0],
                "yaw_rad": 1.5708,
                "speed_mps": 7.0,
                "extent": [2.45, 1.06, 0.75],
                "id": 101,
                "type_id": "vehicle.tesla.model3",
            },
            {
                "type": "car",
                "position": [30.0, 0.4, 0.0],
                "yaw_rad": 0.0,
                "speed_mps": 4.0,
                "extent": [2.45, 1.06, 0.75],
                "id": 102,
            },
            {
                "type": "walker",
                "position": [14.0, 5.5, 0.0],
                "yaw_rad": -1.5708,
                "speed_mps": 1.1,
                "extent": [0.18, 0.18, 0.93],
                "id": 201,
            },
            {
                "type": "traffic_light",
                "position": [22.0, 0.0, 0.0],
                "yaw_rad": 0.0,
                "state": "Red",
            },
        ],
        # One point per metre from 2.5 m ahead, straight on: the sampling
        # ``PlanT/PlanT_agent.py`` feeds the model.
        "route": [[2.5 + index, 0.0] for index in range(ROUTE_POINTS)],
        "speed_limit_kph": 50,
    }
    if with_bev:
        import numpy as np  # noqa: PLC0415

        # Class ids follow ``PlanTVariables.bev_colors``: 0 background,
        # 1 street, 2 sidewalk, 3 solid lines, 4 broken lines.
        bev = np.zeros((256, 256), dtype=np.int64)
        bev[:, 108:148] = 1
        bev[:, 127:129] = 4
        observation["bev"] = {"semantic_classes": bev}
    return observation


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def self_test(verbose: bool = True) -> int:
    """Exercise the contract surface, skipping what this environment cannot run.

    The checks are tiered on purpose: everything up to checkpoint resolution runs
    on a bare stdlib Python, so the contract can be validated from the harness's
    own environment, while the inference checks need this repository's
    environment (torch, transformers, timm, scipy).
    """
    checks: list[tuple[str, str, str]] = []

    def record(name: str, status: str, detail: str = "") -> None:
        checks.append((name, status, detail))
        if verbose:
            mark = {"pass": "ok  ", "skip": "skip", "fail": "FAIL"}[status]
            suffix = f": {detail}" if detail else ""
            print(f"[{mark}] {name}{suffix}")

    def run(name: str, fn: Any, skip_on: tuple[type, ...] = ()) -> Any:
        try:
            result = fn()
        except skip_on as exc:  # noqa: BLE001 - the point is to classify it
            record(name, "skip", f"{type(exc).__name__}: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001
            record(name, "fail", f"{type(exc).__name__}: {exc}")
            return None
        record(name, "pass", "" if result is None else str(result))
        return result

    # -- declaration -------------------------------------------------------

    def declaration() -> str:
        described = describe()
        assert described["interface"] == INTERFACE
        assert described["observation_space"] == OBSERVATION_SPACE
        assert described["action_space"] == ACTION_SPACE
        assert described["entry_point"] == "scenario_orchestration/policy.py"
        return f"{POLICY_NAME} {OBSERVATION_SPACE} -> {ACTION_SPACE}"

    run("declaration matches configs/policy/plant2.yaml", declaration)

    # The document the harness actually writes for this policy.
    harness_request = {
        "schema_version": "1.0.0",
        "experiment_id": "red_light__orchestration__plant2__s003",
        "name": "plant2",
        "interface": "ego_policy_v1",
        "implementation": "plant2.agent.PlanTAgent",
        "observation_space": "state",
        "action_space": "waypoints",
        "seed": 3,
        "repository": "third_party/plant2",
        "entry_point": "scenario_orchestration/policy.py",
        "checkpoint": "third_party/plant2/checkpoints/plant2.ckpt",
        "parameters": {"num_waypoints": 4},
        "requires": ["submodule", "checkpoint", "torch"],
        "an_unknown_future_field": True,
    }

    def parse_request() -> str:
        request = PolicyRequest.from_dict(harness_request)
        request.check()
        assert request.seed == 3
        assert request.parameters["num_waypoints"] == 4
        return f"seed={request.seed} params={request.parameters}"

    run("parse a harness PolicyRequest", parse_request)

    def reject_foreign() -> str:
        rejected = 0
        for bad in (
            {"name": "idm"},
            {"interface": "ego_policy_v99"},
            {"observation_space": "sensor"},
            {"action_space": "control"},
        ):
            try:
                PolicyRequest.from_dict({**harness_request, **bad}).check()
            except PolicyError:
                rejected += 1
        assert rejected == 4, f"only {rejected}/4 incompatible requests rejected"
        return "4/4 incompatible requests rejected"

    run("reject requests this policy cannot honour", reject_foreign)

    def resolve() -> str:
        request = PolicyRequest.from_dict(harness_request)
        checkpoint, reason = resolve_checkpoint(request)
        assert checkpoint.exists(), checkpoint
        return f"{checkpoint.name} ({reason})"

    run("resolve a checkpoint", resolve, skip_on=(CheckpointError,))

    def leaderboard() -> str:
        info = leaderboard_agent(PolicyRequest.from_dict(harness_request))
        assert Path(info["agent"]).exists(), info["agent"]
        assert Path(info["evaluator"]).exists(), info["evaluator"]
        assert info["track"] == "MAP"
        return f"track={info['track']} agent={Path(info['agent']).name}"

    run("expose PlanT as a CARLA leaderboard agent", leaderboard, skip_on=(CheckpointError,))

    # -- tokenisation (needs this repository's PlanT modules) --------------

    policy = PlanT2Policy(PolicyRequest.from_dict(harness_request))

    def objects() -> str:
        # The flags a released checkpoint carries, so the range gate and the
        # static-car class are exercised without loading 450 MB of weights.
        policy._model_flags = {
            "input_bev": True,
            "input_static_cars": True,
            "range": 50,
            "range_factor_front": 2,
        }
        tokens = policy._encode_objects(
            [
                {"type": "vehicle", "position": [10.0, 0.0], "speed_mps": 5.0,
                 "extent": [2.4, 1.0, 0.8], "id": 1},
                {"type": "car", "position": [10.0, 0.0], "speed_mps": 5.0,
                 "extent": [2.4, 1.0, 0.8], "id": 2,
                 "type_id": "vehicle.ford.ambulance"},
                {"type": "pedestrian", "position": [5.0, 2.0], "speed_mps": 0.0,
                 "extent": [0.2, 0.2, 0.9], "id": 3},
                {"type": "pedestrian", "position": [5.0, 3.0], "speed_mps": 1.4,
                 "extent": [0.2, 0.2, 0.9], "id": 4},
                {"type": "traffic_light", "position": [20.0, 0.0], "state": "Green"},
                {"type": "traffic_light", "position": [20.0, 0.0], "state": "Red"},
                {"type": "car", "position": [400.0, 0.0], "speed_mps": 5.0,
                 "extent": [2.4, 1.0, 0.8], "id": 5},
            ]
        )
        classes = [row[0] for row in tokens]
        # car, emergency, the walker that is moving, then the red light.
        assert classes == [1.0, 6.0, 2.0, 5.0], classes
        # [class, x, y, yaw_deg, speed_kph, width, length]
        assert abs(tokens[0][4] - 18.0) < 1e-6, tokens[0]
        assert abs(tokens[0][5] - 2.0) < 1e-6, tokens[0]
        assert abs(tokens[0][6] - 4.8) < 1e-6, tokens[0]
        return (
            "green light dropped, still walker dropped, ambulance promoted, "
            "out-of-range car gated"
        )

    run("encode objects into PlanT tokens", objects, skip_on=(ImportError,))

    def speed_limits() -> str:
        assert snap_speed_limit_kph(60) == 50
        assert snap_speed_limit_kph(None) == 50
        assert snap_speed_limit_kph(115) == 120
        return "60 -> 50, 115 -> 120"

    run("snap speed limits to a trained embedding", speed_limits)

    def route() -> str:
        short = policy._encode_route([[1.0, 0.0], [2.0, 0.0]])
        long = policy._encode_route([[float(i), 0.0] for i in range(40)])
        assert short.shape == (ROUTE_POINTS, 2), short.shape
        assert long.shape == (ROUTE_POINTS, 2), long.shape
        assert abs(float(short[-1][0]) - 2.0) < 1e-6
        assert abs(float(long[-1][0]) - 19.0) < 1e-6
        return f"2 -> {ROUTE_POINTS} padded, 40 -> {ROUTE_POINTS} truncated"

    run("fit a route to PlanT's fixed route embedding", route, skip_on=(ImportError,))

    def missing_bev() -> str:
        policy._model_flags["input_bev"] = True
        try:
            policy._encode_bev(None)
        except ObservationError as exc:
            assert "input_bev" in str(exc)
            return "a missing BEV is an explicit ObservationError"
        raise AssertionError("a missing BEV was accepted")

    run("refuse a BEV-less observation", missing_bev, skip_on=(ImportError,))

    def bev_shape() -> str:
        encoded = policy._encode_bev(example_observation()["bev"])
        assert tuple(encoded.shape) == (3, 128, 128), tuple(encoded.shape)
        return f"256x256 classes -> {tuple(encoded.shape)}"

    run("colourise a BEV raster", bev_shape, skip_on=(ImportError,))

    # -- inference (needs torch, transformers, timm, a checkpoint) ---------

    def inference() -> str:
        ego = build_policy(harness_request)
        ego.reset(seed=3)
        action = ego.act(example_observation())
        assert action["waypoints"] is not None, action
        assert len(action["waypoints"]) == 4, action["waypoints"]
        assert all(len(point) == 2 for point in action["waypoints"])
        assert action["meta"]["waypoints_predicted"] >= 4
        assert action["path"] is not None and len(action["path"]) == 20
        assert action["target_speed_mps"] >= 0.0
        summary = (
            f"{len(action['waypoints'])} waypoints, "
            f"target_speed={action['target_speed_mps']:.2f} m/s"
        )
        if "control" in action:
            assert action["meta"].get("settling") is True, action["meta"]
            assert action["control"]["brake"] == 1.0, action["control"]
            # Past the settling window the real controllers take over.
            ego._step = INITIAL_FRAMES_DELAY
            control = ego.act(example_observation())["control"]
            assert -1.0 <= control["steer"] <= 1.0, control
            assert 0.0 <= control["throttle"] <= 1.0, control
            assert 0.0 <= control["brake"] <= 1.0, control
            summary += (
                f", settling brake held, then control(steer={control['steer']:.3f}, "
                f"throttle={control['throttle']:.3f}, brake={control['brake']:.0f})"
            )
        ego.close()
        return summary

    run(
        "run inference on the example observation",
        inference,
        skip_on=(ImportError, CheckpointError),
    )

    def responds_to_scene() -> str:
        """Guards against a wiring bug that would still produce plausible output.

        A policy that ignored its observation, or that received a mis-tokenised
        one, would happily return waypoints; what it would not do is slow down
        for a conflict and speed up without one.
        """
        ego = build_policy(harness_request)
        blocked = example_observation()
        clear = {**blocked, "objects": []}
        slow = ego.act(blocked)["target_speed_mps"]
        fast = ego.act(clear)["target_speed_mps"]
        ego.close()
        assert fast > slow + 2.0, (
            f"expected a clear road ({fast:.2f} m/s) to be driven faster than a "
            f"red light with a crossing vehicle ({slow:.2f} m/s)"
        )
        return f"conflict {slow:.2f} m/s vs clear road {fast:.2f} m/s"

    run(
        "slow down for a conflict and speed up without one",
        responds_to_scene,
        skip_on=(ImportError, CheckpointError),
    )

    def determinism() -> str:
        observation = example_observation()
        first = build_policy(harness_request)
        second = build_policy(harness_request)
        a = first.act(observation)["waypoints"]
        b = second.act(observation)["waypoints"]
        first.close()
        second.close()
        assert a == b, f"{a} != {b}"
        return "two policies at the same seed agree"

    run(
        "produce a deterministic action for a seed",
        determinism,
        skip_on=(ImportError, CheckpointError),
    )

    failures = [name for name, status, _ in checks if status == "fail"]
    skipped = [name for name, status, _ in checks if status == "skip"]
    if verbose:
        print(
            f"\n{len(checks) - len(failures) - len(skipped)} passed, "
            f"{len(skipped)} skipped, {len(failures)} failed"
        )
        for name in failures:
            print(f"  failed: {name}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _read_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(payload: Any, path: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=False)
    if path is None or path == "-":
        print(text)
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Drive the policy from the shell.

    The interface a method repository is expected to use is ``build_policy``;
    this exists so the contract can be validated without a simulator, and so a
    runner whose own environment cannot import this repository's dependencies can
    still drive PlanT over a pipe (``--serve``).
    """
    parser = argparse.ArgumentParser(
        description=f"PlanT 2.0 as a scenario_orchestration {INTERFACE} ego policy",
    )
    parser.add_argument(
        "--policy-request",
        help="path to the harness's policy.json ('-' for stdin)",
    )
    parser.add_argument(
        "--observation",
        help=(
            "path to a JSON 'state' observation ('-' for stdin); omit with "
            "--example to use the built-in one"
        ),
    )
    parser.add_argument("--output", help="where to write the action (default stdout)")
    parser.add_argument(
        "--example",
        action="store_true",
        help="step the built-in example observation",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "read one JSON observation per line on stdin and write one JSON "
            "action per line on stdout, for runners in another environment"
        ),
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print this policy's declaration and exit",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="print the resolved policy metadata (checkpoint, device) and exit",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="check the contract surface in this environment",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.describe:
        _write_json(describe(), args.output)
        return 0

    request = (
        PolicyRequest.from_dict(_read_json(args.policy_request))
        if args.policy_request
        else PolicyRequest()
    )

    if args.metadata:
        policy = build_policy(request)
        try:
            policy.load()
        except PolicyError as exc:
            print(f"[{POLICY_NAME}] {exc}", file=sys.stderr)
        _write_json(policy.metadata(), args.output)
        return 0

    policy = build_policy(request)

    if args.serve:
        policy.load()
        policy.reset()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                action = policy.act(json.loads(line))
            except (PolicyError, ValueError) as exc:
                action = {"error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(action), flush=True)
        policy.close()
        return 0

    if args.example:
        observation = example_observation()
    elif args.observation:
        observation = _read_json(args.observation)
    else:
        parser.error("one of --observation, --example, --serve, --describe, "
                     "--metadata or --self-test is required")

    policy.reset()
    action = policy.act(observation)
    policy.close()
    _write_json(action, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
