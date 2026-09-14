"""Drive LASER's VUT with the harness's own ego policy, the way the other methods do.

LASER's native egos (``idm_ego``, ``plant2_ego``, ``tfv6_ego``, ``simlingo_ego``)
are LASER's re-implementations: its own IDM and lane-centre steering, modified
copies of the PlanT2 and TFv6 adapters, and observations built its own way. The
other methods in the harness drive every ego the same way instead, and this
module does too:

* the POLICY is the policy repository's own ``scenario_orchestration/policy.py``
  named by the harness's ``policy.json`` (``$LASER_POLICY_REQUEST``, written by
  ``scenario_orchestration/run.py``) -- ``third_party/idm`` for every ``idm*``,
  ``third_party/plant2``, ``third_party/tfv6``, the simlingo checkout;
* the method-side DRIVER is ``carla_port.ego_driver.PolicyEgoDriver`` from the
  highway orchestrator, imported rather than re-derived: the ``state``
  observation (``carla_obs.ObservationBuilder``), the policy's own sensor rig
  captured for the tick it decides on (``carla_sensors.CameraRig``), decisions
  held at ``$POLICY_HZ``, and an acceleration realised by ``SpawnGear`` +
  ``AccelerationTracker`` -- the law all three other methods share;
* PlanT2's BEV raster comes from the same ``scenario_orchestration/bev.py``
  wrapper around its repository's renderer.

``$LASER_POLICY_DRIVER_ROOT`` names the orchestrator_highway tree to import from
(default: the sibling submodule). LASER provides only what that driver needs to
know about a world it did not build: CARLA world coordinates as the "script"
frame, the route LASER planned as the reference path, LASER's agent names as
actor ids, and the traffic light that governs the ego's approach.

A policy whose inference stack cannot share LASER's interpreter (TFv6, SimLingo)
runs in its own process: set ``$LASER_POLICY_PYTHON_<NAME>`` and it is built by
``scenario_orchestration/policy_worker.py`` there, behind a proxy with the same
``sensors`` / ``load`` / ``reset`` / ``act`` / ``close`` / ``metadata`` surface.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import carla

LASER_REPO = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_HZ = 20.0
WORKER_READY = "POLICY_WORKER_READY"
#: Policies whose `state` observation carries a BEV raster (the highway runner's
#: BEV_POLICIES).
BEV_POLICIES = ("plant2",)


# --------------------------------------------------------------------------- #
# The driver, from orchestrator_highway
# --------------------------------------------------------------------------- #
def driver_root() -> Path:
    env = os.environ.get("LASER_POLICY_DRIVER_ROOT")
    return Path(env) if env else LASER_REPO.parent / "orchestrator_highway"


def _load_package(name: str, directory: Path):
    """Import a package from its own directory, without putting its parent on
    sys.path (orchestrator_highway's top level has a `scenario_orchestration`
    package and flat modules that could shadow names in LASER's process)."""
    existing = sys.modules.get(name)
    if existing is not None:
        loaded_from = Path(getattr(existing, "__file__", "") or "").resolve().parent
        if loaded_from != directory.resolve():
            raise RuntimeError(
                f"{name} is already imported from {loaded_from}, not {directory}; "
                "point every importer at one tree")
        return existing
    spec = importlib.util.spec_from_file_location(
        name, str(directory / "__init__.py"), submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_file(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ego_driver_module():
    _load_package("carla_port", driver_root() / "carla_port")
    import carla_port.ego_driver as ego_driver  # noqa: PLC0415
    return ego_driver


# --------------------------------------------------------------------------- #
# Loading the policy
# --------------------------------------------------------------------------- #
def harness_root() -> Path:
    env = os.environ.get("LASER_HARNESS_ROOT")
    return Path(env) if env else LASER_REPO.parent.parent


def resolve_repository(request: Dict[str, Any]) -> tuple:
    """`$<NAME>_ROOT` first, then `policy.repository` -- the same order as the
    highway port's `policies.resolve_repository`."""
    name = str(request.get("name") or "")
    env_key = f"{name.upper().replace('-', '_')}_ROOT"
    if os.environ.get(env_key):
        return Path(os.environ[env_key]), f"${env_key}"
    declared = Path(str(request.get("repository") or ""))
    candidates = [declared] if declared.is_absolute() else [harness_root() / declared]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate, "policy.repository"
    raise RuntimeError(
        f"cannot locate the repository of policy {name!r}: {declared} is not a "
        f"directory under {harness_root()} and ${env_key} is not set")


def _entry(request: Dict[str, Any], repository: Path) -> Path:
    entry = repository / str(request.get("entry_point") or "scenario_orchestration/policy.py")
    if not entry.is_file():
        raise RuntimeError(f"policy {request.get('name')!r}: no entry point at {entry}")
    return entry


def load_in_process(request: Dict[str, Any], repository: Path):
    """What `policies.load_policy` does: import the entry point by path, build,
    load. The repository goes at the END of sys.path: LASER's own `leaderboard`,
    `srunner` and `agents` must keep winning in this process."""
    entry = _entry(request, repository)
    for path in (str(entry.parent), str(repository)):
        if path not in sys.path:
            sys.path.append(path)
    module = _load_file(f"_harness_policy_{request.get('name')}", entry)
    factory = None
    for attr in ("build_policy", "make_policy", "load_policy", "Policy"):
        factory = getattr(module, attr, None)
        if callable(factory):
            break
    if factory is None:
        raise RuntimeError(f"{entry} exposes no build_policy() factory")
    policy = factory(dict(request))
    loader = getattr(policy, "load", None)
    if callable(loader):
        policy = loader() or policy
    return policy


class RemotePolicy:
    """A policy built by `scenario_orchestration/policy_worker.py` in another
    interpreter, behind the ego_policy_v1 surface PolicyEgoDriver calls."""

    def __init__(self, request: Dict[str, Any], repository: Path, python: str,
                 workdir: Optional[str] = None, timeout_s: float = 900.0):
        self.name = str(request.get("name") or "policy")
        self._sock_path = os.path.join(tempfile.mkdtemp(prefix="laser_policy_"), "worker.sock")
        worker = LASER_REPO / "scenario_orchestration" / "policy_worker.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repository)
        env["PYTHONNOUSERSITE"] = "1"
        cmd = [python, str(worker), "--socket", self._sock_path,
               "--entry", str(_entry(request, repository)), "--repository", str(repository)]
        self._proc = subprocess.Popen(
            cmd, cwd=workdir or str(repository), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
        deadline, lines = time.time() + 120.0, []
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"{self.name} worker exited:\n" + "".join(lines))
                time.sleep(0.05)
                continue
            lines.append(line)
            print(f"[{self.name}-worker] {line.rstrip()}", flush=True)
            if WORKER_READY in line:
                break
        else:
            self.close()
            raise TimeoutError(f"{self.name} worker never became ready:\n" + "".join(lines[-30:]))
        self._conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._conn.connect(self._sock_path)
        built = self._rpc({"op": "build", "request": request}, timeout_s)
        self._sensors = list(built.get("sensors") or [])
        self._timeout_s = timeout_s
        print(f"[{self.name}-worker] built in python {built.get('python')}, "
              f"{len(self._sensors)} sensor(s) declared", flush=True)

    def sensors(self) -> List[Dict[str, Any]]:
        return [dict(s) for s in self._sensors]

    def load(self):
        self._rpc({"op": "load"}, self._timeout_s)
        return self

    def reset(self) -> None:
        self._rpc({"op": "reset"}, 60.0)

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return self._rpc({"op": "act", "observation": observation}, 120.0)["action"]

    def metadata(self) -> Dict[str, Any]:
        return self._rpc({"op": "metadata"}, 60.0).get("metadata") or {}

    def close(self) -> None:
        conn, self._conn = getattr(self, "_conn", None), None
        if conn is not None:
            try:
                raw = pickle.dumps({"op": "close"}, protocol=pickle.HIGHEST_PROTOCOL)
                conn.sendall(struct.pack("!Q", len(raw)) + raw)
            except OSError:
                pass
            conn.close()
        proc, self._proc = getattr(self, "_proc", None), None
        if proc is not None:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _rpc(self, message: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
        if self._conn is None:
            raise RuntimeError(f"{self.name} worker is not connected")
        raw = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        self._conn.settimeout(timeout_s)
        self._conn.sendall(struct.pack("!Q", len(raw)) + raw)
        (size,) = struct.unpack("!Q", self._recv(8))
        reply = pickle.loads(self._recv(size))
        if not reply.get("ok"):
            raise RuntimeError(f"{self.name} worker: {reply.get('error')}\n{reply.get('traceback', '')}")
        return reply

    def _recv(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError(f"{self.name} worker closed its socket")
            buf.extend(chunk)
        return bytes(buf)


#: orchestration/scenario_orchestration/policies.py :: PATH_PARAMETERS, the keys
#: that method and osc2runner's bridge rewrite.
PATH_PARAMETERS = ("weights", "checkpoint", "config", "config_path",
                   "model_path", "weights_path")


def absolutize(payload: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """The other methods' `_absolutize`: the harness writes checkpoint paths
    relative to ITS root, a policy resolves them against its working directory.
    Rewritten only when the rewrite lands on something that exists."""
    payload = json.loads(json.dumps(payload))

    def resolve(value):
        if not isinstance(value, str) or not value or os.path.isabs(value):
            return value
        candidate = root / value
        return os.path.abspath(str(candidate)) if candidate.exists() else value

    payload["checkpoint"] = resolve(payload.get("checkpoint"))
    parameters = payload.get("parameters")
    if isinstance(parameters, dict):
        for key in PATH_PARAMETERS:
            if key in parameters:
                parameters[key] = resolve(parameters[key])
    return payload


def load_policy(request: Dict[str, Any]):
    request = absolutize(request, harness_root())
    repository, how = resolve_repository(request)
    name = str(request.get("name") or "").upper().replace("-", "_")
    python = os.environ.get(f"LASER_POLICY_PYTHON_{name}")
    if python:
        workdir = os.environ.get(f"LASER_POLICY_WORKDIR_{name}")
        return RemotePolicy(request, repository, python, workdir), repository, f"{how}, worker {python}"
    return load_in_process(request, repository), repository, f"{how}, in process"


def bev_source(request: Dict[str, Any], repository: Path):
    """The highway runner's `_bev_source`: PlanT2's raster from its own renderer."""
    if str(request.get("name") or "") not in BEV_POLICIES:
        return None
    bev = _load_file("_harness_policy_bev", driver_root() / "scenario_orchestration" / "bev.py")
    if bev.blank_allowed():
        return None
    return bev.build(str(repository))


# --------------------------------------------------------------------------- #
# What PolicyEgoDriver needs to know about LASER's world
# --------------------------------------------------------------------------- #
class WorldFrame:
    """LASER has no script frame: CARLA world coordinates play that role."""

    def __init__(self, light=None):
        self._light = light

    @staticmethod
    def to_carla_xy(x: float, y: float):
        return x, y

    def traffic_light(self, _arm: str):
        return self._light


class AgentBindings:
    """Actor -> LASER agent name, looked up when asked (the VUT is built before
    the other agents exist)."""

    def __init__(self, agent_manager):
        self._manager = agent_manager

    def script_id_of(self, actor) -> Optional[str]:
        actor_id = getattr(actor, "id", None)
        for group in ("_target_vehicle", "_vehicles", "_pedestrians"):
            for agent in getattr(self._manager, group, None) or []:
                if getattr(getattr(agent, "carla_actor", None), "id", None) == actor_id:
                    return str(agent.name)
        return None


class _Ego:
    v = 0.0


class RouteCompanion:
    """The measurement companion's surface: the reference path, the ego's pose
    and its measured speed."""

    def __init__(self, vehicle, route):
        self.vehicle = vehicle
        self.reference_path = [(float(tf.location.x), float(tf.location.y)) for tf, _opt in route]
        self.ego = _Ego()

    def update(self) -> None:
        v = self.vehicle.get_velocity()
        self.ego.v = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    def pose(self):
        tf = self.vehicle.get_transform()
        return float(tf.location.x), float(tf.location.y), float(tf.rotation.yaw)


def ego_approach_light(world, route):
    """The light whose stop line is on the ego's lane just before the first
    junction of its route, or None."""
    carla_map = world.get_map()
    before = None
    for tf, _opt in route:
        wp = carla_map.get_waypoint(tf.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        if wp.is_junction:
            break
        before = wp
    if before is None:
        return None
    best, best_d = None, 30.0
    for light in world.get_actors().filter("traffic.traffic_light*"):
        try:
            stops = light.get_stop_waypoints()
        except RuntimeError:
            continue
        for stop in stops:
            if (stop.road_id, stop.lane_id) != (before.road_id, before.lane_id):
                continue
            d = stop.transform.location.distance(before.transform.location)
            if d < best_d:
                best, best_d = light, d
    return best


# --------------------------------------------------------------------------- #
class HarnessPolicyEgo:
    """LASER's ego controller for any harness ego_policy_v1 policy."""

    def __init__(self, vehicle, route, request_path: str, agent_manager):
        with open(request_path) as fh:
            self.request = json.load(fh)
        self.name = str(self.request.get("name") or "policy")
        world = vehicle.get_world()
        self.vehicle = vehicle
        self.agent_manager = agent_manager

        policy, repository, how = load_policy(self.request)
        self.repository = repository
        ego_driver = ego_driver_module()
        hz = float(os.environ.get("POLICY_HZ") or DEFAULT_POLICY_HZ)
        self.driver = ego_driver.PolicyEgoDriver(
            policy, name=self.name, hz=hz, bev=bev_source(self.request, repository))
        self.companion = RouteCompanion(vehicle, route)
        self.light = ego_approach_light(world, route)
        settings = world.get_settings()
        ctx = ego_driver.EgoContext(
            world=world, frame=WorldFrame(self.light), bindings=AgentBindings(agent_manager),
            ego_id="VUT", ego_actor=vehicle, mode="physics", policy=self.companion,
            fixed_delta=float(settings.fixed_delta_seconds or 0.05))
        self.driver.attach(ctx)
        self.driver.reset()
        self.provenance = {
            "policy": self.name,
            "implementation": self.request.get("implementation"),
            "repository": str(repository),
            "loaded": how,
            "driver": f"{driver_root()}/carla_port/ego_driver.py :: PolicyEgoDriver",
            "decision_hz": hz,
            "ego_approach_light": getattr(self.light, "id", None),
        }
        print(f"harness ego: {json.dumps(self.provenance)}", flush=True)

    def run_step(self, dt: float):
        self.companion.update()
        return self.driver.control(dt)

    def vision(self) -> Dict[str, Any]:
        return self.driver.vision()

    def close(self) -> None:
        try:
            meta = self.driver.metadata()
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            meta = {"metadata_error": f"{type(exc).__name__}: {exc}"}
        out_dir = os.environ.get("SAVE_PATH")
        if out_dir:
            try:
                with open(os.path.join(out_dir, "ego_driver.json"), "w") as fh:
                    json.dump({"provenance": self.provenance, "driver": meta}, fh,
                              indent=2, default=str)
            except OSError as exc:
                print(f"harness ego: could not write ego_driver.json: {exc}")
        self.driver.close()
