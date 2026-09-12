#!/usr/bin/env python3
"""Standardized execution entry point for the scenario_orchestration harness.

The harness never imports LASER internals. It writes two JSON documents, runs

    python scenario_orchestration/run.py \\
        --scenario-request request.json \\
        --policy-request policy.json \\
        --output-dir <results>/raw/<experiment_id>

and reads ``method_result.json`` back out of the output directory.

This file mirrors the OSC2Runner contract shape, but launches LASER's existing
``laser_se.py`` path instead of ``python -m osc2carla``.

MVP scope
---------
* Families: red_light, cut_in, lane_change, overtake, right_turn, left_turn
* Ego policies: transfuser / interfuser / plant2 / idm / tfv6 / simlingo (native via LASER_EGO)
* CARLA: assumed already running (override with LASER_CARLA_HOST/PORT)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHOD_RESULT_FILE = "method_result.json"
LASER_LOG_FILE = "laser.log"

#: Family -> (road flag for laser_se.py, script path relative to REPO_ROOT)
NATIVE_SCENARIOS = {
    "red_light": ("T10J189", "laser_scenes/RedLight/script.json"),
    "cut_in": ("T04CutIn", "laser_scenes/Cut-in/script.json"),
    "lane_change": ("T04HardBrake", "laser_scenes/LaneChanging/script.json"),
    "overtake": ("T01Overtake", "laser_scenes/VehiclePassing/script.json"),
    "right_turn": ("T10J189Right", "laser_scenes/RightTurn/script.json"),
    "left_turn": ("T10J189Left", "laser_scenes/UnprotectedLeftTurn/script.json"),
}

#: Harness implementations.yaml native_ids and other aliases -> family keys.
NATIVE_ALIASES = {
    "vehicle_passing": "overtake",
    "lane_changing": "lane_change",
    "red_light_running": "red_light",
    "red_light_violation": "red_light",
    # scenario_orchestration scenarios/*/implementations.yaml (laser: native_id)
    "laser_red_light_violation": "red_light",
    "laser_adjacent_cut_in": "cut_in",
    "laser_blocked_lane_change": "lane_change",
    "laser_unprotected_left": "left_turn",
    "laser_right_turn_conflict": "right_turn",
}

NATIVE_POLICY_NAMES = {
    "transfuser": "transfuser",
    "laser_transfuser": "transfuser",
    "interfuser": "interfuser",
    "laser_interfuser": "interfuser",
    "plant2": "plant2",
    "plant": "plant2",
    "laser_plant2": "plant2",
    "idm": "idm",
    "idm_conservative": "idm",
    "idm_assertive": "idm",
    "idm_highway": "idm",
    "idm_mobil": "idm",
    "laser_idm": "idm",
    "tfv6": "tfv6",
    "transfuser_v6": "tfv6",
    "laser_tfv6": "tfv6",
    "simlingo": "simlingo",
    "laser_simlingo": "simlingo",
}

DEFAULT_HORIZON_S = 15.0
DEFAULT_RUN_TIMEOUT_S = 900.0


class RequestError(Exception):
    """The request cannot be turned into a LASER invocation."""


def _read_json(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object, got %s" % type(payload).__name__)
    return payload


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_float(name: str) -> Optional[float]:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_plant2_checkpoint(
    declared: str, plant_root: Optional[str], seed: int
) -> str:
    """Resolve a PlanT2 checkpoint file, including seed-indexed directories."""
    expanded = os.path.expanduser(str(declared))
    candidates = [expanded]
    if not os.path.isabs(expanded):
        # Policy configs are relative to the central scenario_orchestration repo,
        # while this adapter executes from third_party/laser.
        harness_root = os.path.dirname(os.path.dirname(REPO_ROOT))
        candidates.extend(
            [
                os.path.join(harness_root, expanded),
                os.path.join(REPO_ROOT, expanded),
            ]
        )
        if plant_root:
            candidates.append(
                os.path.join(str(plant_root), "checkpoints", os.path.basename(expanded))
            )

    seen = set()
    for candidate in candidates:
        path = os.path.abspath(candidate)
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            checkpoints = sorted(
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.endswith(".ckpt") and os.path.isfile(os.path.join(path, name))
            )
            if not checkpoints:
                raise RequestError(
                    "PlanT2 checkpoint directory contains no .ckpt files: %s" % path
                )
            return checkpoints[int(seed) % len(checkpoints)]

    raise RequestError("PlanT2 checkpoint does not exist: %s" % declared)


_EXCEPTION_LINE = re.compile(r"^\w+(?:Error|Exception|Exit)\b.*: .+")


def _first_line_of_error(stderr: Optional[str]) -> str:
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        if _EXCEPTION_LINE.match(line):
            return line
    for line in reversed(lines):
        if "ERROR" in line or "error:" in line or "Traceback" in line:
            return line
    return lines[-1] if lines else ""


def scenario_candidates(request: Dict[str, Any]) -> List[str]:
    implementation = request.get("implementation") or {}
    parameters = implementation.get("parameters") or {}
    out: List[str] = []
    for value in (
        parameters.get("scenario"),
        implementation.get("native_id"),
        request.get("scenario_family"),
        request.get("semantic_id"),
    ):
        if not value:
            continue
        stem = os.path.basename(str(value))
        for candidate in (stem, NATIVE_ALIASES.get(stem)):
            if candidate and candidate not in out:
                out.append(candidate)
    return out


def resolve_scenario(request: Dict[str, Any]) -> Tuple[str, str, str]:
    """``(family, road, absolute script path)``."""
    parameters = (request.get("implementation") or {}).get("parameters") or {}

    explicit_script = parameters.get("script") or parameters.get("script_file")
    explicit_road = parameters.get("road")
    if explicit_script and explicit_road:
        script_path = (
            str(explicit_script)
            if os.path.isabs(str(explicit_script))
            else os.path.join(REPO_ROOT, str(explicit_script))
        )
        if not os.path.exists(script_path):
            raise RequestError("script %r does not exist" % explicit_script)
        family = str(
            request.get("scenario_family")
            or (request.get("implementation") or {}).get("native_id")
            or "custom"
        )
        return family, str(explicit_road), script_path

    tried = scenario_candidates(request)
    for name in tried:
        if name in NATIVE_SCENARIOS:
            road, rel_script = NATIVE_SCENARIOS[name]
            road = str(parameters.get("road") or road)
            script_path = os.path.join(REPO_ROOT, rel_script)
            if not os.path.exists(script_path):
                raise RequestError("native script missing: %s" % rel_script)
            return name, road, script_path

    raise RequestError(
        "no LASER scenario for family %r (tried %s; available: %s)"
        % (
            request.get("scenario_family"),
            ", ".join(tried) or "nothing",
            ", ".join(sorted(NATIVE_SCENARIOS)),
        )
    )


def resolve_horizon(evaluation: Dict[str, Any], parameters: Dict[str, Any]) -> float:
    override = _env_float("LASER_SIM_DURATION") or _as_float(parameters.get("sim_duration"))
    if override and override > 0:
        return override
    horizon = _as_float(evaluation.get("horizon_s"))
    if horizon and horizon > 0:
        return horizon
    return DEFAULT_HORIZON_S


def build_policy_plan(policy_request: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Return ``(extra_env, notes)`` for a native LASER ego stack."""
    name = str(policy_request.get("name") or "").strip().lower()
    interface = str(policy_request.get("interface") or "ego_policy_v1")
    observation_space = str(policy_request.get("observation_space") or "sensor")
    action_space = str(policy_request.get("action_space") or "control")
    parameters = dict(policy_request.get("parameters") or {})

    if interface != "ego_policy_v1":
        raise RequestError(
            "policy %r declares interface %r; LASER currently speaks 'ego_policy_v1'"
            % (name, interface)
        )
    # Harness TransFuser declares waypoints; InterFuser-style stacks declare
    # control. LASER drives both natively inside laser_se / the AD agent.
    if action_space not in ("control", "waypoints"):
        raise RequestError(
            "policy %r emits %r; native LASER stacks use control or waypoints"
            % (name, action_space)
        )

    resolved = NATIVE_POLICY_NAMES.get(name)
    if resolved is None:
        raise RequestError(
            "policy %r is not one LASER realizes natively (%s)."
            % (name or "(unnamed)", ", ".join(sorted(set(NATIVE_POLICY_NAMES.values()))))
        )

    if observation_space not in ("sensor", "state"):
        raise RequestError(
            "policy %r wants observation_space %r; native LASER policies use "
            "sensor or state"
            % (name, observation_space)
        )
    if resolved in ("plant2", "idm"):
        notes_space = "state"
    elif observation_space == "state":
        notes_space = "requested state; native stack still consumes sensors"
    else:
        notes_space = "sensor"

    env: Dict[str, str] = {"LASER_EGO": resolved}
    if resolved == "interfuser":
        # InterfuserAgent opens a pygame debug window; headless nodes need dummy SDL.
        env.setdefault("SDL_VIDEODRIVER", "dummy")
    ckpt = (
        parameters.get("checkpoint")
        or policy_request.get("checkpoint")
        or _env("TRANSFUSER_CKPT")
    )
    if resolved == "transfuser":
        if not ckpt:
            default_ckpt = os.path.expanduser(
                "~/scratch/transfuser/model_ckpt/models_2022/transfuser"
            )
            if os.path.isdir(default_ckpt):
                ckpt = default_ckpt
        if ckpt:
            env["TRANSFUSER_CKPT"] = str(ckpt)
        else:
            raise RequestError(
                "transfuser requires TRANSFUSER_CKPT or policy.checkpoint / "
                "parameters.checkpoint"
            )

    if resolved == "plant2":
        seed = int(policy_request.get("seed") or 0)
        plant_root = _env("PLANT2_ROOT")
        if not plant_root:
            sibling = os.path.join(os.path.dirname(REPO_ROOT), "plant2")
            if os.path.isdir(sibling):
                plant_root = sibling
        if plant_root:
            env["PLANT2_ROOT"] = os.path.abspath(os.path.expanduser(str(plant_root)))

        plant_ckpt = (
            parameters.get("checkpoint")
            or policy_request.get("checkpoint")
            or _env("PLANT2_CHECKPOINT")
            or _env("PLANT_CHECKPOINT")
        )
        if plant_ckpt:
            checkpoint = _resolve_plant2_checkpoint(
                str(plant_ckpt), plant_root, seed
            )
            env["PLANT2_CHECKPOINT"] = checkpoint
            env["PLANT_CHECKPOINT"] = checkpoint
        # Seed for checkpoint selection among epoch=029_final_{1,2,3}.ckpt
        env["PLANT2_SEED"] = str(seed)
        # Map IDM-style aggressiveness knobs onto nothing for plant2; speed limit only.
        if "desired_speed_mps" in parameters:
            # Informative only — PlanT is not IDM; ignore for control.
            pass

    if resolved == "idm":
        # Propagate harness IDM parameters into the LASER IDM env knobs.
        mapping = {
            "desired_speed_mps": "IDM_DESIRED_SPEED_MPS",
            "time_headway_s": "IDM_TIME_HEADWAY_S",
            "min_gap_m": "IDM_MIN_GAP_M",
            "max_accel_mps2": "IDM_MAX_ACCEL_MPS2",
            "comfort_decel_mps2": "IDM_COMFORT_DECEL_MPS2",
        }
        for src, dst in mapping.items():
            if src in parameters and parameters[src] is not None:
                env[dst] = str(parameters[src])
        # idm_mobil is a named policy (always MOBIL on). Other IDM names still
        # get MOBIL auto-enabled only for lane_change / overtake in main().
        if name == "idm_mobil":
            env["IDM_ENABLE_MOBIL"] = "1"
            if "lane_width_m" in parameters and parameters["lane_width_m"] is not None:
                # LASER IDM uses half-width for corridor checks.
                env["IDM_LANE_HALF_WIDTH_M"] = str(float(parameters["lane_width_m"]) / 2.0)
            mobil_map = {
                "politeness": "MOBIL_P",
                "a_thr": "MOBIL_A_THR",
                "b_safe": "MOBIL_B_SAFE",
            }
            for src, dst in mobil_map.items():
                if src in parameters and parameters[src] is not None:
                    env[dst] = str(parameters[src])

    if resolved == "tfv6":
        tfv6_root = _env("TFV6_ROOT")
        if not tfv6_root:
            sibling = os.path.join(os.path.dirname(REPO_ROOT), "tfv6")
            if os.path.isdir(sibling):
                tfv6_root = sibling
        if tfv6_root:
            env["TFV6_ROOT"] = os.path.abspath(os.path.expanduser(str(tfv6_root)))

        tfv6_ckpt = (
            parameters.get("checkpoint")
            or policy_request.get("checkpoint")
            or _env("TFV6_CHECKPOINT")
        )
        if tfv6_ckpt:
            # Policy yaml may use a harness-relative path (third_party/...).
            ckpt_path = str(tfv6_ckpt)
            if not os.path.isabs(ckpt_path):
                harness_root = os.path.dirname(os.path.dirname(REPO_ROOT))
                candidate = os.path.join(harness_root, ckpt_path)
                if os.path.isdir(candidate) or os.path.isfile(candidate):
                    ckpt_path = candidate
            env["TFV6_CHECKPOINT"] = os.path.abspath(os.path.expanduser(ckpt_path))
        seed = int(policy_request.get("seed") or 0)
        env["TFV6_SEED"] = str(seed)
        # Real RGB for sensorimotor eval (unless the caller already opted out).
        env.setdefault("LASER_NO_RENDERING", "0")

    if resolved == "simlingo":
        simlingo_root = _env("SIMLINGO_ROOT")
        if not simlingo_root:
            sibling = os.path.join(os.path.dirname(REPO_ROOT), "simlingo")
            if os.path.isdir(sibling):
                simlingo_root = sibling
        if simlingo_root:
            env["SIMLINGO_ROOT"] = os.path.abspath(os.path.expanduser(str(simlingo_root)))

        harness_root = os.path.dirname(os.path.dirname(REPO_ROOT))

        def _abs_harness(path: Any) -> Optional[str]:
            if path is None:
                return None
            raw = str(path)
            if not os.path.isabs(raw):
                candidate = os.path.join(harness_root, raw)
                if os.path.isdir(candidate) or os.path.isfile(candidate):
                    raw = candidate
            return os.path.abspath(os.path.expanduser(raw))

        simlingo_ckpt = (
            parameters.get("checkpoint")
            or policy_request.get("checkpoint")
            or _env("SIMLINGO_CHECKPOINT")
        )
        resolved_ckpt = _abs_harness(simlingo_ckpt)
        if resolved_ckpt:
            env["SIMLINGO_CHECKPOINT"] = resolved_ckpt

        simlingo_weights = parameters.get("weights") or _env("SIMLINGO_WEIGHTS")
        resolved_weights = _abs_harness(simlingo_weights)
        if resolved_weights:
            env["SIMLINGO_WEIGHTS"] = resolved_weights

        for src, dst in (
            ("device", "SIMLINGO_DEVICE"),
            ("mode_token", "SIMLINGO_MODE_TOKEN"),
            ("instruction", "SIMLINGO_INSTRUCTION"),
        ):
            if src in parameters and parameters[src] is not None:
                env[dst] = str(parameters[src])

        seed = int(policy_request.get("seed") or 0)
        env["SIMLINGO_SEED"] = str(seed)
        env.setdefault("AV_CKPT", os.path.join(harness_root, "third_party", "checkpoints"))
        env.setdefault("LASER_NO_RENDERING", "0")

    notes = {
        "policy_mode": "native",
        "policy_resolved": resolved,
        "observation_space_note": notes_space,
    }
    return env, notes


def build_command(road: str, script_path: str, horizon_s: float) -> List[str]:
    interpreter = _env("LASER_PYTHON") or sys.executable
    return [
        interpreter,
        os.path.join(REPO_ROOT, "laser_se.py"),
        "-r",
        road,
        "-s",
        script_path,
        "-t",
        str(int(round(horizon_s))),
        "--host",
        _env("LASER_CARLA_HOST") or _env("CARLA_HOST") or "127.0.0.1",
        "-p",
        _env("LASER_CARLA_PORT") or _env("CARLA_PORT") or "2000",
    ]


def child_environment(extra: Dict[str, str], seed: int) -> Dict[str, str]:
    env = dict(os.environ)
    # Prefer TransFuser team_code ahead of LASER's own utils to avoid import clashes.
    transfuser_code = os.path.expanduser("~/scratch/transfuser/team_code_transfuser")
    python_path = []
    if os.path.isdir(transfuser_code):
        python_path.append(transfuser_code)
    python_path.append(REPO_ROOT)
    existing = [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p]
    for p in existing:
        if p not in python_path:
            python_path.append(p)
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONHASHSEED", str(seed))
    env.setdefault("SCENARIO_ORCHESTRATION_SEED", str(seed))
    if not env.get("OPENAI_API_KEY"):
        # LASER constructs ChatOpenAI even when 1-step scripts skip GPT calls.
        env.setdefault("OPENAI_API_KEY", "unused-for-one-step-scripts")
    env.update(extra)
    return env


def latest_se_record(before: float) -> Optional[str]:
    """Pick the se_records directory created for this run."""
    root = os.path.join(REPO_ROOT, "se_records")
    if not os.path.isdir(root):
        return None
    candidates = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime + 1.0 >= before:
            candidates.append((mtime, path))
    if not candidates:
        # Fall back to newest directory.
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                try:
                    candidates.append((os.path.getmtime(path), path))
                except OSError:
                    pass
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def collect_artifacts(record_dir: Optional[str], output_dir: str) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {"se_record_dir": None}
    if not record_dir or not os.path.isdir(record_dir):
        return artifacts

    rel_name = os.path.basename(record_dir)
    dest = os.path.join(output_dir, "se_records", rel_name)
    try:
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(record_dir, dest)
        artifacts["se_record_dir"] = os.path.join("se_records", rel_name)
    except OSError as exc:
        artifacts["se_record_copy_error"] = str(exc)
        artifacts["se_record_dir"] = record_dir

    collisions_path = os.path.join(record_dir, "collisions.txt")
    time_path = os.path.join(record_dir, "time.txt")
    token_path = os.path.join(record_dir, "token.txt")

    collision = False
    time_to_event = None
    if os.path.exists(collisions_path):
        with open(collisions_path) as fh:
            text = fh.read().strip()
        if text:
            collision = True
            first = text.splitlines()[0]
            stamp = first.split(",", 1)[0].strip()
            time_to_event = _as_float(stamp)

    scenario_duration = None
    if os.path.exists(time_path):
        try:
            payload = _read_json(time_path)
            scenario_duration = _as_float(payload.get("simulation_time"))
        except (OSError, ValueError):
            pass

    token_usage = None
    if os.path.exists(token_path):
        try:
            token_usage = _read_json(token_path)
        except (OSError, ValueError):
            pass

    videos = [f for f in os.listdir(record_dir) if f.endswith(".mp4")]
    if videos:
        artifacts["video_path"] = os.path.join(
            artifacts.get("se_record_dir") or record_dir, videos[0]
        )

    artifacts.update(
        {
            "collision": collision,
            "time_to_event": time_to_event,
            "scenario_duration": scenario_duration,
            "token_usage": token_usage,
        }
    )
    return artifacts


def run_laser(
    command: Sequence[str], env: Dict[str, str], timeout_s: float, output_dir: str
) -> Tuple[Optional[int], str, str, bool]:
    started = time.time()
    timed_out = False
    try:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout_s,
        )
        stdout, stderr, returncode = (
            completed.stdout,
            completed.stderr,
            completed.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (
            exc.stdout.decode("utf-8", "replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", "replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        returncode = None

    sys.stdout.write(stdout or "")
    sys.stderr.write(stderr or "")
    try:
        with open(os.path.join(output_dir, LASER_LOG_FILE), "w") as fh:
            fh.write(stdout or "")
            fh.write(stderr or "")
    except OSError:
        pass
    sys.stderr.write(
        "[run.py] laser_se finished in %.1fs (returncode=%s)\n"
        % (time.time() - started, returncode)
    )
    return returncode, stdout, stderr, timed_out


def write_result(
    output_dir: str,
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    method_metrics: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    trace_path: Optional[str] = None,
) -> int:
    report = {
        "status": status,
        "metrics": dict(metrics or {}),
        "method_metrics": dict(method_metrics or {}),
        "trace_path": trace_path,
        "reason": reason,
    }
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, METHOD_RESULT_FILE), "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    except OSError as exc:
        sys.stderr.write("[run.py] could not write %s: %s\n" % (METHOD_RESULT_FILE, exc))
        return 1
    sys.stderr.write(
        "[run.py] %s -> status=%s%s\n"
        % (METHOD_RESULT_FILE, status, " (%s)" % reason if reason else "")
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one scenario_orchestration experiment on LASER"
    )
    parser.add_argument("--scenario-request", required=True)
    parser.add_argument("--policy-request", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    output_dir = os.path.abspath(args.output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        sys.stderr.write("[run.py] cannot create %s: %s\n" % (output_dir, exc))
        return 1

    try:
        request = _read_json(args.scenario_request)
    except (OSError, ValueError) as exc:
        return write_result(
            output_dir, "error", reason="unreadable scenario request: %s" % exc
        )
    try:
        policy_request = _read_json(args.policy_request)
    except (OSError, ValueError) as exc:
        return write_result(
            output_dir, "error", reason="unreadable policy request: %s" % exc
        )

    experiment_id = str(
        request.get("experiment_id")
        or _env("SCENARIO_ORCHESTRATION_EXPERIMENT_ID")
        or ""
    )
    seed = int(
        request.get("seed")
        or _env("SCENARIO_ORCHESTRATION_SEED")
        or 0
    )
    evaluation = dict(request.get("evaluation") or {})
    implementation = dict(request.get("implementation") or {})
    parameters = dict(implementation.get("parameters") or {})

    context: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "seed": seed,
        "evaluation_horizon_s": evaluation.get("horizon_s"),
        "requested_native_id": implementation.get("native_id"),
        "requested_parameters": parameters,
        "policy_requested": policy_request.get("name"),
    }

    try:
        family, road, script_path = resolve_scenario(request)
        horizon_s = resolve_horizon(evaluation, parameters)
        policy_env, policy_notes = build_policy_plan(policy_request)
    except RequestError as exc:
        return write_result(
            output_dir, "failure", method_metrics=context, reason=str(exc)
        )

    # IDM+MOBIL only for Merge (lane_change) and Overtake — not cut-in / turns.
    if policy_env.get("LASER_EGO") == "idm" and family in ("lane_change", "overtake"):
        policy_env["IDM_ENABLE_MOBIL"] = "1"
        policy_notes = dict(policy_notes)
        policy_notes["idm_mobil"] = True

    context.update(policy_notes)
    context.update(
        {
            "scenario_family_resolved": family,
            "road": road,
            "script": os.path.relpath(script_path, REPO_ROOT),
            "sim_duration_s": horizon_s,
        }
    )

    command = build_command(road, script_path, horizon_s)
    timeout_s = _env_float("LASER_RUN_TIMEOUT_S") or DEFAULT_RUN_TIMEOUT_S
    context["command"] = list(command)
    context["timeout_s"] = timeout_s

    if context.get("policy_resolved") == "interfuser":
        # TargetVehicleRecorder sets SAVE_PATH; InterfuserAgent.setup expects SCENE too.
        policy_env.setdefault("SCENE", script_path)

    sys.stderr.write(
        "[run.py] %s: %s on %s (%s, %.1fs)\n"
        % (
            experiment_id or family,
            family,
            road,
            context.get("policy_resolved", "?"),
            horizon_s,
        )
    )

    started = time.time()
    returncode, _stdout, stderr, timed_out = run_laser(
        command, child_environment(policy_env, seed), timeout_s, output_dir
    )
    context["wall_time_s"] = round(time.time() - started, 3)
    context["returncode"] = returncode

    artifacts = collect_artifacts(latest_se_record(started), output_dir)
    context.update({k: v for k, v in artifacts.items() if k not in ("collision", "time_to_event", "scenario_duration")})

    if timed_out:
        return write_result(
            output_dir,
            "timeout",
            metrics={
                "collision": bool(artifacts.get("collision")),
                "scenario_duration": artifacts.get("scenario_duration"),
            },
            method_metrics=context,
            reason="LASER exceeded its %.0fs wall-clock budget on %s"
            % (timeout_s, family),
        )
    if returncode not in (0, None):
        detail = _first_line_of_error(stderr)
        return write_result(
            output_dir,
            "failure",
            metrics={
                "collision": bool(artifacts.get("collision")),
                "scenario_duration": artifacts.get("scenario_duration"),
            },
            method_metrics=context,
            reason="laser_se exited with code %s%s"
            % (returncode, ": " + detail if detail else ""),
        )

    collision = bool(artifacts.get("collision"))
    metrics: Dict[str, Any] = {
        "collision": collision,
        # MVP proxy: episode completed under LASER. Stronger intent checks can
        # be added per family later (cf. OSC2 expect_collision).
        "scenario_realized": True,
        "scenario_success": bool(not collision),
    }
    if artifacts.get("time_to_event") is not None:
        metrics["time_to_event"] = artifacts["time_to_event"]
    if artifacts.get("scenario_duration") is not None:
        metrics["scenario_duration"] = artifacts["scenario_duration"]

    context["log_path"] = LASER_LOG_FILE
    return write_result(
        output_dir,
        "success",
        metrics=metrics,
        method_metrics=context,
        trace_path=None,
    )


if __name__ == "__main__":
    sys.exit(main())
