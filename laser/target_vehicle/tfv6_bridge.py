"""TFv6 3.10 worker client used by LASER (Python 3.8).

Spawns ``third_party/tfv6/.venv/bin/python`` running
``scenario_orchestration/tfv6_bridge_worker.py`` and talks pickle-framed
messages over a Unix socket.
"""

from __future__ import annotations

import os
import pickle
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _resolve_tfv6_root() -> Path:
    env = os.environ.get("TFV6_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    laser_repo = here.parents[2]
    sibling = laser_repo.parent / "tfv6"
    if sibling.is_dir() and (sibling / "scenario_orchestration" / "policy.py").is_file():
        return sibling
    scratch = (
        Path.home() / "scratch" / "scenario_orchestration" / "third_party" / "tfv6"
    )
    if scratch.is_dir() and (scratch / "scenario_orchestration" / "policy.py").is_file():
        return scratch
    raise RuntimeError(
        "Cannot find tfv6 checkout. Set TFV6_ROOT "
        f"(tried {sibling} and {scratch})."
    )


def _resolve_tfv6_python(root: Path) -> Path:
    # Do NOT Path.resolve() the interpreter: venv `bin/python` is a symlink to
    # the base CPython, and following it drops the venv's site-packages.
    env = os.environ.get("TFV6_PYTHON")
    if env:
        return Path(os.path.abspath(os.path.expanduser(env)))
    venv_py = (root / ".venv" / "bin" / "python").absolute()
    if venv_py.is_file():
        return venv_py
    raise RuntimeError(
        f"No TFv6 venv python at {venv_py}. "
        "Run scripts/bootstrap_tfv6_venv.sh or set TFV6_PYTHON."
    )


class TFv6BridgeClient:
    """Parent-side handle to the 3.10 inference worker."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else _resolve_tfv6_root()
        self.python = _resolve_tfv6_python(self.root)
        self._proc: Optional[subprocess.Popen] = None
        self._conn: Optional[socket.socket] = None
        self._sock_path: Optional[str] = None
        self.info: Dict[str, Any] = {}

    def start(self, request: Dict[str, Any], timeout_s: float = 180.0) -> Dict[str, Any]:
        if self._proc is not None:
            raise RuntimeError("bridge already started")

        fd, sock_path = tempfile.mkstemp(prefix="tfv6_bridge_", suffix=".sock")
        os.close(fd)
        os.unlink(sock_path)
        self._sock_path = sock_path

        # Prefer LASER-vendored worker + policy overlay so stock tfv6 pins work
        # without committing into INPUTrrr0/TFv6.
        laser_repo = Path(__file__).resolve().parents[2]
        overlay = laser_repo / "scenario_orchestration" / "overlays" / "tfv6"
        overlay_worker = overlay / "scenario_orchestration" / "tfv6_bridge_worker.py"
        stock_worker = self.root / "scenario_orchestration" / "tfv6_bridge_worker.py"
        if overlay_worker.is_file():
            worker = overlay_worker
            # Overlay first so scenario_orchestration.policy is the fixed copy.
            pythonpath = os.pathsep.join([str(overlay), str(self.root)])
            print(f"TFv6 bridge overlay: {overlay}")
        elif stock_worker.is_file():
            worker = stock_worker
            pythonpath = str(self.root)
        else:
            raise RuntimeError(
                "tfv6_bridge_worker.py not found. Expected LASER overlay at "
                f"{overlay_worker} or stock worker at {stock_worker}. "
                "Run scripts/bootstrap_tfv6_venv.sh after TFV6_ROOT is set."
            )

        env = os.environ.copy()
        # Keep the worker on the 3.10 tree; drop LASER 3.8 PYTHONPATH noise.
        env["PYTHONPATH"] = pythonpath
        env["PIP_CONFIG_FILE"] = "/dev/null"
        # Always set (do not setdefault): frozen-red LASER scenes need creeping.
        env["LEAD_CLOSED_LOOP_CONFIG"] = os.environ.get(
            "LEAD_CLOSED_LOOP_CONFIG",
            "sensor_agent_creeping=True sensor_agent_stuck_threshold=40 "
            "sensor_agent_stuck_move_duration=40 sensor_agent_stuck_throttle=0.55 "
            "use_kalman_filter=True slower_for_stop_sign=True",
        )
        env.setdefault(
            "TFV6_CHECKPOINT",
            str(
                Path.home()
                / "scratch"
                / "scenario_orchestration"
                / "third_party"
                / "checkpoints"
                / "tfv6_cvpr2026"
                / "tfv6_resnet34"
            ),
        )

        self._proc = subprocess.Popen(
            [str(self.python), str(worker), "--socket", sock_path],
            cwd=str(self.root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        deadline = time.time() + timeout_s
        ready = False
        lines = []
        assert self._proc.stdout is not None
        while time.time() < deadline:
            if self._proc.poll() is not None:
                rest = self._proc.stdout.read() or ""
                raise RuntimeError(
                    "TFv6 worker exited early:\n" + "\n".join(lines + [rest])
                )
            line = self._proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            line = line.rstrip()
            lines.append(line)
            print(f"[tfv6-worker] {line}")
            if line.startswith("TFV6_BRIDGE_READY"):
                ready = True
                break
        if not ready:
            self.close()
            raise TimeoutError(
                "TFv6 worker did not become ready:\n" + "\n".join(lines[-40:])
            )

        self._conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._conn.connect(sock_path)
        self._conn.settimeout(timeout_s)

        ping = self._rpc({"op": "ping"})
        if not ping.get("ok"):
            raise RuntimeError(f"TFv6 ping failed: {ping}")
        print(
            f"TFv6 bridge up: python={ping.get('python')} pid={ping.get('pid')} "
            f"root={self.root}"
        )

        loaded = self._rpc({"op": "load", "request": request}, timeout_s=timeout_s)
        if not loaded.get("ok"):
            raise RuntimeError(
                f"TFv6 load failed: {loaded.get('error')}\n{loaded.get('traceback', '')}"
            )
        self.info = loaded
        print(
            f"TFv6 loaded checkpoint={loaded.get('checkpoint')} "
            f"radars={loaded.get('use_radars')} "
            f"img={loaded.get('final_image_height')}x{loaded.get('final_image_width')}"
        )
        return loaded

    def act(self, observation: Dict[str, Any], timeout_s: float = 60.0) -> Dict[str, Any]:
        resp = self._rpc({"op": "act", "observation": observation}, timeout_s=timeout_s)
        if not resp.get("ok"):
            raise RuntimeError(
                f"TFv6 act failed: {resp.get('error')}\n{resp.get('traceback', '')}"
            )
        return resp["action"]

    def reset(self) -> None:
        resp = self._rpc({"op": "reset"})
        if not resp.get("ok"):
            raise RuntimeError(f"TFv6 reset failed: {resp}")

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._rpc({"op": "close"}, timeout_s=5.0)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._proc = None
        if self._sock_path and os.path.exists(self._sock_path):
            try:
                os.unlink(self._sock_path)
            except OSError:
                pass
            self._sock_path = None

    # -- framing -----------------------------------------------------------

    def _rpc(self, payload: Dict[str, Any], timeout_s: float = 60.0) -> Dict[str, Any]:
        if self._conn is None:
            raise RuntimeError("bridge not connected")
        self._conn.settimeout(timeout_s)
        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self._conn.sendall(struct.pack("!Q", len(raw)) + raw)
        header = self._recv_exact(8)
        (size,) = struct.unpack("!Q", header)
        body = self._recv_exact(size)
        return pickle.loads(body)

    def _recv_exact(self, n: int) -> bytes:
        assert self._conn is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("TFv6 worker socket closed")
            buf.extend(chunk)
        return bytes(buf)
