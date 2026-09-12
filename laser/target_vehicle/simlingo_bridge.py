"""SimLingo worker client used by LASER (torch 1.13 venv).

Spawns ``third_party/simlingo/.venv/bin/python`` running
``scenario_orchestration/simlingo_bridge_worker.py`` and talks pickle-framed
messages over a Unix socket (same protocol as tfv6_bridge).
"""

from __future__ import annotations

import os
import pickle
import socket
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _resolve_simlingo_root() -> Path:
    env = os.environ.get("SIMLINGO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    laser_repo = here.parents[2]
    sibling = laser_repo.parent / "simlingo"
    if sibling.is_dir() and (sibling / "scenario_orchestration" / "policy.py").is_file():
        return sibling
    scratch = (
        Path.home() / "scratch" / "scenario_orchestration" / "third_party" / "simlingo"
    )
    if scratch.is_dir() and (scratch / "scenario_orchestration" / "policy.py").is_file():
        return scratch
    raise RuntimeError(
        "Cannot find simlingo checkout. Set SIMLINGO_ROOT "
        f"(tried {sibling} and {scratch})."
    )


def _resolve_simlingo_python(root: Path) -> Path:
    # Do NOT Path.resolve() the interpreter: venv `bin/python` is a symlink to
    # the base CPython, and following it drops the venv's site-packages.
    env = os.environ.get("SIMLINGO_PYTHON")
    if env:
        return Path(os.path.abspath(os.path.expanduser(env)))
    venv_py = (root / ".venv" / "bin" / "python").absolute()
    if venv_py.is_file():
        return venv_py
    raise RuntimeError(
        f"No SimLingo venv python at {venv_py}. "
        "Run third_party/simlingo/scripts/bootstrap_simlingo_venv.sh "
        "or set SIMLINGO_PYTHON."
    )


def _default_checkpoint() -> str:
    return str(
        Path.home()
        / "scratch"
        / "scenario_orchestration"
        / "third_party"
        / "checkpoints"
        / "simlingo"
        / "simlingo"
    )


class SimLingoBridgeClient:
    """Parent-side handle to the SimLingo inference worker."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else _resolve_simlingo_root()
        self.python = _resolve_simlingo_python(self.root)
        self._proc: Optional[subprocess.Popen] = None
        self._conn: Optional[socket.socket] = None
        self._sock_path: Optional[str] = None
        self.info: Dict[str, Any] = {}

    def start(self, request: Dict[str, Any], timeout_s: float = 600.0) -> Dict[str, Any]:
        if self._proc is not None:
            raise RuntimeError("bridge already started")

        fd, sock_path = tempfile.mkstemp(prefix="simlingo_bridge_", suffix=".sock")
        os.close(fd)
        os.unlink(sock_path)
        self._sock_path = sock_path

        worker = self.root / "scenario_orchestration" / "simlingo_bridge_worker.py"
        workdir = os.environ.get("SIMLINGO_WORKDIR") or str(
            Path.home() / "scratch" / "scenario_orchestration_policies" / "simlingo"
        )
        Path(workdir).mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root)
        env["PIP_CONFIG_FILE"] = "/dev/null"
        env["PYTHONNOUSERSITE"] = "1"
        env.setdefault("AV_CKPT", str(Path(self.root).parent / "checkpoints"))
        env.setdefault(
            "CARLA_ROOT",
            os.environ.get("CARLA_ROOT")
            or str(Path.home() / "scratch" / "carla"),
        )
        env.setdefault("HF_HOME", str(Path.home() / "scratch" / "hf_home"))
        env.setdefault("TRANSFORMERS_CACHE", env["HF_HOME"])
        env["SIMLINGO_WORKDIR"] = workdir

        # Symlink pretrained cache into workdir if a shared tree exists.
        av = Path(env["AV_CKPT"])
        if (av / "pretrained").is_dir():
            link = Path(workdir) / "pretrained"
            if not link.exists():
                try:
                    link.symlink_to(av / "pretrained")
                except OSError:
                    pass

        self._proc = subprocess.Popen(
            [
                str(self.python),
                str(worker),
                "--socket",
                sock_path,
                "--workdir",
                workdir,
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        deadline = time.time() + min(30.0, timeout_s)
        ready = False
        assert self._proc.stdout is not None
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line and self._proc.poll() is not None:
                break
            if "SIMLINGO_BRIDGE_READY" in line:
                ready = True
                break
        if not ready:
            out = ""
            try:
                out = self._proc.stdout.read() if self._proc.stdout else ""
            except Exception:
                pass
            self.close()
            raise RuntimeError(f"SimLingo bridge failed to start:\n{out}")

        # Connect
        deadline = time.time() + 10.0
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn.connect(sock_path)
                self._conn = conn
                break
            except OSError as exc:
                last_err = exc
                time.sleep(0.05)
        if self._conn is None:
            self.close()
            raise RuntimeError(f"could not connect to SimLingo bridge: {last_err}")

        ping = self._rpc({"op": "ping"}, timeout_s=30.0)
        load = self._rpc({"op": "load", "request": request}, timeout_s=timeout_s)
        self.info = {**ping, **load}
        print(
            f"SimLingo bridge up: python={ping.get('python')} pid={ping.get('pid')} "
            f"ckpt={load.get('checkpoint')}",
            flush=True,
        )
        return self.info

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._rpc({"op": "act", "observation": observation}, timeout_s=120.0)
        return dict(resp.get("action") or {})

    def reset(self) -> None:
        self._rpc({"op": "reset"}, timeout_s=30.0)

    def close(self) -> None:
        try:
            if self._conn is not None:
                try:
                    self._rpc({"op": "close"}, timeout_s=10.0)
                except Exception:
                    pass
                try:
                    self._conn.close()
                except OSError:
                    pass
        finally:
            self._conn = None
            if self._proc is not None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=15)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                self._proc = None
            if self._sock_path and os.path.exists(self._sock_path):
                try:
                    os.unlink(self._sock_path)
                except OSError:
                    pass
            self._sock_path = None

    def _rpc(self, msg: Dict[str, Any], timeout_s: float = 60.0) -> Dict[str, Any]:
        if self._conn is None:
            raise RuntimeError("bridge not connected")
        raw = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        self._conn.settimeout(timeout_s)
        self._conn.sendall(struct.pack("!Q", len(raw)) + raw)
        header = self._recv_exact(8)
        (size,) = struct.unpack("!Q", header)
        payload = pickle.loads(self._recv_exact(size))
        if not payload.get("ok"):
            err = payload.get("error") or "unknown bridge error"
            tb = payload.get("traceback") or ""
            raise RuntimeError(f"SimLingo bridge: {err}\n{tb}")
        return payload

    def _recv_exact(self, n: int) -> bytes:
        assert self._conn is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("SimLingo bridge socket closed")
            buf.extend(chunk)
        return bytes(buf)
