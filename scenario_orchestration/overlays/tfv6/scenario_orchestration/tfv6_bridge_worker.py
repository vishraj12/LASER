#!/usr/bin/env python3
"""Long-lived TFv6 inference worker (Python 3.10).

LASER stays on 3.8 + real CARLA bindings; this process owns torch + lead.
Protocol: length-prefixed pickle frames over a Unix domain socket.

  request: {"op": "ping"|"load"|"act"|"reset"|"close"|"sensors", ...}
  response: {"ok": True, ...} or {"ok": False, "error": "..."}
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import sys
import traceback
from typing import Any, Dict, Optional


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(conn: socket.socket) -> Dict[str, Any]:
    header = _recv_exact(conn, 8)
    (size,) = struct.unpack("!Q", header)
    if size > 512 * 1024 * 1024:
        raise ValueError(f"message too large: {size}")
    return pickle.loads(_recv_exact(conn, size))


def _send_msg(conn: socket.socket, payload: Dict[str, Any]) -> None:
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("!Q", len(raw)) + raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True, help="Unix socket path")
    args = parser.parse_args()

    sock_path = args.socket
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    os.chmod(sock_path, 0o600)
    server.listen(1)
    # Parent waits on this line.
    print(f"TFV6_BRIDGE_READY {sock_path}", flush=True)

    conn, _addr = server.accept()
    policy = None
    try:
        while True:
            try:
                msg = _recv_msg(conn)
            except ConnectionError:
                break
            op = msg.get("op")
            try:
                if op == "ping":
                    _send_msg(conn, {"ok": True, "pid": os.getpid(),
                                     "python": sys.version.split()[0]})
                elif op == "load":
                    from scenario_orchestration.policy import build_policy

                    request = dict(msg.get("request") or {})
                    policy = build_policy(request)
                    policy.load()
                    policy.reset()
                    _send_msg(conn, {
                        "ok": True,
                        "checkpoint": policy.checkpoint,
                        "sensors": policy.sensors(),
                        "use_discrete_command": bool(
                            getattr(policy.training_config,
                                    "use_discrete_command", False)),
                        "use_radars": bool(
                            getattr(policy.training_config, "use_radars", False)),
                        "final_image_height": int(
                            policy.training_config.final_image_height),
                        "final_image_width": int(
                            policy.training_config.final_image_width),
                    })
                elif op == "sensors":
                    if policy is None:
                        raise RuntimeError("load first")
                    _send_msg(conn, {"ok": True, "sensors": policy.sensors()})
                elif op == "act":
                    if policy is None:
                        raise RuntimeError("load first")
                    action = policy.act(msg.get("observation") or {})
                    _send_msg(conn, {"ok": True, "action": action})
                elif op == "reset":
                    if policy is not None:
                        policy.reset()
                    _send_msg(conn, {"ok": True})
                elif op == "close":
                    if policy is not None:
                        policy.close()
                    _send_msg(conn, {"ok": True})
                    break
                else:
                    _send_msg(conn, {"ok": False, "error": f"unknown op {op!r}"})
            except Exception as exc:  # noqa: BLE001
                _send_msg(conn, {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            server.close()
        except Exception:  # noqa: BLE001
            pass
        if os.path.exists(sock_path):
            try:
                os.unlink(sock_path)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
