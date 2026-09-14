#!/usr/bin/env python3
"""Host one harness ego policy in its own interpreter for LASER.

    python policy_worker.py --socket S --entry <repo>/scenario_orchestration/policy.py
                            [--repository <repo>]

`laser/target_vehicle/harness_policy_ego.py :: RemotePolicy` starts this when a
policy's inference stack cannot share LASER's interpreter (TFv6's torch/lead,
SimLingo's Python 3.8). It imports the entry point exactly as the highway port's
`policies.load_policy` does and serves it over length-prefixed pickle frames:

  build     {"request": policy.json}  -> {"sensors": policy.sensors() or []}
  load / reset / close / metadata
  act       {"observation": ...}      -> {"action": policy.act(observation)}

Every reply is {"ok": True, ...} or {"ok": False, "error", "traceback"}; a frame
that cannot be decoded is answered with its error rather than ending the loop.
Kept to Python 3.8.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import socket
import struct
import sys
import traceback

READY = "POLICY_WORKER_READY"


def _recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def _send(conn, payload):
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("!Q", len(raw)) + raw)


def _error(exc):
    return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc()}


def _build(entry, repository, request):
    for path in (os.path.dirname(entry), repository):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("_policy_%s" % request.get("name"), entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for attr in ("build_policy", "make_policy", "load_policy", "Policy"):
        factory = getattr(module, attr, None)
        if callable(factory):
            return factory(dict(request))
    raise RuntimeError("%s exposes no build_policy() factory" % entry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--entry", required=True)
    ap.add_argument("--repository", default=None)
    args = ap.parse_args()

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    os.chmod(args.socket, 0o600)
    server.listen(1)
    print("%s %s" % (READY, args.socket), flush=True)
    conn, _ = server.accept()
    policy = None
    try:
        while True:
            try:
                (size,) = struct.unpack("!Q", _recv_exact(conn, 8))
                body = _recv_exact(conn, size)
            except ConnectionError:
                break
            try:
                message = pickle.loads(body)
            except Exception as exc:  # noqa: BLE001
                _send(conn, _error(exc))
                continue
            op = message.get("op")
            try:
                if op == "build":
                    policy = _build(args.entry, args.repository, message.get("request") or {})
                    sensors = getattr(policy, "sensors", None)
                    _send(conn, {"ok": True, "python": sys.version.split()[0],
                                 "sensors": list(sensors()) if callable(sensors) else []})
                elif op == "load":
                    loader = getattr(policy, "load", None)
                    if callable(loader):
                        policy = loader() or policy
                    _send(conn, {"ok": True})
                elif op == "reset":
                    resetter = getattr(policy, "reset", None)
                    if callable(resetter):
                        resetter()
                    _send(conn, {"ok": True})
                elif op == "act":
                    _send(conn, {"ok": True, "action": policy.act(message.get("observation") or {})})
                elif op == "metadata":
                    described = getattr(policy, "metadata", None)
                    _send(conn, {"ok": True, "metadata": described() if callable(described) else {}})
                elif op == "close":
                    closer = getattr(policy, "close", None)
                    if callable(closer):
                        closer()
                    break
                else:
                    _send(conn, {"ok": False, "error": "unknown op %r" % op})
            except Exception as exc:  # noqa: BLE001
                _send(conn, _error(exc))
    finally:
        conn.close()
        server.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
