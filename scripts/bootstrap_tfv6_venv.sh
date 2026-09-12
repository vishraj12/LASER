#!/usr/bin/env bash
# Bootstrap TFV6_ROOT/.venv on Python 3.10 (Compute Canada).
# This script lives in the LASER repo; the venv is created under third_party/tfv6.
# Skips carla (no cp310 wheel on PyPI) and open3d>=0.19 (needs >=3.11 on PyPI).
# Installs a minimal carla stub so checkpoint load() works offline.
set -euo pipefail

LASER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${TFV6_ROOT:-}" ]]; then
  ROOT="$(cd "$TFV6_ROOT" && pwd)"
elif [[ -d "$LASER_ROOT/../tfv6" ]]; then
  ROOT="$(cd "$LASER_ROOT/../tfv6" && pwd)"
else
  echo "bootstrap_tfv6_venv: set TFV6_ROOT to the tfv6 checkout" >&2
  exit 1
fi
cd "$ROOT"
echo "bootstrap_tfv6_venv: target $ROOT"

if command -v module >/dev/null 2>&1; then
  module load StdEnv/2020 python/3.10.2 2>/dev/null || module load python/3.10.2 || true
fi

PY="$(command -v python3)"
VER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "$VER" != "3.10" ]]; then
  echo "bootstrap_tfv6_venv: need Python 3.10, got $VER ($PY)" >&2
  exit 1
fi

rm -rf .venv
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
export PIP_CONFIG_FILE=/dev/null
unset PYTHONPATH || true

python -m pip install -U 'pip>=24' 'setuptools>=70' wheel

REQ="$(mktemp)"
python - "$REQ" <<'PY'
import sys
from pathlib import Path
req_path = Path(sys.argv[1])
text = Path("pyproject.toml").read_text()
start = text.index("dependencies = [")
end = text.index("]", start)
deps = []
for line in text[start:end].splitlines():
    line = line.strip().rstrip(",")
    if line.startswith('"') and line.endswith('"'):
        deps.append(line.strip('"'))
skip_prefixes = ("carla", "open3d", "pyqt5")
out = [d for d in deps if not any(d.lower().startswith(p) for p in skip_prefixes)]
req_path.write_text("\n".join(out) + "\n")
print("deps:", len(out))
PY

python -m pip install \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  -r "$REQ"
rm -f "$REQ"

python -m pip install --no-deps .

# Agents package for RoadOption etc.
mkdir -p 3rd_party/CARLA_0915/PythonAPI
if [[ -d "${CARLA_ROOT:-$HOME/scratch/carla}/PythonAPI/carla" ]]; then
  ln -sfn "${CARLA_ROOT:-$HOME/scratch/carla}/PythonAPI/carla" \
    3rd_party/CARLA_0915/PythonAPI/carla
fi

STUB=".venv/lib/python3.10/site-packages/carla"
mkdir -p "$STUB"
cat > "$STUB/__init__.py" <<'EOF'
"""Minimal stub: no cp310 CARLA wheel on this host. Enough for ExpertConfig import / load()."""
class Color:
    def __init__(self, r, g, b, a=255):
        self.r, self.g, self.b, self.a = int(r), int(g), int(b), int(a)
class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)
class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)
class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()
class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)
class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        self.throttle, self.steer, self.brake = throttle, steer, brake
        self.hand_brake, self.reverse = hand_brake, reverse
EOF

echo "bootstrap_tfv6_venv: done -> $ROOT/.venv"
echo "Worker/policy overlay lives in LASER: scenario_orchestration/overlays/tfv6/"
