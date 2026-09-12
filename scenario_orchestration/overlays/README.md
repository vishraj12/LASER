# Policy overlays

Stock `third_party/plant2` and `third_party/tfv6` submodule pins do not always
include the small harness patches LASER needs (and we may lack write access to
those upstreams). These overlays ship with LASER so closed-loop cells stay
runnable from a clean mega-repo checkout.

| Overlay | Used by | Provides |
| --- | --- | --- |
| `plant2/` | `plant2_ego.py` | `policy.py` torch/device/index fixes |
| `tfv6/` | `tfv6_bridge.py` | `tfv6_bridge_worker.py` + closed-loop `policy.py` |

`simlingo` is intentionally not covered here yet.
