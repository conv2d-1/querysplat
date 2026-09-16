"""V3 ablation: keep dense pair-decoder flow (no motion propagation)."""
from __future__ import annotations

import copy
import importlib.util
import os

_BASE_REL = (
    "hAlgorithm/configs/motion_head/baseline/"
    "wfm_rgb_query_dual_4dgs_perpixel_trainquality_v3_envision.py"
)
_BASE_PATH = os.path.join(os.getcwd(), _BASE_REL)
if not os.path.isfile(_BASE_PATH):
    raise FileNotFoundError(f"Base config not found: {_BASE_PATH}")
_SPEC = importlib.util.spec_from_file_location("v3_envision_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BASE)

_SKIP = frozenset({"model", "output_dir", "ckpt_dir", "tb_dir", "eval_dir", "vis_dir"})
for _name in dir(_BASE):
    if _name.startswith("_") or _name in _SKIP:
        continue
    globals()[_name] = getattr(_BASE, _name)

model = copy.deepcopy(_BASE.model)
model["model"] = copy.deepcopy(model["model"])
model["model"]["gs_flow_source"] = "dense"

output_dir = (
    "/mnt/home/tcchen/workspace/TMA-origin-dev/results/sparse_pair_4dgs_v2/"
    "wfm_rgb_query_dual_4dgs_perpixel_trainquality_v3_no_motion_single_source"
)
ckpt_dir = f"{output_dir}/checkpoint"
tb_dir = f"{output_dir}/tensorboard"
eval_dir = f"{output_dir}/evaluation"
vis_dir = f"{output_dir}/visualization"

trainer["output_dir"] = output_dir
trainer["ckpt_dir"] = ckpt_dir
trainer["tb_dir"] = tb_dir
trainer["eval_dir"] = eval_dir
trainer["vis_dir"] = vis_dir
