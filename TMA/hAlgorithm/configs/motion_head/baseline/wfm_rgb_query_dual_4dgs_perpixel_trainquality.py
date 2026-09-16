"""Train-quality finetune on top of per-pixel 4DGS baseline.

Extends :mod:`wfm_rgb_query_dual_4dgs_perpixel_finetune` with:

  * ``use_rgb_color_anchor=True`` — SH DC from ref-frame RGB; head predicts residual only.
  * ``num_render_frames=3`` — render-supervise all non-ref frames in 4-view clips.
  * ``motion_gs_displacement_consistency_loss`` — align dense GS flow with sparse motion at traj UVs.
  * ``dgs_scale_dropout_prob=0.25`` — mix metric/normalized render for infer alignment.
  * ``task_weight.dgs=2.0`` — stronger render supervision.

Init checkpoint (``trainer.load_from``): latest per-pixel 4DGS finetune ckpt.
"""
from __future__ import annotations

import copy
import importlib.util
import os

_BASE_REL = (
    "hAlgorithm/configs/motion_head/baseline/"
    "wfm_rgb_query_dual_4dgs_perpixel_finetune.py"
)
_BASE_PATH = os.path.join(os.getcwd(), _BASE_REL)
if not os.path.isfile(_BASE_PATH):
    raise FileNotFoundError(
        f"Base config not found: {_BASE_PATH}. "
        "Run training from the TMA repo root (cwd must contain hAlgorithm/).",
    )
_SPEC = importlib.util.spec_from_file_location("perpixel_finetune_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BASE)

_SKIP = frozenset({
    "model",
    "trainer",
    "output_dir",
    "ckpt_dir",
    "tb_dir",
    "eval_dir",
    "vis_dir",
})
for _name in dir(_BASE):
    if _name.startswith("_") or _name in _SKIP:
        continue
    globals()[_name] = getattr(_BASE, _name)

_PERPIXEL_FINETUNE_CKPT = (
    "/mnt/home/tcchen/workspace/TMA-origin-dev/results/sparse_pair_4dgs_v2/"
    "wfm_rgb_query_dual_4dgs_perpixel_finetune_20260611-115145/checkpoint/latest/ckpt.pth"
)

model = copy.deepcopy(_BASE.model)
model["model"] = copy.deepcopy(model["model"])
model["model"]["sparse_gaussian_head"] = copy.deepcopy(model["model"]["sparse_gaussian_head"])
model["model"]["sparse_gaussian_head"]["use_rgb_color_anchor"] = True
model["model"]["sparse_gaussian_head"]["predict_attribute_delta"] = False

model["motion_gs_displacement_consistency_loss"] = dict(
    type=(
        "hAlgorithm.modules.losses2.sparse_motion_gs_consistency_loss."
        "SparseMotionGsDisplacementConsistencyLoss"
    ),
    loss_weight=dict(default=0.25),
    dynamic_threshold=0.02,
    huber_delta=0.05,
)
model["dgs_scale_dropout_prob"] = 0.25

model["sparse_dynamic_gaussian_render_loss"] = copy.deepcopy(
    model["sparse_dynamic_gaussian_render_loss"],
)
model["sparse_dynamic_gaussian_render_loss"]["num_render_frames"] = 3

model["task_weight"] = copy.deepcopy(model["task_weight"])
model["task_weight"]["dgs"] = 2.0

trainer = copy.deepcopy(_BASE.trainer)
trainer["load_from"] = _PERPIXEL_FINETUNE_CKPT

output_dir = (
    "/mnt/home/tcchen/workspace/TMA-origin-dev/results/sparse_pair_4dgs_v2/"
    "wfm_rgb_query_dual_4dgs_perpixel_trainquality"
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
