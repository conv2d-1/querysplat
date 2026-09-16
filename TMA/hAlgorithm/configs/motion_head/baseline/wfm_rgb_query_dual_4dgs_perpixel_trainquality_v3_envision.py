"""V3 Envision4D-inspired dual-query 4DGS (new classes, old paths unchanged).

Loads ``trainquality_20260612`` checkpoint and enables:

  * :class:`SparsePairDynamicGaussianHeadV2` + :class:`SparsePairGaussianAdapter`
  * Motion-single-source dense flow (``gs_flow_source="motion"``)
  * Target rendered depth loss + floater suppression
  * Stronger motion–GS consistency (0.5) with debug stats

Init checkpoint:
  ``.../wfm_rgb_query_dual_4dgs_perpixel_trainquality_20260612-165820/checkpoint/latest/ckpt.pth``
"""
from __future__ import annotations

import copy
import importlib.util
import os

_BASE_REL = (
    "hAlgorithm/configs/motion_head/baseline/"
    "wfm_rgb_query_dual_4dgs_perpixel_trainquality.py"
)
_BASE_PATH = os.path.join(os.getcwd(), _BASE_REL)
if not os.path.isfile(_BASE_PATH):
    raise FileNotFoundError(
        f"Base config not found: {_BASE_PATH}. "
        "Run training from the TMA repo root (cwd must contain hAlgorithm/).",
    )
_SPEC = importlib.util.spec_from_file_location("trainquality_base", _BASE_PATH)
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
    "max_iter",
})
for _name in dir(_BASE):
    if _name.startswith("_") or _name in _SKIP:
        continue
    globals()[_name] = getattr(_BASE, _name)

_TRAINQUALITY_CKPT = (
    "/mnt/home/tcchen/workspace/TMA-origin-dev/results/sparse_pair_4dgs_v2/"
    "wfm_rgb_query_dual_4dgs_perpixel_trainquality_20260612-165820/"
    "checkpoint/latest/ckpt.pth"
)

max_iter = 20000

model = copy.deepcopy(_BASE.model)
model["type"] = (
    "hAlgorithm.modules.pipelines2.mvfr_query_dual_4dgs_v2.WFMQueryDual4DGSPipelineV2"
)
model["model"] = copy.deepcopy(model["model"])
model["model"]["type"] = (
    "hAlgorithm.modules.models2.sdk.query_dual_4dgs_v2.MVQueryDual4DGSV2"
)
model["model"]["gs_flow_source"] = "motion"
model["model"]["gaussian_adapter"] = dict(
    type=(
        "hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_adapter."
        "SparsePairGaussianAdapter"
    ),
    sh_degree=0,
    depth_condition_scale=True,
    use_rgb_color_anchor=True,
)
model["model"]["sparse_gaussian_head"] = dict(
    type=(
        "hAlgorithm.modules.models2.head.sparse_pair_gaussian_head_v2."
        "SparsePairDynamicGaussianHeadV2"
    ),
    in_dim=256,
    hidden_dim=64,
    sh_degree=0,
    motion_feat_dim=32,
    enhanced_motion=True,
    random_src_frame=True,
    predict_attribute_delta=False,
    use_rgb_color_anchor=True,
    fuse_depth_in_head=True,
    fuse_edge_in_head=True,
    predict_raw_only=True,
)

model["motion_gs_displacement_consistency_loss"] = dict(
    type=(
        "hAlgorithm.modules.losses2.sparse_motion_gs_consistency_loss."
        "SparseMotionGsDisplacementConsistencyLoss"
    ),
    loss_weight=dict(default=0.5),
    dynamic_threshold=0.02,
    huber_delta=0.05,
)

model["sparse_dynamic_gaussian_render_loss"] = dict(
    type=(
        "hAlgorithm.modules.losses2.sparse_pair_gaussian_loss_v2."
        "SparsePairDynamicGaussianRenderLossV2"
    ),
    l1_weight=0.5,
    ssim_weight=0.2,
    num_render_frames=3,
    max_render_points=0,
    background_color=[0.0, 0.0, 0.0],
    target_depth_weight=0.1,
    floater_weight=0.01,
    floater_depth_margin=0.05,
)

trainer = copy.deepcopy(_BASE.trainer)
trainer["load_from"] = _TRAINQUALITY_CKPT
trainer["max_iter"] = max_iter
trainer["lr_scheduler"] = copy.deepcopy(trainer["lr_scheduler"])
trainer["lr_scheduler"]["max_iters"] = max_iter
trainer["lr_scheduler"]["warmup_iters"] = 500

trainer["optimizer"] = copy.deepcopy(trainer["optimizer"])
trainer["optimizer"]["model.sparse_gaussian_head"] = dict(
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-3,
    eps=1e-10,
)
trainer["optimizer"]["model.gaussian_adapter"] = dict(
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-3,
    eps=1e-10,
)

output_dir = (
    "/mnt/home/tcchen/workspace/TMA-origin-dev/results/sparse_pair_4dgs_v2/"
    "wfm_rgb_query_dual_4dgs_perpixel_trainquality_v3_envision"
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
