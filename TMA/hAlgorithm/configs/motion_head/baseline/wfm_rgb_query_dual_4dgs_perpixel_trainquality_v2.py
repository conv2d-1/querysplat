"""Continue from trainquality with corrected Gaussian head design.

Extends :mod:`wfm_rgb_query_dual_4dgs_perpixel_trainquality` and loads the
first trainquality checkpoint (trained with the deprecated shared ``delta_mlp``).

Head corrections applied in this run:

  * ``use_rgb_color_anchor=True`` — ref-frame RGB initializes SH DC; head learns residual.
  * ``predict_attribute_delta=False`` — remove misleading shared per-frame delta branch.

Inherited trainquality settings (unchanged):

  * ``num_render_frames=3``
  * ``motion_gs_displacement_consistency_loss``
  * ``dgs_scale_dropout_prob=0.25``
  * ``task_weight.dgs=2.0``

Checkpoint loading uses ``strict=False`` (pipeline default). Weights from the removed
``delta_mlp`` are ignored; ``attr_mlp`` and the rest of the motion / encoder stack
are warm-started from trainquality.

Init checkpoint (``trainer.load_from``):
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

# Shorter second-stage finetune; increase if metrics still improving.
max_iter = 15000

model = copy.deepcopy(_BASE.model)
model["model"] = copy.deepcopy(model["model"])
model["model"]["sparse_gaussian_head"] = copy.deepcopy(model["model"]["sparse_gaussian_head"])
model["model"]["sparse_gaussian_head"]["use_rgb_color_anchor"] = True
model["model"]["sparse_gaussian_head"]["predict_attribute_delta"] = False

trainer = copy.deepcopy(_BASE.trainer)
trainer["load_from"] = _TRAINQUALITY_CKPT
trainer["max_iter"] = max_iter
trainer["lr_scheduler"] = copy.deepcopy(trainer["lr_scheduler"])
trainer["lr_scheduler"]["max_iters"] = max_iter
trainer["lr_scheduler"]["warmup_iters"] = 300

# Slightly lower head LR while adapting to RGB color anchor.
trainer["optimizer"] = copy.deepcopy(trainer["optimizer"])
trainer["optimizer"]["model.sparse_gaussian_head"] = dict(
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-3,
    eps=1e-10,
)

output_dir = (
    "/mnt/home/tcchen/workspace/TMA-origin-dev/results/sparse_pair_4dgs_v2/"
    "wfm_rgb_query_dual_4dgs_perpixel_trainquality_v2"
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
