"""Sparse 2D flow evaluation metrics in normalized UV space.

Expects ``outputs`` to carry:
    - pred_flow_2d:             [Q, 2] predicted (du, dv) in normalized UV
    - motion_queries_uv:        [Q, 2] source UV in normalized coordinates
    - motion_queries_gt_2d_tgt: [Q, 2] GT target UV in normalized coordinates
    - track_vis_pred:           [Q]    validity mask

Optional fields used for dynamic-only evaluation:
    - motion_queries_gt_3d_src: [Q, 3] GT source 3D positions
    - track_gt:                 [Q, 3] GT target 3D positions

Metrics:
    - epe:   mean 2D endpoint error in normalized UV
    - apd:   average accuracy over ``acc2d_*`` thresholds
    - tau:   inlier ratio at ``inlier_threshold``
    - acc2d_*: per-threshold 2D accuracy
"""

import logging
from typing import Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _compute_flow_metrics(
    errors: np.ndarray,
    inlier_threshold: float,
    apd_thresholds: List[float],
    prefix: str = "",
) -> Dict[str, float]:
    """Compute EPE / APD / tau / acc2d from a 1-D 2D-flow error array."""
    if len(errors) == 0:
        nan = float("nan")
        result = {f"{prefix}epe": nan, f"{prefix}apd": nan, f"{prefix}tau": nan}
        for t in apd_thresholds:
            result[f"{prefix}acc2d_{t}"] = nan
        return result

    epe = float(np.mean(errors))
    tau = float(np.mean(errors < inlier_threshold))

    accs, acc_dict = [], {}
    for t in apd_thresholds:
        a = float(np.mean(errors < t))
        accs.append(a)
        acc_dict[f"{prefix}acc2d_{t}"] = a

    return {
        f"{prefix}epe": epe,
        f"{prefix}apd": float(np.mean(accs)),
        f"{prefix}tau": tau,
        **acc_dict,
    }


class SparseFlow2DEvalMetrics:
    """Query-based 2D flow evaluation in normalized UV space."""

    def __init__(
        self,
        dynamic_threshold: float = 0.01,
        apd_thresholds: List[float] = [0.005, 0.01, 0.02, 0.05, 0.1],
        inlier_threshold: float = 0.02,
        dynamic_only: bool = False,
        metric_prefix: str = "",
        **kwargs,
    ):
        self.dynamic_threshold = dynamic_threshold
        self.apd_thresholds = apd_thresholds
        self.inlier_threshold = inlier_threshold
        self.dynamic_only = dynamic_only
        self.metric_prefix = metric_prefix

        self.metrics = [
            f"{metric_prefix}{m}"
            for m in ["epe", "apd", "tau"] + [f"acc2d_{t}" for t in apd_thresholds]
        ]

    def __call__(self, inputs, outputs) -> Dict[str, torch.Tensor]:
        if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
            return {}

        out = outputs[0]

        pred_flow = getattr(out, "pred_flow_2d", None)
        src_uv = getattr(out, "motion_queries_uv", None)
        gt_tgt_uv = getattr(out, "motion_queries_gt_2d_tgt", None)
        vis = getattr(out, "track_vis_pred", None)
        gt_src = getattr(out, "motion_queries_gt_3d_src", None)
        gt_tgt = getattr(out, "track_gt", None)

        if pred_flow is None or src_uv is None or gt_tgt_uv is None:
            return {}

        if isinstance(pred_flow, torch.Tensor):
            pred_flow = pred_flow.detach().cpu().numpy()
        if isinstance(src_uv, torch.Tensor):
            src_uv = src_uv.detach().cpu().numpy()
        if isinstance(gt_tgt_uv, torch.Tensor):
            gt_tgt_uv = gt_tgt_uv.detach().cpu().numpy()
        if vis is not None and isinstance(vis, torch.Tensor):
            vis = vis.detach().cpu().numpy()
        if gt_src is not None and isinstance(gt_src, torch.Tensor):
            gt_src = gt_src.detach().cpu().numpy()
        if gt_tgt is not None and isinstance(gt_tgt, torch.Tensor):
            gt_tgt = gt_tgt.detach().cpu().numpy()

        n = min(pred_flow.shape[0], src_uv.shape[0], gt_tgt_uv.shape[0])
        if n == 0:
            return {}

        pred_flow = pred_flow[:n]
        src_uv = src_uv[:n]
        gt_tgt_uv = gt_tgt_uv[:n]
        if vis is not None:
            vis = vis[:n]
        if gt_src is not None:
            gt_src = gt_src[:n]
        if gt_tgt is not None:
            gt_tgt = gt_tgt[:n]

        valid = np.ones(n, dtype=bool)
        if vis is not None:
            valid = valid & (vis > 0.5)

        is_dynamic = None
        if gt_src is not None and gt_tgt is not None:
            gt_disp = gt_tgt - gt_src
            gt_motion_mag = np.linalg.norm(gt_disp, axis=-1)
            is_dynamic = gt_motion_mag > self.dynamic_threshold

            if self.dynamic_only and valid.sum() > 0:
                n_dyn = (is_dynamic & valid).sum()
                if n_dyn == 0:
                    valid_mag = gt_motion_mag[valid]
                    logger.warning(
                        "[%sSparseFlow2DEvalMetrics] dynamic_only=True but 0/%d valid "
                        "points exceed threshold %.4f. "
                        "3D displacement stats: max=%.6f, mean=%.6f, p95=%.6f.",
                        self.metric_prefix,
                        int(valid.sum()),
                        self.dynamic_threshold,
                        float(valid_mag.max()) if len(valid_mag) > 0 else 0.0,
                        float(valid_mag.mean()) if len(valid_mag) > 0 else 0.0,
                        float(np.percentile(valid_mag, 95)) if len(valid_mag) > 0 else 0.0,
                    )
        elif self.dynamic_only:
            logger.warning(
                "[%sSparseFlow2DEvalMetrics] dynamic_only=True but 3D GT positions are missing. "
                "Returning nan.",
                self.metric_prefix,
            )

        if self.dynamic_only and is_dynamic is not None:
            valid = valid & is_dynamic

        pfx = self.metric_prefix
        if valid.sum() == 0:
            nan = float("nan")
            result = {f"{pfx}epe": nan, f"{pfx}apd": nan, f"{pfx}tau": nan}
            for t in self.apd_thresholds:
                result[f"{pfx}acc2d_{t}"] = nan
            return result

        gt_flow = gt_tgt_uv - src_uv
        errors = np.linalg.norm(pred_flow[valid] - gt_flow[valid], axis=-1)

        result = _compute_flow_metrics(
            errors,
            self.inlier_threshold,
            self.apd_thresholds,
            prefix=pfx,
        )

        if is_dynamic is not None:
            valid_base = np.ones(n, dtype=bool)
            if vis is not None:
                valid_base = valid_base & (vis > 0.5)
            n_valid = valid_base.sum()
            if n_valid > 0:
                result[f"{pfx}dyn_ratio"] = float(is_dynamic[valid_base].sum() / n_valid)

        return result
