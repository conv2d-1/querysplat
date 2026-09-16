"""Sparse Motion Evaluation Metrics

Evaluates query-based 3D motion predictions against GT trajectory positions.
Works with ReconstructOutput objects that carry:
    - track_pred:              [Q, 3] predicted 3D positions (src-camera frame, metric scale)
    - track_gt:                [Q, 3] ground-truth 3D positions (same frame & scale)
    - track_vis_pred:          [Q]    validity mask (1 = valid, 0 = ignore)
    - motion_queries_gt_3d_src:[Q, 3] GT source 3D positions (for dynamic classification)

Metrics:
    - epe:   End-Point Error (mean 3D L2 distance over valid queries)
    - apd:   Average Points within Delta (mean acc over thresholds)
    - tau:   Inlier ratio at ``inlier_threshold``
    - acc3d_*: Per-threshold accuracy
"""

import logging
from typing import Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _compute_metrics(
    errors: np.ndarray,
    inlier_threshold: float,
    apd_thresholds: List[float],
    prefix: str = "",
) -> Dict[str, float]:
    """Compute EPE / APD / tau / acc3d from a 1-D error array."""
    if len(errors) == 0:
        nan = float("nan")
        result = {f"{prefix}epe": nan, f"{prefix}apd": nan, f"{prefix}tau": nan}
        for t in apd_thresholds:
            result[f"{prefix}acc3d_{t}"] = nan
        return result

    epe = float(np.mean(errors))
    tau = float(np.mean(errors < inlier_threshold))

    accs, acc_dict = [], {}
    for t in apd_thresholds:
        a = float(np.mean(errors < t))
        accs.append(a)
        acc_dict[f"{prefix}acc3d_{t}"] = a

    return {
        f"{prefix}epe": epe,
        f"{prefix}apd": float(np.mean(accs)),
        f"{prefix}tau": tau,
        **acc_dict,
    }


class SparseMotionEvalMetrics:
    """Query-based 3D motion evaluation aligned with D4RT / Any4D conventions.

    Expects ``outputs`` to be a list of ``ReconstructOutput``.
    Only the **first** element is used (track data is shared across views).
    """

    def __init__(
        self,
        dynamic_threshold: float = 0.01,
        apd_thresholds: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5],
        inlier_threshold: float = 0.1,
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
            for m in ["epe", "apd", "tau"] + [f"acc3d_{t}" for t in apd_thresholds]
        ]

    # ------------------------------------------------------------------
    def __call__(self, inputs, outputs) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs:  batch dict (unused beyond compatibility)
            outputs: list[ReconstructOutput]

        Returns:
            dict  metric_name -> scalar tensor
        """
        if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
            return {}

        out = outputs[0]

        pred = getattr(out, "track_pred", None)
        gt = getattr(out, "track_gt", None)
        vis = getattr(out, "track_vis_pred", None)
        gt_src = getattr(out, "motion_queries_gt_3d_src", None)

        if pred is None or gt is None:
            return {}

        # Ensure numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        if vis is not None and isinstance(vis, torch.Tensor):
            vis = vis.detach().cpu().numpy()
        if gt_src is not None and isinstance(gt_src, torch.Tensor):
            gt_src = gt_src.detach().cpu().numpy()

        # Build validity mask  [Q]
        valid = np.ones(pred.shape[0], dtype=bool)
        if vis is not None:
            if vis.shape[0] != pred.shape[0]:
                logger.warning(
                    "[SparseMotionEvalMetrics] vis shape %s != pred shape %s; "
                    "truncating vis to match pred",
                    vis.shape, pred.shape,
                )
                vis = vis[:pred.shape[0]]
            valid = valid & (vis > 0.5)

        # Classify dynamic / static using GT displacement magnitude
        is_dynamic = None
        if gt_src is not None:
            gt_disp = gt - gt_src  # [Q, 3]
            gt_motion_mag = np.linalg.norm(gt_disp, axis=-1)  # [Q]
            is_dynamic = gt_motion_mag > self.dynamic_threshold

            if self.dynamic_only and valid.sum() > 0:
                n_dyn = (is_dynamic & valid).sum()
                if n_dyn == 0:
                    valid_mag = gt_motion_mag[valid]
                    logger.warning(
                        "[%sSparseMotionEvalMetrics] dynamic_only=True but 0/%d valid "
                        "points exceed threshold %.4f. "
                        "Displacement stats: max=%.6f, mean=%.6f, p95=%.6f. "
                        "Consider lowering dynamic_threshold or increasing eval_view_num.",
                        self.metric_prefix, int(valid.sum()),
                        self.dynamic_threshold,
                        float(valid_mag.max()) if len(valid_mag) > 0 else 0.0,
                        float(valid_mag.mean()) if len(valid_mag) > 0 else 0.0,
                        float(np.percentile(valid_mag, 95)) if len(valid_mag) > 0 else 0.0,
                    )
        else:
            if self.dynamic_only:
                logger.warning(
                    "[%sSparseMotionEvalMetrics] dynamic_only=True but "
                    "motion_queries_gt_3d_src is None — cannot classify dynamic points. "
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
                result[f"{pfx}acc3d_{t}"] = nan
            return result

        errors = np.linalg.norm(pred[valid] - gt[valid], axis=-1)  # [M]

        result = _compute_metrics(
            errors, self.inlier_threshold, self.apd_thresholds, prefix=pfx,
        )

        # Log dynamic point ratio for diagnostics
        if is_dynamic is not None:
            valid_base = np.ones(pred.shape[0], dtype=bool)
            if vis is not None:
                valid_base = valid_base & (vis[:pred.shape[0]] > 0.5)
            n_valid = valid_base.sum()
            if n_valid > 0:
                result[f"{pfx}dyn_ratio"] = float(is_dynamic[valid_base].sum() / n_valid)

        return result
