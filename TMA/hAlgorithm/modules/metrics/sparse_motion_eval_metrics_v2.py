"""Sparse Motion Evaluation Metrics V2

Evaluates query-based 3D motion predictions from Track3DOutput objects
produced by WFMQueryPipeline (mvfr_query_v4).

Each ReconstructOutput carries a ``track_3d`` dict mapping
    tgt_index -> Track3DOutput
with fields:
    - warp3d:          [Q, 3]  predicted target 3D position (world frame)
    - warp3d_uv:       [Q, 2]  predicted target 2D pixel coords (optional)
    - tgt_3d_gt:       [Q, 3]  GT target 3D position
    - src_3d_gt:       [Q, 3]  GT source 3D position (for dynamic classification)
    - tgt_2d_gt:       [Q, 2]  GT target 2D normalised coords
    - src_valids_gt:   [Q]     source point validity
    - tgt_valids_gt:   [Q]     target point validity
    - src_visibs_gt:   [Q]     source point visibility
    - tgt_visibs_gt:   [Q]     target point visibility

Metrics (3D):
    - epe_3d:   End-Point Error in 3D (mean L2 distance)
    - apd_3d:   Average Points within Delta (mean acc over thresholds)
    - tau_3d:   Inlier ratio at ``inlier_threshold_3d``
    - acc3d_*:  Per-threshold 3D accuracy

Metrics (2D, optional when warp3d_uv is available):
    - epe_2d:   End-Point Error in 2D pixel space
    - apd_2d:   Average Points within Delta (pixel thresholds)
    - tau_2d:   Inlier ratio at ``inlier_threshold_2d``
    - acc2d_*:  Per-threshold 2D accuracy
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _to_np(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _compute_metrics(
    errors: np.ndarray,
    inlier_threshold: float,
    apd_thresholds: List[float],
    prefix: str = "",
    acc_tag: str = "acc",
) -> Dict[str, float]:
    """Compute EPE / APD / tau / acc from a 1-D error array.

    Args:
        acc_tag: tag used for per-threshold keys, e.g. "acc3d" → ``{prefix}acc3d_{t}``.
    """
    if len(errors) == 0:
        nan = float("nan")
        result = {f"{prefix}epe": nan, f"{prefix}apd": nan, f"{prefix}tau": nan}
        for t in apd_thresholds:
            result[f"{prefix}{acc_tag}_{t}"] = nan
        return result

    epe = float(np.mean(errors))

    if inlier_threshold is not None:
        tau = float(np.mean(errors < inlier_threshold))
    else:
        tau = None

    accs, acc_dict = [], {}
    for t in apd_thresholds:
        a = float(np.mean(errors < t))
        accs.append(a)
        acc_dict[f"{prefix}{acc_tag}_{t}"] = a

    return {
        f"{prefix}epe": epe,
        f"{prefix}apd": float(np.mean(accs)),
        f"{prefix}tau": tau,
        **acc_dict,
    }


class SparseMotionEvalMetrics:
    """Query-based 3D/2D motion evaluation for Track3DOutput from WFMQueryPipeline.

    Iterates over all (src, tgt) pairs across all output views, aggregates
    errors, then reports 3D and (optionally) 2D metrics.
    """

    def __init__(
        self,
        dynamic_threshold: float = 0.01,
        apd_thresholds_3d: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5],
        inlier_threshold_3d: float = 0.1,
        apd_thresholds_2d: Optional[List[float]] = [1.0, 3.0],
        inlier_threshold_2d: float = 1.0,
        image_hw: Optional[tuple] = None,
        dynamic_only: bool = False,
        require_visible: bool = True,
        metric_prefix: str = "",
        with_motion_mask: bool = False,
        with_delta_motion_mask: bool = False,
        with_warp3d_delta: bool = False,
        with_warp3d_to_delta: bool = False,
        with_src_points: bool = False,
        with_dy_ratio: bool = True,
        **kwargs,
    ):
        """
        Args:
            dynamic_threshold: 3D displacement magnitude above which a point
                is classified as dynamic.
            apd_thresholds_3d: list of 3D distance thresholds (metres) for accuracy.
            inlier_threshold_3d: threshold for 3D tau (inlier ratio).
            apd_thresholds_2d: list of 2D pixel thresholds for accuracy.
                Set to None to disable 2D evaluation.
            inlier_threshold_2d: threshold for 2D tau (pixels).
            image_hw: (H, W) used for denormalising GT 2D coords when they are
                in [0, 1].  If None, inferred from ``output.rgb`` shape.
            dynamic_only: if True, only evaluate dynamic points.
            require_visible: if True, additionally require src/tgt visibility
                (not just validity) for a point to be evaluated.
            metric_prefix: string prepended to every metric key.
        """
        self.dynamic_threshold = dynamic_threshold
        self.apd_thresholds_3d = apd_thresholds_3d
        self.inlier_threshold_3d = inlier_threshold_3d
        self.apd_thresholds_2d = apd_thresholds_2d
        self.inlier_threshold_2d = inlier_threshold_2d
        self.image_hw = image_hw
        self.dynamic_only = dynamic_only
        self.require_visible = require_visible
        self.metric_prefix = metric_prefix
        self.with_motion_mask = with_motion_mask
        self.with_delta_motion_mask = with_delta_motion_mask
        self.with_warp3d_delta = with_warp3d_delta
        self.with_warp3d_to_delta = with_warp3d_to_delta
        self.with_src_points = with_src_points
        self.with_dy_ratio = with_dy_ratio

        self.metrics = self._build_metric_names()

    def _build_metric_names(self) -> List[str]:
        pfx = self.metric_prefix

        names = [f"{pfx}epe_3d", f"{pfx}apd_3d"]
        if self.inlier_threshold_3d is not None:
            names.append(f"{pfx}tau_3d")
        for t in self.apd_thresholds_3d:
            names.append(f"{pfx}acc3d_{t}")

        if self.inlier_threshold_2d is not None:
            names.append(f"{pfx}tau_2d")
        if self.apd_thresholds_2d is not None:
            names += [f"{pfx}epe_2d", f"{pfx}apd_2d"]
            for t in self.apd_thresholds_2d:
                names.append(f"{pfx}acc2d_{t}")

        if self.with_dy_ratio:
            names.append(f"{pfx}dyn_ratio")

        return names

    def _build_valid_mask(self, track) -> np.ndarray:
        """Build a boolean validity mask from src/tgt valids and visibs."""
        src_valids = _to_np(track.src_valids_gt)
        tgt_valids = _to_np(track.tgt_valids_gt)
        tgt_3d_gt = _to_np(track.tgt_3d_gt)

        Q = tgt_3d_gt.shape[0]
        valid = np.ones(Q, dtype=bool)

        if src_valids is not None:
            valid &= (src_valids > 0.5)
        if tgt_valids is not None:
            valid &= (tgt_valids > 0.5)

        if self.require_visible:
            src_visibs = _to_np(track.src_visibs_gt)
            tgt_visibs = _to_np(track.tgt_visibs_gt)
            if src_visibs is not None:
                valid &= (src_visibs > 0.5)
            if tgt_visibs is not None:
                valid &= (tgt_visibs > 0.5)

        return valid

    def _classify_dynamic(self, track):
        """Classify points as dynamic based on GT 3D displacement."""
        src_3d = _to_np(track.src_3d_gt)
        tgt_3d = _to_np(track.tgt_3d_gt)

        if src_3d is None or tgt_3d is None:
            return None

        disp = np.linalg.norm(tgt_3d - src_3d, axis=-1)
        return disp > self.dynamic_threshold

    def _get_2d_errors(self, track, valid: np.ndarray, tgt_output=None):
        """Compute 2D pixel errors between warp3d_uv and tgt_2d_gt.

        ``warp3d_uv`` is already in pixel coords; ``tgt_2d_gt`` is normalised
        [0, 1] and will be converted to pixel coords using image size.
        """
        warp2d = _to_np(track.warp2d)
        tgt_2d_gt = _to_np(track.tgt_2d_gt)

        if warp2d is None or tgt_2d_gt is None:
            return None

        H, W = track.query_height, track.query_width

        tgt_2d_gt = tgt_2d_gt.copy()
        tgt_2d_gt = tgt_2d_gt.astype(np.float64)
        tgt_2d_gt[:, 0] *= (W - 1)
        tgt_2d_gt[:, 1] *= (H - 1)
        
        warp2d = warp2d.copy()
        warp2d = warp2d.astype(np.float64)
        warp2d[:, 0] *= (W - 1)
        warp2d[:, 1] *= (H - 1)

        return np.linalg.norm(warp2d[valid] - tgt_2d_gt[valid], axis=-1)

    # ------------------------------------------------------------------
    def __call__(self, inputs, outputs) -> Dict[str, float]:
        """
        Args:
            inputs:  batch dict (unused, kept for interface compatibility)
            outputs: list[ReconstructOutput]

        Returns:
            dict  metric_name -> float
        """
        if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
            return {}

        all_errors_3d: List[np.ndarray] = []
        all_errors_2d: List[np.ndarray] = []
        all_is_dynamic: List[np.ndarray] = []

        for out in outputs:
            track_3d_dict = getattr(out, "track_3d", None)
            if track_3d_dict is None or len(track_3d_dict) == 0:
                continue

            for tgt_idx, track in track_3d_dict.items():
                warp3d = _to_np(track.warp3d)
                warp3d_delta = _to_np(track.warp3d_delta)

                src_3d_gt = _to_np(track.src_3d_gt)
                tgt_3d_gt = _to_np(track.tgt_3d_gt)

                src_points = _to_np(track.src_points)

                if warp3d is None and warp3d_delta is None:
                    continue
                if tgt_3d_gt is None:
                    continue

                if self.with_warp3d_delta:
                    if self.with_src_points:
                        warp3d = src_points + warp3d_delta
                    else:
                        warp3d = src_3d_gt + warp3d_delta

                if self.with_warp3d_to_delta:
                    warp3d_delta = warp3d - src_points
                    if self.with_src_points:
                        warp3d = src_points + warp3d_delta
                    else:
                        warp3d = src_3d_gt + warp3d_delta

                if self.with_motion_mask and track.motion_mask is not None:
                    static_mask = ~track.motion_mask
                    if self.with_src_points:
                        warp3d[static_mask] = src_points[static_mask]
                    else:
                        warp3d[static_mask] = src_3d_gt[static_mask]

                if self.with_delta_motion_mask and warp3d_delta is not None:
                    static_mask = np.abs(warp3d_delta) < 0.01
                    if self.with_src_points:
                        warp3d[static_mask] = src_points[static_mask]
                    else:
                        warp3d[static_mask] = src_3d_gt[static_mask]

                valid = self._build_valid_mask(track)
                is_dynamic = self._classify_dynamic(track)

                if self.dynamic_only:
                    if is_dynamic is None:
                        logger.warning(
                            "[%sSparseMotionEvalMetrics] dynamic_only=True but "
                            "cannot classify dynamic (missing src/tgt 3D GT). "
                            "Skipping pair src→tgt=%d.",
                            self.metric_prefix, tgt_idx,
                        )
                        continue
                    valid = valid & is_dynamic

                if valid.sum() == 0:
                    continue

                # 3D errors
                all_errors_3d.append(
                    np.linalg.norm(warp3d[valid] - tgt_3d_gt[valid], axis=-1)
                )

                # 2D errors
                tgt_output = outputs[tgt_idx] if tgt_idx < len(outputs) else None
                errors_2d = self._get_2d_errors(track, valid, tgt_output=tgt_output)
                if errors_2d is not None:
                    all_errors_2d.append(errors_2d)

                # Dynamic ratio stats (over valid-mask points, before dynamic filtering)
                if is_dynamic is not None:
                    all_is_dynamic.append(is_dynamic[self._build_valid_mask(track)])

        pfx = self.metric_prefix

        if len(all_errors_3d) == 0:
            return self._nan_result()

        # ── 3D metrics ──
        errors_3d = np.concatenate(all_errors_3d)
        result_3d = _compute_metrics(
            errors_3d, self.inlier_threshold_3d, self.apd_thresholds_3d,
            prefix=f"{pfx}", acc_tag="acc3d",
        )
        result = {
            f"{pfx}epe_3d": result_3d[f"{pfx}epe"],
            # f"{pfx}apd_3d": result_3d[f"{pfx}apd"],
            # f"{pfx}tau_3d": result_3d[f"{pfx}tau"],
        }

        if self.inlier_threshold_3d is not None:
            result[f"{pfx}tau_3d"] = result_3d[f"{pfx}tau"]

        for t in self.apd_thresholds_3d:
            result[f"{pfx}acc3d_{t}"] = result_3d[f"{pfx}acc3d_{t}"]

        # ── 2D metrics ──
        if self.apd_thresholds_2d is not None and len(all_errors_2d) > 0:
            errors_2d = np.concatenate(all_errors_2d)
            result_2d = _compute_metrics(
                errors_2d, self.inlier_threshold_2d, self.apd_thresholds_2d,
                prefix=f"{pfx}", acc_tag="acc2d",
            )
            result[f"{pfx}epe_2d"] = result_2d[f"{pfx}epe"]
            # result[f"{pfx}apd_2d"] = result_2d[f"{pfx}apd"]

            if self.inlier_threshold_2d is not None:
                result[f"{pfx}tau_2d"] = result_2d[f"{pfx}tau"]

            for t in self.apd_thresholds_2d:
                result[f"{pfx}acc2d_{t}"] = result_2d[f"{pfx}acc2d_{t}"]

        # ── Diagnostic stats ──
        if len(all_is_dynamic) > 0 and self.with_dy_ratio:
            all_dyn = np.concatenate(all_is_dynamic)
            result[f"{pfx}dyn_ratio"] = float(all_dyn.mean())

        return result

    def _nan_result(self) -> Dict[str, float]:
        pfx = self.metric_prefix
        nan = float("nan")
        result = {
            f"{pfx}epe_3d": nan, 
            # f"{pfx}apd_3d": nan, 
            # f"{pfx}tau_3d": nan,
        }
        if self.inlier_threshold_3d is not None:
            result[f"{pfx}tau_3d"] = nan
        for t in self.apd_thresholds_3d:
            result[f"{pfx}acc3d_{t}"] = nan

        if self.inlier_threshold_2d is not None:
            result[f"{pfx}tau_2d"] = nan
        if self.apd_thresholds_2d is not None:
            result[f"{pfx}epe_2d"] = nan
            # result[f"{pfx}apd_2d"] = nan
            for t in self.apd_thresholds_2d:
                result[f"{pfx}acc2d_{t}"] = nan

        if self.with_dy_ratio:
            result[f"{pfx}dyn_ratio"] = nan

        return result
