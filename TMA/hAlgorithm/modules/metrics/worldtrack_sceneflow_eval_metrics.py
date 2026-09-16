"""WorldTrack / Tapvid3d Scene Flow evaluation metrics (sparse, ref0-based).

Protocol
--------
- GT: ``gt_tracks_world[T,Q,3]`` (frame-0 ref / JSON trajs_3d).
- Pred flow: ``pred_flow_ref0[T,Q,3]`` (``pred_flow_ref0[0]=0``), e.g. warp3d_delta.
- Pred positions (optional): ``pred_tracks_ref0[T,Q,3]`` in the same ref0 frame.
  If omitted, uses ``gt_tracks_world[0] + pred_flow_ref0`` (frame-0 anchored to GT).

Scale alignment
-----------------
On **all** finite (t, q) 3D positions — same as Open-d4rt ``run_eval_worldtrack.sh`` /
``eval_track3d_in_worldtrack.py`` global median-norm scale for Dynamic Points:

    scale = median(||GT_pos||) / median(||Pred_pos||)
            over all finite pred_tracks_ref0 vs gt_tracks_world.

Apply ``scale`` to **pred scene flow** when comparing to GT flow (APD/EPE).
``pred_dyn_count`` is diagnostic only (pred-dynamic query subset), not used for scale.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hAlgorithm.modules.metrics.worldtrack_eval_metrics import WORLDTRACK_APD_THRESHOLDS, compute_scale_factor_global
from hAlgorithm.modules.metrics.worldtrack_eval_metrics import WORLDTRACK_APD_TAU_M

SF_APD_GLOBAL_KEY = "avg_sf_global"
SF_EPE_GLOBAL_KEY = "epe_sf_global"
# GT-dynamic query subset: scene-flow APD/EPE (same global scale, no re-fit on subset).
SF_APD_DYN_KEY = "avg_sf_global_dyn"
SF_EPE_DYN_KEY = "epe_sf_global_dyn"
SF_APD_PRED_DYN_KEY = "avg_sf_pred_dyn"
SF_EPE_PRED_DYN_KEY = "epe_sf_pred_dyn"
DYN_FLOW_COUNT_KEY = "dyn_flow_count"

SCALE_SOURCE_ALL_POSITIONS = "all_positions_vs_gt_trajs"
SCALE_SOURCE_ALL_EMPTY = "all_positions_empty"
# Legacy names (pred-dynamic scale); kept for log compatibility only.
SCALE_SOURCE_PRED_DYNAMIC_POSITIONS = SCALE_SOURCE_ALL_POSITIONS
SCALE_SOURCE_PRED_DYNAMIC_EMPTY = SCALE_SOURCE_ALL_EMPTY


def _gt_dynamic_query_mask_from_tracks(
    gt_tracks_world: np.ndarray,
    *,
    dynamic_threshold: float,
) -> np.ndarray:
    """GT-dynamic queries: sum_t ||gt[t,q]-gt[t-1,q]|| > threshold (metres)."""
    gt = np.asarray(gt_tracks_world, dtype=np.float64)
    if gt.ndim != 3:
        raise ValueError(f"Expected gt_tracks_world [T,Q,3], got shape {gt.shape}")
    if gt.shape[0] < 2:
        return np.zeros((gt.shape[1],), dtype=bool)
    motion = gt[1:] - gt[:-1]
    total_motion_norm = np.linalg.norm(motion, axis=-1).sum(axis=0)
    return total_motion_norm > float(dynamic_threshold)


def _pred_dynamic_query_mask_from_tracks(
    pred_tracks_ref0: np.ndarray,
    *,
    dynamic_threshold: float,
) -> np.ndarray:
    """Per-query mask from predicted trajectory motion (metres)."""
    pt = np.asarray(pred_tracks_ref0, dtype=np.float64)
    if pt.ndim != 3:
        raise ValueError(f"Expected pred_tracks_ref0 [T,Q,3], got shape {pt.shape}")
    if pt.shape[0] < 2:
        return np.zeros((pt.shape[1],), dtype=bool)
    motion = pt[1:] - pt[:-1]
    total_motion_norm = np.linalg.norm(motion, axis=-1).sum(axis=0)
    return total_motion_norm > float(dynamic_threshold)


def _dynamic_flow_vector_count(num_frames: int, dyn_query_count: int) -> int:
    """Number of flow vectors (t>=1) on dynamic queries: (T-1) * |dyn_queries|."""
    if int(num_frames) < 2 or int(dyn_query_count) <= 0:
        return 0
    return int((int(num_frames) - 1) * int(dyn_query_count))


def _subset_scene_flow_metrics(
    gt_tracks_world: np.ndarray,
    pred_flow_ref0: np.ndarray,
    scale: float,
    query_mask: np.ndarray,
    *,
    apd_thresholds: Dict[int, float],
) -> Tuple[float, float, Dict[int, float], int]:
    """Scene-flow APD/EPE on a query subset; returns (apd, epe, fractions, n_flow_vectors)."""
    qmask = np.asarray(query_mask, dtype=bool)
    if not np.any(qmask):
        return float("nan"), float("nan"), {}, 0
    errors = _flow_errors_with_scale(
        gt_tracks_world, pred_flow_ref0, scale, query_mask=qmask
    )
    avg_pts, fractions, epe = _apd_epe_from_errors(errors, apd_thresholds=apd_thresholds)
    return float(avg_pts), float(epe), fractions, int(errors.size)


def _resolve_pred_tracks_ref0(
    gt_tracks_world: np.ndarray,
    pred_flow_ref0: np.ndarray,
    pred_tracks_ref0: Optional[np.ndarray],
) -> np.ndarray:
    if pred_tracks_ref0 is not None:
        pt = np.asarray(pred_tracks_ref0, dtype=np.float64)
    else:
        gt = np.asarray(gt_tracks_world, dtype=np.float64)
        pf = np.asarray(pred_flow_ref0, dtype=np.float64)
        if gt.ndim != 3 or pf.ndim != 3 or gt.shape != pf.shape:
            raise ValueError(f"Expected [T,Q,3] gt/flow with same shape, got {gt.shape} vs {pf.shape}")
        pt = gt[:1] + pf
    if pt.shape != np.asarray(gt_tracks_world).shape:
        raise ValueError(
            f"pred_tracks_ref0 shape {pt.shape} != gt_tracks_world {np.asarray(gt_tracks_world).shape}"
        )
    return pt


def compute_scale_pred_dynamic_positions_vs_gt(
    gt_tracks_world: np.ndarray,
    pred_tracks_ref0: np.ndarray,
    *,
    dynamic_threshold: float = 0.01,
) -> Tuple[float, str, int]:
    """Median-norm scale: **all** pred vs GT 3D positions (ref0).

    ``pred_dyn_count`` is returned for diagnostics only (not used to select scale points).
    """
    gt = np.asarray(gt_tracks_world, dtype=np.float64)
    pt = np.asarray(pred_tracks_ref0, dtype=np.float64)
    pred_dyn_count = int(
        _pred_dynamic_query_mask_from_tracks(pt, dynamic_threshold=dynamic_threshold).sum()
    )

    gt_flat = gt.reshape(-1, 3)
    pr_flat = pt.reshape(-1, 3)
    finite = np.isfinite(gt_flat).all(axis=-1) & np.isfinite(pr_flat).all(axis=-1)
    if not np.any(finite):
        return 1.0, SCALE_SOURCE_ALL_EMPTY, pred_dyn_count

    scale = float(compute_scale_factor_global(gt_flat[finite], pr_flat[finite]))
    return scale, SCALE_SOURCE_ALL_POSITIONS, pred_dyn_count


def _finite_flow_pairs_from_gt_and_pred_flow(
    gt_tracks_world: np.ndarray,
    pred_flow_ref0: np.ndarray,
    *,
    query_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    gt = np.asarray(gt_tracks_world, dtype=np.float64)
    pf = np.asarray(pred_flow_ref0, dtype=np.float64)
    if gt.ndim != 3 or pf.ndim != 3 or gt.shape != pf.shape:
        raise ValueError(f"Expected [T,Q,3] arrays with same shape, got {gt.shape} vs {pf.shape}")
    if gt.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    gt_flow = gt[1:] - gt[:1]
    pred_flow = pf[1:]
    if query_mask is not None:
        qmask = np.asarray(query_mask, dtype=bool)
        if qmask.shape != (gt.shape[1],):
            raise ValueError(f"query_mask must be [Q], got {qmask.shape} vs Q={gt.shape[1]}")
        gt_flow = gt_flow[:, qmask, :]
        pred_flow = pred_flow[:, qmask, :]
    gt_flat = gt_flow.reshape(-1, 3)
    pr_flat = pred_flow.reshape(-1, 3)
    finite = np.isfinite(gt_flat).all(axis=-1) & np.isfinite(pr_flat).all(axis=-1)
    return gt_flat[finite], pr_flat[finite]


def _flow_errors_with_scale(
    gt_tracks_world: np.ndarray,
    pred_flow_ref0: np.ndarray,
    scale: float,
    *,
    query_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    gt_flow, pr_flow = _finite_flow_pairs_from_gt_and_pred_flow(
        gt_tracks_world, pred_flow_ref0, query_mask=query_mask
    )
    if gt_flow.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm((pr_flow * float(scale)) - gt_flow, axis=-1)


def _flow_scale_pred_dynamic_legacy(
    gt_tracks_world: np.ndarray,
    pred_flow_ref0: np.ndarray,
    *,
    dynamic_threshold: float,
) -> Tuple[float, str]:
    """Legacy diagnostic: scale on pred-dynamic *flow* vectors (not used for metrics)."""
    pt = _resolve_pred_tracks_ref0(gt_tracks_world, pred_flow_ref0, None)
    pred_dyn_mask = _pred_dynamic_query_mask_from_tracks(pt, dynamic_threshold=dynamic_threshold)
    gt_flow, pr_flow = _finite_flow_pairs_from_gt_and_pred_flow(
        gt_tracks_world, pred_flow_ref0, query_mask=pred_dyn_mask
    )
    if gt_flow.shape[0] == 0:
        return 1.0, "pred_dynamic_flow_empty"
    return float(compute_scale_factor_global(gt_flow, pr_flow)), "pred_dynamic_flow"


def _apd_epe_from_errors(
    errors: np.ndarray,
    *,
    apd_thresholds: Dict[int, float],
) -> Tuple[float, Dict[int, float], float]:
    if int(errors.size) <= 0:
        fractions = {int(k): float("nan") for k in apd_thresholds.keys()}
        return float("nan"), fractions, float("nan")
    fractions = {int(k): float(np.mean(errors <= float(thr_m))) for k, thr_m in apd_thresholds.items()}
    tau = float(WORLDTRACK_APD_TAU_M)
    avg_pts = float(np.mean(errors <= tau))
    epe = float(np.mean(errors))
    return avg_pts, fractions, epe


def compute_average_flow_pts_within_thresh(
    gt_tracks_world: np.ndarray,
    pred_flow_ref0: np.ndarray,
    *,
    apd_thresholds: Optional[Dict[int, float]] = None,
    fixed_scale: Optional[float] = None,
) -> Tuple[float, Dict[int, float], float, float]:
    if apd_thresholds is None:
        apd_thresholds = WORLDTRACK_APD_THRESHOLDS
    if fixed_scale is None:
        raise ValueError("fixed_scale is required; estimate scale before calling this helper.")
    scale = float(fixed_scale)
    errors = _flow_errors_with_scale(gt_tracks_world, pred_flow_ref0, scale)
    avg_pts, fractions, epe = _apd_epe_from_errors(errors, apd_thresholds=apd_thresholds)
    return avg_pts, fractions, epe, scale


def metrics_for_sequence_scene_flow(
    gt_tracks_world: np.ndarray,
    pred_flow_ref0: np.ndarray,
    *,
    pred_tracks_ref0: Optional[np.ndarray] = None,
    compute_dyn: bool = True,
    dynamic_threshold: float = 0.01,
    apd_thresholds: Optional[Dict[int, float]] = None,
    fixed_scale: Optional[float] = None,
    scale_align_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Scene-flow metrics: global scale from all pred vs GT positions; errors on scaled flow."""
    if apd_thresholds is None:
        apd_thresholds = WORLDTRACK_APD_THRESHOLDS

    gt = np.asarray(gt_tracks_world, dtype=np.float64)
    pred_flow = np.asarray(pred_flow_ref0, dtype=np.float64)
    pred_tracks = _resolve_pred_tracks_ref0(gt, pred_flow, pred_tracks_ref0)

    if fixed_scale is not None:
        scale = float(fixed_scale)
        src = str(scale_align_source or "fixed_scale")
        _, _, pred_dyn_count = compute_scale_pred_dynamic_positions_vs_gt(
            gt, pred_tracks, dynamic_threshold=dynamic_threshold
        )
    else:
        scale, src, pred_dyn_count = compute_scale_pred_dynamic_positions_vs_gt(
            gt, pred_tracks, dynamic_threshold=dynamic_threshold
        )

    flow_scale, flow_src = _flow_scale_pred_dynamic_legacy(
        gt, pred_flow, dynamic_threshold=dynamic_threshold
    )

    avg_sf, fracs_sf, epe_sf, _ = compute_average_flow_pts_within_thresh(
        gt, pred_flow, apd_thresholds=apd_thresholds, fixed_scale=scale
    )

    out: Dict[str, Any] = {
        SF_APD_GLOBAL_KEY: float(avg_sf),
        SF_EPE_GLOBAL_KEY: float(epe_sf),
        "fractions_sf_global": fracs_sf,
        "scale_global": float(scale),
        "scale_align_source": src,
        "pred_dyn_count": int(pred_dyn_count),
        "scale_global_flow_pred_dynamic": float(flow_scale),
        "scale_align_source_flow": flow_src,
        "num_queries": int(gt.shape[1]) if gt.ndim == 3 else 0,
    }

    if compute_dyn and gt.ndim == 3 and gt.shape[0] >= 2:
        gt_dyn_mask = _gt_dynamic_query_mask_from_tracks(gt, dynamic_threshold=dynamic_threshold)
        pred_dyn_mask = _pred_dynamic_query_mask_from_tracks(
            pred_tracks, dynamic_threshold=dynamic_threshold
        )
        out["dyn_count"] = int(gt_dyn_mask.sum())
        out["dyn_fraction"] = float(gt_dyn_mask.mean()) if gt_dyn_mask.size > 0 else float("nan")
        out["pred_dyn_query_count"] = int(pred_dyn_mask.sum())

        if int(out["dyn_count"]) > 0:
            avg_dyn, epe_dyn, fr_dyn, n_dyn_flow = _subset_scene_flow_metrics(
                gt, pred_flow, scale, gt_dyn_mask, apd_thresholds=apd_thresholds
            )
            out[SF_APD_DYN_KEY] = float(avg_dyn)
            out[SF_EPE_DYN_KEY] = float(epe_dyn)
            out["fractions_sf_global_dyn"] = fr_dyn
            out[DYN_FLOW_COUNT_KEY] = int(n_dyn_flow)
        else:
            out[SF_APD_DYN_KEY] = float("nan")
            out[SF_EPE_DYN_KEY] = float("nan")
            out["fractions_sf_global_dyn"] = {}
            out[DYN_FLOW_COUNT_KEY] = 0

        if int(out["pred_dyn_query_count"]) > 0:
            avg_pd, epe_pd, fr_pd, n_pd_flow = _subset_scene_flow_metrics(
                gt, pred_flow, scale, pred_dyn_mask, apd_thresholds=apd_thresholds
            )
            out[SF_APD_PRED_DYN_KEY] = float(avg_pd)
            out[SF_EPE_PRED_DYN_KEY] = float(epe_pd)
            out["fractions_sf_pred_dyn"] = fr_pd
            out["pred_dyn_flow_count"] = int(n_pd_flow)
        else:
            out[SF_APD_PRED_DYN_KEY] = float("nan")
            out[SF_EPE_PRED_DYN_KEY] = float("nan")
            out["fractions_sf_pred_dyn"] = {}
            out["pred_dyn_flow_count"] = 0

    return out


def _weighted_micro_average(
    results: List[Dict[str, Any]],
    metric_key: str,
    weight_key: str,
) -> float:
    """Flow-vector–weighted mean of per-sequence metrics (for dynamic subsets)."""
    num = 0.0
    den = 0.0
    for item in results:
        if metric_key not in item or weight_key not in item:
            continue
        val = float(item[metric_key])
        weight = int(item[weight_key])
        if weight <= 0 or not np.isfinite(val):
            continue
        num += val * float(weight)
        den += float(weight)
    return float(num / den) if den > 0 else float("nan")


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scalar_keys = [
        SF_APD_GLOBAL_KEY,
        SF_EPE_GLOBAL_KEY,
        SF_APD_DYN_KEY,
        SF_EPE_DYN_KEY,
        SF_APD_PRED_DYN_KEY,
        SF_EPE_PRED_DYN_KEY,
        "dyn_fraction",
    ]
    fraction_keys = ["fractions_sf_global", "fractions_sf_global_dyn", "fractions_sf_pred_dyn"]
    summary: Dict[str, Any] = {"num_sequences": int(len(results))}
    for key in scalar_keys:
        values = [
            float(item[key])
            for item in results
            if key in item and np.isfinite(float(item[key]))
        ]
        summary[key] = float(np.mean(values)) if values else float("nan")
    summary["total_queries"] = int(sum(int(item.get("num_queries", 0)) for item in results))
    summary["total_dynamic_queries"] = int(sum(int(item.get("dyn_count", 0)) for item in results))
    summary["total_pred_dynamic_queries"] = int(sum(int(item.get("pred_dyn_count", 0)) for item in results))
    summary["total_dyn_flow_vectors"] = int(sum(int(item.get(DYN_FLOW_COUNT_KEY, 0)) for item in results))
    summary["total_pred_dyn_flow_vectors"] = int(
        sum(int(item.get("pred_dyn_flow_count", 0)) for item in results)
    )
    summary[f"{SF_APD_DYN_KEY}_micro"] = _weighted_micro_average(
        results, SF_APD_DYN_KEY, DYN_FLOW_COUNT_KEY
    )
    summary[f"{SF_EPE_DYN_KEY}_micro"] = _weighted_micro_average(
        results, SF_EPE_DYN_KEY, DYN_FLOW_COUNT_KEY
    )

    for frac_key in fraction_keys:
        agg: Dict[int, List[float]] = defaultdict(list)
        for item in results:
            payload = item.get(frac_key, {})
            if not isinstance(payload, dict):
                continue
            for thr, value in payload.items():
                if np.isfinite(float(value)):
                    agg[int(thr)].append(float(value))
        summary[frac_key] = {int(thr): float(np.mean(vals)) for thr, vals in sorted(agg.items())}
    return summary


def format_subset_summary(subset: str, summary: Dict[str, Any]) -> str:
    dyn_apd = summary.get(SF_APD_DYN_KEY, summary.get("avg_sf_global_dyn", float("nan")))
    dyn_epe = summary.get(SF_EPE_DYN_KEY, summary.get("epe_sf_global_dyn", float("nan")))
    dyn_apd_micro = summary.get(f"{SF_APD_DYN_KEY}_micro", float("nan"))
    dyn_epe_micro = summary.get(f"{SF_EPE_DYN_KEY}_micro", float("nan"))
    return (
        f"{subset}: "
        f"SF-APD/tau(all)={summary.get(SF_APD_GLOBAL_KEY, float('nan')):.4f} "
        f"SF-EPE(all)={summary.get(SF_EPE_GLOBAL_KEY, float('nan')):.4f} | "
        f"DynPoints-APD/tau(GT)={dyn_apd:.4f} DynPoints-EPE(GT)={dyn_epe:.4f} "
        f"[micro APD={dyn_apd_micro:.4f} EPE={dyn_epe_micro:.4f}] "
        f"dyn_q={int(summary.get('total_dynamic_queries', 0))}/"
        f"{int(summary.get('total_queries', 0))}"
    )
