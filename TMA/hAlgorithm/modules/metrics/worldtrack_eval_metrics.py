"""WorldTrack / St4RTrack 3D tracking evaluation metrics.

Aligned with Open-d4rt ``eval_track3d_in_worldtrack.py`` / ``run_eval_worldtrack.sh``:
    - APD = ``avg_pts_global`` = mean of hit-rate fractions at 0.1 / 0.3 / 0.5 / 1.0 m
    - tau = ``tau_global`` = hit-rate at 0.1 m only (= ``fractions_global[1]``)
    - EPE = ``epe_global`` (mean 3D error after global scale alignment)

Inputs are trajectories in the frame-0 reference world coordinate system:
    gt_tracks_world, pred_tracks_ref0: [T, Q, 3]
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# TAPVid-style keys -> metric thresholds in metres (St4RTrack / WorldTrack protocol).
WORLDTRACK_APD_THRESHOLDS: Dict[int, float] = {
    1: 0.1,
    2: 0.3,
    4: 0.5,
    8: 1.0,
}

# tau threshold (metres); ``fractions_global[1]`` uses WORLDTRACK_APD_THRESHOLDS[1].
WORLDTRACK_APD_TAU_M: float = 0.1

# Summary JSON keys (Open-d4rt ``eval_track3d_in_worldtrack.py`` naming).
APD_GLOBAL_KEY = "avg_pts_global"
TAU_GLOBAL_KEY = "tau_global"
EPE_GLOBAL_KEY = "epe_global"
TAU_GLOBAL_DYN_KEY = "tau_global_dyn"


def dynamic_point_mask_ref0(
    tracks_tq3: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """True for queries whose GT displacement from frame 0 exceeds ``threshold`` (m).

    Matches ``worldtrack_json_loader.compute_dynamic_point_mask`` (SynthVerse / PO use
    world ``trajs_3d``; TapVid3D uses ref0-world tracks).
    """
    tracks = np.asarray(tracks_tq3, dtype=np.float64)
    if tracks.ndim != 3:
        raise ValueError(f"Expected tracks [T,Q,3], got {tracks.shape}")
    disp = np.linalg.norm(tracks - tracks[0:1], axis=-1)
    with np.errstate(invalid="ignore"):
        max_disp = np.nanmax(disp, axis=0)
    return np.isfinite(max_disp) & (max_disp > float(threshold))


def compute_scale_factor_global(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    query_mask: Optional[np.ndarray] = None,
) -> float:
    """Median-norm scale: median(||GT||) / median(||Pred||) over finite points."""
    gt = np.asarray(gt_points, dtype=np.float64)
    pred = np.asarray(pred_points, dtype=np.float64)
    gt_flat = gt.reshape(-1, 3)
    pred_flat = pred.reshape(-1, 3)
    finite = np.isfinite(gt_flat).all(axis=-1) & np.isfinite(pred_flat).all(axis=-1)
    if query_mask is not None:
        qmask = np.asarray(query_mask, dtype=bool).reshape(-1)
        if qmask.size != gt.shape[1]:
            raise ValueError(f"query_mask length {qmask.size} != Q={gt.shape[1]}")
        frame_q = np.broadcast_to(qmask.reshape(1, -1), gt.shape[:2]).reshape(-1)
        finite = finite & frame_q
    if not np.any(finite):
        return 1.0
    gt_norm = np.linalg.norm(gt_flat[finite], axis=-1)
    pred_norm = np.linalg.norm(pred_flat[finite], axis=-1)
    eps = 1e-12
    if gt_norm.size <= 0 or pred_norm.size <= 0:
        return 1.0
    gt_norm = np.maximum(gt_norm, eps)
    pred_norm = np.maximum(pred_norm, eps)
    return float(np.median(gt_norm) / max(float(np.median(pred_norm)), eps))


def scale_per_trajectory(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
) -> np.ndarray:
    """Per-trajectory median scale alignment."""
    gt = np.asarray(gt_points, dtype=np.float64)
    pred = np.asarray(pred_points, dtype=np.float64)
    out = pred.copy()
    eps = 1e-12
    for idx in range(gt.shape[1]):
        finite = np.isfinite(gt[:, idx]).all(axis=-1) & np.isfinite(pred[:, idx]).all(axis=-1)
        if not np.any(finite):
            continue
        gt_norm = np.linalg.norm(gt[finite, idx], axis=-1)
        pred_norm = np.linalg.norm(pred[finite, idx], axis=-1)
        if gt_norm.size <= 0 or pred_norm.size <= 0:
            continue
        gt_norm = np.maximum(gt_norm, eps)
        pred_norm = np.maximum(pred_norm, eps)
        scale = float(np.median(gt_norm) / max(float(np.median(pred_norm)), eps))
        out[:, idx] = pred[:, idx] * scale
    return out


def estimate_sim3_closed_form(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    centroid_src = src.mean(axis=0, keepdims=True)
    centroid_dst = dst.mean(axis=0, keepdims=True)
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    h = src_centered.T @ dst_centered
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    var_src = float((src_centered**2).sum())
    scale = float(np.sum(s) / max(var_src, 1e-12))
    t = centroid_dst[0] - scale * (r @ centroid_src[0])
    return scale, r, t


def estimate_sim3_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    iterations: int = 1000,
    inlier_threshold: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 points for Sim3 estimation.")
    if rng is None:
        rng = np.random.default_rng(0)
    best_count = -1
    best_model: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
    best_mask: Optional[np.ndarray] = None
    for _ in range(int(iterations)):
        subset_idx = rng.choice(src.shape[0], size=3, replace=False)
        try:
            scale, rot, trans = estimate_sim3_closed_form(src[subset_idx], dst[subset_idx])
        except np.linalg.LinAlgError:
            continue
        transformed = scale * (rot @ src.T).T + trans
        dists = np.linalg.norm(transformed - dst, axis=1)
        mask = dists < float(inlier_threshold)
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_model = (scale, rot, trans)
            best_mask = mask
    if best_model is None:
        return estimate_sim3_closed_form(src, dst)
    if best_mask is not None and int(best_mask.sum()) >= 3:
        return estimate_sim3_closed_form(src[best_mask], dst[best_mask])
    return best_model


def finite_correspondence_pairs(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    src_flat = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst_flat = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(src_flat).all(axis=-1) & np.isfinite(dst_flat).all(axis=-1)
    return src_flat[finite], dst_flat[finite]


def compute_average_pts_within_thresh(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    scaling: str = "global",
    compute_epe: bool = True,
    apd_thresholds: Optional[Dict[int, float]] = None,
    rng: Optional[np.random.Generator] = None,
    align_query_mask: Optional[np.ndarray] = None,
) -> Tuple[
    float,
    np.ndarray,
    Dict[int, float],
    Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]],
    float,
]:
    """Compute APD, per-threshold fractions, and EPE after the chosen alignment.

    APD = mean(fractions) over WORLDTRACK_APD_THRESHOLDS (Open-d4rt protocol).
    tau@0.1m = fractions[1].

    Returns:
        apd, pred_aligned, fractions, alignment_params, epe
    """
    if apd_thresholds is None:
        apd_thresholds = WORLDTRACK_APD_THRESHOLDS
    if rng is None:
        rng = np.random.default_rng(0)

    gt = np.asarray(gt_points, dtype=np.float64)
    pred = np.asarray(pred_points, dtype=np.float64)
    params: Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]

    if scaling == "global":
        scale = compute_scale_factor_global(gt, pred, query_mask=align_query_mask)
        pred_aligned = pred * scale
        params = (scale, np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64))
    elif scaling == "per_traj":
        pred_aligned = scale_per_trajectory(gt, pred)
        params = (1.0, np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64))
    elif scaling == "sim3_closed":
        src_fit, dst_fit = finite_correspondence_pairs(pred, gt)
        if src_fit.shape[0] < 3:
            pred_aligned = np.full_like(pred, np.nan, dtype=np.float64)
            params = (None, None, None)
        else:
            scale, rot, trans = estimate_sim3_closed_form(src_fit, dst_fit)
            src = pred.reshape(-1, 3)
            finite = np.isfinite(src).all(axis=-1)
            pred_aligned = np.full_like(src, np.nan, dtype=np.float64)
            pred_aligned[finite] = (scale * (rot @ src[finite].T)).T + trans
            pred_aligned = pred_aligned.reshape(gt.shape)
            params = (scale, rot, trans)
    elif scaling == "sim3":
        src_fit, dst_fit = finite_correspondence_pairs(pred, gt)
        if src_fit.shape[0] < 3:
            pred_aligned = np.full_like(pred, np.nan, dtype=np.float64)
            params = (None, None, None)
        else:
            if src_fit.shape[0] > 16384:
                pick = rng.choice(src_fit.shape[0], size=16384, replace=False)
                src_sample = src_fit[pick]
                dst_sample = dst_fit[pick]
            else:
                src_sample = src_fit
                dst_sample = dst_fit
            scale, rot, trans = estimate_sim3_ransac(src_sample, dst_sample, rng=rng)
            src = pred.reshape(-1, 3)
            finite = np.isfinite(src).all(axis=-1)
            pred_aligned = np.full_like(src, np.nan, dtype=np.float64)
            pred_aligned[finite] = (scale * (rot @ src[finite].T)).T + trans
            pred_aligned = pred_aligned.reshape(gt.shape)
            params = (scale, rot, trans)
    else:
        raise ValueError(f"Unknown scaling: {scaling}")

    dists = np.linalg.norm(pred_aligned - gt, axis=-1)
    total_points = int(np.isfinite(dists).sum())
    fractions: Dict[int, float] = {}
    for thr_key, fixed_threshold in apd_thresholds.items():
        within_dist = np.isfinite(dists) & (dists <= float(fixed_threshold))
        fractions[int(thr_key)] = float(np.sum(within_dist) / max(total_points, 1))
    apd = float(np.mean(list(fractions.values()))) if fractions else float("nan")
    if compute_epe and np.any(np.isfinite(dists)):
        epe = float(np.mean(dists[np.isfinite(dists)]))
    else:
        epe = float("inf")
    return apd, pred_aligned, fractions, params, epe


def metrics_for_sequence(
    gt_tracks_world: np.ndarray,
    pred_tracks_ref0: np.ndarray,
    compute_dyn: bool = True,
    dynamic_threshold: float = 0.01,
    eval_dynamic_only: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Per-sequence WorldTrack metrics for trajectories in ref0 world frame.

    Args:
        gt_tracks_world: [T, Q, 3] ground truth.
        pred_tracks_ref0: [T, Q, 3] predictions in the same frame.
        compute_dyn: Whether to compute dynamic-query subset metrics.
        dynamic_threshold: Ref0 max displacement (m) above which a query is dynamic.
        eval_dynamic_only: If True, score only dynamic queries and fit global scale on
            those queries only (SynthVerse dynamic-points protocol).
        rng: RNG for Sim3 RANSAC subsampling (seed=0 when None).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    gt_full = np.asarray(gt_tracks_world, dtype=np.float64)
    pred_full = np.asarray(pred_tracks_ref0, dtype=np.float64)
    ref0_dyn_mask = dynamic_point_mask_ref0(gt_full, dynamic_threshold)
    ref0_dyn_count = int(ref0_dyn_mask.sum())

    gt_use = gt_full
    pred_use = pred_full
    align_mask: Optional[np.ndarray] = None
    if eval_dynamic_only:
        if ref0_dyn_count <= 0:
            return {
                APD_GLOBAL_KEY: float("nan"),
                TAU_GLOBAL_KEY: float("nan"),
                "avg_pts_pertraj": float("nan"),
                "avg_pts_sim3": float("nan"),
                "avg_pts_sim3_closed": float("nan"),
                EPE_GLOBAL_KEY: float("nan"),
                "epe_pertraj": float("nan"),
                "epe_sim3": float("nan"),
                "epe_sim3_closed": float("nan"),
                "avg_pts_global_dyn": float("nan"),
                TAU_GLOBAL_DYN_KEY: float("nan"),
                "epe_global_dyn": float("nan"),
                "avg_pts_sim3_closed_dyn": float("nan"),
                "epe_sim3_closed_dyn": float("nan"),
                "fractions_global": {},
                "fractions_pertraj": {},
                "fractions_sim3": {},
                "fractions_sim3_closed": {},
                "fractions_global_dyn": {},
                "fractions_sim3_closed_dyn": {},
                "dyn_fraction": float("nan"),
                "dyn_count": 0,
                "ref0_dyn_count": 0,
                "num_queries": 0,
                "eval_dynamic_only": True,
                "dynamic_threshold": float(dynamic_threshold),
            }
        gt_use = gt_full[:, ref0_dyn_mask]
        pred_use = pred_full[:, ref0_dyn_mask]

    avg_pts_global, _, fractions_global, _, epe_global = compute_average_pts_within_thresh(
        gt_use,
        pred_use,
        scaling="global",
        compute_epe=True,
        rng=rng,
        align_query_mask=align_mask,
    )
    avg_pts_pertraj, _, fractions_pertraj, _, epe_pertraj = compute_average_pts_within_thresh(
        gt_use,
        pred_use,
        scaling="per_traj",
        compute_epe=True,
        rng=rng,
    )
    avg_pts_sim3, _, fractions_sim3, _, epe_sim3 = compute_average_pts_within_thresh(
        gt_use,
        pred_use,
        scaling="sim3",
        compute_epe=True,
        rng=rng,
    )
    avg_pts_sim3_closed, _, fractions_sim3_closed, _, epe_sim3_closed = compute_average_pts_within_thresh(
        gt_use,
        pred_use,
        scaling="sim3_closed",
        compute_epe=True,
        rng=rng,
    )

    avg_pts_global_dyn = float("nan")
    epe_global_dyn = float("nan")
    avg_pts_sim3_closed_dyn = float("nan")
    epe_sim3_closed_dyn = float("nan")
    fractions_global_dyn: Dict[int, float] = {}
    fractions_sim3_closed_dyn: Dict[int, float] = {}
    dyn_fraction = float("nan")
    dyn_count = 0

    if compute_dyn and gt_use.shape[0] >= 2:
        total_motion = gt_use[1:] - gt_use[:-1]
        total_motion_norm = np.linalg.norm(total_motion, axis=-1).sum(axis=0)
        dyn_mask = total_motion_norm > float(dynamic_threshold)
        dyn_count = int(dyn_mask.sum())
        dyn_fraction = float(dyn_mask.mean()) if dyn_mask.size > 0 else float("nan")
        if dyn_count > 0:
            avg_pts_global_dyn, _, fractions_global_dyn, _, epe_global_dyn = compute_average_pts_within_thresh(
                gt_use[:, dyn_mask],
                pred_use[:, dyn_mask],
                scaling="global",
                compute_epe=True,
                rng=rng,
            )
            avg_pts_sim3_closed_dyn, _, fractions_sim3_closed_dyn, _, epe_sim3_closed_dyn = compute_average_pts_within_thresh(
                gt_use[:, dyn_mask],
                pred_use[:, dyn_mask],
                scaling="sim3_closed",
                compute_epe=True,
                rng=rng,
            )

    tau_global = float(fractions_global.get(1, float("nan")))
    tau_global_dyn = float(fractions_global_dyn.get(1, float("nan")))

    return {
        APD_GLOBAL_KEY: float(avg_pts_global),
        TAU_GLOBAL_KEY: tau_global,
        "avg_pts_pertraj": float(avg_pts_pertraj),
        "avg_pts_sim3": float(avg_pts_sim3),
        "avg_pts_sim3_closed": float(avg_pts_sim3_closed),
        EPE_GLOBAL_KEY: float(epe_global),
        "epe_pertraj": float(epe_pertraj),
        "epe_sim3": float(epe_sim3),
        "epe_sim3_closed": float(epe_sim3_closed),
        "avg_pts_global_dyn": float(avg_pts_global_dyn),
        TAU_GLOBAL_DYN_KEY: tau_global_dyn,
        "epe_global_dyn": float(epe_global_dyn),
        "avg_pts_sim3_closed_dyn": float(avg_pts_sim3_closed_dyn),
        "epe_sim3_closed_dyn": float(epe_sim3_closed_dyn),
        "fractions_global": fractions_global,
        "fractions_pertraj": fractions_pertraj,
        "fractions_sim3": fractions_sim3,
        "fractions_sim3_closed": fractions_sim3_closed,
        "fractions_global_dyn": fractions_global_dyn,
        "fractions_sim3_closed_dyn": fractions_sim3_closed_dyn,
        "dyn_fraction": float(dyn_fraction),
        "dyn_count": int(dyn_count),
        "ref0_dyn_count": int(ref0_dyn_count),
        "num_queries": int(gt_use.shape[1]),
        "eval_dynamic_only": bool(eval_dynamic_only),
        "dynamic_threshold": float(dynamic_threshold),
    }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean-pool per-sequence metrics into a subset summary."""
    scalar_keys = [
        APD_GLOBAL_KEY,
        TAU_GLOBAL_KEY,
        "avg_pts_pertraj",
        "avg_pts_sim3",
        "avg_pts_sim3_closed",
        EPE_GLOBAL_KEY,
        "epe_pertraj",
        "epe_sim3",
        "epe_sim3_closed",
        "avg_pts_global_dyn",
        TAU_GLOBAL_DYN_KEY,
        "epe_global_dyn",
        "avg_pts_sim3_closed_dyn",
        "epe_sim3_closed_dyn",
        "dyn_fraction",
    ]
    fraction_keys = [
        "fractions_global",
        "fractions_pertraj",
        "fractions_sim3",
        "fractions_sim3_closed",
        "fractions_global_dyn",
        "fractions_sim3_closed_dyn",
    ]
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
    """One-line summary matching Open-d4rt ``_format_subset_summary`` field names."""
    return (
        f"{subset}: "
        f"APD(global)={summary.get(APD_GLOBAL_KEY, float('nan')):.4f} "
        f"tau(global)={summary.get(TAU_GLOBAL_KEY, float('nan')):.4f} "
        f"EPE(global)={summary.get(EPE_GLOBAL_KEY, float('nan')):.4f} "
        f"APD(global,dyn)={summary.get('avg_pts_global_dyn', float('nan')):.4f} "
        f"tau(global,dyn)={summary.get(TAU_GLOBAL_DYN_KEY, float('nan')):.4f} "
        f"EPE(global,dyn)={summary.get('epe_global_dyn', float('nan')):.4f} "
        f"queries={int(summary.get('total_queries', 0))}"
    )


def tracks_cam_to_ref0_world(
    tracks_xyz_cam: np.ndarray,
    extrinsics_w2c: np.ndarray,
) -> np.ndarray:
    """Camera-space tracks [T, N, 3] -> ref0 world [T, N, 3].

    Extrinsics are world-to-camera; frame 0 is used as the reference.
    """
    tracks_xyz_cam = np.asarray(tracks_xyz_cam, dtype=np.float64)
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float64)
    frame_count = tracks_xyz_cam.shape[0]
    first_inv = np.linalg.inv(extrinsics_w2c[0])
    extrinsics_n = np.asarray([extr @ first_inv for extr in extrinsics_w2c[:frame_count]], dtype=np.float64)
    extrinsics_c2w = np.linalg.inv(extrinsics_n)
    tracks_xyz_world = np.empty_like(tracks_xyz_cam, dtype=np.float64)
    for frame_idx in range(frame_count):
        rot = extrinsics_c2w[frame_idx, :3, :3]
        trans = extrinsics_c2w[frame_idx, :3, 3]
        tracks_xyz_world[frame_idx] = (rot @ tracks_xyz_cam[frame_idx].T).T + trans
    return tracks_xyz_world
