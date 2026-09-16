"""Pipeline-level flow-based scale alignment.

Computes a single global scale ``s`` so that ``s * pred_flow ≈ gt_flow``,
then applies it **in-place** to the prediction stored on ``mv_outputs[0]``.

This ensures both the downstream *visualiser* and *metrics* see the same
aligned predictions, avoiding the previous discrepancy where evaluation
applied scale alignment internally while visualisation displayed raw output.

Dense models (Any4D / VDPM / Track4World)
    scene_flow_pred *= s

Sparse model (Query)
    track_pred = gt_src + s * (track_pred − gt_src)

The alignment method is always least-squares by default:
    s = (pred_flow · gt_flow) / (pred_flow · pred_flow)
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────

def _to_np(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _bilinear_sample(feature_map: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Sample [C, H, W] at [N, 2] pixel coords → [N, C]."""
    C, H, W = feature_map.shape
    x = np.clip(coords[:, 0], 0, W - 1)
    y = np.clip(coords[:, 1], 0, H - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    f00 = feature_map[:, y0, x0].T
    f01 = feature_map[:, y0, x1].T
    f10 = feature_map[:, y1, x0].T
    f11 = feature_map[:, y1, x1].T
    return (1 - wx) * (1 - wy) * f00 + wx * (1 - wy) * f01 + \
           (1 - wx) * wy * f10 + wx * wy * f11


def _least_squares_scale(pred_flat: np.ndarray, gt_flat: np.ndarray) -> float:
    """s = (pred · gt) / (pred · pred), with NaN/zero filtering."""
    finite = np.isfinite(pred_flat) & np.isfinite(gt_flat) & (np.abs(pred_flat) > 1e-8)
    if finite.sum() == 0:
        return 1.0
    pf = pred_flat[finite]
    gf = gt_flat[finite]
    denom = np.dot(pf, pf)
    return float(np.dot(pf, gf) / denom) if denom > 1e-8 else 1.0


# ── dense alignment ─────────────────────────────────────────────────────

def align_dense_scene_flow(mv_outputs) -> Optional[float]:
    """Align ``scene_flow_pred`` on *mv_outputs[0]* using GT trajectories.

    Replicates the same sampling / alignment logic that was previously inside
    ``compute_vdpm_metrics``, but applies the scale **in-place** so that the
    visualiser sees the corrected flow.

    Returns the computed scale, or *None* if alignment could not run.
    """
    out0 = mv_outputs[0]

    sf_raw = getattr(out0, "scene_flow_pred", None)
    trajs_3d = _to_np(getattr(out0, "trajs_3d", None))
    trajs_2d = _to_np(getattr(out0, "trajs_2d", None))
    if sf_raw is None or trajs_3d is None or trajs_2d is None:
        return None

    sf = _to_np(sf_raw)  # (T, 3, H, W)
    H, W = sf.shape[-2:]
    num_views, num_trajs = trajs_3d.shape[:2]

    motion_ext = _to_np(getattr(out0, "motion_extrinsics", None))
    visibs = _to_np(getattr(out0, "trajs_visibs", None))
    valids = _to_np(getattr(out0, "trajs_valids", None))
    origin_h = getattr(out0, "origin_height", None)
    origin_w = getattr(out0, "origin_width", None)

    scale_x = W / origin_w if origin_w and origin_w > 0 else 1.0
    scale_y = H / origin_h if origin_h and origin_h > 0 else 1.0

    ref_idx = 0
    all_pred, all_gt = [], []

    for tgt_idx in range(num_views):
        if tgt_idx == ref_idx or tgt_idx >= sf.shape[0]:
            continue

        gt_flow_world = trajs_3d[tgt_idx] - trajs_3d[ref_idx]  # (N, 3)
        if motion_ext is not None:
            R = motion_ext[ref_idx, :3, :3]
            gt_flow_cam = (R @ gt_flow_world.T).T
        else:
            gt_flow_cam = gt_flow_world

        src_2d = trajs_2d[ref_idx].copy()
        src_2d[:, 0] *= scale_x
        src_2d[:, 1] *= scale_y

        mask = np.ones(num_trajs, dtype=bool)
        if visibs is not None:
            mask &= (visibs[ref_idx] > 0) & (visibs[tgt_idx] > 0)
        if valids is not None:
            mask &= (valids[ref_idx] > 0) & (valids[tgt_idx] > 0)
        mask &= np.isfinite(src_2d).all(axis=1)
        mask &= (src_2d[:, 0] >= 0) & (src_2d[:, 0] < W) & \
                (src_2d[:, 1] >= 0) & (src_2d[:, 1] < H)

        if mask.sum() == 0:
            continue

        sampled = _bilinear_sample(sf[tgt_idx], src_2d[mask])  # (M, 3)
        all_pred.append(sampled)
        all_gt.append(gt_flow_cam[mask])

    if not all_pred:
        return None

    all_pred_cat = np.concatenate(all_pred, axis=0)
    all_gt_cat = np.concatenate(all_gt, axis=0)

    scale = _least_squares_scale(all_pred_cat.flatten(), all_gt_cat.flatten())

    # Apply in-place
    if isinstance(sf_raw, torch.Tensor):
        out0.scene_flow_pred = sf_raw * scale
    else:
        out0.scene_flow_pred = sf_raw * scale

    out0.scene_flow_align_scale = scale
    logger.info("[align_dense] scale=%.6f  (#pairs=%d)", scale, len(all_pred_cat))
    return scale


# ── sparse alignment ────────────────────────────────────────────────────

def align_sparse_motion(mv_outputs) -> Optional[float]:
    """Align ``track_pred`` on *mv_outputs[0]* for the Query model.

    pred_flow = track_pred − gt_src
    gt_flow   = track_gt   − gt_src
    s         = (pred_flow · gt_flow) / (pred_flow · pred_flow)
    track_pred = gt_src + s * pred_flow

    Returns the computed scale, or *None* if alignment could not run.
    """
    out0 = mv_outputs[0]

    pred = _to_np(getattr(out0, "track_pred", None))
    gt = _to_np(getattr(out0, "track_gt", None))
    gt_src = _to_np(getattr(out0, "motion_queries_gt_3d_src", None))
    vis = _to_np(getattr(out0, "track_vis_pred", None))

    if pred is None or gt is None or gt_src is None:
        return None

    # Validity mask (use all valid points for alignment, not just dynamic)
    align_mask = np.ones(pred.shape[0], dtype=bool)
    if vis is not None:
        align_mask &= vis > 0.5

    pred_flow = pred - gt_src
    gt_flow = gt - gt_src

    pf = pred_flow[align_mask].flatten()
    gf = gt_flow[align_mask].flatten()

    scale = _least_squares_scale(pf, gf)

    aligned_pred = gt_src + scale * pred_flow
    out0.track_pred = aligned_pred
    out0.scene_flow_align_scale = scale
    logger.info("[align_sparse] scale=%.6f  (#queries=%d)", scale, int(align_mask.sum()))
    return scale
