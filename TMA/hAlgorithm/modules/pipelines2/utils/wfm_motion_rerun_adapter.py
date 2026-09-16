"""Bridge WFM ``track_3d`` outputs to ``SparseMotionVisualizer3D`` / Rerun.

``SparseMotionVisualizer3D`` expects ``mv_outputs[ref_flat].track_pred`` and
aligned ``motion_queries_*`` fields.  WFM stores per-target tracks under
``track_3d`` (``Track3DOutput``) keyed by the flattened ``mv_outputs`` index of
the target view — the same convention as ``vis_motion_results``.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _to_numpy(x: Any, min_ndim: int = 1) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    while x.ndim > min_ndim and x.shape[0] == 1:
        x = x[0]
    return np.asarray(x)


def attach_track3d_for_sparse_rerun(
    mv_outputs: List[Any],
    frame_num: int,
    view_num: int,
    *,
    ref_frame: int = 0,
    ref_view: int = 0,
    max_queries: Optional[int] = None,
) -> bool:
    """Populate sparse-motion fields on the reference output for Rerun.

    Uses ``mv_outputs[ref_frame * view_num + ref_view].track_3d`` and writes
    concatenated arrays onto that same object:

    - ``track_pred``: final ``warp3d`` (absolute camera-space, same convention
      as the training head — not displacement).
    - ``motion_queries_uv``, ``motion_queries_tgt_frame``, ``track_vis_pred``
    - ``motion_queries_gt_3d_src`` when ``src_3d_gt`` is present
    - ``track_pred_is_displacement``: ``False``

    Args:
        mv_outputs: List of per-view outputs from ``WFMQueryPipeline.infer``.
        frame_num: Number of temporal frames.
        view_num: Number of views per frame.
        ref_frame: Reference frame index (default 0, matches ``REF_FRAME``).
        ref_view: Reference view index within the frame (default 0).
        max_queries: If set, randomly subsample after concatenation (repro seed
            fixed for stability).

    Returns:
        True if sparse fields were attached, False if nothing to do.
    """
    ref_flat = ref_frame * view_num + ref_view
    if ref_flat < 0 or ref_flat >= len(mv_outputs):
        logger.warning("[WfmRerunAdapter] ref_flat=%d out of range len=%d", ref_flat, len(mv_outputs))
        return False
    ref_out = mv_outputs[ref_flat]
    trd = getattr(ref_out, "track_3d", None)
    if not trd:
        logger.info("[WfmRerunAdapter] no track_3d on ref_flat=%d — skip Rerun adapter", ref_flat)
        return False

    pred_list, uv_list, tf_list, vis_list, src3_list = [], [], [], [], []

    for tgt_key in sorted(trd.keys()):
        tr = trd[tgt_key]
        warp3d = _to_numpy(getattr(tr, "warp3d", None), min_ndim=2)
        quv = _to_numpy(getattr(tr, "query_uv", None), min_ndim=2)
        if warp3d is None or quv is None or warp3d.shape[0] == 0:
            continue
        n = min(warp3d.shape[0], quv.shape[0])
        if n == 0:
            continue
        warp3d = warp3d[:n].astype(np.float64, copy=False)
        quv = quv[:n].astype(np.float64, copy=False)

        tgt_frame = int(tgt_key) // int(view_num)
        tf = np.full((n,), tgt_frame, dtype=np.int64)
        vis = np.ones((n,), dtype=np.float64)

        src3 = _to_numpy(getattr(tr, "src_3d_gt", None), min_ndim=2)
        if src3 is not None and src3.shape[0] >= n:
            src3 = src3[:n].astype(np.float64, copy=False)
        else:
            src3 = None
        
        motion_mask = _to_numpy(getattr(tr, "motion_mask", None), min_ndim=1)
        if motion_mask is not None and src3 is not None:
            static_mask = motion_mask <= 0
            warp3d[static_mask] = src3[static_mask]

        pred_list.append(warp3d)
        uv_list.append(quv)
        tf_list.append(tf)
        vis_list.append(vis)
        src3_list.append(src3)

    if not pred_list:
        logger.info("[WfmRerunAdapter] track_3d present but no valid warp3d rows — skip")
        return False

    track_pred = np.concatenate(pred_list, axis=0)
    motion_queries_uv = np.concatenate(uv_list, axis=0)
    motion_queries_tgt_frame = np.concatenate(tf_list, axis=0)
    track_vis_pred = np.concatenate(vis_list, axis=0)
    if all(s is not None for s in src3_list):
        motion_queries_gt_3d_src = np.concatenate(src3_list, axis=0)
    else:
        motion_queries_gt_3d_src = None

    if max_queries is not None and track_pred.shape[0] > max_queries:
        rng = np.random.RandomState(0)
        sel = rng.choice(track_pred.shape[0], size=max_queries, replace=False)
        track_pred = track_pred[sel]
        motion_queries_uv = motion_queries_uv[sel]
        motion_queries_tgt_frame = motion_queries_tgt_frame[sel]
        track_vis_pred = track_vis_pred[sel]
        if motion_queries_gt_3d_src is not None:
            motion_queries_gt_3d_src = motion_queries_gt_3d_src[sel]

    ref_out.track_pred = track_pred
    ref_out.motion_queries_uv = motion_queries_uv
    ref_out.motion_queries_tgt_frame = motion_queries_tgt_frame
    ref_out.track_vis_pred = track_vis_pred
    ref_out.track_pred_is_displacement = False
    if motion_queries_gt_3d_src is not None:
        ref_out.motion_queries_gt_3d_src = motion_queries_gt_3d_src
    logger.info(
        "[WfmRerunAdapter] attached %d queries from %d track_3d pairs on ref_flat=%d",
        track_pred.shape[0],
        len(pred_list),
        ref_flat,
    )
    return True
