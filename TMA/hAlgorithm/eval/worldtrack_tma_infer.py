"""TMA inference helpers for WorldTrack-style evaluation.

Coordinate convention (authoritative for WorldTrack eval)
---------------------------------------------------------
``warp3d`` on pair ``(src=0, tgt=t)`` is the predicted 3D position expressed in the
**frame-0 camera / reference coordinate system** (same as ``get_motion_inputs``:
"Align GT trajectory coords to first camera", and the same ref0 frame used by
Open-d4rt ``pred_ref["xyz_3d"]`` and JSON ``gt_tracks_world``).

Pipeline steps inside ``infer``:

    1. Model outputs ``warp3d`` (normalized, then ``× pair_src_scale`` for metric).
    2. ``pair_tgt_trajs_3d`` used for training loss lives in this frame-0 frame.
    3. UV debug projection uses ``w2c_tgt @ warp3d`` because points are stored in
       ref frame while ``extrinsics`` in the batch are relative w2c — not because
       ``warp3d`` is in target-native camera coordinates.

WorldTrack eval conversion:

    ``pred_tracks_ref0[t, q] = warp3d[t, q]`` for ``t > 0`` (direct, no extra ref0 transform).

    ``pred_tracks_ref0[0, q]`` — model ``src_points`` on any ``(0, t!=0)`` track, else
    ``warp3d`` on ``(0, 0)``. Eval never derives points via ``warp3d - warp3d_delta``,
    pointmap sampling, or GT.

    Then ``metrics_for_sequence`` applies **global median scale** vs JSON GT, identical
    to Open-d4rt.

Prediction source (configurable)
--------------------------------
``--pred-3d-source`` / ``PRED_3D_SOURCE`` selects how each ``(src=0, tgt=t)`` 3D point is read:

- ``warp3d`` (default): model ``Track3DOutput.warp3d`` (absolute position in ref0).
- ``warp3d_delta``: ``src_points + warp3d_delta`` only (no GT); raises if either is missing.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch

Pred3DSource = Literal["warp3d", "warp3d_delta"]
PRED_3D_SOURCE_WARP3D: Pred3DSource = "warp3d"
PRED_3D_SOURCE_WARP3D_DELTA: Pred3DSource = "warp3d_delta"

from hAlgorithm.eval.worldtrack_json_loader import WorldTrackSequence, normalize_uv_to_01
from hAlgorithm.modules.metrics.worldtrack_eval_metrics import (
    compute_scale_factor_global,
    tracks_cam_to_ref0_world,
)


def _to_np(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_pred_3d_source(value: str) -> Pred3DSource:
    """Normalize CLI / shell config to a supported prediction source."""
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "warp3d": PRED_3D_SOURCE_WARP3D,
        "absolute": PRED_3D_SOURCE_WARP3D,
        "abs": PRED_3D_SOURCE_WARP3D,
        "warp3d_delta": PRED_3D_SOURCE_WARP3D_DELTA,
        "delta": PRED_3D_SOURCE_WARP3D_DELTA,
        "src_points_warp3d_delta": PRED_3D_SOURCE_WARP3D_DELTA,
        "src_points+warp3d_delta": PRED_3D_SOURCE_WARP3D_DELTA,
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown pred_3d_source {value!r}; use 'warp3d' or 'warp3d_delta' "
            f"(aliases: absolute, delta, src_points+warp3d_delta)."
        )
    return aliases[key]


def resolve_pred_3d_source(
    *,
    pred_3d_source: Union[str, Pred3DSource, None] = None,
    use_warp3d_delta: bool = False,
) -> Pred3DSource:
    """Resolve eval prediction source; ``use_warp3d_delta=True`` selects delta mode."""
    if use_warp3d_delta:
        return PRED_3D_SOURCE_WARP3D_DELTA
    if pred_3d_source is None:
        return PRED_3D_SOURCE_WARP3D
    return normalize_pred_3d_source(str(pred_3d_source))


def pred_field_label(pred_3d_source: Pred3DSource) -> str:
    if pred_3d_source == PRED_3D_SOURCE_WARP3D_DELTA:
        return "warp3d_delta+src_points"
    return "warp3d"


def use_warp3d_delta_from_source(pred_3d_source: Pred3DSource) -> bool:
    return pred_3d_source == PRED_3D_SOURCE_WARP3D_DELTA


def ref0_to_cam_tracks(
    points_ref0_tq3: np.ndarray,
    extrinsics_w2c: np.ndarray,
) -> np.ndarray:
    """Frame-0 ref coordinates → per-frame native camera (for sanity vs JSON ``trajs_3d``).

    ``X_cam_t = w2c_t @ c2w_0 @ X_ref0`` with ``c2w_0 = inv(w2c_0)``.
    """
    pts = np.asarray(points_ref0_tq3, dtype=np.float64)
    ext = np.asarray(extrinsics_w2c, dtype=np.float64)
    c2w0 = np.linalg.inv(ext[0])
    out = np.full_like(pts, np.nan, dtype=np.float64)
    for t in range(pts.shape[0]):
        w2c_t = ext[t]
        p = pts[t]
        fin = np.isfinite(p).all(axis=-1)
        if not np.any(fin):
            continue
        ph = np.concatenate([p[fin], np.ones((int(fin.sum()), 1), dtype=np.float64)], axis=-1)
        world = (ph @ c2w0.T)[:, :3]
        wh = np.concatenate([world, np.ones((world.shape[0], 1), dtype=np.float64)], axis=-1)
        out[t, fin] = (wh @ w2c_t.T)[:, :3]
    return out


def _track3d_dict(outputs: Sequence[Any], src_index: int) -> Dict[Any, Any]:
    if src_index < 0 or src_index >= len(outputs):
        return {}
    track_dict = getattr(outputs[src_index], "track_3d", None)
    return track_dict if isinstance(track_dict, dict) else {}


def _find_track_for_tgt(
    outputs: Sequence[Any],
    src_index: int,
    tgt_index: int,
) -> Any:
    track_dict = _track3d_dict(outputs, src_index)
    track = track_dict.get(tgt_index, track_dict.get(int(tgt_index), None))
    if track is not None and (
        getattr(track, "src_index", None) == src_index
        and getattr(track, "tgt_index", None) == tgt_index
    ):
        return track
    for val in track_dict.values():
        if (
            getattr(val, "src_index", None) == src_index
            and getattr(val, "tgt_index", None) == tgt_index
        ):
            return val
    return None


def _track_pair_label(track: Any) -> str:
    src = getattr(track, "src_index", None)
    tgt = getattr(track, "tgt_index", None)
    if src is not None and tgt is not None:
        return f"(src={src}, tgt={tgt})"
    return "unknown pair"


def _src_points_field_from_track(track: Any) -> Optional[np.ndarray]:
    """``Track3DOutput.src_points`` only (model field; no eval-side derivation)."""
    src_points = _to_np(getattr(track, "src_points", None))
    if src_points is None:
        return None
    return np.asarray(src_points, dtype=np.float64)


def _frame0_pred_positions(
    outputs: Sequence[Any],
    *,
    src_index: int = 0,
) -> np.ndarray:
    """Frame-0 ref0 positions for WorldTrack eval (prediction only).

    1. ``src_points`` on any ``(src, tgt!=src)`` track (e.g. ``(0, 1)``).
    2. Else ``warp3d`` on the ``(src, src)`` pair ``(0, 0)``.

    Forbidden in eval: ``warp3d - warp3d_delta``, GT ``src_3d_gt``, pointmap sampling.
    """
    track_dict = _track3d_dict(outputs, src_index)
    for tgt_key in sorted(track_dict.keys()):
        if int(tgt_key) == int(src_index):
            continue
        src_points = _src_points_field_from_track(track_dict[tgt_key])
        if src_points is not None:
            return src_points

    track_00 = _find_track_for_tgt(outputs, src_index, src_index)
    if track_00 is not None:
        warp3d_00 = _to_np(getattr(track_00, "warp3d", None))
        if warp3d_00 is not None:
            return np.asarray(warp3d_00, dtype=np.float64)

    raise RuntimeError(
        f"WorldTrack eval: frame-0 needs model src_points on (src={src_index}, tgt!=src) "
        f"or warp3d on ({src_index}, {src_index}); got track keys {sorted(track_dict.keys())}."
    )


def compose_warp3d_from_track(
    track: Any,
    *,
    pred_3d_source: Pred3DSource = PRED_3D_SOURCE_WARP3D,
    frame0_src: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Read ref0 3D position for one ``(src=0, tgt=t)`` track (prediction only, no GT)."""
    pair = _track_pair_label(track)
    if pred_3d_source == PRED_3D_SOURCE_WARP3D_DELTA:
        warp3d_delta = _to_np(getattr(track, "warp3d_delta", None))
        if warp3d_delta is None:
            raise RuntimeError(
                f"WorldTrack eval {pair}: missing model warp3d_delta "
                f"(pred_3d_source={pred_3d_source!r})."
            )
        src_points = _src_points_field_from_track(track)
        if src_points is None:
            if frame0_src is None:
                raise RuntimeError(
                    f"WorldTrack eval {pair}: missing src_points and no frame-0 src "
                    f"for warp3d_delta eval."
                )
            src_points = np.asarray(frame0_src, dtype=np.float64)
        return src_points + np.asarray(warp3d_delta, dtype=np.float64)

    warp3d = _to_np(getattr(track, "warp3d", None))
    if warp3d is None:
        raise RuntimeError(
            f"WorldTrack eval {pair}: missing model warp3d "
            f"(pred_3d_source={pred_3d_source!r})."
        )
    return np.asarray(warp3d, dtype=np.float64)


def _warp3d_from_track(
    track: Any,
    *,
    use_warp3d_delta: bool = False,
    pred_3d_source: Optional[Pred3DSource] = None,
) -> Optional[np.ndarray]:
    """Backward-compatible wrapper around :func:`compose_warp3d_from_track`."""
    source = (
        pred_3d_source
        if pred_3d_source is not None
        else resolve_pred_3d_source(use_warp3d_delta=use_warp3d_delta)
    )
    return compose_warp3d_from_track(track, pred_3d_source=source)


def extract_pred_tracks_ref0_from_outputs(
    outputs: Sequence[Any],
    *,
    query_indices: np.ndarray,
    num_frames: int,
    global_scale: Optional[float],
    extrinsics_w2c_metric: np.ndarray,
    use_warp3d_delta: bool = False,
    pred_3d_source: Optional[Pred3DSource] = None,
    src_index: int = 0,
) -> np.ndarray:
    """Stack per-frame ref0 3D as ``[T, Q, 3]`` (= WorldTrack / Open-d4rt ref0)."""
    del global_scale, extrinsics_w2c_metric
    source = resolve_pred_3d_source(
        pred_3d_source=pred_3d_source, use_warp3d_delta=use_warp3d_delta
    )
    q_count = int(np.asarray(query_indices).shape[0])
    pred = np.full((int(num_frames), q_count, 3), np.nan, dtype=np.float64)
    if not outputs:
        return pred

    frame0_src = _frame0_pred_positions(outputs, src_index=src_index)

    for t in range(int(num_frames)):
        if t == int(src_index):
            pts = frame0_src
        else:
            track = _find_track_for_tgt(outputs, src_index, t)
            if track is None:
                raise RuntimeError(
                    f"WorldTrack eval: missing model track (src={src_index}, tgt={t})."
                )
            pts = compose_warp3d_from_track(
                track, pred_3d_source=source, frame0_src=frame0_src
            )
        n = min(int(pts.shape[0]), q_count)
        pred[t, :n] = np.asarray(pts[:n], dtype=np.float64)
    return pred


def extract_pred_tracks_cam_from_outputs(
    outputs: Sequence[Any],
    *,
    num_frames: int,
    q_count: int,
    extrinsics_w2c_metric: np.ndarray,
    use_warp3d_delta: bool = False,
    pred_3d_source: Optional[Pred3DSource] = None,
    src_index: int = 0,
) -> np.ndarray:
    """Frame-0 ref0 tracks → native camera ``t`` (sanity vs JSON ``trajs_3d`` only)."""
    pred_ref0 = extract_pred_tracks_ref0_from_outputs(
        outputs,
        query_indices=np.arange(q_count, dtype=np.int64),
        num_frames=num_frames,
        global_scale=None,
        extrinsics_w2c_metric=extrinsics_w2c_metric,
        use_warp3d_delta=use_warp3d_delta,
        pred_3d_source=pred_3d_source,
        src_index=src_index,
    )
    return ref0_to_cam_tracks(pred_ref0, extrinsics_w2c_metric)


def extract_tma_internal_pred_cam(
    outputs: Sequence[Any],
    *,
    num_frames: int,
    q_count: int,
    extrinsics_w2c_metric: np.ndarray,
    use_warp3d_delta: bool = False,
    pred_3d_source: Optional[Pred3DSource] = None,
    src_index: int = 0,
) -> np.ndarray:
    return extract_pred_tracks_cam_from_outputs(
        outputs,
        num_frames=num_frames,
        q_count=q_count,
        extrinsics_w2c_metric=extrinsics_w2c_metric,
        use_warp3d_delta=use_warp3d_delta,
        pred_3d_source=pred_3d_source,
        src_index=src_index,
    )


def extract_tma_gt_cam_from_outputs(
    outputs: Sequence[Any],
    *,
    num_frames: int,
    q_count: int,
    src_index: int = 0,
) -> np.ndarray:
    """``tgt_3d_gt`` (training frame-0-aligned GT inside infer; debug only)."""
    gt = np.full((int(num_frames), int(q_count), 3), np.nan, dtype=np.float64)
    for t in range(int(num_frames)):
        track = _find_track_for_tgt(outputs, src_index, t)
        if track is None:
            continue
        tgt = _to_np(getattr(track, "tgt_3d_gt", None))
        if tgt is None:
            continue
        n = min(int(tgt.shape[0]), int(q_count))
        gt[t, :n] = tgt[:n]
    return gt


def _resolve_inner_model(model: Any) -> Any:
    inner = getattr(model, "model", model)
    if hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod
    return inner


@contextmanager
def worldtrack_query_bank_override(model: Any, batch: dict) -> Iterator[None]:
    """Sparse queries at frame-0 UVs (full grid if no query bank).

  Prefers ``batch["worldtrack_query_uv"]`` (from ``align_batch_queries_with_sequence``,
  same projection + bounds gate as Scene Flow). Falls back to collated ``trajs_2d``.
    """
    inner = _resolve_inner_model(model)
    query_bank = getattr(inner, "query_banck", None)
    if query_bank is None:
        yield
        return

    uv_np: Optional[np.ndarray] = None
    wq = batch.get("worldtrack_query_uv")
    if wq is not None:
        uv_np = np.asarray(_to_np(wq), dtype=np.float64)
        if uv_np.ndim == 3:
            uv_np = uv_np[0]
        if uv_np.ndim != 2 or uv_np.shape[-1] != 2:
            raise ValueError(f"worldtrack_query_uv must be [Q,2], got {uv_np.shape}")
        num_frames = int(batch.get("worldtrack_num_frames", 0))
        if num_frames <= 0:
            trajs_2d = batch.get("trajs_2d")
            if trajs_2d is not None:
                t2d = trajs_2d[0] if isinstance(trajs_2d, (list, tuple)) else trajs_2d
                num_frames = int(t2d.shape[1])
            else:
                img = batch.get("image") or batch.get("image_show")
                if img is not None and hasattr(img, "shape"):
                    num_frames = int(img.shape[1]) if int(getattr(img, "ndim", 0)) >= 5 else 1
        if num_frames <= 0:
            num_frames = 1
    else:
        trajs_2d = batch.get("trajs_2d")
        if trajs_2d is None:
            yield
            return
        if isinstance(trajs_2d, (list, tuple)):
            trajs_2d = trajs_2d[0]
        num_frames = int(trajs_2d.shape[1])
        uv_np = _to_np(trajs_2d[0, 0]).astype(np.float64, copy=False)

    uv = torch.from_numpy(uv_np).float()
    meta = batch.get("meta_data", {})
    ow = int(meta.get("origin_width", meta.get("input_width", [[uv.shape[-1]]])[0]))
    oh = int(meta.get("origin_height", meta.get("input_height", [[uv.shape[-2]]])[0]))
    iw = int(meta.get("input_width", [[ow]])[0])
    ih = int(meta.get("input_height", [[oh]])[0])

    orig_forward = query_bank.forward

    def _forward(image, edge_mask=None, meta_data=None, **kwargs):
        from hAlgorithm.modules.models2.query_bank.query import BaseQuery

        uv_norm = uv.to(device=image.device, dtype=image.dtype).clone()
        uv_norm[..., 0] /= max(float(ow - 1), 1.0)
        uv_norm[..., 1] /= max(float(oh - 1), 1.0)
        # Do not clamp: queries are pre-gated to in-bounds projected UV (Open-d4rt / Scene Flow).
        uv_norm = uv_norm.unsqueeze(0).unsqueeze(0).expand(1, num_frames, -1, -1).contiguous()
        return BaseQuery(uv=uv_norm, full_uv=False, width=iw, height=ih)

    query_bank.forward = _forward
    old_chunk = getattr(inner, "chunk_size", None)
    if old_chunk is not None:
        inner.chunk_size = min(int(old_chunk), 2048)
    try:
        yield
    finally:
        query_bank.forward = orig_forward
        if old_chunk is not None:
            inner.chunk_size = old_chunk


def ref0_space_epe(pred_ref0: np.ndarray, gt_ref0: np.ndarray) -> float:
    """EPE in frame-0 ref / WorldTrack ref0 (before global scale in metrics)."""
    fin = np.isfinite(pred_ref0) & np.isfinite(gt_ref0)
    if not np.any(fin):
        return float("nan")
    return float(np.linalg.norm(pred_ref0[fin] - gt_ref0[fin], axis=-1).mean())


def cam_space_epe_vs_json(pred_cam: np.ndarray, gt_cam: np.ndarray) -> float:
    fin = np.isfinite(pred_cam) & np.isfinite(gt_cam)
    if not np.any(fin):
        return float("nan")
    return float(np.linalg.norm(pred_cam[fin] - gt_cam[fin], axis=-1).mean())


def summarize_pred_alignment(
    gt_tracks_ref0: np.ndarray,
    pred_tracks_ref0: np.ndarray,
) -> Dict[str, float]:
    scale = compute_scale_factor_global(gt_tracks_ref0, pred_tracks_ref0)
    pred_aligned = pred_tracks_ref0 * scale
    fin = np.isfinite(gt_tracks_ref0) & np.isfinite(pred_aligned)
    epe = (
        float(np.linalg.norm(gt_tracks_ref0[fin] - pred_aligned[fin], axis=-1).mean())
        if np.any(fin)
        else float("nan")
    )
    return {"scale_global": float(scale), "epe_after_scale_global": epe}


def extract_tma_gt_tracks_ref0_from_outputs(
    outputs: Sequence[Any],
    *,
    query_indices: np.ndarray,
    num_frames: int,
    global_scale: Optional[float],
    extrinsics_w2c_metric: np.ndarray,
    src_index: int = 0,
) -> np.ndarray:
    del global_scale
    q_count = int(query_indices.shape[0])
    gt_cam = extract_tma_gt_cam_from_outputs(
        outputs, num_frames=num_frames, q_count=q_count, src_index=src_index
    )
    return tracks_cam_to_ref0_world(gt_cam, extrinsics_w2c_metric)


def get_global_scale_from_batch(batch: Dict[str, Any]) -> Optional[float]:
    scale = batch.get("scale", None)
    if scale is None:
        return None
    s = _to_np(scale)
    if s is None:
        return None
    return float(np.nanmean(s))


def get_metric_extrinsics_from_batch(
    batch: Dict[str, Any],
    num_frames: int,
) -> np.ndarray:
    ext = batch.get("extrinsics", None)
    if ext is None:
        raise ValueError("batch has no 'extrinsics'")
    e = _to_np(ext)[0]
    scale = get_global_scale_from_batch(batch)
    if scale is not None and np.isfinite(scale) and scale > 0:
        e = e.copy()
        e[:, :3, 3] *= scale
    return e[: int(num_frames)]


def get_metric_extrinsics_for_sequence(
    seq: WorldTrackSequence,
    num_frames: int,
) -> np.ndarray:
    return np.asarray(seq.extrinsics_w2c[: int(num_frames)], dtype=np.float64)


def resize_video_for_model(
    video_rgb: np.ndarray,
    image_hw: Tuple[int, int],
) -> np.ndarray:
    import cv2

    h, w = int(image_hw[0]), int(image_hw[1])
    out = []
    for frame in video_rgb:
        interp = cv2.INTER_AREA if frame.shape[0] >= h else cv2.INTER_LINEAR
        out.append(cv2.resize(frame, (w, h), interpolation=interp))
    return np.stack(out, axis=0)


def remap_query_uv_norm_after_resize(
    query_uv_norm: np.ndarray,
    orig_hw: Tuple[int, int],
    model_hw: Tuple[int, int],
) -> np.ndarray:
    oh, ow = int(orig_hw[0]), int(orig_hw[1])
    mh, mw = int(model_hw[0]), int(model_hw[1])
    uv = np.asarray(query_uv_norm, dtype=np.float64).copy()
    uv[:, 0] *= float(ow - 1)
    uv[:, 1] *= float(oh - 1)
    sx = float(mw - 1) / float(max(ow - 1, 1))
    sy = float(mh - 1) / float(max(oh - 1, 1))
    uv[:, 0] *= sx
    uv[:, 1] *= sy
    return normalize_uv_to_01(uv, (mh, mw))
