import logging
import os
from collections import deque

import cv2
import numpy as np
import torch
import colorsys

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:
    rr = None
    rrb = None


logger = logging.getLogger(__name__)
_FRUSTUM_GT_COLOR = np.array([60, 220, 80, 230], dtype=np.uint8)
_FRUSTUM_PRED_COLOR = np.array([255, 140, 60, 230], dtype=np.uint8)


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _sample_query_rgb(ref_out, query_uv, n):
    rgb = _to_numpy(getattr(ref_out, "rgb", None))
    if rgb is None or query_uv is None or n <= 0:
        return np.full((n, 3), 200, dtype=np.uint8)

    quv = np.asarray(_to_numpy(query_uv), dtype=np.float64).reshape(-1, 2)[:n]
    h, w = rgb.shape[:2]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    px = np.clip(
        np.round(quv * np.array([w - 1, h - 1], dtype=np.float64)).astype(np.int64),
        [0, 0],
        [w - 1, h - 1],
    )
    return rgb[px[:, 1], px[:, 0]].astype(np.uint8)


def _collect_track_rows(track, ref_out):
    # For t1 arrows, prefer frame-0 src points as the origin baseline.
    # Fall back to src_3d_gt only when src_points is unavailable.
    src = _to_numpy(getattr(track, "src_points", None))
    if src is None:
        src = _to_numpy(getattr(track, "src_3d_gt", None))
    pred = _to_numpy(getattr(track, "warp3d", None))
    delta = _to_numpy(getattr(track, "warp3d_delta", None))
    gt_tgt = _to_numpy(getattr(track, "tgt_3d_gt", None))
    if src is None or pred is None:
        return None, None, None, None, None, None, None

    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    pred_from_delta = None
    if delta is not None:
        delta = np.asarray(delta, dtype=np.float64).reshape(-1, 3)
        pred_from_delta = src + delta
    n = min(src.shape[0], pred.shape[0])
    if pred_from_delta is not None:
        n = min(n, pred_from_delta.shape[0])
    if n <= 0:
        return None, None, None, None, None, None, None

    src = src[:n]
    pred = pred[:n]
    if pred_from_delta is not None:
        pred_from_delta = pred_from_delta[:n]
    if gt_tgt is not None:
        gt_tgt = np.asarray(gt_tgt, dtype=np.float64).reshape(-1, 3)[:n]

    finite = np.isfinite(src).all(axis=1) & np.isfinite(pred).all(axis=1)
    if pred_from_delta is not None:
        finite &= np.isfinite(pred_from_delta).all(axis=1)
    if gt_tgt is not None:
        finite &= np.isfinite(gt_tgt).all(axis=1)
    valid = finite
    if not valid.any():
        return None, None, None, None, None, None, None

    query_uv = _to_numpy(getattr(track, "query_uv", None))
    query_uv_valid = None
    if query_uv is not None:
        query_uv = np.asarray(query_uv, dtype=np.float64).reshape(-1, 2)[:n]
        query_uv_valid = query_uv[valid]
    colors = _sample_query_rgb(ref_out, query_uv, n)
    valid_indices = np.flatnonzero(valid).astype(np.int64)
    return (
        src[valid],
        pred[valid],
        (pred_from_delta[valid] if pred_from_delta is not None else None),
        (gt_tgt[valid] if gt_tgt is not None else None),
        colors[valid],
        query_uv_valid,
        valid_indices,
    )


def _direction_colors(vectors):
    """Map 3D direction to RGB colors.

    Unit direction ``[-1, 1]`` is linearly mapped to ``[0, 255]`` per axis.
    Near-zero vectors fall back to neutral gray.
    """
    if vectors is None or len(vectors) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    v = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    mag = np.linalg.norm(v, axis=1, keepdims=True)
    unit = np.zeros_like(v)
    nz = mag[:, 0] > 1e-8
    if np.any(nz):
        unit[nz] = v[nz] / mag[nz]
    rgb = np.clip((unit + 1.0) * 0.5 * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb[~nz] = np.array([170, 170, 170], dtype=np.uint8)
    return rgb


def _index_colors(n):
    """Assign pseudo-random but stable RGB colors per row index."""
    n = int(n)
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if n == 1:
        return np.array([[255, 80, 80]], dtype=np.uint8)
    # Fixed seed ensures same row index gets the same color across frames.
    rng = np.random.default_rng(seed=20260522)
    return rng.integers(0, 256, size=(n, 3), dtype=np.uint8)


def _x_position_rainbow_colors(
    points_xyz,
    *,
    use_percentile_clip=True,
    clip_percentiles=(1.0, 99.0),
    reverse=False,
):
    """Map x position to rainbow RGB colors."""
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)

    x = pts[:, 0]
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.full((n, 3), 170, dtype=np.uint8)

    x_finite = x[finite]
    if use_percentile_clip:
        lo, hi = np.percentile(x_finite, clip_percentiles)
    else:
        lo, hi = np.min(x_finite), np.max(x_finite)

    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        t = np.full(n, 0.5, dtype=np.float64)
    else:
        t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    if reverse:
        t = 1.0 - t

    # Hue from violet(0.75) to red(0.0): common rainbow-like ordering.
    h = 0.75 * (1.0 - t)
    s = np.ones(n, dtype=np.float64)
    v = np.ones(n, dtype=np.float64)

    rgb = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        if not np.isfinite(x[i]):
            rgb[i] = np.array([170, 170, 170], dtype=np.uint8)
            continue
        r, g, b = colorsys.hsv_to_rgb(float(h[i]), float(s[i]), float(v[i]))
        rgb[i] = np.array([int(r * 255.0), int(g * 255.0), int(b * 255.0)], dtype=np.uint8)
    return rgb


def _xy_position_rainbow_colors(
    points_xyz,
    *,
    use_percentile_clip=True,
    clip_percentiles=(1.0, 99.0),
    reverse=False,
    y_hue_offset_strength=0.04,
):
    """Map x to rainbow hue, then add a small y-based hue offset."""
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)

    x = pts[:, 0]
    y = pts[:, 1]
    finite_x = np.isfinite(x)
    if not np.any(finite_x):
        return np.full((n, 3), 170, dtype=np.uint8)

    x_finite = x[finite_x]
    if use_percentile_clip:
        x_lo, x_hi = np.percentile(x_finite, clip_percentiles)
    else:
        x_lo, x_hi = np.min(x_finite), np.max(x_finite)

    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or (x_hi - x_lo) < 1e-12:
        tx = np.full(n, 0.5, dtype=np.float64)
    else:
        tx = np.clip((x - x_lo) / (x_hi - x_lo), 0.0, 1.0)

    if reverse:
        tx = 1.0 - tx

    base_h = 0.75 * (1.0 - tx)

    # y controls a small hue perturbation around x-rainbow base hue.
    finite_y = np.isfinite(y)
    if np.any(finite_y):
        y_finite = y[finite_y]
        if use_percentile_clip:
            y_lo, y_hi = np.percentile(y_finite, clip_percentiles)
        else:
            y_lo, y_hi = np.min(y_finite), np.max(y_finite)
        if np.isfinite(y_lo) and np.isfinite(y_hi) and (y_hi - y_lo) >= 1e-12:
            ty = np.clip((y - y_lo) / (y_hi - y_lo), 0.0, 1.0)
            y_offset = (ty - 0.5) * 2.0 * float(y_hue_offset_strength)
        else:
            y_offset = np.zeros(n, dtype=np.float64)
    else:
        y_offset = np.zeros(n, dtype=np.float64)

    h = np.mod(base_h + y_offset, 1.0)
    s = np.ones(n, dtype=np.float64)
    v = np.ones(n, dtype=np.float64)

    rgb = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        if not np.isfinite(x[i]):
            rgb[i] = np.array([170, 170, 170], dtype=np.uint8)
            continue
        r, g, b = colorsys.hsv_to_rgb(float(h[i]), float(s[i]), float(v[i]))
        rgb[i] = np.array([int(r * 255.0), int(g * 255.0), int(b * 255.0)], dtype=np.uint8)
    return rgb


def _arrow_colors_by_cfg(origin, pred, cfg):
    """Get arrow colors from config; default maps x-position to rainbow."""
    cfg = cfg or {}
    color_mode = str(cfg.get("arrow_color_mode", "x_rainbow_y_offset")).strip().lower()
    x_source = str(cfg.get("arrow_color_x_source", "origin")).strip().lower()
    x_points = pred if x_source == "pred" else origin
    y_source = str(cfg.get("arrow_color_y_source", x_source)).strip().lower()
    y_points = pred if y_source == "pred" else origin

    if color_mode == "index":
        return _index_colors(origin.shape[0])
    if color_mode == "direction":
        return _direction_colors(pred - origin)
    if color_mode == "x_rainbow":
        return _x_position_rainbow_colors(
            x_points,
            use_percentile_clip=bool(cfg.get("arrow_color_x_clip", True)),
            clip_percentiles=tuple(cfg.get("arrow_color_x_clip_percentiles", (1.0, 99.0))),
            reverse=bool(cfg.get("arrow_color_x_reverse", False)),
        )
    if color_mode == "x_rainbow_y_offset":
        points_for_xy = np.asarray(x_points, dtype=np.float64).reshape(-1, 3).copy()
        points_for_xy[:, 1] = np.asarray(y_points, dtype=np.float64).reshape(-1, 3)[:, 1]
        return _xy_position_rainbow_colors(
            points_for_xy,
            use_percentile_clip=bool(cfg.get("arrow_color_x_clip", True)),
            clip_percentiles=tuple(cfg.get("arrow_color_x_clip_percentiles", (1.0, 99.0))),
            reverse=bool(cfg.get("arrow_color_x_reverse", False)),
            y_hue_offset_strength=float(cfg.get("arrow_color_y_offset_strength", 0.15)),
        )
    return _index_colors(origin.shape[0])


def _enforce_min_arrow_length(vectors, min_length):
    """Enforce a minimum visual arrow length without changing endpoints data."""
    v = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    if v.shape[0] == 0:
        return v
    min_length = float(min_length)
    if min_length <= 0:
        return v

    mag = np.linalg.norm(v, axis=1, keepdims=True)
    nz = mag[:, 0] > 1e-8

    out = v.copy()
    short = mag[:, 0] < min_length
    if np.any(short & nz):
        scale = (min_length / np.clip(mag[short & nz], 1e-8, None))
        out[short & nz] = out[short & nz] * scale
    if np.any(short & ~nz):
        out[short & ~nz] = np.array([0.0, 0.0, min_length], dtype=np.float64)
    return out


def _to_line_segments(origins, vectors):
    """Convert origin+vector arrows to line-strip segments."""
    o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    v = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    if o.shape[0] == 0:
        return np.zeros((0, 2, 3), dtype=np.float64)
    return np.stack([o, o + v], axis=1)


def _frame0_gt_points_and_colors(ref_out):
    pmap = _to_numpy(getattr(ref_out, "pointmap_gt_global", None))
    if pmap is None:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    pts_map = np.asarray(pmap, dtype=np.float64)
    pts = pts_map.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)

    h, w = ref_out.pointmap_gt_h, ref_out.pointmap_gt_w

    rgb = _to_numpy(getattr(ref_out, "rgb", None))

    if rgb is not None:
        rgb = np.asarray(rgb)
        cols = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        cols = (np.clip(cols, 0.0, 1.0) * 255.0).reshape(-1, 3).astype(np.uint8)
    else:
        cols = np.full((h * w, 3), 180, dtype=np.uint8)
    return pts[valid], cols[valid]


def _camera_hw(out):
    h = int(getattr(out, "pointmap_h", 0) or 0)
    w = int(getattr(out, "pointmap_w", 0) or 0)
    if h > 0 and w > 0:
        return h, w
    rgb = _to_numpy(getattr(out, "rgb", None))
    if rgb is not None and rgb.ndim >= 2:
        return int(rgb.shape[0]), int(rgb.shape[1])
    return 1, 1


def _stride_sample_points(points, colors, step):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    s = max(int(step), 1)
    if s <= 1 or pts.shape[0] <= 1:
        return pts, cols
    return pts[::s], cols[::s]


def _frustum_lines(c2w, k, h, w, scale=0.15):
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    d = float(scale)
    corners_cam = np.array(
        [
            [-cx / fx * d, -cy / fy * d, d],
            [(w - cx) / fx * d, -cy / fy * d, d],
            [(w - cx) / fx * d, (h - cy) / fy * d, d],
            [-cx / fx * d, (h - cy) / fy * d, d],
        ],
        dtype=np.float64,
    )
    r = c2w[:3, :3]
    t = c2w[:3, 3]
    cw = (r @ corners_cam.T).T + t
    center = t.copy()
    segs = np.array(
        [
            [center, cw[0]],
            [center, cw[1]],
            [center, cw[2]],
            [center, cw[3]],
            [cw[0], cw[1]],
            [cw[1], cw[2]],
            [cw[2], cw[3]],
            [cw[3], cw[0]],
        ],
        dtype=np.float64,
    )
    return center, segs


def _cfg_bool(cfg, *keys, default=False):
    for key in keys:
        if key in cfg:
            return bool(cfg.get(key))
    return default


def _cfg_int(cfg, *keys, default=1, min_value=1):
    for key in keys:
        if key in cfg:
            try:
                value = int(cfg.get(key))
            except (TypeError, ValueError):
                continue
            return max(value, min_value)
    return max(int(default), min_value)


def _log_camera_frustums(
    step_idx,
    mv_outputs,
    cfg,
    show_dynamic=False,
    show_dynamic_delta=True,
    show_dynamic_pred=False,
    show_dynamic_pred_delta=False,
):
    if rr is None or step_idx < 0 or step_idx >= len(mv_outputs):
        return
    out = mv_outputs[step_idx]
    h, w = _camera_hw(out)
    frustum_scale = float(cfg.get("frustum_scale", cfg.get("vis_3d_frustum_scale", 0.15)))
    line_radius = float(cfg.get("frustum_line_radius", 0.003))

    def _log_one(ext_name, k_name, path, color, label):
        ext = _to_numpy(getattr(out, ext_name, None))
        k = _to_numpy(getattr(out, k_name, None))
        if ext is None or k is None:
            return
        ext = np.asarray(ext, dtype=np.float64).reshape(4, 4)
        k = np.asarray(k, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(ext).all() or not np.isfinite(k).all():
            return
        try:
            c2w = np.linalg.inv(ext)
        except np.linalg.LinAlgError:
            return
        center, segs = _frustum_lines(c2w, k, h, w, scale=frustum_scale)
        rr.log(path, rr.LineStrips3D(segs, colors=color, radii=line_radius))
        rr.log(
            f"{path}_center",
            rr.Points3D([center], colors=[color[:3]], labels=[f"{step_idx}"], radii=0.01),
        )

    gt_path = f"dynamic/cameras/gt_frustum/step_{step_idx:03d}"
    pred_path = f"dynamic/cameras/pred_frustum/step_{step_idx:03d}"
    gt_path_delta = f"dynamic_delta/cameras/gt_frustum/step_{step_idx:03d}"
    pred_path_delta = f"dynamic_delta/cameras/pred_frustum/step_{step_idx:03d}"
    pred_path_right = f"dynamic_pred/cameras/pred_frustum/step_{step_idx:03d}"
    pred_path_right_delta = f"dynamic_pred_delta/cameras/pred_frustum/step_{step_idx:03d}"
    if show_dynamic:
        _log_one("extrinsics", "intrinsics", gt_path, _FRUSTUM_GT_COLOR, "GT Cam")
        _log_one("extrinsics_pred", "intrinsics_pred", pred_path, _FRUSTUM_PRED_COLOR, "Pred Cam")
    if show_dynamic_delta:
        _log_one("extrinsics", "intrinsics", gt_path_delta, _FRUSTUM_GT_COLOR, "GT Cam")
        _log_one("extrinsics_pred", "intrinsics_pred", pred_path_delta, _FRUSTUM_PRED_COLOR, "Pred Cam")
    if show_dynamic_pred:
        # Right panel (dynamic_pred): predicted camera frustums only, accumulated over time.
        _log_one("extrinsics_pred", "intrinsics_pred", pred_path_right, _FRUSTUM_PRED_COLOR, "Pred Cam")
    if show_dynamic_pred_delta:
        # Right-bottom panel (dynamic_pred_delta): predicted camera frustums only, accumulated over time.
        _log_one("extrinsics_pred", "intrinsics_pred", pred_path_right_delta, _FRUSTUM_PRED_COLOR, "Pred Cam")


def _setup_blueprint(show_dynamic=False, show_dynamic_delta=True, show_dynamic_pred=False, show_dynamic_pred_delta=False):
    if rr is None or rrb is None:
        return
    top_views = []
    if show_dynamic:
        top_views.append(rrb.Spatial3DView(name="Dynamic GT & Pred", origin="dynamic/"))
    if show_dynamic_pred:
        top_views.append(rrb.Spatial3DView(name="Dynamic points Pred", origin="dynamic_pred/"))

    bottom_views = []
    if show_dynamic_delta:
        bottom_views.append(rrb.Spatial3DView(name="Dynamic GT & Pred (delta+src)", origin="dynamic_delta/"))
    if show_dynamic_pred_delta:
        bottom_views.append(
            rrb.Spatial3DView(name="Dynamic points Pred (delta+src)", origin="dynamic_pred_delta/")
        )

    rows = []
    if top_views:
        rows.append(rrb.Horizontal(*top_views))
    if bottom_views:
        rows.append(rrb.Horizontal(*bottom_views))
    if not rows:
        return
    rr.send_blueprint(rrb.Blueprint(rrb.Vertical(*rows), collapse_panels=True))


def _log_view_coordinates(cfg):
    """Log rerun world-axis convention from config.

    Supported config keys:
    - vis_3d_view_coordinates (preferred)
    - view_coordinates (fallback)
    """
    if rr is None:
        return
    name = cfg.get("vis_3d_view_coordinates", cfg.get("view_coordinates", "RUB"))
    if name is None:
        return
    vc_name = str(name).strip().upper()
    if not vc_name:
        return
    vc = getattr(rr.ViewCoordinates, vc_name, None)
    if vc is None:
        logger.warning(
            "[DynamicRerun] unknown view coordinates %r, fallback to RUB",
            name,
        )
        vc = getattr(rr.ViewCoordinates, "RUB", None)
        if vc is None:
            return
    rr.log("/", vc, static=True)


def vis_dynamic_points_rerun(cfg, mv_outputs, out_dir, data_idx, meta_data=None, max_queries=None):
    """Rerun visualization for sparse dynamic points.

    - GT:
      - frame 0 uses ``target_global_points`` (stored as ``pointmap_gt_global``)
      - later frames use per-pair ``tgt_3d_gt``
    - Pred:
      - arrows from ``src_3d_gt`` (or fallback ``src_points``) to ``warp3d``
    """
    if rr is None:
        logger.warning("[DynamicRerun] rerun-sdk is not installed, skip export.")
        return False
    if not mv_outputs:
        return False

    ref_out = mv_outputs[0]
    track_3d = getattr(ref_out, "track_3d", None)
    if not track_3d:
        logger.info("[DynamicRerun] no track_3d on mv_outputs[0], skip export.")
        return False

    cfg = cfg or {}
    total_steps = len(mv_outputs)
    max_points = max_queries if max_queries is not None else cfg.get("max_queries")
    show_dynamic = _cfg_bool(
        cfg,
        "show_dynamic",
        default=False,
    )
    show_dynamic_delta = _cfg_bool(
        cfg,
        "show_dynamic_delta",
        default=True,
    )
    show_dynamic_pred = _cfg_bool(
        cfg,
        "show_dynamic_pred",
        default=False,
    )
    show_dynamic_pred_delta = _cfg_bool(
        cfg,
        "show_dynamic_pred_delta",
        default=show_dynamic_pred,
    )
    show_pred_arrows_short = _cfg_bool(
        cfg,
        "show_pred_arrows_short",
        default=False,
    )
    replace_gt_with_pred_points = _cfg_bool(
        cfg,
        "replace_gt_with_pred_points",
        default=False,
    )
    pred_points_step = _cfg_int(
        cfg,
        "pred_points_step",
        default=32,
        min_value=1,
    )
    pred_arrows_long_hist_alpha_decay = max(
        float((cfg or {}).get("pred_arrows_long_hist_alpha_decay", 0.4)),
        0.0,
    )
    pred_arrows_long_hist_alpha_min = float((cfg or {}).get("pred_arrows_long_hist_alpha_min", 0.08))
    pred_arrows_long_hist_alpha_min = float(np.clip(pred_arrows_long_hist_alpha_min, 0.0, 1.0))

    rr.init(f"DynamicPoints_{data_idx}", spawn=False)
    _setup_blueprint(
        show_dynamic=show_dynamic,
        show_dynamic_delta=show_dynamic_delta,
        show_dynamic_pred=show_dynamic_pred,
        show_dynamic_pred_delta=show_dynamic_pred_delta,
    )
    _log_view_coordinates(cfg)

    save_path = os.path.join(out_dir, "rerun_vis", f"vis_dynamic_{data_idx:06d}.rrd")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    gt_by_frame = {}
    gt_col_by_frame = {}
    gt_uv_by_frame = {}
    arrow_org_by_frame = {}
    arrow_pred_by_frame = {}
    arrow_rgb_by_frame = {}
    arrow_vis_col_by_frame = {}
    arrow_org_delta_by_frame = {}
    arrow_pred_delta_by_frame = {}
    arrow_rgb_delta_by_frame = {}
    arrow_vis_col_delta_by_frame = {}
    src_init_by_stream = {}
    stream_arrow_color_lut = {}

    # Temporal state per stream: first step uses src_3d_gt/src_points, then uses previous warp3d.
    # Here stream is keyed by source index (or 0 fallback), without requiring frame/view metadata.
    prev_pred_by_stream = {}
    prev_pred_delta_by_stream = {}

    for tgt_key in sorted(track_3d.keys(), key=lambda k: int(k)):
        tgt_flat = int(tgt_key)
        tgt_frame = tgt_flat
        if tgt_frame >= total_steps:
            continue
        track = track_3d[tgt_key]
        stream_id = int(getattr(track, "src_index", 0) or 0)
        src, pred, pred_from_delta, gt_tgt, colors, query_uv_valid, valid_indices = _collect_track_rows(track, ref_out)
        if src is None:
            continue

        if stream_id not in src_init_by_stream:
            src_init_by_stream[stream_id] = (src, colors)

        prev_pred = prev_pred_by_stream.get(stream_id)
        if prev_pred is not None and prev_pred.shape == pred.shape:
            origin = prev_pred
        else:
            origin = src

        if stream_id not in stream_arrow_color_lut:
            stream_arrow_color_lut[stream_id] = {}
        color_lut = stream_arrow_color_lut[stream_id]
        if valid_indices is None:
            valid_indices = np.arange(origin.shape[0], dtype=np.int64)
        valid_indices = np.asarray(valid_indices, dtype=np.int64).reshape(-1)

        arrow_colors = np.zeros((origin.shape[0], 3), dtype=np.uint8)
        missing_pos = []
        for row_i, row_idx in enumerate(valid_indices):
            cached = color_lut.get(int(row_idx))
            if cached is None:
                missing_pos.append(row_i)
            else:
                arrow_colors[row_i] = cached
        if missing_pos:
            current_colors = _arrow_colors_by_cfg(origin, pred, cfg)
            for row_i in missing_pos:
                c = current_colors[row_i].astype(np.uint8)
                arrow_colors[row_i] = c
                color_lut[int(valid_indices[row_i])] = c

        arrow_org_by_frame.setdefault(tgt_frame, []).append(origin)
        arrow_pred_by_frame.setdefault(tgt_frame, []).append(pred)
        arrow_rgb_by_frame.setdefault(tgt_frame, []).append(colors)
        arrow_vis_col_by_frame.setdefault(tgt_frame, []).append(arrow_colors)

        prev_pred_by_stream[stream_id] = pred
        if pred_from_delta is not None:
            prev_pred_delta = prev_pred_delta_by_stream.get(stream_id)
            if prev_pred_delta is not None and prev_pred_delta.shape == pred_from_delta.shape:
                origin_delta = prev_pred_delta
            else:
                origin_delta = src
            arrow_org_delta_by_frame.setdefault(tgt_frame, []).append(origin_delta)
            arrow_pred_delta_by_frame.setdefault(tgt_frame, []).append(pred_from_delta)
            arrow_rgb_delta_by_frame.setdefault(tgt_frame, []).append(colors)
            arrow_vis_col_delta_by_frame.setdefault(tgt_frame, []).append(arrow_colors)
            prev_pred_delta_by_stream[stream_id] = pred_from_delta
        if gt_tgt is not None:
            gt_by_frame.setdefault(tgt_frame, []).append(gt_tgt)
            gt_col_by_frame.setdefault(tgt_frame, []).append(colors)
            if query_uv_valid is not None:
                gt_uv_by_frame.setdefault(tgt_frame, []).append(query_uv_valid)

    # frame0_gt, frame0_gt_cols = _frame0_gt_points_and_colors(ref_out)
    gt_acc_pts = []
    gt_acc_cols = []
    pred_acc_pts = []
    pred_acc_rgb_col = []
    pred_win_short = deque()
    pred_win_long = deque()
    pred_all_short = []
    pred_all_long = []
    pred_acc_pts_delta = []
    pred_acc_rgb_col_delta = []
    pred_win_short_delta = deque()
    pred_win_long_delta = deque()
    pred_all_short_delta = []
    pred_all_long_delta = []

    # Right view t0: show source points.
    src_init_pts_all = None
    src_init_cols_all = None
    if src_init_by_stream:
        src_init_pts = np.concatenate([v[0] for v in src_init_by_stream.values()], axis=0)
        src_init_cols = np.concatenate([v[1] for v in src_init_by_stream.values()], axis=0)
        src_init_pts_all = src_init_pts
        src_init_cols_all = src_init_cols
        if max_points is not None and src_init_pts.shape[0] > max_points:
            rng = np.random.default_rng(seed=1900)
            sel = rng.choice(src_init_pts.shape[0], size=max_points, replace=False)
            src_init_pts, src_init_cols = src_init_pts[sel], src_init_cols[sel]
        pred_acc_pts.append(src_init_pts)
        pred_acc_rgb_col.append(src_init_cols)
        pred_acc_pts_delta.append(src_init_pts)
        pred_acc_rgb_col_delta.append(src_init_cols)

    for tgt_frame in range(total_steps):
        rr.set_time_sequence("target_frame", tgt_frame)
        # Ensure each timestamp only contains current-frame arrows/points.
        if show_dynamic:
            rr.log("dynamic/pred_arrows_long", rr.Clear(recursive=True))
            rr.log("dynamic/pred_arrows_long_hist", rr.Clear(recursive=True))
            if show_pred_arrows_short:
                rr.log("dynamic/pred_arrows_short", rr.Clear(recursive=True))
                rr.log("dynamic/pred_arrows_short_hist", rr.Clear(recursive=True))

        if show_dynamic_pred:
            rr.log("dynamic_pred/pred_arrows_long", rr.Clear(recursive=True))
            rr.log("dynamic_pred/pred_arrows_long_hist", rr.Clear(recursive=True))
            rr.log("dynamic_pred/pred_points", rr.Clear(recursive=True))
            if show_pred_arrows_short:
                rr.log("dynamic_pred/pred_arrows_short", rr.Clear(recursive=True))
                rr.log("dynamic_pred/pred_arrows_short_hist", rr.Clear(recursive=True))

        if show_dynamic_delta:
            rr.log("dynamic_delta/pred_arrows_long", rr.Clear(recursive=True))
            rr.log("dynamic_delta/pred_arrows_long_hist", rr.Clear(recursive=True))
            if show_pred_arrows_short:
                rr.log("dynamic_delta/pred_arrows_short", rr.Clear(recursive=True))
                rr.log("dynamic_delta/pred_arrows_short_hist", rr.Clear(recursive=True))

        if show_dynamic_pred_delta:
            rr.log("dynamic_pred_delta/pred_arrows_long", rr.Clear(recursive=True))
            rr.log("dynamic_pred_delta/pred_arrows_long_hist", rr.Clear(recursive=True))
            rr.log("dynamic_pred_delta/pred_points", rr.Clear(recursive=True))
            if show_pred_arrows_short:
                rr.log("dynamic_pred_delta/pred_arrows_short", rr.Clear(recursive=True))
                rr.log("dynamic_pred_delta/pred_arrows_short_hist", rr.Clear(recursive=True))

        _log_camera_frustums(
            tgt_frame,
            mv_outputs,
            cfg,
            show_dynamic=show_dynamic,
            show_dynamic_delta=show_dynamic_delta,
            show_dynamic_pred=show_dynamic_pred,
            show_dynamic_pred_delta=show_dynamic_pred_delta,
        )

        h_gt = int(getattr(ref_out, "pointmap_gt_h", 0) or 0)
        w_gt = int(getattr(ref_out, "pointmap_gt_w", 0) or 0)

        frame0_gt, frame0_gt_cols = _frame0_gt_points_and_colors(mv_outputs[tgt_frame])

        # if tgt_frame == 0 and frame0_gt.shape[0] > 0:
        if frame0_gt.shape[0] > 0:
            gt0 = frame0_gt.reshape(-1, 3)
            gt0_cols = frame0_gt_cols.reshape(-1, 3)

            step = 2
            if h_gt > 0 and w_gt > 0 and gt0.shape[0] == h_gt * w_gt:
                gt0 = gt0.reshape(h_gt, w_gt, 3)[::step, ::step, :].reshape(-1, 3)
                gt0_cols = gt0_cols.reshape(h_gt, w_gt, 3)[::step, ::step, :].reshape(-1, 3)
            else:
                gt0 = gt0[::step]
                gt0_cols = gt0_cols[::step]
            gt_acc_pts.append(gt0)
            gt_acc_cols.append(gt0_cols)


        # 拆成两个分量：第一帧 (frame0) + 当前帧 (cur)，便于分 topic 显示
        if gt_acc_pts:
            gt_pts_t0 = gt_acc_pts[0]
            gt_cols_t0 = gt_acc_cols[0]
            if len(gt_acc_pts) > 1:
                gt_pts_cur = gt_acc_pts[-1]
                gt_cols_cur = gt_acc_cols[-1]
            else:
                gt_pts_cur = np.zeros((0, 3), dtype=np.float64)
                gt_cols_cur = np.zeros((0, 3), dtype=np.uint8)

            if max_points is not None and gt_pts_t0.shape[0] > max_points:
                # 第一帧点是静态的，用固定 seed 保证跨步采样一致
                rng = np.random.default_rng(seed=2026)
                sel = rng.choice(gt_pts_t0.shape[0], size=max_points, replace=False)
                gt_pts_t0, gt_cols_t0 = gt_pts_t0[sel], gt_cols_t0[sel]
            if max_points is not None and gt_pts_cur.shape[0] > max_points:
                rng = np.random.default_rng(seed=2027 + tgt_frame)
                sel = rng.choice(gt_pts_cur.shape[0], size=max_points, replace=False)
                gt_pts_cur, gt_cols_cur = gt_pts_cur[sel], gt_cols_cur[sel]

            gt_pts = (
                np.concatenate([gt_pts_t0, gt_pts_cur], axis=0)
                if gt_pts_cur.shape[0] > 0
                else gt_pts_t0
            )
            gt_cols = (
                np.concatenate([gt_cols_t0, gt_cols_cur], axis=0)
                if gt_pts_cur.shape[0] > 0
                else gt_cols_t0
            )
        else:
            gt_pts_t0 = np.zeros((0, 3), dtype=np.float64)
            gt_cols_t0 = np.zeros((0, 3), dtype=np.uint8)
            gt_pts_cur = np.zeros((0, 3), dtype=np.float64)
            gt_cols_cur = np.zeros((0, 3), dtype=np.uint8)
            gt_pts = np.zeros((0, 3), dtype=np.float64)
            gt_cols = np.zeros((0, 3), dtype=np.uint8)

        def _build_pred_display_points():
            # Pred display follows the same rule as GT display:
            # frame-0 points + current-frame points.
            # 返回拆开的分量：(t0_pts, t0_cols, cur_pts, cur_cols, delta_cur_pts, delta_cur_cols)
            # 调用方负责把第一帧 (t0) 写到 gt_points_frame0 topic，把 cur/delta_cur 写到 gt_points topic。
            dyn_t0_pts = src_init_pts_all if src_init_pts_all is not None else np.zeros((0, 3), dtype=np.float64)
            dyn_t0_cols = src_init_cols_all if src_init_cols_all is not None else np.zeros((0, 3), dtype=np.uint8)

            dyn_cur_pts = np.zeros((0, 3), dtype=np.float64)
            dyn_cur_cols = np.zeros((0, 3), dtype=np.uint8)
            if tgt_frame > 0 and tgt_frame in arrow_pred_by_frame:
                dyn_cur_pts = np.concatenate(arrow_pred_by_frame[tgt_frame], axis=0)
                dyn_cur_cols = (
                    np.concatenate(arrow_rgb_by_frame[tgt_frame], axis=0)
                    if tgt_frame in arrow_rgb_by_frame
                    else np.full((dyn_cur_pts.shape[0], 3), 180, dtype=np.uint8)
                )

            dyn_delta_cur_pts = np.zeros((0, 3), dtype=np.float64)
            dyn_delta_cur_cols = np.zeros((0, 3), dtype=np.uint8)
            if tgt_frame > 0 and tgt_frame in arrow_pred_delta_by_frame:
                dyn_delta_cur_pts = np.concatenate(arrow_pred_delta_by_frame[tgt_frame], axis=0)
                dyn_delta_cur_cols = (
                    np.concatenate(arrow_rgb_delta_by_frame[tgt_frame], axis=0)
                    if tgt_frame in arrow_rgb_delta_by_frame
                    else np.full((dyn_delta_cur_pts.shape[0], 3), 180, dtype=np.uint8)
                )
            else:
                dyn_delta_cur_pts = dyn_cur_pts
                dyn_delta_cur_cols = dyn_cur_cols

            dyn_t0_pts, dyn_t0_cols = _stride_sample_points(dyn_t0_pts, dyn_t0_cols, pred_points_step)
            dyn_cur_pts, dyn_cur_cols = _stride_sample_points(dyn_cur_pts, dyn_cur_cols, pred_points_step)
            dyn_delta_cur_pts, dyn_delta_cur_cols = _stride_sample_points(
                dyn_delta_cur_pts,
                dyn_delta_cur_cols,
                pred_points_step,
            )

            if max_points is not None and dyn_t0_pts.shape[0] > max_points:
                # 第一帧静态点云，用固定 seed 保证跨步采样一致
                rng = np.random.default_rng(seed=4000)
                sel = rng.choice(dyn_t0_pts.shape[0], size=max_points, replace=False)
                dyn_t0_pts, dyn_t0_cols = dyn_t0_pts[sel], dyn_t0_cols[sel]
            if max_points is not None and dyn_cur_pts.shape[0] > max_points:
                rng = np.random.default_rng(seed=4010 + tgt_frame)
                sel = rng.choice(dyn_cur_pts.shape[0], size=max_points, replace=False)
                dyn_cur_pts, dyn_cur_cols = dyn_cur_pts[sel], dyn_cur_cols[sel]
            if max_points is not None and dyn_delta_cur_pts.shape[0] > max_points:
                rng = np.random.default_rng(seed=4510 + tgt_frame)
                sel = rng.choice(dyn_delta_cur_pts.shape[0], size=max_points, replace=False)
                dyn_delta_cur_pts, dyn_delta_cur_cols = dyn_delta_cur_pts[sel], dyn_delta_cur_cols[sel]
            return (
                dyn_t0_pts,
                dyn_t0_cols,
                dyn_cur_pts,
                dyn_cur_cols,
                dyn_delta_cur_pts,
                dyn_delta_cur_cols,
            )

        # 第一帧静态点云写到独立 topic `<root>/gt_points_frame0`，当前帧动态点写到 `<root>/gt_points`，
        # 这样在 rerun UI 里可以独立切换/隐藏首帧点云。
        def _log_split(root, t0_pts, t0_cols, cur_pts, cur_cols):
            if t0_pts.shape[0] > 0:
                rr.log(f"{root}/gt_points_frame0", rr.Points3D(t0_pts, colors=t0_cols, radii=0.008))
            else:
                rr.log(f"{root}/gt_points_frame0", rr.Clear(recursive=True))
            if cur_pts.shape[0] > 0:
                rr.log(f"{root}/gt_points", rr.Points3D(cur_pts, colors=cur_cols, radii=0.008))
            else:
                rr.log(f"{root}/gt_points", rr.Clear(recursive=True))

        if replace_gt_with_pred_points:
            (
                dyn_t0_pts,
                dyn_t0_cols,
                dyn_cur_pts,
                dyn_cur_cols,
                dyn_delta_cur_pts,
                dyn_delta_cur_cols,
            ) = _build_pred_display_points()

            if show_dynamic:
                _log_split("dynamic", dyn_t0_pts, dyn_t0_cols, dyn_cur_pts, dyn_cur_cols)
            if show_dynamic_delta:
                _log_split(
                    "dynamic_delta",
                    dyn_t0_pts,
                    dyn_t0_cols,
                    dyn_delta_cur_pts,
                    dyn_delta_cur_cols,
                )
        else:
            if gt_pts.shape[0] > 0:
                if show_dynamic:
                    _log_split("dynamic", gt_pts_t0, gt_cols_t0, gt_pts_cur, gt_cols_cur)
                if show_dynamic_delta:
                    _log_split(
                        "dynamic_delta",
                        gt_pts_t0,
                        gt_cols_t0,
                        gt_pts_cur,
                        gt_cols_cur,
                    )
            else:
                # No GT available: fallback to pred points for visualization.
                (
                    dyn_t0_pts,
                    dyn_t0_cols,
                    dyn_cur_pts,
                    dyn_cur_cols,
                    dyn_delta_cur_pts,
                    dyn_delta_cur_cols,
                ) = _build_pred_display_points()
                if show_dynamic:
                    _log_split("dynamic", dyn_t0_pts, dyn_t0_cols, dyn_cur_pts, dyn_cur_cols)
                if show_dynamic_delta:
                    _log_split(
                        "dynamic_delta",
                        dyn_t0_pts,
                        dyn_t0_cols,
                        dyn_delta_cur_pts,
                        dyn_delta_cur_cols,
                    )

        if tgt_frame in arrow_org_by_frame and tgt_frame in arrow_pred_by_frame:
            origin = np.concatenate(arrow_org_by_frame[tgt_frame], axis=0)
            pred = np.concatenate(arrow_pred_by_frame[tgt_frame], axis=0)
            rgb_colors = (
                np.concatenate(arrow_rgb_by_frame[tgt_frame], axis=0)
                if tgt_frame in arrow_rgb_by_frame
                else np.full((origin.shape[0], 3), 180, dtype=np.uint8)
            )
            arrow_vis_colors = (
                np.concatenate(arrow_vis_col_by_frame[tgt_frame], axis=0)
                if tgt_frame in arrow_vis_col_by_frame
                else _arrow_colors_by_cfg(origin, pred, cfg)
            )
            # Apply index-step sampling to arrow data as well.
            if pred_points_step > 1 and origin.shape[0] > 1:
                origin = origin[::pred_points_step]
                pred = pred[::pred_points_step]
                rgb_colors = rgb_colors[::pred_points_step]
                arrow_vis_colors = arrow_vis_colors[::pred_points_step]
            if max_points is not None and origin.shape[0] > max_points:
                rng = np.random.default_rng(seed=3000 + tgt_frame)
                sel = rng.choice(origin.shape[0], size=max_points, replace=False)
                origin, pred, rgb_colors, arrow_vis_colors = origin[sel], pred[sel], rgb_colors[sel], arrow_vis_colors[sel]

            vec = pred - origin
            dir_colors = arrow_vis_colors
            min_arrow_len = float((cfg or {}).get("min_arrow_length", 0.01))
            vec_vis = _enforce_min_arrow_length(vec, min_arrow_len)
            mags = np.linalg.norm(vec, axis=1)
            short_mask = mags < min_arrow_len
            long_mask = ~short_mask

            # Right view: predicted-only accumulation (no GT points).
            if show_dynamic_pred and tgt_frame > 0:
                pred_acc_pts.append(pred)
                pred_acc_rgb_col.append(rgb_colors)

            if show_pred_arrows_short and np.any(short_mask):
                short_entry = (
                    tgt_frame,
                    origin[short_mask],
                    vec_vis[short_mask],
                    dir_colors[short_mask],
                )
                pred_win_short.append(short_entry)
                pred_all_short.append(short_entry)
            if np.any(long_mask):
                long_entry = (
                    tgt_frame,
                    origin[long_mask],
                    vec_vis[long_mask],
                    dir_colors[long_mask],
                )
                pred_win_long.append(long_entry)
                pred_all_long.append(long_entry)

        if tgt_frame in arrow_org_delta_by_frame and tgt_frame in arrow_pred_delta_by_frame:
            origin_delta = np.concatenate(arrow_org_delta_by_frame[tgt_frame], axis=0)
            pred_delta = np.concatenate(arrow_pred_delta_by_frame[tgt_frame], axis=0)
            rgb_colors_delta = (
                np.concatenate(arrow_rgb_delta_by_frame[tgt_frame], axis=0)
                if tgt_frame in arrow_rgb_delta_by_frame
                else np.full((origin_delta.shape[0], 3), 180, dtype=np.uint8)
            )
            arrow_vis_colors_delta = (
                np.concatenate(arrow_vis_col_delta_by_frame[tgt_frame], axis=0)
                if tgt_frame in arrow_vis_col_delta_by_frame
                else _arrow_colors_by_cfg(origin_delta, pred_delta, cfg)
            )
            # Apply index-step sampling to delta arrow data as well.
            if pred_points_step > 1 and origin_delta.shape[0] > 1:
                origin_delta = origin_delta[::pred_points_step]
                pred_delta = pred_delta[::pred_points_step]
                rgb_colors_delta = rgb_colors_delta[::pred_points_step]
                arrow_vis_colors_delta = arrow_vis_colors_delta[::pred_points_step]
            if max_points is not None and origin_delta.shape[0] > max_points:
                rng = np.random.default_rng(seed=3500 + tgt_frame)
                sel = rng.choice(origin_delta.shape[0], size=max_points, replace=False)
                origin_delta, pred_delta, rgb_colors_delta, arrow_vis_colors_delta = (
                    origin_delta[sel],
                    pred_delta[sel],
                    rgb_colors_delta[sel],
                    arrow_vis_colors_delta[sel],
                )

            vec_delta = pred_delta - origin_delta
            dir_colors_delta = arrow_vis_colors_delta
            min_arrow_len = float((cfg or {}).get("min_arrow_length", 0.01))
            vec_vis_delta = _enforce_min_arrow_length(vec_delta, min_arrow_len)
            mags_delta = np.linalg.norm(vec_delta, axis=1)
            short_mask_delta = mags_delta < min_arrow_len
            long_mask_delta = ~short_mask_delta

            # Right-bottom view: predicted-only accumulation using src + warp3d_delta.
            if show_dynamic_pred_delta and tgt_frame > 0:
                pred_acc_pts_delta.append(pred_delta)
                pred_acc_rgb_col_delta.append(rgb_colors_delta)

            if show_pred_arrows_short and np.any(short_mask_delta):
                short_entry_delta = (
                    tgt_frame,
                    origin_delta[short_mask_delta],
                    vec_vis_delta[short_mask_delta],
                    dir_colors_delta[short_mask_delta],
                )
                pred_win_short_delta.append(short_entry_delta)
                pred_all_short_delta.append(short_entry_delta)
            if np.any(long_mask_delta):
                long_entry_delta = (
                    tgt_frame,
                    origin_delta[long_mask_delta],
                    vec_vis_delta[long_mask_delta],
                    dir_colors_delta[long_mask_delta],
                )
                pred_win_long_delta.append(long_entry_delta)
                pred_all_long_delta.append(long_entry_delta)

        win_begin = tgt_frame - 10
        while pred_win_short and pred_win_short[0][0] < win_begin:
            pred_win_short.popleft()
        while pred_win_long and pred_win_long[0][0] < win_begin:
            pred_win_long.popleft()
        while pred_win_short_delta and pred_win_short_delta[0][0] < win_begin:
            pred_win_short_delta.popleft()
        while pred_win_long_delta and pred_win_long_delta[0][0] < win_begin:
            pred_win_long_delta.popleft()

        def _log_hist_and_current(root, short_data, long_data):
            if show_pred_arrows_short:
                short_hist = [x for x in short_data if x[0] < tgt_frame]
                short_cur = [x for x in short_data if x[0] == tgt_frame]
                if short_hist:
                    rr.log(
                        f"{root}/pred_arrows_short_hist",
                        rr.LineStrips3D(
                            strips=np.concatenate([_to_line_segments(x[1], x[2]) for x in short_hist], axis=0),
                            colors=np.concatenate([x[3] for x in short_hist], axis=0),
                            radii=0.002,
                        ),
                    )
                if short_cur:
                    rr.log(
                        f"{root}/pred_arrows_short",
                        rr.Arrows3D(
                            origins=np.concatenate([x[1] for x in short_cur], axis=0),
                            vectors=np.concatenate([x[2] for x in short_cur], axis=0),
                            colors=np.concatenate([x[3] for x in short_cur], axis=0),
                            radii=0.002,
                        ),
                    )

            long_hist = [x for x in long_data if x[0] < tgt_frame]
            long_cur = [x for x in long_data if x[0] == tgt_frame]
            if long_hist:
                long_hist_strips = np.concatenate([_to_line_segments(x[1], x[2]) for x in long_hist], axis=0)
                if root in ("dynamic", "dynamic_delta") and pred_arrows_long_hist_alpha_decay > 0.0:
                    hist_colors = []
                    for x in long_hist:
                        c = np.asarray(x[3], dtype=np.uint8).reshape(x[3].shape[0], -1)
                        age = max(int(tgt_frame - x[0]), 0)
                        alpha_scale = max(
                            pred_arrows_long_hist_alpha_min,
                            float(np.exp(-pred_arrows_long_hist_alpha_decay * age)),
                        )
                        if c.shape[1] >= 4:
                            rgba = c[:, :4].copy()
                            rgba[:, 3] = np.clip(
                                rgba[:, 3].astype(np.float32) * alpha_scale,
                                0.0,
                                255.0,
                            ).astype(np.uint8)
                        else:
                            alpha = np.full(
                                (c.shape[0], 1),
                                int(np.clip(alpha_scale * 255.0, 0.0, 255.0)),
                                dtype=np.uint8,
                            )
                            rgba = np.concatenate([c[:, :3], alpha], axis=1)
                        hist_colors.append(rgba)
                    long_hist_colors = np.concatenate(hist_colors, axis=0)
                else:
                    long_hist_colors = np.concatenate([x[3] for x in long_hist], axis=0)
                rr.log(
                    f"{root}/pred_arrows_long_hist",
                    rr.LineStrips3D(
                        strips=long_hist_strips,
                        colors=long_hist_colors,
                        radii=0.002,
                    ),
                )
            if long_cur:
                rr.log(
                    f"{root}/pred_arrows_long",
                    rr.Arrows3D(
                        origins=np.concatenate([x[1] for x in long_cur], axis=0),
                        vectors=np.concatenate([x[2] for x in long_cur], axis=0),
                        colors=np.concatenate([x[3] for x in long_cur], axis=0),
                        radii=0.002,
                    ),
                )

        if show_dynamic:
            _log_hist_and_current("dynamic", list(pred_win_short), list(pred_win_long))
        if show_dynamic_pred:
            _log_hist_and_current("dynamic_pred", pred_all_short, pred_all_long)
        if show_dynamic_delta:
            _log_hist_and_current("dynamic_delta", list(pred_win_short_delta), list(pred_win_long_delta))
        if show_dynamic_pred_delta:
            _log_hist_and_current("dynamic_pred_delta", pred_all_short_delta, pred_all_long_delta)

        if show_dynamic_pred and pred_acc_pts:
            pred_pts_all = np.concatenate(pred_acc_pts, axis=0)
            rgb_col_all = np.concatenate(pred_acc_rgb_col, axis=0)
            pred_pts_all, rgb_col_all = _stride_sample_points(pred_pts_all, rgb_col_all, pred_points_step)
            rr.log("dynamic_pred/pred_points", rr.Points3D(pred_pts_all, colors=rgb_col_all, radii=0.008))
        if show_dynamic_pred_delta and pred_acc_pts_delta:
            pred_pts_all_delta = np.concatenate(pred_acc_pts_delta, axis=0)
            rgb_col_all_delta = np.concatenate(pred_acc_rgb_col_delta, axis=0)
            pred_pts_all_delta, rgb_col_all_delta = _stride_sample_points(
                pred_pts_all_delta,
                rgb_col_all_delta,
                pred_points_step,
            )
            rr.log(
                "dynamic_pred_delta/pred_points",
                rr.Points3D(pred_pts_all_delta, colors=rgb_col_all_delta, radii=0.008),
            )

    rr.save(save_path)
    logger.info("[DynamicRerun] saved: %s", save_path)
    return True
