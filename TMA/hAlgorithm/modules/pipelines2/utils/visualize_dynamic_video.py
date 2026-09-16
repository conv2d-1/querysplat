"""Motion trajectory videos (warp2d / track3d_2d), separate from static JPG vis in visualize_dynamic."""

import logging
import os

import cv2
import numpy as np

from hAlgorithm.modules.pipelines2.utils.visualize_dynamic import (
    _generate_colors,
    _grid_subsample_ids,
    _subsample_ids,
    _to_np,
    _uv_to_pixel,
)


def _soften_bgr_colors(colors, mix=0.45):
    """Blend BGR colours toward white. mix=0 unchanged, mix=1 white."""
    if mix <= 0:
        return colors
    mix = min(max(float(mix), 0.0), 1.0)
    softened = []
    for b, g, r in colors:
        softened.append(
            (
                int(b * (1.0 - mix) + 255 * mix),
                int(g * (1.0 - mix) + 255 * mix),
                int(r * (1.0 - mix) + 255 * mix),
            )
        )
    return softened


def _rgb_to_bgr(img_rgb):
    return cv2.cvtColor((np.clip(img_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _get_frame_bgr(mv_output):
    return _rgb_to_bgr(mv_output.rgb)


def _draw_points_on_frame(
    canvas,
    uv,
    H,
    W,
    point_ids,
    colors,
    is_normalized=True,
    marker="circle",
):
    """Draw coloured tracking points on a single image canvas."""
    radius = max(2, min(H, W) // 200)
    thickness = max(1, radius // 2)
    out = canvas.copy()

    def _px(uv_row):
        if is_normalized:
            return _uv_to_pixel(uv_row, H, W)
        return uv_row

    for ci, qi in enumerate(point_ids):
        if qi >= len(uv):
            continue
        color = colors[ci]
        x, y = _px(uv[qi : qi + 1])[0]
        px, py = int(x), int(y)
        try:
            if marker == "circle":
                cv2.circle(out, (px, py), radius, color, -1, cv2.LINE_AA)
            else:
                d = radius
                cv2.line(out, (px - d, py - d), (px + d, py + d), color, thickness, cv2.LINE_AA)
                cv2.line(out, (px - d, py + d), (px + d, py - d), color, thickness, cv2.LINE_AA)
        except Exception:
            logging.error("draw point failed at (%s, %s)", px, py)
    return out


def _draw_trajectory_trails(
    canvas,
    trail_pixels,
    colors,
    min_alpha=0.12,
    trail_length=4,
    decay=1.2,
):
    """Draw recent trajectory segments with age-based fade."""
    if len(trail_pixels) < 2:
        return canvas

    out = canvas.copy()
    n_steps = len(trail_pixels)
    h, w = out.shape[:2]
    start = max(0, n_steps - int(trail_length) - 1)

    for step_i in range(start + 1, n_steps):
        age = n_steps - 1 - step_i
        fade = np.exp(-age / max(float(decay), 1e-3))
        alpha = min_alpha + (1.0 - min_alpha) * fade
        overlay = out.copy()
        for ci in range(len(colors)):
            pt0 = trail_pixels[step_i - 1][ci]
            pt1 = trail_pixels[step_i][ci]
            if not (np.isfinite(pt0).all() and np.isfinite(pt1).all()):
                continue
            x0, y0 = int(round(pt0[0])), int(round(pt0[1]))
            x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
            if not (0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h):
                continue
            cv2.line(
                overlay,
                (x0, y0),
                (x1, y1),
                colors[ci],
                max(1, min(h, w) // 256),
                cv2.LINE_AA,
            )
        out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
    return out


def _write_bgr_video(frames, output_path, fps=10):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        logging.error("failed to open video writer: %s", output_path)
        return
    for frame in frames:
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)
    writer.release()


def _collect_warp2d_timeline(src_idx, track_3d):
    tgt_keys = sorted(int(k) for k in track_3d.keys() if int(k) != int(src_idx))
    timeline = [int(src_idx)] + tgt_keys
    deduped = []
    seen = set()
    for idx in timeline:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def _get_warp2d_positions_at_step(step_idx, frame_id, src_idx, track_3d, query_uv, pred_ids):
    if step_idx == 0 and int(frame_id) == int(src_idx):
        uv = _to_np(query_uv, min_ndim=2)
    else:
        track = track_3d.get(frame_id)
        if track is None:
            return None
        uv = _to_np(track.warp2d, min_ndim=2)
    if uv is None or len(uv) == 0:
        return None
    if pred_ids is not None:
        return uv[pred_ids]
    return uv


def _get_track3d_2d_positions_at_step(step_idx, frame_id, src_idx, track_3d, query_uv, pred_ids):
    """Return (uv, is_normalized) for single-frame track3d_2d video drawing."""
    if step_idx == 0 and int(frame_id) == int(src_idx):
        uv = _to_np(query_uv, min_ndim=2)
        is_normalized = True
    else:
        track = track_3d.get(frame_id)
        if track is None:
            return None, True
        uv = _to_np(track.warp3d_uv, min_ndim=2)
        is_normalized = False
    if uv is None or len(uv) == 0:
        return None, True
    if pred_ids is not None:
        return uv[pred_ids], is_normalized
    return uv, is_normalized


def _uv_to_canvas_pixel(uv, h, w, is_normalized):
    if is_normalized:
        return _uv_to_pixel(uv, h, w)
    return np.asarray(uv, dtype=np.float64).reshape(-1, 2)


def _save_warp2d_trajectory_video(
    cfg,
    mv_outputs,
    src_idx,
    track_3d,
    query_uv,
    pred_ids,
    colors,
    out_path,
):
    """Save warp2d tracking as a single-frame-per-step video with coloured trails."""
    timeline = _collect_warp2d_timeline(src_idx, track_3d)
    if len(timeline) == 0:
        return

    fps = int(cfg.get("motion_warp2d_video_fps", 6))
    hold_first = int(cfg.get("motion_warp2d_hold_first", 4))
    marker = cfg.get("motion_warp2d_marker", "circle")
    color_mix = float(cfg.get("motion_warp2d_video_color_mix", 0.45))
    trail_min_alpha = float(cfg.get("motion_warp2d_trail_min_alpha", 0.12))
    trail_length = int(cfg.get("motion_warp2d_trail_length", 4))
    trail_decay = float(cfg.get("motion_warp2d_trail_decay", 1.2))
    video_colors = _soften_bgr_colors(colors, mix=color_mix)

    query_uv = _to_np(query_uv, min_ndim=2)
    if query_uv is None or len(query_uv) == 0:
        return

    video_frames = []
    trail_pixels = []
    skipped_total = 0

    for step_idx, frame_id in enumerate(timeline):
        if frame_id >= len(mv_outputs):
            continue
        frame_out = mv_outputs[frame_id]
        if frame_out.rgb is None:
            continue

        img_bgr = _get_frame_bgr(frame_out)
        H, W = img_bgr.shape[:2]

        uv_step = _get_warp2d_positions_at_step(
            step_idx, frame_id, src_idx, track_3d, query_uv, pred_ids
        )
        if uv_step is None or len(uv_step) == 0:
            skipped_total += len(pred_ids)
            continue

        px_step = _uv_to_pixel(uv_step, H, W)
        trail_pixels.append(px_step)

        draw_ids = np.arange(len(pred_ids), dtype=int)

        canvas = img_bgr.copy()
        canvas = _draw_trajectory_trails(
            canvas,
            trail_pixels,
            video_colors,
            min_alpha=trail_min_alpha,
            trail_length=trail_length,
            decay=trail_decay,
        )
        canvas = _draw_points_on_frame(
            canvas,
            uv_step,
            H,
            W,
            draw_ids,
            video_colors,
            is_normalized=True,
            marker=marker,
        )

        label = f"frame {frame_id}  step {step_idx + 1}/{len(timeline)}"
        _put_frame_label(canvas, label)

        repeats = hold_first if step_idx == 0 else 1
        video_frames.extend([canvas.copy() for _ in range(repeats)])

    if not video_frames:
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _write_bgr_video(video_frames, out_path, fps=fps)
    if skipped_total > 0:
        logging.info(
            "vis_motion_video save: %s (%d frames), skipped %d invalid track points",
            out_path,
            len(video_frames),
            skipped_total,
        )
    else:
        logging.info("vis_motion_video save: %s (%d frames)", out_path, len(video_frames))


def _select_warp2d_pred_ids(track, max_vis_points):
    """Pick point indices for warp2d video (same rules as static JPG vis)."""
    query_uv = _to_np(track.query_uv, min_ndim=2)
    src_2d_gt = _to_np(track.src_2d_gt, min_ndim=2)
    src_3d_gt = _to_np(track.src_3d_gt, min_ndim=2)
    tgt_3d_gt = _to_np(track.tgt_3d_gt, min_ndim=2)
    src_valids = _to_np(track.src_valids_gt)
    tgt_valids = _to_np(track.tgt_valids_gt)
    qh = int(track.query_height) if track.query_height is not None else None
    qw = int(track.query_width) if track.query_width is not None else None

    has_gt = not (src_3d_gt is None and tgt_3d_gt is None)
    Q_pred = query_uv.shape[0] if query_uv is not None else 0

    if has_gt and src_2d_gt is not None and src_2d_gt.shape[0] > 0:
        gt_valid = np.ones(src_2d_gt.shape[0], dtype=bool)
        for m in (src_valids, tgt_valids):
            if m is not None:
                gt_valid &= (m > 0.5)
        pred_ids = np.where(gt_valid)[0]
        if len(pred_ids) > 0:
            pred_ids = _subsample_ids(pred_ids, max_vis_points)
    elif qh is not None and qw is not None and Q_pred > 0 and qh * qw == Q_pred:
        pred_ids = _grid_subsample_ids(qh, qw, max_vis_points)
    else:
        pred_ids = (
            np.arange(min(Q_pred, max_vis_points), dtype=int)
            if max_vis_points is not None
            else np.arange(Q_pred, dtype=int)
        )
    return pred_ids, query_uv, qh, qw


def _put_frame_label(canvas, label):
    h = canvas.shape[0]
    cv2.putText(
        canvas,
        label,
        (10, max(24, h // 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        label,
        (10, max(24, h // 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 220, 30),
        1,
        cv2.LINE_AA,
    )


def _save_track3d_2d_trajectory_video(
    cfg,
    mv_outputs,
    src_idx,
    track_3d,
    query_uv,
    pred_ids,
    colors,
    out_path,
):
    """Save 3D→2D reprojection as single-frame-per-step video (same style as warp2d mp4)."""
    timeline = _collect_warp2d_timeline(src_idx, track_3d)
    if len(timeline) == 0:
        return

    fps = int(cfg.get("motion_track3d_2d_video_fps", cfg.get("motion_warp2d_video_fps", 6)))
    hold_first = int(cfg.get("motion_track3d_2d_hold_first", cfg.get("motion_warp2d_hold_first", 4)))
    marker = cfg.get("motion_track3d_2d_marker", cfg.get("motion_warp2d_marker", "cross"))
    color_mix = float(
        cfg.get("motion_track3d_2d_video_color_mix", cfg.get("motion_warp2d_video_color_mix", 0.45))
    )
    trail_min_alpha = float(
        cfg.get("motion_track3d_2d_trail_min_alpha", cfg.get("motion_warp2d_trail_min_alpha", 0.12))
    )
    trail_length = int(
        cfg.get("motion_track3d_2d_trail_length", cfg.get("motion_warp2d_trail_length", 4))
    )
    trail_decay = float(
        cfg.get("motion_track3d_2d_trail_decay", cfg.get("motion_warp2d_trail_decay", 1.2))
    )
    video_colors = _soften_bgr_colors(colors, mix=color_mix)

    query_uv = _to_np(query_uv, min_ndim=2)
    if query_uv is None or len(query_uv) == 0:
        return

    video_frames = []
    trail_pixels = []
    skipped_total = 0

    for step_idx, frame_id in enumerate(timeline):
        if frame_id >= len(mv_outputs):
            continue
        frame_out = mv_outputs[frame_id]
        if frame_out.rgb is None:
            continue

        img_bgr = _get_frame_bgr(frame_out)
        H, W = img_bgr.shape[:2]

        uv_step, is_normalized = _get_track3d_2d_positions_at_step(
            step_idx, frame_id, src_idx, track_3d, query_uv, pred_ids
        )
        if uv_step is None or len(uv_step) == 0:
            skipped_total += len(pred_ids)
            continue

        px_step = _uv_to_canvas_pixel(uv_step, H, W, is_normalized)
        trail_pixels.append(px_step)

        draw_ids = np.arange(len(pred_ids), dtype=int)
        canvas = img_bgr.copy()
        canvas = _draw_trajectory_trails(
            canvas,
            trail_pixels,
            video_colors,
            min_alpha=trail_min_alpha,
            trail_length=trail_length,
            decay=trail_decay,
        )
        canvas = _draw_points_on_frame(
            canvas,
            uv_step,
            H,
            W,
            draw_ids,
            video_colors,
            is_normalized=is_normalized,
            marker=marker,
        )

        label = f"frame {frame_id}  step {step_idx + 1}/{len(timeline)}"
        _put_frame_label(canvas, label)

        repeats = hold_first if step_idx == 0 else 1
        video_frames.extend([canvas.copy() for _ in range(repeats)])

    if not video_frames:
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _write_bgr_video(video_frames, out_path, fps=fps)
    if skipped_total > 0:
        logging.info(
            "vis_motion_video save: %s (%d frames), skipped %d invalid track points",
            out_path,
            len(video_frames),
            skipped_total,
        )
    else:
        logging.info("vis_motion_video save: %s (%d frames)", out_path, len(video_frames))


def save_track3d_2d_trajectory_video_for_src(
    cfg,
    mv_outputs,
    motion_out_dir,
    data_idx,
    frame_index,
    src_idx,
    track_3d,
    max_vis_points=512,
):
    """Write ``track3d_2d_pred_{data_idx}.mp4`` for one source view when warp3d_uv exists."""
    if track_3d is None or len(track_3d) == 0:
        return

    first_track = next(iter(track_3d.values()))
    warp3d_uv = _to_np(first_track.warp3d_uv, min_ndim=2)
    has_track3d_2d = warp3d_uv is not None or any(
        _to_np(t.warp3d_uv, min_ndim=2) is not None for t in track_3d.values()
    )
    if not has_track3d_2d:
        return

    pred_ids, query_uv, _, _ = _select_warp2d_pred_ids(first_track, max_vis_points)
    if query_uv is None or len(pred_ids) == 0:
        return

    pred_src_prefix = f"src{frame_index:03d}"
    cur_dir = os.path.join(motion_out_dir, pred_src_prefix)
    os.makedirs(cur_dir, exist_ok=True)
    video_path = os.path.join(cur_dir, f"track3d_2d_pred_{data_idx:06d}.mp4")
    colors = _generate_colors(len(pred_ids))

    _save_track3d_2d_trajectory_video(
        cfg=cfg,
        mv_outputs=mv_outputs,
        src_idx=src_idx,
        track_3d=track_3d,
        query_uv=query_uv,
        pred_ids=pred_ids,
        colors=colors,
        out_path=video_path,
    )


def save_warp2d_trajectory_video_for_src(
    cfg,
    mv_outputs,
    motion_out_dir,
    data_idx,
    frame_index,
    src_idx,
    track_3d,
    max_vis_points=512,
):
    """Write ``track2d_pred_{data_idx}.mp4`` for one source view when warp2d exists."""
    if track_3d is None or len(track_3d) == 0:
        return

    first_track = next(iter(track_3d.values()))
    first_warp2d = _to_np(first_track.warp2d, min_ndim=2)
    has_warp2d = first_warp2d is not None or any(
        _to_np(t.warp2d, min_ndim=2) is not None for t in track_3d.values()
    )
    if not has_warp2d:
        return

    pred_ids, query_uv, _, _ = _select_warp2d_pred_ids(first_track, max_vis_points)
    if query_uv is None or len(pred_ids) == 0:
        return

    pred_src_prefix = f"src{frame_index:03d}"
    cur_dir = os.path.join(motion_out_dir, pred_src_prefix)
    os.makedirs(cur_dir, exist_ok=True)
    video_path = os.path.join(cur_dir, f"track2d_pred_{data_idx:06d}.mp4")
    colors = _generate_colors(len(pred_ids))

    _save_warp2d_trajectory_video(
        cfg=cfg,
        mv_outputs=mv_outputs,
        src_idx=src_idx,
        track_3d=track_3d,
        query_uv=query_uv,
        pred_ids=pred_ids,
        colors=colors,
        out_path=video_path,
    )
