"""WorldTrack track visualization (Open-d4rt ``vis_like_demo`` style)."""

from __future__ import annotations

import colorsys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg

from hAlgorithm.modules.metrics.worldtrack_eval_metrics import compute_scale_factor_global


def _track_colors(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    cols = []
    for i in range(n):
        rgb = colorsys.hsv_to_rgb(i / max(n, 1), 0.75, 1.0)
        cols.append([int(round(c * 255.0)) for c in rgb])
    return np.asarray(cols, dtype=np.uint8)


def _sample_track_subset(
    gt_tracks_world: np.ndarray,
    pred_tracks_world: np.ndarray,
    visibility_tq: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_points = int(gt_tracks_world.shape[1])
    if num_points <= int(max_points) or int(max_points) <= 0:
        idx = np.arange(num_points, dtype=np.int64)
    else:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(num_points, size=int(max_points), replace=False).astype(np.int64))
    return gt_tracks_world[:, idx], pred_tracks_world[:, idx], visibility_tq[:, idx]


def uv_norm_to_pixel(uv_norm: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalized ``[0,1]`` UV → pixel ``(u, v)`` (same as ``visualize_dynamic._uv_to_pixel``)."""
    uv = np.asarray(uv_norm, dtype=np.float64)
    px = uv[..., 0] * float(max(int(width) - 1, 1))
    py = uv[..., 1] * float(max(int(height) - 1, 1))
    return np.stack([px, py], axis=-1).astype(np.float64)


def sample_rgb_at_norm_uv(
    video_rgb: np.ndarray,
    uv_norm: np.ndarray,
    *,
    query_height: int,
    query_width: int,
) -> np.ndarray:
    """Sample RGB at normalized query UV on an image (handles query vs image size mismatch)."""
    video_rgb = np.asarray(video_rgb, dtype=np.uint8)
    uv_norm = np.asarray(uv_norm, dtype=np.float64).reshape(-1, 2)
    h_img, w_img = int(video_rgb.shape[0]), int(video_rgb.shape[1])
    qh, qw = int(query_height), int(query_width)
    uv_pix = uv_norm_to_pixel(uv_norm, qh, qw)
    if qh != h_img or qw != w_img:
        sx = float(max(w_img - 1, 1)) / float(max(qw - 1, 1))
        sy = float(max(h_img - 1, 1)) / float(max(qh - 1, 1))
        uv_pix[:, 0] *= sx
        uv_pix[:, 1] *= sy
    tracks_uv = uv_pix[None, ...]
    mask = np.ones((1, int(uv_pix.shape[0])), dtype=bool)
    return sample_video_rgb_at_track_uv(video_rgb[None, ...], tracks_uv, mask)


def sample_video_rgb_at_track_uv(
    video_rgb: np.ndarray,
    tracks_uv: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Per-point RGB from ``video_rgb[t]`` at integer pixel ``(u, v)`` for valid ``(t, q)``.

    Args:
        video_rgb: [T, H, W, 3] uint8 RGB.
        tracks_uv: [T, Q, 2] pixel coordinates (same convention as ``project_world_tracks_to_uv``).
        valid_mask: [T, Q] bool — only these entries are sampled.

    Returns:
        colors: [N, 3] uint8 in row-major order of ``np.nonzero(valid_mask)``.
    """
    video_rgb = np.asarray(video_rgb, dtype=np.uint8)
    tracks_uv = np.asarray(tracks_uv, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    t_idx, q_idx = np.nonzero(mask)
    if t_idx.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    h, w = int(video_rgb.shape[1]), int(video_rgb.shape[2])
    uv = tracks_uv[t_idx, q_idx]
    finite = np.isfinite(uv).all(axis=-1)
    if not np.any(finite):
        return np.zeros((0, 3), dtype=np.uint8)

    t_idx = t_idx[finite]
    q_idx = q_idx[finite]
    uv = uv[finite]
    ui = np.clip(np.rint(uv[:, 0]), 0, w - 1).astype(np.int64)
    vi = np.clip(np.rint(uv[:, 1]), 0, h - 1).astype(np.int64)
    return video_rgb[t_idx, vi, ui].astype(np.uint8)


def project_world_tracks_to_uv(
    points_ref0_tq3: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Ref0 / WorldTrack ``gt_tracks_world`` [T,Q,3] -> pixel UV [T,Q,2].

    Points are in the frame-0 reference frame (``tracks_cam_to_ref0_world``), not
    metric world. Project via ``ref0_to_cam`` then pinhole, matching JSON ``trajs_2d``.
    """
    from hAlgorithm.eval.worldtrack_json_loader import _project_cam_to_uv

    # Keep this module runnable in minimal environments (e.g. visualization-only boxes)
    # without importing torch via `worldtrack_tma_infer`.
    cam = _ref0_to_cam_tracks_np(points_ref0_tq3, extrinsics_w2c)
    return _project_cam_to_uv(cam, intrinsics).astype(np.float32)


def _ref0_to_cam_tracks_np(points_ref0_tq3: np.ndarray, extrinsics_w2c: np.ndarray) -> np.ndarray:
    """Frame-0 ref coordinates → per-frame native camera coords (NumPy-only)."""
    pts = np.asarray(points_ref0_tq3, dtype=np.float64)
    ext = np.asarray(extrinsics_w2c, dtype=np.float64)
    c2w0 = np.linalg.inv(ext[0])
    out = np.full_like(pts, np.nan, dtype=np.float64)
    for t in range(int(pts.shape[0])):
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


def _figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    return np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()


def export_video_from_frames(
    *,
    video_rgb: np.ndarray,
    fps: float,
    dst_video: Path,
) -> Tuple[str, str]:
    """Write mp4 (+ poster jpg). Uses ffmpeg when available."""
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    poster_name = "video_poster.jpg"
    poster_path = dst_video.parent / poster_name
    first = np.asarray(video_rgb[0], dtype=np.uint8)
    cv2.imwrite(str(poster_path), first[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    temp_video = dst_video.parent / "_tmp_input_video.mp4"
    h, w = int(video_rgb.shape[1]), int(video_rgb.shape[2])
    writer = cv2.VideoWriter(
        str(temp_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(fps, 1.0)),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {temp_video}")
    try:
        for frame_rgb in video_rgb:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_video),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst_video),
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        shutil.copy2(temp_video, dst_video)
    finally:
        if temp_video.exists():
            temp_video.unlink()
    return dst_video.name, poster_name


def render_gt_tracks_2d_video(
    *,
    video_rgb: np.ndarray,
    tracks_uv_pixel: np.ndarray,
    visibility_tq: np.ndarray,
    output_path: Path,
    trace_frames: int = 0,
    fps: float = 15.0,
    point_radius: int = 4,
    line_thickness: int = 2,
) -> str:
    """Render GT-only 2D trajectories: one color per track, solid=visible, hollow=occluded.

    Args:
        video_rgb: [T, H, W, 3] uint8 RGB frames.
        tracks_uv_pixel: [T, Q, 2] pixel UV (same convention as ``WorldTrackSequence.tracks_uv``).
        visibility_tq: [T, Q] bool visibility.
        output_path: Destination mp4 path.
        trace_frames: History length for trajectory lines; ``<=0`` means full history.
    """
    video_rgb = np.asarray(video_rgb, dtype=np.uint8)
    tracks_uv = np.asarray(tracks_uv_pixel, dtype=np.float64)
    visibility = np.asarray(visibility_tq, dtype=bool)
    if tracks_uv.ndim != 3 or visibility.ndim != 2:
        raise ValueError("tracks_uv_pixel must be [T,Q,2] and visibility_tq must be [T,Q].")
    if tracks_uv.shape[:2] != visibility.shape:
        raise ValueError("tracks_uv and visibility shape mismatch.")

    num_frames = int(video_rgb.shape[0])
    num_queries = int(tracks_uv.shape[1])
    colors_rgb = _track_colors(num_queries)
    colors_bgr = [tuple(int(c) for c in colors_rgb[qi][::-1].tolist()) for qi in range(num_queries)]

    frames_out = []
    for frame_idx in range(num_frames):
        frame = video_rgb[frame_idx].copy()
        hist_start = 0 if int(trace_frames) <= 0 else max(0, frame_idx - int(trace_frames))

        for qi in range(num_queries):
            color = colors_bgr[qi]
            for t in range(hist_start + 1, frame_idx + 1):
                if not (bool(visibility[t - 1, qi]) and bool(visibility[t, qi])):
                    continue
                p0 = tracks_uv[t - 1, qi]
                p1 = tracks_uv[t, qi]
                if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                    continue
                cv2.line(
                    frame,
                    tuple(np.rint(p0).astype(np.int32)),
                    tuple(np.rint(p1).astype(np.int32)),
                    color,
                    int(line_thickness),
                    cv2.LINE_AA,
                )

            pt = tracks_uv[frame_idx, qi]
            if not np.isfinite(pt).all():
                continue
            center = tuple(np.rint(pt).astype(np.int32))
            if bool(visibility[frame_idx, qi]):
                cv2.circle(frame, center, int(point_radius) + 1, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, center, int(point_radius), color, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, center, int(point_radius) + 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, center, int(point_radius), color, 2, cv2.LINE_AA)

        cv2.putText(
            frame,
            f"GT tracks | frame {frame_idx + 1}/{num_frames} | Q={num_queries}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "solid=visible  hollow=occluded",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        frames_out.append(frame)

    video_name, _ = export_video_from_frames(
        video_rgb=np.stack(frames_out, axis=0),
        fps=fps,
        dst_video=Path(output_path),
    )
    return video_name


def render_track_comparison_videos(
    *,
    video_rgb: np.ndarray,
    gt_tracks_world: np.ndarray,
    pred_tracks_world: np.ndarray,
    pred_tracks_raw_world: Optional[np.ndarray] = None,
    visibility_tq: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    output_dir: Path,
    max_points: int = 300,
    trace_frames: int = 8,
    fps: float = 15.0,
) -> Dict[str, str]:
    """2D overlay + side-by-side 3D (GT vs aligned pred), matching Open-d4rt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_sub, pred_sub, vis_sub = _sample_track_subset(
        gt_tracks_world, pred_tracks_world, visibility_tq, max_points
    )
    pred_raw_sub = None
    if pred_tracks_raw_world is not None:
        # Keep the same sampled point indices by reusing the sampling choice via deterministic RNG.
        # `_sample_track_subset` uses rng seed=0 when subsampling, so calling it with same shapes
        # yields the same point indices.
        _, pred_raw_sub, _ = _sample_track_subset(
            gt_tracks_world, np.asarray(pred_tracks_raw_world), visibility_tq, max_points
        )
    gt_uv = project_world_tracks_to_uv(gt_sub, extrinsics_w2c, intrinsics)
    pred_uv = project_world_tracks_to_uv(pred_sub, extrinsics_w2c, intrinsics)
    colors = _track_colors(int(gt_sub.shape[1]))

    frames_2d = []
    for frame_idx in range(int(video_rgb.shape[0])):
        frame = np.asarray(video_rgb[frame_idx], dtype=np.uint8).copy()
        for qi in range(int(gt_sub.shape[1])):
            tint = tuple(int(v) for v in colors[qi].tolist())
            gt_line_color = tuple(
                int(0.55 * c + 0.45 * g) for c, g in zip(tint, _OVERLAY_GT_BGR)
            )
            pred_line_color = tuple(
                int(0.55 * c + 0.45 * p) for c, p in zip(tint, _OVERLAY_PRED_BGR)
            )
            for hist_idx in range(max(0, frame_idx - int(trace_frames)), frame_idx):
                if bool(vis_sub[hist_idx, qi]) and bool(vis_sub[hist_idx + 1, qi]):
                    p0, p1 = gt_uv[hist_idx, qi], gt_uv[hist_idx + 1, qi]
                    if np.isfinite(p0).all() and np.isfinite(p1).all():
                        cv2.line(
                            frame,
                            tuple(np.rint(p0).astype(np.int32)),
                            tuple(np.rint(p1).astype(np.int32)),
                            gt_line_color,
                            2,
                            cv2.LINE_AA,
                        )
                    p0p, p1p = pred_uv[hist_idx, qi], pred_uv[hist_idx + 1, qi]
                    if np.isfinite(p0p).all() and np.isfinite(p1p).all():
                        cv2.line(
                            frame,
                            tuple(np.rint(p0p).astype(np.int32)),
                            tuple(np.rint(p1p).astype(np.int32)),
                            pred_line_color,
                            1,
                            cv2.LINE_AA,
                        )
            if bool(vis_sub[frame_idx, qi]):
                gt_p, pred_p = gt_uv[frame_idx, qi], pred_uv[frame_idx, qi]
                if np.isfinite(gt_p).all():
                    cv2.circle(
                        frame,
                        tuple(np.rint(gt_p).astype(np.int32)),
                        4,
                        _OVERLAY_GT_BGR,
                        -1,
                        lineType=cv2.LINE_AA,
                    )
                if np.isfinite(pred_p).all():
                    cv2.drawMarker(
                        frame,
                        tuple(np.rint(pred_p).astype(np.int32)),
                        _OVERLAY_PRED_BGR,
                        markerType=cv2.MARKER_CROSS,
                        markerSize=12,
                        thickness=2,
                        line_type=cv2.LINE_AA,
                    )
        # Legend: green dot = GT, red cross = Pred (global-scale aligned).
        cv2.putText(
            frame,
            "GT",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            _OVERLAY_GT_BGR,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Pred",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            _OVERLAY_PRED_BGR,
            2,
            cv2.LINE_AA,
        )
        frames_2d.append(frame)

    frames_2d_np = np.stack(frames_2d, axis=0)
    video_2d_name, poster_2d_name = export_video_from_frames(
        video_rgb=frames_2d_np,
        fps=fps,
        dst_video=output_dir / "tracks_2d_overlay.mp4",
    )

    valid = np.isfinite(gt_sub).all(axis=-1) | np.isfinite(pred_sub).all(axis=-1)
    clouds = [gt_sub, pred_sub]
    if pred_raw_sub is not None:
        clouds.append(pred_raw_sub)
    flat = (
        np.concatenate([c[valid] for c in clouds], axis=0)
        if np.any(valid)
        else np.zeros((1, 3), dtype=np.float32)
    )
    xyz_min = np.nanmin(flat, axis=0)
    xyz_max = np.nanmax(flat, axis=0)
    center = (xyz_min + xyz_max) * 0.5
    half_extent = max(float(np.nanmax(xyz_max - xyz_min) * 0.55), 0.5)

    frames_3d = []
    frames_3d_overlay = []
    # Robust error scaling for colormap (meters in ref0/worldtrack ref0 coords).
    fin_err = _valid_track_mask(gt_sub, pred_sub, vis_sub)
    if np.any(fin_err):
        err_all = np.linalg.norm((pred_sub - gt_sub)[fin_err], axis=-1)
        # Use p90 so outliers don't saturate everything.
        err_scale = float(np.percentile(err_all, 90))
    else:
        err_scale = 1.0
    err_scale = max(err_scale, 1e-6)
    err_norm = Normalize(vmin=0.0, vmax=err_scale, clip=True)
    err_cmap = cm.get_cmap("plasma")
    for frame_idx in range(int(video_rgb.shape[0])):
        fig = plt.figure(figsize=(10, 5), dpi=140)
        ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
        ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
        for ax, title in ((ax_gt, "GT Tracks"), (ax_pred, "Pred (global scale aligned)")):
            ax.set_title(title)
            ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
            ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
            ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=24.0, azim=45.0)
        for qi in range(int(gt_sub.shape[1])):
            rgb = colors[qi].astype(np.float32) / 255.0
            t0 = max(0, frame_idx - int(trace_frames))
            gt_hist = gt_sub[t0 : frame_idx + 1, qi]
            pred_hist = pred_sub[t0 : frame_idx + 1, qi]
            gt_ok = np.isfinite(gt_hist).all(axis=-1) & vis_sub[t0 : frame_idx + 1, qi]
            pred_ok = np.isfinite(pred_hist).all(axis=-1) & vis_sub[t0 : frame_idx + 1, qi]
            if int(gt_ok.sum()) >= 2:
                pts = gt_hist[gt_ok]
                ax_gt.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=rgb, linewidth=1.2)
            if int(pred_ok.sum()) >= 2:
                pts = pred_hist[pred_ok]
                ax_pred.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=rgb, linewidth=1.2)
            if bool(vis_sub[frame_idx, qi]):
                gt_now = gt_sub[frame_idx, qi]
                pred_now = pred_sub[frame_idx, qi]
                if np.isfinite(gt_now).all():
                    ax_gt.scatter(gt_now[0], gt_now[1], gt_now[2], color=rgb, s=18)
                if np.isfinite(pred_now).all():
                    ax_pred.scatter(
                        pred_now[0], pred_now[1], pred_now[2], color=rgb, s=18, marker="x"
                    )
        fig.tight_layout()
        frames_3d.append(_figure_to_rgb(fig))
        plt.close(fig)

        # Overlay view: GT + Pred in same 3D axes, with error vectors colored by magnitude.
        fig = plt.figure(figsize=(7.2, 5.4), dpi=140)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_title("GT vs Pred overlay (error vectors colored by |Pred-GT|)")
        ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
        ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
        ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=24.0, azim=45.0)

        t0 = max(0, frame_idx - int(trace_frames))
        for qi in range(int(gt_sub.shape[1])):
            rgb = colors[qi].astype(np.float32) / 255.0
            gt_hist = gt_sub[t0 : frame_idx + 1, qi]
            pred_hist = pred_sub[t0 : frame_idx + 1, qi]
            ok = np.isfinite(gt_hist).all(axis=-1) & np.isfinite(pred_hist).all(axis=-1) & vis_sub[
                t0 : frame_idx + 1, qi
            ]
            if int(ok.sum()) >= 2:
                pts_gt = gt_hist[ok]
                pts_pred = pred_hist[ok]
                ax.plot(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2], color=rgb, linewidth=1.0, alpha=0.75)
                ax.plot(
                    pts_pred[:, 0],
                    pts_pred[:, 1],
                    pts_pred[:, 2],
                    color=rgb,
                    linewidth=1.0,
                    alpha=0.35,
                    linestyle="--",
                )
            if bool(vis_sub[frame_idx, qi]):
                gt_now = gt_sub[frame_idx, qi]
                pred_now = pred_sub[frame_idx, qi]
                if np.isfinite(gt_now).all() and np.isfinite(pred_now).all():
                    e = float(np.linalg.norm(pred_now - gt_now))
                    c = err_cmap(err_norm(e))
                    # Error vector segment GT -> Pred.
                    ax.plot(
                        [gt_now[0], pred_now[0]],
                        [gt_now[1], pred_now[1]],
                        [gt_now[2], pred_now[2]],
                        color=c,
                        linewidth=2.0,
                        alpha=0.9,
                    )
                    ax.scatter(gt_now[0], gt_now[1], gt_now[2], color=(0.15, 0.8, 0.25), s=22, alpha=0.95)
                    ax.scatter(
                        pred_now[0],
                        pred_now[1],
                        pred_now[2],
                        color=(0.9, 0.15, 0.15),
                        s=28,
                        marker="x",
                        alpha=0.95,
                    )

        # Colorbar for error magnitude (meters, clipped at p90).
        sm = cm.ScalarMappable(norm=err_norm, cmap=err_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(f"|Pred-GT| (m), clipped @p90={err_scale:.3f}")
        try:
            fig.tight_layout()
        except RuntimeError:
            pass
        frames_3d_overlay.append(_figure_to_rgb(fig))
        plt.close(fig)

    frames_3d_np = np.stack(frames_3d, axis=0)
    video_3d_name, poster_3d_name = export_video_from_frames(
        video_rgb=frames_3d_np,
        fps=fps,
        dst_video=output_dir / "tracks_3d.mp4",
    )
    frames_3d_overlay_np = np.stack(frames_3d_overlay, axis=0)
    video_3d_overlay_name, poster_3d_overlay_name = export_video_from_frames(
        video_rgb=frames_3d_overlay_np,
        fps=fps,
        dst_video=output_dir / "tracks_3d_overlay.mp4",
    )

    # Triple overlay: GT + Pred(aligned) + Pred(raw), plus both distances to GT.
    triple_paths: Dict[str, str] = {}
    if pred_raw_sub is not None:
        fin_err_raw = _valid_track_mask(gt_sub, pred_raw_sub, vis_sub)
        if np.any(fin_err_raw):
            err_all_raw = np.linalg.norm((pred_raw_sub - gt_sub)[fin_err_raw], axis=-1)
            err_scale_raw = float(np.percentile(err_all_raw, 90))
        else:
            err_scale_raw = 1.0
        err_scale_raw = max(err_scale_raw, 1e-6)
        err_norm_raw = Normalize(vmin=0.0, vmax=err_scale_raw, clip=True)

        frames_3d_triple = []
        for frame_idx in range(int(video_rgb.shape[0])):
            fig = plt.figure(figsize=(7.6, 5.6), dpi=140)
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            ax.set_title("GT vs Pred(aligned) vs Pred(raw) (vectors: GT→aligned solid, GT→raw dashed)")
            ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
            ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
            ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=24.0, azim=45.0)

            t0 = max(0, frame_idx - int(trace_frames))
            for qi in range(int(gt_sub.shape[1])):
                rgb = colors[qi].astype(np.float32) / 255.0
                gt_hist = gt_sub[t0 : frame_idx + 1, qi]
                pred_hist = pred_sub[t0 : frame_idx + 1, qi]
                raw_hist = pred_raw_sub[t0 : frame_idx + 1, qi]
                ok = (
                    np.isfinite(gt_hist).all(axis=-1)
                    & np.isfinite(pred_hist).all(axis=-1)
                    & np.isfinite(raw_hist).all(axis=-1)
                    & vis_sub[t0 : frame_idx + 1, qi]
                )
                if int(ok.sum()) >= 2:
                    pts_gt = gt_hist[ok]
                    pts_al = pred_hist[ok]
                    pts_raw = raw_hist[ok]
                    ax.plot(
                        pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2], color=rgb, linewidth=1.0, alpha=0.75
                    )
                    ax.plot(
                        pts_al[:, 0],
                        pts_al[:, 1],
                        pts_al[:, 2],
                        color=rgb,
                        linewidth=1.0,
                        alpha=0.35,
                        linestyle="--",
                    )
                    ax.plot(
                        pts_raw[:, 0],
                        pts_raw[:, 1],
                        pts_raw[:, 2],
                        color=rgb,
                        linewidth=0.9,
                        alpha=0.25,
                        linestyle=":",
                    )

                if bool(vis_sub[frame_idx, qi]):
                    gt_now = gt_sub[frame_idx, qi]
                    al_now = pred_sub[frame_idx, qi]
                    raw_now = pred_raw_sub[frame_idx, qi]
                    if np.isfinite(gt_now).all():
                        ax.scatter(gt_now[0], gt_now[1], gt_now[2], color=(0.15, 0.8, 0.25), s=22, alpha=0.95)
                    if np.isfinite(al_now).all():
                        ax.scatter(al_now[0], al_now[1], al_now[2], color=(0.9, 0.15, 0.15), s=28, marker="x")
                    if np.isfinite(raw_now).all():
                        ax.scatter(raw_now[0], raw_now[1], raw_now[2], color=(1.0, 0.55, 0.1), s=18, marker="^")

                    if np.isfinite(gt_now).all() and np.isfinite(al_now).all():
                        e_al = float(np.linalg.norm(al_now - gt_now))
                        c_al = err_cmap(err_norm(e_al))
                        ax.plot(
                            [gt_now[0], al_now[0]],
                            [gt_now[1], al_now[1]],
                            [gt_now[2], al_now[2]],
                            color=c_al,
                            linewidth=2.0,
                            alpha=0.9,
                        )
                    if np.isfinite(gt_now).all() and np.isfinite(raw_now).all():
                        e_raw = float(np.linalg.norm(raw_now - gt_now))
                        c_raw = err_cmap(err_norm_raw(e_raw))
                        ax.plot(
                            [gt_now[0], raw_now[0]],
                            [gt_now[1], raw_now[1]],
                            [gt_now[2], raw_now[2]],
                            color=c_raw,
                            linewidth=1.6,
                            alpha=0.85,
                            linestyle="--",
                        )

            sm_al = cm.ScalarMappable(norm=err_norm, cmap=err_cmap)
            sm_al.set_array([])
            cbar1 = fig.colorbar(sm_al, ax=ax, fraction=0.04, pad=0.02)
            cbar1.set_label(f"|aligned-GT| (m), clipped @p90={err_scale:.3f}")

            try:
                fig.tight_layout()
            except RuntimeError:
                pass
            frames_3d_triple.append(_figure_to_rgb(fig))
            plt.close(fig)

        frames_3d_triple_np = np.stack(frames_3d_triple, axis=0)
        video_3d_triple_name, poster_3d_triple_name = export_video_from_frames(
            video_rgb=frames_3d_triple_np,
            fps=fps,
            dst_video=output_dir / "tracks_3d_triple_overlay.mp4",
        )
        triple_paths = {
            "tracks_3d_triple_overlay": video_3d_triple_name,
            "tracks_3d_triple_overlay_poster": poster_3d_triple_name,
        }
    return {
        "tracks_2d_overlay": video_2d_name,
        "tracks_2d_overlay_poster": poster_2d_name,
        "tracks_3d": video_3d_name,
        "tracks_3d_poster": poster_3d_name,
        "tracks_3d_overlay": video_3d_overlay_name,
        "tracks_3d_overlay_poster": poster_3d_overlay_name,
        **triple_paths,
    }


# PLY / 2D overlay colors (RGB 0–255).
_PLY_GT_COLOR = np.array([40, 200, 80], dtype=np.uint8)       # green
_PLY_PRED_ALIGNED_COLOR = np.array([230, 50, 50], dtype=np.uint8)  # red
_PLY_PRED_RAW_COLOR = np.array([255, 160, 40], dtype=np.uint8)   # orange
_OVERLAY_GT_BGR = (40, 200, 80)
_OVERLAY_PRED_BGR = (50, 50, 230)


def write_ply_xyz_rgb(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write ASCII PLY with per-vertex RGB (MeshLab / CloudCompare compatible)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if pts.shape[0] != cols.shape[0]:
        raise ValueError(f"points/colors length mismatch: {pts.shape[0]} vs {cols.shape[0]}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _find_track3d(outputs_t: Any, tgt_index: int = 0, src_index: int = 0) -> Any:
    """Best-effort lookup of Track3DOutput for (src, tgt) within one timestep output."""
    track_dict = getattr(outputs_t, "track_3d", None)
    if not isinstance(track_dict, dict):
        return None
    tr = track_dict.get(tgt_index, track_dict.get(int(tgt_index), None))
    if tr is not None and getattr(tr, "tgt_index", tgt_index) == tgt_index:
        if getattr(tr, "src_index", src_index) == src_index:
            return tr
    for v in track_dict.values():
        if getattr(v, "src_index", None) == src_index and getattr(v, "tgt_index", None) == tgt_index:
            return v
    return None


# Sparse frame-0 PLY legend (RGB 0–255).
_SPARSE_GT_GREEN = np.array([40, 200, 80], dtype=np.uint8)
_SPARSE_PRED_RED = np.array([230, 50, 50], dtype=np.uint8)


def export_worldtrack_frame0_vis_ply(
    out_path: Path,
    *,
    gt_tracks_ref0_frame0: np.ndarray,
    pred_tracks_ref0_frame0: np.ndarray,
    outputs_dense: Any | None = None,
    video_rgb_frame0: np.ndarray | None = None,
    align_pred: bool = True,
) -> Dict[str, Any]:
    """Frame-0 PLY: sparse GT (green) + sparse pred (red) + optional dense pred (image RGB).

    - No dense pseudo-GT pointmap.
    - Dense pred: full-UV ``(src=0,tgt=0)`` ``warp3d``, globally scaled like sparse pred.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt = np.asarray(gt_tracks_ref0_frame0, dtype=np.float64).reshape(-1, 3)
    pred = np.asarray(pred_tracks_ref0_frame0, dtype=np.float64).reshape(-1, 3)
    scale = 1.0
    if align_pred:
        scale = compute_scale_factor_global(gt, pred)
        pred = pred * float(scale)

    fin_gt = np.isfinite(gt).all(axis=-1)
    fin_pred = np.isfinite(pred).all(axis=-1)
    gt = gt[fin_gt]
    pred = pred[fin_pred]

    points_all = [gt, pred]
    colors_all = [
        np.tile(_SPARSE_GT_GREEN.reshape(1, 3), (gt.shape[0], 1)),
        np.tile(_SPARSE_PRED_RED.reshape(1, 3), (pred.shape[0], 1)),
    ]

    meta: Dict[str, Any] = {
        "ply_path": str(out_path),
        "frame": 0,
        "coordinate": "ref0",
        "scale_global": float(scale),
        "gt_sparse_points": int(gt.shape[0]),
        "pred_sparse_points": int(pred.shape[0]),
        "pred_dense_points": 0,
        "gt_color_rgb": _SPARSE_GT_GREEN.tolist(),
        "pred_sparse_color_rgb": _SPARSE_PRED_RED.tolist(),
        "pred_dense_color": "unknown",
    }

    if outputs_dense is not None:
        d0 = outputs_dense[0] if isinstance(outputs_dense, (list, tuple)) else outputs_dense
        tr_d = _find_track3d(d0, tgt_index=0, src_index=0)
        if tr_d is None or getattr(tr_d, "warp3d", None) is None:
            raise RuntimeError("frame0 dense Track3DOutput (0,0) missing warp3d.")
        pred_dense = np.asarray(getattr(tr_d, "warp3d"), dtype=np.float64).reshape(-1, 3) * float(scale)
        fin_d = np.isfinite(pred_dense).all(axis=-1)
        pred_dense = pred_dense[fin_d]
        dense_cols = None
        color_source = "solid_red"
        query_uv = getattr(tr_d, "query_uv", None)
        qh = getattr(tr_d, "query_height", None)
        qw = getattr(tr_d, "query_width", None)
        # Use query_uv (actual dense grid), NOT warp3d_uv (3D reprojection can drift → wrong RGB).
        if (
            query_uv is not None
            and video_rgb_frame0 is not None
            and qh is not None
            and qw is not None
            and int(qh) > 0
            and int(qw) > 0
        ):
            uv_norm = np.asarray(query_uv, dtype=np.float64).reshape(-1, 2)[fin_d]
            dense_cols = sample_rgb_at_norm_uv(
                video_rgb_frame0,
                uv_norm,
                query_height=int(qh),
                query_width=int(qw),
            )
            color_source = "image_rgb@query_uv_norm"
        if dense_cols is None:
            dense_cols = np.tile(_SPARSE_PRED_RED.reshape(1, 3), (pred_dense.shape[0], 1))
        points_all.append(pred_dense)
        colors_all.append(np.asarray(dense_cols, dtype=np.uint8))
        meta["pred_dense_points"] = int(pred_dense.shape[0])
        meta["pred_dense_color"] = color_source
        meta["pred_dense_query_hw"] = [int(qh) if qh is not None else 0, int(qw) if qw is not None else 0]
        if video_rgb_frame0 is not None:
            meta["pred_dense_image_hw"] = [
                int(np.asarray(video_rgb_frame0).shape[0]),
                int(np.asarray(video_rgb_frame0).shape[1]),
            ]

    pts = np.concatenate(points_all, axis=0) if points_all else np.zeros((0, 3))
    cols = np.concatenate(colors_all, axis=0) if colors_all else np.zeros((0, 3), dtype=np.uint8)
    write_ply_xyz_rgb(out_path, pts, cols)

    meta["note"] = (
        "Green=sparse GT queries; Red=sparse pred (aligned); "
        "Dense pred=full-UV warp3d (aligned); dense RGB from query_uv on model-resolution image."
    )
    return meta


def export_frame0_gt_colored_ply(
    out_path: Path,
    *,
    gt_tracks_ref0_frame0: np.ndarray,
    video_rgb_frame0: np.ndarray,
    gt_tracks_uv_pixel_frame0: np.ndarray,
    visibility_frame0: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Frame-0 GT query point cloud colored by image RGB at query UV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt = np.asarray(gt_tracks_ref0_frame0, dtype=np.float64).reshape(-1, 3)
    uv = np.asarray(gt_tracks_uv_pixel_frame0, dtype=np.float64).reshape(-1, 2)
    fin = np.isfinite(gt).all(axis=-1) & np.isfinite(uv).all(axis=-1)
    if visibility_frame0 is not None:
        vis = np.asarray(visibility_frame0, dtype=bool).reshape(-1)
        fin = fin & vis
    gt = gt[fin]
    uv = uv[fin]

    video_rgb_frame0 = np.asarray(video_rgb_frame0, dtype=np.uint8)
    h_img, w_img = int(video_rgb_frame0.shape[0]), int(video_rgb_frame0.shape[1])
    u = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, max(w_img - 1, 0))
    v = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, max(h_img - 1, 0))
    cols = video_rgb_frame0[v, u]

    write_ply_xyz_rgb(out_path, gt, cols)
    return {
        "ply_path": str(out_path),
        "frame": 0,
        "layer": "gt_queries",
        "num_points": int(gt.shape[0]),
        "color_mode": "image_rgb@query_uv_pixel",
        "image_hw": [h_img, w_img],
    }


def export_frame0_pred_dense_colored_ply(
    out_path: Path,
    *,
    gt_tracks_ref0_frame0: np.ndarray,
    pred_tracks_ref0_frame0: np.ndarray,
    outputs_dense: Any,
    video_rgb_frame0: np.ndarray,
    align_pred: bool = True,
) -> Dict[str, Any]:
    """Frame-0 dense warp3d prediction colored by model-resolution image RGB."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt = np.asarray(gt_tracks_ref0_frame0, dtype=np.float64).reshape(-1, 3)
    pred = np.asarray(pred_tracks_ref0_frame0, dtype=np.float64).reshape(-1, 3)
    scale = 1.0
    if align_pred:
        scale = compute_scale_factor_global(gt, pred)

    d0 = outputs_dense[0] if isinstance(outputs_dense, (list, tuple)) else outputs_dense
    tr_d = _find_track3d(d0, tgt_index=0, src_index=0)
    if tr_d is None or getattr(tr_d, "warp3d", None) is None:
        raise RuntimeError("frame0 dense Track3DOutput (0,0) missing warp3d.")
    pred_dense = np.asarray(getattr(tr_d, "warp3d"), dtype=np.float64).reshape(-1, 3) * float(scale)
    fin_d = np.isfinite(pred_dense).all(axis=-1)
    pred_dense = pred_dense[fin_d]

    dense_cols = None
    color_source = "solid_red"
    query_uv = getattr(tr_d, "query_uv", None)
    qh = getattr(tr_d, "query_height", None)
    qw = getattr(tr_d, "query_width", None)
    if (
        query_uv is not None
        and video_rgb_frame0 is not None
        and qh is not None
        and qw is not None
        and int(qh) > 0
        and int(qw) > 0
    ):
        uv_norm = np.asarray(query_uv, dtype=np.float64).reshape(-1, 2)[fin_d]
        dense_cols = sample_rgb_at_norm_uv(
            video_rgb_frame0,
            uv_norm,
            query_height=int(qh),
            query_width=int(qw),
        )
        color_source = "image_rgb@query_uv_norm"
    if dense_cols is None:
        dense_cols = np.tile(_SPARSE_PRED_RED.reshape(1, 3), (pred_dense.shape[0], 1))

    write_ply_xyz_rgb(out_path, pred_dense, np.asarray(dense_cols, dtype=np.uint8))
    return {
        "ply_path": str(out_path),
        "frame": 0,
        "layer": "pred_dense",
        "num_points": int(pred_dense.shape[0]),
        "scale_global": float(scale),
        "color_mode": color_source,
        "query_hw": [int(qh) if qh is not None else 0, int(qw) if qw is not None else 0],
        "image_hw": [
            int(np.asarray(video_rgb_frame0).shape[0]),
            int(np.asarray(video_rgb_frame0).shape[1]),
        ],
    }


def export_worldtrack_frame0_sparse_gt_pred_ply(
    out_path: Path,
    *,
    gt_tracks_ref0_frame0: np.ndarray,
    pred_tracks_ref0_frame0: np.ndarray,
    align_pred: bool = True,
) -> Dict[str, Any]:
    """Backward-compat alias: sparse-only (no dense pred)."""
    return export_worldtrack_frame0_vis_ply(
        out_path,
        gt_tracks_ref0_frame0=gt_tracks_ref0_frame0,
        pred_tracks_ref0_frame0=pred_tracks_ref0_frame0,
        outputs_dense=None,
        align_pred=align_pred,
    )


def export_worldtrack_frame0_dense_warp3d_vs_gt_pointmap_ply(
    out_path: Path,
    *,
    outputs_dense: Any,
    outputs_query: Any,
    gt_tracks_world_frame0_queries: np.ndarray,
    gt_pointmap_ref0: Optional[np.ndarray] = None,
    gt_pointmap_colors: Optional[np.ndarray] = None,
    video_rgb_frame0: Optional[np.ndarray] = None,
    include_dense: bool = True,
    include_query: bool = True,
    pred_color_dense_rgb: Tuple[int, int, int] = (220, 40, 40),
    pred_color_query_rgb: Tuple[int, int, int] = (40, 40, 230),
) -> Dict[str, Any]:
    """Export a single PLY for frame-0 only.

    Contains:
    - GT pointmap (colored) in ref0 coordinates.
    - Predicted warp3d points at frame0:
      - dense (per-pixel full query bank) from outputs_dense[0] track (0,0)
      - query (WorldTrack queries) from outputs_query[0] track (0,0)

    Alignment:
    - Compute global median scale using *query* predictions vs GT(query) at frame0.
    - Apply this scale to both dense and query predictions.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- GT pointmap ---
    # Try to recover missing GT pointmap / colors from model outputs when caller doesn't provide them.
    if gt_pointmap_ref0 is None or gt_pointmap_colors is None:
        ref0 = outputs_query[0] if isinstance(outputs_query, (list, tuple)) else outputs_query
        if gt_pointmap_ref0 is None:
            gt_pointmap_ref0 = getattr(ref0, "pointmap_gt_global", None)
        if gt_pointmap_colors is None:
            gt_pointmap_colors = getattr(ref0, "pointmap_color", None)
    gt_pm = np.asarray(gt_pointmap_ref0) if gt_pointmap_ref0 is not None else np.zeros((0, 3))
    gt_pm = gt_pm.reshape(-1, 3).astype(np.float64, copy=False)

    if gt_pointmap_colors is not None:
        gt_col = np.asarray(gt_pointmap_colors).reshape(-1, 3).astype(np.uint8, copy=False)
    elif video_rgb_frame0 is not None and gt_pm.shape[0] > 0:
        # Color GT pointmap using frame-0 RGB in raster order.
        rgb0 = np.asarray(video_rgb_frame0, dtype=np.uint8)
        gt_col = rgb0.reshape(-1, 3)
    else:
        gt_col = np.zeros((gt_pm.shape[0], 3), dtype=np.uint8)
    fin_gt = np.isfinite(gt_pm).all(axis=-1)
    gt_pm = gt_pm[fin_gt]
    if gt_col.shape[0] == fin_gt.shape[0]:
        gt_col = gt_col[fin_gt]
    else:
        gt_col = gt_col[: gt_pm.shape[0]]

    # --- Query pred for alignment ---
    q0 = outputs_query[0] if isinstance(outputs_query, (list, tuple)) else outputs_query
    tr_q = _find_track3d(q0, tgt_index=0, src_index=0)
    if tr_q is None or getattr(tr_q, "warp3d", None) is None:
        raise RuntimeError("frame0 query Track3DOutput (0,0) missing warp3d; cannot align.")
    pred_q = np.asarray(getattr(tr_q, "warp3d"), dtype=np.float64).reshape(-1, 3)
    gt_q0 = np.asarray(gt_tracks_world_frame0_queries, dtype=np.float64).reshape(-1, 3)
    scale = compute_scale_factor_global(gt_q0, pred_q)

    points_all: list[np.ndarray] = []
    colors_all: list[np.ndarray] = []

    # Add GT pointmap first (colored by RGB).
    points_all.append(gt_pm)
    colors_all.append(gt_col)

    # --- Dense pred ---
    meta: Dict[str, Any] = {
        "scale_global": float(scale),
        "gt_pointmap_points": int(gt_pm.shape[0]),
        "gt_query_points": int(gt_q0.shape[0]),
        "pred_dense_points": 0,
        "pred_query_points": 0,
    }

    if include_dense:
        d0 = outputs_dense[0] if isinstance(outputs_dense, (list, tuple)) else outputs_dense
        tr_d = _find_track3d(d0, tgt_index=0, src_index=0)
        if tr_d is None or getattr(tr_d, "warp3d", None) is None:
            raise RuntimeError("frame0 dense Track3DOutput (0,0) missing warp3d.")
        pred_dense = np.asarray(getattr(tr_d, "warp3d"), dtype=np.float64).reshape(-1, 3) * float(scale)
        fin = np.isfinite(pred_dense).all(axis=-1)
        pred_dense = pred_dense[fin]
        points_all.append(pred_dense)
        # Dense prediction colors: sample from frame-0 RGB via warp3d_uv if available.
        dense_cols = None
        uv = getattr(tr_d, "warp3d_uv", None)
        if uv is not None and video_rgb_frame0 is not None:
            uv0 = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
            uv0 = uv0[fin]
            video = np.asarray(video_rgb_frame0, dtype=np.uint8)[None, ...]  # [1,H,W,3]
            tracks_uv = uv0[None, ...]  # [1,N,2]
            mask = np.ones((1, int(uv0.shape[0])), dtype=bool)
            dense_cols = sample_video_rgb_at_track_uv(video, tracks_uv, mask)
        if dense_cols is None:
            # Fallback: if dense point count matches pointmap colors, reuse them (assumes raster order).
            if gt_col.shape[0] == pred_dense.shape[0]:
                dense_cols = gt_col.copy()
            else:
                dense_cols = np.tile(
                    np.asarray(pred_color_dense_rgb, dtype=np.uint8).reshape(1, 3),
                    (pred_dense.shape[0], 1),
                )
        colors_all.append(np.asarray(dense_cols, dtype=np.uint8))
        meta["pred_dense_points"] = int(pred_dense.shape[0])

    # --- Query pred (extra, may overlap dense) ---
    if include_query:
        pred_q_aligned = pred_q * float(scale)
        fin = np.isfinite(pred_q_aligned).all(axis=-1)
        pred_q_aligned = pred_q_aligned[fin]
        points_all.append(pred_q_aligned)
        # Keep query predictions in a solid color for readability (dense already carries RGB).
        colors_all.append(
            np.tile(
                np.asarray(pred_color_query_rgb, dtype=np.uint8).reshape(1, 3),
                (pred_q_aligned.shape[0], 1),
            )
        )
        meta["pred_query_points"] = int(pred_q_aligned.shape[0])

    pts = np.concatenate(points_all, axis=0) if points_all else np.zeros((0, 3), dtype=np.float64)
    cols = np.concatenate(colors_all, axis=0) if colors_all else np.zeros((0, 3), dtype=np.uint8)
    write_ply_xyz_rgb(out_path, pts, cols)
    meta["ply_path"] = str(out_path)
    return meta


def _valid_track_mask(
    gt_tracks: np.ndarray,
    pred_tracks: np.ndarray,
    visibility: Optional[np.ndarray],
) -> np.ndarray:
    fin = np.isfinite(gt_tracks).all(axis=-1) & np.isfinite(pred_tracks).all(axis=-1)
    if visibility is not None:
        fin &= np.asarray(visibility, dtype=bool)
    return fin


def _track_error_stats(
    gt_tracks: np.ndarray,
    pred_tracks: np.ndarray,
    visibility: Optional[np.ndarray],
) -> Dict[str, float]:
    fin = _valid_track_mask(gt_tracks, pred_tracks, visibility)
    if not np.any(fin):
        return {"count": 0.0, "epe_mean_m": float("nan"), "epe_median_m": float("nan")}
    err = np.linalg.norm(
        np.asarray(gt_tracks, dtype=np.float64)[fin]
        - np.asarray(pred_tracks, dtype=np.float64)[fin],
        axis=-1,
    )
    return {
        "count": float(fin.sum()),
        "epe_mean_m": float(err.mean()),
        "epe_median_m": float(np.median(err)),
        "epe_p90_m": float(np.percentile(err, 90)),
        "epe_max_m": float(err.max()),
    }


def _image_color_mode_enabled(
    *,
    use_image_colors: bool,
    video_rgb: Optional[np.ndarray],
    extrinsics_w2c: Optional[np.ndarray],
    intrinsics: Optional[np.ndarray],
) -> bool:
    return (
        bool(use_image_colors)
        and video_rgb is not None
        and extrinsics_w2c is not None
        and intrinsics is not None
    )


def _colors_for_tracks(
    *,
    tracks_ref0: np.ndarray,
    valid_mask: np.ndarray,
    video_rgb: np.ndarray,
    tracks_uv_pixel: Optional[np.ndarray],
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    use_image_colors: bool,
) -> Tuple[np.ndarray, str]:
    """Sample per-point RGB; prefer dataset ``tracks_uv`` (pixel) when provided."""
    tracks = np.asarray(tracks_ref0, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if not _image_color_mode_enabled(
        use_image_colors=use_image_colors,
        video_rgb=video_rgb,
        extrinsics_w2c=extrinsics_w2c,
        intrinsics=intrinsics,
    ):
        return np.tile(_PLY_PRED_ALIGNED_COLOR, (int(mask.sum()), 1)), "solid_legend"

    if tracks_uv_pixel is not None:
        uv = np.asarray(tracks_uv_pixel, dtype=np.float64)
        cols = sample_video_rgb_at_track_uv(video_rgb, uv, mask)
        return cols, "video_rgb@tracks_uv_pixel"

    uv = project_world_tracks_to_uv(tracks, extrinsics_w2c, intrinsics)
    cols = sample_video_rgb_at_track_uv(video_rgb, uv, mask)
    return cols, "video_rgb@projected_uv"


def export_tracks_imagecolor_ply(
    out_path: Path,
    *,
    tracks_ref0: np.ndarray,
    reference_tracks_ref0: np.ndarray,
    visibility_tq: Optional[np.ndarray],
    video_rgb: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    tracks_uv_pixel: Optional[np.ndarray] = None,
    use_image_colors: bool = True,
    layer_name: str = "tracks",
) -> Dict[str, Any]:
    """Single-layer PLY: 3D points colored by original video RGB (one point cloud)."""
    ref = np.asarray(reference_tracks_ref0, dtype=np.float64)
    tr = np.asarray(tracks_ref0, dtype=np.float64)
    fin = _valid_track_mask(ref, tr, visibility_tq)
    pts = tr[fin]
    cols, color_src = _colors_for_tracks(
        tracks_ref0=tr,
        valid_mask=fin,
        video_rgb=video_rgb,
        tracks_uv_pixel=tracks_uv_pixel,
        extrinsics_w2c=extrinsics_w2c,
        intrinsics=intrinsics,
        use_image_colors=use_image_colors,
    )
    write_ply_xyz_rgb(out_path, pts, cols)
    return {
        "ply_path": str(out_path),
        "layer": layer_name,
        "num_points": int(pts.shape[0]),
        "color_mode": color_src if use_image_colors else "solid_legend",
        "color_source": color_src,
    }


def export_tracks_pred_gt_ply(
    out_path: Path,
    *,
    gt_tracks_world: np.ndarray,
    pred_tracks_ref0: np.ndarray,
    visibility_tq: Optional[np.ndarray] = None,
    pred_tracks_raw: Optional[np.ndarray] = None,
    include_raw_pred: bool = False,
    label_gt: str = "gt",
    label_pred: str = "pred_aligned",
    video_rgb: Optional[np.ndarray] = None,
    extrinsics_w2c: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    gt_tracks_uv_pixel: Optional[np.ndarray] = None,
    use_image_colors: bool = True,
) -> Dict[str, Any]:
    """Backward-compat alias: writes **pred aligned only** as one image-colored cloud."""
    del include_raw_pred, pred_tracks_raw, label_gt, label_pred
    if video_rgb is None or extrinsics_w2c is None or intrinsics is None:
        raise ValueError("export_tracks_pred_gt_ply requires video_rgb and camera parameters.")
    return export_tracks_imagecolor_ply(
        out_path,
        tracks_ref0=pred_tracks_ref0,
        reference_tracks_ref0=gt_tracks_world,
        visibility_tq=visibility_tq,
        video_rgb=video_rgb,
        extrinsics_w2c=extrinsics_w2c,
        intrinsics=intrinsics,
        tracks_uv_pixel=None,
        use_image_colors=use_image_colors,
        layer_name="pred_aligned",
    )


def export_tracks_error_lines_ply(
    out_path: Path,
    *,
    gt_tracks_world: np.ndarray,
    pred_tracks_ref0: np.ndarray,
    visibility_tq: Optional[np.ndarray] = None,
    max_lines: int = 5000,
    worst_first: bool = True,
    video_rgb: Optional[np.ndarray] = None,
    extrinsics_w2c: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    use_image_colors: bool = True,
) -> Dict[str, Any]:
    """PLY segment endpoints GT→Pred for largest errors (video RGB or yellow)."""
    gt = np.asarray(gt_tracks_world, dtype=np.float64)
    pred = np.asarray(pred_tracks_ref0, dtype=np.float64)
    fin = _valid_track_mask(gt, pred, visibility_tq)
    if not np.any(fin):
        write_ply_xyz_rgb(out_path, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))
        return {"num_lines": 0}

    gt_f = gt[fin]
    pred_f = pred[fin]
    err = np.linalg.norm(pred_f - gt_f, axis=-1)
    order = np.argsort(err)[::-1] if worst_first else np.arange(err.shape[0])
    if max_lines > 0:
        order = order[: int(max_lines)]

    seg_gt = gt_f[order]
    seg_pred = pred_f[order]
    pts = np.stack([seg_gt, seg_pred], axis=1).reshape(-1, 3)

    image_color_mode = (
        bool(use_image_colors)
        and video_rgb is not None
        and extrinsics_w2c is not None
        and intrinsics is not None
    )
    if image_color_mode:
        gt_uv = project_world_tracks_to_uv(gt, extrinsics_w2c, intrinsics)
        pred_uv = project_world_tracks_to_uv(pred, extrinsics_w2c, intrinsics)
        gt_cols_all = sample_video_rgb_at_track_uv(video_rgb, gt_uv, fin)
        pred_cols_all = sample_video_rgb_at_track_uv(video_rgb, pred_uv, fin)
        cols = np.stack([gt_cols_all[order], pred_cols_all[order]], axis=1).reshape(-1, 3)
        color_note = "video_rgb@endpoints"
    else:
        line_color = np.array([255, 220, 60], dtype=np.uint8)
        cols = np.tile(line_color, (pts.shape[0], 1))
        color_note = line_color.tolist()

    write_ply_xyz_rgb(out_path, pts, cols)
    return {
        "num_lines": int(order.shape[0]),
        "max_error_m": float(err[order[0]]) if order.size else float("nan"),
        "line_color_rgb": color_note,
    }


def export_tracks_debug_ply_package(
    out_dir: Path,
    *,
    gt_tracks_world: np.ndarray,
    pred_tracks_ref0: np.ndarray,
    pred_tracks_raw: Optional[np.ndarray] = None,
    visibility_tq: Optional[np.ndarray] = None,
    scale_global: Optional[float] = None,
    video_rgb: Optional[np.ndarray] = None,
    extrinsics_w2c: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    gt_tracks_uv_pixel: Optional[np.ndarray] = None,
    use_image_colors: bool = True,
) -> Dict[str, Any]:
    """Write separate image-RGB PLYs under ``out_dir/vis/`` (one layer per file)."""
    if video_rgb is None or extrinsics_w2c is None or intrinsics is None:
        raise ValueError("PLY export requires video_rgb, extrinsics_w2c, and intrinsics.")

    vis_dir = Path(out_dir) / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    raw = pred_tracks_raw
    if raw is None and scale_global is not None and np.isfinite(scale_global) and scale_global > 0:
        raw = np.asarray(pred_tracks_ref0, dtype=np.float64) / float(scale_global)

    ply_kw = dict(
        visibility_tq=visibility_tq,
        video_rgb=video_rgb,
        extrinsics_w2c=extrinsics_w2c,
        intrinsics=intrinsics,
        use_image_colors=use_image_colors,
    )

    meta_gt = export_tracks_imagecolor_ply(
        vis_dir / "tracks_gt_imagecolor.ply",
        tracks_ref0=gt_tracks_world,
        reference_tracks_ref0=gt_tracks_world,
        tracks_uv_pixel=gt_tracks_uv_pixel,
        layer_name="gt",
        **ply_kw,
    )
    meta_pred = export_tracks_imagecolor_ply(
        vis_dir / "tracks_pred_aligned_imagecolor.ply",
        tracks_ref0=pred_tracks_ref0,
        reference_tracks_ref0=gt_tracks_world,
        layer_name="pred_aligned",
        **ply_kw,
    )
    # Primary file for older scripts: pred-only, video RGB (not green/red/orange stack).
    meta_main = export_tracks_imagecolor_ply(
        vis_dir / "tracks_pred_gt_aligned.ply",
        tracks_ref0=pred_tracks_ref0,
        reference_tracks_ref0=gt_tracks_world,
        layer_name="pred_aligned",
        **ply_kw,
    )

    meta_raw = None
    if raw is not None:
        meta_raw = export_tracks_imagecolor_ply(
            vis_dir / "tracks_pred_raw_imagecolor.ply",
            tracks_ref0=raw,
            reference_tracks_ref0=gt_tracks_world,
            layer_name="pred_raw",
            **ply_kw,
        )

    meta_lines = export_tracks_error_lines_ply(
        vis_dir / "tracks_error_lines.ply",
        gt_tracks_world=gt_tracks_world,
        pred_tracks_ref0=pred_tracks_ref0,
        **ply_kw,
    )
    manifest = {
        "tracks_gt_imagecolor": "tracks_gt_imagecolor.ply",
        "tracks_pred_aligned_imagecolor": "tracks_pred_aligned_imagecolor.ply",
        "tracks_pred_gt_aligned": "tracks_pred_gt_aligned.ply",
        "note": "Each PLY is a single layer with vertex colors = video RGB at track UV.",
        "scale_global": scale_global,
        "gt": meta_gt,
        "pred_aligned": meta_pred,
        "pred_main": meta_main,
        "pred_raw": meta_raw,
        "error_lines": meta_lines,
        "aligned_stats": _track_error_stats(gt_tracks_world, pred_tracks_ref0, visibility_tq),
    }
    (vis_dir / "tracks_ply_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def save_tracks_package(
    out_dir: Path,
    *,
    video_rgb: np.ndarray,
    gt_tracks_world: np.ndarray,
    pred_tracks_ref0: np.ndarray,
    visibility_tq: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    metrics: Dict[str, Any],
    pred_tracks_ref0_raw: Optional[np.ndarray] = None,
    gt_tracks_uv_pixel: Optional[np.ndarray] = None,
    export_ply: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_kw: Dict[str, Any] = dict(
        video_rgb=video_rgb.astype(np.uint8),
        gt_tracks_world=gt_tracks_world.astype(np.float64),
        pred_tracks_ref0=pred_tracks_ref0.astype(np.float64),
        visibility=visibility_tq.astype(bool),
        extrinsics_w2c=extrinsics_w2c.astype(np.float64),
        intrinsics=np.asarray(intrinsics, dtype=np.float64).reshape(-1),
    )
    if pred_tracks_ref0_raw is not None:
        save_kw["pred_tracks_ref0_raw"] = np.asarray(pred_tracks_ref0_raw, dtype=np.float64)
    if gt_tracks_uv_pixel is not None:
        save_kw["gt_tracks_uv_pixel"] = np.asarray(gt_tracks_uv_pixel, dtype=np.float64)
    np.savez_compressed(out_dir / "tracks.npz", **save_kw)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if export_ply:
        export_tracks_debug_ply_package(
            out_dir,
            gt_tracks_world=gt_tracks_world,
            pred_tracks_ref0=pred_tracks_ref0,
            pred_tracks_raw=pred_tracks_ref0_raw,
            visibility_tq=visibility_tq,
            scale_global=metrics.get("scale_global"),
            video_rgb=video_rgb,
            extrinsics_w2c=extrinsics_w2c,
            intrinsics=intrinsics,
            gt_tracks_uv_pixel=gt_tracks_uv_pixel,
        )


def visualize_tracks_package(
    package_dir: Path,
    *,
    max_points: int = 300,
    trace_frames: int = 8,
    fps: float = 15.0,
    export_ply: bool = True,
) -> Dict[str, str]:
    pack = np.load(package_dir / "tracks.npz", allow_pickle=False)
    metrics_path = package_dir / "metrics.json"
    metrics: Dict[str, Any] = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    if export_ply:
        raw = pack["pred_tracks_ref0_raw"] if "pred_tracks_ref0_raw" in pack.files else None
        gt_uv = pack["gt_tracks_uv_pixel"] if "gt_tracks_uv_pixel" in pack.files else None
        export_tracks_debug_ply_package(
            package_dir,
            gt_tracks_world=pack["gt_tracks_world"],
            pred_tracks_ref0=pack["pred_tracks_ref0"],
            pred_tracks_raw=raw,
            visibility_tq=pack["visibility"],
            scale_global=metrics.get("scale_global"),
            video_rgb=pack["video_rgb"],
            extrinsics_w2c=pack["extrinsics_w2c"],
            intrinsics=pack["intrinsics"],
            gt_tracks_uv_pixel=gt_uv,
        )

    paths = render_track_comparison_videos(
        video_rgb=pack["video_rgb"],
        gt_tracks_world=pack["gt_tracks_world"],
        pred_tracks_world=pack["pred_tracks_ref0"],
        pred_tracks_raw_world=(
            pack["pred_tracks_ref0_raw"] if "pred_tracks_ref0_raw" in pack.files else None
        ),
        visibility_tq=pack["visibility"],
        extrinsics_w2c=pack["extrinsics_w2c"],
        intrinsics=pack["intrinsics"],
        output_dir=package_dir / "vis",
        max_points=max_points,
        trace_frames=trace_frames,
        fps=fps,
    )
    (package_dir / "vis" / "manifest.json").write_text(
        json.dumps(paths, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths
