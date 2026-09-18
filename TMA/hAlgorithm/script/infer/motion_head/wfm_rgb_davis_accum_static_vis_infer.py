#!/usr/bin/env python3
"""
DAVIS accumulated per-frame static + frame-0 dynamic warp3d visualization.

Per sequence:
  1. Sample RGB frames (default 50, debug_trajectory).
  2. For each frame t: pair (t, t) dense RGB grid → static point cloud,
     colored by frame-t RGB at the same query UVs.
  3. Pairs (0, t): dense RGB-grid query UVs → dynamic tracks.
  4. Rerun: at time t, static clouds from frames 0..t are accumulated; dynamic pred + trajectories.
  5. Predicted camera frustums per frame (rainbow: frame 0 red → last frame violet), accumulated over time.

Usage:
    python hAlgorithm/script/infer/motion_head/wfm_rgb_davis_accum_static_vis_infer.py \\
        --motion_config results/.../backup.py \\
        --load_from results/.../checkpoint/latest/ckpt.pth \\
        --sequence parkour
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.append(os.getcwd())

import cv2
import numpy as np
import torch
from tqdm import tqdm

from hAlgorithm.utils import config_merge_args, file2dict, parse_unknown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[3]

_spec_davis = importlib.util.spec_from_file_location(
    "_wfm_rgb_davis_dynamic_vis_infer",
    _SCRIPT_DIR / "wfm_rgb_davis_dynamic_vis_infer.py",
)
_davis = importlib.util.module_from_spec(_spec_davis)
assert _spec_davis.loader is not None
_spec_davis.loader.exec_module(_davis)

_spec_seq = importlib.util.spec_from_file_location(
    "_wfm_rgb_sequence_vis_infer",
    _SCRIPT_DIR / "wfm_rgb_sequence_vis_infer.py",
)
_seq = importlib.util.module_from_spec(_spec_seq)
assert _spec_seq.loader is not None
_spec_seq.loader.exec_module(_seq)

_spec_pair = importlib.util.spec_from_file_location(
    "_mv_query_pair2_motion_heatmap_infer",
    _SCRIPT_DIR / "mv_query_pair2_motion_heatmap_infer.py",
)
_pair = importlib.util.module_from_spec(_spec_pair)
assert _spec_pair.loader is not None
_spec_pair.loader.exec_module(_pair)

try:
    import rerun as rr
except ImportError:
    rr = None


def _setup_davis_blueprint() -> None:
    if rr is None:
        return
    try:
        import rerun.blueprint as rrb
    except ImportError:
        return
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Accum Static + Dynamic 3D", origin="dynamic_pred/"),
                rrb.Spatial2DView(name="DAVIS RGB (original)", origin="davis_orig/"),
            ),
            collapse_panels=True,
        ),
    )


def frame_rainbow_colors(num_frames: int, saturation: float = 0.85) -> np.ndarray:
    """Rainbow by frame index: frame 0 → red, last frame → violet."""
    if num_frames <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    import colorsys

    sat = float(np.clip(saturation, 0.0, 1.0))
    out = np.zeros((num_frames, 3), dtype=np.uint8)
    for i in range(num_frames):
        t_norm = i / max(num_frames - 1, 1)
        hue = 0.75 * t_norm
        r, g, b = colorsys.hsv_to_rgb(hue, sat, 1.0)
        out[i] = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
    return out


def camera_frustum_lines(
    c2w: np.ndarray,
    k: np.ndarray,
    h: int,
    w: int,
    scale: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Eight wireframe segments for a pinhole frustum in world space."""
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


def predict_cameras_per_frame(
    pipeline,
    batch: dict,
    args,
    amp_dtype: torch.dtype,
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Run ``pipeline.infer`` once and extract per-frame predicted w2c → c2w + K."""
    infer_batch = {
        "image": batch["image"],
        "meta_data": batch["meta_data"],
    }
    for key in ("scale", "extrinsics_reff", "time_idx"):
        if key in batch:
            infer_batch[key] = batch[key]
    if "extrinsics_reff" in batch:
        infer_batch["extrinsics"] = batch["extrinsics_reff"]
    if args.use_amp:
        infer_batch["use_amp"] = True
        infer_batch["amp_dtype"] = amp_dtype

    mv_outputs = pipeline.infer(**infer_batch)
    cameras: list[tuple[np.ndarray, np.ndarray] | None] = []
    for out in mv_outputs:
        w2c = getattr(out, "extrinsics_pred", None)
        k = getattr(out, "intrinsics_pred", None)
        if w2c is None or k is None:
            cameras.append(None)
            continue
        w2c = np.asarray(w2c, dtype=np.float64).reshape(4, 4)
        k = np.asarray(k, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(w2c).all() or not np.isfinite(k).all():
            cameras.append(None)
            continue
        try:
            c2w = np.linalg.inv(w2c)
        except np.linalg.LinAlgError:
            cameras.append(None)
            continue
        cameras.append((c2w, k))

    n_expected = int(batch["meta_data"]["frames"][0])
    if len(cameras) != n_expected:
        logging.warning(
            "Predicted camera count %d != frame count %d",
            len(cameras),
            n_expected,
        )
    n_valid = sum(1 for c in cameras if c is not None)
    logging.info("Predicted cameras: %d / %d frames valid", n_valid, len(cameras))
    return cameras


def log_accum_camera_frustums(
    t: int,
    cameras: list[tuple[np.ndarray, np.ndarray] | None],
    h: int,
    w: int,
    frustum_scale: float,
    line_radius: float,
) -> None:
    """Log wireframe frustums for frames 0..t with rainbow colors."""
    if rr is None or not cameras:
        return

    colors = frame_rainbow_colors(len(cameras))
    all_strips: list[np.ndarray] = []
    strip_colors: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    center_colors: list[np.ndarray] = []

    for fi in range(t + 1):
        cam = cameras[fi]
        if cam is None:
            continue
        c2w, k = cam
        center, segs = camera_frustum_lines(c2w, k, h, w, scale=frustum_scale)
        color = colors[fi]
        for seg in segs:
            all_strips.append(seg)
            strip_colors.append(color)
        centers.append(center)
        center_colors.append(color)

    if all_strips:
        rr.log(
            "dynamic_pred/accum_camera_frustums",
            rr.LineStrips3D(
                strips=all_strips,
                colors=np.stack(strip_colors, axis=0),
                radii=line_radius,
            ),
        )
    if centers:
        rr.log(
            "dynamic_pred/accum_camera_centers",
            rr.Points3D(
                np.stack(centers, axis=0),
                colors=np.stack(center_colors, axis=0),
                radii=0.012,
            ),
        )


def frame_rgb_from_tensor(ten: torch.Tensor) -> np.ndarray:
    """Model-input tensor ``[-1,1]`` → uint8 RGB, same pixels the network sees."""
    arr = ten.detach().float().cpu().permute(1, 2, 0).numpy()
    return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)


def build_static_layer_outside_mask(
    raw_static: np.ndarray,
    uv_static: np.ndarray,
    frame_rgb: np.ndarray,
    frame_mask: np.ndarray | None,
    mask_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep (t,t) warp3d outside frame-t VOS mask; sample RGB on frame-t image."""
    uv = np.asarray(uv_static, dtype=np.float64)
    if frame_mask is None:
        pts = np.asarray(raw_static, dtype=np.float64)
        return pts, _davis.sample_rgb_at_uv(frame_rgb, uv)
    h, w = frame_mask.shape[:2]
    px = np.clip(np.round(uv[:, 0] * max(w - 1, 1)).astype(int), 0, w - 1)
    py = np.clip(np.round(uv[:, 1] * max(h - 1, 1)).astype(int), 0, h - 1)
    outside = frame_mask[py, px] <= int(mask_threshold)
    if not np.any(outside):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    pts = np.asarray(raw_static, dtype=np.float64)[outside]
    cols = _davis.sample_rgb_at_uv(frame_rgb, uv[outside])
    return pts, cols


def subsample_points(
    pts: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or pts.shape[0] <= max_points:
        return pts, colors
    rng = np.random.default_rng(seed)
    sel = rng.choice(pts.shape[0], max_points, replace=False)
    return pts[sel], colors[sel]


def build_accum_static_at_t(
    static_per_frame: list[tuple[np.ndarray, np.ndarray]],
    t: int,
    static_vis_step: int,
    max_per_frame_vis_points: int,
    max_accum_vis_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate static clouds from frames 0..t with optional subsampling."""
    pts_list: list[np.ndarray] = []
    col_list: list[np.ndarray] = []

    for fi in range(t + 1):
        pts, cols = static_per_frame[fi]
        if static_vis_step > 1:
            pts = pts[::static_vis_step]
            cols = cols[::static_vis_step]
        pts, cols = subsample_points(pts, cols, max_per_frame_vis_points, seed=fi)
        if pts.shape[0] == 0:
            continue
        pts_list.append(pts)
        col_list.append(cols)

    if not pts_list:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    accum_pts = np.concatenate(pts_list, axis=0)
    accum_cols = np.concatenate(col_list, axis=0)
    accum_pts, accum_cols = subsample_points(accum_pts, accum_cols, max_accum_vis_points, seed=t)
    return accum_pts, accum_cols


def export_davis_accum_rerun(
    out_dir: str,
    seq_name: str,
    static_per_frame: list[tuple[np.ndarray, np.ndarray]],
    dynamic_tracks: list[np.ndarray],
    dynamic_colors: np.ndarray,
    frame_rgbs: list[np.ndarray],
    frame_rgbs_orig: list[np.ndarray],
    ref_mask_model: np.ndarray | None,
    mask_threshold: int,
    static_vis_step: int,
    max_per_frame_vis_points: int,
    max_accum_vis_points: int,
    max_traj_vis_points: int,
    traj_vis_downsample: int,
    cameras: list[tuple[np.ndarray, np.ndarray] | None] | None,
    image_h: int,
    image_w: int,
    frustum_scale: float,
    frustum_line_radius: float,
    show_camera_frustums: bool,
) -> str:
    if rr is None:
        raise RuntimeError("rerun-sdk is not installed.")

    save_path = os.path.join(out_dir, "rerun_vis", f"vis_accum_static_{seq_name}.rrd")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rr.init(f"DavisAccumStatic_{seq_name}", spawn=False)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    _setup_davis_blueprint()

    tracks = np.stack([np.asarray(x, dtype=np.float64) for x in dynamic_tracks], axis=0)
    num_frames, _, _ = tracks.shape
    valid_q = np.isfinite(tracks).all(axis=(0, 2))
    tracks = tracks[:, valid_q]
    if tracks.shape[1] == 0:
        raise RuntimeError("No finite dynamic tracks to visualize.")

    vis_idx = _davis._subsample_traj_indices(tracks.shape[1], max_traj_vis_points)
    tracks_vis = tracks[:, vis_idx]
    point_cols = np.asarray(dynamic_colors, dtype=np.uint8)[valid_q][vis_idx]

    ds = max(int(traj_vis_downsample), 1)
    traj_vis_idx = np.arange(0, tracks_vis.shape[1], ds, dtype=np.int64)
    tracks_traj = tracks_vis[:, traj_vis_idx]
    traj_cols = _davis.height_rainbow_colors(tracks_traj[0])

    for t in range(num_frames):
        rr.set_time_sequence("target_frame", t)

        if frame_rgbs_orig[t] is not None:
            img_orig = frame_rgbs_orig[t]
            if img_orig.dtype != np.uint8:
                img_orig = (np.clip(img_orig, 0.0, 1.0) * 255.0).astype(np.uint8)
            rr.log("davis_orig/rgb", rr.Image(img_orig))

        if frame_rgbs[t] is not None:
            img = frame_rgbs[t]
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            rr.log("camera/rgb", rr.Image(img))
            if t == 0 and ref_mask_model is not None:
                rr.log(
                    "camera/rgb_with_ref_mask",
                    rr.Image(_davis.overlay_mask_on_rgb(img, ref_mask_model, mask_threshold)),
                )

        accum_pts, accum_cols = build_accum_static_at_t(
            static_per_frame,
            t,
            static_vis_step,
            max_per_frame_vis_points,
            max_accum_vis_points,
        )
        rr.log(
            "dynamic_pred/accum_static_pointcloud",
            rr.Points3D(accum_pts, colors=accum_cols, radii=0.004),
        )

        cur_pts = tracks_vis[t]
        rr.log(
            "dynamic_pred/pred_points",
            rr.Points3D(cur_pts, colors=point_cols, radii=0.006),
        )

        if t >= 1 and tracks_traj.shape[1] > 0:
            traj_strips = [tracks_traj[: t + 1, qi] for qi in range(tracks_traj.shape[1])]
            rr.log(
                "dynamic_pred/trajectories",
                rr.LineStrips3D(
                    strips=traj_strips,
                    colors=traj_cols,
                    radii=0.0005,
                ),
            )

        if show_camera_frustums and cameras:
            log_accum_camera_frustums(
                t,
                cameras,
                image_h,
                image_w,
                frustum_scale,
                frustum_line_radius,
            )

    rr.save(save_path)
    n_cam = sum(1 for c in (cameras or []) if c is not None)
    logging.info(
        "Saved Rerun: %s (%d frames, final accum static pts=%d, %d pred points, "
        "%d trajectory lines, %d camera frustums)",
        save_path,
        num_frames,
        build_accum_static_at_t(
            static_per_frame, num_frames - 1, static_vis_step,
            max_per_frame_vis_points, max_accum_vis_points,
        )[0].shape[0],
        tracks_vis.shape[1],
        tracks_traj.shape[1],
        n_cam,
    )
    return save_path


def parse_args():
    p = argparse.ArgumentParser(description="DAVIS per-frame (n,n) accum static + (0,t) dynamic Rerun vis")
    p.add_argument("--motion_config", type=str, required=True)
    p.add_argument("--load_from", type=str, required=True)
    p.add_argument("--davis_root", type=str, default=_davis.DEFAULT_DAVIS_ROOT)
    p.add_argument("--resolution", type=str, default="480p", choices=["480p", "1080p"])
    p.add_argument(
        "--use_davis_masks",
        action="store_true",
        help="Opt in to legacy DAVIS annotation-mask queries and filtering.",
    )
    p.add_argument("--sequence", type=str, default=None)
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--output_dir", type=str, default="./results/wfm_rgb_davis_accum_static_vis")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument(
        "--frame_sampling",
        type=str,
        default="debug_trajectory",
        choices=["debug_trajectory", "sequential"],
    )
    p.add_argument("--static_downsample", type=int, default=4)
    p.add_argument("--dynamic_downsample", type=int, default=1)
    p.add_argument("--mask_threshold", type=int, default=127)
    p.add_argument("--save_mask_debug", action="store_true", default=True)
    p.add_argument("--max_dynamic_queries", type=int, default=8192, help="0 = no cap.")
    p.add_argument("--static_vis_step", type=int, default=1)
    p.add_argument(
        "--max_per_frame_vis_points",
        type=int,
        default=40000,
        help="Random cap per frame before accumulation (0 = no cap).",
    )
    p.add_argument(
        "--max_accum_vis_points",
        type=int,
        default=250000,
        help="Random cap on total accumulated static points per timestep (0 = no cap).",
    )
    p.add_argument("--max_traj_vis_points", type=int, default=512)
    p.add_argument("--traj_vis_downsample", type=int, default=10)
    p.add_argument("--frustum_scale", type=float, default=0.05)
    p.add_argument("--frustum_line_radius", type=float, default=0.003)
    p.add_argument("--no_camera_frustums", action="store_true")
    p.add_argument("--export_rrd_only", action="store_true")
    p.add_argument("--process_res", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--dense_query_batch_size", type=int, default=65536)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--no_time", action="store_true")
    args, unknown = p.parse_known_args()
    unknown = parse_unknown(unknown)

    if args.no_amp:
        args.use_amp = False
    if args.static_downsample < 1:
        p.error("--static_downsample must be >= 1.")
    if args.dynamic_downsample < 1:
        p.error("--dynamic_downsample must be >= 1.")

    if args.no_time:
        tag = Path(args.motion_config).stem
        args.output_dir = os.path.join(args.output_dir, tag)
    else:
        tag = Path(args.motion_config).stem + datetime.datetime.now().strftime("_%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(args.output_dir, tag)
    os.makedirs(args.output_dir, exist_ok=True)
    return args, unknown


def run_sequence(
    pipeline,
    args,
    seq_name: str,
    device: torch.device,
    process_res: int,
    patch_size: int,
) -> None:
    rgb_dir, mask_dir = _davis.davis_paths(
        args.davis_root,
        args.resolution,
        seq_name,
        require_masks=args.use_davis_masks,
    )
    rgb_paths = _davis.list_sorted_frames(rgb_dir)
    rgb_paths = _seq.sample_frame_paths(rgb_paths, args.max_frames, args.frame_sampling)
    n_frames = len(rgb_paths)
    logging.info("Sequence %r: %d sampled frames from %s", seq_name, n_frames, rgb_dir)

    frame_tensors = []
    frame_rgbs_model: list[np.ndarray] = []
    frame_rgbs_orig = []
    frame_masks_model: list[np.ndarray] = []
    for p in rgb_paths:
        ten, nh, nw = _seq.load_frame_tensor(p, process_res, patch_size)
        frame_tensors.append(ten)
        frame_rgbs_model.append(frame_rgb_from_tensor(ten))
        rgb_orig = _davis.load_frame_rgb_orig(p)
        frame_rgbs_orig.append(rgb_orig)
        if args.use_davis_masks:
            mask_path = _davis.mask_path_for_frame(mask_dir, p)
            frame_masks_model.append(
                _davis.resize_mask_nearest(_davis.load_mask_gray(mask_path), nh, nw),
            )
        else:
            frame_masks_model.append(None)

    h, w = frame_tensors[0].shape[1:]
    ref_rgb_path = rgb_paths[0]
    uv_static, _, _ = _pair.sample_dense_grid(h, w, args.static_downsample)
    ref_mask_path = None
    ref_mask_model = None
    ref_fg_orig = None
    if args.use_davis_masks:
        ref_mask_path = _davis.mask_path_for_frame(mask_dir, ref_rgb_path)
        uv_dynamic, ref_fg_orig, _ = _davis.mask_to_query_uv_from_paths(
            ref_rgb_path,
            ref_mask_path,
            downsample=args.dynamic_downsample,
            max_queries=args.max_dynamic_queries,
            mask_threshold=args.mask_threshold,
        )
        ref_mask_model = frame_masks_model[0]
    else:
        uv_dynamic = _davis.dense_query_uv(
            h, w, args.dynamic_downsample, args.max_dynamic_queries
        )

    if args.use_davis_masks and args.save_mask_debug:
        _davis.save_mask_debug_overlays(
            os.path.join(args.output_dir, seq_name),
            seq_name,
            ref_rgb_path,
            ref_mask_path,
            frame_rgbs_model[0],
            args.mask_threshold,
        )

    logging.info(
        "Queries: static_grid=%d (downsample=%d), dynamic=%d (source=%s)",
        len(uv_static),
        args.static_downsample,
        len(uv_dynamic),
        "DAVIS mask" if args.use_davis_masks else "RGB grid",
    )

    frame_data_list = [
        dict(image=ten, image_rgb=rgb, frame_id=Path(p).stem, rgb_path=p)
        for ten, rgb, p in zip(frame_tensors, frame_rgbs_model, rgb_paths)
    ]
    batch = _davis.build_rgb_only_window_batch(frame_tensors, seq_name, device)

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_ctx = (
        torch.autocast("cuda", enabled=True, dtype=amp_dtype)
        if args.use_amp
        else nullcontext()
    )
    sdk = pipeline.model
    chunk_size = getattr(sdk, "chunk_size", None)

    cameras: list[tuple[np.ndarray, np.ndarray] | None] | None = None
    if not args.no_camera_frustums:
        with torch.no_grad():
            cameras = predict_cameras_per_frame(pipeline, batch, args, amp_dtype)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    static_per_frame: list[tuple[np.ndarray, np.ndarray]] = []
    static_raw_per_frame: list[np.ndarray] = []
    dynamic_tracks: list[np.ndarray] = []
    dynamic_colors = _davis.sample_rgb_at_uv(frame_rgbs_model[0], uv_dynamic)

    with torch.no_grad():
        patch_tokens, time_token = _pair.encode_pair_features(sdk, batch, amp_ctx)

        for t in tqdm(range(n_frames), desc=f"{seq_name} static (t,t)"):
            raw_static = _davis.decode_dense_pair_warp3d(
                sdk, batch, patch_tokens, time_token,
                (t, t), uv_static, device,
                args.dense_query_batch_size, chunk_size,
            )
            static_raw_per_frame.append(raw_static)
            static_pts, static_cols = build_static_layer_outside_mask(
                raw_static,
                uv_static,
                frame_rgbs_model[t],
                frame_masks_model[t],
                args.mask_threshold,
            )
            static_per_frame.append((static_pts, static_cols))

        for t in tqdm(range(n_frames), desc=f"{seq_name} dynamic (0,t)"):
            pair = (0, 0) if t == 0 else (0, t)
            dyn_pts = _davis.decode_dense_pair_warp3d(
                sdk, batch, patch_tokens, time_token,
                pair, uv_dynamic, device,
                args.dense_query_batch_size, chunk_size,
            )
            if dyn_pts.shape[0] != len(uv_dynamic):
                raise RuntimeError(
                    f"t={t}: dynamic warp3d rows {dyn_pts.shape[0]} != uv_dynamic {len(uv_dynamic)}",
                )
            dynamic_tracks.append(dyn_pts)

    out_root = os.path.join(args.output_dir, seq_name)
    os.makedirs(out_root, exist_ok=True)

    np.savez_compressed(
        os.path.join(out_root, "warp3d_static_per_frame.npz"),
        uv=uv_static,
        warp3d_per_frame=np.stack(static_raw_per_frame, axis=0),
        downsample=args.static_downsample,
        frame_ids=[Path(p).stem for p in rgb_paths],
    )
    np.savez_compressed(
        os.path.join(out_root, "warp3d_dynamic.npz"),
        uv=uv_dynamic,
        tracks=np.stack(dynamic_tracks, axis=0),
        frame_ids=[Path(p).stem for p in rgb_paths],
    )
    if cameras is not None:
        c2w_stack = np.full((n_frames, 4, 4), np.nan, dtype=np.float64)
        k_stack = np.full((n_frames, 3, 3), np.nan, dtype=np.float64)
        for fi, cam in enumerate(cameras):
            if cam is None:
                continue
            c2w_stack[fi] = cam[0]
            k_stack[fi] = cam[1]
        np.savez_compressed(
            os.path.join(out_root, "pred_cameras.npz"),
            c2w=c2w_stack,
            intrinsics=k_stack,
            frame_ids=[Path(p).stem for p in rgb_paths],
            image_height=h,
            image_width=w,
        )

    rrd_path = export_davis_accum_rerun(
        out_root,
        seq_name,
        static_per_frame,
        dynamic_tracks,
        dynamic_colors,
        frame_rgbs_model,
        frame_rgbs_orig,
        ref_mask_model,
        args.mask_threshold,
        args.static_vis_step,
        args.max_per_frame_vis_points,
        args.max_accum_vis_points,
        args.max_traj_vis_points,
        args.traj_vis_downsample,
        cameras,
        h,
        w,
        args.frustum_scale,
        args.frustum_line_radius,
        not args.no_camera_frustums,
    )

    summary = dict(
        sequence=seq_name,
        num_frames=n_frames,
        rgb_dir=rgb_dir,
        mask_dir=mask_dir if args.use_davis_masks else None,
        ref_rgb=ref_rgb_path,
        ref_mask=ref_mask_path,
        static_mode=(
            "per_frame_nn_outside_frame_t_mask"
            if args.use_davis_masks
            else "per_frame_rgb_grid"
        ),
        dynamic_mode=(
            "ref0_mask_pair_0t"
            if args.use_davis_masks
            else "ref0_rgb_grid_pair_0t"
        ),
        orig_fg_pixels=int(ref_fg_orig.sum()) if ref_fg_orig is not None else None,
        static_grid_queries=int(len(uv_static)),
        dynamic_queries=int(uv_dynamic.shape[0]),
        static_downsample=args.static_downsample,
        dynamic_downsample=args.dynamic_downsample,
        per_frame_static_counts=[int(x[0].shape[0]) for x in static_per_frame],
        pred_cameras=int(sum(1 for c in (cameras or []) if c is not None)),
        rrd=rrd_path,
    )
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info("Done %r → %s", seq_name, out_root)


def load_cameras_from_npz(
    out_root: str,
    n_frames: int,
) -> list[tuple[np.ndarray, np.ndarray] | None] | None:
    cam_npz = os.path.join(out_root, "pred_cameras.npz")
    if not os.path.isfile(cam_npz):
        return None
    data = np.load(cam_npz)
    c2w_stack = data["c2w"]
    k_stack = data["intrinsics"]
    cameras: list[tuple[np.ndarray, np.ndarray] | None] = []
    for fi in range(min(n_frames, c2w_stack.shape[0])):
        c2w = c2w_stack[fi]
        k = k_stack[fi]
        if not np.isfinite(c2w).all() or not np.isfinite(k).all():
            cameras.append(None)
        else:
            cameras.append((c2w, k))
    while len(cameras) < n_frames:
        cameras.append(None)
    return cameras


def export_rrd_from_saved(
    args,
    seq_name: str,
    process_res: int,
    patch_size: int,
) -> None:
    out_root = os.path.join(args.output_dir, seq_name)
    static_npz = os.path.join(out_root, "warp3d_static_per_frame.npz")
    dyn_npz = os.path.join(out_root, "warp3d_dynamic.npz")
    if not os.path.isfile(static_npz) or not os.path.isfile(dyn_npz):
        raise FileNotFoundError(f"Missing npz under {out_root}; run inference first.")

    static = np.load(static_npz)
    uv_static = static["uv"]
    static_pts_stack = static["warp3d_per_frame"]
    frame_ids = list(static["frame_ids"])

    dyn = np.load(dyn_npz)
    tracks = dyn["tracks"]
    uv_dynamic = dyn["uv"]

    rgb_dir, mask_dir = _davis.davis_paths(
        args.davis_root,
        args.resolution,
        seq_name,
        require_masks=args.use_davis_masks,
    )
    rgb_paths = _davis.list_sorted_frames(rgb_dir)
    rgb_paths = _seq.sample_frame_paths(rgb_paths, args.max_frames, args.frame_sampling)
    id_to_path = {Path(p).stem: p for p in rgb_paths}
    ordered_paths = [id_to_path[str(fid)] for fid in frame_ids]

    frame_rgbs_model: list[np.ndarray] = []
    frame_rgbs_orig = []
    frame_masks_model = []
    for p in ordered_paths:
        ten, nh, nw = _seq.load_frame_tensor(p, process_res, patch_size)
        frame_rgbs_model.append(frame_rgb_from_tensor(ten))
        rgb_orig = _davis.load_frame_rgb_orig(p)
        frame_rgbs_orig.append(rgb_orig)
        if args.use_davis_masks:
            mask_path = _davis.mask_path_for_frame(mask_dir, p)
            frame_masks_model.append(
                _davis.resize_mask_nearest(_davis.load_mask_gray(mask_path), nh, nw),
            )
        else:
            frame_masks_model.append(None)

    ref_mask_model = frame_masks_model[0]
    static_per_frame: list[tuple[np.ndarray, np.ndarray]] = []
    for t in range(static_pts_stack.shape[0]):
        pts, cols = build_static_layer_outside_mask(
            static_pts_stack[t],
            uv_static,
            frame_rgbs_model[t],
            frame_masks_model[t],
            args.mask_threshold,
        )
        static_per_frame.append((pts, cols))

    dynamic_colors = _davis.sample_rgb_at_uv(frame_rgbs_model[0], uv_dynamic)
    h, w = frame_rgbs_model[0].shape[:2]
    cameras = load_cameras_from_npz(out_root, static_pts_stack.shape[0])
    if cameras is None and not args.no_camera_frustums:
        logging.warning("Missing pred_cameras.npz — RRD will omit camera frustums.")

    export_davis_accum_rerun(
        out_root,
        seq_name,
        static_per_frame,
        list(tracks),
        dynamic_colors,
        frame_rgbs_model,
        frame_rgbs_orig,
        ref_mask_model,
        args.mask_threshold,
        args.static_vis_step,
        args.max_per_frame_vis_points,
        args.max_accum_vis_points,
        args.max_traj_vis_points,
        args.traj_vis_downsample,
        cameras,
        h,
        w,
        args.frustum_scale,
        args.frustum_line_radius,
        not args.no_camera_frustums,
    )


def main():
    args, unknown = parse_args()
    cfg = file2dict(args.motion_config)
    config_merge_args(cfg, unknown)

    process_res = args.process_res or int(cfg.get("max_size", 504))
    patch_size = args.patch_size or int(cfg.get("patch_size", 14))

    seqs: list[str] = []
    if args.sequences:
        seqs.extend(args.sequences)
    if args.sequence:
        seqs.append(args.sequence)
    if not seqs:
        raise SystemExit("Provide --sequence or --sequences.")

    pipeline = None if args.export_rrd_only else _davis.load_pipeline(args, cfg)
    device = torch.device("cuda")

    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    failures = []
    for seq in seqs:
        try:
            if args.export_rrd_only:
                export_rrd_from_saved(args, seq, process_res, patch_size)
            else:
                run_sequence(pipeline, args, seq, device, process_res, patch_size)
        except Exception as exc:
            logging.exception("Failed %s: %s", seq, exc)
            failures.append((seq, str(exc)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if failures:
        for name, msg in failures:
            logging.error("  %s: %s", name, msg)
        sys.exit(1)
    logging.info("All done → %s", args.output_dir)


if __name__ == "__main__":
    main()
