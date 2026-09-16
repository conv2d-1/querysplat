#!/usr/bin/env python3
"""
DAVIS dynamic 3D visualization via WFM pair warp3d.

For each sequence:
  1. Sample up to ``max_frames`` RGB frames (default 50, debug_trajectory).
  2. Pair (0, 0): dense grid queries → static reference point cloud (warp3d).
  3. Pairs (0, t): frame-0 **DAVIS Annotations** (VOS object mask) query UVs → dynamic tracks.
  4. Export a single Rerun ``.rrd`` (static background + mask dynamic points).

Note: DAVIS ``Annotations/480p`` are **video object segmentation** masks (foreground
object silhouette), not optical-flow / model-style ``motion_mask`` logits. Query UVs
are computed in **original image normalized coordinates** so they stay aligned with
the resized RGB fed to the model.

Usage:
    python hAlgorithm/script/infer/motion_head/wfm_rgb_davis_dynamic_vis_infer.py \\
        --motion_config results/.../backup.py \\
        --load_from results/.../checkpoint/latest/ckpt.pth \\
        --sequence bear
"""

from __future__ import annotations

import argparse
import colorsys
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

from hAlgorithm.modules.models2.query_bank.query import BaseQuery
from hAlgorithm.modules.pipelines2.utils.motion_utils import prepare_decoupled_warp3d_delta_inplace
from hAlgorithm.utils import config_merge_args, file2dict, instantiate_from_config, parse_unknown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[3]

DEFAULT_DAVIS_ROOT = "/mnt/netdata/Team/AI/datasets/VLM/DAVIS"
DEFAULT_RUN_DIR = (
    _REPO_ROOT
    / "results/baseline_v4/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_20260609-115200"
)

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
    """3D pred view + original DAVIS RGB panel, synced on ``target_frame``."""
    if rr is None:
        return
    try:
        import rerun.blueprint as rrb
    except ImportError:
        return
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Dynamic Pred 3D", origin="dynamic_pred/"),
                rrb.Spatial2DView(name="DAVIS RGB (original)", origin="davis_orig/"),
            ),
            collapse_panels=True,
        ),
    )


def load_frame_rgb_orig(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise RuntimeError(f"Failed to read RGB: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_args():
    p = argparse.ArgumentParser(description="DAVIS warp3d dynamic 3D Rerun vis")
    p.add_argument("--motion_config", type=str, required=True)
    p.add_argument("--load_from", type=str, required=True)
    p.add_argument(
        "--davis_root",
        type=str,
        default=DEFAULT_DAVIS_ROOT,
        help="DAVIS dataset root (JPEGImages + Annotations).",
    )
    p.add_argument(
        "--resolution",
        type=str,
        default="480p",
        choices=["480p", "1080p"],
        help="DAVIS resolution subfolder.",
    )
    p.add_argument("--sequence", type=str, default=None)
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--output_dir", type=str, default="./results/wfm_rgb_davis_dynamic_vis")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument(
        "--frame_sampling",
        type=str,
        default="debug_trajectory",
        choices=["debug_trajectory", "sequential"],
    )
    p.add_argument(
        "--static_downsample",
        type=int,
        default=4,
        help="Grid stride for (0,0) full-scene warp3d (1 = densest).",
    )
    p.add_argument(
        "--dynamic_downsample",
        type=int,
        default=1,
        help="Keep every k-th foreground pixel (raster order) on frame-0 VOS mask.",
    )
    p.add_argument(
        "--mask_threshold",
        type=int,
        default=127,
        help="Foreground if annotation pixel > threshold (DAVIS uses 255).",
    )
    p.add_argument(
        "--save_mask_debug",
        action="store_true",
        default=True,
        help="Save frame-0 mask overlay PNGs under mask_debug/.",
    )
    p.add_argument(
        "--max_dynamic_queries",
        type=int,
        default=8192,
        help="Cap mask query count (0 = no cap).",
    )
    p.add_argument(
        "--static_vis_step",
        type=int,
        default=1,
        help="Subsample static cloud for Rerun display.",
    )
    p.add_argument(
        "--max_static_vis_points",
        type=int,
        default=120000,
        help="Random cap on static points logged to Rerun (0 = no cap).",
    )
    p.add_argument(
        "--max_traj_vis_points",
        type=int,
        default=512,
        help="Max mask query trajectories drawn in Rerun (0 = all).",
    )
    p.add_argument(
        "--traj_vis_downsample",
        type=int,
        default=10,
        help="Extra stride for trajectory LineStrips only (pred_points unchanged).",
    )
    p.add_argument(
        "--min_arrow_length",
        type=float,
        default=0.01,
        help="Minimum visual length for step arrows in ref0 3D units.",
    )
    p.add_argument(
        "--export_rrd_only",
        action="store_true",
        help="Skip inference; rebuild RRD from saved npz under output_dir/sequence/.",
    )
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


def resolve_load_from(load_from: str, motion_config: str) -> str:
    if load_from == "latest":
        return os.path.join(os.path.dirname(motion_config), "checkpoint/latest/ckpt.pth")
    if load_from == "best":
        return os.path.join(os.path.dirname(motion_config), "checkpoint/best/ckpt.pth")
    return os.path.expanduser(load_from)


def load_pipeline(args, cfg):
    pipeline = instantiate_from_config(cfg["model"]).cuda()
    pipeline.eval()
    pipeline.device = torch.device("cuda")
    if args.use_amp:
        pipeline.dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    else:
        pipeline.dtype = torch.float32

    if not hasattr(pipeline.model, "pair_forward"):
        raise TypeError("Model must expose pair_forward (WFMQueryPipeline).")

    ckpt = resolve_load_from(args.load_from, args.motion_config)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    pipeline.load_checkpoint(ckpt_path=ckpt)
    logging.info("Loaded checkpoint: %s", ckpt)
    pipeline.testing_sub_pixel_scale = int(args.static_downsample)
    return pipeline


def davis_paths(davis_root: str, resolution: str, seq: str) -> tuple[str, str]:
    root = os.path.abspath(davis_root)
    rgb_dir = os.path.join(root, "JPEGImages", resolution, seq)
    mask_dir = os.path.join(root, "Annotations", resolution, seq)
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"RGB dir not found: {rgb_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")
    return rgb_dir, mask_dir


def list_sorted_frames(rgb_dir: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [
        os.path.join(rgb_dir, f)
        for f in sorted(os.listdir(rgb_dir))
        if os.path.splitext(f.lower())[1] in exts
    ]
    if len(paths) < 2:
        raise RuntimeError(f"Need >= 2 frames under {rgb_dir}, got {len(paths)}")
    return paths


def mask_path_for_frame(mask_dir: str, rgb_path: str) -> str:
    stem = Path(rgb_path).stem
    png = os.path.join(mask_dir, f"{stem}.png")
    if os.path.isfile(png):
        return png
    raise FileNotFoundError(f"Mask not found for {rgb_path}: {png}")


def load_mask_gray(mask_path: str) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    return mask


def mask_to_query_uv_from_paths(
    rgb_path: str,
    mask_path: str,
    *,
    downsample: int,
    max_queries: int,
    mask_threshold: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Foreground query UVs in original-image normalized coords (u, v) ∈ [0, 1]."""
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise RuntimeError(f"Failed to read RGB: {rgb_path}")
    oh, ow = rgb.shape[:2]
    mask = load_mask_gray(mask_path)
    if mask.shape[:2] != (oh, ow):
        raise ValueError(
            f"Mask/RGB size mismatch for {rgb_path}: mask={mask.shape[:2]} rgb={(oh, ow)}",
        )

    fg = mask > int(mask_threshold)
    ys, xs = np.where(fg)
    if ys.size == 0:
        raise RuntimeError(f"Empty VOS mask on frame 0: {mask_path}")

    order = np.lexsort((xs, ys))
    ys, xs = ys[order], xs[order]
    if downsample > 1:
        ys, xs = ys[::downsample], xs[::downsample]

    u = xs.astype(np.float32) / max(ow - 1, 1)
    v = ys.astype(np.float32) / max(oh - 1, 1)
    uv = np.stack([u, v], axis=1)
    if max_queries > 0 and uv.shape[0] > max_queries:
        step = max(int(np.ceil(uv.shape[0] / max_queries)), 1)
        uv = uv[::step][:max_queries]

    return uv.astype(np.float32), fg, (oh, ow)


def resize_mask_nearest(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if mask.shape[:2] == (target_h, target_w):
        return mask
    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def save_mask_debug_overlays(
    out_dir: str,
    seq_name: str,
    rgb_path: str,
    mask_path: str,
    model_rgb: np.ndarray,
    mask_threshold: int,
) -> None:
    dbg_dir = os.path.join(out_dir, "mask_debug")
    os.makedirs(dbg_dir, exist_ok=True)

    rgb_bgr = cv2.imread(rgb_path)
    mask = load_mask_gray(mask_path)
    oh, ow = rgb_bgr.shape[:2]
    if mask.shape[:2] != (oh, ow):
        raise ValueError(f"Mask/RGB size mismatch: {mask_path}")

    fg = mask > int(mask_threshold)
    overlay_orig = rgb_bgr.copy()
    overlay_orig[fg] = (
        0.45 * overlay_orig[fg].astype(np.float32) + 0.55 * np.array([0, 0, 255], np.float32)
    ).astype(np.uint8)
    cv2.imwrite(os.path.join(dbg_dir, f"{seq_name}_mask_overlay_orig.jpg"), overlay_orig)

    nh, nw = model_rgb.shape[:2]
    mask_model = resize_mask_nearest(mask, nh, nw)
    model_bgr = cv2.cvtColor(model_rgb, cv2.COLOR_RGB2BGR)
    overlay_model = model_bgr.copy()
    fg_model = mask_model > int(mask_threshold)
    overlay_model[fg_model] = (
        0.45 * overlay_model[fg_model].astype(np.float32) + 0.55 * np.array([0, 0, 255], np.float32)
    ).astype(np.uint8)
    cv2.imwrite(os.path.join(dbg_dir, f"{seq_name}_mask_overlay_modelres.jpg"), overlay_model)
    logging.info("Saved mask debug overlays → %s", dbg_dir)


def sample_rgb_at_uv(rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    px = np.clip(np.round(uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
    py = np.clip(np.round(uv[:, 1] * (h - 1)).astype(int), 0, h - 1)
    cols = rgb[py, px].astype(np.uint8)
    return cols


def decode_dense_pair_warp3d(
    sdk,
    batch,
    patch_tokens,
    time_token,
    pair,
    uv_np,
    device,
    query_batch_size,
    chunk_size,
) -> np.ndarray:
    pair_idx = [pair]
    segments = []
    q_step = min(query_batch_size, chunk_size or query_batch_size)

    for q0 in range(0, len(uv_np), q_step):
        q1 = min(q0 + q_step, len(uv_np))
        uv_t = torch.from_numpy(uv_np[q0:q1]).to(device)
        query = BaseQuery(uv=uv_t)
        pair_out = sdk.pair_forward(
            rgb=batch["image"],
            pair_idx=pair_idx,
            patch_tokens=patch_tokens,
            time_token=time_token,
            query=query,
            query_rgb=None,
            meta_data=batch["meta_data"],
        )
        prepare_decoupled_warp3d_delta_inplace(pair_out)
        warp3d = pair_out.get("warp3d")
        if warp3d is None:
            delta = pair_out.get("warp3d_delta")
            if delta is None:
                raise RuntimeError(f"pair {pair}: no warp3d or warp3d_delta in pair_forward output.")
            warp3d = delta
        arr = warp3d.float().reshape(-1, 3).cpu().numpy()
        segments.append(arr.astype(np.float64))

    out = np.concatenate(segments, axis=0)
    if out.shape[0] != len(uv_np):
        raise RuntimeError(
            f"pair {pair}: warp3d rows {out.shape[0]} != query count {len(uv_np)}",
        )
    return out


def overlay_mask_on_rgb(
    rgb: np.ndarray,
    mask_gray: np.ndarray,
    mask_threshold: int,
    alpha: float = 0.5,
) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    fg = mask_gray > int(mask_threshold)
    if not fg.any():
        return rgb.astype(np.uint8)
    color = np.array([255, 50, 50], dtype=np.float32)
    out[fg] = out[fg] * (1.0 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def height_rainbow_colors(
    points_xyz: np.ndarray,
    *,
    height_axis: int = 1,
    invert: bool = True,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
    saturation: float = 0.55,
) -> np.ndarray:
    """Rainbow by 3D height: high → red, low → violet (RDF: axis 1 is Down, use -Y)."""
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)

    h_val = -pts[:, height_axis] if invert else pts[:, height_axis]
    finite = np.isfinite(h_val)
    if not np.any(finite):
        return np.full((n, 3), 170, dtype=np.uint8)

    h_finite = h_val[finite]
    lo, hi = np.percentile(h_finite, clip_percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        t = np.full(n, 0.5, dtype=np.float64)
    else:
        t = np.clip((h_val - lo) / (hi - lo), 0.0, 1.0)

    sat = float(np.clip(saturation, 0.0, 1.0))
    # High (t→1) → red (hue 0); low (t→0) → violet (hue 0.75).
    hue = 0.75 * (1.0 - t)
    rgb = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        if not finite[i]:
            rgb[i] = np.array([170, 170, 170], dtype=np.uint8)
            continue
        r, g, b = colorsys.hsv_to_rgb(float(hue[i]), sat, 1.0)
        rgb[i] = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
    return rgb


def _subsample_traj_indices(num_points: int, max_points: int, seed: int = 42) -> np.ndarray:
    if max_points <= 0 or num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_points, max_points, replace=False).astype(np.int64))


def filter_static_outside_mask(
    uv: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    mask_gray: np.ndarray | None,
    mask_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep static grid points whose UV lies outside frame-0 VOS foreground."""
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    if mask_gray is None:
        return points, colors

    uv = np.asarray(uv, dtype=np.float64)
    h, w = mask_gray.shape[:2]
    px = np.clip(np.round(uv[:, 0] * max(w - 1, 1)).astype(int), 0, w - 1)
    py = np.clip(np.round(uv[:, 1] * max(h - 1, 1)).astype(int), 0, h - 1)
    outside = mask_gray[py, px] <= int(mask_threshold)
    if not np.any(outside):
        logging.warning("No static points outside ref mask; static cloud will be empty.")
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    return points[outside], colors[outside]


def export_davis_rerun(
    out_dir: str,
    seq_name: str,
    uv_static: np.ndarray,
    static_pts: np.ndarray,
    static_colors: np.ndarray,
    dynamic_tracks: list[np.ndarray],
    dynamic_colors: np.ndarray,
    frame_rgbs: list[np.ndarray],
    frame_rgbs_orig: list[np.ndarray],
    ref_mask_model: np.ndarray | None,
    mask_threshold: int,
    static_vis_step: int,
    max_static_vis_points: int,
    max_traj_vis_points: int,
    traj_vis_downsample: int,
) -> str:
    if rr is None:
        raise RuntimeError("rerun-sdk is not installed.")

    save_path = os.path.join(out_dir, "rerun_vis", f"vis_dynamic_{seq_name}.rrd")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rr.init(f"DavisDynamic_{seq_name}", spawn=False)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    _setup_davis_blueprint()

    static_pts, static_colors = filter_static_outside_mask(
        uv_static, static_pts, static_colors, ref_mask_model, mask_threshold,
    )
    if static_vis_step > 1:
        static_pts = static_pts[::static_vis_step]
        static_colors = static_colors[::static_vis_step]
    if max_static_vis_points > 0 and static_pts.shape[0] > max_static_vis_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(static_pts.shape[0], max_static_vis_points, replace=False)
        static_pts = static_pts[sel]
        static_colors = static_colors[sel]

    tracks = np.stack([np.asarray(x, dtype=np.float64) for x in dynamic_tracks], axis=0)
    num_frames, num_queries, _ = tracks.shape
    valid_q = np.isfinite(tracks).all(axis=(0, 2))
    tracks = tracks[:, valid_q]
    if tracks.shape[1] == 0:
        raise RuntimeError("No finite dynamic tracks to visualize.")

    vis_idx = _subsample_traj_indices(tracks.shape[1], max_traj_vis_points)
    tracks_vis = tracks[:, vis_idx]
    point_cols = np.asarray(dynamic_colors, dtype=np.uint8)[valid_q][vis_idx]

    ds = max(int(traj_vis_downsample), 1)
    traj_vis_idx = np.arange(0, tracks_vis.shape[1], ds, dtype=np.int64)
    tracks_traj = tracks_vis[:, traj_vis_idx]
    traj_cols = height_rainbow_colors(tracks_traj[0])

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
                    rr.Image(overlay_mask_on_rgb(img, ref_mask_model, mask_threshold)),
                )

        rr.log(
            "dynamic_pred/static_ref0_pointcloud",
            rr.Points3D(static_pts, colors=static_colors, radii=0.005),
        )

        cur_pts = tracks_vis[t]
        rr.log(
            "dynamic_pred/pred_points",
            rr.Points3D(cur_pts, colors=point_cols, radii=0.005),
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

    rr.save(save_path)
    logging.info(
        "Saved Rerun: %s (%d frames, %d static-outside-mask pts, %d pred points, "
        "%d trajectory lines, downsample=%dx)",
        save_path,
        num_frames,
        static_pts.shape[0],
        tracks_vis.shape[1],
        tracks_traj.shape[1],
        ds,
    )
    return save_path


def export_rrd_from_saved(
    args,
    seq_name: str,
    process_res: int,
    patch_size: int,
) -> None:
    out_root = os.path.join(args.output_dir, seq_name)
    dyn_npz = os.path.join(out_root, "warp3d_dynamic.npz")
    static_npz = os.path.join(out_root, "warp3d_static.npz")
    if not os.path.isfile(dyn_npz):
        raise FileNotFoundError(f"Missing {dyn_npz}; run inference first.")

    dyn = np.load(dyn_npz)
    tracks = dyn["tracks"]
    uv_dynamic = dyn["uv"]
    frame_ids = list(dyn["frame_ids"])

    rgb_dir, mask_dir = davis_paths(args.davis_root, args.resolution, seq_name)
    rgb_paths = list_sorted_frames(rgb_dir)
    rgb_paths = _seq.sample_frame_paths(rgb_paths, args.max_frames, args.frame_sampling)
    id_to_path = {Path(p).stem: p for p in rgb_paths}
    ordered_paths = [id_to_path[str(fid)] for fid in frame_ids]

    frame_rgbs = []
    frame_rgbs_orig = []
    for p in ordered_paths:
        _, nh, nw = _seq.load_frame_tensor(p, process_res, patch_size)
        rgb_orig = load_frame_rgb_orig(p)
        frame_rgbs_orig.append(rgb_orig)
        rgb = rgb_orig
        if rgb.shape[:2] != (nh, nw):
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        frame_rgbs.append(rgb)

    ref_mask_path = mask_path_for_frame(mask_dir, ordered_paths[0])
    ref_mask_model = resize_mask_nearest(
        load_mask_gray(ref_mask_path),
        frame_rgbs[0].shape[0],
        frame_rgbs[0].shape[1],
    )
    dynamic_colors = sample_rgb_at_uv(frame_rgbs[0], uv_dynamic)

    static = np.load(static_npz)
    static_warp3d = static["warp3d"]
    uv_static = static["uv"]
    static_colors = sample_rgb_at_uv(frame_rgbs[0], uv_static)

    export_davis_rerun(
        out_root,
        seq_name,
        uv_static,
        static_warp3d,
        static_colors,
        list(tracks),
        dynamic_colors,
        frame_rgbs,
        frame_rgbs_orig,
        ref_mask_model,
        args.mask_threshold,
        args.static_vis_step,
        args.max_static_vis_points,
        args.max_traj_vis_points,
        args.traj_vis_downsample,
    )


def run_sequence(
    pipeline,
    args,
    seq_name: str,
    device: torch.device,
    process_res: int,
    patch_size: int,
) -> None:
    rgb_dir, mask_dir = davis_paths(args.davis_root, args.resolution, seq_name)
    rgb_paths = list_sorted_frames(rgb_dir)
    rgb_paths = _seq.sample_frame_paths(rgb_paths, args.max_frames, args.frame_sampling)
    n_frames = len(rgb_paths)
    logging.info("Sequence %r: %d sampled frames from %s", seq_name, n_frames, rgb_dir)

    frame_tensors = []
    frame_rgbs = []
    frame_rgbs_orig = []
    for p in rgb_paths:
        ten, nh, nw = _seq.load_frame_tensor(p, process_res, patch_size)
        frame_tensors.append(ten)
        rgb_orig = load_frame_rgb_orig(p)
        frame_rgbs_orig.append(rgb_orig)
        rgb = rgb_orig
        if rgb.shape[:2] != (nh, nw):
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        frame_rgbs.append(rgb)

    h, w = frame_tensors[0].shape[1:]
    ref_rgb_path = rgb_paths[0]
    ref_mask_path = mask_path_for_frame(mask_dir, ref_rgb_path)
    uv_dynamic, ref_fg_orig, (orig_h, orig_w) = mask_to_query_uv_from_paths(
        ref_rgb_path,
        ref_mask_path,
        downsample=args.dynamic_downsample,
        max_queries=args.max_dynamic_queries,
        mask_threshold=args.mask_threshold,
    )
    ref_mask_model = resize_mask_nearest(load_mask_gray(ref_mask_path), h, w)
    uv_static, _, _ = _pair.sample_dense_grid(h, w, args.static_downsample)

    if args.save_mask_debug:
        save_mask_debug_overlays(
            os.path.join(args.output_dir, seq_name),
            seq_name,
            ref_rgb_path,
            ref_mask_path,
            frame_rgbs[0],
            args.mask_threshold,
        )

    logging.info(
        "Queries: static=%d (downsample=%d), dynamic=%d (VOS fg orig=%d, "
        "model_res=%d, dynamic_downsample=%d)",
        len(uv_static),
        args.static_downsample,
        len(uv_dynamic),
        int(ref_fg_orig.sum()),
        int((ref_mask_model > args.mask_threshold).sum()),
        args.dynamic_downsample,
    )
    logging.info(
        "Ref frame 0: rgb=%s mask=%s orig_size=%dx%d model_size=%dx%d",
        ref_rgb_path,
        ref_mask_path,
        orig_w,
        orig_h,
        w,
        h,
    )

    frame_data_list = [
        dict(image=ten, image_rgb=rgb, frame_id=Path(p).stem, rgb_path=p)
        for ten, rgb, p in zip(frame_tensors, frame_rgbs, rgb_paths)
    ]
    depth_scale = 1.0
    batch = _pair.build_window_batch(frame_data_list, device, depth_scale)

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_ctx = (
        torch.autocast("cuda", enabled=True, dtype=amp_dtype)
        if args.use_amp
        else nullcontext()
    )
    sdk = pipeline.model
    chunk_size = getattr(sdk, "chunk_size", None)

    with torch.no_grad():
        patch_tokens, time_token = _pair.encode_pair_features(sdk, batch, amp_ctx)

        static_warp3d = decode_dense_pair_warp3d(
            sdk, batch, patch_tokens, time_token,
            (0, 0), uv_static, device,
            args.dense_query_batch_size, chunk_size,
        )
        if static_warp3d.shape[0] != len(uv_static):
            raise RuntimeError(
                f"Static warp3d rows {static_warp3d.shape[0]} != uv_static {len(uv_static)}",
            )
        static_colors = sample_rgb_at_uv(frame_rgbs[0], uv_static)

        dynamic_tracks: list[np.ndarray] = []
        dynamic_colors = sample_rgb_at_uv(frame_rgbs[0], uv_dynamic)

        for t in tqdm(range(n_frames), desc=f"{seq_name} (0,t)"):
            if t == 0:
                dyn_pts = decode_dense_pair_warp3d(
                    sdk, batch, patch_tokens, time_token,
                    (0, 0), uv_dynamic, device,
                    args.dense_query_batch_size, chunk_size,
                )
            else:
                dyn_pts = decode_dense_pair_warp3d(
                    sdk, batch, patch_tokens, time_token,
                    (0, t), uv_dynamic, device,
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
        os.path.join(out_root, "warp3d_static.npz"),
        uv=uv_static,
        warp3d=static_warp3d,
        downsample=args.static_downsample,
    )
    np.savez_compressed(
        os.path.join(out_root, "warp3d_dynamic.npz"),
        uv=uv_dynamic,
        tracks=np.stack(dynamic_tracks, axis=0),
        frame_ids=[Path(p).stem for p in rgb_paths],
    )

    rrd_path = export_davis_rerun(
        out_root,
        seq_name,
        uv_static,
        static_warp3d,
        static_colors,
        dynamic_tracks,
        dynamic_colors,
        frame_rgbs,
        frame_rgbs_orig,
        ref_mask_model,
        args.mask_threshold,
        args.static_vis_step,
        args.max_static_vis_points,
        args.max_traj_vis_points,
        args.traj_vis_downsample,
    )

    summary = dict(
        sequence=seq_name,
        num_frames=n_frames,
        rgb_dir=rgb_dir,
        mask_dir=mask_dir,
        ref_rgb=ref_rgb_path,
        ref_mask=ref_mask_path,
        mask_type="DAVIS_VOS_Annotations",
        orig_fg_pixels=int(ref_fg_orig.sum()),
        model_res_fg_pixels=int((ref_mask_model > args.mask_threshold).sum()),
        static_queries=int(static_warp3d.shape[0]),
        dynamic_queries=int(uv_dynamic.shape[0]),
        static_downsample=args.static_downsample,
        dynamic_downsample=args.dynamic_downsample,
        mask_threshold=args.mask_threshold,
        rrd=rrd_path,
    )
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info("Done %r → %s", seq_name, out_root)


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

    pipeline = None if args.export_rrd_only else load_pipeline(args, cfg)
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
