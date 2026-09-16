#!/usr/bin/env python3
"""
WFMQueryPipeline (MVQuery6): dense motion heatmap inference with 2-frame encoder pairs.

Unlike ``mv_query_fisheye_motion_heatmap_infer.py`` (fisheye + sliding window) or
``mv_query_streaming_motion_heatmap_infer.py`` (streaming state), this script:

  1. Groups frames into non-overlapping pairs: (0,1), (2,3), ...
  2. If the frame count is odd, appends one overlapping pair (N-2, N-1) so every
     frame receives a motion heatmap.
  3. For each pair, encodes exactly 2 frames, then runs pair_forward for both
     (src→tgt) and (tgt→src) to obtain per-frame warp3d_delta magnitudes.

Designed for RGB WFM checkpoints (e.g. wfm_rgb_query DA3-BASE finetune).

Usage (RGB sequence):
    python hAlgorithm/script/infer/motion_head/mv_query_pair2_motion_heatmap_infer.py \\
        --config results/.../wfm_rgb_query_..._backup.py \\
        --load_from results/.../checkpoint/latest/ckpt.pth \\
        --sequence_dir /path/to/frames \\
        --output_dir ./results/pair2_motion_heatmap

Usage (Kosmo / mf_files JSON):
    python ... --data /path/to/scene.json --view_id 0
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

sys.path.append(os.getcwd())

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from hAlgorithm.datasets_4d_fisheye.utils import (
        detect_fisheye_valid_radius,
        generate_fisheye_boundary_mask,
    )
    from hAlgorithm.datasets_4d_fisheye.vis_utils import compute_effective_sensor_size
except ImportError:
    def detect_fisheye_valid_radius(rgb, cx, cy, brightness_threshold=5):
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        nz = np.where(gray > brightness_threshold)
        if len(nz[0]) == 0:
            return float(min(cx, cy, w - cx, h - cy))
        radius = min((nz[1].max() - nz[1].min()) / 2, (nz[0].max() - nz[0].min()) / 2)
        return float(min(radius, cx, cy, w - cx, h - cy))

    def generate_fisheye_boundary_mask(height, width, cx, cy, radius):
        y, x = np.ogrid[:height, :width]
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius

    def compute_effective_sensor_size(original_sensor_size, original_image_size, current_image_size):
        ow, oh = original_image_size
        cw, ch = current_image_size
        px = original_sensor_size[0] / ow
        py = original_sensor_size[1] / oh
        return [px * cw, py * ch]

from hAlgorithm.modules.models2.query_bank.query import BaseQuery
from hAlgorithm.modules.pipelines2.utils.motion_utils import (
    prepare_decoupled_warp3d_delta_inplace,
)
from hAlgorithm.utils import (
    config_merge_args,
    file2dict,
    instantiate_from_config,
    parse_unknown,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_SKIP_WALK_DIR_NAMES = frozenset({
    "groundtruth", "annotations", "annotation", "labels", "masks",
    "depth", "disp", "disparity", "video", "videos", "flow", "flows",
    "occlusion", "segmentation", "segmentations", "segm", "meta", "lists", "list",
    "instances", "instancemasks", "inpainting", "object proposals",
})
_NON_RGB_DIR_FRAGMENTS = (
    "depth_anything", "depthanything", "metric_depth", "monodepth",
    "midas", "zoedepth", "dapth", "disp_anything", "depth_v2", "da_v2",
)

DEFAULT_SENSOR_SIZE_MM = [6.43, 4.87]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="WFMQueryPipeline 2-frame-pair dense motion heatmap inference",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--load_from", default="latest")
    parser.add_argument(
        "--sequence_dir",
        type=str,
        default=None,
        help="Folder of RGB frames (or DAVIS-style tree).",
    )
    parser.add_argument("--sequence_dirs", nargs="+", default=None)
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional mf_files JSON (Kosmo/HaSim).",
    )
    parser.add_argument("--output_dir", type=str, default="./results/pair2_motion_heatmap")
    parser.add_argument("--nums", type=int, default=None, help="Max frames (JSON mode).")
    parser.add_argument("--view_id", type=int, default=None,
                        help="Camera view id (Kosmo fisheye: 8 or 9; pinhole: 0).")
    parser.add_argument(
        "--fisheye",
        action="store_true",
        help="Load Kosmo/JSON frames as fisheye (crop + FISHEYE_BLENDER meta).",
    )
    parser.add_argument(
        "--sensor_size", type=float, nargs=2,
        default=DEFAULT_SENSOR_SIZE_MM,
        metavar=("W_MM", "H_MM"),
        help="Fisheye physical sensor size in mm.",
    )
    parser.add_argument(
        "--fisheye_radius", type=float, default=None,
        help="Valid circle radius in pixels (auto-detect if unset).",
    )
    parser.add_argument(
        "--skip_fisheye_crop", action="store_true",
        help="Disable circular fisheye crop.",
    )
    parser.add_argument("--process_res", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Cap frames per sequence (None = use all).",
    )
    parser.add_argument(
        "--frame_sampling",
        type=str,
        default="sequential",
        choices=["sequential", "debug_trajectory"],
    )
    parser.add_argument("--flat_images_only", action="store_true")
    parser.add_argument("--no_auto_image_dir", action="store_true")
    parser.add_argument("--no_time", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
    )
    parser.add_argument("--dense_downsample", type=int, default=4)
    parser.add_argument("--dense_motion_threshold", type=float, default=0.02)
    parser.add_argument("--dense_query_batch_size", type=int, default=65536)
    parser.add_argument(
        "--disp_aggregation",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="When a frame appears in two overlapping pairs, merge disp maps.",
    )
    parser.add_argument(
        "--scale_mode",
        type=str,
        default="global",
        choices=["global", "per_pair"],
        help=(
            "global: one depth_scale for the whole sequence (stable GIF colors). "
            "per_pair: legacy per-chunk scale."
        ),
    )
    parser.add_argument(
        "--encoder_window",
        type=int,
        default=2,
        help="Frames per encoder forward (e.g. 4, 8, 16, 50).",
    )
    parser.add_argument(
        "--pair_schedule",
        type=str,
        default="adjacent",
        choices=["adjacent", "pair2"],
        help=(
            "adjacent: encode W frames; pairs (0,1)..(W-2,W-1) each with (src,tgt) "
            "and (tgt,src) so every frame gets a mask. "
            "pair2: legacy non-overlapping 2-frame encode pairs."
        ),
    )
    parser.add_argument("--save_gif", action="store_true", default=True)
    parser.add_argument("--gif_fps", type=float, default=4.0)
    parser.add_argument(
        "--gif_suffix",
        type=str,
        default=None,
        help=(
            "GIF filename suffix, e.g. win8 → dense_heatmap_overlay_win8.gif. "
            "Default: win{encoder_window}."
        ),
    )
    # ── Visualization ─────────────────────────────────────────────────────
    parser.add_argument(
        "--heatmap_norm",
        type=str,
        default="global_log_percentile",
        choices=[
            "per_frame_max", "percentile", "log_percentile",
            "global_percentile", "global_log_percentile",
        ],
    )
    parser.add_argument("--heatmap_percentile", type=float, default=95.0)
    parser.add_argument("--heatmap_gamma", type=float, default=1.2)
    parser.add_argument("--heatmap_vmin_percentile", type=float, default=85.0)
    parser.add_argument(
        "--heatmap_colormap",
        type=str,
        default="turbo",
        choices=["jet", "turbo", "inferno", "hot"],
    )
    parser.add_argument("--heatmap_overlay_alpha", type=float, default=0.6)
    parser.add_argument(
        "--overlay_motion_only",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_overlay_motion_only",
        action="store_false",
        dest="overlay_motion_only",
    )
    parser.add_argument(
        "--overlay_use_mask",
        action="store_true",
        default=True,
        help="Overlay only where motion_map > dense_motion_threshold.",
    )
    parser.add_argument(
        "--no_overlay_use_mask",
        action="store_false",
        dest="overlay_use_mask",
    )
    parser.add_argument("--save_raw_max_heatmap", action="store_true", default=False)
    parser.add_argument(
        "--vis_only",
        action="store_true",
        help="Re-render heatmaps/overlays from existing dense_motion_npy (no GPU inference).",
    )
    parser.add_argument(
        "--vis_from",
        type=str,
        default=None,
        help="Source sequence output dir containing dense_motion_npy/ (vis_only).",
    )
    parser.add_argument(
        "--vis_to",
        type=str,
        default=None,
        help="Output dir for vis_only (default: <vis_from>_vis).",
    )

    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown(unknown_args)

    if args.vis_only:
        if args.vis_from is None:
            parser.error("--vis_only requires --vis_from.")
        if not os.path.isdir(os.path.join(args.vis_from, "dense_motion_npy")):
            parser.error(f"dense_motion_npy/ not found under --vis_from: {args.vis_from}")
    elif args.config is None:
        parser.error("--config is required unless --vis_only.")

    has_seq = args.sequence_dir is not None or args.sequence_dirs
    if not args.vis_only and not has_seq and args.data is None:
        parser.error("Provide --sequence_dir, --sequence_dirs, or --data.")

    if args.dense_downsample < 1:
        parser.error("--dense_downsample must be >= 1.")
    if args.encoder_window < 2:
        parser.error("--encoder_window must be >= 2.")
    if args.pair_schedule == "pair2" and args.encoder_window != 2:
        parser.error("pair_schedule=pair2 requires --encoder_window 2.")

    if not args.vis_only:
        if args.no_time:
            config_name = os.path.splitext(os.path.basename(args.config))[0]
            args.output_dir = os.path.join(args.output_dir, config_name)
        else:
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            config_name = os.path.splitext(os.path.basename(args.config))[0]
            args.output_dir = os.path.join(args.output_dir, config_name + "_" + now)
        os.makedirs(args.output_dir, exist_ok=True)

    if args.load_from in ["None", "none", "null"]:
        args.load_from = None

    return args, unknown_args


# ─────────────────────────────────────────────────────────────────────────────
# Image listing
# ─────────────────────────────────────────────────────────────────────────────

def _skip_walk_dir(name: str) -> bool:
    nl = name.lower()
    if nl in _SKIP_WALK_DIR_NAMES:
        return True
    return any(frag in nl for frag in _NON_RGB_DIR_FRAGMENTS)


def _filter_walk_dirnames(_walk_root: str, dirnames: list[str]) -> None:
    lowered = {d.lower() for d in dirnames}
    skip_parallel_gt = "images" in lowered
    dirnames[:] = [
        d for d in dirnames
        if not _skip_walk_dir(d) and not (skip_parallel_gt and d.lower() == "gt")
    ]


def _keep_path_for_rgb_infer(path: str) -> bool:
    s = path.replace("\\", "/").lower()
    banned = (
        "/segmentations/", "/segmentation/", "/annotations/", "/annot/",
        "/labels/", "/masks/", "/depth/", "/disp/", "/disparity/",
        "/optical_flow/", "/flow/", "/inpainting/", "/instances/",
        "/instancemasks/", "/object proposals/",
    )
    if any(b in s for b in banned):
        return False
    return not any(frag in s for frag in _NON_RGB_DIR_FRAGMENTS)


def list_image_paths(image_dir: str, *, recursive: bool = True) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    root = os.path.abspath(os.path.expanduser(image_dir))
    if not os.path.exists(root):
        raise FileNotFoundError(f"sequence_dir does not exist: {root}")
    if not os.path.isdir(root):
        ext = os.path.splitext(root.lower())[1]
        if ext in exts:
            return [root]
        raise NotADirectoryError(f"Not a directory or image: {root}")

    paths: list[str] = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isfile(p) and os.path.splitext(name.lower())[1] in exts:
            paths.append(p)

    if not paths and recursive:
        for walk_root, dirnames, files in os.walk(root):
            _filter_walk_dirnames(walk_root, dirnames)
            for f in sorted(files):
                if os.path.splitext(f.lower())[1] in exts:
                    paths.append(os.path.join(walk_root, f))
        paths.sort()

    return [p for p in paths if _keep_path_for_rgb_infer(p)]


def resolve_image_dir(image_dir: str, *, flat_only: bool, auto: bool = True) -> tuple[str, list[str]]:
    root = os.path.abspath(os.path.expanduser(image_dir))
    if not auto:
        return root, list_image_paths(root, recursive=not flat_only)

    candidates: list[str] = []
    seen: set[str] = set()

    def add(path: str) -> None:
        ap = os.path.abspath(os.path.expanduser(path))
        if ap in seen or not os.path.isdir(ap):
            return
        seen.add(ap)
        candidates.append(ap)

    add(root)
    for parts in (
        ("JPEGImages", "480p"), ("JPEGImages", "1080p"), ("JPEGImages",),
        ("480p",), ("images",), ("Images",), ("RGB",), ("frames",),
    ):
        add(os.path.join(root, *parts))

    try:
        for name in sorted(os.listdir(root)):
            child = os.path.join(root, name)
            if os.path.isdir(child) and not _skip_walk_dir(name):
                add(child)
    except OSError:
        pass

    best_n, chosen, best_paths = 0, root, []
    for c in candidates:
        ps = list_image_paths(c, recursive=not flat_only)
        if len(ps) > best_n:
            best_n, chosen, best_paths = len(ps), c, ps

    if best_n >= 2 and os.path.abspath(chosen) != os.path.abspath(root):
        logging.info("Auto-resolved sequence root: %s → %s (%d images).", root, chosen, best_n)
    return chosen, best_paths


def debug_trajectory_indices(n_total: int, k: int) -> list[int]:
    if n_total < 1:
        return []
    k = int(min(max(k, 1), n_total))
    if k == 1:
        return [0]
    if k == n_total:
        return list(range(n_total))
    step = (n_total - 1) / (k - 1)
    mid = [round(i * step) for i in range(1, k - 1)]
    return [0] + mid + [n_total - 1]


def sample_frame_paths(
    paths: list[str],
    max_frames: int | None,
    sampling: str,
) -> list[str]:
    if max_frames is None or len(paths) <= max_frames:
        return paths
    if sampling == "sequential":
        return paths[:max_frames]
    idx = debug_trajectory_indices(len(paths), max_frames)
    return [paths[i] for i in idx]


def load_frame_tensor(
    path: str,
    process_res: int,
    patch_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(path) as pil_im:
        rgb = np.asarray(pil_im.convert("RGB"))
    oh, ow = rgb.shape[:2]
    scale = process_res / max(oh, ow)
    nh = max(int(oh * scale) // patch_size * patch_size, patch_size)
    nw = max(int(ow * scale) // patch_size * patch_size, patch_size)
    resized = np.ascontiguousarray(cv2.resize(rgb, (nw, nh)))
    f32 = (resized.astype(np.float32, copy=True) / 127.5) - 1.0
    ten = torch.from_numpy(f32).permute(2, 0, 1).contiguous().clone()
    return ten, resized


def sequence_name_from_path(sequence_dir: str, resolved_root: str) -> str:
    root = Path(resolved_root).resolve()
    if root.is_file():
        return root.stem
    return root.name or Path(sequence_dir).name


# ─────────────────────────────────────────────────────────────────────────────
# JSON data
# ─────────────────────────────────────────────────────────────────────────────

def resolve_rgb_path(path: str) -> Optional[str]:
    if path is None:
        return None
    if os.path.isfile(path):
        return path
    for ext in (".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"):
        candidate = path + ext
        if os.path.isfile(candidate):
            return candidate
    return None


def load_data_from_json(json_path, data_root=None):
    with open(json_path, "r") as f:
        data = json.load(f)
    if data_root is None:
        data_root = os.path.dirname(os.path.dirname(json_path))
    mf_files = data.get("mf_files", {})
    if not mf_files:
        raise ValueError(f"No mf_files in {json_path}")
    scene_name = sorted(mf_files.keys())[0]
    scene_data = mf_files[scene_name]
    cam_params = {}
    for cam_key, cam_val in scene_data.get("cam_params", {}).items():
        cam_params[int(cam_key.replace("cam", ""))] = cam_val
    logging.info("Loaded JSON scene: %s", scene_name)
    return scene_data.get("frames", []), scene_name, data_root, cam_params


def parse_fisheye_intrinsics(cam_in: list) -> Tuple[np.ndarray, np.ndarray]:
    if cam_in is None or len(cam_in) < 4:
        raise ValueError("Fisheye cam_in must contain at least fx, fy, cx, cy.")
    fx, fy, cx, cy = cam_in[:4]
    K = np.eye(3, dtype=np.float32)
    K[0, 0], K[1, 1] = fx, fy
    K[0, 2], K[1, 2] = cx, cy
    if len(cam_in) >= 9:
        k = np.array(cam_in[4:9], dtype=np.float32)
    elif len(cam_in) >= 8:
        k = np.array([0.0, *cam_in[4:8]], dtype=np.float32)
    elif len(cam_in) == 5:
        k = np.array(cam_in[4:5], dtype=np.float32)
        k = np.pad(k, (0, 4), mode="constant")
    else:
        raise ValueError(f"Unexpected fisheye K length: {len(cam_in)}")
    if k.shape[0] < 5:
        k = np.pad(k, (0, 5 - k.shape[0]), mode="constant")
    return K, k[:5]


def scale_intrinsics(K: np.ndarray, scale: float) -> np.ndarray:
    Ks = K.copy()
    Ks[0, 0] *= scale
    Ks[1, 1] *= scale
    Ks[0, 2] *= scale
    Ks[1, 2] *= scale
    return Ks


def apply_fisheye_crop(
    rgb: np.ndarray,
    K: np.ndarray,
    radius: Optional[float],
    radius_cfg: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    h, w = rgb.shape[:2]
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if radius_cfg is not None:
        r = float(radius_cfg)
    elif radius is not None:
        r = float(radius)
    else:
        r = float(detect_fisheye_valid_radius(rgb, cx, cy))

    crop_size = int(2 * r)
    crop_x0 = int(np.clip(cx - r, 0, max(w - crop_size, 0)))
    crop_y0 = int(np.clip(cy - r, 0, max(h - crop_size, 0)))

    rgb_c = rgb[crop_y0:crop_y0 + crop_size, crop_x0:crop_x0 + crop_size].copy()
    K_c = K.copy()
    K_c[0, 2] -= crop_x0
    K_c[1, 2] -= crop_y0

    boundary = generate_fisheye_boundary_mask(
        crop_size, crop_size, K_c[0, 2], K_c[1, 2], r,
    )
    return rgb_c, K_c, dict(
        crop_x0=crop_x0, crop_y0=crop_y0, crop_size=crop_size,
        radius=r, boundary_mask=boundary,
    )


def resize_to_model(rgb: np.ndarray, process_res: int, patch_size: int = 14):
    orig_h, orig_w = rgb.shape[:2]
    scale = process_res / max(orig_h, orig_w)
    new_h = max((int(orig_h * scale) // patch_size) * patch_size, patch_size)
    new_w = max((int(orig_w * scale) // patch_size) * patch_size, patch_size)
    rgb_r = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return rgb_r, scale, (orig_h, orig_w)


def load_fisheye_json_frame(
    frame,
    data_root,
    cam_params,
    view_id,
    process_res,
    patch_size,
    sensor_size_mm,
    fisheye_radius_cfg,
    skip_crop,
):
    rgb_rel = frame.get("rgb")
    if rgb_rel is None:
        return None
    full_rgb = rgb_rel if os.path.isabs(rgb_rel) else os.path.join(data_root, rgb_rel)
    full_rgb = resolve_rgb_path(full_rgb)
    if full_rgb is None:
        return None

    img = cv2.cvtColor(cv2.imread(full_rgb), cv2.COLOR_BGR2RGB)
    if img is None:
        return None

    vid = frame.get("view_id", view_id)
    cp = cam_params.get(vid, {})
    if cp.get("type", "").lower() != "fisheye":
        logging.warning("view_id=%s type=%s is not fisheye; skipping.", vid, cp.get("type"))
        return None
    if view_id is not None and vid != view_id:
        return None

    cam_in = frame.get("cam_in") or frame.get("K") or cp.get("K")
    try:
        K_orig, distort_k = parse_fisheye_intrinsics(cam_in)
    except ValueError as exc:
        logging.warning("Frame %s: %s", frame.get("frame_id"), exc)
        return None

    if skip_crop:
        img_work = img
        K_work = K_orig.copy()
        boundary = np.ones(img_work.shape[:2], dtype=bool)
    else:
        img_work, K_work, crop_meta = apply_fisheye_crop(
            img, K_orig, cp.get("fisheye_radius"), fisheye_radius_cfg,
        )
        boundary = crop_meta["boundary_mask"]

    img_resized, img_scale, _ = resize_to_model(img_work, process_res, patch_size)
    K_scaled = scale_intrinsics(K_work, img_scale)
    eff_sensor = compute_effective_sensor_size(
        list(sensor_size_mm),
        (img_work.shape[1], img_work.shape[0]),
        (img_resized.shape[1], img_resized.shape[0]),
    )

    if boundary.shape[:2] != img_resized.shape[:2]:
        boundary = cv2.resize(
            boundary.astype(np.uint8), (img_resized.shape[1], img_resized.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    img_tensor = torch.from_numpy(
        (img_resized.astype(np.float32) / 127.5 - 1.0)
    ).permute(2, 0, 1)

    c2w_raw = frame.get("cam2world_pose")
    c2w = (
        torch.from_numpy(np.array(c2w_raw, dtype=np.float32))
        if c2w_raw is not None else None
    )

    return dict(
        image=img_tensor,
        image_rgb=img_resized,
        intrinsics=torch.from_numpy(K_scaled),
        distort_k=distort_k,
        sensor_size=np.array(eff_sensor, dtype=np.float32),
        boundary_mask=boundary,
        c2w=c2w,
        frame_id=frame.get("frame_id", 0),
        view_id=vid,
        rgb_path=full_rgb,
        is_fisheye=True,
    )


def load_json_frame(frame, data_root, process_res, patch_size, view_id):
    rgb_rel = frame.get("rgb")
    if rgb_rel is None:
        return None
    full_rgb = rgb_rel if os.path.isabs(rgb_rel) else os.path.join(data_root, rgb_rel)
    full_rgb = resolve_rgb_path(full_rgb)
    if full_rgb is None:
        return None

    img = cv2.cvtColor(cv2.imread(full_rgb), cv2.COLOR_BGR2RGB)
    if img is None:
        return None

    vid = frame.get("view_id", view_id if view_id is not None else 0)
    if view_id is not None and vid != view_id:
        return None

    oh, ow = img.shape[:2]
    scale = process_res / max(oh, ow)
    nh = max(int(oh * scale) // patch_size * patch_size, patch_size)
    nw = max(int(ow * scale) // patch_size * patch_size, patch_size)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    img_tensor = torch.from_numpy((img_resized.astype(np.float32) / 127.5) - 1.0).permute(2, 0, 1)

    fd = dict(
        image=img_tensor,
        image_rgb=img_resized,
        frame_id=frame.get("frame_id", 0),
        view_id=vid,
        rgb_path=full_rgb,
        c2w=None,
        intrinsics=None,
    )

    cam_in = frame.get("cam_in") or frame.get("K")
    if cam_in is not None and len(cam_in) >= 4:
        fx, fy, cx, cy = cam_in[0], cam_in[1], cam_in[2], cam_in[3]
        K = np.eye(3, dtype=np.float32)
        K[0, 0], K[1, 1] = fx * scale, fy * scale
        K[0, 2], K[1, 2] = cx * scale, cy * scale
        fd["intrinsics"] = torch.from_numpy(K)

    c2w_raw = frame.get("cam2world_pose")
    if c2w_raw is not None:
        fd["c2w"] = torch.from_numpy(np.array(c2w_raw, dtype=np.float32))

    return fd


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def resolve_load_from(load_from: str, config_path: str) -> Optional[str]:
    if load_from is None:
        return None
    if load_from == "latest":
        return os.path.join(os.path.dirname(config_path), "checkpoint/latest/ckpt.pth")
    if load_from == "best":
        return os.path.join(os.path.dirname(config_path), "checkpoint/best/ckpt.pth")
    return os.path.expanduser(load_from)


def load_pipeline(args, cfg):
    logging.info("Loading WFMQueryPipeline (MVQuery6 pair2 heatmap)...")
    pipeline = instantiate_from_config(cfg["model"]).cuda()
    pipeline.eval()
    pipeline.device = torch.device("cuda")
    pipeline.dtype = (
        torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    )

    if not hasattr(pipeline.model, "pair_forward"):
        raise TypeError(
            "Config model must expose pair_forward (WFMQueryPipeline + MVQuery4/6)."
        )

    ckpt = resolve_load_from(args.load_from, args.config)
    if ckpt is None and cfg.get("trainer", {}).get("load_from"):
        ckpt = cfg["trainer"]["load_from"]
    if ckpt is not None:
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        pipeline.load_checkpoint(ckpt_path=ckpt)
        logging.info("Loaded checkpoint: %s", ckpt)
    else:
        logging.warning("No checkpoint loaded.")

    pipeline.testing_sub_pixel_scale = int(args.dense_downsample)
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Encoder window + pair schedules
# ─────────────────────────────────────────────────────────────────────────────

def build_encoder_chunk_ranges(n_frames: int, window_size: int) -> list[tuple[int, int]]:
    """Non-overlapping [start, end) ranges; each has >= 2 frames; cover all indices."""
    if n_frames < 2:
        return []
    if n_frames <= window_size:
        return [(0, n_frames)]

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < n_frames:
        end = min(start + window_size, n_frames)
        if end - start < 2:
            break
        ranges.append((start, end))
        if end >= n_frames:
            break
        start += window_size

    if ranges and ranges[-1][1] < n_frames:
        tail = n_frames - ranges[-1][1]
        if tail == 1:
            ls, _ = ranges[-1]
            new_start = max(0, n_frames - window_size)
            if new_start <= ls:
                ranges[-1] = (new_start, n_frames)
            else:
                ranges.append((n_frames - 2, n_frames))
    return ranges


def build_pair2_schedule(n_frames: int) -> list[tuple[int, int]]:
    """(0,1), (2,3), ...; if odd N>=3, append overlapping (N-2, N-1)."""
    if n_frames < 2:
        return []
    pairs: list[tuple[int, int]] = []
    for i in range(0, n_frames - 1, 2):
        pairs.append((i, i + 1))
    if n_frames % 2 == 1:
        overlap = (n_frames - 2, n_frames - 1)
        if pairs[-1] != overlap:
            pairs.append(overlap)
    return pairs


def compute_depth_scale(frame_data_list, w2c_aligned):
    all_norms = []
    for i, fd in enumerate(frame_data_list):
        depth, K = fd.get("depth"), fd.get("K_orig")
        if depth is None or K is None:
            continue
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.flatten()
        valid = (z > 1e-3) & (z < 1000.0)
        if valid.sum() < 100:
            continue
        x_c = (u.flatten()[valid] - K[0, 2]) * z[valid] / K[0, 0]
        y_c = (v.flatten()[valid] - K[1, 2]) * z[valid] / K[1, 1]
        pts = np.stack([x_c, y_c, z[valid]], axis=1)
        c2w_al = torch.linalg.inv(w2c_aligned[i]).numpy()
        pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        pts_ref = (c2w_al @ pts_h.T).T[:, :3]
        all_norms.append(np.linalg.norm(pts_ref, axis=1))

    if all_norms:
        return max(float(np.mean(np.concatenate(all_norms))), 0.1)
    if w2c_aligned is not None:
        cam_pos = torch.linalg.inv(w2c_aligned)[:, :3, 3]
        return max(float(cam_pos.norm(dim=-1).max().clamp(min=0.1)), 0.1)
    return 1.0


def compute_pair_scale(pair_fds: list[dict]) -> float:
    c2ws = [fd["c2w"] for fd in pair_fds if fd.get("c2w") is not None]
    if len(c2ws) < 2:
        return 1.0
    c2w_stack = torch.stack(c2ws)
    w2c = torch.linalg.inv(c2w_stack)
    w2c_aligned = w2c @ c2w_stack[0:1]
    return compute_depth_scale(pair_fds, w2c_aligned)


def compute_sequence_scale(frame_data_list: list[dict]) -> float:
    """Single depth_scale from all frames (stable magnitude across pair2 chunks)."""
    c2ws = [fd["c2w"] for fd in frame_data_list if fd.get("c2w") is not None]
    if len(c2ws) < 2:
        return 1.0
    c2w_stack = torch.stack(c2ws)
    w2c = torch.linalg.inv(c2w_stack)
    w2c_aligned = w2c @ c2w_stack[0:1]
    return compute_depth_scale(frame_data_list, w2c_aligned)


def build_window_batch(chunk_fds: list[dict], device: torch.device, depth_scale: float) -> dict:
    W = len(chunk_fds)
    assert W >= 2
    images = torch.stack([fd["image"] for fd in chunk_fds]).unsqueeze(0)
    h, w = chunk_fds[0]["image"].shape[1:]

    has_pose = all(fd.get("c2w") is not None for fd in chunk_fds)
    if has_pose:
        c2w_stack = torch.stack([fd["c2w"] for fd in chunk_fds])
        w2c = torch.linalg.inv(c2w_stack)
        w2c_aligned = w2c @ c2w_stack[0:1]
        w2c_aligned[..., :3, 3] /= depth_scale
        extrinsics = w2c_aligned.unsqueeze(0)
    else:
        eye = torch.eye(4, dtype=torch.float32)
        extrinsics = eye.unsqueeze(0).unsqueeze(0).repeat(1, W, 1, 1)

    scale_t = torch.full((1, W, 1, 1, 1), depth_scale, dtype=torch.float32)
    time_idx = (
        torch.linspace(0.0, 1.0, W).unsqueeze(0)
        if W > 1 else torch.zeros(1, 1)
    )

    meta_data = {
        "name": ["window_infer"],
        "frames": torch.tensor([W]),
        "views": torch.tensor([1]),
        "input_height": torch.tensor([h]),
        "input_width": torch.tensor([w]),
        "origin_height": torch.tensor([h]),
        "origin_width": torch.tensor([w]),
        "data_idx": torch.tensor([0]),
        "camera_type": [["FISHEYE_BLENDER" if chunk_fds[0].get("is_fisheye") else "PINHOLE"]],
        "crop_offset": torch.zeros(1, W, 2, dtype=torch.float32),
    }

    if chunk_fds[0].get("is_fisheye"):
        distort_k = torch.from_numpy(
            np.stack([fd["distort_k"] for fd in chunk_fds], axis=0)
        ).float()
        sensor_size = torch.from_numpy(
            np.stack([fd["sensor_size"] for fd in chunk_fds], axis=0)
        ).float()
        meta_data["distort_k"] = distort_k.unsqueeze(0)
        meta_data["sensor_size"] = sensor_size.unsqueeze(0)

    batch = dict(
        image=images.to(device),
        extrinsics_reff=extrinsics.to(device),
        scale=scale_t.to(device),
        time_idx=time_idx.to(device),
        meta_data=meta_data,
    )

    if all(fd.get("intrinsics") is not None for fd in chunk_fds):
        intrinsics = torch.stack([fd["intrinsics"] for fd in chunk_fds]).unsqueeze(0)
        batch["intrinsics"] = intrinsics.to(device)

    return batch


def build_pair_batch(pair_fds: list[dict], device: torch.device, depth_scale: float) -> dict:
    assert len(pair_fds) == 2
    return build_window_batch(pair_fds, device, depth_scale)


def _assemble_motion_results(
    frame_data_list,
    frame_disp_lists,
    depth_scales,
    use_global_scale,
    global_depth_scale,
    dh,
    dw,
    h,
    w,
    args,
) -> list[dict]:
    N = len(frame_data_list)
    results_list = []
    for gi in range(N):
        disp_list = frame_disp_lists.get(gi, [])
        if not disp_list:
            results_list.append(dict(
                global_i=gi,
                image_rgb=frame_data_list[gi]["image_rgb"],
                motion_map=None,
                motion_mask=None,
                frame_id=frame_data_list[gi].get("frame_id", gi),
            ))
            continue

        disp_norm = disp_list[0] if len(disp_list) == 1 else aggregate_disp(
            disp_list, args.disp_aggregation,
        )
        if use_global_scale:
            scale_used = global_depth_scale
        else:
            scale_used = max(depth_scales.get(gi, [1.0]))  # type: ignore[union-attr]

        map_small = (disp_norm * scale_used).reshape(dh, dw)
        motion_map = cv2.resize(map_small, (w, h), interpolation=cv2.INTER_LINEAR)

        bmask = frame_data_list[gi].get("boundary_mask")
        if bmask is not None:
            if bmask.shape[:2] != (h, w):
                bmask = cv2.resize(
                    bmask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            motion_map = motion_map * bmask.astype(np.float32)

        motion_mask = motion_map > args.dense_motion_threshold
        results_list.append(dict(
            global_i=gi,
            image_rgb=frame_data_list[gi]["image_rgb"],
            motion_map=motion_map,
            motion_mask=motion_mask,
            frame_id=frame_data_list[gi].get("frame_id", gi),
        ))
    return results_list


# ─────────────────────────────────────────────────────────────────────────────
# Dense grid + pair_forward decode
# ─────────────────────────────────────────────────────────────────────────────

def sample_dense_grid(h: int, w: int, downsample: int) -> tuple:
    dh = max(h // downsample, 1)
    dw = max(w // downsample, 1)
    v_coords = np.linspace(0.0, 1.0, dh, dtype=np.float32)
    u_coords = np.linspace(0.0, 1.0, dw, dtype=np.float32)
    uu, vv = np.meshgrid(u_coords, v_coords)
    uv_np = np.stack([uu.flatten(), vv.flatten()], axis=1)
    return uv_np, dh, dw


def aggregate_disp(disp_list: list, mode: str) -> np.ndarray:
    stacked = np.stack(disp_list, axis=0)
    return stacked.max(axis=0) if mode == "max" else stacked.mean(axis=0)


def decode_dense_pair_batched(
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
    Q = len(uv_np)
    segments = []
    q_step = min(query_batch_size, chunk_size or query_batch_size)

    for q0 in range(0, Q, q_step):
        q1 = min(q0 + q_step, Q)
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
        delta = pair_out.get("warp3d_delta")
        if delta is None:
            segments.append(np.zeros(q1 - q0, dtype=np.float32))
            continue
        mag = torch.norm(delta.float(), dim=-1).squeeze(0).cpu().numpy()
        segments.append(mag.astype(np.float32))

    return np.concatenate(segments, axis=0)


def encode_pair_features(sdk, batch, amp_ctx):
    with amp_ctx:
        patch_features, pos, patch_start_idx, _ = sdk.aggregator(
            rgb=batch["image"],
            scale=batch["scale"],
            prompt_depth=None,
            intrinsics=batch.get("intrinsics"),
            ray_directions=None,
            w2c=batch["extrinsics_reff"],
            c2w=None,
            ray_world=None,
            rgb_mask=None,
            meta_data=batch["meta_data"],
        )

    if isinstance(patch_features, (list, tuple)):
        patch_tokens = [feat[:, :, patch_start_idx:] for feat in patch_features]
    else:
        patch_tokens = [patch_features[:, :, patch_start_idx:]]

    time_token = None
    if sdk.time_token_index is not None:
        if isinstance(patch_features, (list, tuple)):
            time_token = patch_features[-1][:, :, sdk.time_token_index]
        else:
            time_token = patch_features[:, :, sdk.time_token_index]

    return patch_tokens, time_token


def run_motion_inference(
    pipeline,
    frame_data_list: list[dict],
    device: torch.device,
    args,
) -> tuple[list[dict], dict]:
    if args.pair_schedule == "pair2":
        return run_pair2_inference(pipeline, frame_data_list, device, args)
    return run_adjacent_window_inference(pipeline, frame_data_list, device, args)


def run_adjacent_window_inference(
    pipeline,
    frame_data_list: list[dict],
    device: torch.device,
    args,
) -> tuple[list[dict], dict]:
    """Encode W-frame windows; adjacent pairs (i,i+1) and (i+1,i) per chunk."""
    sdk = pipeline.model
    N = len(frame_data_list)
    W = args.encoder_window
    chunk_ranges = build_encoder_chunk_ranges(N, W)
    logging.info(
        "Adjacent window schedule: %d frames, encoder_window=%d, %d chunks: %s",
        N, W, len(chunk_ranges), chunk_ranges,
    )

    h, w = frame_data_list[0]["image"].shape[1:]
    grid_uv, dh, dw = sample_dense_grid(h, w, args.dense_downsample)
    chunk_size = getattr(sdk, "chunk_size", None)

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_ctx = (
        torch.autocast("cuda", enabled=True, dtype=amp_dtype)
        if args.use_amp else nullcontext()
    )

    frame_disp_lists: dict[int, list[np.ndarray]] = {}
    depth_scales: dict[int, list[float]] | None = (
        {} if args.scale_mode == "per_pair" else None
    )

    use_global_scale = args.scale_mode == "global"
    global_depth_scale = (
        compute_sequence_scale(frame_data_list) if use_global_scale else None
    )
    if use_global_scale:
        logging.info("Global depth_scale=%.4f (scale_mode=global)", global_depth_scale)

    with torch.no_grad():
        for chunk_start, chunk_end in tqdm(
            chunk_ranges, desc=f"win{W} adjacent encode+decode",
        ):
            chunk_fds = frame_data_list[chunk_start:chunk_end]
            local_len = len(chunk_fds)
            depth_scale = (
                global_depth_scale if use_global_scale
                else compute_sequence_scale(chunk_fds)
            )
            batch = build_window_batch(chunk_fds, device, depth_scale)
            patch_tokens, time_token = encode_pair_features(sdk, batch, amp_ctx)

            for i in range(local_len - 1):
                for local_src, local_tgt in ((i, i + 1), (i + 1, i)):
                    mag = decode_dense_pair_batched(
                        sdk, batch, patch_tokens, time_token,
                        (local_src, local_tgt), grid_uv, device,
                        args.dense_query_batch_size, chunk_size,
                    )
                    global_idx = chunk_start + local_src
                    frame_disp_lists.setdefault(global_idx, []).append(mag)
                    if depth_scales is not None:
                        depth_scales.setdefault(global_idx, []).append(depth_scale)

    results_list = _assemble_motion_results(
        frame_data_list, frame_disp_lists, depth_scales or {},
        use_global_scale, global_depth_scale, dh, dw, h, w, args,
    )
    infer_meta = dict(
        global_depth_scale=global_depth_scale,
        scale_mode=args.scale_mode,
        encoder_window=W,
        pair_schedule=args.pair_schedule,
        encoder_chunk_ranges=chunk_ranges,
    )
    return results_list, infer_meta


def run_pair2_inference(
    pipeline,
    frame_data_list: list[dict],
    device: torch.device,
    args,
) -> tuple[list[dict], dict]:
    sdk = pipeline.model
    N = len(frame_data_list)
    pairs = build_pair2_schedule(N)
    logging.info("Pair2 schedule (%d frames → %d pairs): %s", N, len(pairs), pairs)

    h, w = frame_data_list[0]["image"].shape[1:]
    grid_uv, dh, dw = sample_dense_grid(h, w, args.dense_downsample)
    chunk_size = getattr(sdk, "chunk_size", None)

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_ctx = (
        torch.autocast("cuda", enabled=True, dtype=amp_dtype)
        if args.use_amp else nullcontext()
    )

    # global frame index -> list of displacement vectors (small grid, pre-scale)
    frame_disp_lists: dict[int, list[np.ndarray]] = {}
    depth_scales: dict[int, list[float]] | None = (
        {} if args.scale_mode == "per_pair" else None
    )

    use_global_scale = args.scale_mode == "global"
    global_depth_scale = (
        compute_sequence_scale(frame_data_list) if use_global_scale else None
    )
    if use_global_scale:
        logging.info("Global depth_scale=%.4f (scale_mode=global)", global_depth_scale)

    with torch.no_grad():
        for global_a, global_b in tqdm(pairs, desc="Pair2 encode+decode"):
            pair_fds = [frame_data_list[global_a], frame_data_list[global_b]]
            depth_scale = (
                global_depth_scale if use_global_scale
                else compute_pair_scale(pair_fds)
            )
            batch = build_pair_batch(pair_fds, device, depth_scale)
            patch_tokens, time_token = encode_pair_features(sdk, batch, amp_ctx)

            for local_src, global_idx in ((0, global_a), (1, global_b)):
                mag = decode_dense_pair_batched(
                    sdk, batch, patch_tokens, time_token,
                    (local_src, 1 - local_src), grid_uv, device,
                    args.dense_query_batch_size, chunk_size,
                )
                frame_disp_lists.setdefault(global_idx, []).append(mag)
                if depth_scales is not None:
                    depth_scales.setdefault(global_idx, []).append(depth_scale)

    results_list = _assemble_motion_results(
        frame_data_list, frame_disp_lists, depth_scales or {},
        use_global_scale, global_depth_scale, dh, dw, h, w, args,
    )

    infer_meta = dict(
        global_depth_scale=global_depth_scale,
        scale_mode=args.scale_mode,
        encoder_window=2,
        pair_schedule="pair2",
    )
    return results_list, infer_meta


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

_COLORMAP = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "hot": cv2.COLORMAP_HOT,
}


def _robust_scale(motion_map, percentile, global_linear_scale):
    if global_linear_scale is not None and global_linear_scale > 1e-8:
        return global_linear_scale
    pos = motion_map[motion_map > 1e-6]
    if pos.size < 16:
        return max(float(motion_map.max()), 1e-6)
    return max(float(np.percentile(pos, percentile)), 1e-6)


def compute_global_vis_params(motion_maps, norm_mode, percentile):
    """Shared colormap scale(s) for global_percentile / global_log_percentile."""
    if norm_mode not in ("global_percentile", "global_log_percentile"):
        return {}
    vals = [m[m > 1e-6] for m in motion_maps if m is not None]
    if not vals:
        return {"global_vis_linear_scale": 1.0}
    all_pos = np.concatenate(vals)
    linear_scale = max(float(np.percentile(all_pos, percentile)), 1e-6)
    params = {"global_vis_linear_scale": linear_scale}
    if norm_mode == "global_log_percentile":
        log_vals = np.log1p(all_pos / linear_scale)
        params["global_vis_log_hi"] = max(
            float(np.percentile(log_vals, percentile)), 1e-6,
        )
    return params


def normalize_motion_for_vis(
    motion_map,
    norm_mode,
    percentile,
    gamma,
    global_vis_params=None,
):
    gparams = global_vis_params or {}
    global_linear = gparams.get("global_vis_linear_scale")
    global_log_hi = gparams.get("global_vis_log_hi")

    scale = _robust_scale(motion_map, percentile, global_linear)
    if norm_mode == "per_frame_max":
        normed = motion_map / max(float(motion_map.max()), 1e-6)
    elif norm_mode in ("log_percentile", "global_log_percentile"):
        normed = np.log1p(motion_map / scale)
        if norm_mode == "global_log_percentile" and global_log_hi is not None:
            hi = global_log_hi
        else:
            hi = max(float(np.percentile(normed, percentile)), 1e-6)
        normed = np.clip(normed / hi, 0.0, 1.0)
    else:
        normed = np.clip(motion_map / scale, 0.0, 1.0)
    if gamma > 0 and abs(gamma - 1.0) > 1e-6:
        normed = np.power(normed, gamma)
    return (np.clip(normed, 0.0, 1.0) * 255).astype(np.uint8)


def motion_map_to_heatmap_bgr(map_u8, colormap):
    cmap = _COLORMAP.get(colormap, cv2.COLORMAP_TURBO)
    return cv2.applyColorMap(map_u8, cmap)


def blend_motion_overlay(
    rgb,
    motion_map,
    heat_bgr,
    alpha,
    vmin_percentile,
    motion_only,
    overlay_use_mask=False,
    motion_threshold=0.02,
):
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not motion_only:
        return cv2.addWeighted(rgb_bgr, 1.0 - alpha, heat_bgr, alpha, 0)
    if overlay_use_mask:
        fg = motion_map > motion_threshold
    else:
        pos = motion_map[motion_map > 1e-6]
        if pos.size < 16:
            return rgb_bgr
        vmin = float(np.percentile(pos, vmin_percentile))
        fg = motion_map > vmin
    out = rgb_bgr.copy()
    if fg.any():
        out[fg] = (
            rgb_bgr[fg].astype(np.float32) * (1.0 - alpha)
            + heat_bgr[fg].astype(np.float32) * alpha
        ).astype(np.uint8)
    return out


def load_motion_npy_results(npy_dir: str) -> list[dict]:
    """Load per-frame motion maps from dense_motion_npy/."""
    files = sorted(Path(npy_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {npy_dir}")

    results = []
    for f in files:
        stem = f.stem
        if "_frame" in stem:
            global_s, frame_id = stem.split("_frame", 1)
            global_i = int(global_s)
        else:
            global_i = int(stem)
            frame_id = str(global_i)
        motion_map = np.load(str(f))
        results.append(dict(
            global_i=global_i,
            frame_id=frame_id,
            motion_map=motion_map.astype(np.float32),
        ))
    results.sort(key=lambda r: r["global_i"])
    return results


def load_infer_meta_from_summary(vis_from: str) -> dict:
    summary_path = os.path.join(vis_from, "summary.json")
    if not os.path.isfile(summary_path):
        return {}
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def build_frame_data_list(args, process_res: int, patch_size: int) -> list[dict]:
    """Load RGB frames only (no model) for vis_only overlay."""
    frame_data_list: list[dict] = []

    if args.data is not None:
        frames, _scene_name, data_root, cam_params = load_data_from_json(args.data)
        if args.view_id is not None:
            frames = [f for f in frames if f.get("view_id", args.view_id) == args.view_id]
        if args.nums is not None:
            frames = frames[: args.nums]
        if args.max_frames is not None:
            frames = frames[: args.max_frames]

        desc = "Loading fisheye JSON frames (vis)" if args.fisheye else "Loading JSON frames (vis)"
        for fr in tqdm(frames, desc=desc):
            if args.fisheye:
                fd = load_fisheye_json_frame(
                    fr, data_root, cam_params, args.view_id,
                    process_res, patch_size, args.sensor_size,
                    args.fisheye_radius, args.skip_fisheye_crop,
                )
            else:
                fd = load_json_frame(fr, data_root, process_res, patch_size, args.view_id)
            if fd is not None:
                if args.fisheye and fd.get("c2w") is None:
                    continue
                frame_data_list.append(fd)
        return frame_data_list

    seq_dirs: list[str] = []
    if args.sequence_dirs:
        seq_dirs.extend(args.sequence_dirs)
    if args.sequence_dir:
        seq_dirs.append(args.sequence_dir)
    if len(seq_dirs) != 1:
        raise RuntimeError("vis_only with --sequence_dir expects exactly one sequence.")

    resolved, paths = resolve_image_dir(
        seq_dirs[0],
        flat_only=args.flat_images_only,
        auto=not args.no_auto_image_dir,
    )
    paths = sample_frame_paths(paths, args.max_frames, args.frame_sampling)
    for p in paths:
        ten, rgb = load_frame_tensor(p, process_res, patch_size)
        frame_data_list.append(dict(
            image=ten,
            image_rgb=rgb,
            frame_id=Path(p).stem,
            rgb_path=p,
            c2w=None,
            intrinsics=None,
        ))
    return frame_data_list


def merge_vis_results(motion_results: list[dict], frame_data_list: list[dict], threshold: float):
    """Attach RGB and recompute motion_mask for vis_only."""
    rgb_by_idx = {i: fd["image_rgb"] for i, fd in enumerate(frame_data_list)}
    merged = []
    for res in motion_results:
        gi = res["global_i"]
        if gi >= len(frame_data_list):
            logging.warning("Skip npy global_i=%d (only %d RGB frames loaded)", gi, len(frame_data_list))
            continue
        motion_map = res["motion_map"]
        merged.append(dict(
            global_i=gi,
            frame_id=res.get("frame_id", frame_data_list[gi].get("frame_id", gi)),
            image_rgb=rgb_by_idx[gi],
            motion_map=motion_map,
            motion_mask=motion_map > threshold,
        ))
    return merged


def run_vis_only(args):
    vis_from = os.path.abspath(args.vis_from)
    vis_to = os.path.abspath(args.vis_to or f"{vis_from}_vis")
    os.makedirs(vis_to, exist_ok=True)

    process_res = args.process_res or 504
    patch_size = args.patch_size or 14

    motion_results = load_motion_npy_results(os.path.join(vis_from, "dense_motion_npy"))
    frame_data_list = build_frame_data_list(args, process_res, patch_size)
    if len(frame_data_list) < len(motion_results):
        logging.warning(
            "Loaded %d RGB frames but %d npy maps; truncating to RGB count.",
            len(frame_data_list), len(motion_results),
        )
        motion_results = [r for r in motion_results if r["global_i"] < len(frame_data_list)]

    results_list = merge_vis_results(
        motion_results, frame_data_list, args.dense_motion_threshold,
    )
    if not results_list:
        raise RuntimeError("No frames to render in vis_only mode.")

    infer_meta = load_infer_meta_from_summary(vis_from)
    if infer_meta.get("encoder_window") is not None:
        args.encoder_window = int(infer_meta["encoder_window"])
    seq_name = infer_meta.get("sequence") or Path(vis_from).name
    pair_schedule = infer_meta.get("encoder_chunk_ranges") or infer_meta.get("pair_schedule")

    logging.info(
        "vis_only: %d frames from %s → %s (overlay_use_mask=%s, gamma=%.2f, p=%.0f)",
        len(results_list), vis_from, vis_to,
        args.overlay_use_mask, args.heatmap_gamma, args.heatmap_percentile,
    )
    save_sequence_outputs(results_list, vis_to, args, seq_name, pair_schedule, infer_meta)

    with open(os.path.join(vis_to, "vis_only_args.json"), "w", encoding="utf-8") as f:
        payload = {k: getattr(args, k) for k in vars(args)}
        payload["vis_from_resolved"] = vis_from
        payload["vis_to_resolved"] = vis_to
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logging.info("vis_only done → %s", vis_to)


def get_gif_filename(args) -> str:
    suffix = args.gif_suffix if args.gif_suffix is not None else f"win{args.encoder_window}"
    return f"dense_heatmap_overlay_{suffix}.gif" if suffix else "dense_heatmap_overlay.gif"


def compute_global_vis_scale(motion_maps, norm_mode, percentile):
    """Backward-compatible helper; returns linear scale only."""
    params = compute_global_vis_params(motion_maps, norm_mode, percentile)
    return params.get("global_vis_linear_scale")


def save_sequence_outputs(results_list, out_root, args, seq_name, pair_schedule, infer_meta=None):
    out_dirs = {
        "mask": os.path.join(out_root, "dense_motion_masks"),
        "heatmap": os.path.join(out_root, "dense_motion_heatmaps"),
        "npy": os.path.join(out_root, "dense_motion_npy"),
        "overlay": os.path.join(out_root, "dense_heatmap_overlay"),
        "heatmap_raw": os.path.join(out_root, "dense_motion_heatmaps_raw"),
    }
    for key, d in out_dirs.items():
        if key == "heatmap_raw" and not args.save_raw_max_heatmap:
            continue
        os.makedirs(d, exist_ok=True)

    all_maps = [r["motion_map"] for r in results_list if r.get("motion_map") is not None]
    global_vis_params = compute_global_vis_params(
        all_maps, args.heatmap_norm, args.heatmap_percentile,
    )
    if global_vis_params:
        logging.info("Global vis params: %s", global_vis_params)

    gif_frames = []
    saved = 0
    for res in results_list:
        motion_map = res.get("motion_map")
        if motion_map is None:
            continue

        stem = f"{res['global_i']:06d}_frame{res['frame_id']}"
        map_u8 = normalize_motion_for_vis(
            motion_map,
            args.heatmap_norm,
            args.heatmap_percentile,
            args.heatmap_gamma,
            global_vis_params=global_vis_params or None,
        )
        heat_bgr = motion_map_to_heatmap_bgr(map_u8, args.heatmap_colormap)
        cv2.imwrite(os.path.join(out_dirs["heatmap"], f"{stem}.png"), heat_bgr)
        np.save(os.path.join(out_dirs["npy"], f"{stem}.npy"), motion_map)

        if args.save_raw_max_heatmap:
            raw_u8 = normalize_motion_for_vis(motion_map, "per_frame_max", 100.0, 1.0)
            cv2.imwrite(
                os.path.join(out_dirs["heatmap_raw"], f"{stem}.png"),
                motion_map_to_heatmap_bgr(raw_u8, "jet"),
            )

        motion_mask = res.get("motion_mask")
        if motion_mask is not None:
            cv2.imwrite(
                os.path.join(out_dirs["mask"], f"{stem}.png"),
                (motion_mask.astype(np.uint8) * 255),
            )

        overlay = blend_motion_overlay(
            res["image_rgb"],
            motion_map,
            heat_bgr,
            args.heatmap_overlay_alpha,
            args.heatmap_vmin_percentile,
            args.overlay_motion_only,
            overlay_use_mask=args.overlay_use_mask,
            motion_threshold=args.dense_motion_threshold,
        )
        cv2.imwrite(os.path.join(out_dirs["overlay"], f"{stem}.png"), overlay)
        gif_frames.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        saved += 1

    if args.save_gif and gif_frames:
        try:
            from PIL import Image as PILImage
            pil = [PILImage.fromarray(f) for f in gif_frames]
            gif_name = get_gif_filename(args)
            gif_path = os.path.join(out_root, gif_name)
            pil[0].save(
                gif_path, save_all=True, append_images=pil[1:],
                duration=int(1000 / args.gif_fps), loop=0,
            )
            logging.info("Saved GIF: %s", gif_path)
        except ImportError:
            logging.warning("Pillow not available; skipping GIF.")

    summary = dict(
        sequence=seq_name,
        total_frames=len(results_list),
        saved_frames=saved,
        pair_schedule=args.pair_schedule,
        encoder_window=args.encoder_window,
        encoder_chunk_ranges=(infer_meta or {}).get("encoder_chunk_ranges"),
        scale_mode=args.scale_mode,
        global_depth_scale=(infer_meta or {}).get("global_depth_scale"),
        dense_downsample=args.dense_downsample,
        dense_motion_threshold=args.dense_motion_threshold,
        disp_aggregation=args.disp_aggregation,
        heatmap_norm=args.heatmap_norm,
        heatmap_gamma=args.heatmap_gamma,
        heatmap_percentile=args.heatmap_percentile,
        heatmap_vmin_percentile=args.heatmap_vmin_percentile,
        overlay_use_mask=args.overlay_use_mask,
        gif_filename=get_gif_filename(args),
        global_vis_params=global_vis_params or None,
        output_dir=out_root,
    )
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info("Sequence %r: saved %d heatmaps → %s", seq_name, saved, out_root)
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Job runners
# ─────────────────────────────────────────────────────────────────────────────

def run_rgb_sequence(pipeline, sequence_dir, args, device, process_res, patch_size):
    resolved, paths = resolve_image_dir(
        sequence_dir,
        flat_only=args.flat_images_only,
        auto=not args.no_auto_image_dir,
    )
    paths = sample_frame_paths(paths, args.max_frames, args.frame_sampling)
    if len(paths) < 2:
        raise RuntimeError(f"Need >= 2 frames under {resolved}, got {len(paths)}")

    seq_name = sequence_name_from_path(sequence_dir, resolved)
    out_root = os.path.join(args.output_dir, seq_name)
    os.makedirs(out_root, exist_ok=True)

    frame_data_list = []
    for p in paths:
        ten, rgb = load_frame_tensor(p, process_res, patch_size)
        frame_data_list.append(dict(
            image=ten,
            image_rgb=rgb,
            frame_id=Path(p).stem,
            rgb_path=p,
            c2w=None,
            intrinsics=None,
        ))

    pair_schedule = (
        build_pair2_schedule(len(frame_data_list))
        if args.pair_schedule == "pair2"
        else build_encoder_chunk_ranges(len(frame_data_list), args.encoder_window)
    )
    results, infer_meta = run_motion_inference(pipeline, frame_data_list, device, args)
    save_sequence_outputs(results, out_root, args, seq_name, pair_schedule, infer_meta)


def run_json_sequence(pipeline, args, device, process_res, patch_size):
    frames, scene_name, data_root, cam_params = load_data_from_json(args.data)
    if args.view_id is not None:
        frames = [f for f in frames if f.get("view_id", args.view_id) == args.view_id]
    if args.nums is not None:
        frames = frames[: args.nums]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    seq_name = scene_name if args.view_id is None else f"{scene_name}_cam{args.view_id}"
    out_root = os.path.join(args.output_dir, seq_name)
    os.makedirs(out_root, exist_ok=True)

    frame_data_list = []
    desc = "Loading fisheye JSON frames" if args.fisheye else "Loading JSON frames"
    for fr in tqdm(frames, desc=desc):
        if args.fisheye:
            fd = load_fisheye_json_frame(
                fr, data_root, cam_params, args.view_id,
                process_res, patch_size, args.sensor_size,
                args.fisheye_radius, args.skip_fisheye_crop,
            )
        else:
            fd = load_json_frame(fr, data_root, process_res, patch_size, args.view_id)
        if fd is not None:
            if args.fisheye and fd.get("c2w") is None:
                continue
            frame_data_list.append(fd)

    if len(frame_data_list) < 2:
        raise RuntimeError("Need >= 2 valid frames from JSON.")

    pair_schedule = (
        build_pair2_schedule(len(frame_data_list))
        if args.pair_schedule == "pair2"
        else build_encoder_chunk_ranges(len(frame_data_list), args.encoder_window)
    )
    results, infer_meta = run_motion_inference(pipeline, frame_data_list, device, args)
    save_sequence_outputs(results, out_root, args, seq_name, pair_schedule, infer_meta)


def main():
    args, unknown_args = parse_args()

    if args.vis_only:
        if args.data is None and args.sequence_dir is None and not args.sequence_dirs:
            raise SystemExit("vis_only requires --data or --sequence_dir to reload RGB frames.")
        run_vis_only(args)
        return

    if args.config is None:
        raise SystemExit("--config is required for inference.")
    cfg = file2dict(args.config)
    config_merge_args(cfg, unknown_args)

    process_res = args.process_res or int(cfg.get("max_size", 504))
    patch_size = args.patch_size or int(cfg.get("patch_size", 14))

    pipeline = load_pipeline(args, cfg)
    device = torch.device("cuda")

    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    failures: list[tuple[str, str]] = []

    seq_dirs: list[str] = []
    if args.sequence_dirs:
        seq_dirs.extend(args.sequence_dirs)
    if args.sequence_dir:
        seq_dirs.append(args.sequence_dir)

    for seq_dir in seq_dirs:
        try:
            run_rgb_sequence(pipeline, seq_dir, args, device, process_res, patch_size)
        except Exception as exc:
            logging.exception("Failed RGB sequence %s: %s", seq_dir, exc)
            failures.append((seq_dir, str(exc)))

    if args.data is not None:
        try:
            run_json_sequence(pipeline, args, device, process_res, patch_size)
        except Exception as exc:
            logging.exception("Failed JSON sequence: %s", exc)
            failures.append((args.data, str(exc)))

    if failures:
        for name, msg in failures:
            logging.error("  %s: %s", name, msg)
        sys.exit(1)

    logging.info("All done → %s", args.output_dir)


if __name__ == "__main__":
    main()
