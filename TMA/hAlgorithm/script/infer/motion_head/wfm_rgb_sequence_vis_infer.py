#!/usr/bin/env python3
"""
WFM RGB sequence inference: vis-only (no eval), model-predicted depth/pose.

For each RGB image sequence directory or video file:
  1. Collect sorted frames (DAVIS-style auto-resolve optional) or extract sampled
     frames from video (cv2).
  2. Sample ``max_frames`` with DebugTrajectory-style uniform indices (default 30).
  3. Resize/normalize like val (max_size=504, patch 14).
  4. ``WFMQueryPipeline.infer`` then ``visualize`` (same path as ``test_dist.sh`` + ``--test_vis``).

Query density is controlled by ``--query_stride`` (maps to ``pipeline.testing_sub_pixel_scale``,
default 4 ≈ 1/16 queries vs full grid). Use ``--query_stride 1`` to match dense test infer.

No DA3, no GT depth/pose, no eval metrics, no 4DGS vis.

Usage:
    python hAlgorithm/script/infer/motion_head/wfm_rgb_sequence_vis_infer.py \\
        --motion_config results/.../test_hasim_benchmark_hard.py \\
        --load_from results/.../checkpoint/latest/ckpt.pth \\
        --sequence_dir /path/to/JPEGImages/480p/bear \\
        --output_dir ./results/rgb_sequence_vis

    python ... --sequence_dirs /path/a /path/b --max_frames 30 --query_stride 4

    python ... --video /path/to/clip.mp4 --videos a.mp4 b.mp4 --keep_frames
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path

sys.path.append(os.getcwd())

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

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
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v", ".gif"}


def parse_args():
    p = argparse.ArgumentParser(description="WFM RGB sequence vis-only infer")
    p.add_argument("--motion_config", type=str, required=True)
    p.add_argument("--load_from", type=str, required=True)
    p.add_argument(
        "--sequence_dir",
        type=str,
        default=None,
        help="Single RGB sequence root (folder of frames or DAVIS-style tree).",
    )
    p.add_argument(
        "--sequence_dirs",
        nargs="+",
        default=None,
        help="Multiple sequence roots (one vis job each).",
    )
    p.add_argument("--video", type=str, default=None, help="Single video file.")
    p.add_argument("--videos", nargs="+", default=None, help="Multiple video files.")
    p.add_argument(
        "--keep_frames",
        action="store_true",
        help="Keep extracted video frames under output_dir/frames/.",
    )
    p.add_argument("--output_dir", type=str, default="./results/wfm_rgb_sequence_vis")
    p.add_argument(
        "--max_frames",
        type=int,
        default=30,
        help="Frames per sequence (DebugTrajectory uniform sampling by default). "
             "Ignored when --all_frames is set.",
    )
    p.add_argument(
        "--all_frames",
        action="store_true",
        help="Use every frame (no temporal subsampling). Long sequences are split "
             "into --infer_chunk_size chunks to avoid OOM.",
    )
    p.add_argument(
        "--infer_chunk_size",
        type=int,
        default=0,
        help="Max frames per infer/visualize chunk (0 = single batch). "
             "Recommended 20–30 for full-length / native-res videos.",
    )
    p.add_argument(
        "--frame_sampling",
        type=str,
        default="debug_trajectory",
        choices=["debug_trajectory", "sequential"],
        help="debug_trajectory: uniform incl. first/last (val DebugTrajectory). "
             "sequential: first max_frames frames.",
    )
    p.add_argument(
        "--query_stride",
        type=int,
        default=4,
        help="Uniform layout: maps to testing_sub_pixel_scale. "
             "center_dense_columns: sparse side-column stride.",
    )
    p.add_argument(
        "--query_layout",
        type=str,
        default="uniform",
        choices=["uniform", "center_dense_columns"],
        help="uniform: regular grid. center_dense_columns: 4 vertical bands; "
             "middle two denser, outer two sparser (no detector).",
    )
    p.add_argument(
        "--query_dense_stride",
        type=int,
        default=2,
        help="center_dense_columns: stride for middle two vertical bands.",
    )
    p.add_argument(
        "--query_sparse_stride",
        type=int,
        default=None,
        help="center_dense_columns: stride for outer bands (default: max(4, 2*query_stride)).",
    )
    p.add_argument(
        "--query_dense_band_indices",
        type=str,
        default="1,2",
        help="center_dense_columns: which of 4 vertical bands use dense_stride "
             "(0=left … 3=right). Default 1,2 = center half; use 2,3 for right half.",
    )
    p.add_argument(
        "--process_res",
        type=int,
        default=None,
        help="Max side length; default: max_size from motion config or 504.",
    )
    p.add_argument(
        "--native_resolution",
        action="store_true",
        help="Set process_res to max(w,h) of the first frame (no downscale; "
             "still aligned to patch_size).",
    )
    p.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="Default: patch_size from motion config or 14.",
    )
    p.add_argument(
        "--flat_images_only",
        action="store_true",
        help="Do not recurse subfolders when listing images.",
    )
    p.add_argument(
        "--no_auto_image_dir",
        action="store_true",
        help="Disable DAVIS-style auto-resolve of image roots.",
    )
    p.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use autocast in pipeline.infer (default on).",
    )
    p.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable autocast in infer.",
    )
    p.add_argument(
        "--amp_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
    )
    p.add_argument(
        "--ckpt_iter_tag",
        type=str,
        default=None,
        help="Visualization subfolder iter_XXXXXX; default parsed from --load_from.",
    )
    p.add_argument("--no_time", action="store_true")
    args, unknown = p.parse_known_args()
    unknown = parse_unknown(unknown)

    if args.no_amp:
        args.use_amp = False

    if (
        args.sequence_dir is None
        and not args.sequence_dirs
        and args.video is None
        and not args.videos
    ):
        p.error("Provide --sequence_dir/--sequence_dirs or --video/--videos.")

    if args.query_stride < 1:
        p.error("--query_stride must be >= 1.")
    if args.query_dense_stride < 1:
        p.error("--query_dense_stride must be >= 1.")
    if args.query_sparse_stride is None:
        args.query_sparse_stride = max(4, int(args.query_stride) * 2)
    if args.query_sparse_stride < 1:
        p.error("--query_sparse_stride must be >= 1.")
    try:
        args.query_dense_band_indices = tuple(
            int(x.strip()) for x in str(args.query_dense_band_indices).split(",") if x.strip()
        )
    except ValueError:
        p.error("--query_dense_band_indices must be comma-separated integers, e.g. 1,2")
    if not args.query_dense_band_indices:
        p.error("--query_dense_band_indices must list at least one band index.")
    if any(i < 0 or i > 3 for i in args.query_dense_band_indices):
        p.error("--query_dense_band_indices band indices must be in [0, 3].")

    if args.all_frames:
        args.max_frames = 10**9
    elif args.max_frames < 2:
        p.error("--max_frames must be >= 2 for motion pairs.")

    if args.infer_chunk_size != 0 and args.infer_chunk_size < 2:
        p.error("--infer_chunk_size must be >= 2 when set.")

    if args.no_time:
        stem = os.path.splitext(os.path.basename(args.motion_config))[0]
        args.output_dir = os.path.join(args.output_dir, stem)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = os.path.splitext(os.path.basename(args.motion_config))[0]
        args.output_dir = os.path.join(args.output_dir, stem + "_" + stamp)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.load_from in ("None", "none", "null"):
        args.load_from = None

    return args, unknown


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
    """Match ``DebugTrajectory`` in datasets_mv/clip_sampler/debug_trajectory.py."""
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


def sample_frame_indices(n_total: int, max_frames: int, sampling: str) -> list[int]:
    if n_total <= max_frames:
        return list(range(n_total))
    if sampling == "sequential":
        return list(range(max_frames))
    return debug_trajectory_indices(n_total, max_frames)


def sample_frame_paths(
    paths: list[str],
    max_frames: int,
    sampling: str,
) -> list[str]:
    idx = sample_frame_indices(len(paths), max_frames, sampling)
    return [paths[i] for i in idx]


def resolve_video_path(path: str) -> str:
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Video not found: {p}")
    if os.path.splitext(p.lower())[1] not in VIDEO_EXTS:
        raise ValueError(f"Unsupported video extension: {p}")
    return p


def _read_all_video_frames(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames_bgr: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()
    return frames_bgr


def _write_sampled_frames(
    frames_bgr: list[np.ndarray],
    indices: list[int],
    cache_dir: str | None,
) -> tuple[list[str], str | None]:
    tmp_obj = tempfile.mkdtemp(prefix="wfm_vid_frames_") if cache_dir is None else cache_dir
    os.makedirs(tmp_obj, exist_ok=True)
    paths: list[str] = []
    for i, idx in enumerate(indices):
        out = os.path.join(tmp_obj, f"frame_{i:06d}.png")
        cv2.imwrite(out, frames_bgr[idx])
        paths.append(out)
    cleanup = None if cache_dir else tmp_obj
    return paths, cleanup


def extract_video_frame_paths(
    video_path: str,
    max_frames: int,
    sampling: str,
    cache_dir: str | None,
) -> tuple[list[str], str | None]:
    """Return sampled frame image paths and optional temp dir to clean up."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total >= 2:
        indices = sample_frame_indices(total, max_frames, sampling)
        tmp_obj = tempfile.mkdtemp(prefix="wfm_vid_frames_") if cache_dir is None else cache_dir
        os.makedirs(tmp_obj, exist_ok=True)
        paths: list[str] = []
        seek_ok = True
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                seek_ok = False
                break
            out = os.path.join(tmp_obj, f"frame_{i:06d}.png")
            cv2.imwrite(out, frame)
            paths.append(out)
        cap.release()
        if seek_ok:
            cleanup = None if cache_dir else tmp_obj
            return paths, cleanup
        if cache_dir is None and os.path.isdir(tmp_obj):
            shutil.rmtree(tmp_obj, ignore_errors=True)
        logging.warning(
            "Seek-based frame read failed for %s; falling back to sequential decode.",
            video_path,
        )
    else:
        cap.release()

    frames_bgr = _read_all_video_frames(video_path)
    if len(frames_bgr) < 2:
        raise RuntimeError(f"Video has < 2 frames: {video_path}")
    indices = sample_frame_indices(len(frames_bgr), max_frames, sampling)
    return _write_sampled_frames(frames_bgr, indices, cache_dir)


def native_process_res_from_path(path: str) -> int:
    ext = os.path.splitext(path.lower())[1]
    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for resolution probe: {path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if w < 1 or h < 1:
            raise RuntimeError(f"Invalid video resolution: {path}")
        return max(w, h)
    with Image.open(path) as pil_im:
        w, h = pil_im.size
    return max(w, h)


def iter_frame_chunks(
    paths: list[str],
    chunk_size: int,
) -> list[tuple[int, list[str]]]:
    if chunk_size <= 0 or len(paths) <= chunk_size:
        return [(0, paths)]
    chunks: list[tuple[int, list[str]]] = []
    start = 0
    while start < len(paths):
        end = min(start + chunk_size, len(paths))
        chunk = paths[start:end]
        if len(chunk) < 2:
            if chunks:
                prev_start, prev = chunks[-1]
                chunks[-1] = (prev_start, prev + chunk)
            break
        chunks.append((start, chunk))
        start = end
    return chunks


def load_frame_tensor(
    path: str,
    process_res: int,
    patch_size: int,
) -> tuple[torch.Tensor, int, int]:
    with Image.open(path) as pil_im:
        rgb = np.asarray(pil_im.convert("RGB"))
    oh, ow = rgb.shape[:2]
    scale = process_res / max(oh, ow)
    nh = max(int(oh * scale) // patch_size * patch_size, patch_size)
    nw = max(int(ow * scale) // patch_size * patch_size, patch_size)
    resized = np.ascontiguousarray(cv2.resize(rgb, (nw, nh)))
    f32 = (resized.astype(np.float32, copy=True) / 127.5) - 1.0
    ten = torch.from_numpy(f32).permute(2, 0, 1).contiguous().clone()
    return ten, nh, nw


def sequence_name_from_path(sequence_dir: str, resolved_root: str) -> str:
    root = Path(resolved_root).resolve()
    if root.is_file():
        return root.stem
    return root.name or Path(sequence_dir).name


def parse_ckpt_iter_tag(load_from: str | None, override: str | None) -> str:
    if override:
        return override if override.startswith("iter_") else f"iter_{override}"
    if not load_from:
        return "iter_000001"
    m = re.search(r"iter[_-]?(\d+)", load_from.replace("\\", "/"), re.I)
    if m:
        return f"iter_{int(m.group(1)):06d}"
    return "iter_000001"


def build_center_dense_columns_query_uv(
    input_width: int,
    input_height: int,
    *,
    dense_stride: int,
    sparse_stride: int,
    num_bands: int = 4,
    dense_band_indices: tuple[int, ...] = (1, 2),
    offset: float = 0.0,
) -> tuple[np.ndarray, int, int]:
    """Vertical 4-band layout: outer bands sparse, inner two bands dense."""
    iw, ih = int(input_width), int(input_height)
    qh = max(ih // dense_stride, 1)
    qw = max(iw // dense_stride, 1)
    parts: list[np.ndarray] = []
    for band in range(num_bands):
        stride = dense_stride if band in dense_band_indices else sparse_stride
        u0 = (band * iw) // num_bands
        u1 = ((band + 1) * iw) // num_bands
        if u1 <= u0:
            continue
        us = np.arange(u0, u1, stride, dtype=np.float32) + offset
        vs = np.arange(0, ih, stride, dtype=np.float32) + offset
        uu, vv = np.meshgrid(us, vs, indexing="xy")
        parts.append(np.stack([uu, vv], axis=-1).reshape(-1, 2))
    if not parts:
        raise RuntimeError("center_dense_columns produced zero queries.")
    uv = np.concatenate(parts, axis=0)
    uv[:, 0] /= max(float(iw - 1), 1.0)
    uv[:, 1] /= max(float(ih - 1), 1.0)
    np.clip(uv, 0.0, 1.0, out=uv)
    return uv, qw, qh


@contextmanager
def center_dense_columns_query_override(
    pipeline,
    input_width: int,
    input_height: int,
    dense_stride: int,
    sparse_stride: int,
    dense_band_indices: tuple[int, ...] = (1, 2),
):
    """Replace QueryBank5 uniform grid with 4-band vertical layout at infer time."""
    inner = getattr(pipeline, "model", pipeline)
    query_bank = getattr(inner, "query_banck", None)
    if query_bank is None:
        yield
        return

    uv_np, qw, qh = build_center_dense_columns_query_uv(
        input_width,
        input_height,
        dense_stride=dense_stride,
        sparse_stride=sparse_stride,
        dense_band_indices=dense_band_indices,
    )
    logging.info(
        "center_dense_columns: %dx%d input, dense_bands=%s dense_stride=%d "
        "sparse_stride=%d → %d queries (ref grid %dx%d)",
        input_width,
        input_height,
        dense_band_indices,
        dense_stride,
        sparse_stride,
        uv_np.shape[0],
        qw,
        qh,
    )

    orig_forward = query_bank.forward

    def _forward(image, edge_mask=None, meta_data=None, **kwargs):
        from hAlgorithm.modules.models2.query_bank.query import BaseQuery

        uv = torch.from_numpy(uv_np).to(device=image.device, dtype=image.dtype)
        return BaseQuery(uv=uv, full_uv=False, width=qw, height=qh)

    query_bank.forward = _forward
    try:
        yield
    finally:
        query_bank.forward = orig_forward


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
        pipeline.dtype = (
            torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        )
    else:
        pipeline.dtype = torch.float32

    ckpt = resolve_load_from(args.load_from, args.motion_config)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    pipeline.load_checkpoint(ckpt_path=ckpt)
    logging.info("Loaded checkpoint: %s", ckpt)

    if args.query_layout == "center_dense_columns":
        stride = int(args.query_dense_stride)
        pipeline.testing_sub_pixel_scale = stride
        logging.info(
            "query_layout=center_dense_columns: dense_bands=%s dense_stride=%d "
            "sparse_stride=%d, testing_sub_pixel_scale=%d.",
            args.query_dense_band_indices,
            args.query_dense_stride,
            args.query_sparse_stride,
            stride,
        )
    else:
        stride = int(args.query_stride)
        pipeline.testing_sub_pixel_scale = stride
        logging.info(
            "testing_sub_pixel_scale=%d (query grid ~ input/%d per side, ≈ 1/%d queries vs dense).",
            stride,
            stride,
            stride * stride,
        )

    soc = pipeline.save_output_cfg
    soc["save_4dgs_results"] = False
    soc["save_gaussians"] = False
    soc["save_render_results"] = False
    soc["save_glb_results"] = False
    soc["save_local_results"] = False
    soc["save_cameras"] = False
    soc["save_match"] = False
    soc.setdefault("save_motion", True)
    soc.setdefault("save_motion_rerun", soc.get("save_motion", True))
    soc.setdefault("motion_vis_ref_frame", 0)
    soc.setdefault("motion_vis_skip_identity_tgt", True)
    return pipeline


def build_infer_batch(
    frame_tensors: list[torch.Tensor],
    sequence_name: str,
    data_idx: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict:
    h, w = frame_tensors[0].shape[1], frame_tensors[0].shape[2]
    for t in frame_tensors[1:]:
        if t.shape[1] != h or t.shape[2] != w:
            raise ValueError("Mixed resolutions in one sequence chunk.")
    n = len(frame_tensors)
    images = torch.stack(frame_tensors, dim=0).unsqueeze(0)
    meta = {
        "name": [sequence_name],
        "data_idx": torch.tensor([data_idx]),
        "frames": torch.tensor([n]),
        "views": torch.tensor([1]),
        "input_height": torch.tensor([h]),
        "input_width": torch.tensor([w]),
        "origin_height": torch.tensor([h]),
        "origin_width": torch.tensor([w]),
    }
    batch = {
        "image": images.to(device),
        "meta_data": meta,
    }
    if use_amp:
        batch["use_amp"] = True
        batch["amp_dtype"] = amp_dtype
    return batch


def run_one_sequence(
    pipeline,
    frame_paths: list[str],
    seq_name: str,
    args,
    device: torch.device,
    ckpt_tag: str,
    amp_dtype: torch.dtype,
    process_res: int,
    patch_size: int,
    source: str | None = None,
) -> None:
    if len(frame_paths) < 2:
        raise RuntimeError(
            f"Need >= 2 frames for {seq_name!r}, found {len(frame_paths)}.",
        )

    vis_out_dir = os.path.join(
        args.output_dir,
        "visualization",
        ckpt_tag,
        seq_name,
    )
    os.makedirs(vis_out_dir, exist_ok=True)

    chunks = iter_frame_chunks(frame_paths, args.infer_chunk_size)
    logging.info(
        "Sequence %r: %d frames from %s → vis %s (%d chunk(s), process_res=%d)",
        seq_name,
        len(frame_paths),
        source or seq_name,
        vis_out_dir,
        len(chunks),
        process_res,
    )

    for chunk_idx, (global_start, chunk_paths) in enumerate(chunks):
        chunk_out = (
            vis_out_dir
            if len(chunks) == 1
            else os.path.join(vis_out_dir, f"chunk_{global_start:06d}")
        )
        os.makedirs(chunk_out, exist_ok=True)
        chunk_name = (
            seq_name
            if len(chunks) == 1
            else f"{seq_name}_f{global_start:04d}"
        )
        logging.info(
            "Chunk %d/%d: frames [%d, %d) (%d frames) → %s",
            chunk_idx + 1,
            len(chunks),
            global_start,
            global_start + len(chunk_paths),
            len(chunk_paths),
            chunk_out,
        )

        frame_tensors: list[torch.Tensor] = []
        for p in chunk_paths:
            ten, nh, nw = load_frame_tensor(p, process_res, patch_size)
            frame_tensors.append(ten)

        batch = build_infer_batch(
            frame_tensors,
            chunk_name,
            data_idx=chunk_idx,
            device=device,
            use_amp=args.use_amp,
            amp_dtype=amp_dtype,
        )

        query_ctx = nullcontext()
        if args.query_layout == "center_dense_columns":
            h, w = frame_tensors[0].shape[1], frame_tensors[0].shape[2]
            query_ctx = center_dense_columns_query_override(
                pipeline,
                w,
                h,
                dense_stride=args.query_dense_stride,
                sparse_stride=args.query_sparse_stride,
                dense_band_indices=args.query_dense_band_indices,
            )

        with query_ctx:
            with torch.inference_mode():
                outputs = pipeline.infer(**batch)
                pipeline.visualize(
                    outputs,
                    meta_data=batch["meta_data"],
                    out_dir=chunk_out,
                )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info("Done sequence %r → %s", seq_name, vis_out_dir)


def run_one_image_sequence(
    pipeline,
    sequence_dir: str,
    args,
    device: torch.device,
    ckpt_tag: str,
    amp_dtype: torch.dtype,
    process_res: int,
    patch_size: int,
) -> None:
    resolved, paths = resolve_image_dir(
        sequence_dir,
        flat_only=args.flat_images_only,
        auto=not args.no_auto_image_dir,
    )
    paths = sample_frame_paths(paths, args.max_frames, args.frame_sampling)
    if len(paths) < 2:
        raise RuntimeError(
            f"Need >= 2 frames under {resolved}, found {len(paths)} after sampling.",
        )
    seq_name = sequence_name_from_path(sequence_dir, resolved)
    run_one_sequence(
        pipeline,
        paths,
        seq_name,
        args,
        device,
        ckpt_tag,
        amp_dtype,
        process_res,
        patch_size,
        source=resolved,
    )


def run_one_video(
    pipeline,
    video_path: str,
    args,
    device: torch.device,
    ckpt_tag: str,
    amp_dtype: torch.dtype,
    process_res: int,
    patch_size: int,
) -> None:
    video_path = resolve_video_path(video_path)
    seq_name = Path(video_path).stem

    cache_dir = None
    cleanup_tmp = None
    if args.keep_frames:
        cache_dir = os.path.join(args.output_dir, "frames", seq_name)
        os.makedirs(cache_dir, exist_ok=True)

    frame_paths, cleanup_tmp = extract_video_frame_paths(
        video_path,
        max_frames=args.max_frames,
        sampling=args.frame_sampling,
        cache_dir=cache_dir,
    )

    try:
        run_one_sequence(
            pipeline,
            frame_paths,
            seq_name,
            args,
            device,
            ckpt_tag,
            amp_dtype,
            process_res,
            patch_size,
            source=video_path,
        )
    finally:
        if cleanup_tmp and os.path.isdir(cleanup_tmp):
            shutil.rmtree(cleanup_tmp, ignore_errors=True)


def main():
    args, unknown = parse_args()
    cfg = file2dict(args.motion_config)
    config_merge_args(cfg, unknown)

    process_res = args.process_res
    if process_res is None:
        process_res = int(cfg.get("max_size", 504))
    patch_size = args.patch_size
    if patch_size is None:
        patch_size = int(cfg.get("patch_size", 14))

    if args.native_resolution:
        probe_path: str | None = None
        if args.video:
            probe_path = args.video
        elif args.videos:
            probe_path = args.videos[0]
        elif args.sequence_dir:
            probe_path = args.sequence_dir
        elif args.sequence_dirs:
            probe_path = args.sequence_dirs[0]
        if probe_path and os.path.isfile(probe_path):
            process_res = native_process_res_from_path(probe_path)
        elif probe_path and os.path.isdir(probe_path):
            _, probe_paths = resolve_image_dir(
                probe_path,
                flat_only=args.flat_images_only,
                auto=not args.no_auto_image_dir,
            )
            if probe_paths:
                process_res = native_process_res_from_path(probe_paths[0])
        logging.info("native_resolution: process_res=%d", process_res)

    device = torch.device("cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    ckpt_tag = parse_ckpt_iter_tag(
        resolve_load_from(args.load_from, args.motion_config),
        args.ckpt_iter_tag,
    )

    pipeline = load_pipeline(args, cfg)

    video_list: list[str] = []
    if args.videos:
        video_list.extend(args.videos)
    if args.video:
        video_list.append(args.video)

    sequence_list: list[str] = []
    if args.sequence_dirs:
        sequence_list.extend(args.sequence_dirs)
    if args.sequence_dir:
        sequence_list.append(args.sequence_dir)

    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "video_list": video_list,
                "sequence_list": sequence_list,
                "ckpt_tag": ckpt_tag,
            },
            f,
            indent=2,
        )

    failures: list[tuple[str, str]] = []
    jobs: list[tuple[str, str]] = [("video", v) for v in video_list]
    jobs.extend(("sequence", s) for s in sequence_list)

    for kind, path in tqdm(jobs, desc="inputs"):
        try:
            if kind == "video":
                run_one_video(
                    pipeline,
                    path,
                    args,
                    device,
                    ckpt_tag,
                    amp_dtype,
                    process_res,
                    patch_size,
                )
            else:
                run_one_image_sequence(
                    pipeline,
                    path,
                    args,
                    device,
                    ckpt_tag,
                    amp_dtype,
                    process_res,
                    patch_size,
                )
        except Exception as ex:
            logging.exception("Failed %s %s: %s", kind, path, ex)
            failures.append((path, str(ex)))

    if failures:
        logging.error("Failed %d / %d inputs.", len(failures), len(jobs))
        for inp, msg in failures:
            logging.error("  %s: %s", inp, msg)
        sys.exit(1)

    logging.info("All inputs done. Root: %s", args.output_dir)


if __name__ == "__main__":
    main()
