#!/usr/bin/env python3
"""
YOLO person-mask dynamic 3D visualization via WFM pair warp3d.

For each video or RGB image sequence (no GT mask):
  1. Sample up to ``max_frames`` RGB frames (default 50, debug_trajectory).
  2. Run YOLO on frame 0 → union person mask (seg masks or bbox fill).
  3. Pair (0, 0): dense grid queries → static reference point cloud (warp3d).
  4. Pairs (0, t): frame-0 person-mask query UVs → dynamic tracks.
  5. Export a single Rerun ``.rrd`` (same layout as DAVIS dynamic vis).

Usage:
    python hAlgorithm/script/infer/motion_head/wfm_rgb_yolo_person_dynamic_vis_infer.py \\
        --motion_config results/.../backup.py \\
        --load_from results/.../checkpoint/latest/ckpt.pth \\
        --video /path/to/clip.mp4
"""

from __future__ import annotations

import argparse
import datetime
import glob
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

sys.path.append(os.getcwd())

import cv2
import numpy as np
import torch
from tqdm import tqdm

from hAlgorithm.script.infer.motion_head.wfm_rgb_sequence_vis_infer import (
    extract_video_frame_paths,
    resolve_video_path,
    sequence_name_from_path,
)
from hAlgorithm.utils import config_merge_args, file2dict, instantiate_from_config, parse_unknown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[3]

DEFAULT_RUN_DIR = (
    _REPO_ROOT
    / "results/baseline_v4/wfm_rgb_query_260526_bs1_4f_50k_hyp0.1_dl3dv0.1_waymo_kitti2_stereo4d_bigdata_v1_finetune_psdutio_20260609-115200"
)

YOLO_PERSON_CLASS_ID = 0

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


def parse_args():
    p = argparse.ArgumentParser(description="YOLO person-mask warp3d dynamic 3D Rerun vis")
    p.add_argument("--motion_config", type=str, required=True)
    p.add_argument("--load_from", type=str, required=True)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--videos", nargs="+", default=None)
    p.add_argument("--sequence_dir", type=str, default=None, help="RGB image folder (no mask).")
    p.add_argument("--sequence_dirs", nargs="+", default=None)
    p.add_argument("--output_dir", type=str, default="./results/wfm_rgb_yolo_person_dynamic_vis")
    p.add_argument("--max_frames", type=int, default=50)
    p.add_argument(
        "--frame_sampling",
        type=str,
        default="debug_trajectory",
        choices=["debug_trajectory", "sequential"],
    )
    p.add_argument("--static_downsample", type=int, default=4)
    p.add_argument("--dynamic_downsample", type=int, default=1)
    p.add_argument(
        "--dynamic_subpixel_factor",
        type=int,
        default=1,
        help="Per mask pixel, sample subpixel_factor^2 UVs (2 => 4x dynamic queries).",
    )
    p.add_argument("--mask_threshold", type=int, default=127)
    p.add_argument("--save_mask_debug", action="store_true", default=True)
    p.add_argument("--max_dynamic_queries", type=int, default=0, help="0 = no cap.")
    p.add_argument("--static_vis_step", type=int, default=1)
    p.add_argument("--max_static_vis_points", type=int, default=120000)
    p.add_argument("--max_traj_vis_points", type=int, default=2048)
    p.add_argument("--traj_vis_downsample", type=int, default=1)
    p.add_argument("--min_arrow_length", type=float, default=0.01)
    p.add_argument("--export_rrd_only", action="store_true")
    p.add_argument("--process_res", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--dense_query_batch_size", type=int, default=65536)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--no_time", action="store_true")
    p.add_argument(
        "--yolo_weights",
        type=str,
        default="yolov8n-seg.pt",
        help="Ultralytics YOLO weights (seg preferred for person mask).",
    )
    p.add_argument("--yolo_conf", type=float, default=0.25)
    p.add_argument("--yolo_iou", type=float, default=0.45)
    p.add_argument("--yolo_imgsz", type=int, default=None, help="YOLO inference size (default: auto).")
    p.add_argument(
        "--yolo_device",
        type=str,
        default=None,
        help="YOLO device (default: cuda if available else cpu).",
    )
    p.add_argument(
        "--video_start_time",
        type=str,
        default=None,
        help="Start offset for video clip splitting (e.g. 2:14, 134, 134.5 seconds).",
    )
    p.add_argument(
        "--video_clip_frames",
        type=int,
        default=None,
        help="Split each video into fixed-length frame clips from --video_start_time.",
    )
    p.add_argument(
        "--skip_no_person",
        action="store_true",
        default=True,
        help="Skip clips where YOLO finds no person on frame 0 (default: on).",
    )
    p.add_argument(
        "--no_skip_no_person",
        action="store_true",
        help="Fail when YOLO finds no person on frame 0.",
    )
    p.add_argument(
        "--mask_source",
        type=str,
        default="yolo",
        choices=["yolo", "manual", "auto"],
        help="yolo=always YOLO; manual=require manual_ref_mask.png; auto=prefer manual if present.",
    )
    p.add_argument(
        "--rerun_clip_dirs",
        nargs="+",
        default=None,
        help="Re-run inference from existing clip output dirs (uses cached _frames/).",
    )
    args, unknown = p.parse_known_args()
    unknown = parse_unknown(unknown)

    if args.no_amp:
        args.use_amp = False
    if args.static_downsample < 1:
        p.error("--static_downsample must be >= 1.")
    if args.dynamic_downsample < 1:
        p.error("--dynamic_downsample must be >= 1.")
    if args.dynamic_subpixel_factor < 1:
        p.error("--dynamic_subpixel_factor must be >= 1.")
    if args.video_clip_frames is not None:
        if args.video_clip_frames < 2:
            p.error("--video_clip_frames must be >= 2.")
        if not args.video_start_time:
            args.video_start_time = "0"

    if args.no_skip_no_person:
        args.skip_no_person = False

    if args.no_time:
        tag = Path(args.motion_config).stem
        args.output_dir = os.path.join(args.output_dir, tag)
    else:
        tag = Path(args.motion_config).stem + datetime.datetime.now().strftime("_%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(args.output_dir, tag)
    os.makedirs(args.output_dir, exist_ok=True)
    return args, unknown


def load_yolo_model(weights: str, device: str):
    from ultralytics import YOLO

    model = YOLO(weights)
    model.to(device)
    logging.info("Loaded YOLO weights: %s (device=%s)", weights, device)
    return model


def detect_person_mask(
    rgb: np.ndarray,
    model,
    *,
    conf: float,
    iou: float,
    imgsz: int | None,
) -> tuple[np.ndarray, dict]:
    """Return grayscale mask (uint8, 255=fg) with same HxW as ``rgb`` (RGB)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    kwargs = dict(
        conf=conf,
        iou=iou,
        classes=[YOLO_PERSON_CLASS_ID],
        verbose=False,
    )
    if imgsz is not None:
        kwargs["imgsz"] = int(imgsz)

    results = model.predict(bgr, **kwargs)
    if not results:
        raise RuntimeError("YOLO returned no results on reference frame.")

    r = results[0]
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    num_boxes = 0

    if r.masks is not None and len(r.masks) > 0:
        for m in r.masks.data:
            m_np = m.detach().cpu().numpy()
            if m_np.shape[:2] != (h, w):
                m_np = cv2.resize(m_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, (m_np > 0.5).astype(np.uint8) * 255)
        num_boxes = len(r.masks)
    elif r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            if int(box.cls.item()) != YOLO_PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
                num_boxes += 1

    if num_boxes == 0:
        raise RuntimeError("YOLO found no person instances on reference frame.")
    if not (mask > 127).any():
        raise RuntimeError("Empty person mask after YOLO on reference frame.")

    meta = dict(num_detections=int(num_boxes), mask_source="yolo_person")
    return mask, meta


def mask_array_to_query_uv(
    mask: np.ndarray,
    *,
    downsample: int,
    max_queries: int,
    mask_threshold: int,
    subpixel_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    oh, ow = mask.shape[:2]
    fg = mask > int(mask_threshold)
    ys, xs = np.where(fg)
    if ys.size == 0:
        raise RuntimeError("Empty person mask on reference frame.")

    order = np.lexsort((xs, ys))
    ys, xs = ys[order], xs[order]
    if downsample > 1:
        ys, xs = ys[::downsample], xs[::downsample]

    sf = max(int(subpixel_factor), 1)
    if sf > 1:
        uv_parts: list[np.ndarray] = []
        xs_f = xs.astype(np.float32)
        ys_f = ys.astype(np.float32)
        for dy in range(sf):
            for dx in range(sf):
                u = (xs_f + (dx + 0.5) / sf) / max(ow - 1, 1)
                v = (ys_f + (dy + 0.5) / sf) / max(oh - 1, 1)
                uv_parts.append(np.stack([u, v], axis=1))
        uv = np.concatenate(uv_parts, axis=0)
    else:
        u = xs.astype(np.float32) / max(ow - 1, 1)
        v = ys.astype(np.float32) / max(oh - 1, 1)
        uv = np.stack([u, v], axis=1)

    if max_queries > 0 and uv.shape[0] > max_queries:
        step = max(int(np.ceil(uv.shape[0] / max_queries)), 1)
        uv = uv[::step][:max_queries]

    return uv.astype(np.float32), fg


MANUAL_MASK_SUFFIX = "_manual_ref_mask.png"
YOLO_MASK_SUFFIX = "_yolo_ref_mask.png"


def manual_mask_path(out_root: str, seq_name: str) -> str:
    return os.path.join(out_root, "mask_debug", f"{seq_name}{MANUAL_MASK_SUFFIX}")


def yolo_mask_path(out_root: str, seq_name: str) -> str:
    return os.path.join(out_root, "mask_debug", f"{seq_name}{YOLO_MASK_SUFFIX}")


def resolve_ref_mask(
    ref_rgb_orig: np.ndarray,
    out_root: str,
    seq_name: str,
    args,
    yolo_model,
) -> tuple[np.ndarray, dict]:
    oh, ow = ref_rgb_orig.shape[:2]
    manual_path = manual_mask_path(out_root, seq_name)
    use_manual = args.mask_source == "manual" or (
        args.mask_source == "auto" and os.path.isfile(manual_path)
    )

    if use_manual:
        if not os.path.isfile(manual_path):
            raise FileNotFoundError(
                f"Manual mask required but not found: {manual_path}. "
                "Use manual_person_mask_editor.py first.",
            )
        mask = _davis.load_mask_gray(manual_path)
        if mask.shape[:2] != (oh, ow):
            mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        if not (mask > args.mask_threshold).any():
            raise RuntimeError(f"Empty manual mask: {manual_path}")
        return mask, dict(mask_source="manual", manual_path=manual_path, num_detections=0)

    if args.mask_source == "manual":
        raise FileNotFoundError(f"Manual mask not found: {manual_path}")

    mask, meta = detect_person_mask(
        ref_rgb_orig,
        yolo_model,
        conf=args.yolo_conf,
        iou=args.yolo_iou,
        imgsz=args.yolo_imgsz,
    )
    meta = dict(meta)
    meta["mask_source"] = "yolo"
    return mask, meta


def save_mask_debug_overlays(
    out_dir: str,
    seq_name: str,
    rgb_path: str,
    mask: np.ndarray,
    model_rgb: np.ndarray,
    mask_threshold: int,
    *,
    tag: str,
) -> str:
    dbg_dir = os.path.join(out_dir, "mask_debug")
    os.makedirs(dbg_dir, exist_ok=True)

    # Never overwrite user-edited manual_ref_mask.png during infer debug export.
    if tag == "manual":
        mask_path = manual_mask_path(out_dir, seq_name)
        vis_path = os.path.join(dbg_dir, f"{seq_name}_manual_ref_mask_vis.png")
        fg = mask > int(mask_threshold)
        vis = np.zeros(mask.shape[:2], dtype=np.uint8)
        vis[fg] = 255
        cv2.imwrite(vis_path, vis)
        orig_overlay_name = f"{seq_name}_manual_infer_overlay_orig.jpg"
        model_overlay_name = f"{seq_name}_manual_infer_overlay_modelres.jpg"
    else:
        mask_path = os.path.join(dbg_dir, f"{seq_name}_{tag}_ref_mask.png")
        cv2.imwrite(mask_path, mask)
        orig_overlay_name = f"{seq_name}_mask_overlay_orig.jpg"
        model_overlay_name = f"{seq_name}_mask_overlay_modelres.jpg"

    rgb_bgr = cv2.imread(rgb_path)
    if rgb_bgr is None:
        raise RuntimeError(f"Failed to read RGB: {rgb_path}")
    oh, ow = rgb_bgr.shape[:2]
    if mask.shape[:2] != (oh, ow):
        raise ValueError(f"Mask/RGB size mismatch: mask={mask.shape[:2]} rgb={(oh, ow)}")

    fg = mask > int(mask_threshold)
    overlay_orig = rgb_bgr.copy()
    overlay_orig[fg] = (
        0.45 * overlay_orig[fg].astype(np.float32) + 0.55 * np.array([0, 0, 255], np.float32)
    ).astype(np.uint8)
    cv2.imwrite(os.path.join(dbg_dir, orig_overlay_name), overlay_orig)

    nh, nw = model_rgb.shape[:2]
    mask_model = _davis.resize_mask_nearest(mask, nh, nw)
    model_bgr = cv2.cvtColor(model_rgb, cv2.COLOR_RGB2BGR)
    overlay_model = model_bgr.copy()
    fg_model = mask_model > int(mask_threshold)
    overlay_model[fg_model] = (
        0.45 * overlay_model[fg_model].astype(np.float32) + 0.55 * np.array([0, 0, 255], np.float32)
    ).astype(np.uint8)
    cv2.imwrite(os.path.join(dbg_dir, model_overlay_name), overlay_model)
    logging.info("Saved %s mask debug overlays → %s", tag, dbg_dir)
    return mask_path


def save_yolo_mask_debug(
    out_dir: str,
    seq_name: str,
    rgb_path: str,
    mask: np.ndarray,
    model_rgb: np.ndarray,
    mask_threshold: int,
) -> str:
    return save_mask_debug_overlays(
        out_dir, seq_name, rgb_path, mask, model_rgb, mask_threshold, tag="yolo",
    )


def parse_timestamp_to_seconds(text: str) -> float:
    raw = str(text).strip()
    if not raw:
        raise ValueError("Empty timestamp.")
    if ":" in raw:
        parts = [float(p) for p in raw.split(":")]
        if len(parts) == 2:
            return parts[0] * 60.0 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
        raise ValueError(f"Invalid timestamp: {text}")
    return float(raw)


def _ffprobe_fps(video_path: str) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            video_path,
        ],
        text=True,
    ).strip()
    if "/" in out:
        num, den = out.split("/", 1)
        return float(num) / float(den)
    return float(out)


def _ffprobe_frame_count(video_path: str) -> int:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "csv=p=0",
            video_path,
        ],
        text=True,
    ).strip()
    if not out or out == "N/A":
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                video_path,
            ],
            text=True,
        ).strip()
    total = int(out)
    if total < 2:
        raise RuntimeError(f"Video has < 2 frames (ffprobe): {video_path}")
    return total


def extract_video_frame_paths_ffmpeg(
    video_path: str,
    max_frames: int,
    sampling: str,
    cache_dir: str | None,
) -> tuple[list[str], str | None]:
    """Extract sampled frames with ffmpeg (fallback when cv2 cannot decode, e.g. AV1)."""
    total = _ffprobe_frame_count(video_path)
    indices = _seq.sample_frame_indices(total, max_frames, sampling)
    tmp_obj = tempfile.mkdtemp(prefix="wfm_vid_frames_ffmpeg_") if cache_dir is None else cache_dir
    os.makedirs(tmp_obj, exist_ok=True)

    select_expr = "+".join(f"eq(n\\,{idx})" for idx in indices)
    out_pattern = os.path.join(tmp_obj, "frame_%06d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"select='{select_expr}'",
        "-vsync",
        "vfr",
        out_pattern,
    ]
    subprocess.run(cmd, check=True)

    paths = sorted(glob.glob(os.path.join(tmp_obj, "frame_*.png")))
    if len(paths) < 2:
        raise RuntimeError(
            f"ffmpeg extracted {len(paths)} frames from {video_path} (need >= 2).",
        )
    cleanup = None if cache_dir else tmp_obj
    return paths, cleanup


def plan_video_clip_ranges(
    video_path: str,
    start_time: str,
    clip_frames: int,
) -> list[dict]:
    total = _ffprobe_frame_count(video_path)
    fps = _ffprobe_fps(video_path)
    start_sec = parse_timestamp_to_seconds(start_time)
    start_frame = int(round(start_sec * fps))
    if start_frame >= total:
        raise RuntimeError(
            f"video_start_time={start_time} ({start_frame}) >= total frames {total} for {video_path}",
        )

    clips: list[dict] = []
    cur = start_frame
    clip_idx = 0
    while cur < total:
        n = min(int(clip_frames), total - cur)
        if n < 2:
            logging.warning(
                "Skip tail clip %d at frame %d (%d frame); need >= 2.",
                clip_idx,
                cur,
                n,
            )
            break
        clips.append(
            dict(
                clip_idx=clip_idx,
                start_frame=cur,
                num_frames=n,
                end_frame_exclusive=cur + n,
            ),
        )
        cur += int(clip_frames)
        clip_idx += 1

    if not clips:
        raise RuntimeError(f"No valid clips planned for {video_path} from {start_time}.")
    logging.info(
        "Planned %d clips for %s from %s (frame %d, fps=%.3f, total=%d)",
        len(clips),
        video_path,
        start_time,
        start_frame,
        fps,
        total,
    )
    return clips


def extract_video_clip_frames_ffmpeg(
    video_path: str,
    start_frame: int,
    num_frames: int,
    cache_dir: str | None,
) -> tuple[list[str], str | None]:
    if num_frames < 2:
        raise RuntimeError(f"Clip needs >= 2 frames, got {num_frames}.")
    tmp_obj = tempfile.mkdtemp(prefix="wfm_vid_clip_frames_") if cache_dir is None else cache_dir
    os.makedirs(tmp_obj, exist_ok=True)

    indices = list(range(start_frame, start_frame + num_frames))
    select_expr = "+".join(f"eq(n\\,{idx})" for idx in indices)
    out_pattern = os.path.join(tmp_obj, "frame_%06d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"select='{select_expr}'",
        "-vsync",
        "vfr",
        out_pattern,
    ]
    subprocess.run(cmd, check=True)

    paths = sorted(glob.glob(os.path.join(tmp_obj, "frame_*.png")))
    if len(paths) < 2:
        raise RuntimeError(
            f"ffmpeg extracted {len(paths)} clip frames from {video_path} "
            f"(start={start_frame}, n={num_frames}).",
        )
    if len(paths) != num_frames:
        logging.warning(
            "Clip frame count mismatch for %s: expected %d, got %d (start=%d).",
            video_path,
            num_frames,
            len(paths),
            start_frame,
        )
    cleanup = None if cache_dir else tmp_obj
    return paths, cleanup


def extract_video_frames_robust(
    video_path: str,
    max_frames: int,
    sampling: str,
    cache_dir: str | None,
) -> tuple[list[str], str | None]:
    try:
        paths, cleanup = extract_video_frame_paths(
            video_path,
            max_frames,
            sampling,
            cache_dir=cache_dir,
        )
        if len(paths) >= 2:
            return paths, cleanup
    except RuntimeError as exc:
        logging.warning("cv2 frame extraction failed for %s: %s", video_path, exc)
    logging.info("Using ffmpeg fallback for %s", video_path)
    return extract_video_frame_paths_ffmpeg(video_path, max_frames, sampling, cache_dir)


def clip_sequence_name(video_stem: str, clip_idx: int, start_frame: int, num_frames: int) -> str:
    return f"{video_stem}_f{start_frame:06d}_n{num_frames:03d}_clip{clip_idx:04d}"


def resolve_rgb_paths_for_source(
    source_path: str,
    max_frames: int,
    frame_sampling: str,
    cache_parent: str,
    *,
    clip_start_frame: int | None = None,
    clip_num_frames: int | None = None,
    seq_name: str | None = None,
) -> tuple[list[str], str, str | None]:
    """Return (rgb_paths, seq_name, temp_dir_to_cleanup)."""
    p = os.path.abspath(os.path.expanduser(source_path))
    if os.path.isfile(p):
        video_stem = sequence_name_from_path(p, p)
        seq_name = seq_name or video_stem
        cache_dir = os.path.join(cache_parent, seq_name, "_frames")
        os.makedirs(cache_dir, exist_ok=True)
        if clip_start_frame is not None and clip_num_frames is not None:
            rgb_paths, cleanup = extract_video_clip_frames_ffmpeg(
                resolve_video_path(p),
                clip_start_frame,
                clip_num_frames,
                cache_dir=cache_dir,
            )
        else:
            rgb_paths, cleanup = extract_video_frames_robust(
                resolve_video_path(p),
                max_frames,
                frame_sampling,
                cache_dir=cache_dir,
            )
        return rgb_paths, seq_name, cleanup

    if not os.path.isdir(p):
        raise FileNotFoundError(f"Input not found: {p}")
    seq_name = seq_name or sequence_name_from_path(p, p)
    rgb_paths = _davis.list_sorted_frames(p)
    rgb_paths = _seq.sample_frame_paths(rgb_paths, max_frames, frame_sampling)
    return rgb_paths, seq_name, None


def export_rrd_from_saved_yolo(args, seq_name: str, process_res: int, patch_size: int) -> None:
    out_root = os.path.join(args.output_dir, seq_name)
    dyn_npz = os.path.join(out_root, "warp3d_dynamic.npz")
    static_npz = os.path.join(out_root, "warp3d_static.npz")
    if not os.path.isfile(dyn_npz):
        raise FileNotFoundError(f"Missing {dyn_npz}; run inference first.")

    dyn = np.load(dyn_npz)
    tracks = dyn["tracks"]
    uv_dynamic = dyn["uv"]
    frame_ids = list(dyn["frame_ids"])

    meta_path = os.path.join(out_root, "source_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}; cannot rebuild RRD without source paths.")
    with open(meta_path, "r", encoding="utf-8") as f:
        source_meta = json.load(f)
    rgb_paths = source_meta["rgb_paths"]
    id_to_path = {Path(p).stem: p for p in rgb_paths}
    ordered_paths = [id_to_path[str(fid)] for fid in frame_ids]

    frame_rgbs = []
    frame_rgbs_orig = []
    for path in ordered_paths:
        _, nh, nw = _seq.load_frame_tensor(path, process_res, patch_size)
        rgb_orig = _davis.load_frame_rgb_orig(path)
        frame_rgbs_orig.append(rgb_orig)
        rgb = rgb_orig
        if rgb.shape[:2] != (nh, nw):
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        frame_rgbs.append(rgb)

    ref_mask_path = manual_mask_path(out_root, seq_name)
    if not os.path.isfile(ref_mask_path):
        ref_mask_path = yolo_mask_path(out_root, seq_name)
    if not os.path.isfile(ref_mask_path):
        raise FileNotFoundError(f"Missing ref mask under {out_root}/mask_debug/")
    ref_mask_model = _davis.resize_mask_nearest(
        _davis.load_mask_gray(ref_mask_path),
        frame_rgbs[0].shape[0],
        frame_rgbs[0].shape[1],
    )
    dynamic_colors = _davis.sample_rgb_at_uv(frame_rgbs[0], uv_dynamic)

    static = np.load(static_npz)
    static_warp3d = static["warp3d"]
    uv_static = static["uv"]
    static_colors = _davis.sample_rgb_at_uv(frame_rgbs[0], uv_static)

    _davis.export_davis_rerun(
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


def resolve_rgb_paths_for_rerun(out_root: str, seq_name: str) -> tuple[list[str], str, None]:
    meta_path = os.path.join(out_root, "source_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}; cannot rerun from cached frames.")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    rgb_paths = meta.get("rgb_paths") or []
    if not rgb_paths or not all(os.path.isfile(p) for p in rgb_paths):
        raise FileNotFoundError(
            f"Cached frame paths missing under {out_root}/_frames; re-extract from video first.",
        )
    return rgb_paths, seq_name, None


def run_source(
    pipeline,
    yolo_model,
    args,
    source_path: str,
    device: torch.device,
    process_res: int,
    patch_size: int,
    *,
    seq_name: str | None = None,
    clip_start_frame: int | None = None,
    clip_num_frames: int | None = None,
    clip_idx: int | None = None,
    rerun_out_root: str | None = None,
) -> None:
    out_parent = args.output_dir
    if rerun_out_root is not None:
        out_root = os.path.abspath(rerun_out_root)
        seq_name = seq_name or Path(out_root).name
        rgb_paths, seq_name, cleanup = resolve_rgb_paths_for_rerun(out_root, seq_name)
    else:
        rgb_paths, seq_name, cleanup = resolve_rgb_paths_for_source(
            source_path,
            args.max_frames,
            args.frame_sampling,
            out_parent,
            clip_start_frame=clip_start_frame,
            clip_num_frames=clip_num_frames,
            seq_name=seq_name,
        )
        out_root = os.path.join(args.output_dir, seq_name)
    n_frames = len(rgb_paths)
    logging.info("Source %r (%s): %d sampled frames", seq_name, source_path, n_frames)

    frame_tensors = []
    frame_rgbs = []
    frame_rgbs_orig = []
    for path in rgb_paths:
        ten, nh, nw = _seq.load_frame_tensor(path, process_res, patch_size)
        frame_tensors.append(ten)
        rgb_orig = _davis.load_frame_rgb_orig(path)
        frame_rgbs_orig.append(rgb_orig)
        rgb = rgb_orig
        if rgb.shape[:2] != (nh, nw):
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        frame_rgbs.append(rgb)

    h, w = frame_tensors[0].shape[1:]
    ref_rgb_path = rgb_paths[0]
    ref_rgb_orig = frame_rgbs_orig[0]
    os.makedirs(out_root, exist_ok=True)
    ref_mask, mask_meta = resolve_ref_mask(
        ref_rgb_orig,
        out_root,
        seq_name,
        args,
        yolo_model,
    )
    orig_h, orig_w = ref_rgb_orig.shape[:2]
    uv_dynamic, ref_fg_orig = mask_array_to_query_uv(
        ref_mask,
        downsample=args.dynamic_downsample,
        max_queries=args.max_dynamic_queries,
        mask_threshold=args.mask_threshold,
        subpixel_factor=args.dynamic_subpixel_factor,
    )
    ref_mask_model = _davis.resize_mask_nearest(ref_mask, h, w)
    uv_static, _, _ = _pair.sample_dense_grid(h, w, args.static_downsample)

    if args.save_mask_debug:
        tag = str(mask_meta.get("mask_source", "yolo"))
        save_mask_debug_overlays(
            out_root,
            seq_name,
            ref_rgb_path,
            ref_mask,
            frame_rgbs[0],
            args.mask_threshold,
            tag=tag,
        )

    logging.info(
        "Queries: static=%d (downsample=%d), dynamic=%d (%s fg orig=%d, "
        "model_res=%d, dynamic_downsample=%d, dynamic_subpixel_factor=%d, yolo_dets=%d)",
        len(uv_static),
        args.static_downsample,
        len(uv_dynamic),
        mask_meta.get("mask_source", "yolo"),
        int(ref_fg_orig.sum()),
        int((ref_mask_model > args.mask_threshold).sum()),
        args.dynamic_downsample,
        args.dynamic_subpixel_factor,
        mask_meta.get("num_detections", 0),
    )
    logging.info(
        "Ref frame 0: rgb=%s orig_size=%dx%d model_size=%dx%d",
        ref_rgb_path,
        orig_w,
        orig_h,
        w,
        h,
    )

    frame_data_list = [
        dict(image=ten, image_rgb=rgb, frame_id=Path(path).stem, rgb_path=path)
        for ten, rgb, path in zip(frame_tensors, frame_rgbs, rgb_paths)
    ]
    batch = _pair.build_window_batch(frame_data_list, device, 1.0)

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

        static_warp3d = _davis.decode_dense_pair_warp3d(
            sdk, batch, patch_tokens, time_token,
            (0, 0), uv_static, device,
            args.dense_query_batch_size, chunk_size,
        )
        static_colors = _davis.sample_rgb_at_uv(frame_rgbs[0], uv_static)

        dynamic_tracks: list[np.ndarray] = []
        dynamic_colors = _davis.sample_rgb_at_uv(frame_rgbs[0], uv_dynamic)

        for t in tqdm(range(n_frames), desc=f"{seq_name} (0,t)"):
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
        frame_ids=[Path(path).stem for path in rgb_paths],
    )

    rrd_path = _davis.export_davis_rerun(
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

    with open(os.path.join(out_root, "source_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            dict(source_path=source_path, rgb_paths=rgb_paths, seq_name=seq_name),
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary = dict(
        sequence=seq_name,
        source_path=source_path,
        clip_idx=clip_idx,
        clip_start_frame=clip_start_frame,
        clip_num_frames=clip_num_frames,
        num_frames=n_frames,
        ref_rgb=ref_rgb_path,
        mask_type=str(mask_meta.get("mask_source", "YOLO_person")),
        yolo_weights=args.yolo_weights,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_detections=int(mask_meta.get("num_detections", 0)),
        manual_mask=mask_meta.get("manual_path"),
        orig_fg_pixels=int(ref_fg_orig.sum()),
        model_res_fg_pixels=int((ref_mask_model > args.mask_threshold).sum()),
        static_queries=int(static_warp3d.shape[0]),
        dynamic_queries=int(uv_dynamic.shape[0]),
        static_downsample=args.static_downsample,
        dynamic_downsample=args.dynamic_downsample,
        dynamic_subpixel_factor=args.dynamic_subpixel_factor,
        mask_threshold=args.mask_threshold,
        rrd=rrd_path,
    )
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info("Done %r → %s", seq_name, out_root)

    if cleanup and os.path.isdir(cleanup):
        shutil.rmtree(cleanup, ignore_errors=True)


def collect_sources(args) -> list[str]:
    if args.rerun_clip_dirs:
        return []
    sources: list[str] = []
    if args.videos:
        sources.extend(args.videos)
    if args.video:
        sources.append(args.video)
    if args.sequence_dirs:
        sources.extend(args.sequence_dirs)
    if args.sequence_dir:
        sources.append(args.sequence_dir)
    if not sources and not args.rerun_clip_dirs:
        raise SystemExit("Provide --video/--videos, --sequence_dir(s), or --rerun_clip_dirs.")
    return sources


def expand_rerun_jobs(args) -> list[dict]:
    jobs: list[dict] = []
    for clip_dir in args.rerun_clip_dirs:
        out_root = os.path.abspath(os.path.expanduser(clip_dir))
        seq_name = Path(out_root).name
        meta_path = os.path.join(out_root, "source_meta.json")
        summary_path = os.path.join(out_root, "summary.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Missing {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        clip_idx = None
        clip_start_frame = None
        clip_num_frames = None
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            clip_idx = summary.get("clip_idx")
            clip_start_frame = summary.get("clip_start_frame")
            clip_num_frames = summary.get("clip_num_frames")
        jobs.append(
            dict(
                source_path=meta.get("source_path", ""),
                seq_name=seq_name,
                clip_idx=clip_idx,
                clip_start_frame=clip_start_frame,
                clip_num_frames=clip_num_frames,
                rerun_out_root=out_root,
            ),
        )
    return jobs


def expand_inference_jobs(args) -> list[dict]:
    if args.rerun_clip_dirs:
        return expand_rerun_jobs(args)

    jobs: list[dict] = []
    for source in collect_sources(args):
        p = os.path.abspath(os.path.expanduser(source))
        if args.video_clip_frames is not None and os.path.isfile(p):
            video_stem = sequence_name_from_path(p, p)
            for clip in plan_video_clip_ranges(p, args.video_start_time, args.video_clip_frames):
                jobs.append(
                    dict(
                        source_path=source,
                        seq_name=clip_sequence_name(
                            video_stem,
                            clip["clip_idx"],
                            clip["start_frame"],
                            clip["num_frames"],
                        ),
                        clip_idx=clip["clip_idx"],
                        clip_start_frame=clip["start_frame"],
                        clip_num_frames=clip["num_frames"],
                    ),
                )
        else:
            jobs.append(
                dict(
                    source_path=source,
                    seq_name=sequence_name_from_path(source, source),
                    clip_idx=None,
                    clip_start_frame=None,
                    clip_num_frames=None,
                ),
            )
    return jobs


def main():
    args, unknown = parse_args()
    cfg = file2dict(args.motion_config)
    config_merge_args(cfg, unknown)

    process_res = args.process_res or int(cfg.get("max_size", 504))
    patch_size = args.patch_size or int(cfg.get("patch_size", 14))
    jobs = expand_inference_jobs(args)

    yolo_device = args.yolo_device
    if yolo_device is None:
        yolo_device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = None if args.export_rrd_only else _davis.load_pipeline(args, cfg)
    need_yolo = (
        not args.export_rrd_only
        and args.mask_source != "manual"
        and not (args.mask_source == "auto" and args.rerun_clip_dirs)
    )
    yolo_model = None
    if need_yolo:
        yolo_model = load_yolo_model(args.yolo_weights, yolo_device)
    elif not args.export_rrd_only:
        jobs_need_yolo = True
        if args.mask_source == "manual":
            jobs_need_yolo = False
        elif args.mask_source == "auto" and args.rerun_clip_dirs:
            jobs_need_yolo = any(
                not os.path.isfile(manual_mask_path(job["rerun_out_root"], job["seq_name"]))
                for job in jobs
            )
        if jobs_need_yolo:
            yolo_model = load_yolo_model(args.yolo_weights, yolo_device)
    device = torch.device("cuda")

    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    failures = []
    skipped = []
    for job in jobs:
        source = job["source_path"]
        seq_name = job["seq_name"]
        try:
            if args.export_rrd_only:
                export_rrd_from_saved_yolo(args, seq_name, process_res, patch_size)
            else:
                run_source(
                    pipeline,
                    yolo_model,
                    args,
                    source,
                    device,
                    process_res,
                    patch_size,
                    seq_name=seq_name,
                    clip_start_frame=job["clip_start_frame"],
                    clip_num_frames=job["clip_num_frames"],
                    clip_idx=job["clip_idx"],
                    rerun_out_root=job.get("rerun_out_root"),
                )
        except RuntimeError as exc:
            if args.skip_no_person and "no person instances" in str(exc).lower():
                logging.warning("Skipped %s: %s", seq_name, exc)
                skipped.append((seq_name, str(exc)))
            else:
                logging.exception("Failed %s (%s): %s", seq_name, source, exc)
                failures.append((seq_name, str(exc)))
        except Exception as exc:
            logging.exception("Failed %s (%s): %s", seq_name, source, exc)
            failures.append((seq_name, str(exc)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if skipped:
        for name, msg in skipped:
            logging.warning("  skipped %s: %s", name, msg)
    if failures:
        for name, msg in failures:
            logging.error("  %s: %s", name, msg)
        sys.exit(1)
    logging.info("All done → %s", args.output_dir)


if __name__ == "__main__":
    main()
