#!/usr/bin/env python3
"""
WFM video Gaussian render inference (vis-only, no eval).

For each input video:
  1. Extract frames (cv2), sample ``max_frames`` (DebugTrajectory or sequential).
  2. Resize/normalize like val (max_size=504, patch 14).
  3. ``WFMQueryPipeline.infer`` with predicted cameras when GT is absent.
  4. Save Gaussian render: per-frame JPG, grid JPG, MP4.
  5. Optional ``web_viewer/`` export (PLY timeline + manifest) for WebGL / 4DGS viewers.

Requires a 4DGS checkpoint (``sparse_gaussian_head`` + ``sparse_dynamic_gaussian_render_loss``).

Usage:
    python hAlgorithm/script/infer/motion_head/wfm_video_gaussian_vis_infer.py \\
        --motion_config results/.../wfm_rgb_query_260526_sparse_pair_4dgs_v1.py \\
        --load_from results/.../checkpoint/latest/ckpt.pth \\
        --video /path/to/clip.mp4 \\
        --output_dir ./results/wfm_video_gaussian_vis

    python ... --videos a.mp4 b.mp4 --max_frames 30 --query_stride 4
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from hAlgorithm.script.infer.motion_head.wfm_rgb_sequence_vis_infer import (
    extract_video_frame_paths,
    iter_frame_chunks,
    native_process_res_from_path,
    resolve_image_dir,
    resolve_video_path,
    sample_frame_paths,
    sequence_name_from_path,
)
from hAlgorithm.script.infer.motion_head.wfm_video_gaussian_web_export import (
    attach_model_results_capture,
    export_web_viewer_bundle,
)
from hAlgorithm.utils import (
    config_merge_args,
    file2dict,
    grid_images,
    instantiate_from_config,
    parse_unknown,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_4DGS_RUN_DIR = (
    "/mnt/home/tcchen/workspace/TMA-origin-dev/results/sparse_pair_4dgs_v2/"
    "wfm_rgb_query_dual_4dgs_perpixel_trainquality_v2_wild_norm_20260622-203044"
)
DEFAULT_4DGS_CONFIG = os.path.join(
    DEFAULT_4DGS_RUN_DIR,
    "wfm_rgb_query_dual_4dgs_perpixel_trainquality_v2_wild_norm_backup.py",
)
DEFAULT_4DGS_CKPT = os.path.join(DEFAULT_4DGS_RUN_DIR, "checkpoint/latest/ckpt.pth")


def parse_args():
    p = argparse.ArgumentParser(description="WFM video Gaussian render vis-only infer")
    p.add_argument(
        "--motion_config",
        type=str,
        default=DEFAULT_4DGS_CONFIG,
        help="4DGS model config (.py).",
    )
    p.add_argument(
        "--load_from",
        type=str,
        default=DEFAULT_4DGS_CKPT,
        help="Checkpoint path (or latest/best).",
    )
    p.add_argument("--video", type=str, default=None, help="Single video file.")
    p.add_argument("--videos", nargs="+", default=None, help="Multiple video files.")
    p.add_argument(
        "--sequence_dir",
        type=str,
        default=None,
        help="Single RGB image sequence directory (e.g. DAVIS JPEGImages/480p/xxx).",
    )
    p.add_argument(
        "--sequence_dirs",
        nargs="+",
        default=None,
        help="Multiple RGB image sequence directories.",
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
    p.add_argument("--output_dir", type=str, default="./results/wfm_video_gaussian_vis")
    p.add_argument(
        "--max_frames",
        type=int,
        default=30,
        help="Frames per video (DebugTrajectory uniform sampling by default). "
             "Ignored when --all_frames is set.",
    )
    p.add_argument(
        "--all_frames",
        action="store_true",
        help="Use every frame (no temporal subsampling). Split long clips with "
             "--infer_chunk_size to avoid OOM.",
    )
    p.add_argument(
        "--infer_chunk_size",
        type=int,
        default=0,
        help="Max frames per infer chunk (0 = single batch). E.g. 15 for long videos.",
    )
    p.add_argument(
        "--frame_sampling",
        type=str,
        default="debug_trajectory",
        choices=["debug_trajectory", "sequential"],
    )
    p.add_argument(
        "--query_stride",
        type=int,
        default=4,
        help="Maps to pipeline.testing_sub_pixel_scale (1=dense H×W queries).",
    )
    p.add_argument("--process_res", type=int, default=None)
    p.add_argument(
        "--native_resolution",
        action="store_true",
        help="Use max(w,h) of the first input frame (no downscale; still aligned to patch_size).",
    )
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument(
        "--render_fps",
        type=float,
        default=24.0,
        help="FPS for output render MP4.",
    )
    p.add_argument(
        "--keep_frames",
        action="store_true",
        help="Keep extracted frames under output_dir/frames/.",
    )
    p.add_argument(
        "--export_web_viewer",
        action="store_true",
        default=True,
        help="Export web_viewer/ (PLY timeline + manifest.json) (default on).",
    )
    p.add_argument(
        "--no_export_web_viewer",
        action="store_true",
        help="Disable web_viewer/ export.",
    )
    p.add_argument(
        "--web_max_ply_points",
        type=int,
        default=8192,
        help="Max Gaussians per PLY frame in web export (0 = all queries).",
    )
    p.add_argument(
        "--max_render_points",
        type=int,
        default=None,
        help="Cap Gaussians at render time (default 8192). 0 = per-pixel (all H×W queries).",
    )
    p.add_argument(
        "--per_pixel",
        action="store_true",
        help="Shorthand: query_stride=1, max_render_points=0 (full-resolution gsplat).",
    )
    p.add_argument(
        "--infer_gs_log_scale_bias",
        type=float,
        default=None,
        help="Gsplat log-scale bias at infer (0=model scale, None=auto shrink for vis).",
    )
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument(
        "--amp_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
    )
    p.add_argument("--ckpt_iter_tag", type=str, default=None)
    p.add_argument("--no_time", action="store_true")
    args, unknown = p.parse_known_args()
    unknown = parse_unknown(unknown)

    if args.no_amp:
        args.use_amp = False
    if args.no_export_web_viewer:
        args.export_web_viewer = False

    if args.per_pixel:
        args.query_stride = 1
        if args.max_render_points is None:
            args.max_render_points = 0
        if args.web_max_ply_points == 8192:
            args.web_max_ply_points = 0
        if args.infer_gs_log_scale_bias is None:
            # Per-pixel dense gsplat: keep model splat scale (no auto shrink → dotty vis).
            args.infer_gs_log_scale_bias = 0.0

    if (
        args.video is None
        and not args.videos
        and args.sequence_dir is None
        and not args.sequence_dirs
    ):
        p.error("Provide --video/--videos or --sequence_dir/--sequence_dirs.")

    if args.query_stride < 1:
        p.error("--query_stride must be >= 1.")
    if args.all_frames:
        args.max_frames = 10**9
    elif args.max_frames < 2:
        p.error("--max_frames must be >= 2 for dynamic 4DGS.")
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


def load_frame_tensor(
    path: str,
    process_res: int,
    patch_size: int,
) -> torch.Tensor:
    with Image.open(path) as pil_im:
        rgb = np.asarray(pil_im.convert("RGB"))
    oh, ow = rgb.shape[:2]
    scale = process_res / max(oh, ow)
    nh = max(int(oh * scale) // patch_size * patch_size, patch_size)
    nw = max(int(ow * scale) // patch_size * patch_size, patch_size)
    resized = np.ascontiguousarray(cv2.resize(rgb, (nw, nh)))
    f32 = (resized.astype(np.float32, copy=True) / 127.5) - 1.0
    return torch.from_numpy(f32).permute(2, 0, 1).contiguous().clone()


def parse_ckpt_iter_tag(load_from: str | None, override: str | None) -> str:
    if override:
        return override if override.startswith("iter_") else f"iter_{override}"
    if not load_from:
        return "iter_000001"
    m = re.search(r"iter[_-]?(\d+)", load_from.replace("\\", "/"), re.I)
    if m:
        return f"iter_{int(m.group(1)):06d}"
    return "iter_000001"


def resolve_load_from(load_from: str, motion_config: str) -> str:
    if load_from == "latest":
        return os.path.join(os.path.dirname(motion_config), "checkpoint/latest/ckpt.pth")
    if load_from == "best":
        return os.path.join(os.path.dirname(motion_config), "checkpoint/best/ckpt.pth")
    return os.path.expanduser(load_from)


def assert_4dgs_pipeline(pipeline) -> None:
    has_dgs_loss = getattr(pipeline, "sparse_dynamic_gaussian_render_loss", None) is not None
    if not has_dgs_loss:
        raise RuntimeError(
            "Config/pipeline has no sparse_dynamic_gaussian_render_loss. "
            "Use a 4DGS checkpoint (e.g. wfm_rgb_query_*_sparse_pair_4dgs_v1)."
        )


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

    assert_4dgs_pipeline(pipeline)

    stride = int(args.query_stride)
    pipeline.testing_sub_pixel_scale = stride
    logging.info(
        "testing_sub_pixel_scale=%d (query grid ~ input/%d per side).",
        stride,
        stride,
    )

    soc = pipeline.save_output_cfg
    soc["save_4dgs_results"] = False
    soc["save_gaussians"] = False
    soc["save_render_results"] = False
    soc["save_glb_results"] = False
    soc["save_local_results"] = False
    soc["save_cameras"] = False
    soc["save_match"] = False
    soc["save_motion"] = False
    soc["save_motion_rerun"] = False

    attach_model_results_capture(pipeline)

    dgs_loss = getattr(pipeline, "sparse_dynamic_gaussian_render_loss", None)
    if dgs_loss is not None:
        if args.max_render_points is not None:
            dgs_loss.max_render_points = int(args.max_render_points)
        normalized_render = bool(
            getattr(pipeline, "dgs_render_normalized_only", False)
            or getattr(dgs_loss, "render_in_normalized_space", False)
        )
        if normalized_render and args.infer_gs_log_scale_bias is None:
            # Per-pixel normalized render: auto pixel-scale bias (no global scale guess).
            args.infer_gs_log_scale_bias = 0.0
        if args.infer_gs_log_scale_bias is not None:
            dgs_loss.infer_vis_gs_log_scale_bias = float(args.infer_gs_log_scale_bias)
        coord_mode = "normalized (no wild scale)" if normalized_render else "metric (wild scale from pair_depth)"
        logging.info(
            "DGS infer render: backend=gsplat, max_render_points=%d, "
            "infer_gs_log_scale_bias=%s, coord=%s",
            dgs_loss.max_render_points,
            dgs_loss.infer_vis_gs_log_scale_bias,
            coord_mode,
        )

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
            raise ValueError("Mixed resolutions in one video chunk.")
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


def save_gaussian_render_outputs(
    outputs_list: list,
    out_dir: str,
    data_idx: int,
    sequence_name: str,
    render_fps: float,
) -> None:
    """Save per-frame JPG, grid JPG, and MP4 from ``dgs_render_rgb``."""
    render_dir = os.path.join(out_dir, "4dgs", f"{data_idx:06d}")
    frames_dir = os.path.join(render_dir, "render_frames")
    os.makedirs(frames_dir, exist_ok=True)

    rgb_list: list[np.ndarray] = []
    for i, out in enumerate(outputs_list):
        ren = getattr(out, "dgs_render_rgb", None)
        if ren is None:
            continue
        ren = np.clip(ren, 0.0, 1.0)
        rgb_u8 = (ren * 255).astype(np.uint8)
        bgr_u8 = rgb_u8[..., ::-1]
        frame_path = os.path.join(frames_dir, f"{i:06d}.jpg")
        cv2.imwrite(frame_path, bgr_u8)
        rgb_list.append(bgr_u8)

    if not rgb_list:
        raise RuntimeError(
            "No dgs_render_rgb in pipeline outputs. "
            "Check 4DGS checkpoint and sparse_gaussian_head."
        )

    grid_path = os.path.join(render_dir, f"render_rgb_grid_{data_idx:06d}.jpg")
    col = min(4, len(rgb_list))
    grid_images(save_path=grid_path, images=rgb_list, col=col)
    logging.info("Render grid → %s", grid_path)

    # Side-by-side input | render for easier visual QA.
    compare_list: list[np.ndarray] = []
    rgb_dir = os.path.join(out_dir, "rgb")
    for i, ren_bgr in enumerate(rgb_list):
        src_path = os.path.join(rgb_dir, f"{i:06d}.jpg")
        if os.path.isfile(src_path):
            src_bgr = cv2.imread(src_path)
            if src_bgr is not None:
                src_bgr = cv2.resize(src_bgr, (ren_bgr.shape[1], ren_bgr.shape[0]))
                compare_list.append(np.concatenate([src_bgr, ren_bgr], axis=1))
    if compare_list:
        compare_path = os.path.join(render_dir, f"compare_rgb_grid_{data_idx:06d}.jpg")
        grid_images(save_path=compare_path, images=compare_list, col=col)
        logging.info("Compare grid (input|render) → %s", compare_path)

    mp4_path = os.path.join(render_dir, f"render_{sequence_name}_{data_idx:06d}.mp4")
    _write_browser_compatible_mp4(rgb_list, mp4_path, render_fps)
    logging.info("Render MP4 (H.264, %sfps) → %s", render_fps, mp4_path)


def _write_browser_compatible_mp4(
    bgr_frames: list[np.ndarray],
    mp4_path: str,
    fps: float,
) -> None:
    """Encode BGR frames as H.264/yuv420p for browser-compatible MP4 playback."""
    if not bgr_frames:
        raise ValueError("Cannot encode an empty frame list.")

    height, width = bgr_frames[0].shape[:2]
    for frame in bgr_frames:
        if frame.shape[:2] != (height, width):
            raise ValueError("All MP4 frames must have the same resolution.")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to encode browser-compatible H.264 MP4 output."
        )

    tmp_path = f"{mp4_path}.tmp.mp4"
    command = [
        ffmpeg,
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s:v", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        tmp_path,
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        for frame in bgr_frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        assert process.stderr is not None
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if return_code != 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"ffmpeg H.264 encoding failed: {stderr.strip()}")
    os.replace(tmp_path, mp4_path)


def stitch_chunk_render_mp4(
    vis_out_dir: str,
    sequence_name: str,
    render_fps: float,
) -> str | None:
    """Concatenate per-chunk render frames into one full-length MP4."""
    chunk_dirs = sorted(
        d for d in glob.glob(os.path.join(vis_out_dir, "chunk_*"))
        if os.path.isdir(d)
    )
    if not chunk_dirs:
        return None

    rgb_list: list[np.ndarray] = []
    for chunk_dir in chunk_dirs:
        frames_dir = os.path.join(chunk_dir, "4dgs", "000000", "render_frames")
        if not os.path.isdir(frames_dir):
            continue
        for fname in sorted(os.listdir(frames_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(frames_dir, fname))
            if img is not None:
                rgb_list.append(img)
    if not rgb_list:
        return None

    full_dir = os.path.join(vis_out_dir, "4dgs", "000000")
    os.makedirs(full_dir, exist_ok=True)
    mp4_path = os.path.join(full_dir, f"render_{sequence_name}_full.mp4")
    _write_browser_compatible_mp4(rgb_list, mp4_path, render_fps)
    logging.info(
        "Stitched full render MP4 (H.264, %d frames, %sfps) → %s",
        len(rgb_list), render_fps, mp4_path,
    )
    return mp4_path


def save_sampled_rgb_frames(frame_paths: list[str], out_dir: str) -> None:
    """Save sampled input frames under ``out_dir/rgb/``."""
    rgb_dir = os.path.join(out_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    for i, src in enumerate(frame_paths):
        img = cv2.imread(src)
        if img is None:
            raise RuntimeError(f"Failed to read sampled frame: {src}")
        cv2.imwrite(os.path.join(rgb_dir, f"{i:06d}.jpg"), img)
    logging.info("Sampled RGB (%d frames) → %s", len(frame_paths), rgb_dir)


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
) -> None:
    if len(frame_paths) < 2:
        raise RuntimeError(f"Need >= 2 frames for {seq_name!r}, found {len(frame_paths)}.")

    vis_out_dir = os.path.join(
        args.output_dir,
        "visualization",
        ckpt_tag,
        seq_name,
    )
    os.makedirs(vis_out_dir, exist_ok=True)

    chunks = iter_frame_chunks(frame_paths, args.infer_chunk_size)
    logging.info(
        "Sequence %r: %d frames → %s (%d chunk(s), process_res=%d)",
        seq_name,
        len(frame_paths),
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

        frame_tensors = [
            load_frame_tensor(p, process_res, patch_size) for p in chunk_paths
        ]
        save_sampled_rgb_frames(chunk_paths, chunk_out)
        batch = build_infer_batch(
            frame_tensors,
            chunk_name,
            data_idx=0,
            device=device,
            use_amp=args.use_amp,
            amp_dtype=amp_dtype,
        )

        pipeline._captured_gs_results = None
        with torch.inference_mode():
            outputs = pipeline.infer(**batch)

        save_gaussian_render_outputs(
            outputs,
            chunk_out,
            data_idx=0,
            sequence_name=chunk_name,
            render_fps=args.render_fps,
        )

        if args.export_web_viewer and pipeline._captured_gs_results:
            export_web_viewer_bundle(
                captured=pipeline._captured_gs_results,
                outputs_list=outputs,
                image=batch["image"],
                out_dir=chunk_out,
                sequence_name=chunk_name,
                fps=args.render_fps,
                max_ply_points=args.web_max_ply_points,
            )

        del outputs, batch, frame_tensors
        pipeline._captured_gs_results = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(chunks) > 1:
        stitch_chunk_render_mp4(vis_out_dir, seq_name, args.render_fps)

    logging.info("Done sequence %r → %s", seq_name, vis_out_dir)


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
        )
    finally:
        if cleanup_tmp and os.path.isdir(cleanup_tmp):
            shutil.rmtree(cleanup_tmp, ignore_errors=True)


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
    )


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
    else:
        logging.info("process_res=%d patch_size=%d", process_res, patch_size)

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
        for vp, msg in failures:
            logging.error("  %s: %s", vp, msg)
        sys.exit(1)

    logging.info("All inputs done. Root: %s", args.output_dir)


if __name__ == "__main__":
    main()
