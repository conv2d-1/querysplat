#!/usr/bin/env python3
"""
WFMQueryPipeline (MVQuery6): dense motion mask inference via query-level ``motion_mask`` head.

Unlike ``mv_query_pair2_motion_heatmap_infer.py`` (warp3d_delta magnitude heatmap), this
script reads the per-query ``motion_mask`` logit from ``query_pair_decoder`` and
visualizes binary dynamic/static masks directly.

Pair logic (``pair_schedule=adjacent``): encode W frames per chunk; for each adjacent
pair run (src→tgt) and (tgt→src) so every frame receives a mask.

Usage:
    python hAlgorithm/script/infer/motion_head/mv_query_motion_mask_infer.py \\
        --config results/.../wfm_rgb_query_..._query_motion_mask_backup.py \\
        --load_from results/.../checkpoint/latest/ckpt.pth \\
        --data /path/to/scene.json --view_id 8 --fisheye
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
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd())

from hAlgorithm.modules.models2.query_bank.query import BaseQuery
from hAlgorithm.utils import config_merge_args, file2dict, parse_unknown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_mv_query_pair2_motion_heatmap_infer",
    _SCRIPT_DIR / "mv_query_pair2_motion_heatmap_infer.py",
)
_hm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hm)


def parse_args():
    parser = argparse.ArgumentParser(
        description="WFMQueryPipeline dense motion mask inference (query motion_mask head)",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--load_from", default="latest")
    parser.add_argument("--sequence_dir", type=str, default=None)
    parser.add_argument("--sequence_dirs", nargs="+", default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results/query_motion_mask")
    parser.add_argument("--nums", type=int, default=None)
    parser.add_argument("--view_id", type=int, default=None)
    parser.add_argument("--fisheye", action="store_true")
    parser.add_argument(
        "--sensor_size", type=float, nargs=2,
        default=_hm.DEFAULT_SENSOR_SIZE_MM, metavar=("W_MM", "H_MM"),
    )
    parser.add_argument("--fisheye_radius", type=float, default=None)
    parser.add_argument("--skip_fisheye_crop", action="store_true")
    parser.add_argument("--process_res", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument(
        "--frame_sampling", type=str, default="sequential",
        choices=["sequential", "debug_trajectory"],
    )
    parser.add_argument("--flat_images_only", action="store_true")
    parser.add_argument("--no_auto_image_dir", action="store_true")
    parser.add_argument("--no_time", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--dense_downsample", type=int, default=4)
    parser.add_argument("--dense_query_batch_size", type=int, default=65536)
    parser.add_argument(
        "--mask_aggregation", type=str, default="max",
        choices=["max", "mean"],
        help="Merge logits when a frame appears in multiple pairs.",
    )
    parser.add_argument(
        "--mask_logit_threshold", type=float, default=0.0,
        help="Dynamic if motion_mask logit >= threshold (training eval uses 0).",
    )
    parser.add_argument(
        "--scale_mode", type=str, default="global", choices=["global", "per_pair"],
    )
    parser.add_argument("--encoder_window", type=int, default=4)
    parser.add_argument(
        "--pair_schedule", type=str, default="adjacent",
        choices=["adjacent", "pair2"],
    )
    parser.add_argument("--save_gif", action="store_true", default=True)
    parser.add_argument("--gif_fps", type=float, default=4.0)
    parser.add_argument("--gif_suffix", type=str, default=None)
    parser.add_argument("--overlay_alpha", type=float, default=0.55)
    parser.add_argument(
        "--dynamic_color", type=int, nargs=3, default=[255, 50, 50],
        metavar=("R", "G", "B"),
    )
    parser.add_argument("--save_prob_heatmap", action="store_true", default=True)

    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown(unknown_args)

    if args.config is None:
        parser.error("--config is required.")
    has_seq = args.sequence_dir is not None or args.sequence_dirs
    if not has_seq and args.data is None:
        parser.error("Provide --sequence_dir, --sequence_dirs, or --data.")
    if args.dense_downsample < 1:
        parser.error("--dense_downsample must be >= 1.")
    if args.encoder_window < 2:
        parser.error("--encoder_window must be >= 2.")
    if args.pair_schedule == "pair2" and args.encoder_window != 2:
        parser.error("pair_schedule=pair2 requires --encoder_window 2.")

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


def aggregate_logits(logit_list: list[np.ndarray], mode: str) -> np.ndarray:
    stacked = np.stack(logit_list, axis=0)
    return stacked.max(axis=0) if mode == "max" else stacked.mean(axis=0)


def decode_dense_pair_motion_mask(
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
        mask_logit = pair_out.get("motion_mask")
        if mask_logit is None:
            raise RuntimeError(
                "pair_forward did not return motion_mask; "
                "use a checkpoint with query_pair_decoder.motion_mask head."
            )
        logit = mask_logit.float().squeeze(-1).squeeze(0).cpu().numpy()
        segments.append(logit.astype(np.float32))

    return np.concatenate(segments, axis=0)


def _assemble_mask_results(
    frame_data_list,
    frame_logit_lists,
    dh,
    dw,
    h,
    w,
    args,
) -> list[dict]:
    results_list = []
    for gi, fd in enumerate(frame_data_list):
        logit_list = frame_logit_lists.get(gi, [])
        if not logit_list:
            results_list.append(dict(
                global_i=gi,
                image_rgb=fd["image_rgb"],
                mask_logit=None,
                mask_prob=None,
                motion_mask=None,
                frame_id=fd.get("frame_id", gi),
            ))
            continue

        logit_small = aggregate_logits(logit_list, args.mask_aggregation).reshape(dh, dw)
        logit_full = cv2.resize(logit_small, (w, h), interpolation=cv2.INTER_LINEAR)

        bmask = fd.get("boundary_mask")
        if bmask is not None:
            if bmask.shape[:2] != (h, w):
                bmask = cv2.resize(
                    bmask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            logit_full = np.where(bmask, logit_full, -1e6)

        prob_full = 1.0 / (1.0 + np.exp(-logit_full))
        motion_mask = logit_full >= args.mask_logit_threshold
        results_list.append(dict(
            global_i=gi,
            image_rgb=fd["image_rgb"],
            mask_logit=logit_full,
            mask_prob=prob_full,
            motion_mask=motion_mask,
            frame_id=fd.get("frame_id", gi),
        ))
    return results_list


def run_adjacent_window_inference(pipeline, frame_data_list, device, args):
    sdk = pipeline.model
    N = len(frame_data_list)
    W = args.encoder_window
    chunk_ranges = _hm.build_encoder_chunk_ranges(N, W)
    logging.info(
        "Adjacent window mask: %d frames, encoder_window=%d, %d chunks: %s",
        N, W, len(chunk_ranges), chunk_ranges,
    )

    h, w = frame_data_list[0]["image"].shape[1:]
    grid_uv, dh, dw = _hm.sample_dense_grid(h, w, args.dense_downsample)
    chunk_size = getattr(sdk, "chunk_size", None)

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_ctx = (
        torch.autocast("cuda", enabled=True, dtype=amp_dtype)
        if args.use_amp else nullcontext()
    )

    frame_logit_lists: dict[int, list[np.ndarray]] = {}
    use_global_scale = args.scale_mode == "global"
    global_depth_scale = (
        _hm.compute_sequence_scale(frame_data_list) if use_global_scale else None
    )
    if use_global_scale:
        logging.info("Global depth_scale=%.4f", global_depth_scale)

    with torch.no_grad():
        for chunk_start, chunk_end in tqdm(
            chunk_ranges, desc=f"win{W} adjacent mask",
        ):
            chunk_fds = frame_data_list[chunk_start:chunk_end]
            depth_scale = (
                global_depth_scale if use_global_scale
                else _hm.compute_sequence_scale(chunk_fds)
            )
            batch = _hm.build_window_batch(chunk_fds, device, depth_scale)
            patch_tokens, time_token = _hm.encode_pair_features(sdk, batch, amp_ctx)

            for i in range(len(chunk_fds) - 1):
                for local_src, local_tgt in ((i, i + 1), (i + 1, i)):
                    logits = decode_dense_pair_motion_mask(
                        sdk, batch, patch_tokens, time_token,
                        (local_src, local_tgt), grid_uv, device,
                        args.dense_query_batch_size, chunk_size,
                    )
                    global_idx = chunk_start + local_src
                    frame_logit_lists.setdefault(global_idx, []).append(logits)

    results_list = _assemble_mask_results(
        frame_data_list, frame_logit_lists, dh, dw, h, w, args,
    )
    infer_meta = dict(
        global_depth_scale=global_depth_scale,
        scale_mode=args.scale_mode,
        encoder_window=W,
        pair_schedule=args.pair_schedule,
        encoder_chunk_ranges=chunk_ranges,
        mask_logit_threshold=args.mask_logit_threshold,
    )
    return results_list, infer_meta


def run_pair2_inference(pipeline, frame_data_list, device, args):
    sdk = pipeline.model
    N = len(frame_data_list)
    pairs = _hm.build_pair2_schedule(N)
    logging.info("Pair2 mask schedule (%d frames → %d pairs): %s", N, len(pairs), pairs)

    h, w = frame_data_list[0]["image"].shape[1:]
    grid_uv, dh, dw = _hm.sample_dense_grid(h, w, args.dense_downsample)
    chunk_size = getattr(sdk, "chunk_size", None)

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    amp_ctx = (
        torch.autocast("cuda", enabled=True, dtype=amp_dtype)
        if args.use_amp else nullcontext()
    )

    frame_logit_lists: dict[int, list[np.ndarray]] = {}
    use_global_scale = args.scale_mode == "global"
    global_depth_scale = (
        _hm.compute_sequence_scale(frame_data_list) if use_global_scale else None
    )

    with torch.no_grad():
        for global_a, global_b in tqdm(pairs, desc="Pair2 mask"):
            pair_fds = [frame_data_list[global_a], frame_data_list[global_b]]
            depth_scale = (
                global_depth_scale if use_global_scale
                else _hm.compute_pair_scale(pair_fds)
            )
            batch = _hm.build_pair_batch(pair_fds, device, depth_scale)
            patch_tokens, time_token = _hm.encode_pair_features(sdk, batch, amp_ctx)

            for local_src, global_idx in ((0, global_a), (1, global_b)):
                logits = decode_dense_pair_motion_mask(
                    sdk, batch, patch_tokens, time_token,
                    (local_src, 1 - local_src), grid_uv, device,
                    args.dense_query_batch_size, chunk_size,
                )
                frame_logit_lists.setdefault(global_idx, []).append(logits)

    results_list = _assemble_mask_results(
        frame_data_list, frame_logit_lists, dh, dw, h, w, args,
    )
    infer_meta = dict(
        global_depth_scale=global_depth_scale,
        scale_mode=args.scale_mode,
        encoder_window=2,
        pair_schedule="pair2",
        mask_logit_threshold=args.mask_logit_threshold,
    )
    return results_list, infer_meta


def run_motion_mask_inference(pipeline, frame_data_list, device, args):
    if args.pair_schedule == "pair2":
        return run_pair2_inference(pipeline, frame_data_list, device, args)
    return run_adjacent_window_inference(pipeline, frame_data_list, device, args)


def blend_mask_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float, color) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    dyn = mask.astype(bool)
    if dyn.any():
        color_arr = np.array(color, dtype=np.float32)
        out[dyn] = out[dyn] * (1.0 - alpha) + color_arr * alpha
    return out.astype(np.uint8)


def prob_to_heatmap_bgr(prob: np.ndarray) -> np.ndarray:
    prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(prob_u8, cv2.COLORMAP_TURBO)


def get_gif_filename(args) -> str:
    suffix = args.gif_suffix if args.gif_suffix is not None else f"win{args.encoder_window}"
    return f"motion_mask_overlay_{suffix}.gif" if suffix else "motion_mask_overlay.gif"


def save_sequence_outputs(results_list, out_root, args, seq_name, pair_schedule, infer_meta=None):
    out_dirs = {
        "mask": os.path.join(out_root, "motion_masks"),
        "prob": os.path.join(out_root, "motion_mask_probs"),
        "overlay": os.path.join(out_root, "motion_mask_overlay"),
        "npy": os.path.join(out_root, "motion_mask_logits"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    gif_frames = []
    saved = 0
    for res in results_list:
        motion_mask = res.get("motion_mask")
        if motion_mask is None:
            continue

        stem = f"{res['global_i']:06d}_frame{res['frame_id']}"
        cv2.imwrite(
            os.path.join(out_dirs["mask"], f"{stem}.png"),
            (motion_mask.astype(np.uint8) * 255),
        )
        if res.get("mask_logit") is not None:
            np.save(os.path.join(out_dirs["npy"], f"{stem}.npy"), res["mask_logit"])

        prob = res.get("mask_prob")
        if args.save_prob_heatmap and prob is not None:
            cv2.imwrite(
                os.path.join(out_dirs["prob"], f"{stem}.png"),
                prob_to_heatmap_bgr(prob),
            )

        overlay_rgb = blend_mask_overlay(
            res["image_rgb"], motion_mask, args.overlay_alpha, args.dynamic_color,
        )
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dirs["overlay"], f"{stem}.png"), overlay_bgr)
        gif_frames.append(overlay_rgb)
        saved += 1

    if args.save_gif and gif_frames:
        gif_path = os.path.join(out_root, get_gif_filename(args))
        pil = [Image.fromarray(f) for f in gif_frames]
        pil[0].save(
            gif_path, save_all=True, append_images=pil[1:],
            duration=int(1000 / args.gif_fps), loop=0,
        )
        logging.info("Saved GIF: %s", gif_path)

    summary = dict(
        sequence=seq_name,
        total_frames=len(results_list),
        saved_frames=saved,
        pair_schedule=args.pair_schedule,
        encoder_window=args.encoder_window,
        encoder_chunk_ranges=(infer_meta or {}).get("encoder_chunk_ranges"),
        mask_logit_threshold=args.mask_logit_threshold,
        mask_aggregation=args.mask_aggregation,
        dense_downsample=args.dense_downsample,
        gif_filename=get_gif_filename(args),
    )
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info("Sequence %r: saved %d motion masks → %s", seq_name, saved, out_root)
    return saved


def run_rgb_sequence(pipeline, sequence_dir, args, device, process_res, patch_size):
    resolved, paths = _hm.resolve_image_dir(
        sequence_dir,
        flat_only=args.flat_images_only,
        auto=not args.no_auto_image_dir,
    )
    paths = _hm.sample_frame_paths(paths, args.max_frames, args.frame_sampling)
    if len(paths) < 2:
        raise RuntimeError(f"Need >= 2 frames under {resolved}, got {len(paths)}")

    seq_name = _hm.sequence_name_from_path(sequence_dir, resolved)
    out_root = os.path.join(args.output_dir, seq_name)
    os.makedirs(out_root, exist_ok=True)

    frame_data_list = []
    for p in paths:
        ten, rgb = _hm.load_frame_tensor(p, process_res, patch_size)
        frame_data_list.append(dict(
            image=ten, image_rgb=rgb, frame_id=Path(p).stem, rgb_path=p,
            c2w=None, intrinsics=None,
        ))

    pair_schedule = (
        _hm.build_pair2_schedule(len(frame_data_list))
        if args.pair_schedule == "pair2"
        else _hm.build_encoder_chunk_ranges(len(frame_data_list), args.encoder_window)
    )
    results, infer_meta = run_motion_mask_inference(pipeline, frame_data_list, device, args)
    save_sequence_outputs(results, out_root, args, seq_name, pair_schedule, infer_meta)


def run_json_sequence(pipeline, args, device, process_res, patch_size):
    frames, scene_name, data_root, cam_params = _hm.load_data_from_json(args.data)
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
            fd = _hm.load_fisheye_json_frame(
                fr, data_root, cam_params, args.view_id,
                process_res, patch_size, args.sensor_size,
                args.fisheye_radius, args.skip_fisheye_crop,
            )
        else:
            fd = _hm.load_json_frame(fr, data_root, process_res, patch_size, args.view_id)
        if fd is not None:
            if args.fisheye and fd.get("c2w") is None:
                continue
            frame_data_list.append(fd)

    if len(frame_data_list) < 2:
        raise RuntimeError("Need >= 2 valid frames from JSON.")

    pair_schedule = (
        _hm.build_pair2_schedule(len(frame_data_list))
        if args.pair_schedule == "pair2"
        else _hm.build_encoder_chunk_ranges(len(frame_data_list), args.encoder_window)
    )
    results, infer_meta = run_motion_mask_inference(pipeline, frame_data_list, device, args)
    save_sequence_outputs(results, out_root, args, seq_name, pair_schedule, infer_meta)


def main():
    args, unknown_args = parse_args()
    cfg = file2dict(args.config)
    config_merge_args(cfg, unknown_args)

    process_res = args.process_res or int(cfg.get("max_size", 504))
    patch_size = args.patch_size or int(cfg.get("patch_size", 14))

    pipeline = _hm.load_pipeline(args, cfg)
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
