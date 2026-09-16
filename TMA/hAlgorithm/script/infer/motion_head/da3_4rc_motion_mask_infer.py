#!/usr/bin/env python3
"""
DA3 + MotionHead4RC: Per-Frame Motion Mask via Chunk-wise Inference.

Architecture (encode once, decode per-pair):
    1. Load all N frames, group into chunks of W (window_size).
    2. For each chunk, run the backbone encoder ONCE for all W frames.
    3. Run the motion_head W times (one per source frame) — only the
       lightweight cross-attention decoder runs, not the backbone.
    4. Extract the scene flow to the adjacent target frame and threshold
       its L2 norm to produce a binary motion mask.

Pairing strategy:
    frame_i (src) → frame_{i+1} (tgt)   for i = 0..N-2
    frame_{N-1} (src) → frame_{N-2} (tgt) for the last frame

Usage:
    python hAlgorithm/script/infer/motion_head/da3_4rc_motion_mask_infer.py \
        --config <config_path> \
        --load_from latest \
        --data <json_path> \
        --output_dir ./results/motion_mask_demo
"""

import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import json
import logging
import torch
import numpy as np
import cv2
from contextlib import nullcontext
from tqdm import tqdm

from hAlgorithm.utils import (
    config_merge_args,
    file2dict,
    instantiate_from_config,
    parse_unknown,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DA3 4RC Chunk-wise Motion Mask Inference"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--load_from", default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results/motion_mask_demo")
    parser.add_argument("--nums", type=int, default=None)
    parser.add_argument("--view_id", type=int, default=None)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--no_time", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--depth_key", type=str, default="depth")
    parser.add_argument("--motion_threshold", type=float, default=0.05,
                        help="L2 norm threshold on scene flow (real-world scale).")
    parser.add_argument("--overlay_alpha", type=float, default=0.5)
    parser.add_argument("--save_gif", action="store_true", default=True)
    parser.add_argument("--gif_fps", type=float, default=4.0)
    parser.add_argument("--window_size", type=int, default=8,
                        help="Number of frames per encoder chunk.")

    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown(unknown_args)

    if args.no_time:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        args.output_dir = os.path.join(args.output_dir, config_name)
    else:
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        args.output_dir = os.path.join(
            args.output_dir, config_name + "_" + now
        )
    os.makedirs(args.output_dir, exist_ok=True)

    if args.load_from in ["None", "none", "null"]:
        args.load_from = None

    return args, unknown_args


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

def load_model(args, cfg):
    logging.info("Loading model...")
    model = instantiate_from_config(cfg["model"]).cuda()
    model.eval()
    model.device = torch.device("cuda")
    model.dtype = (torch.bfloat16 if args.amp_dtype == "bfloat16"
                   else torch.float16)

    if args.load_from is not None:
        if args.load_from == "latest":
            args.load_from = os.path.join(
                os.path.dirname(args.config),
                "checkpoint/latest/ckpt.pth",
            )
        elif args.load_from == "best":
            args.load_from = os.path.join(
                os.path.dirname(args.config),
                "checkpoint/best/ckpt.pth",
            )
        model.load_checkpoint(ckpt_path=args.load_from)
    elif ("trainer" in cfg
          and "load_from" in cfg["trainer"]
          and cfg["trainer"]["load_from"] is not None):
        model.load_checkpoint(ckpt_path=cfg["trainer"]["load_from"])

    return model


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_data_from_json(json_path, data_root=None):
    with open(json_path, "r") as f:
        data = json.load(f)
    if data_root is None:
        data_root = os.path.dirname(os.path.dirname(json_path))
    mf_files = data.get("mf_files", {})
    if not mf_files:
        raise ValueError(f"No mf_files found in {json_path}")
    scene_name = sorted(list(mf_files.keys()))[0]
    scene_data = mf_files[scene_name]
    logging.info(f"Loaded scene: {scene_name}")

    # Extract scene-level cam_params (intrinsics indexed by view_id)
    cam_params = {}
    raw_cam_params = scene_data.get("cam_params", {})
    if raw_cam_params:
        for cam_key, cam_val in raw_cam_params.items():
            cam_id = int(cam_key.replace("cam", ""))
            cam_params[cam_id] = cam_val
        logging.info(f"Loaded cam_params for {len(cam_params)} cameras")

    frames = scene_data.get("frames", [])
    if isinstance(scene_data, list):
        frames = scene_data

    return frames, scene_name, data_root, cam_params


def _parse_intrinsics(cam_in, img_scale):
    """Parse intrinsics from various formats → (K_scaled, K_orig) or (None, None)."""
    if cam_in is None:
        return None, None
    if isinstance(cam_in, list) and len(cam_in) >= 4:
        fx, fy, cx, cy = cam_in[:4]
        K_orig = np.eye(3, dtype=np.float32)
        K_orig[0, 0], K_orig[1, 1] = fx, fy
        K_orig[0, 2], K_orig[1, 2] = cx, cy
        K = K_orig.copy()
        K[0, 0] *= img_scale
        K[1, 1] *= img_scale
        K[0, 2] *= img_scale
        K[1, 2] *= img_scale
        return K, K_orig
    if (isinstance(cam_in, list) and len(cam_in) == 3
            and isinstance(cam_in[0], list)):
        K_orig = np.array(cam_in, dtype=np.float32)
        K = K_orig.copy()
        K[0, 0] *= img_scale
        K[1, 1] *= img_scale
        K[0, 2] *= img_scale
        K[1, 2] *= img_scale
        return K, K_orig
    return None, None


def load_single_frame(frame, data_root, process_res,
                      depth_key="depth", cam_params=None):
    """Load one frame → dict with image tensor, intrinsics, c2w, rgb, depth."""
    rgb_path = frame.get("rgb")
    if rgb_path is None:
        return None

    full_rgb_path = (os.path.join(data_root, rgb_path)
                     if not os.path.isabs(rgb_path) else rgb_path)
    if not os.path.exists(full_rgb_path):
        return None

    img = cv2.cvtColor(cv2.imread(full_rgb_path), cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    img_scale = process_res / max(orig_h, orig_w)
    new_h = (int(orig_h * img_scale) // 14) * 14
    new_w = (int(orig_w * img_scale) // 14) * 14
    img_resized = cv2.resize(img, (new_w, new_h))

    img_tensor = (img_resized.astype(np.float32) / 127.5 - 1.0)
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1)

    # Intrinsics: try per-frame first, then fall back to scene-level cam_params
    cam_in = frame.get("cam_in") or frame.get("K")
    if cam_in is None and cam_params is not None:
        view_id = frame.get("view_id", 0)
        cp = cam_params.get(view_id, {})
        cam_in = cp.get("K")
    K, K_orig = _parse_intrinsics(cam_in, img_scale)

    # Extrinsics (c2w)
    c2w_raw = frame.get("cam2world_pose")
    c2w = (torch.from_numpy(np.array(c2w_raw, dtype=np.float32))
           if c2w_raw is not None else None)

    # Depth (optional, for scale computation)
    depth = None
    depth_path = frame.get(depth_key)
    if depth_path is not None:
        full_dp = (os.path.join(data_root, depth_path)
                   if not os.path.isabs(depth_path) else depth_path)
        if os.path.exists(full_dp):
            try:
                if full_dp.endswith('.npy'):
                    depth = np.load(full_dp).astype(np.float32)
                elif full_dp.endswith('.npz'):
                    npz = np.load(full_dp)
                    depth = npz[list(npz.keys())[0]].astype(np.float32)
                else:
                    depth = cv2.imread(
                        full_dp, cv2.IMREAD_ANYDEPTH
                    ).astype(np.float32)
                    depth /= frame.get("depth_scale", 1.0)
                if depth.ndim == 3:
                    depth = depth[..., 0]
            except Exception as e:
                logging.warning(f"Depth load failed {full_dp}: {e}")

    return dict(
        image=img_tensor,
        image_rgb=img_resized,
        intrinsics=torch.from_numpy(K) if K is not None else None,
        K_orig=K_orig,
        c2w=c2w,
        depth=depth,
        frame_id=frame.get("frame_id", 0),
        view_id=frame.get("view_id", 0),
    )


# ──────────────────────────────────────────────────────────────────────
# Scale computation
# ──────────────────────────────────────────────────────────────────────

def compute_depth_scale(frame_data_list, w2c_aligned):
    all_norms = []
    for i, fd in enumerate(frame_data_list):
        depth, K_orig = fd.get("depth"), fd.get("K_orig")
        if depth is None or K_orig is None:
            continue
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.flatten()
        valid = (z > 1e-3) & (z < 1000.0)
        if valid.sum() < 100:
            continue
        x_cam = (u.flatten()[valid] - K_orig[0, 2]) * z[valid] / K_orig[0, 0]
        y_cam = (v.flatten()[valid] - K_orig[1, 2]) * z[valid] / K_orig[1, 1]
        pts = np.stack([x_cam, y_cam, z[valid]], axis=1)
        c2w_al = torch.linalg.inv(w2c_aligned[i]).numpy()
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], 1)
        pts_ref = (c2w_al @ pts_h.T).T[:, :3]
        all_norms.append(np.linalg.norm(pts_ref, axis=1))

    if all_norms:
        return max(float(np.mean(np.concatenate(all_norms))), 0.1)

    if w2c_aligned is not None:
        cam_pos = torch.linalg.inv(w2c_aligned)[:, :3, 3]
        return max(float(cam_pos.norm(dim=-1).max().clamp(min=0.1)), 0.1)
    return 1.0


def compute_chunk_scale(chunk_fds):
    """Compute depth-based global scale for a chunk of frames."""
    c2ws = [fd["c2w"] for fd in chunk_fds if fd["c2w"] is not None]
    if len(c2ws) < 2:
        return 1.0
    c2w_stack = torch.stack(c2ws, dim=0)
    w2c = torch.linalg.inv(c2w_stack)
    w2c_aligned = w2c @ c2w_stack[0:1]
    return compute_depth_scale(chunk_fds, w2c_aligned)


# ──────────────────────────────────────────────────────────────────────
# Batch building for a chunk of W frames
# ──────────────────────────────────────────────────────────────────────

def build_chunk_batch(chunk_fds, device, depth_scale):
    """
    Build a model-ready batch for W frames.

    Returns batch dict with all tensors normalised the same way
    MVFRPipeline.get_inputs does (extrinsics translations / scale).
    """
    W = len(chunk_fds)
    images = torch.stack([fd["image"] for fd in chunk_fds]).unsqueeze(0)

    # Intrinsics [1, W, 3, 3]
    K_list = [fd["intrinsics"] for fd in chunk_fds]
    if not all(k is not None for k in K_list):
        raise ValueError("All frames in chunk must have valid intrinsics.")
    intrinsics = torch.stack(K_list).unsqueeze(0)

    # Extrinsics [1, W, 4, 4]: w2c aligned to first frame, normalised
    c2w_list = [fd["c2w"] for fd in chunk_fds]
    if not all(e is not None for e in c2w_list):
        raise ValueError("All frames in chunk must have valid extrinsics.")
    c2w = torch.stack(c2w_list)
    w2c = torch.linalg.inv(c2w)
    w2c_aligned = w2c @ c2w[0:1]
    # Normalise translations by scale (matching pipeline.get_inputs)
    w2c_aligned[..., :3, 3] /= depth_scale
    extrinsics = w2c_aligned.unsqueeze(0)

    # Scale [1, W, 1, 1, 1]
    scale = torch.full(
        (1, W, 1, 1, 1), depth_scale, dtype=torch.float32,
    )

    h, w = chunk_fds[0]["image"].shape[1:]
    meta_data = {
        "name": ["chunk_infer"],
        "frames": torch.tensor([W]),
        "views": torch.tensor([1]),
        "input_height": torch.tensor([h]),
        "input_width": torch.tensor([w]),
        "origin_height": torch.tensor([h]),
        "origin_width": torch.tensor([w]),
        "data_idx": torch.tensor([0]),
    }

    return dict(
        image=images.to(device),
        intrinsics=intrinsics.to(device),
        extrinsics_reff=extrinsics.to(device),
        scale=scale.to(device),
        meta_data=meta_data,
    )


# ──────────────────────────────────────────────────────────────────────
# Core: chunk-wise encode-once, decode-per-pair
# ──────────────────────────────────────────────────────────────────────

def run_chunk_motion(pipeline, chunk_fds, device, args, global_offset, N_total):
    """
    Encode a chunk of W frames once through the backbone,
    then run the motion decoder for each source frame.

    Returns:
        list of (flow_magnitude_hw, global_frame_idx) tuples
    """
    amp_dtype = (torch.float16 if args.amp_dtype == "float16"
                 else torch.bfloat16)
    amp_ctx = (torch.autocast("cuda", enabled=True, dtype=amp_dtype)
               if args.use_amp else nullcontext())

    depth_scale = compute_chunk_scale(chunk_fds)
    batch = build_chunk_batch(chunk_fds, device, depth_scale)

    sdk = pipeline.model  # MVBaseMotionWithTime
    W = len(chunk_fds)

    with torch.no_grad(), amp_ctx:
        # ── 1. Backbone encoder: run ONCE for all W frames ──
        patch_features, pos, patch_start_idx, _ = sdk.aggregator(
            rgb=batch["image"],
            scale=batch["scale"],
            prompt_depth=None,
            intrinsics=batch["intrinsics"],
            ray_directions=None,
            w2c=batch["extrinsics_reff"],
            c2w=None,
            ray_world=None,
            meta_data=batch["meta_data"],
        )

        # ── 2. Motion decoder: run per source frame ──
        query_times = torch.zeros(1, 1, device=device)
        results_list = []

        for local_i in range(W):
            global_i = global_offset + local_i

            # Skip the last frame of a non-final chunk — it will be
            # processed as the first frame overlap in the next chunk.
            is_last_chunk_frame = (local_i == W - 1)
            is_last_global_frame = (global_i == N_total - 1)
            if is_last_chunk_frame and not is_last_global_frame:
                continue

            # Adjacent target
            if global_i < N_total - 1:
                tgt_local = local_i + 1
            else:
                tgt_local = local_i - 1

            motion_out = sdk.motion_head(
                patch_features,
                query_times=query_times,
                meta_data=batch["meta_data"],
                patch_start_idx=patch_start_idx,
                src_frame_idx=local_i,
            )

            sf = motion_out.get("scene_flow")  # [B, 1, W, 3, H, W]
            if sf is None:
                sf = motion_out.get("dynamic_pointmap")

            if sf is not None:
                # Extract flow from src to target, denormalise
                flow = sf[0, 0, tgt_local].float().cpu()  # [3, H, W]
                flow = flow * depth_scale
                mag = torch.norm(flow, dim=0).numpy()  # [H, W]
            else:
                h, w = chunk_fds[0]["image"].shape[1:]
                mag = np.zeros((h, w), dtype=np.float32)

            results_list.append((mag, global_i))

    return results_list


# ──────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────

def magnitude_to_mask(magnitude, threshold):
    return (magnitude > threshold).astype(np.uint8) * 255


def visualize_motion_mask(rgb, mask, magnitude, alpha=0.5):
    h, w = rgb.shape[:2]
    mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mag_r = cv2.resize(magnitude, (w, h), interpolation=cv2.INTER_LINEAR)

    overlay = rgb.copy()
    dyn = mask_r > 0
    overlay[dyn] = (
        overlay[dyn].astype(np.float32) * (1 - alpha)
        + np.array([255, 50, 50], dtype=np.float32) * alpha
    ).astype(np.uint8)

    mag_norm = mag_r / (mag_r.max() + 1e-8)
    mag_u8 = (mag_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mag_u8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return overlay, heatmap


def generate_gif(images, path, fps=4.0):
    try:
        from PIL import Image
        pil = [Image.fromarray(img) for img in images]
        pil[0].save(path, save_all=True, append_images=pil[1:],
                     duration=int(1000 / fps), loop=0)
        logging.info(f"Saved GIF: {path}")
    except ImportError:
        logging.warning("Pillow not available, skipping GIF.")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    args, unknown_args = parse_args()
    cfg = file2dict(args.config)
    config_merge_args(cfg, unknown_args)

    pipeline = load_model(args, cfg)
    if hasattr(pipeline, 'save_output_cfg'):
        pipeline.save_output_cfg.update(dict(
            save_motion_results=True,
            save_motion_3d=False,
        ))

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    logging.info(f"Output: {args.output_dir}")
    logging.info(f"Threshold: {args.motion_threshold}, Window: {args.window_size}")

    if args.data is None:
        logging.error("No --data provided.")
        return

    frames, scene_name, data_root, cam_params = load_data_from_json(args.data)

    if args.view_id is not None:
        frames = [f for f in frames if f.get("view_id") == args.view_id]
        logging.info(f"Filtered to {len(frames)} frames (view_id={args.view_id})")
    if args.nums is not None:
        frames = frames[:args.nums]

    N = len(frames)
    if N < 2:
        logging.error(f"Need ≥ 2 frames, got {N}")
        return

    # Load all frames
    frame_data_list = []
    for fr in tqdm(frames, desc="Loading frames"):
        fd = load_single_frame(
            fr, data_root, args.process_res, args.depth_key,
            cam_params=cam_params,
        )
        if fd is None:
            continue
        if fd["intrinsics"] is None or fd["c2w"] is None:
            logging.warning(f"Skipping frame {fr.get('frame_id', '?')}: "
                            "missing intrinsics or extrinsics")
            continue
        frame_data_list.append(fd)

    N = len(frame_data_list)
    if N < 2:
        logging.error("Not enough valid frames.")
        return

    logging.info(f"Processing {N} frames with window_size={args.window_size}")

    device = next(pipeline.parameters()).device
    W = args.window_size

    mask_dir = os.path.join(args.output_dir, "motion_masks")
    overlay_dir = os.path.join(args.output_dir, "motion_overlays")
    magnitude_dir = os.path.join(args.output_dir, "flow_magnitude")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(magnitude_dir, exist_ok=True)

    # Collect per-frame results (indexed by global_i)
    all_magnitudes = [None] * N
    overlay_images = []

    # Process chunks with stride W-1 (overlap of 1 frame at boundary)
    stride = max(W - 1, 1)
    chunk_starts = list(range(0, N, stride))
    # Ensure last chunk covers the tail
    if chunk_starts[-1] + 1 < N and chunk_starts[-1] + W < N:
        chunk_starts.append(N - W)

    for cs in tqdm(chunk_starts, desc="Chunk inference"):
        ce = min(cs + W, N)
        if ce - cs < 2:
            break
        chunk_fds = frame_data_list[cs:ce]

        results = run_chunk_motion(
            pipeline, chunk_fds, device, args,
            global_offset=cs, N_total=N,
        )

        for mag, gi in results:
            if all_magnitudes[gi] is None:
                all_magnitudes[gi] = mag

    # Visualise
    for gi in tqdm(range(N), desc="Saving visualizations"):
        mag = all_magnitudes[gi]
        if mag is None:
            logging.warning(f"Frame {gi}: no motion result, skipping")
            continue

        mask = magnitude_to_mask(mag, args.motion_threshold)
        rgb = frame_data_list[gi]["image_rgb"]
        overlay, heatmap = visualize_motion_mask(
            rgb, mask, mag, args.overlay_alpha
        )

        fid = frame_data_list[gi]["frame_id"]
        cv2.imwrite(
            os.path.join(mask_dir, f"{gi:06d}_frame{fid}.png"), mask,
        )
        cv2.imwrite(
            os.path.join(overlay_dir, f"{gi:06d}_frame{fid}.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(magnitude_dir, f"{gi:06d}_frame{fid}.png"),
            cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR),
        )
        overlay_images.append(overlay)

        dyn_ratio = mask.sum() / mask.size * 100
        logging.debug(f"Frame {gi} (id={fid}): dynamic={dyn_ratio:.1f}%")

    if args.save_gif and overlay_images:
        generate_gif(
            overlay_images,
            os.path.join(args.output_dir, "motion_mask_overlay.gif"),
            fps=args.gif_fps,
        )

    summary = dict(
        total_frames=N, motion_threshold=args.motion_threshold,
        window_size=args.window_size, scene=scene_name,
    )
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Done. Results → {args.output_dir}")
    logging.info(f"  motion_masks/  overlay/  flow_magnitude/")


if __name__ == "__main__":
    main()
