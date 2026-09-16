#!/usr/bin/env python3
"""
VDPM Network Inference with Scene Flow Visualization.

VDPM is a pure image-based algorithm that predicts depth, camera poses, and
dynamic point maps from RGB video frames alone - no external camera parameters needed.

Usage:
    python hAlgorithm/script/infer/motion_head/vdpm_kosmo_infer.py \
        --config <config_path> \
        --data <json_path> \
        --output_dir ./results/vdpm_output

Example:
    python hAlgorithm/script/infer/motion_head/vdpm_kosmo_infer.py \
        --config hAlgorithm/configs/open/vdpm/vdpm_260131_50f_eval.py \
        --data /path/to/data.json \
        --output_dir ./results/vdpm_demo
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
    parser = argparse.ArgumentParser(description="VDPM Network Inference with Scene Flow Visualization")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file.",
    )
    parser.add_argument(
        "--load_from",
        default=None,
        help="Path of checkpoint to be loaded. Use 'latest' or 'best' for automatic path resolution. If None, uses config's checkpoint path.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to JSON data file (mf_files format).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/vdpm_demo",
        help="The output directory for predictions and visualizations.",
    )
    parser.add_argument(
        "--nums",
        type=int,
        default=None,
        help="Maximum number of samples to process.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=50,
        help="Number of frames to process per batch (default: 50).",
    )
    parser.add_argument(
        "--view_id",
        type=int,
        default=None,
        help="Specific camera view ID to use. If None, use all views.",
    )
    parser.add_argument(
        "--process_res",
        type=int,
        default=504,
        help="Processing resolution (height and width will be adjusted to this).",
    )
    parser.add_argument(
        "--no_time",
        action="store_true",
        help="If set, don't append timestamp to output directory.",
    )
    parser.add_argument(
        "--save_motion_3d",
        action="store_true",
        default=True,
        help="Save 3D motion visualization using Rerun.",
    )
    parser.add_argument(
        "--save_motion_results",
        action="store_true",
        default=True,
        help="Save 2D motion visualization results.",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Save 3D reconstruction as GLB.",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision for inference.",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Data type for AMP.",
    )

    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown(unknown_args)

    # Setup output directory
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


def load_model(args, cfg):
    """Load model from config and checkpoint."""
    logging.info("Loading VDPM model...")
    model = instantiate_from_config(cfg["model"]).cuda()
    model.eval()
    
    # Set device and dtype for inference
    model.device = torch.device("cuda")
    model.dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    # Resolve checkpoint path
    load_path = None
    
    # Priority 1: Explicit --load_from argument
    if args.load_from is not None and args.load_from != "":
        load_path = args.load_from
        if load_path == "latest":
            load_path = os.path.join(
                os.path.dirname(args.config), "checkpoint/latest/ckpt.pth"
            )
        elif load_path == "best":
            load_path = os.path.join(
                os.path.dirname(args.config), "checkpoint/best/ckpt.pth"
            )
        logging.info(f"Loading checkpoint from --load_from: {load_path}")
        model.load_checkpoint(ckpt_path=load_path)
    # Priority 2: Config file's checkpoint path
    elif "trainer" in cfg and "load_from" in cfg["trainer"] and cfg["trainer"]["load_from"] is not None:
        load_path = cfg["trainer"]["load_from"]
        logging.info(f"Loading checkpoint from config: {load_path}")
        model.load_checkpoint(ckpt_path=load_path)
    else:
        logging.warning("No checkpoint path found. Using model as-is (not recommended for inference).")

    return model


def load_data_from_json(json_path, data_root=None):
    """
    Load data from JSON file (mf_files format).
    
    Returns:
        scene_data: Scene data dict containing 'frames', 'cam_params', etc.
        scene_name: Name of the scene
        data_root: Root directory for data paths
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    if data_root is None:
        data_root = os.path.dirname(os.path.dirname(json_path))
    
    mf_files = data.get("mf_files", {})
    
    # Get first scene
    if not mf_files:
        raise ValueError(f"No mf_files found in {json_path}")
    
    scene_name = sorted(list(mf_files.keys()))[0]
    scene_data = mf_files[scene_name]
    
    logging.info(f"Loaded scene: {scene_name}")
    
    return scene_data, scene_name, data_root


def prepare_batch_from_frame(frame_data, data_root, device, process_res=504, view_id=None):
    """
    Prepare a batch dictionary from frame data for VDPM inference.
    
    VDPM is a pure image-based model - it only needs RGB images as input.
    Camera intrinsics/extrinsics are predicted by the model internally.
    
    Args:
        frame_data: List of frame dictionaries
        data_root: Root directory for data paths
        device: Target device
        process_res: Processing resolution
        view_id: Specific camera view ID to use (None for all views)
        
    Returns:
        batch: Dictionary containing model inputs
        
    Expected output tensor shapes:
        - image: [B, N, 3, H, W]
    """
    import cv2
    
    images = []
    frame_ids = []
    view_ids = []
    
    for frame in frame_data:
        frame_id = frame.get("frame_id", 0)
        curr_view_id = frame.get("view_id", 0)
        
        # Filter by specific view_id if specified
        if view_id is not None and curr_view_id != view_id:
            continue
        
        rgb_path = frame.get("rgb")
        
        if rgb_path is None:
            continue
        
        # Load image
        full_rgb_path = os.path.join(data_root, rgb_path) if not os.path.isabs(rgb_path) else rgb_path
        if not os.path.exists(full_rgb_path):
            logging.warning(f"Image not found: {full_rgb_path}")
            continue
        
        img = cv2.imread(full_rgb_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original size
        orig_h, orig_w = img.shape[:2]
        
        # Resize to process_res
        img_scale = process_res / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * img_scale), int(orig_w * img_scale)
        # Make divisible by 14 (patch size)
        new_h = (new_h // 14) * 14
        new_w = (new_w // 14) * 14
        img = cv2.resize(img, (new_w, new_h))
        
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
        images.append(img)
        
        frame_ids.append(frame_id)
        view_ids.append(curr_view_id)
    
    if not images:
        return None
    
    num_images = len(images)
    
    # Stack tensors: [N, ...] -> [1, N, ...]
    image_tensor = torch.stack(images, dim=0).unsqueeze(0)  # [1, N, 3, H, W]
    
    # Build meta_data
    unique_frame_ids = sorted(set(frame_ids))
    unique_view_ids = sorted(set(view_ids))
    num_frames = len(unique_frame_ids)
    num_views = len(unique_view_ids)
    
    h, w = images[0].shape[1], images[0].shape[2]
    meta_data = {
        "name": ["vdpm_demo"],
        "frames": torch.tensor([num_frames]),
        "views": torch.tensor([num_views]),
        "input_height": torch.tensor([h]),
        "input_width": torch.tensor([w]),
        "origin_height": torch.tensor([h]),
        "origin_width": torch.tensor([w]),
        "data_idx": torch.tensor([0]),
        "data_info": [[{"frame_id": fid, "view_id": vid} for fid, vid in zip(frame_ids, view_ids)]],
    }
    
    # VDPM only needs images - no intrinsics/extrinsics needed
    batch = {
        "image": image_tensor.to(device),
        "meta_data": meta_data,
    }
    
    return batch


def run_inference(model, batch, args):
    """
    Run model inference and return outputs.
    """
    # Prepare AMP dtype
    amp_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    amp_dtype = amp_dtype_map.get(args.amp_dtype, torch.bfloat16)
    
    batch["use_amp"] = args.use_amp
    batch["amp_dtype"] = amp_dtype
    
    with torch.no_grad():
        if args.use_amp:
            with torch.autocast("cuda", enabled=True, dtype=amp_dtype):
                outputs = model.infer(**batch)
        else:
            outputs = model.infer(**batch)
    
    return outputs


def save_motion_visualizations(model, outputs, meta_data, output_dir, data_idx):
    """
    Save motion visualizations (2D and 3D).
    """
    from hAlgorithm.modules.pipelines2.utils.visualize_motion import (
        vis_motion_head_results,
        vis_motion_3d_rerun,
    )
    
    # Check if outputs have scene_flow
    if not outputs or not hasattr(outputs[0], 'scene_flow_pred') or outputs[0].scene_flow_pred is None:
        logging.warning("No scene_flow_pred found in outputs, skipping motion visualization")
        return
    
    # Create motion output directories
    motion_out_dir = os.path.join(output_dir, f"motion/{data_idx:06d}")
    os.makedirs(motion_out_dir, exist_ok=True)
    
    cfg = model.save_output_cfg
    
    # 2D Motion Visualization
    if cfg.get("save_motion_results", True):
        logging.info(f"Saving 2D motion visualizations to {motion_out_dir}")
        vis_motion_head_results(
            cfg=cfg,
            mv_outputs=outputs,
            motion_out_dir=motion_out_dir,
            data_idx=data_idx,
            meta_data=meta_data,
        )
    
    # 3D Rerun Visualization
    if cfg.get("save_motion_3d", True):
        logging.info(f"Saving 3D motion visualization (Rerun)")
        vis_motion_3d_rerun(
            cfg=cfg,
            mv_outputs=outputs,
            out_dir=output_dir,
            data_idx=data_idx,
            meta_data=meta_data,
        )


def save_reconstruction_outputs(outputs, output_dir, data_idx):
    """
    Save 3D reconstruction as PLY/GLB files.
    """
    from hAlgorithm.modules.pipelines2.utils.save_outputs import save_mv_outputs
    
    recon_out_dir = os.path.join(output_dir, f"reconstruction/{data_idx:06d}")
    os.makedirs(recon_out_dir, exist_ok=True)
    
    # Save outputs (PLY, GLB, camera info, etc.)
    try:
        cfg = {
            "save_everything": False,
            "save_output_conf": True,
            "save_gaussians": False,
            "save_glb_results": True,
            "save_cameras": True,
            "output_conf_ratio": 0.2,
            "output_match_input_res": False,
        }
        save_mv_outputs(
            outputs=outputs,
            out_dir=recon_out_dir,
            cfg=cfg,
        )
        logging.info(f"Saved reconstruction outputs to {recon_out_dir}")
    except Exception as e:
        logging.warning(f"Failed to save reconstruction outputs: {e}")


def main():
    args, unknown_args = parse_args()
    
    # Load config
    cfg = file2dict(args.config)
    config_merge_args(cfg, unknown_args)
    
    # Update save_output_cfg
    if "model" in cfg and hasattr(cfg["model"], "get"):
        save_cfg = cfg["model"].get("save_output_cfg", {})
    else:
        save_cfg = {}
    save_cfg["save_motion_results"] = args.save_motion_results
    save_cfg["save_motion_3d"] = args.save_motion_3d
    save_cfg["vis_3d_prefer_gt"] = False
    
    # Load model
    model = load_model(args, cfg)
    
    # Update model's save_output_cfg
    if hasattr(model, 'save_output_cfg'):
        model.save_output_cfg.update(save_cfg)
    
    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Processing config: frames={args.frames}, view_id={args.view_id}")
    logging.info(f"VDPM is a pure image-based model - no external camera parameters needed")
    
    # Load data
    if args.data is not None:
        scene_data, scene_name, data_root = load_data_from_json(args.data)
        
        # Get frames
        if isinstance(scene_data, list):
            frames = scene_data
        elif isinstance(scene_data, dict):
            frames = scene_data.get("frames", scene_data.get("data", []))
        else:
            frames = []
        
        # Filter frames by view_id if specified
        if args.view_id is not None:
            frames = [f for f in frames if f.get("view_id") == args.view_id]
            logging.info(f"Filtered to {len(frames)} frames with view_id={args.view_id}")
        
        # Limit total frames if specified
        if args.nums is not None:
            frames = frames[:args.nums]
        
        total_frames = len(frames)
        batch_size = args.frames  # Number of frames per batch
        num_batches = (total_frames + batch_size - 1) // batch_size
        
        logging.info(f"Processing {total_frames} frames in {num_batches} batches (batch_size={batch_size}, view_id={args.view_id})")
        
        # Get device from model
        device = next(model.parameters()).device
        
        # Process frames in batches
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_frames)
            frame_batch = frames[start_idx:end_idx]
            
            # Prepare batch - VDPM only needs images
            batch = prepare_batch_from_frame(
                frame_batch, 
                data_root, 
                device,
                args.process_res,
                view_id=None,  # Already filtered above
            )
            
            if batch is None:
                logging.warning(f"Failed to prepare batch {batch_idx}")
                continue
            
            # Update data_idx in meta_data
            batch["meta_data"]["data_idx"] = torch.tensor([batch_idx])
            
            # Run inference
            outputs = run_inference(model, batch, args)
            
            # Save motion visualizations
            save_motion_visualizations(
                model=model,
                outputs=outputs,
                meta_data=batch["meta_data"],
                output_dir=args.output_dir,
                data_idx=batch_idx,
            )
            
            # Save reconstruction outputs if requested
            if args.save_glb:
                save_reconstruction_outputs(
                    outputs=outputs,
                    output_dir=args.output_dir,
                    data_idx=batch_idx,
                )
            
            logging.info(f"Processed batch {batch_idx}: frames {start_idx}-{end_idx-1}")
    else:
        logging.warning("No data provided. Use --data to specify input JSON file.")
    
    logging.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
