#!/usr/bin/env python3
"""
Any4D Network Inference with Scene Flow Visualization.

This script is designed for inference with the Any4D network that includes
a scene flow head for 3D motion prediction. After inference, motion visualizations
are automatically saved.

Usage:
    python hAlgorithm/script/infer/any4d_kosmo_infer.py \
        --config <config_path> \
        --data <json_path> \
        --output_dir ./results/any4d_output

If no --load_from is specified, the script will automatically use the checkpoint path
from the config file's trainer.load_from field.

Example:
    python hAlgorithm/script/infer/any4d_kosmo_infer.py \
        --config results/any4d_exp/any4d_config.py \
        --data /path/to/data.json \
        --output_dir ./results/any4d_demo

    # Or with explicit checkpoint:
    python hAlgorithm/script/infer/any4d_kosmo_infer.py \
        --config results/any4d_exp/any4d_config.py \
        --load_from latest \
        --data /path/to/data.json \
        --output_dir ./results/any4d_demo
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
    parser = argparse.ArgumentParser(description="Any4D Network Inference with Scene Flow Visualization")
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
        default="./results/any4d_demo",
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
    parser.add_argument(
        "--depth_key",
        type=str,
        default="lidar_depth",
        help="Key in JSON for depth file path (e.g., 'depth', 'pred_depth', 'lidar_depth').",
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
    logging.info("Loading model...")
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
        cam_params: Camera parameters dict (cam_params[view_id] = {K, height, width, ...})
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
    
    # Extract cam_params (intrinsics are stored at scene level, not per-frame)
    cam_params = None
    if isinstance(scene_data, dict):
        cam_params = scene_data.get("cam_params", None)
        if cam_params:
            logging.info(f"Found cam_params for cameras: {list(cam_params.keys())}")
    
    logging.info(f"Loaded scene: {scene_name}")
    
    return scene_data, scene_name, data_root, cam_params


def prepare_batch_from_frame(frame_data, data_root, device, process_res=504, view_id=None, depth_key="depth", cam_params=None):
    """
    Prepare a batch dictionary from frame data for model inference.
    
    Args:
        frame_data: List of frame dictionaries
        data_root: Root directory for data paths
        device: Target device
        process_res: Processing resolution
        view_id: Specific camera view ID to use (None for all views)
        depth_key: Key in JSON for depth file path (e.g., "depth", "pred_depth", "lidar_depth")
        cam_params: Scene-level camera parameters dict (cam_params['camX'] = {K, height, width, ...})
        
    Returns:
        batch: Dictionary containing model inputs
        
    Expected output tensor shapes:
        - image: [B, N, 3, H, W]
        - intrinsics: [B, N, 3, 3]
        - extrinsics_reff: [B, N, 4, 4] (w2c aligned to frame 0)
        - scale: [B, N]
    """
    import cv2
    
    images = []
    intrinsics_list = []
    c2w_list = []  # Store c2w first, then convert to w2c and align
    depth_list = []  # Store depth maps for scale computation
    cam_in_list = []  # Store original cam_in for depth unprojection
    frame_ids = []
    view_ids = []
    img_scale = None  # Image resize scale (consistent across all images)
    orig_h, orig_w = None, None  # Original image size for depth scaling
    
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
        
        # Load intrinsics: Priority order:
        # 1. Per-frame cam_in or K field
        # 2. Scene-level cam_params based on view_id
        cam_in = frame.get("cam_in") or frame.get("K")
        
        # If not in frame, try to get from scene-level cam_params
        if cam_in is None and cam_params is not None:
            cam_key = f"cam{curr_view_id}"  # e.g., "cam0", "cam1"
            if cam_key in cam_params:
                cam_info = cam_params[cam_key]
                cam_in = cam_info.get("K")  # [fx, fy, cx, cy] format
                logging.debug(f"Using cam_params[{cam_key}] for intrinsics: {cam_in}")
        
        cam_in_list.append(cam_in)  # Store for depth unprojection (original scale)
        K_tensor = None
        
        if cam_in is not None:
            # Handle both [fx, fy, cx, cy] (cam_in) and [[fx,0,cx],[0,fy,cy],[0,0,1]] (K matrix) formats
            if isinstance(cam_in, list) and len(cam_in) >= 4:
                # Extract first 4: [fx, fy, cx, cy]
                fx, fy, cx, cy = cam_in[0], cam_in[1], cam_in[2], cam_in[3]
                K = np.eye(3, dtype=np.float32)
                K[0, 0] = fx * img_scale  # fx
                K[1, 1] = fy * img_scale  # fy
                K[0, 2] = cx * img_scale  # cx
                K[1, 2] = cy * img_scale  # cy
                K_tensor = torch.from_numpy(K)
            elif isinstance(cam_in, list) and len(cam_in) == 3 and isinstance(cam_in[0], list):
                # K is 3x3 matrix format
                K = np.array(cam_in, dtype=np.float32)
                K[0, 0] *= img_scale  # fx
                K[1, 1] *= img_scale  # fy
                K[0, 2] *= img_scale  # cx
                K[1, 2] *= img_scale  # cy
                K_tensor = torch.from_numpy(K)
            else:
                # Invalid format
                logging.debug(f"Frame {frame_id}: Invalid intrinsics format")
        
        intrinsics_list.append(K_tensor)
        
        # Load extrinsics: JSON contains c2w (cam2world_pose)
        # WARNING: JSON stores c2w, must invert to get w2c for model
        c2w = frame.get("cam2world_pose")
        if c2w is not None:
            c2w = np.array(c2w, dtype=np.float32)
            c2w_list.append(torch.from_numpy(c2w))
        else:
            c2w_list.append(None)
        
        # Load depth for scale computation
        depth_path = frame.get(depth_key)
        if depth_path is not None:
            full_depth_path = os.path.join(data_root, depth_path) if not os.path.isabs(depth_path) else depth_path
            if os.path.exists(full_depth_path):
                try:
                    if full_depth_path.endswith('.npy'):
                        depth = np.load(full_depth_path).astype(np.float32)
                    elif full_depth_path.endswith('.npz'):
                        npz_data = np.load(full_depth_path)
                        depth = npz_data[list(npz_data.keys())[0]].astype(np.float32)
                    else:
                        # PNG/other image formats
                        depth = cv2.imread(full_depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
                        # Check for depth_scale in frame
                        depth_scale = frame.get("depth_scale", 1.0)
                        depth = depth / depth_scale
                    
                    # Ensure 2D
                    if len(depth.shape) == 3:
                        depth = depth[..., 0]
                    
                    depth_list.append(depth)
                    logging.debug(f"Loaded depth: {full_depth_path}, shape={depth.shape}, range=[{depth.min():.2f}, {depth.max():.2f}]")
                except Exception as e:
                    logging.warning(f"Failed to load depth {full_depth_path}: {e}")
                    depth_list.append(None)
            else:
                logging.warning(f"Depth file not found: {full_depth_path}")
                depth_list.append(None)
        else:
            logging.debug(f"No '{depth_key}' key in frame {frame_id}")
            depth_list.append(None)
        
        frame_ids.append(frame_id)
        view_ids.append(curr_view_id)
    
    if not images:
        return None
    
    num_images = len(images)
    
    # Stack tensors: [N, ...] -> [1, N, ...]
    image_tensor = torch.stack(images, dim=0).unsqueeze(0)  # [1, N, 3, H, W]
    
    # Build intrinsics tensor [1, N, 3, 3]
    num_valid_intrinsics = sum(1 for k in intrinsics_list if k is not None)
    if all(k is not None for k in intrinsics_list):
        intrinsics_tensor = torch.stack(intrinsics_list, dim=0).unsqueeze(0)
    else:
        intrinsics_tensor = None
        logging.warning(f"Intrinsics: {num_valid_intrinsics}/{len(intrinsics_list)} frames have cam_in")
    
    # Build extrinsics tensor with proper preprocessing:
    # 1. Convert c2w to w2c (model expects w2c)
    # 2. Align to first frame's camera coordinate system
    extrinsics_tensor = None
    w2c_aligned = None
    first_c2w = None
    
    if all(e is not None for e in c2w_list):
        c2w_tensor = torch.stack(c2w_list, dim=0)  # [N, 4, 4]
        
        # Step 1: Convert c2w to w2c by inversion
        w2c_tensor = torch.linalg.inv(c2w_tensor)  # [N, 4, 4]
        
        # Step 2: Align to first frame's camera coordinate system
        # extrinsics_reff = w2c @ c2w[0], so first camera becomes identity
        first_c2w = c2w_tensor[0:1]  # [1, 4, 4]
        w2c_aligned = w2c_tensor @ first_c2w  # [N, 4, 4]
        
        extrinsics_tensor = w2c_aligned.unsqueeze(0)  # [1, N, 4, 4]
    else:
        num_valid_c2w = sum(1 for e in c2w_list if e is not None)
        logging.warning(f"Extrinsics: {num_valid_c2w}/{len(c2w_list)} frames have cam2world_pose")
    
    # Compute scale from depth
    depth_scale_value = 1.0  # Default fallback
    
    # Log depth loading summary
    num_valid_depths = sum(1 for d in depth_list if d is not None)
    logging.info(f"Depth loading: {num_valid_depths}/{len(depth_list)} frames have valid depth (key='{depth_key}')")
    
    # Debug: check prerequisites for depth-based scale
    has_depth = any(d is not None for d in depth_list)
    has_intrinsics = intrinsics_tensor is not None
    has_extrinsics = w2c_aligned is not None
    logging.info(f"Scale prerequisites: has_depth={has_depth}, has_intrinsics={has_intrinsics}, has_extrinsics={has_extrinsics}")
    
    if has_depth and has_intrinsics and has_extrinsics:
        all_points_norms = []
        
        for i, depth in enumerate(depth_list):
            if depth is None:
                continue
            
            # Get intrinsics (original scale, not resized) from the stored cam_in_list
            cam_in = cam_in_list[i] if i < len(cam_in_list) else None
            if cam_in is None:
                logging.warning(f"Frame {i}: No cam_in available for depth unprojection")
                continue
            
            # Parse intrinsics - handle both formats
            if isinstance(cam_in, list) and len(cam_in) >= 4:
                # K is [fx, fy, cx, cy] format
                fx, fy, cx, cy = cam_in[0], cam_in[1], cam_in[2], cam_in[3]
                K_orig = np.eye(3, dtype=np.float32)
                K_orig[0, 0] = fx  # fx
                K_orig[1, 1] = fy  # fy
                K_orig[0, 2] = cx  # cx
                K_orig[1, 2] = cy  # cy
            elif isinstance(cam_in, list) and len(cam_in) == 3 and isinstance(cam_in[0], list):
                # K is 3x3 matrix format
                K_orig = np.array(cam_in, dtype=np.float32)
            else:
                logging.warning(f"Frame {i}: Invalid intrinsics format")
                continue
            
            # Unproject depth to 3D points in camera frame
            h, w = depth.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            z = depth.flatten()
            valid_mask = (z > 1e-3) & (z < 1000.0)  # Filter invalid depth
            
            if valid_mask.sum() < 100:
                continue
            
            u_flat = u.flatten()[valid_mask]
            v_flat = v.flatten()[valid_mask]
            z_valid = z[valid_mask]
            
            # Camera coordinates: P_cam = K^-1 @ [u*z, v*z, z]
            x_cam = (u_flat - K_orig[0, 2]) * z_valid / K_orig[0, 0]
            y_cam = (v_flat - K_orig[1, 2]) * z_valid / K_orig[1, 1]
            points_cam = np.stack([x_cam, y_cam, z_valid], axis=1)  # [N_pts, 3]
            
            # Transform to aligned coordinate system (first camera frame)
            c2w_aligned_i = torch.linalg.inv(w2c_aligned[i]).numpy()  # [4, 4]
            
            # Transform points: P_reff = c2w_aligned @ P_cam
            points_homo = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)  # [N_pts, 4]
            points_reff = (c2w_aligned_i @ points_homo.T).T[:, :3]  # [N_pts, 3]
            
            # Compute norms (distance from origin in aligned frame)
            norms = np.linalg.norm(points_reff, axis=1)
            all_points_norms.append(norms)
        
        if len(all_points_norms) > 0:
            all_norms = np.concatenate(all_points_norms)
            # Match with_global_scale: scale = mean(||points||)
            depth_scale_value = float(np.mean(all_norms))
            depth_scale_value = max(depth_scale_value, 0.1)  # Ensure minimum scale
            logging.info(f"Computed scale from depth: {depth_scale_value:.4f} (from {len(all_norms)} points)")
    else:
        # Fallback: compute from camera positions
        if w2c_aligned is not None:
            c2w_aligned = torch.linalg.inv(w2c_aligned)  # [N, 4, 4]
            camera_positions = c2w_aligned[:, :3, 3]  # [N, 3]
            pos_norms = torch.norm(camera_positions, dim=-1)  # [N]
            if pos_norms.max() > 1e-6:
                depth_scale_value = float(pos_norms.max().clamp(min=0.1))
            logging.warning(f"No depth available, using camera-based scale: {depth_scale_value:.4f}")
    
    # Scale tensor: [B, N] - same scale for all views in this batch
    scale_tensor = torch.full((1, num_images), depth_scale_value, dtype=torch.float32, device=device)
    
    # Build meta_data
    unique_frame_ids = sorted(set(frame_ids))
    unique_view_ids = sorted(set(view_ids))
    num_frames = len(unique_frame_ids)
    num_views = len(unique_view_ids)
    
    h, w = images[0].shape[1], images[0].shape[2]
    meta_data = {
        "name": ["any4d_demo"],
        "frames": torch.tensor([num_frames]),
        "views": torch.tensor([num_views]),
        "input_height": torch.tensor([h]),
        "input_width": torch.tensor([w]),
        "origin_height": torch.tensor([h]),
        "origin_width": torch.tensor([w]),
        "data_idx": torch.tensor([0]),
        "data_info": [[{"frame_id": fid, "view_id": vid} for fid, vid in zip(frame_ids, view_ids)]],
    }
    
    batch = {
        "image": image_tensor.to(device),
        "meta_data": meta_data,
    }
    
    if intrinsics_tensor is not None:
        batch["intrinsics"] = intrinsics_tensor.to(device)
    
    if extrinsics_tensor is not None:
        batch["extrinsics"] = extrinsics_tensor.to(device)
        batch["extrinsics_reff"] = extrinsics_tensor.to(device)
    
    # Scale: [B, N] - model config uses 'sparse_pointmap_max_range'
    batch["scale"] = scale_tensor
    batch["sparse_pointmap_max_range"] = scale_tensor
    
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
    
    # Load data
    if args.data is not None:
        scene_data, scene_name, data_root, cam_params = load_data_from_json(args.data)
        
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
            
            # Prepare batch
            batch = prepare_batch_from_frame(
                frame_batch, 
                data_root, 
                device,
                args.process_res,
                depth_key=args.depth_key,
                cam_params=cam_params,
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