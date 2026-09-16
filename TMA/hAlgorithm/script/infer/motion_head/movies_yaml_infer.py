#!/usr/bin/env python3
"""
Movies Network Inference with YAML Dataset Config.

This script runs inference on datasets specified in YAML config files,
similar to the evaluation pipeline but with visualization output.

Usage:
    python hAlgorithm/script/infer/movies_yaml_infer.py \
        --config <config_path> \
        --yaml <yaml_dataset_path> \
        --load_from latest \
        --output_dir ./results/movies_yaml_infer

Example:
    python hAlgorithm/script/infer/movies_yaml_infer.py \
        --config hAlgorithm/configs/movies_reproduction/movies_any4d_eval_v3.py \
        --yaml hAlgorithm/configs/mv_v1.0/dataset_configs_2/val_any4d_50frames_v3.yaml \
        --load_from latest \
        --output_dir ./results/movies_yaml_infer \
        --nums 5
"""

import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import json
import logging
import pprint
import torch
import yaml
from tqdm import tqdm

from hAlgorithm.utils import (
    config_merge_args,
    file2dict,
    instantiate_from_config,
    parse_unknown,
    get_obj_from_file,
)
from hAlgorithm.datasets.dataloader.collate import default_collate
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_args():
    parser = argparse.ArgumentParser(description="Movies Network Inference with YAML Dataset Config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config file (e.g., movies_any4d_eval_v3.py).",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        required=True,
        help="Path to YAML dataset config file (e.g., val_any4d_50frames_v3.yaml).",
    )
    parser.add_argument(
        "--load_from",
        default=None,
        help="Path of checkpoint to be loaded. Use 'latest' or 'best' for automatic path resolution.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/movies_yaml_infer",
        help="The output directory for predictions and visualizations.",
    )
    parser.add_argument(
        "--nums",
        type=int,
        default=None,
        help="Maximum number of samples to process per dataset.",
    )
    parser.add_argument(
        "--select_dataset",
        type=str,
        default=None,
        help="Comma-separated dataset names to process (e.g., 'kubric4d,pointodyssey').",
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
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--use_yaml_sampler",
        action="store_true",
        default=True,
        help="Use clip_sampler from YAML config instead of Python config.",
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

    # Parse select_dataset
    if args.select_dataset:
        args.select_dataset = [s.strip() for s in args.select_dataset.split(",")]

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
    if args.load_from is not None:
        if args.load_from == "latest":
            args.load_from = os.path.join(
                os.path.dirname(args.config), "checkpoint/latest/ckpt.pth"
            )
        elif args.load_from == "best":
            args.load_from = os.path.join(
                os.path.dirname(args.config), "checkpoint/best/ckpt.pth"
            )
        logging.info(f"Loading checkpoint from: {args.load_from}")
        model.load_checkpoint(ckpt_path=args.load_from)
    elif "trainer" in cfg and "load_from" in cfg["trainer"] and cfg["trainer"]["load_from"] is not None:
        logging.info(f"Loading checkpoint from config: {cfg['trainer']['load_from']}")
        model.load_checkpoint(ckpt_path=cfg["trainer"]["load_from"])

    return model


def load_datasets_from_yaml(yaml_path, cfg, select_dataset=None, use_yaml_sampler=True):
    """
    Load datasets from YAML config file.
    
    Args:
        yaml_path: Path to YAML dataset config
        cfg: Main config dict (for basic_config)
        select_dataset: Optional list of dataset names to filter
        use_yaml_sampler: If True, use clip_sampler from YAML (ignore Python config)
        
    Returns:
        List of (dataset, dataset_name) tuples
    """
    logging.info(f"Loading datasets from: {yaml_path}")
    
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    
    dataset_configs = yaml_data.get("datasets", [])
    
    # Load basic config from main config
    # Note: val_basic will override basic settings for validation
    basic_config = dict()
    if "data" in cfg:
        if "basic" in cfg["data"] and cfg["data"]["basic"] is not None:
            basic_config.update(cfg["data"]["basic"])
        # Use val_basic for inference (overrides basic)
        if "val_basic" in cfg["data"] and cfg["data"]["val_basic"] is not None:
            basic_config.update(cfg["data"]["val_basic"])
    
    # Remove clip_sampler from basic_config if use_yaml_sampler is True
    if use_yaml_sampler and "clip_sampler" in basic_config:
        logging.info("Ignoring clip_sampler from Python config (using YAML config)")
        basic_config.pop("clip_sampler")
    
    logging.info(f"Basic config keys: {list(basic_config.keys())}")
    if "clip_sampler" in basic_config:
        logging.info(f"Using clip_sampler from Python config: {basic_config['clip_sampler']}")
    
    datasets = []
    
    for di, dataset_cfg in enumerate(dataset_configs):
        dataset_cfg = dataset_cfg.copy()  # Don't modify original
        
        # Get pipeline
        pipeline_ori = dataset_cfg.pop("pipeline")
        if isinstance(pipeline_ori, str):
            pipeline = get_obj_from_file(pipeline_ori, "pipeline").copy()
        else:
            pipeline = pipeline_ori.copy()
        
        # Get dataset name
        dataset_name = dataset_cfg.get("name", pipeline.get("name", f"dataset_{di}"))
        
        # Filter by select_dataset if specified
        if select_dataset is not None and dataset_name not in select_dataset:
            logging.info(f"Skipping dataset: {dataset_name}")
            continue
        
        # First apply dataset_cfg (from YAML), then basic_config overrides
        # Note: basic_config is applied AFTER dataset_cfg, so it takes precedence
        dataset_cfg.pop("prob", None)
        dataset_cfg.pop("batch_scale", None)
        
        pipeline["phase"] = "val"  # Use val phase for inference
        pipeline.update(dataset_cfg)  # First: YAML settings
        pipeline.update(basic_config)  # Second: basic_config overrides
        
        # Log clip_sampler info
        if "clip_sampler" in pipeline:
            sampler_cfg = pipeline["clip_sampler"]
            view_num = sampler_cfg.get("view_num", "unknown")
            sampler_type = sampler_cfg.get("type", "unknown").split(".")[-1]
            logging.info(f"Loading dataset {di}: {dataset_name} with {sampler_type}, view_num={view_num}")
        else:
            logging.info(f"Loading dataset {di}: {dataset_name} (no clip_sampler)")
        
        logging.debug(f"Pipeline config:\n{pprint.pformat(pipeline, compact=True)}")
        
        # Instantiate dataset
        dataset = instantiate_from_config(pipeline)
        datasets.append((dataset, dataset_name))
        
        logging.info(f"  Loaded {dataset_name}: {len(dataset)} samples")
    
    return datasets


def run_inference(model, batch, args):
    """Run model inference and return outputs."""
    amp_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    amp_dtype = amp_dtype_map.get(args.amp_dtype, torch.bfloat16)
    
    batch["use_amp"] = args.use_amp
    batch["amp_dtype"] = amp_dtype
    
    # Debug: print batch shape to verify frame count
    if "image" in batch:
        img_shape = batch["image"].shape
        logging.info(f"  Input shape: {img_shape} (B, N_frames, C, H, W)")
    
    with torch.no_grad():
        if args.use_amp:
            with torch.autocast("cuda", enabled=True, dtype=amp_dtype):
                outputs = model.infer(**batch)
        else:
            outputs = model.infer(**batch)
    
    return outputs


def move_outputs_to_cpu(outputs):
    """Move output tensors to CPU for visualization."""
    if outputs is None:
        return None
    
    cpu_outputs = []
    for output in outputs:
        if output is None:
            cpu_outputs.append(None)
            continue
        
        # Create a copy of the output with CPU tensors
        cpu_output = type(output).__new__(type(output))
        for attr_name in dir(output):
            if attr_name.startswith('_'):
                continue
            try:
                attr_value = getattr(output, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(cpu_output, attr_name, attr_value.detach().cpu())
                elif callable(attr_value):
                    continue  # Skip methods
                else:
                    setattr(cpu_output, attr_name, attr_value)
            except Exception:
                pass
        cpu_outputs.append(cpu_output)
    
    return cpu_outputs


def save_motion_visualizations(model, outputs, meta_data, output_dir, dataset_name, data_idx):
    """Save motion visualizations (2D and 3D)."""
    try:
        from hAlgorithm.modules.pipelines2.utils.visualize_motion import (
            vis_motion_head_results,
            vis_motion_3d_rerun,
        )
    except ImportError:
        logging.warning("Motion visualization modules not found, skipping visualization")
        return
    
    # Check if outputs have scene_flow
    if not outputs or not hasattr(outputs[0], 'scene_flow_pred') or outputs[0].scene_flow_pred is None:
        logging.debug("No scene_flow_pred found in outputs, skipping motion visualization")
        return
    
    # Create motion output directories
    motion_out_dir = os.path.join(output_dir, dataset_name, f"motion/{data_idx:06d}")
    os.makedirs(motion_out_dir, exist_ok=True)
    
    cfg = model.save_output_cfg
    
    # 2D Motion Visualization
    if cfg.get("save_motion_results", True):
        logging.debug(f"Saving 2D motion visualizations to {motion_out_dir}")
        try:
            vis_motion_head_results(
                cfg=cfg,
                mv_outputs=outputs,
                motion_out_dir=motion_out_dir,
                data_idx=data_idx,
                meta_data=meta_data,
            )
        except Exception as e:
            logging.error(f"Error in 2D motion visualization: {e}")
            import traceback
            traceback.print_exc()
    
    # 3D Rerun Visualization
    if cfg.get("save_motion_3d", True):
        logging.debug(f"Saving 3D motion visualization (Rerun)")
        try:
            vis_motion_3d_rerun(
                cfg=cfg,
                mv_outputs=outputs,
                out_dir=os.path.join(output_dir, dataset_name),
                data_idx=data_idx,
                meta_data=meta_data,
            )
        except Exception as e:
            logging.error(f"Error in 3D motion visualization: {e}")
            import traceback
            traceback.print_exc()


def move_batch_to_device(batch, device):
    """Recursively move batch tensors to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    else:
        return batch


def main():
    args, unknown_args = parse_args()
    
    # Load config
    cfg = file2dict(args.config)
    config_merge_args(cfg, unknown_args)
    
    # Update save_output_cfg
    save_cfg = cfg.get("model", {}).get("save_output_cfg", {})
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
    logging.info(f"YAML config: {args.yaml}")
    
    # Load datasets
    datasets = load_datasets_from_yaml(args.yaml, cfg, args.select_dataset, args.use_yaml_sampler)
    
    if not datasets:
        logging.error("No datasets loaded!")
        return
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Process each dataset
    for dataset, dataset_name in datasets:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing dataset: {dataset_name}")
        logging.info(f"{'='*60}")
        
        # Create dataset output directory
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=default_collate,
        )
        
        # Limit samples if specified
        num_samples = len(dataloader)
        if args.nums is not None:
            num_samples = min(num_samples, args.nums)
        
        logging.info(f"Processing {num_samples} samples from {dataset_name}")
        
        if num_samples == 0:
            logging.warning(f"No samples to process for {dataset_name}")
            continue
        
        # Process samples
        processed_count = 0
        for data_idx, batch in enumerate(tqdm(dataloader, total=num_samples, desc=dataset_name)):
            if data_idx >= num_samples:
                break
            
            # Move batch to device
            batch = move_batch_to_device(batch, device)
            
            # Run inference
            try:
                outputs = run_inference(model, batch, args)
                
                # Move outputs to CPU for visualization
                outputs = move_outputs_to_cpu(outputs)
            except Exception as e:
                logging.error(f"Error processing sample {data_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Get meta_data
            meta_data = batch.get("meta_data", {})
            
            # Save visualizations
            save_motion_visualizations(
                model=model,
                outputs=outputs,
                meta_data=meta_data,
                output_dir=args.output_dir,
                dataset_name=dataset_name,
                data_idx=data_idx,
            )
            
            processed_count += 1
            if processed_count % 10 == 0:
                logging.info(f"Processed {processed_count}/{num_samples} samples")
        
        logging.info(f"Completed {dataset_name}: {processed_count} samples processed")
    
    logging.info(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()