#!/usr/bin/env python3

import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import logging
import pprint

import torch
from accelerate.utils import set_seed

from hAlgorithm.modules.pipelines.prompt_pointmap_pipeline import PromptPointMapPipeline
from hAlgorithm.utils import (
    config_logging,
    dict_to_file,
    file2dict,
    instantiate_from_config,
)


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_default_output_path(config_path):
    """Generate default output path based on config filename.

    Args:
        config_path (str): Path to the config file

    Returns:
        str: Default output path in results/merge_lora_checkpoint directory
    """
    # Get config filename without extension
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create output path
    output_dir = os.path.join("results", "merge_lora_checkpoint", f"{config_name}_{now}")
    output_path = os.path.join(output_dir, "model_merged.pth")

    return output_dir, output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into the base model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint with LoRA weights"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Directory to save the merged model (default: results/merge_lora_checkpoint/<config_name>_<timestamp>)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for merging weights"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set default output path if not provided
    if args.output is None:
        args.output_dir, args.output_path = get_default_output_path(args.config)
    else:
        args.output_dir = args.output
        args.output_path = os.path.join(args.output, "model_merged.pth")

    # Create output directory and setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    config_logging(out_dir=args.output_dir)

    # Set the seed
    logging.info(f"seed: {args.seed}")
    set_seed(args.seed)

    # Load configuration
    logging.info(f"Loading config from {args.config}")
    cfg = file2dict(args.config)
    cfg["model"]["seed"] = args.seed

    # Log configuration
    cfg_str = pprint.pformat(cfg, compact=True)
    logging.info("config:\n" + cfg_str + "\n")

    # Save config snapshot
    cfg_save_path = os.path.join(args.output_dir, os.path.basename(args.config))
    assert cfg_save_path != args.config
    dict_to_file(cfg, cfg_save_path)

    # Create model
    logging.info("Creating model from config")
    model = instantiate_from_config(cfg["model"])

    if not isinstance(model, PromptPointMapPipeline):
        raise ValueError("Model must be an instance of PromptPointMapPipeline")

    # Load checkpoint
    logging.info(f"Loading checkpoint from {args.checkpoint}")
    model.load_checkpoint(args.checkpoint)

    # Move model to device
    model = model.to(args.device)

    # Count parameters before merging
    total_params_before, trainable_params_before = count_parameters(model)
    logging.info("Parameters before merging LoRA weights:")
    logging.info(f"  Total parameters: {total_params_before:,}")
    logging.info(f"  Trainable parameters: {trainable_params_before:,}")

    # Merge LoRA weights
    logging.info("Merging LoRA weights")
    model.merge_lora_weights()

    # Count parameters after merging
    total_params_after, trainable_params_after = count_parameters(model)
    logging.info("Parameters after merging LoRA weights:")
    logging.info(f"  Total parameters: {total_params_after:,}")
    logging.info(f"  Trainable parameters: {trainable_params_after:,}")

    # Verify the merge
    if total_params_after < total_params_before:
        logging.info("LoRA weights successfully merged: trainable parameters decreased")
    else:
        logging.warning("Warning: trainable parameters did not decrease after merging")

    # Save the merged model
    logging.info(f"Saving merged model to {args.output_path}")
    state_dict = model.state_dict()
    torch.save(state_dict, args.output_path)

    logging.info("LoRA weights have been successfully merged and saved!")


if __name__ == "__main__":
    main()
