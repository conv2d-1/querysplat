import argparse
import json
import logging
import os
import shutil
from datetime import datetime

import diffusers
import numpy as np
import torch
from diffusers import DDIMScheduler
from tqdm import tqdm

from hAlgorithm.datasets import prepare_data_loaders
from hAlgorithm.modules.pipelines.marigold_dc_pipeline import (
    MarigoldDepthCompletionPipeline,
)
from hAlgorithm.utils import colorize_depth_maps, file2dict


def visualize(pred, meta_data, out_dir):
    depth_pred_colored = colorize_depth_maps(
        pred, pred.min(), pred.max(), cmap="turbo"
    )  # [3, H, W], value in (0, 1)
    save_path = os.path.join(out_dir, f"detph_{meta_data['data_idx'][0]:06d}.jpg")
    depth_pred_colored.save(save_path)


def save_output(pred, meta_data, out_dir):
    save_path = os.path.join(out_dir, f"detph_{meta_data['data_idx'][0]:06d}.npy")
    np.save(save_path, pred)

    data_path = meta_data["data_path"][0]
    new_data_path = os.path.join(out_dir, f"data_info_with_depth.json")
    assert new_data_path != data_path
    if not os.path.exists(new_data_path):
        shutil.copy(data_path, new_data_path)

    with open(new_data_path, "r") as f:
        data_info = json.load(f)
        assert data_info["files"][meta_data["data_idx"]]["rgb"] == meta_data["data_info"]["rgb"][0]
        data_info["files"][meta_data["data_idx"]]["pred_depth"] = save_path
        data_info["files"][meta_data["data_idx"]]["depth_scale"] = meta_data["depth_scale"][
            0
        ].item()

    with open(new_data_path, "w") as f:
        json.dump(data_info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Marigold-DC Pipeline")

    parser.add_argument("--output_dir", type=str, default="output", help="Output dense depth")
    parser.add_argument(
        "--config",
        type=str,
        default="hAlgorithm/configs/marigold/infer_depth_marigold_dc_sparse.py",
        help="Config file",
    )
    args = parser.parse_args()

    cfg = file2dict(args.config)
    num_inference_steps = cfg.get("model", {}).get("denoising_steps")
    if num_inference_steps is None:
        num_inference_steps = 50
    processing_resolution = cfg.get("model", {}).get("processing_resolution")
    if processing_resolution is None:
        processing_resolution = 768
    ckpt = cfg.get("model", {}).get("pretrained_model_name_or_path")
    output_dir = os.path.join(
        os.path.abspath(args.output_dir),
        os.path.splitext(os.path.basename(args.config))[0],
        f'{datetime.now().strftime("%Y%m%d-%H%M%S")}',
    )
    os.makedirs(output_dir, exist_ok=True)
    assert ckpt is not None

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        processing_resolution_non_cuda = 512
        num_inference_steps_non_cuda = 10
        if processing_resolution > processing_resolution_non_cuda:
            logging.warning(
                f"CUDA not found: Reducing processing_resolution to {processing_resolution_non_cuda}"
            )
            processing_resolution = processing_resolution_non_cuda
        if num_inference_steps > num_inference_steps_non_cuda:
            logging.warning(
                f"CUDA not found: Reducing num_inference_steps to {num_inference_steps_non_cuda}"
            )
            num_inference_steps = num_inference_steps_non_cuda

    pipe = MarigoldDepthCompletionPipeline.from_pretrained(ckpt, prediction_type="depth").to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    if not torch.cuda.is_available():
        logging.warning("CUDA not found: Using a lightweight VAE")
        del pipe.vae
        pipe.vae = diffusers.AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device)

    _, data_loaders, _ = prepare_data_loaders("val", cfg=cfg, seed=2024)

    for data_loader in tqdm(data_loaders):
        for batch_idx, batch in tqdm(enumerate(data_loader)):
            image = [batch["image"].squeeze().permute(1, 2, 0).numpy() / 255.0]
            sparse_depth = batch["sparse_depth"].squeeze().numpy()
            pred = pipe(
                image=image,
                sparse_depth=sparse_depth,
                num_inference_steps=num_inference_steps,
                processing_resolution=processing_resolution,
            )
            visualize(pred, batch["meta_data"], output_dir)
            save_output(pred, batch["meta_data"], output_dir)


if __name__ == "__main__":
    main()
