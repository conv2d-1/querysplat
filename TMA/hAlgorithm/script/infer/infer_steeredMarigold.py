# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import argparse
import json
import logging
import os
import shutil
from datetime import datetime

import numpy as np
import torch
from tqdm.auto import tqdm

from hAlgorithm.datasets import prepare_data_loaders
from hAlgorithm.modules.pipelines.steered_marigold_pipeline import SteeredMarigoldPipeline
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


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hAlgorithm/configs/marigold/infer_depth_steeredMariglod_sparse.py",
        help="Path to the meta info",
    )
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory.")
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )
    parser.add_argument("--seed", default=2024, type=int, help="random seed")

    args = parser.parse_args()
    cfg = file2dict(args.config)
    denoise_steps = cfg.get("model", {}).get("denoising_steps")
    if denoise_steps is None:
        denoise_steps = 50
    processing_res = cfg.get("model", {}).get("processing_resolution")
    if processing_res is None:
        processing_res = 768
    ckpt = cfg.get("model", {}).get("pretrained_model_name_or_path")
    output_dir = os.path.join(
        os.path.abspath(args.output_dir),
        os.path.splitext(os.path.basename(args.config))[0],
        f'{datetime.now().strftime("%Y%m%d-%H%M%S")}',
    )
    os.makedirs(output_dir, exist_ok=True)
    assert ckpt is not None

    ensemble_size = 1
    batch_size = 1
    half_precision = args.half_precision
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method
    seed = args.seed

    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    _, data_loaders, _ = prepare_data_loaders("val", cfg=cfg, seed=2024)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(f"Running with half precision ({dtype}), might lead to suboptimal result.")
    else:
        dtype = torch.float32
        variant = None

    pipe: SteeredMarigoldPipeline = SteeredMarigoldPipeline.from_pretrained(
        ckpt, variant=variant, torch_dtype=dtype
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

    pipe = pipe.to(device)
    logging.info(
        f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}"
    )

    # Print out config;
    logging.info(
        f"Inference settings: checkpoint = `{ckpt}`, "
        f"with denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}"
    )

    # -------------------- Inference and saving --------------------

    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    for data_loader in tqdm(data_loaders):
        for batch_idx, batch in tqdm(enumerate(data_loader)):
            image = batch["image"]
            sparse_depth = batch["sparse_depth"]
            pipe_out = pipe(
                image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
                sparse_depth=sparse_depth,
            )
            pred = pipe_out.depth

            visualize(pred, batch["meta_data"], output_dir)
            save_output(pred, batch["meta_data"], output_dir)
