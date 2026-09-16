import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import json
import logging
import math
import pprint

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from accelerate.utils import set_seed
from onnxsim import simplify
from tqdm import tqdm

from hAlgorithm.datasets import prepare_data_loaders
from hAlgorithm.modules.pipelines.outputs import DepthOutput
from hAlgorithm.modules.utils.alignment import align_depth_least_square
from hAlgorithm.utils import (
    config_logging,
    dict_to_file,
    file2dict,
    instantiate_from_config,
)
from hAlgorithm.utils.util import eval_dict_to_text


def parse_args():
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Path to onnx file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file.",
    )
    parser.add_argument(
        "--load_from",
        default=None,
        help="Path of checkpoint to be load.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--test",
        action="store_true",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test dataset file.",
    )
    parser.add_argument(
        "--test_vis",
        action="store_true",
        help="Show test results.",
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug.",
    )

    args = parser.parse_args()

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    args.output_dir = os.path.join(args.output_dir, "torch2onnx", config_name + "_" + now)

    if args.save_outputs:
        args.output_dir = os.path.realpath(args.output_dir)

    if args.load_from in ["None", "none", "null"]:
        args.load_from = None

    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config_logging(out_dir=args.output_dir)

    # Set the seed now.
    logging.info(f"seed: {args.seed}")
    set_seed(args.seed)

    # Load configuration
    cfg = file2dict(args.config)
    if args.test and args.test_data is not None:
        cfg["data"]["val"] = args.test_data

    cfg["model"]["seed"] = args.seed

    # Initialize datasets
    val_datasets, val_dataloaders, val_nums = prepare_data_loaders("val", cfg)
    cfg_str = pprint.pformat(cfg, compact=True)
    logging.info("config:\n" + cfg_str + "\n")
    logging.info("")

    save_out_dir = os.path.join(args.output_dir, "visualization")
    os.makedirs(save_out_dir, exist_ok=True)

    if args.test:
        # Initialize directories
        eval_dir = os.path.join(args.output_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        logging.info("***** Running test *****")
        logging.info(f"  Num test examples = {sum([len(dataset) for dataset in val_datasets])}")
        logging.info(f"  logging_dir = {args.output_dir}")
        logging.info("")

    # -------------------- Snapshot of config --------------------
    cfg_save_path = os.path.join(args.output_dir, os.path.basename(args.config))
    assert cfg_save_path != args.config
    dict_to_file(cfg, cfg_save_path)

    # Initialize model
    model_cfg = cfg["model"].copy()
    model = instantiate_from_config(model_cfg)
    assert model is not None

    if args.onnx is None:
        # NOTE replave atten
        from hAlgorithm.modules.models.facebookresearch_dinov2_main.dinov2.layers.attention import (
            Attention,
        )

        for block in model.model.rgb_encoder.dinov2.blocks:
            block.attn.__class__ = Attention

        # NOTE replave forward
        model.forward = model.trans_onnx

        # NOTE load pretrain
        model.load_checkpoint(args.load_from)

        model.eval()
        model = model.cuda()

        onnx_path = os.path.join(
            args.output_dir, f"{os.path.splitext(os.path.basename(args.config))[0]}.onnx"
        )
        for i, batch in enumerate(val_dataloaders[0]):
            image = batch["image"].cuda()
            prompt_depth = batch[model.prompt_name].cuda()
            prompt_scale = batch[model.prompt_scale_name][:, None, None, None].cuda()

            h, w = image.shape[-2:]
            patch_h, patch_w = (
                h // model.model.rgb_encoder.patch_size,
                w // model.model.rgb_encoder.patch_size,
            )
            setattr(model, "patch_h", patch_h)
            setattr(model, "patch_w", patch_w)

            if model.prompt_set_none:
                input_names = ["image"]
                output_names = ["pointmap", "confidence"]

                torch.onnx.export(
                    model,  # 模型实例
                    (image),  # 输入张量
                    onnx_path,  # 输出文件路径
                    input_names=input_names,  # 输入张量的名称
                    output_names=output_names,  # 输出张量的名称
                    opset_version=11,  # ONNX 操作集版本，通常选择 11 或更高
                    export_params=True,  # 导出模型参数
                    do_constant_folding=True,  # 启用常量折叠优化
                )
            else:
                input_names = [
                    "image",
                    "prompt_depth",
                    "prompt_scale",
                ]
                output_names = ["pointmap", "confidence"]

                torch.onnx.export(
                    model,  # 模型实例
                    (
                        image,
                        prompt_depth,
                        prompt_scale,
                    ),  # 输入张量
                    onnx_path,  # 输出文件路径
                    input_names=input_names,  # 输入张量的名称
                    output_names=output_names,  # 输出张量的名称
                    opset_version=11,  # ONNX 操作集版本，通常选择 11 或更高
                    export_params=True,  # 导出模型参数
                    do_constant_folding=True,  # 启用常量折叠优化
                )

            onnx_model = onnx.load(onnx_path)  # load onnx model
            model_simp, check = simplify(onnx_model)
            onnx.save(model_simp, onnx_path)
            logging.info(f"save onnx: {onnx_path}")

            break

        model = model.cpu()
    else:
        onnx_path = args.onnx

    if args.test:
        provider_options = [{"device_id": 0}]  # 使用 GPU 0
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(
            onnx_path, providers=providers, provider_options=provider_options
        )

        eval_metrics = instantiate_from_config(cfg["trainer"]["eval_metrics"])

        eval_results = {}
        frame_eval_results = []

        for i, batch in enumerate(tqdm(val_dataloaders[0])):
            image = batch["image"].clone()
            image = ((image + 1) * 0.5 - model._mean) / model._std

            intrinsics = batch.get("intrinsics", None)  # [n, 3, 3]

            prompt_depth = batch[model.prompt_name]
            prompt_scale = batch[model.prompt_scale_name][:, None, None, None]

            if model.prompt_set_none:
                inputs = {
                    "image": image.numpy(),
                }
                pointmap_pred = session.run(None, inputs)[0].squeeze()
            else:
                inputs = {
                    "image": image.numpy(),
                    "prompt_depth": prompt_depth.numpy(),
                    "prompt_scale": prompt_scale.numpy(),
                }
                pointmap_pred = session.run(None, inputs)[0].squeeze()

            # Convert the predicted pointmap to a NumPy array and extract the depth channel.
            pointmap_pred = pointmap_pred.reshape(3, -1).transpose(1, 0)
            depth = pointmap_pred[:, 2].reshape((image.shape[-2], image.shape[-1])).clip(1e-3)

            # If matching input resolution is required, resize the predicted pointmap and color pointmap to match the ground truth depth.
            if model.match_input_res:
                depth_gt = batch[model.align_name].squeeze().numpy()
                h, w = depth_gt.shape
                depth = cv2.resize(
                    depth,
                    dsize=(w, h),
                    interpolation=cv2.INTER_LINEAR,
                )

                if model.post_align:
                    depth_gt_valid_mask = batch[model.align_mask_name].squeeze().numpy()
                    depth = align_depth_least_square(
                        gt_arr=depth_gt,
                        pred_arr=depth,
                        valid_mask_arr=depth_gt_valid_mask,
                        return_scale_shift=False,
                        max_resolution=None,
                    )

            # Optionally convert the ground truth pointmap to a NumPy array.
            pointmap_gt = batch.get(model.target_name, None)
            if pointmap_gt is not None:
                pointmap_gt = (
                    pointmap_gt.cpu().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)
                )

            # Convert the color pointmap to a NumPy array and normalize it to the [0, 255] range.
            pointmap_color = (
                batch["image"].clone().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)
            )
            pointmap_color = (pointmap_color + 1) * 0.5 * 255

            output = DepthOutput(
                depth_align=depth,
                pointmap=pointmap_pred,
                pointmap_gt=pointmap_gt,
                pointmap_color=pointmap_color,
                intrinsics=intrinsics,
            )

            if args.test_vis:
                model.visualize(output, batch["meta_data"], out_dir=save_out_dir)

            if args.save_outputs:
                model.save_output(output, batch["meta_data"], out_dir=save_out_dir)

            if eval_metrics is not None:
                frame_eval_results.append(eval_metrics(batch, output))

            if args.debug:
                break

        dataset_name = (
            val_dataloaders[0].dataset.name
            if hasattr(val_dataloaders[0].dataset, "name")
            else "unnamed"
        )
        if isinstance(eval_metrics.metrics[0], str):
            for metric_name in eval_metrics.metrics:
                eval_results[metric_name] = sum(
                    result[metric_name].item() for result in frame_eval_results
                ) / len(frame_eval_results)
        else:
            for metric_obj in eval_metrics.metrics:
                for metric_name in metric_obj.metrics:
                    eval_results[metric_name] = sum(
                        result[metric_name].item() for result in frame_eval_results
                    ) / len(frame_eval_results)

        logging.info(f"Evaluation results on {dataset_name}: {eval_results}")

        # Save evaluation results to a text file
        eval_text = eval_dict_to_text(val_metrics=eval_results, dataset_name=dataset_name)
        eval_text_save_path = os.path.join(
            eval_dir,
            f"eval-{dataset_name}.txt",
        )
        with open(eval_text_save_path, "w+") as f:
            f.write(eval_text)

        logging.debug(f"Evaluation results on {dataset_name}: {eval_text}")
        logging.info(f"Saved evaluation results to: {eval_text_save_path}")


if __name__ == "__main__":
    main()
