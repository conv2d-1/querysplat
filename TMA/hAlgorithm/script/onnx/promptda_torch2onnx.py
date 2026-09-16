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
        help="Path to test results.",
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

        for block in model.pretrained.blocks:
            block.attn.__class__ = Attention

        # NOTE replave forward
        model.forward = model.trans_onnx

        # NOTE load pretrain
        model.load_checkpoint(args.load_from)

        model.eval()
        model = model.cuda()

        onnx_path = os.path.join(args.output_dir, "onnx.pth")
        for i, batch in enumerate(val_dataloaders[0]):
            image = batch["image"].cuda()
            prompt_depth = batch[model.prompt_depth_name].cuda()
            prompt_depth_min = batch[model.prompt_depth_min][:, None, None, None].cuda()
            prompt_depth_max = batch[model.prompt_depth_max][:, None, None, None].cuda()

            h, w = image.shape[-2:]
            patch_h, patch_w = h // model.patch_size, w // model.patch_size

            # prompt_depth = model.normalize(prompt_depth, prompt_depth_min, prompt_depth_max)
            # image = ((image + 1) * 0.5 - model._mean) / model._std

            input_names = [
                "image",
                "prompt_depth",
                "prompt_depth_min",
                "prompt_depth_max",
                "patch_h",
                "patch_w",
            ]
            output_names = ["depth"]

            torch.onnx.export(
                model,  # 模型实例
                (
                    image,
                    prompt_depth,
                    prompt_depth_min,
                    prompt_depth_max,
                    patch_h,
                    patch_w,
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
            image = batch["image"]
            image = ((image + 1) * 0.5 - model._mean) / model._std

            prompt_depth = batch[model.prompt_depth_name]
            prompt_depth_min = batch[model.prompt_depth_min][:, None, None, None]
            prompt_depth_max = batch[model.prompt_depth_max][:, None, None, None]

            h, w = image.shape[-2:]
            patch_h, patch_w = h // model.patch_size, w // model.patch_size

            inputs = {
                "image": image.numpy(),
                "prompt_depth": prompt_depth.numpy(),
                "prompt_depth_min": prompt_depth_min.numpy(),
                "prompt_depth_max": prompt_depth_max.numpy(),
                "patch_h": np.array([patch_h]).astype(int),
                "patch_w": np.array([patch_w]).astype(int),
            }
            depth_pred = session.run(None, inputs)[0].squeeze()

            if model.match_input_res or model.post_align:
                depth_gt = batch[model.align_depth_name]
                depth_gt = batch[model.align_depth_name].squeeze().numpy()

            if model.match_input_res:
                h, w = depth_gt.shape
                depth_pred = cv2.resize(
                    depth_pred,
                    dsize=(w, h),
                    interpolation=cv2.INTER_LINEAR,
                )

            depth_pred = DepthOutput(depth_align=depth_pred)

            model.visualize(depth_pred, batch["meta_data"], out_dir=save_out_dir)

            if eval_metrics is not None:
                frame_eval_results.append(eval_metrics(batch, depth_pred))

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
