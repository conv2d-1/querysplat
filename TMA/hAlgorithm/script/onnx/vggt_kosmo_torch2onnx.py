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
        "--fp16",
        action="store_true",
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

def register_fp32_norm_hook_for_single_layer(model, target_module_path: str):
    """
    仅为指定路径的 LayerNorm 层注册 FP32 输入 / FP16 输出的 hook
    
    Args:
        model: PyTorch 模型
        target_module_path: 模块的完整路径名，如 "model.rgb_encoder.dinov2.norm"
    
    Returns:
        list: 注册的 hook 句柄（可用于后续移除）
    """
    # 1. 根据路径找到目标模块
    target_module = None
    for name, module in model.named_modules():
        if name == target_module_path:
            if not isinstance(module, torch.nn.LayerNorm):
                raise TypeError(f"Module at '{name}' is not a LayerNorm, but {type(module)}")
            target_module = module
            break
    
    if target_module is None:
        raise ValueError(f"LayerNorm module not found at path: '{target_module_path}'")

    # 2. 将该 LayerNorm 的参数转为 FP32
    target_module.float()
    print(f"Converted {target_module_path} parameters to FP32")

    # 3. 定义 pre-hook: 输入转 FP32
    def pre_hook(module, inputs):
        return tuple(
            inp.float() if inp.is_floating_point() else inp
            for inp in inputs
        )

    # 4. 定义 forward-hook: 输出转回 FP16
    def fwd_hook(module, inputs, output):
        if output.is_floating_point():
            return output.half()
        return output

    # 5. 注册 hooks
    pre_handle = target_module.register_forward_pre_hook(pre_hook)
    fwd_handle = target_module.register_forward_hook(fwd_hook)

    print(f"Registered FP32<->FP16 hooks for: {target_module_path}")
    return [pre_handle, fwd_handle]

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
        
    # deal with special tokens
    _prepare_extra_input = model.model.fuse_encoder._onnx_prepare_extra_input
    

    if args.onnx is None:
        # NOTE replace attn func
        from hAlgorithm.modules.models.facebookresearch_dinov2_main.dinov2.layers.attention import (
            Attention,
        )

        for block in model.model.rgb_encoder.dinov2.blocks:
            block.attn.__class__ = Attention
        
        for block in model.model.fuse_encoder.frame_blocks:
            block.attn.fused_attn = False
        
        for block in model.model.fuse_encoder.global_blocks:
            block.attn.fused_attn = False
        
        # NOTE replace forward
        model.forward = model.trans_onnx_vggt

        # NOTE load pretrain
        model.load_checkpoint(args.load_from)

        model.eval()
        model = model.cuda()
        if args.fp16:
            model = model.half()
            hooks = register_fp32_norm_hook_for_single_layer(
                model, 
                target_module_path="model.rgb_encoder.dinov2.norm"
            )
        
        onnx_path = os.path.join(
            args.output_dir, f"{os.path.splitext(os.path.basename(args.config))[0]}.onnx"
        )
        for i, batch in enumerate(val_dataloaders[0]):
            image = batch["image"].cuda()[:, 0]
            image = torch.nn.functional.interpolate(image, size=(840,840))
            h, w = image.shape[-2:]
            patch_h, patch_w = (
                h // model.model.rgb_encoder.patch_size,
                w // model.model.rgb_encoder.patch_size,
            )
            setattr(model, "patch_h", patch_h)
            setattr(model, "patch_w", patch_w)
            if args.fp16:
                image = image.half()
            camera_token, register_token, pos = _prepare_extra_input(1, 1, patch_h, patch_w)
            
            input_names = [
                "image",
                "camera_token",
                "register_token",
                "pos",
            ]
            output_names = [
                "pred_local_normal",
            ]
            dynamic_axes = {
                name: {0: "batch"} for name in input_names + output_names
            }
            torch.onnx.export(
                model,  # 模型实例
                (
                    image,
                    camera_token,
                    register_token,
                    pos,
                ),  # 输入张量
                onnx_path,  # 输出文件路径
                input_names=input_names,  # 输入张量的名称
                output_names=output_names,  # 输出张量的名称
                dynamic_axes=dynamic_axes,
                opset_version=17,  # ONNX 操作集版本，通常选择 11 或更高
                export_params=True,  # 导出模型参数
                do_constant_folding=True,  # 启用常量折叠优化
            )
            overwrite_input_shapes = {
                "image": [1, 3, 840, 840],
                "camera_token": [1, 1, 1024], # B,1,P
                "register_token": [1, 4, 1024], # B,1,P
                "pos": [1, 3605, 2], # B,P,2
            }
            
            onnx_model = onnx.load(onnx_path)  # load onnx model
            model_simp, check = simplify(onnx_model, overwrite_input_shapes=overwrite_input_shapes)
            onnx.save(model_simp, onnx_path)
            logging.info(f"save onnx: {onnx_path}")
            break
    else:
        onnx_path = args.onnx
        model.load_checkpoint(args.load_from)
        model.eval()
        if args.fp16:
            model = model.half()
    model = model.cpu()

    if args.test:
        provider_options = [{"device_id": 0}, {}]  # 使用 GPU 0
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(
            onnx_path, providers=providers, provider_options=provider_options
        )

        eval_metrics = instantiate_from_config(cfg["trainer"]["eval_metrics"])

        eval_results = {}
        frame_eval_results = []
        extra_input = {}

        for i, batch in enumerate(tqdm(val_dataloaders[0])):
            image = batch["image"][:, 0]
            image = torch.nn.functional.interpolate(image, size=(840, 840))
            h, w = image.shape[-2:]
            patch_h, patch_w = (
                h // model.model.rgb_encoder.patch_size,
                w // model.model.rgb_encoder.patch_size,
            )
            setattr(model, "patch_h", patch_h)
            setattr(model, "patch_w", patch_w)
            if args.fp16:
                image = image.half()
            
            if "camera_token" in extra_input:
                camera_token = extra_input["camera_token"]
                register_token = extra_input["register_token"]
                pos = extra_input["pos"]
            else:
                camera_token, register_token, pos = _prepare_extra_input(1, 1, patch_h, patch_w)
                extra_input["camera_token"] = camera_token.detach().cpu().clone()
                extra_input["register_token"] = register_token.detach().cpu().clone()
                extra_input["pos"] = pos.detach().cpu().clone()

            inputs = {
                "image": image.detach().cpu().numpy(),
                "camera_token": camera_token.detach().cpu().numpy(),
                "register_token": register_token.detach().cpu().numpy(),
                "pos": pos.detach().cpu().numpy(),
            }
            intrinsics = batch.get("intrinsics", None)  # [n, 3, 3]

            out = session.run(None, inputs)
            normal = out[0]
            
            depth_gt = batch[model.target_local_depth_name].squeeze().numpy()
            h, w = depth_gt.shape[-2:]
            normal = torch.nn.functional.interpolate(
                torch.as_tensor(normal), size=(h, w), mode='bilinear'
            )
            normal = torch.nn.functional.normalize(normal, dim=-3).float().numpy()
            
            # Optionally convert the ground truth pointmap to a NumPy array.
            pointmap_gt = batch.get(model.target_local_depth_name, None)
            if pointmap_gt is not None:
                pointmap_gt = (
                    pointmap_gt.cpu().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)
                )

            # Convert the color pointmap to a NumPy array and normalize it to the [0, 255] range.
            pointmap_color = (
                batch["image"][:, 0].clone().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)
            )
            pointmap_color = (pointmap_color + 1) * 0.5 * 255

            output = DepthOutput(
                normal=normal.squeeze(0).transpose(1, 2, 0),
                pointmap_gt=pointmap_gt,
                pointmap_color=pointmap_color,
                intrinsics=intrinsics,
                pointmap_h=h,
                pointmap_w=w,
            )

            if eval_metrics is not None:
                frame_eval_results.append(eval_metrics(batch, [output]))

            if args.test_vis:
                output.pointmap_gt = None
                model.visualize([output], batch["meta_data"], out_dir=save_out_dir)

            if args.save_outputs:
                model.save_output([output], batch["meta_data"], out_dir=save_out_dir)


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
