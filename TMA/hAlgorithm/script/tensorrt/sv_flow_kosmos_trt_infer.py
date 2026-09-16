import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import json
import logging
import math
import pprint
import time

import cv2
import numpy as np
import tensorrt as trt
import torch
from accelerate.utils import set_seed
from tqdm import tqdm

from hAlgorithm.datasets import prepare_data_loaders
from hAlgorithm.modules.pipelines.outputs import DepthOutput, ReconstructOutput
from hAlgorithm.utils import (
    config_logging,
    dict_to_file,
    file2dict,
    instantiate_from_config,
    parse_unknown,
    config_merge_args
)

# from hAlgorithm.script.tensorrt.utils.trt_utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Path to engine file.",
    )
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
        "--test_data",
        type=str,
        default=None,
        help="Path to test dataset file.",
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )

    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown(unknown_args)

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    args.output_dir = os.path.join(args.output_dir, "tensorrt_inference", config_name + "_" + now)

    if args.save_outputs:
        args.output_dir = os.path.realpath(args.output_dir)

    return args, unknown_args

def build_engine(
    onnx_file_path: str,
    engine_file_path: str,
    fp16_mode: bool = False,
    max_workspace_size: int = 4 << 30,  # 4 GB
):
    """
    从 ONNX 构建 TensorRT 引擎（支持动态 shape + FP16）
    
    Args:
        onnx_file_path: ONNX 模型路径
        engine_file_path: 输出引擎路径
        fp16_mode: 是否启用 FP16
        max_workspace_size: Workspace 大小（bytes）
    """
    # 日志设置（VERBOSE 可帮助调试）
    logger = trt.Logger(trt.Logger.WARNING)  # 或 trt.Logger.VERBOSE
    
    # 创建构建器
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # 设置 workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    
    # 启用 FP16（如果支持）
    if fp16_mode:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)  # 安全模式
            logger.log(trt.Logger.INFO, "FP16 enabled")
        else:
            logger.log(trt.Logger.WARNING, "FP16 not supported on this platform")
            fp16_mode = False

    # 解析 ONNX
    parser = trt.OnnxParser(network, logger)
    with open(onnx_file_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(i).desc())
            raise RuntimeError("Failed to parse ONNX file")

    # # ==============================
    # # 设置动态 shape（根据你的模型调整！）
    # # ==============================
    # profile = builder.create_optimization_profile()
    
    # # 假设你的模型支持的尺寸范围（请根据实际需求调整 min/opt/max）
    # # 原始比例: image 840x840 -> prompt_depth 476x476 (ratio ≈ 1.7647)
    # profile.set_shape(
    #     "image",
    #     min=(1, 3, 480, 640),      # 最小输入
    #     opt=(1, 3, 840, 840),      # 最常用输入（性能优化点）
    #     max=(1, 3, 1080, 1920)     # 最大输入
    # )
    # profile.set_shape(
    #     "prompt_depth",
    #     min=(1, 3, 272, 364),      # 480/1.7647≈272, 640/1.7647≈364
    #     opt=(1, 3, 476, 476),      # 840/1.7647=476
    #     max=(1, 3, 612, 1088)      # 1080/1.7647≈612, 1920/1.7647≈1088
    # )
    # profile.set_shape(
    #     "prompt_scale",
    #     min=(1, 1, 1, 1),
    #     opt=(1, 1, 1, 1),
    #     max=(1, 1, 1, 1)
    # )
    
    # config.add_optimization_profile(profile)

    # # ==============================
    # # 【可选】强制敏感层使用 FP32（防止 NaN）
    # # 如果 FP16 模式下仍出现 NaN，取消注释以下代码
    # # ==============================
    # if fp16_mode:
    #     for i in range(network.num_layers):
    #         layer = network.get_layer(i)
    #         # 对 ElementWise (Add, Div 等) 和 Reduce (Softmax 等) 层强制 FP32
    #         if layer.type in [trt.LayerType.ELEMENTWISE, trt.LayerType.REDUCE]:
    #             layer.precision = trt.float32
    #             for j in range(layer.num_outputs):
    #                 layer.set_output_type(j, trt.float32)

    # 构建序列化引擎
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # 保存引擎
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
    
    print(f"✅ Engine saved to: {engine_file_path}")
    print(f"   FP16: {'Enabled' if fp16_mode else 'Disabled'}")
    logging.info(f"Build Engine From: {onnx_file_path}, save engine to {engine_file_path}")
    return engine_file_path

def load_engine(engine_file):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_file, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f"Unsupported TRT dtype: {dtype}")

# 全局缓存
global_context = None
global_output_tensors = {}

def init_trt_engine(engine):
    global global_context
    global_context = engine.create_execution_context()
    logging.info("TRT context initialized once.")

def infer_with_torch(engine, input_tensors_dict):
    global global_context, global_output_tensors
    """
    input_tensors_dict: dict like {'image': tensor1, 'prompt_depth': tensor2, ...}
    Returns: dict of output tensors
    """
    image_shape = input_tensors_dict["image"].shape  # [B, 3, H, W]
    B, _, H, W = image_shape

    # ✅ 预定义输出 shape（根据 image 的 H/W）
    output_shapes = {
        "pred_local_depth": (B, 1, H, W),
        "pred_local_conf": (B, 1, H, W),
        "pred_local_normal": (B, 3, H, W),
        "pred_local_invalid_mask": (B, 1, H, W),
        "update_state": (B, 768, 768),
    }
    # 1. 设置输入张量地址
    for name, tensor in input_tensors_dict.items():
        global_context.set_input_shape(name, tensor.shape)
        global_context.set_tensor_address(name, tensor.data_ptr())

    # 2. 为输出张量分配内存（使用预计算 shape）
    output_tensors = {}
    for name, shape in output_shapes.items():
        dtype = torch_dtype_from_trt(engine.get_tensor_dtype(name))
        # output_tensor = input_tensors_dict["image"].new_zeros(shape)#.to(dtype)
        if name not in global_output_tensors:
            output_tensor = torch.empty(shape, dtype=dtype, device='cuda')
            global_output_tensors[name] = output_tensor
        else:
            output_tensor = global_output_tensors[name]
        global_context.set_tensor_address(name, output_tensor.data_ptr())
        output_tensors[name] = output_tensor
    # 3. 执行推理
    stream = torch.cuda.Stream()
    global_context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return output_tensors

def np_depth_to_point(depth, K):
    B, C, H, W = depth.shape
    grid_x, grid_y = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5, indexing="xy")
    points = (
        np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=0).reshape(3, -1).astype(np.float32)
    )
    rays_d = np.linalg.inv(K) @ points  # (3, HW)
    pts = depth.reshape(B, C, -1) * rays_d  # (B, 3, HW)
    pc = pts.reshape(B, 3, H, W)
    return pc

def filter_pointmap(pointmap_pred, confidence_pred, image, conf_thresh):
    pointmap_color = image.float().squeeze(0).numpy().transpose(1, 2, 0)
    pointmap_color = pointmap_color.reshape(-1, 3)
    filtered_pointmap = pointmap_pred[confidence_pred.reshape(-1) > conf_thresh]
    filtered_pointmap_color = pointmap_color[confidence_pred.reshape(-1) > conf_thresh]
    return filtered_pointmap, filtered_pointmap_color

def main():
    args, unknown_args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config_logging(out_dir=args.output_dir)

    # Set the seed now.
    logging.info(f"seed: {args.seed}")
    set_seed(args.seed)

    # Load configuration
    cfg = file2dict(args.config)
    config_merge_args(cfg, unknown_args)
    
    if args.test_data is not None:
        cfg["data"]["val"] = args.test_data

    cfg["model"]["seed"] = args.seed

    # Initialize datasets
    val_datasets, val_dataloaders, val_nums = prepare_data_loaders("val", cfg)
    cfg_str = pprint.pformat(cfg, compact=True)
    logging.info("config:\n" + cfg_str + "\n")
    logging.info("")

    save_out_dir = os.path.join(args.output_dir, "outputs")
    os.makedirs(save_out_dir, exist_ok=True)

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
    model.eval()
    use_fp16 = True
    # -------------------- Running training --------------------
    # Build model
    engine = None
    if args.engine is not None:
        engine = load_engine(args.engine)
        if engine is None:
            logging.info(f"Loading Engine From: {args.engine} Failed. Try to rebuild")
        else:
            logging.info(f"Loading Engine From: {args.engine}")
    if engine is None:
        assert args.onnx is not None, "Onnx file or Engine File Must be Provided"
        engine_path = os.path.join(
            args.output_dir, f"{os.path.splitext(os.path.basename(args.config))[0]}.engine"
        )
        build_engine(args.onnx, engine_path, fp16_mode=use_fp16)
        engine = load_engine(engine_path)
    init_trt_engine(engine)
    model.device = "cpu"
    save_meta_dict = dict()
    # 推理循环
    for dataloader in val_dataloaders[:1]:
        dataset_name = dataloader.dataset.name if hasattr(dataloader.dataset, "name") else "unnamed"
        vis_dir = os.path.join(save_out_dir, dataset_name)
        os.makedirs(vis_dir, exist_ok=True)
        average_time = 0
        _count = 0
        view_states = {}
        for i, batch in enumerate(tqdm(dataloader)):
            # 准备输入数据
            image = batch["image"].clone()

            intrinsics = batch.get("intrinsics", None)  # [n, 3, 3]

            prompt_depth = batch[model.prompt_depth_name]
            if args.debug:
                import open3d as o3d
                points = prompt_depth.clone().numpy().reshape(3, -1).T  # shape: (H*W, 3)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(f"{vis_dir}/prompt_{i:04d}.ply", pcd) 
                
            prompt_scale = batch[model.scale_name][:, None, None, None]
            view_id = int(batch["meta_data"]["data_info"][0]["view_id"])
            if view_id not in view_states:
                flow_state = model.get_init_param(image)
                logging.info(f"View {view_id} Init State")
            else:
                flow_state = view_states[view_id]

            dtype = torch.float16 if use_fp16 else torch.float32
            inputs = {
                "image": image.to(device='cuda',dtype=dtype),
                "prompt_depth": prompt_depth.to(device='cuda',dtype=dtype),
                "prompt_scale": prompt_scale.to(device='cuda',dtype=dtype),
                "flow_state": flow_state.to(device='cuda',dtype=dtype),
            }
            start_time = time.time()
            outputs = infer_with_torch(engine, inputs)
            end_time = time.time()
            # 打印推理时间和结果
            total_time = (end_time - start_time) * 1000  # 转换为毫秒
            logging.info(f"Inference time: {total_time:.2f} ms")
            average_time = (average_time * _count + total_time) / (_count+1)
            _count = _count + 1
            #update state
            view_states[view_id] = outputs["update_state"].clone()
            if args.save_outputs:
                pred_local_depth = outputs["pred_local_depth"].clone().cpu()
                pred_local_conf = outputs["pred_local_conf"].clone().cpu()
                pred_local_normal = outputs["pred_local_normal"].clone().cpu()
                pred_local_invalid_mask = outputs["pred_local_invalid_mask"].clone().cpu()
                from hAlgorithm.modules.pipelines2.utils.save_kosmo import save_kosmo_outputs
                from hAlgorithm.modules.pipelines.outputs import ReconstructOutput
                output = ReconstructOutput(
                    depth_align=pred_local_depth.squeeze().numpy(),
                    normal=pred_local_normal.squeeze().numpy().transpose(1, 2, 0),
                    confidence=pred_local_conf.squeeze().numpy(),
                    invalid_mask=pred_local_invalid_mask.squeeze().numpy(),
                )
                save_kosmo_outputs({}, output, batch["meta_data"], out_dir=vis_dir, output_meta_dict=save_meta_dict)

            if args.debug:
                pred_local_depth = outputs["pred_local_depth"].clone().cpu()
                import open3d as o3d
                if pred_local_depth is not None:
                    pred_local_depth = model.depth_to_points(pred_local_depth, K=intrinsics, device="cpu")
                points = pred_local_depth.clone().numpy().reshape(3, -1).T  # shape: (H*W, 3)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(f"{vis_dir}/output_local_{i:04d}.ply", pcd)
                breakpoint()

        logging.info(f"Average Inference time on {dataset_name}: {average_time:.2f} ms")
        output_json_path = os.path.join(vis_dir, "data_info_with_depth.json")
        with open(output_json_path, "w") as f:
            json.dump(save_meta_dict, f, indent=2, ensure_ascii=False)
        logging.info(f"output meta save: {output_json_path}")

if __name__ == "__main__":
    main()
