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
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
from accelerate.utils import set_seed
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

# from hAlgorithm.script.tensorrt.utils.trt_utils import *



def parse_args():
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained engine file.",
    )
    parser.add_argument(
        "--head",
        type=str,
        default=None,
        help="Path to head engine file.",
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
        "--vis_results",
        action="store_true",
        help="Show test results.",
    )

    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    args.output_dir = os.path.join(args.output_dir, "tensorrt", config_name + "_" + now)

    return args


# 加载 TensorRT engine
def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    with open(engine_file_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def prepare_engine(engine_file_path):
    # 加载 engine
    engine = load_engine(engine_file_path)
    # 创建执行上下文
    context = engine.create_execution_context()
    return engine, context


def prepare_engine_info(engine):
    input_info = {}  # 存储输入的名称、形状和 GPU 内存地址
    output_info = {}  # 存储输出的名称、形状和 GPU 内存地址

    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        print(f"Tensor {idx}: Name={name}, Shape={shape}, Dtype={dtype}, IsInput={is_input}")

        # 分配 GPU 内存
        size = trt.volume(shape) * dtype.itemsize
        device_mem = cuda.mem_alloc(size)

        if is_input:
            input_info[name] = {
                "shape": shape,
                "dtype": dtype,
                "device_mem": device_mem,
            }
        else:
            output_info[name] = {
                "shape": shape,
                "dtype": dtype,
                "device_mem": device_mem,
            }
    return input_info, output_info


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config_logging(out_dir=args.output_dir)

    # Set the seed now.
    logging.info(f"seed: {args.seed}")
    set_seed(args.seed)

    # Load configuration
    cfg = file2dict(args.config)
    if args.test_data is not None:
        cfg["data"]["val"] = args.test_data

    cfg["model"]["seed"] = args.seed

    # Initialize datasets
    val_datasets, val_dataloaders, val_nums = prepare_data_loaders("val", cfg)
    cfg_str = pprint.pformat(cfg, compact=True)
    logging.info("config:\n" + cfg_str + "\n")
    logging.info("")

    save_out_dir = os.path.join(args.output_dir, "visualization")
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

    # -------------------- Running training --------------------
    # Build model
    pre_engine, pre_context = prepare_engine(args.pretrained)
    head_engine, head_context = prepare_engine(args.head)
    stream = cuda.Stream()
    # 准备输入和输出信息
    pre_input_info, pre_output_info = prepare_engine_info(pre_engine)
    head_input_info, head_output_info = prepare_engine_info(head_engine)

    for name, info in pre_output_info.items():
        if name in head_input_info.keys():
            head_context.set_tensor_address(name, int(info["device_mem"]))
            print(f"set output {name} for head_inputs")

    eval_metrics = instantiate_from_config(cfg["trainer"]["eval_metrics"])

    eval_results = {}
    frame_eval_results = []

    # 推理循环
    for i, batch in enumerate(tqdm(val_dataloaders[0])):
        # 准备输入数据
        image = batch["image"].clone()
        image = ((image + 1) * 0.5 - model._mean) / model._std

        intrinsic = batch.get("intrinsic", None)  # [n, 3, 3]

        prompt_depth = batch[model.prompt_name]
        prompt_scale = batch[model.prompt_scale_name][:, None, None, None]

        ##### pretrain推理 #####
        head_inputs = {"prompt_depth": prompt_depth.numpy(), "prompt_scale": prompt_scale.numpy()}

        for name, data in head_inputs.items():
            if name in head_input_info.keys():
                input_data = head_inputs[name].astype(trt.nptype(head_input_info[name]["dtype"]))
                input_host = np.ascontiguousarray(input_data)
                cuda.memcpy_htod(head_input_info[name]["device_mem"], input_host)
                head_context.set_tensor_address(name, int(head_input_info[name]["device_mem"]))
                print(f"htod copy {name} for head_inputs")

        # 准备输出数据
        head_output_hosts = {}
        for name, info in head_output_info.items():
            shape = info["shape"]
            dtype = info["dtype"]
            head_output_hosts[name] = np.empty(shape, dtype=trt.nptype(dtype))
            head_context.set_tensor_address(name, int(info["device_mem"]))

        pre_inputs = {
            "image": image.numpy(),
        }

        # 更新输入数据
        for name in pre_inputs:
            input_data = pre_inputs[name].astype(trt.nptype(pre_input_info[name]["dtype"]))
            input_host = np.ascontiguousarray(input_data)
            cuda.memcpy_htod(pre_input_info[name]["device_mem"], input_host)
            pre_context.set_tensor_address(name, int(pre_input_info[name]["device_mem"]))

        # 准备输出数据
        pre_output_hosts = {}
        for name, info in pre_output_info.items():
            shape = info["shape"]
            dtype = info["dtype"]
            pre_output_hosts[name] = np.empty(shape, dtype=trt.nptype(dtype))
            pre_context.set_tensor_address(name, int(info["device_mem"]))

        # 执行推理
        stream.synchronize()
        start_time = time.time()

        pre_context.execute_async_v3(stream_handle=stream.handle)

        stream.synchronize()

        # # 将输出数据从 GPU 复制到 CPU
        # # NOTE: 如果没有这一步，仅第一张图正确
        # for name, output in pre_output_hosts.items():
        #     cuda.memcpy_dtoh(output, pre_output_info[name]["device_mem"])

        ##### head推理 #####
        stream.synchronize()

        head_context.execute_async_v3(stream_handle=stream.handle)

        stream.synchronize()
        end_time = time.time()

        # 打印推理时间和结果
        total_time = (end_time - start_time) * 1000  # 转换为毫秒
        logging.info(f"Inference time: {total_time:.2f} ms")

        pointmap_pred = head_output_hosts["pointmap"]

        # Convert the predicted pointmap to a NumPy array and extract the depth channel.
        pointmap_pred = pointmap_pred.reshape(-1)
        depth = pointmap_pred.clip(1e-3)

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
            pointmap=None,
            pointmap_gt=pointmap_gt,
            pointmap_color=pointmap_color,
            intrinsic=intrinsic,
            pointmap_h=image.shape[-2],
            pointmap_w=image.shape[-1],
        )

        if args.vis_results:
            model.visualize(output, batch["meta_data"], out_dir=save_out_dir)

        if eval_metrics is not None:
            frame_eval_results.append(eval_metrics(batch, output))

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
