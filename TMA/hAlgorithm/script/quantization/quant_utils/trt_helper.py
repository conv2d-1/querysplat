import os, sys
sys.path.append(os.getcwd())

from itertools import islice
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

import torch
torch.zeros(1).cuda() # 先让 PyTorch 初始化 CUDA 上下文

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from hAlgorithm.utils import instantiate_from_config
from hAlgorithm.utils.util import eval_dict_to_text
from hAlgorithm.script.quantization.quant_utils.data_helper import normalize_depth


def normalize(depth: torch.Tensor, scale: torch.Tensor, center: torch.Tensor):
    if center is not None:
        depth = depth - center
    if scale is not None:
        depth = depth / scale
    return depth

def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    with open(engine_file_path, "rb") as f, trt.Runtime(
        trt.Logger(trt.Logger.WARNING)
    ) as runtime:
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
        print(
            f"Tensor {idx}: Name={name}, Shape={shape}, Dtype={dtype}, IsInput={is_input}"
        )

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

    

class TrtModel:
    def __init__(self, engine_path):
        self.engine_path = engine_path

        self.engine, self.context = prepare_engine(self.engine_path)
        self.stream = cuda.Stream()
        self.input_info, self.output_info = prepare_engine_info(self.engine)

        self.output_hosts = {}
        for name, info in self.output_info.items():
            shape = info["shape"]
            dtype = info["dtype"]
            self.output_hosts[name] = np.empty(shape, dtype=trt.nptype(dtype))
            self.context.set_tensor_address(name, int(info["device_mem"]))

    def __call__(self, input_data, meta_data=None):
        inputs = {
            "image": input_data['image'].cpu().numpy(),
            "prompt_depth": input_data['prompt_depth'].cpu().numpy()[:,-1:,:,:], #[:,-1:,:,:]
            "prompt_scale": input_data['prompt_scale'].cpu().numpy(),
        }

        # 更新输入数据
        for name in inputs:
            trt_input_data = inputs[name].astype(trt.nptype(self.input_info[name]["dtype"]))
            input_host = np.ascontiguousarray(trt_input_data)
            cuda.memcpy_htod(self.input_info[name]["device_mem"], input_host)
            self.context.set_tensor_address(name, int(self.input_info[name]["device_mem"]))

        # 执行推理
        self.stream.synchronize()
        start_time = time.time()

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        self.stream.synchronize()
        end_time = time.time()

        # 将输出数据从 GPU 复制到 CPU
        # NOTE: 如果没有这一步，仅第一张图正确
        for name, output in self.output_hosts.items():
            cuda.memcpy_dtoh(output, self.output_info[name]["device_mem"])

        # 打印推理时间和结果
        total_time = (end_time - start_time) * 1000  # 转换为毫秒
        print(f"Inference time: {total_time:.2f} ms")

        if "depth" in self.output_hosts.keys():
            pointmap_pred = self.output_hosts["depth"]
        elif "pointmap" in self.output_hosts.keys():
            pointmap_pred = self.output_hosts["pointmap"]
        else:
            raise KeyError("No Valid Depth/Pointmap found in output")
        pointmap_pred = torch.from_numpy(pointmap_pred).cuda()

        if "confidence" in self.output_hosts.keys():
            confidence_pred = self.output_hosts["confidence"]
        else:
            confidence_pred = None
        confidence_pred = torch.from_numpy(confidence_pred).cuda()

        pointmap_pred = normalize_depth(pointmap_pred, input_data['prompt_scale'].cuda(), None)

        results = {
            "pointmap": pointmap_pred,
            "confidence": confidence_pred
        }

        return results