import os
import sys

sys.path.append(os.path.abspath(__file__))

import argparse

import tensorrt as trt
from utils import allocate_buffers, build_engine_onnx

# 假设TRT_LOGGER已经被定义
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def create_engine(onnx_file_path, engine_file_path):
    with trt.Builder(
        TRT_LOGGER
    ) as builder, builder.create_builder_config() as config, builder.create_network(
        flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser:

        # 配置最大工作空间大小
        # config.max_workspace_size = 1 << 30  # 1 GB
        config = builder.create_builder_config()

        # 加载ONNX模型
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # 配置 builder
        # config = builder.create_builder_config()
        # config.set_flag(trt.BuilderFlag.FP16)  # 启用 FP16 模式

        # 构建序列化引擎
        serialized_engine = builder.build_serialized_network(network, config)
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print("save:", engine_file_path)

        return serialized_engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser("convert onnx to trt engine")
    parser.add_argument("onnx_path")
    args = parser.parse_args()

    save_path = args.onnx_path[:-4] + ".engine"
    serialized_engine = create_engine(args.onnx_path, save_path)
