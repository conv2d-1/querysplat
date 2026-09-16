import os
import sys
import time

import numpy as np
import pandas as pd
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from tabulate import tabulate


# 设置 profiler
class Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.layer_times = []

    def report_layer_time(self, layer_name, ms):
        self.layer_times.append((layer_name, ms))


class TRTProfiler:
    def __init__(self):
        self.profiler = None
        self.engine = None
        self.context = None
        self.input_info = {}
        self.output_info = {}
        self.output_hosts = {}
        self.data_table = {}

    # 加载 TensorRT engine
    def load_engine(self, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        with open(engine_file_path, "rb") as f, trt.Runtime(
            trt.Logger(trt.Logger.WARNING)
        ) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    # 创建执行上下文
    def create_execution_context(self, engine):
        return engine.create_execution_context()

    def init(self, engine_file_path):
        self.engine = self.load_engine(engine_file_path)
        self.context = self.create_execution_context(self.engine)

        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            # 打印输入/输出信息
            print(f"Tensor {idx}: Name={name}, Shape={shape}, Dtype={dtype}, IsInput={is_input}")

            # 分配 GPU 内存
            size = trt.volume(shape) * dtype.itemsize
            device_mem = cuda.mem_alloc(size)

            # 存储信息
            if is_input:
                self.input_info[name] = {
                    "shape": shape,
                    "dtype": dtype,
                    "device_mem": device_mem,
                }
            else:
                self.output_info[name] = {
                    "shape": shape,
                    "dtype": dtype,
                    "device_mem": device_mem,
                }

        # 准备输入数据
        for name, info in self.input_info.items():
            shape = info["shape"]
            dtype = info["dtype"]

            # 生成随机数据
            input_data = np.random.random(shape).astype(trt.nptype(dtype))
            input_host = np.ascontiguousarray(input_data)

            # 将数据复制到 GPU
            cuda.memcpy_htod(info["device_mem"], input_host)

            # 设置输入张量地址
            self.context.set_tensor_address(name, int(info["device_mem"]))

        # 准备输出数据
        for name, info in self.output_info.items():
            shape = info["shape"]
            dtype = info["dtype"]

            # 分配 CPU 内存
            self.output_hosts[name] = np.empty(shape, dtype=trt.nptype(dtype))

            # 设置输出张量地址
            self.context.set_tensor_address(name, int(info["device_mem"]))

        # 打印输出数据
        for name, output in self.output_hosts.items():
            print(f"Output {name}: Shape={output.shape}")

    def run(self, num_runs=20, collect_beg=2):
        for i in range(num_runs):
            self.profiler = Profiler()
            self.context.profiler = self.profiler
            # 执行推理
            start_time = time.time()
            stream = cuda.Stream()
            self.context.execute_async_v3(stream_handle=stream.handle)
            stream.synchronize()
            end_time = time.time()

            infer_time = (end_time - start_time) * 1000  # 转换为毫秒
            print(f"Total inference time: {infer_time:.2f} ms")

            if i >= collect_beg:
                self.collect_data()
                if "Total" not in self.data_table:
                    self.data_table["Total"] = infer_time
                else:
                    self.data_table["Total"] += infer_time

        for layer_name, ms in self.data_table.items():
            self.data_table[layer_name] /= num_runs - collect_beg

        # 定义表头
        headers = ["Layer Name", "Time (ms)"]

        # 准备数据：将字典转换为 [(layer_name, time_ms), ...]
        rows = [(layer_name, time_ms) for layer_name, time_ms in self.data_table.items()]

        # 打印表格，设置左对齐
        print("\nLayer-wise Performance:")
        print(tabulate(rows, headers=headers, tablefmt="pretty", stralign="left"))

    def collect_data(self):
        for layer_name, ms in self.profiler.layer_times:
            if layer_name not in self.data_table:
                self.data_table[layer_name] = ms
            else:
                self.data_table[layer_name] += ms

    def save_profiler(self, output_path):
        df = pd.DataFrame(list(self.data_table.items()), columns=["Layer Name", "Time (ms)"])
        df.to_csv(output_path, index=False)

    def close(self):
        # 清理资源
        for info in self.input_info.values():
            info["device_mem"].free()

        for info in self.output_info.values():
            info["device_mem"].free()


if __name__ == "__main__":
    engine_file_path = sys.argv[1]
    engine_dir = os.path.dirname(engine_file_path)
    result_path = f"{engine_dir}/profiler_result.csv"

    profiler = TRTProfiler()

    profiler.init(engine_file_path)
    profiler.run()
    profiler.save_profiler(result_path)
    profiler.close()
