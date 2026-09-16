import json
import random
import subprocess
import sys
import time

import torch
import torch.nn as nn


def get_gpustat():
    # 调用 gpustat 并以 JSON 格式输出
    result = subprocess.run(
        ["gpustat", "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        print("Error executing gpustat:", result.stderr)
        return None

    # 解析 JSON 输出
    gpu_info = json.loads(result.stdout)
    gpu_info_new = dict()
    for gpu in gpu_info["gpus"]:
        gpu_info_new[int(gpu["index"])] = gpu
    return gpu_info_new


# 定义子进程的任务函数
def gpu_task(device_id):

    gpu_data = get_gpustat()
    if device_id not in gpu_data:
        return
    gpu = gpu_data[device_id]
    memory_used = gpu["memory.used"]
    memory_total = gpu["memory.total"]
    memory = memory_total - memory_used
    nums = memory // 20000

    if nums == 0:
        return

    rdn = random.random()
    tensor_size = [15 * nums, 64, int(1024 + rdn * 100), int(1024 + rdn * 100)]

    device = torch.device(f"cuda:{device_id}")
    # 构造张量
    tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
    # 定义计算任务
    weight = torch.randn((64, 64, 3, 3), dtype=torch.float32, device=device)
    while True:
        result = torch.nn.functional.conv2d(tensor, weight, padding=1)
        result = torch.matmul(result, result.permute(0, 1, 3, 2))
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gpu_task.py <gpu_id>")
        sys.exit(1)

    try:
        device_id = int(sys.argv[1])
        gpu_task(device_id)
    except ValueError:
        print("Invalid GPU ID. Please provide an integer.")
        sys.exit(1)
