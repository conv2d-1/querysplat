import json
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn


# 使用 nvidia-smi 获取 GPU 显存使用情况
def get_gpu_memory_usage():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = result.stdout.strip().split("\n")
    memory_usages = []
    for line in lines:
        used, total = map(int, line.split(","))
        memory_usages.append((used, total))
    return memory_usages


# 判断 GPU 是否有足够显存
def is_gpu_available(gpu_id, threshold=80000):  # 配置参数跑满
    memory_usages = get_gpu_memory_usage()
    if gpu_id < 0 or gpu_id >= len(memory_usages):
        return False
    used, total = memory_usages[gpu_id]
    free_ratio = total - used
    return free_ratio > threshold


# # 主进程：管理子进程
if __name__ == "__main__":
    print("Auto-starting tasks on available GPUs...")
    num_gpus = torch.cuda.device_count()
    while True:
        # 自动检查所有 GPU 的显存情况
        for gpu_id in range(num_gpus):
            if is_gpu_available(gpu_id):
                os.system(f"python3 hAlgorithm/script/auto_mem/gpu_task.py {gpu_id}&")
                print(f"Auto-started subprocess for GPU {gpu_id}")

        time.sleep(60 * 10)
