import subprocess

import torch


def get_gpu_memory(device_id):
    """
    获取指定 GPU 设备的剩余显存量（以 MB 为单位）。

    参数:
    device_id (int): 要查询的 GPU 设备 ID。

    返回:
    float: 总体显存量 MB ,如果设备不存在或命令执行失败则返回 None。
    float: 剩余显存量 MB ,如果设备不存在或命令执行失败则返回 None。
    """
    try:
        if isinstance(device_id, str):
            if len(device_id) > 1 and (":" in device_id or device_id in ["cuda"]):
                device_id = 0 if device_id in ["cuda"] else int(device_id.split(":")[1])
        elif isinstance(device_id, (int, float)):
            device_id = int(device_id)
        else:
            raise NotImplementedError
        # 使用 nvidia-smi 查询指定设备的总显存和已用显存
        command = f"nvidia-smi --id={device_id} --query-gpu=memory.total,memory.used --format=csv,noheader,nounits"
        result = subprocess.check_output(command, shell=True).decode("utf-8").strip()

        # 解析结果
        total_memory_str, used_memory_str = result.split(", ")
        total_memory = float(total_memory_str)
        used_memory = float(used_memory_str)

        # 计算剩余显存
        free_memory = total_memory - used_memory

        return total_memory, free_memory

    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi command for device {device_id}: {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred while processing the output for device {device_id}: {e}")
        return None, None


def get_gpu_memory_torch(device):
    device = torch.device(device) if torch.cuda.is_available() else None
    if device is not None:
        properties = torch.cuda.get_device_properties(device)
        total_memory = properties.total_memory / (1024**2)  # Convert to MB
        return total_memory
    else:
        print("No CUDA device found.")
        return None


class DummyMemoryReserver(object):

    def __init__(
        self, memory_capacity=30, min_reserve_memory=10, check_iters=999, reserve_ratio=0.7
    ) -> None:
        self.memory_capacity = memory_capacity
        self.min_reserve_memory = min_reserve_memory
        self.check_iters = check_iters
        self.reserve_ratio = reserve_ratio
        self.dummy = {}
        self.reserved_memory = {}

    def reserve_cuda_memory(self, device, iter):
        max_memory, free_memory = get_gpu_memory(device)
        if (
            max_memory / 1024 > self.memory_capacity
            and free_memory / 1024 > self.min_reserve_memory
        ):
            if iter == 10 or iter % self.check_iters == 0:
                reserved_memory = self.reserved_memory.get(device, 0)
                free_memory = free_memory + reserved_memory
                reserve_size = int((free_memory * self.reserve_ratio) // 8)
                if reserve_size != int(self.reserved_memory.get(device, 0) * 8):
                    self.reserved_memory[device] = reserve_size * 8
                    try:
                        del self.dummy[device]
                    except Exception:
                        pass
                    self.dummy[device] = torch.zeros(reserve_size, 1024, 1024, dtype=float).to(
                        device=device
                    )

    def watch_for_all_gpus(self):
        device_count = torch.cuda.device_count()
        for device in range(device_count):
            self.reserve_cuda_memory("cuda:" + str(device), iter=10)


if __name__ == "__main__":
    a = DummyMemoryReserver()
    import time

    while True:
        a.watch_for_all_gpus()
        time.sleep(10)
