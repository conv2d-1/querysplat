import logging
import time
from contextlib import contextmanager

import torch


# 定义一个装饰器，用于测量函数执行时间
def cuda_timing_decorator(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.time()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time()
        logging.info(f"{func.__name__} 执行时间: {end - start:.6f} 秒")
        return result

    return wrapper


# 定义一个上下文管理器，用于测量代码块的执行时间
@contextmanager
def cuda_timing_context(name="", timing=True):
    if timing:
        torch.cuda.synchronize()
        start = time.time()
    try:
        yield
    finally:
        if timing:
            torch.cuda.synchronize()
            end = time.time()
            logging.info(f"代码块|{name}|执行时间: {end - start:.6f} 秒")


# 示例使用装饰器
@cuda_timing_decorator
def example_function():
    # 模拟一些 CUDA 操作
    tensor = torch.randn(10000, 10000).cuda()
    result = tensor * tensor
    return result


# 示例使用上下文管理器
def example_with_context():
    with cuda_timing_context("init", False):
        # 模拟一些 CUDA 操作
        tensor = torch.randn(10000, 10000).cuda()
        result = tensor * tensor
        return result


# 测试
if __name__ == "__main__":
    example_function()
    example_with_context()
