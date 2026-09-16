from contextlib import contextmanager, nullcontext

import torch

"""
# 使用示例
total_steps = 100  # 比如总共 100 步

with profiled_context(total_steps, enable=True) as loop:
    for step in loop:
        train_step()  # 每次迭代后，如果开启 profiler，则自动调用 profiler.step()
"""


@contextmanager
def profiled_context(
    total_steps,
    enable=True,
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=10, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profile"),
    with_stack=True,
):
    """
    可选的 profile 上下文管理器，当 enable=True 时，会启动 profiler，
    并在每次迭代后自动调用 profiler.step()；否则仅返回一个普通的迭代器。
    """
    if enable:
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            with_stack=with_stack,
        ) as profiler:
            current_step = 0

            # 定义生成器，每次 yield 后自动调用 profiler.step()
            def loop_generator():
                nonlocal current_step
                for _ in range(total_steps):
                    yield current_step
                    profiler.step()
                    current_step += 1

            yield loop_generator()
    else:
        with nullcontext() as dummy:
            # 如果不启用 profiler，则直接 yield range(total_steps) 作为迭代器
            yield iter(range(total_steps))
