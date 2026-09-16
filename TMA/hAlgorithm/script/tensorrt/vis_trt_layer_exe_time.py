"""
用法:
    python script_name.py --profile path/to/profile.json

说明:
    1. 该脚本用于解析指定JSON格式的profile文件,并绘制各层执行耗时的饼图和排名列表。
    2. profile参数为必选项,表示包含profile数据的JSON文件的路径。
    3. 运行脚本后,会显示一个图形窗口,左侧为饼图,右侧为前10大耗时层的列表。

示例:
    python script_name.py --profile ./data/profile.json
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def parse_profile_data(profile):
    with open(profile, "r") as f:
        profile_data = json.load(f)

    # 解析耗时数据
    layers = []
    times = []

    for entry in profile_data:
        if "name" in entry and "averageMs" in entry and entry["averageMs"] > 0:  # 过滤掉耗时为0的层
            layers.append(entry["name"])
            times.append(entry["averageMs"])

    print(f"total time {sum(times)} ms")
    # 按时间降序排序
    sorted_indices = sorted(range(len(times)), key=lambda i: times[i], reverse=True)
    top_n = min(10, len(times))  # 只取前10大耗时的层
    top_layers = [layers[i] for i in sorted_indices[:top_n]]
    top_times = [times[i] for i in sorted_indices[:top_n]]
    other_time = sum(times[i] for i in sorted_indices[top_n:])

    # 如果有剩余层,把它们合并为 "Others"
    if other_time > 0:
        top_layers.append("Others")
        top_times.append(other_time)

    # 设置颜色,确保 top1 和 Others 颜色不同
    colors = plt.cm.tab10(np.arange(len(top_layers)))  # 选择不同的颜色
    colors[0] = plt.cm.Set1(0)  # 确保 Top1 层颜色明显不同

    # 创建画布,分成左右两部分
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 在左侧绘制饼图
    wedges, texts, autotexts = axs[0].pie(
        top_times,
        labels=None,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        textprops={"fontsize": 10},
    )
    axs[0].set_title("TensorRT Layer Execution Time Breakdown")

    # 在右侧显示文本列表,按耗时排序
    axs[1].axis("off")  # 关闭坐标轴
    axs[1].set_title("Top 10 Layers by Execution Time", fontsize=12, fontweight="bold")

    # 显示层名称和对应的时间,并在前面加上颜色小方块
    for i, (layer, time) in enumerate(zip(top_layers, top_times)):
        layer_text = f"{i+1}. {layer[:40]}..." if len(layer) > 40 else f"{i+1}. {layer}"  # 避免太长

        # 画一个小方块代表颜色
        axs[1].text(
            -0.05, 1 - i * 0.1, "■", fontsize=12, color=colors[i], verticalalignment="center"
        )
        axs[1].text(
            0, 1 - i * 0.1, f"{layer_text} - {time:.2f} ms", fontsize=10, verticalalignment="center"
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解析profile数据并展示图表")
    parser.add_argument("--profile", type=str, required=True, help="包含profile数据的JSON文件路径")
    args = parser.parse_args()
    parse_profile_data(args.profile)
