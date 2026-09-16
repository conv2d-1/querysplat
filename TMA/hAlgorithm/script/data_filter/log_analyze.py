import argparse
import csv
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(file_path):
    line_pattern = re.compile(r"iter\d+,\s*epoch\d+,\s*nbatch\d+")
    kv_pattern = re.compile(r"(\w+):([^\s,]+)")

    data_records = defaultdict(list)
    loss_types = set()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line_pattern.search(line):
                matches = kv_pattern.findall(line)
                record = {k: v for k, v in matches}

                if "data" not in record:
                    continue

                # 转换数值类型（确保使用Python原生float）
                for key in record:
                    try:
                        if key.endswith("_loss") or key == "loss":
                            record[key] = float(record[key])  # 使用标准float而非np.float
                    except ValueError:
                        record[key] = 0.0  # 无法转换时设置为0.0

                data_type = record["data"]
                data_records[data_type].append(record)
                loss_types.update([k for k in record if k.endswith("_loss") or k == "loss"])

    return data_records, sorted(loss_types)


def merge_data_records(all_data):
    merged_records = defaultdict(list)
    all_loss_types = set()

    for data_records, loss_types in all_data:
        all_loss_types.update(loss_types)
        for data_type, records in data_records.items():
            merged_records[data_type].extend(records)

    return merged_records, sorted(all_loss_types)


def process_directory(input_path):
    """处理目录中的所有日志文件"""
    file_paths = []
    for ext in ["*.log", "*.txt"]:
        file_paths.extend(glob(str(Path(input_path) / ext)))

    if not file_paths:
        raise ValueError("未找到日志文件")

    all_data = []
    for file_path in file_paths:
        print(f"正在处理: {file_path}")
        data_records, loss_types = parse_log_file(file_path)
        all_data.append((data_records, loss_types))

    return merge_data_records(all_data)


def save_to_files(data_records, loss_types, output_dir):
    """
    将解析后的数据保存到文件中，包括正常数据和异常数据。
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 初始化全局异常数据列表
    all_abnormal_records = []
    abnormal_file_paths = {}
    for data_type, records in data_records.items():
        # 提取loss字段并计算均值和标准差
        losses = [
            float(record["loss"])
            for record in records
            if "loss" in record and isinstance(record["loss"], (int, float))
        ]
        if not losses:
            print(f"数据类型 {data_type} 没有有效的loss数据，跳过保存")
            continue

        losses = np.nan_to_num(losses, nan=999)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        print(f"{data_type} Loss mean:{mean_loss}; std:{std_loss}")

        # 定义正常和异常数据的范围
        lower_bound = 0
        upper_bound = mean_loss + 3 * std_loss

        # 分离正常数据和异常数据
        all_records = []
        abnormal_records = []
        for rec in records:
            if "loss" in rec and isinstance(rec["loss"], (int, float)):
                all_records.append(rec)
                if lower_bound <= rec["loss"] <= upper_bound:
                    pass
                else:
                    abnormal_records.append(rec)

        # 写入正常数据文件
        all_file_path = Path(output_dir) / f"{data_type}.txt"
        mf_header = ["scene", "frame_id", "view_id"] if "scene" in all_records[0].keys() else []
        with open(all_file_path, "w", encoding="utf-8") as f:
            header = ["record_id"] + list(loss_types) + ["data_idx", "rgb_path"] + mf_header
            f.write("\t".join(header) + "\n")

            for i, record in enumerate(all_records, 1):
                row = [str(i)]
                row += [str(record.get(loss, "N/A")) for loss in loss_types]
                row += [str(record.get("data_idx", "N/A")), str(record.get("rgb_path", "N/A"))]
                row += [str(record.get(mf_key, "N/A")) for mf_key in mf_header]
                f.write("\t".join(row) + "\n")

        print(f"已保存 {data_type} 的正常数据至 {all_file_path}")

        # 写入异常数据文件（如果有
        if abnormal_records:
            abnormal_file_path = Path(output_dir) / f"{data_type}_abnormal_{upper_bound:.4f}.csv"
            abnormal_file_paths[data_type] = abnormal_file_path
            with open(abnormal_file_path, "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["record_id"] + list(loss_types) + ["data_idx", "rgb_path"] + mf_header
                writer.writerow(header)

                for i, record in enumerate(abnormal_records, 1):
                    row = [str(i)]
                    row += [str(record.get(loss, "N/A")) for loss in loss_types]
                    row += [str(record.get("data_idx", "N/A")), str(record.get("rgb_path", "N/A"))]
                    row += [str(record.get(mf_key, "N/A")) for mf_key in mf_header]
                    writer.writerow(row)

            print(f"已保存 {data_type} 的异常数据至 {abnormal_file_path}")

        for record in abnormal_records:
            record["data"] = data_type  # 添加data字段标识数据来源
            record["loss"] = np.nan_to_num(float(record["loss"]), nan=9999)
            all_abnormal_records.append(record)

    if all_abnormal_records:
        mf_header = ["scene", "frame_id", "view_id"]
        global_abnormal_file_path = Path(output_dir) / "all_abnormal_data.csv"
        with open(global_abnormal_file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = ["record_id", "data"] + list(loss_types) + ["data_idx", "rgb_path"] + mf_header
            writer.writerow(header)

            for i, record in enumerate(all_abnormal_records, 1):
                row = [str(i), record["data"]]
                row += [str(record.get(loss, "N/A")) for loss in loss_types]
                row += [str(record.get("data_idx", "N/A")), str(record.get("rgb_path", "N/A"))]
                row += [str(record.get(mf_key, "N/A")) for mf_key in mf_header]
                writer.writerow(row)

        print(f"已保存所有异常数据至 {global_abnormal_file_path}")

    return abnormal_file_paths


def plot_combined_histogram(data_records, output_dir="./output/"):
    """绘制所有数据源的loss合并直方图"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    loss_types = set()
    for records in data_records.values():
        if records:
            loss_types.update([k for k in records[0] if k.endswith("_loss") or k == "loss"])
    loss_types = sorted(loss_types)

    plot_data = defaultdict(list)
    for data_type, records in data_records.items():
        for record in records:
            for loss in loss_types:
                if loss in record and isinstance(record[loss], (int, float)):
                    plot_data[loss].append(record[loss])

    if not plot_data:
        print("没有可绘制的loss数据")
        return

    cols = 3
    rows = (len(plot_data) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if len(plot_data) > 1 else [axes]

    for ax, (loss_name, values) in zip(axes, plot_data.items()):
        ax.hist(values, bins=20, edgecolor="black", alpha=0.7)
        ax.set_title(f"{loss_name}\n(μ={np.mean(values):.4f}, σ={np.std(values):.4f})")
        ax.set_xlabel("Loss Value")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.75)

    for ax in axes[len(plot_data) :]:
        ax.remove()

    plt.suptitle("Combined Loss Distributions", y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(Path(output_dir) / "all_loss_histograms.png")
    plt.close()
    print(f"已生成 all_loss_histograms.png")


def plot_individual_histograms(data_records, output_dir="./output/"):
    """绘制每个数据源的单独直方图"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for data_type, records in data_records.items():
        # 收集该数据源的所有loss类型
        loss_columns = [k for k in records[0] if k.endswith("_loss") or k == "loss"]
        if not loss_columns:
            continue

        # 准备画布
        num_losses = len(loss_columns)
        cols = min(3, num_losses)
        rows = (num_losses + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if num_losses > 1 else [axes]

        # 绘制每个loss的直方图
        for ax, loss_col in zip(axes, loss_columns):
            values = [
                rec[loss_col] for rec in records if isinstance(rec.get(loss_col), (int, float))
            ]

            if not values:
                ax.set_visible(False)
                continue

            ax.hist(values, bins=20, edgecolor="black", alpha=0.7)
            ax.set_title(f"{loss_col}\nμ={np.mean(values):.4f} | σ={np.std(values):.4f}")
            ax.set_xlabel("Loss Value")
            ax.set_ylabel("Frequency")
            ax.grid(axis="y", alpha=0.3)

        # 删除多余子图
        for ax in axes[num_losses:]:
            ax.remove()

        plt.suptitle(f"{data_type} Loss Distributions", y=0.99)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(Path(output_dir) / f"{data_type}_loss_histograms.png")
        plt.close()
        print(f"已生成 {data_type}_loss_histograms.png")


def plot_all_data_histogram(data_records, output_dir="./output/"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_data = []
    data_types = []
    for data_type, records in data_records.items():
        # 确保使用列表推导式生成Python列表
        losses = [
            rec["loss"]
            for rec in records
            if "loss" in rec and isinstance(rec["loss"], (int, float))
        ]
        if losses:
            plot_data.append(losses)
            data_types.append(data_type)

    if not plot_data:
        print("没有可绘制的loss数据")
        return

    # 创建画布
    num_plots = len(plot_data)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if num_plots > 1 else [axes]

    # 绘制子图（明确使用matplotlib的hist方法）
    for idx, (data_type, losses) in enumerate(zip(data_types, plot_data)):
        ax = axes[idx]
        # 确保传递的是Python列表或numpy数组
        ax.hist(losses, bins=20, edgecolor="black", alpha=0.7, density=False)
        ax.set_title(f"{data_type}\nμ={np.mean(losses):.4f} | σ={np.std(losses):.4f}")
        ax.set_xlabel("Loss Value")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.3)

    # 删除多余子图
    for ax in axes[len(plot_data) :]:
        ax.remove()

    plt.suptitle("Loss Distribution by Data Source (Frequency)", y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(Path(output_dir) / "all_data_histograms.png")
    plt.close()
    print(f"已生成 all_data_histograms.png")


def main(path):
    # 构造输出目录路径
    output_dir = os.path.join(path, "output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 处理数据
    data_records, loss_types = process_directory(path)

    # 保存和绘图时使用新的输出路径
    abnormal_file_paths = save_to_files(data_records, loss_types, output_dir)
    plot_combined_histogram(data_records, output_dir)
    plot_individual_histograms(data_records, output_dir)
    # plot_all_data_histogram(data_records, output_dir)

    print(f"处理完成，共处理{len(data_records)}种数据类型，结果保存在{output_dir}目录")
    return abnormal_file_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日志分析工具")
    parser.add_argument("--path", type=str, default=".", help="日志文件所在目录（默认当前目录）")
    args = parser.parse_args()

    main(args.path)
