import os

import onnx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

import argparse

def before_last_slash(s):
    index = s.rfind("/")
    return s[:index] if index != -1 else s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="amax statistics")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        help="Path to onnx path",
    )
    args = parser.parse_args()

    onnx_abs_path = os.path.abspath(args.onnx_path)  # 处理相对路径
    onnx_dir = os.path.dirname(onnx_abs_path)        # 所在目录路径
    basename = os.path.splitext(os.path.basename(onnx_abs_path))[0]

    # 加载ONNX模型
    model = onnx.load(args.onnx_path)
    onnx.checker.check_model(model)

    # 读取每层的名称
    quant_scale_names = []
    for i, node in enumerate(model.graph.node):
        if node.op_type=="QuantizeLinear":
            layer_name = node.name
            print(f"Layer {i}: Name = {layer_name}, Type = {node.op_type}")
            layer_name = before_last_slash(layer_name)
            quant_scale_name = f"{layer_name}/Constant_1"
            quant_scale_names.append(quant_scale_name)

    quant_dict = {}
    for i, node in enumerate(model.graph.node):
        layer_name = node.name
        if layer_name in quant_scale_names:
            print(f"Layer {i}: Name = {layer_name}, Type = {node.op_type}")
            scale = onnx.numpy_helper.to_array(node.attribute[0].t)
            layer_name = before_last_slash(layer_name)
            quant_dict[layer_name] = scale * 127  # amax
    max_amax = {k: v.max() for k, v in quant_dict.items()}

    df = pd.DataFrame(list(max_amax.items()), columns=["Layer", "MaxAmax"])
    df['Index'] = range(len(df))  # 新增索引列（0,1,2,...）
    # 按最大值降序排列并取前10
    df_sorted = df.sort_values(by="MaxAmax", ascending=False).head(10)

    # 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制所有数据点（灰色虚线）
    ax.plot(df['Index'], df['MaxAmax'], color='blue', linestyle='-', label='All Layers')

    # 绘制前10大值（红色实线）
    sorted_indexs = df_sorted['Index'].values
    sorted_scales = df_sorted['MaxAmax'].values
    sorted_layers = df_sorted['Layer'].values
    colors = plt.cm.tab10.colors[:len(sorted_layers)]

    layers = [before_last_slash(layer) for layer in sorted_layers]
    
    with open(f"{onnx_dir}/{basename}_amax.json", 'w') as f:
        json.dump(layers, f, indent=4)

    for i in range(len(sorted_indexs)):
        ax.scatter(sorted_indexs[i], sorted_scales[i], 
                color=colors[i],  
                label=sorted_layers[i],  # 设置标签
                s=100,  # 点大小
                edgecolors='black')  # 添加边框

    # 添加主图例（显示所有点的颜色和层名称）
    ax.legend(loc='upper left', bbox_to_anchor=(0.8, 1))

    # 设置坐标轴标签
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Max Amax Value', fontsize=12)
    ax.set_title('Top 10 Layers by Maximum Amax Value', fontsize=14)

    plt.subplots_adjust(left=0.05, right=0.9)
    plt.show()