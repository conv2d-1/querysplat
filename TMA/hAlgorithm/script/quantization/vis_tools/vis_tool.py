import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def before_last_slash(s):
    index = s.rfind("/")
    return s[:index] if index != -1 else s

if __name__ == "__main__":
    csv_path = "results/quantization/prompt_pointmap_stage2_orig_resnet_v0_all_quant/analyse_onnx_quant_model/eval_results_100.csv"
    
    csv_abs_path = os.path.abspath(csv_path)  # 处理相对路径
    csv_dir = os.path.dirname(csv_abs_path)        # 所在目录路径
    basename = os.path.splitext(os.path.basename(csv_abs_path))[0]

    df = pd.read_csv(csv_path)
    layer_names = df['dequant_layer_name'].shift(-1).dropna()
    abs_rel_changes = df['abs_relative_difference'].diff().dropna()
    abs_changes = df['abs_difference'].diff().dropna()

    df_abs_rel_changes = pd.DataFrame({
        'abs_rel_changes': abs_rel_changes.values,
        'layer_name': layer_names.values
    })
    # filtered_abs_rel_changes = df_abs_rel_changes.sort_values(by=['abs_rel_changes'], ascending=True).head(10)
    std = df_abs_rel_changes['abs_rel_changes'].std()
    mean = df_abs_rel_changes['abs_rel_changes'].mean()
    filtered_abs_rel_changes = df_abs_rel_changes[
        (df_abs_rel_changes['abs_rel_changes'] > mean + std) | 
        (df_abs_rel_changes['abs_rel_changes'] < mean - std)
    ]

    df_abs_changes = pd.DataFrame({
        'abs_changes': abs_changes.values,
        'layer_name': layer_names.values
    })
    # filtered_abs_changes = df_abs_changes.sort_values(by=['abs_changes'], ascending=True).head(10
    std = df_abs_changes['abs_changes'].std()
    mean = df_abs_changes['abs_changes'].mean()
    filtered_abs_changes = df_abs_changes[
        (df_abs_changes['abs_changes'] > mean + std) | 
        (df_abs_changes['abs_changes'] < mean - std)
    ]

    plt.figure(figsize=(12, 5))
    plt.tight_layout()
    plt.subplot(2, 2, 1)
    df['abs_relative_difference'].plot(kind='line', marker='o', label='abs_relative_difference')
    plt.title('abs_relative_difference')
    plt.xlabel('idx')
    plt.ylabel('value')
    plt.legend()

    plt.subplot(2, 2, 3)
    df['abs_difference'].plot(kind='line', marker='s', label='abs_difference')
    plt.title('abs_difference')
    plt.xlabel('idx')
    plt.ylabel('value')
    plt.legend()

    plt.subplot(2, 2, 2)
    abs_rel_changes.plot(kind='line', label='abs_relative_changes')
    filtered_indexs = filtered_abs_rel_changes.index.values
    filtered_values = filtered_abs_rel_changes['abs_rel_changes'].values
    filtered_layers = filtered_abs_rel_changes['layer_name'].values
    colors = plt.cm.tab20.colors
    layers = [before_last_slash(layer) for layer in filtered_layers]
    with open(f"{csv_dir}/{basename}_min_rel_changes.json", 'w') as f:
        json.dump(layers, f, indent=4)
    for i in range(len(filtered_indexs)):
        plt.scatter(filtered_indexs[i], filtered_values[i], 
                color=colors[i % len(colors)],  
                label=filtered_layers[i],  # 设置标签
                s=100,  # 点大小
                edgecolors='black')  # 添加边框
    plt.title('abs_relative_changes')
    plt.xlabel('idx')
    plt.ylabel('value')
    plt.legend()

    # 绘制abs_difference
    plt.subplot(2, 2, 4)
    abs_changes.plot(kind='line', label='abs_difference_changes')
    filtered_indexs = filtered_abs_changes.index.values
    filtered_values = filtered_abs_changes['abs_changes'].values
    filtered_layers = filtered_abs_changes['layer_name'].values
    colors = plt.cm.tab20.colors
    layers = [before_last_slash(layer) for layer in filtered_layers] 
    with open(f"{csv_dir}/{basename}_min_abs_changes.json", 'w') as f:
        json.dump(layers, f, indent=4)
    for i in range(len(filtered_indexs)):
        plt.scatter(filtered_indexs[i], filtered_values[i], 
                color=colors[i % len(colors)],  
                label=filtered_layers[i],  # 设置标签
                s=100,  # 点大小
                edgecolors='black')  # 添加边框
    plt.title('abs_difference_changes')
    plt.xlabel('idx')
    plt.ylabel('value')
    plt.legend()

    plt.show()