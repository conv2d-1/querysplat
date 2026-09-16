import os, sys
sys.path.append(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt

from hAlgorithm.script.quantization.quant_utils.onnx_helper import (
    before_last_slash
)

if __name__ == "__main__":
    csv_path = "/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/habitat_all_quant/quant_layer_analyse_20250415-203641/layer_analyse.csv"
    
    csv_abs_path = os.path.abspath(csv_path)
    csv_dir = os.path.dirname(csv_abs_path)
    basename = os.path.splitext(os.path.basename(csv_abs_path))[0]

    df = pd.read_csv(csv_path)

    df_abs_rel_diff_sorted = df[['config', 'abs_relative_difference']].sort_values(by='abs_relative_difference', ascending=False).reset_index(drop=True)
    df_abs_diff_sorted = df[['config', 'abs_difference']].sort_values(by='abs_difference', ascending=False).reset_index(drop=True)
   
    # draw
    plt.figure(figsize=(24, 10))
    plt.tight_layout()
    plt.subplot(1, 2, 1)
    df_abs_rel_diff_sorted['abs_relative_difference'].plot(kind='line', marker='o', label='abs_relative_difference')
    plt.title('abs_relative_difference')
    plt.xlabel('idx')
    plt.ylabel('value')
    plt.legend()

    for i in range(min(10, len(df_abs_rel_diff_sorted))):
        plt.text(
            x=i,  # x坐标
            y=df_abs_rel_diff_sorted['abs_relative_difference'].iloc[i],  # y坐标
            s=before_last_slash(before_last_slash(df_abs_rel_diff_sorted['config'].iloc[i])),  # 显示的文本
            fontsize=5,  # 字体大小
            rotation=45,  # 旋转角度
            ha='left',  # 水平对齐方式
            va='bottom'  # 垂直对齐方式
        )

    plt.subplot(1, 2, 2)
    df_abs_diff_sorted['abs_difference'].plot(kind='line', marker='o', label='abs_difference')
    plt.title('abs_difference')
    plt.xlabel('idx')
    plt.ylabel('value')
    plt.legend()

    for i in range(min(10, len(df_abs_diff_sorted))):
        plt.text(
            x=i,
            y=df_abs_diff_sorted['abs_difference'].iloc[i],
            s=before_last_slash(before_last_slash(df_abs_diff_sorted['config'].iloc[i])),
            fontsize=5,
            rotation=45,
            ha='left',
            va='bottom'
        )

    plt.savefig(f"{csv_dir}/layer_analyse.png", dpi=300)  # plt save must before show
    plt.show()
    plt.close()

    # save csv
    df_abs_rel_diff_sorted.to_csv(f"{csv_dir}/abs_rel_diff_sorted.csv", index=False)
    df_abs_diff_sorted.to_csv(f"{csv_dir}/abs_diff_sorted.csv", index=False)