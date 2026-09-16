import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def load_csv_as_dict(csv_path, header='infer'):
    df_csv = pd.read_csv(csv_path, header=header)
    dict_csv = {row[0]: row[1:] for row in df_csv.itertuples(index=False)}
    return dict_csv

def clean_amax_search(dict_amax_search):
    dict_tmp = {}
    for key, value in dict_amax_search.items():
        result = {}
        lines = key.split('\n')
        for line in lines:
            # 去除首尾空白字符
            line = line.strip()
            if not line:
                continue
            if '=' in line:
                key2, value2 = line.split('=', 1)
                result[key2] = float(value2)
        new_key = "\n".join([f"{k}={v:.4f}" for k, v in result.items()])
        dict_tmp[new_key] = value
    return dict_tmp

def before_last_slash(s):
    index = s.rfind("/")
    return s[:index] if index != -1 else s

if __name__ == "__main__":
    best_amax_path = '/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/habitat_all_quant/quant_amax_search/best_amax.csv'
    amax_search = '/mnt/naspersonal/lx/projects/TM/models/prompt_pointmap_952_250411_600k/habitat_all_quant/quant_amax_search/amax_search.csv'

    dict_amax_search = load_csv_as_dict(amax_search)

    # dict_amax_search = dict_amax_search1 | dict_amax_search2
    dict_amax_search = clean_amax_search(dict_amax_search)
    dict_best_amax = load_csv_as_dict(best_amax_path, header=None)

    best_amax_cur = {}

    rel_abs_diffs = []
    abs_diffs = []
    layer_names = []

    for key, value in dict_best_amax.items():
        best_amax_cur[key] = value[0]
        search_key = "\n".join([f"{k}={v:.4f}" for k, v in best_amax_cur.items()])
        metrics = dict_amax_search[search_key]

        rel_abs_diffs.append(metrics[0])
        abs_diffs.append(metrics[1])
        layer_names.append(before_last_slash(before_last_slash(key)))

    rel_abs_diffs = np.array(rel_abs_diffs)
    abs_diffs = np.array(abs_diffs)
    layer_names = np.array(layer_names)

    rel_abs_changes = rel_abs_diffs[1:] - rel_abs_diffs[:-1]
    abs_changes = abs_diffs[1:] - abs_diffs[:-1]

    rel_abs_changes = np.insert(rel_abs_changes, 0, 0)
    abs_changes = np.insert(abs_changes, 0, 0)
    idxs = np.array([i for i in range(len(layer_names))])

    filtered_rel_abs_mask = rel_abs_changes > 5e-4
    filtered_rel_abs_changes = rel_abs_changes[filtered_rel_abs_mask]
    filtered_rel_abs_diffs = rel_abs_diffs[filtered_rel_abs_mask]
    filtered_rel_abs_names = layer_names[filtered_rel_abs_mask]
    filtered_rel_abs_idxs = idxs[filtered_rel_abs_mask]

    filtered_abs_mask = abs_changes > 5e-4
    filtered_abs_changes = abs_changes[filtered_abs_mask]
    filtered_abs_diffs = abs_diffs[filtered_abs_mask]
    filtered_abs_names = layer_names[filtered_abs_mask]
    filtered_abs_idxs = idxs[filtered_abs_mask]

    colors = plt.cm.tab20.colors

    plt.figure(figsize=(12, 5))
    plt.tight_layout()

    plt.subplot(2, 2, 1)
    plt.plot(idxs, rel_abs_diffs, label='rel_abs_diffs')
    for i in range(len(filtered_rel_abs_names)):
        plt.scatter(filtered_rel_abs_idxs[i], filtered_rel_abs_diffs[i], 
                color=colors[i % len(colors)],  
                label=filtered_rel_abs_names[i],  # 设置标签
                s=100,  # 点大小
                edgecolors='black')  # 添加边框
    plt.title('rel_abs_diffs')
    plt.xlabel('idx')
    plt.ylabel('value')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(idxs, rel_abs_changes, label='rel_abs_changes')
    for i in range(len(filtered_rel_abs_names)):
        plt.scatter(filtered_rel_abs_idxs[i], filtered_rel_abs_changes[i], 
                color=colors[i % len(colors)],  
                label=filtered_rel_abs_names[i],  # 设置标签
                s=100,  # 点大小
                edgecolors='black')  # 添加边框
    plt.title('rel_abs_changes')
    plt.xlabel('idx')
    plt.ylabel('value')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(idxs, abs_diffs, label='abs_diffs')
    for i in range(len(filtered_abs_names)):
        plt.scatter(filtered_abs_idxs[i], filtered_abs_diffs[i], 
                color=colors[i % len(colors)],  
                label=filtered_abs_names[i],  # 设置标签
                s=100,  # 点大小
                edgecolors='black')  # 添加边框
    plt.title('abs_diffs')
    plt.xlabel('idx')
    plt.ylabel('value')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(idxs, abs_changes, label='abs_changes')
    for i in range(len(filtered_abs_names)):
        plt.scatter(filtered_abs_idxs[i], filtered_abs_changes[i], 
                color=colors[i % len(colors)],  
                label=filtered_abs_names[i],  # 设置标签
                s=100,  # 点大小
                edgecolors='black')  # 添加边框
    plt.title('abs_changes')
    plt.xlabel('idx')
    plt.ylabel('value')
    plt.legend()

    plt.show()