import os, sys
import json

sys.path.append(os.getcwd())
from hAlgorithm.preprocess.hypersim import HypersimPreprocess, preprocess_normal_svd

class HabitatProcessor(HypersimPreprocess):
    def __init__(self, json_path, data_root, output_root, **kwargs):
        super().__init__(json_path, data_root, output_root, **kwargs)
    
    def load_json(self, json_path):
        data = []
        with open(json_path, 'r') as f:
            self.original_mf_infos = json.load(f)["mf_files"]
            for scene, frames in self.original_mf_infos.items():
                for frame in frames:
                    views = frame["views"]
                    for view in views:
                        data.append(view)
        return data
    
    def save_json(self, data_info, json_path):
        idx = 0
        for scene, frames in self.original_mf_infos.items():
            for frame in frames:
                views = frame["views"]
                for view in views:
                    view.update(data_info[idx])
                    idx += 1
        
        with open(json_path, 'w') as f:
            json.dump({"mf_files": self.original_mf_infos}, f, indent=2)
    

if __name__ == '__main__':
    data_root = '/mnt/netdata/Team/AI/datasets/TMD/'
    output_root = '/mnt/netdata/Team/AI/datasets/TMD/'
    output_meta_root = '/mnt/netdata/Team/AI/datasets/TMA_preprocess'
    version = 'v1_251016'
    
    dataset_name = 'habitat'
    json_paths = [
        '/mnt/netdata/Team/AI/datasets/TMD/Habitat-Sim-v2/train_mf_v2_0401_part1_filter_black_50.json',
        '/mnt/netdata/Team/AI/datasets/TMD/Habitat-Sim-v2/train_mf_v2_0401_part2_filter_black_50.json',
        '/mnt/netdata/Team/AI/datasets/TMD/Habitat-Sim-v2/train_mf_v2_0403_part1_filter_black_50.json',
        '/mnt/netdata/Team/AI/datasets/TMD/Habitat-Sim-v2/train_mf_v2_0403_part2_filter_black_50.json',
    ]
    debug = False
    
    
    preprocess_funcs = [
        preprocess_normal_svd
    ]
    for json_path in json_paths:
        json_name = os.path.basename(json_path).split(".")[0]
        preprocessor = HabitatProcessor(
            json_path,
            data_root,
            output_root,
        )
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        def preprocess_single_item(idx):
            """线程安全的单条目预处理函数"""
            return idx, preprocessor.preprocess_items(idx, preprocess_funcs)
        
        # 初始化结果列表（保持原始顺序）
        n_items = len(preprocessor)
        if debug:
            n_items = 100
        processed_infos = [None] * n_items
        
        max_workers = 12
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(preprocess_single_item, i) for i in range(n_items)]
            
            # 使用 tqdm 包装 as_completed 以显示进度
            for future in tqdm(as_completed(futures), total=n_items, desc=f'Preprocessing dataset {dataset_name}, split: {json_name}'):
                idx, result = future.result()
                processed_infos[idx] = result
        if not debug:
            os.makedirs(f'{output_meta_root}/{version}', exist_ok=True)
            preprocessor.save_json(processed_infos, f'{output_meta_root}/{version}/{dataset_name}_{json_name}.json')