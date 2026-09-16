import os, sys
sys.path.append(os.getcwd())
from hAlgorithm.preprocess.hypersim import HypersimPreprocess, preprocess_normal_svd

import numpy as np
from PIL import Image

class DynamicStereoProcessor(HypersimPreprocess):
    def __init__(self, json_path, data_root, output_root, **kwargs):
        super().__init__(json_path, data_root, output_root, **kwargs)
        
    def load_depth(self, depth_path, image, const, dtype):
        data_type = os.path.splitext(depth_path)[-1].lower()
        # List of supported image file types.
        img_file_type = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]
        # Handle different file types.
        if data_type in img_file_type:
            # Open the image using PIL and convert it to a NumPy array.
            data = Image.open(depth_path)
            data = (
                np.frombuffer(np.array(data, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((data.size[1], data.size[0]))
            )
            # Ensure the data is of the specified data type.
            data = data.astype(dtype)
            return data
        else:
            super().load_depth(depth_path, image, const, dtype)

if __name__ == '__main__':
    data_root = '/mnt/netdata/Team/AI/datasets/TMD/'
    output_root = '/mnt/netdata/Team/AI/datasets/TMD/'
    output_meta_root = '/mnt/netdata/Team/AI/datasets/TMA_preprocess'
    version = 'v1_251016'
    
    dataset_name = 'dynamicstereo'
    json_paths = [
        '/mnt/netdata/Team/AI/datasets/TMD/DynamicStereo/test.json',
    ]
    
    preprocess_funcs = [
        preprocess_normal_svd
    ]
    for json_path in json_paths:
        json_name = os.path.basename(json_path).split(".")[0]
        preprocessor = DynamicStereoProcessor(
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
        processed_infos = [None] * n_items
        
        max_workers = 12
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(preprocess_single_item, i) for i in range(n_items)]
            
            # 使用 tqdm 包装 as_completed 以显示进度
            for future in tqdm(as_completed(futures), total=n_items, desc=f'Preprocessing dataset {dataset_name}, split: {json_name}'):
                idx, result = future.result()
                processed_infos[idx] = result
                
        # for i in tqdm(range(len(preprocessor)), desc='preprocessing dataset', total=len(preprocessor)):
        #     processed_infos[i] = preprocessor.preprocess_items(i, preprocess_funcs)
        os.makedirs(f'{output_meta_root}/{version}', exist_ok=True)
        preprocessor.save_json(processed_infos, f'{output_meta_root}/{version}/{dataset_name}_{json_name}.json')