import os, sys
sys.path.append(os.getcwd())
from hAlgorithm.preprocess.hypersim import HypersimPreprocess, preprocess_normal_svd

import numpy as np
import cv2

class SynthiaProcessor(HypersimPreprocess):
    def __init__(self, json_path, data_root, output_root, **kwargs):
        super().__init__(json_path, data_root, output_root, **kwargs)
        
    def load_depth(self, depth_path, image, const, dtype):
        # Return None if both file_path and image are None.
        if depth_path is None and image is None:
            return None

        # Create a constant-valued array based on the image dimensions if file_path is None.
        if depth_path is None or not os.path.exists(depth_path):
            data = np.zeros(image.shape, dtype=dtype) + const
        else:
            image_fp = cv2.imread(depth_path).astype(np.float32)
            data = (
                5000
                * (image_fp[..., 2] + image_fp[..., 1] * 256 + image_fp[..., 0] * 256 * 256)
                / (256 * 256 * 256 - 1)
            )
        return data

if __name__ == '__main__':
    data_root = '/mnt/netdata/Team/AI/datasets/TMD/'
    output_root = '/mnt/netdata/Team/AI/datasets/TMD/'
    output_meta_root = '/mnt/netdata/Team/AI/datasets/TMA_preprocess'
    version = 'v1_251016'
    
    dataset_name = 'synthia'
    json_paths = [
        '/mnt/netdata/Team/AI/datasets/TMD/Synthia/test.json',
        '/mnt/netdata/Team/AI/datasets/TMD/Synthia/train.json',
    ]
    debug = False
    
    preprocess_funcs = [
        preprocess_normal_svd
    ]
    for json_path in json_paths:
        json_name = os.path.basename(json_path).split(".")[0]
        preprocessor = SynthiaProcessor(
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