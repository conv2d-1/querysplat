import json
import torch
import torch.nn.functional as F
import numpy as np
import os

mid_360_json_paths = [
    '/mnt/nasTeam/Kosmo/json/kosmo/20251114_left_test_split.json',
    '/mnt/nasTeam/Kosmo/json/kosmo/20251115_left_test_split.json',
]

data_infos = {}

for json_path in mid_360_json_paths:
    with open(json_path, 'r') as f:
        data_infos.update(json.load(f)["mf_files"])

total_frames = []
for scene, scene_infos in data_infos.items():
    assert scene_infos["version"] == '1.0'
    frame_num = len(scene_infos["frames"])
    print("Scene:", scene, ",  frame num:", frame_num)
    for frame in scene_infos["frames"]:
        frame.update({'K': scene_infos['cam_params']['cam0']['K']})
    total_frames.extend(scene_infos["frames"])
print(f"Total frames: {len(total_frames)}")

def process_frame(info, idx, eps=1e-3, output_root='/mnt/netdata/Team/AI/datasets/TMA_preprocess/PatternMask/mid360_251112'):
    assert "lidar_depth" in info
    K = info['K']
    lidar = np.load(info["lidar_depth"]) / info.get("depth_scale", 1.0)
    lidar_960 = resize_depth_preserve(lidar, (960, 960))
    lidar_840 = resize_depth_preserve(lidar, (840, 840))
    lidar_518 = resize_depth_preserve(lidar, (518, 518))
    pattern_masks = {
        "1920" : os.path.join(output_root, '1920', f"{idx:08d}.npy"),
        "960" : os.path.join(output_root, '960', f"{idx:08d}.npy"),
        "840" : os.path.join(output_root, '840', f"{idx:08d}.npy"),
        "518" : os.path.join(output_root, '518', f"{idx:08d}.npy"),
    }
    os.makedirs(os.path.dirname(pattern_masks["1920"]), exist_ok=True)
    os.makedirs(os.path.dirname(pattern_masks["960"]), exist_ok=True)
    os.makedirs(os.path.dirname(pattern_masks["840"]), exist_ok=True)
    os.makedirs(os.path.dirname(pattern_masks["518"]), exist_ok=True)
    np.save(pattern_masks["1920"], (lidar > eps))
    np.save(pattern_masks["960"], (lidar_960 > eps))
    np.save(pattern_masks["840"], (lidar_840 > eps))
    np.save(pattern_masks["518"], (lidar_518 > eps))
    return pattern_masks

def resize_depth_preserve(depth, shape):
    """
    Resizes depth map preserving all valid depth pixels
    Multiple downsampled points can be assigned to the same pixel.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W,1]
        Resized depth map
    """
    # Store dimensions and reshapes to single column
    depth = np.squeeze(depth)
    h, w = depth.shape
    x = depth.reshape(-1)
    # Create coordinate grid
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    # Filters valid points
    idx = x > 0
    crd, val = uv[idx], x[idx]
    # Downsamples coordinates
    crd[:, 0] = (crd[:, 0] * (shape[0] / h) + 0.5).astype(np.int32)
    crd[:, 1] = (crd[:, 1] * (shape[1] / w) + 0.5).astype(np.int32)
    # Filters points inside image
    idx = (crd[:, 0] < shape[0]) & (crd[:, 1] < shape[1])
    crd, val = crd[idx], val[idx]
    # Creates downsampled depth image and assigns points
    depth = np.zeros(shape)
    depth[crd[:, 0], crd[:, 1]] = val
    # Return resized depth map
    return depth

import random
train_num = 5000
test_num = 1000
total_num = train_num + test_num
random_selected = random.sample(total_frames, total_num)

from tqdm import tqdm
train_pattern = {
    "1920" : [],
    "960" : [],
    "840" : [],
    "518" : [],
}

json_dir="/mnt/netdata/Team/AI/datasets/TMA_preprocess/preprocess_json/PatternMask_251118_left/"
os.makedirs(json_dir, exist_ok=True)

output_root = '/mnt/netdata/Team/AI/datasets/TMA_preprocess/PatternMask/mid360_251118_left'
for i, frame in tqdm(enumerate(random_selected[:train_num]), total=train_num):
    pattern_masks = process_frame(frame, i, output_root=output_root)
    for k, v in pattern_masks.items():
        train_pattern[k].append(v)
with open(f'{json_dir}/mid360_pattern_train.json', 'w') as f:
    json.dump(train_pattern, f, indent=2)

test_pattern = {
    "1920" : [],
    "960" : [],
    "840" : [],
    "518" : [],
}
for i, frame in tqdm(enumerate(random_selected[-test_num:]), total=test_num):
    pattern_masks = process_frame(frame, i+train_num, output_root=output_root)
    for k, v in pattern_masks.items():
        test_pattern[k].append(v)
with open(f'{json_dir}/mid360_pattern_test.json', 'w') as f:
    json.dump(test_pattern, f, indent=2)