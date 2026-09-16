import torch
import os
import numpy as np
from PIL import Image

ADE20K_LABELS = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 
    'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 
    'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 
    'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 
    'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 
    'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 
    'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 
    'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 
    'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 
    'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 
    'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
    'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator',
    'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 
    'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 
    'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 
    'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 
    'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 
    'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor',
    'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag'
]


def get_ade20k_palette():
    """Returns the color palette used in ADE20K dataset."""
    return [
        [120, 120, 120],
        [182, 218, 119],
        [255, 255, 0],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 60],
        [100, 100, 142],
        [70, 130, 180],
        [153, 153, 190],
        [153, 153, 153],
        [30, 30, 150],
        [220, 20, 60],
        [255, 0, 100],
        [255, 0, 200],
        [100, 80, 0],
        [220, 220, 0],
        [45, 60, 150],
        [128, 64, 128],
        [110, 110, 110],
        [0, 0, 70],
        [102, 102, 156],
        [255, 99, 71],
        [90, 30, 150],
        [150, 120, 90],
        [150, 90, 60],
        [150, 150, 90],
        [150, 120, 60],
        [153, 153, 153],
        [30, 30, 30],
        [180, 120, 120],
        [180, 180, 0],
        [180, 180, 180],
        [150, 180, 180],
        [180, 150, 180],
        [150, 180, 90],
        [180, 120, 120],
        [0, 150, 140],
        [0, 170, 120],
        [250, 80, 100],
        [255, 85, 255],
        [0, 150, 160],
        [160, 30, 100],
        [100, 100, 150],
        [100, 120, 160],
        [150, 30, 100],
        [255, 0, 255],
        [255, 0, 200],
        [200, 0, 130],
        [150, 0, 255],
        [255, 0, 255],
        [255, 0, 100],
        [255, 50, 255],
        [255, 0, 0],
        [100, 200, 210],
        [210, 200, 100],
        [100, 210, 200],
        [0, 255, 200],
        [200, 220, 255],
        [255, 0, 100],
        [255, 0, 200],
        [100, 0, 255],
        [200, 255, 255],
        [0, 0, 255],
        [255, 255, 0],
        [255, 255, 255],
        [0, 0, 0],
    ]

def apply_cls_cmap(result):
    # 将分割结果转换为伪彩色图
    palette = get_ade20k_palette()
    result_color = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.float_)

    for label, color in enumerate(palette):
        result_color[result == label] = color
    
    return result_color

def save_seg_image(img: torch.Tensor, result: torch.Tensor, output_dir, filename_prefix, data_idx):

    """
    将输入图像和分割结果保存为图片
    :param img: 输入图像的 ndarray (H, W, C)
    :param result: 分割结果的 ndarray (H, W)
    :param output_dir: 输出目录
    :param filename_prefix: 文件名前缀
    """
    img = img.squeeze(0).permute(1, 2, 0).cpu()
    h, w = img.shape[:2]
    result = result.cpu()

    if result.shape[0] != h or result.shape[1] != w:
        result = torch.nn.functional.interpolate(
            result, size=(h, w), mode='bilinear'
        )

    result = result.squeeze().numpy()
    img = img.numpy()

    # 将分割结果转换为伪彩色图
    palette = get_ade20k_palette()
    result_color = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.float_)

    for label, color in enumerate(palette):
        result_color[result == label] = color
    result_color = (result_color * 0.5 + img * 0.5).astype(np.uint8)
    segmented_img = Image.fromarray(result_color)
    segmented_img.save(os.path.join(output_dir, f"{filename_prefix}_{data_idx:06d}_segmented.png"))