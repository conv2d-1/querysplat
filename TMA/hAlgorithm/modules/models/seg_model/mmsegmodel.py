from collections import defaultdict

import mmseg
import numpy as np
import torch
import torch.nn as nn
from mmengine.dataset import Compose
from mmseg.apis import MMSegInferencer, inference_model, init_model
from mmseg.utils import get_classes


def isin(tensor, values):
    """
    替代 torch.isin()，适用于 PyTorch < 1.9
    Args:
        tensor (Tensor): 输入张量
        values (list or Tensor): 要匹配的值列表
    Returns:
        Tensor: 布尔型张量，表示哪些位置属于 values
    """
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, device=tensor.device)
    return (tensor.unsqueeze(-1) == values).any(dim=-1)


"""
cityscapes ,

ade20k:
['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 
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
'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']

"""


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


def debug_seg_model(img, result, output_dir, filename_prefix="output"):
    import os

    import numpy as np
    from PIL import Image

    """
    将输入图像和分割结果保存为图片
    :param img: 输入图像的 ndarray (H, W, C)
    :param result: 分割结果的 ndarray (H, W)
    :param output_dir: 输出目录
    :param filename_prefix: 文件名前缀
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = img.permute(1, 2, 0).numpy()
    result = result.pred_sem_seg.data[0].cpu()

    # 保存原图
    original_img = Image.fromarray(img.astype(np.uint8))
    original_img.save(os.path.join(output_dir, f"{filename_prefix}_original.png"))

    # 将分割结果转换为伪彩色图
    palette = get_ade20k_palette()
    result_color = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)

    for label, color in enumerate(palette):
        result_color[result == label] = color

    segmented_img = Image.fromarray(result_color)
    segmented_img.save(os.path.join(output_dir, f"{filename_prefix}_segmented.png"))

    print(f"Images saved to {output_dir}")


class SegModel:
    def __init__(self, config, weights, cls_name, **kwargs) -> None:
        self.seg_model = MMSegInferencer(config, weights=weights).model
        self.cls_name = cls_name
        self.classes_label = get_classes(self.cls_name)

    def __call__(self, x, **kwargs):
        x, is_batch = self.prepare_image(x, self.seg_model)  # B,C,H,W
        # forward the model
        with torch.no_grad():
            results = self.seg_model.test_step(x)  # B*[1,H,W]

        results = torch.stack([result.pred_sem_seg.data for result in results], dim=0)

        return results  # B,1,H,W

    def get_cls_mask(self, result, cls_names):
        if not isinstance(cls_names, (list, tuple)):
            cls_names = [cls_names]
        cls_idxes = [self.classes_label.index(cls_name) for cls_name in cls_names]
        cls_mask = isin(result, cls_idxes)
        return cls_mask

    def prepare_image(self, imgs, model):
        cfg = model.cfg
        for t in cfg.test_pipeline:
            if t.get("type") == "LoadAnnotations":
                cfg.test_pipeline.remove(t)

        is_batch = True
        imgs = imgs.detach().cpu().clone().float().permute(0, 2, 3, 1).numpy() * 122.5 + 122.5
        imgs = [img for img in imgs]

        assert isinstance(imgs[0], np.ndarray)
        cfg.test_pipeline[0]["type"] = "LoadImageFromNDArray"

        # TODO: Consider using the singleton pattern to avoid building
        # a pipeline for each inference
        pipeline = Compose(cfg.test_pipeline)

        data = defaultdict(list)
        for img in imgs:
            if isinstance(img, np.ndarray):
                data_ = dict(img=img)
            else:
                data_ = dict(img_path=img)
            data_ = pipeline(data_)
            data["inputs"].append(data_["inputs"])
            data["data_samples"].append(data_["data_samples"])

        return data, is_batch


if __name__ == "__main__":
    model = SegModel(
        config="segformer_mit-b5_8xb2-160k_ade20k-512x512",
        weights="/mnt/naspersonal/wsz/mmseg/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth",
        cls_name="ade20k",
    )
    print(model.seg_model)
