import os
import sys

sys.path.append(os.getcwd())

import re

import cv2
import h5py
import numpy as np
import torch
from PIL import Image

from hAlgorithm.datasets.base_dataset import BaseDataset
from hAlgorithm.datasets.transforms.transforms import resize_depth_preserve


class ARKitDataset(BaseDataset):
    def __init__(
        self,
        name: str = "arkit",
        depth_scale: float = 1000.0,
        min_depth: float = 1e-5,
        max_depth: float = 1000,
        **kwargs,
    ):
        super().__init__(
            name=name,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            **kwargs,
        )

    def load_data(self, data_info):
        data_batch = super().load_data(data_info)

        curr_rgb = data_batch["curr_rgb"]
        curr_depth = data_batch["curr_depth"]
        curr_depth_mask = data_batch["curr_depth_mask"]

        H, W, _ = curr_rgb.shape
        curr_depth = resize_depth_preserve(curr_depth, (H, W))
        curr_depth_mask = resize_depth_preserve(curr_depth_mask, (H, W))

        data_batch["curr_rgb"] = curr_rgb
        data_batch["curr_depth"] = curr_depth
        data_batch["curr_depth_mask"] = curr_depth_mask

        return data_batch


class ARKitHighResDataset(BaseDataset):
    def __init__(
        self,
        name: str = "arkit_highres",
        depth_scale: float = 1000.0,
        min_depth: float = 1e-5,
        max_depth: float = 1000,
        **kwargs,
    ):
        super().__init__(
            name=name,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            **kwargs,
        )


if __name__ == "__main__":
    dataset = ARKitDataset(
        phase="train",
        name="arkit",
        seed=0,
        data_root="/mnt/personal/TMD/datasets",
        data_path="/mnt/personal/TMD/datasets/ARKitScenes/train_highres_new.json",
        sampling_strategy="all",
        interpolate_version="v2",
        train_transforms=[
            # dict(
            #     type="hAlgorithm.datasets.transforms.transforms.RandomCrop",
            #     crop_size=(476, 630),
            #     crop_type='center',
            #     padding=[0, 0, 0],
            # ),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=630,
                height=476,
            ),
            # dict(
            #     type='hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio',
            #     resize_size=(476, 630),
            #     padding=[0, 0, 0],
            # ),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip",
                prob=0.5,
            ),
            # dict(
            #     type="hAlgorithm.datasets.transforms.metric3d_transforms.PhotoMetricDistortion",  # NOTE
            #     to_gray_prob=1.0,
            #     distortion_prob=1.0,
            # ),
            # dict(
            #     type="hAlgorithm.datasets.transforms.metric3d_transforms.Weather", prob=1.0  # NOTE
            # ),
            # dict(
            #     type="hAlgorithm.datasets.transforms.metric3d_transforms.RandomBlur", prob=1.0 # NOTE
            # ),
            # dict(
            #     type="hAlgorithm.datasets.transforms.metric3d_transforms.RGBCompresion",
            #     prob=1.0, compression=(0, 50)
            # ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
            # x / 255 * 2 - 1
        ],
        test_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=640,
                height=480,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        depth_scale=1000.0,
        min_depth=1e-5,
        max_depth=1000.0,
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_pointmap=True,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=False,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        with_edge_mask=False,
        debug=True,
        point_normalize_mode="distance",
        sparse_pattern=dict(
            type="hAlgorithm.datasets.patterns.lidar_pattern.LidarPattern",
            csv_path="/mnt/personal/zm/VLidar/Config/ft2_pattern.csv",
            blur_t=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.BlurPointmap",
                scale_range=(0.3, 0.8),
                p=1.0,
            ),
            pts_jitter_t=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.Random3DJitter",
                noise_std=0.01,
                p=1.0,
            ),
            uv_jitter_t=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.RandomUVJitter",
                max_jitter=5,
                jitter_ratio=(0.05, 0.5),
                p=1.0,
            ),
            patch_crop_t=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.PatchCropMask",
                patch_range=(0.05, 0.25),
                p=1.0,
            ),
        ),
    )
    from hAlgorithm.utils.vis_util import visualize_batch

    for index in range(10):
        print(index)
        batch = dataset.__getitem__(index)
        # visualize_batch(batch)
