import os
import sys

sys.path.append(os.getcwd())

import cv2
import h5py
import numpy as np
from PIL import Image

from hAlgorithm.datasets.base_dataset import BaseDataset


class KenBurnsDataset(BaseDataset):
    def __init__(
        self,
        name: str = "kenburns",
        depth_scale: float = 100.0,
        min_depth: float = 1e-5,
        max_depth: float = 65.0,
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

    dataset = KenBurnsDataset(
        phase="test",
        name="kenburns",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/KenBurns/train.json",
        sampling_strategy="all",
        train_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
            dict(type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip", prob=0.5),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
            # x / 255 * 2 - 1
        ],
        test_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        depth_scale=100.0,
        min_depth=1e-5,
        max_depth=1000.0,
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_edge_mask=False,
        with_pointmap=False,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=False,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=True,
    )

    for index in range(10):
        print(index)
        dataset.__getitem__(index)
