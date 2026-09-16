import os
import sys

sys.path.append(os.getcwd())

import numpy as np

from hAlgorithm.datasets.base_dataset import BaseDataset


class TaskonomyDataset(BaseDataset):
    def __init__(
        self,
        name: str = "taskonomy",
        depth_scale: float = 512.0,
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

    def load_normal(self, normal_path, image, const, dtype):
        return (super().load_normal(normal_path, image, const, dtype) - 127) / 128


if __name__ == "__main__":

    dataset = TaskonomyDataset(
        phase="train",
        name="taskonomy",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/Taskonomy/test.json",
        sampling_strategy="random:10",
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
        depth_scale=512.0,
        min_depth=1e-5,
        max_depth=1000.0,
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_edge_mask=True,
        with_pointmap=True,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=False,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=True,
    )

    for index in range(10):
        print(index)
        batch = dataset.__getitem__(index)
