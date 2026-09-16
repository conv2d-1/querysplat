import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from PIL import Image

from hAlgorithm.datasets.base_dataset import BaseDataset


class DynamicStereoDataset(BaseDataset):
    def __init__(
        self,
        name: str = "dynamicstereo",
        depth_scale: float = 1.0,
        min_depth: float = 1e-5,
        max_depth: float = 100,
        **kwargs,
    ):
        super().__init__(
            name=name,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            **kwargs,
        )

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


if __name__ == "__main__":

    dataset = DynamicStereoDataset(
        phase="train",
        name="dynamicstereo",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/DynamicStereo/test.json",
        sampling_strategy="all",
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=630,
                height=476,
            ),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip",
                prob=0.5,
            ),
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
                width=630,
                height=476,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        depth_scale=1.0,
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
        recalculate_normal=True,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=True,
    )

    for index in range(50):
        print(index)
        dataset.__getitem__(index)
