import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from PIL import Image

from hAlgorithm.datasets4d.base_dataset import BaseDataset4D


class CO3D(BaseDataset4D):
    def __init__(
        self,
        name: str = "co3d",
        min_depth: float = 1e-5,
        max_depth: float = 100,
        **kwargs,
    ):
        super().__init__(
            name=name,
            min_depth=min_depth,
            max_depth=max_depth,
            **kwargs,
        )

    def load_depth(self, depth_path, image, const, dtype):
        with Image.open(depth_path) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )
            depth = depth.astype(dtype)
        return depth


if __name__ == "__main__":

    dataset = CO3D(
        phase="test",
        name="co3d",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/CO3D_v2/train_mf_v2_0410.json",
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=952,
                patch_size=14,
                is_lidar=False,
                low_resolution=True,
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
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=952,
                patch_size=14,
                is_lidar=False,
                low_resolution=True,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        min_depth=1e-3,
        max_depth=25.0,
        with_pointmap=True,
        depth_invalid_sem_names=None,
        recalculate_normal=False,
        debug=True,
        sparse_max_size=546,
        sparse_patch_size=14,
        sparse_pattern=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=0,
            sparse_nums=6000,
        ),
        mf_to_sf=True,
        mf_view_ids=[0],
        mf_frame_ids=[i * 10 for i in range(100)],
        # mask_name="no_mask",
    )

    for index in range(10):
        print(index)
        dataset.__getitem__(index)
