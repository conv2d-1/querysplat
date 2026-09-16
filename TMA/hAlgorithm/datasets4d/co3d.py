import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from PIL import Image

from hAlgorithm.datasets4d.base_dataset_nv import BaseDataset4DNovel


class CO3D(BaseDataset4DNovel):
    def __init__(
        self,
        name: str = "CO3D",
        depth_scale: float = 1.0,
        min_depth: float = 1e-5,
        max_depth: float = 100,
        mask_name="mask",
        **kwargs,
    ):
        super().__init__(
            name=name,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            **kwargs,
        )
        self.mask_name = mask_name

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

    dataset = CO3D(
        phase="test",
        name="co3d",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/CO3D_v2/train_mf_v2_0410.json",
        mf_frame_num=4,
        mf_view_num=1,
        mf_frame_step=30,
        depth_scale=1.0,
        min_depth=1e-3,
        max_depth=30.0,
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=518,
                patch_size=14,
                is_lidar=True,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip", prob=1.5),
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.PhotoMetricDistortion",  # NOTE
                to_gray_prob=0.1,
                distortion_prob=0.05,
            ),
            # dict(type="hAlgorithm.datasets.transforms.metric3d_transforms.Weather", prob=0.1),  # NOTE
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.RandomBlur",
                prob=0.05,  # NOTE
            ),
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.RGBCompresion",
                prob=0.1,
                compression=[0, 50],
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
                max_size=518,
                patch_size=14,
                is_lidar=True,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        with_pointmap=True,
        debug=True,
        sparse_pattern=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=0,
            sparse_ratio=1.0,
        ),
        mf_to_sf=False,
        clip_shuffle=True,
        normalize_cameras=True,
        mask_name="no_mask",
    )

    for index in range(10):
        print(index)
        dataset.__getitem__(index)
