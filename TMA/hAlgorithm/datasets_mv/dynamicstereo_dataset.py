import os
import sys

sys.path.append(os.getcwd())

import numpy as np

from hAlgorithm.datasets_mv.base_dataset import BaseDatasetMV
from PIL import Image


class DynamicStereoDatasetMV(BaseDatasetMV):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    dataset = DynamicStereoDatasetMV(
        phase="test",
        name="hypersim",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/Benchmark/Depth/metadata/benchmark.json",
        # sampling_strategy="first:10",
        mf_scene="NYUv2",
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=518,
                patch_size=14,
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
            # dict(
            #     type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
            #     max_size=518,
            #     patch_size=14,
            # ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        min_depth=1e-3,
        max_depth=1000.0,
        with_pointmap=True,
        debug=True,
        sparse_pattern=None,
        normalize_cameras=True,
        with_sift_mask=True,
        mf_to_sf=True,
    )

    for index in range(min(10, len(dataset))):
        print(index)
        dataset.__getitem__(index)