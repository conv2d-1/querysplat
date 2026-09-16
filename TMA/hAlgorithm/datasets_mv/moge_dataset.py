import os
import sys

sys.path.append(os.getcwd())

import numpy as np

from hAlgorithm.datasets.transforms.transforms import resize_depth_preserve
from hAlgorithm.datasets_mv.base_dataset import BaseDatasetMV
from PIL import Image
from pathlib import Path
import io

class MogeDatasetMV(BaseDatasetMV):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.depth_scale = 1.0

    def load_depth(self, depth_path, image, const, dtype):
        """
        Read a depth image, return float32 depth array of shape (H, W).
        """
        if depth_path is None or not os.path.exists(depth_path):
            depth = np.zeros(image.shape, dtype=dtype) + const
        else:
            if isinstance(depth_path, (str, os.PathLike)):
                data = Path(depth_path).read_bytes()
            else:
                data = depth_path.read()
            pil_image = Image.open(io.BytesIO(data))
            near = float(pil_image.info.get('near'))
            far = float(pil_image.info.get('far'))
            # unit = float(pil_image.info.get('unit')) if 'unit' in pil_image.info else None
            depth = np.array(pil_image)
            mask_nan, mask_inf = depth == 0, depth == 65535
            depth = (depth.astype(np.float32) - 1) / 65533
            depth = near ** (1 - depth) * far ** depth
            depth[mask_nan] = 0
            depth[mask_inf] = 0
        return depth

    def load_data(self, data_info):
        assert self.mf_to_sf
        if 'depth_scale' not in data_info:
            data_info['depth_scale'] = self.depth_scale
        if 'extrinsics' not in data_info:
            data_info['extrinsics'] = np.eye(4).tolist()
        data_path = self.load_data_path(data_info)
        curr_rgb = self.read_image(data_path["rgb_path"])
        h, w = curr_rgb.shape[:2]
        moge_intrinsics = data_info[self.intrinsics_name]
        moge_intrinsics[0] = moge_intrinsics[0] * w
        moge_intrinsics[2] = moge_intrinsics[2] * w
        moge_intrinsics[1] = moge_intrinsics[1] * h
        moge_intrinsics[3] = moge_intrinsics[3] * h
        data_info[self.intrinsics_name] = moge_intrinsics
        data_batch = super().load_data(data_info)
        return data_batch



if __name__ == "__main__":

    dataset = MogeDatasetMV(
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