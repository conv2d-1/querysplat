import os
import sys

sys.path.append(os.getcwd())

import math

from hAlgorithm.datasets4d.lidar_depth_dataset import LidarDepthDataset4D


class ZedDataset4D(LidarDepthDataset4D):

    def __init__(self, fx=1122.73, fy=959.75, fov_x=60, fov_y=46.9, **kwargs):
        super().__init__(**kwargs)

        self.crop_width = int(math.tan(fov_x / 180 * math.pi / 2) * fx * 2)
        self.crop_height = int(math.tan(fov_y / 180 * math.pi / 2) * fy * 2)

    def load_data(self, data_info):
        data_batch = super().load_data(data_info)

        intrinsics = data_batch["curr_intrinsics"]
        image = data_batch["curr_rgb"]
        depth = data_batch["curr_depth"]
        depth_mask = data_batch["curr_depth_mask"]
        prompt = data_batch["curr_prompt"]

        CROP_HEIGHT = self.crop_height
        CROP_WIDTH = self.crop_width

        height, width, _ = image.shape
        top_margin = int((height - CROP_HEIGHT) / 2)
        left_margin = int((width - CROP_WIDTH) / 2)

        image = image[
            top_margin : top_margin + CROP_HEIGHT,
            left_margin : left_margin + CROP_WIDTH,
        ]

        if intrinsics is not None:
            intrinsics[2] = image.shape[1] / 2.0
            intrinsics[3] = image.shape[0] / 2.0

        if depth is not None:
            assert len(depth.shape) == 2
            depth = depth[
                top_margin : top_margin + CROP_HEIGHT,
                left_margin : left_margin + CROP_WIDTH,
            ]

        if depth_mask is not None:
            assert len(depth_mask.shape) == 2
            depth_mask = depth_mask[
                top_margin : top_margin + CROP_HEIGHT,
                left_margin : left_margin + CROP_WIDTH,
            ]

        if prompt is not None:
            assert len(prompt.shape) == 2
            prompt = prompt[
                top_margin : top_margin + CROP_HEIGHT,
                left_margin : left_margin + CROP_WIDTH,
            ]

        data_batch["curr_intrinsics"] = intrinsics
        data_batch["curr_rgb"] = image
        data_batch["curr_depth"] = depth
        data_batch["curr_depth_mask"] = depth_mask
        data_batch["curr_prompt"] = prompt

        return data_batch


if __name__ == "__main__":

    dataset = ZedDataset4D(
        phase="test",
        name="lidardepth",
        mf_to_sf=True,
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/DL3DV-10K/train_mf_1080p_mini.json",
        lidar_name="sfm_depth",
        sampling_strategy="all",
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=630,
                height=476,
                is_lidar=True,
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
                is_lidar=True,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        sparse_pattern=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.SFMRandomPattern",
            seed=0,
            sparse_nums=6000,
            sfm_prob=1.0,
        ),
        depth_scale=1.0,
        min_depth=1e-3,
        max_depth=60.0,
        with_pointmap=True,
        depth_invalid_sem_names=None,
        recalculate_normal=False,
        debug=True,
    )

    for index in range(min(10, len(dataset))):
        print(index)
        dataset.__getitem__(index)
