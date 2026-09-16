import os
import sys

sys.path.append(os.getcwd())

import logging

import numpy as np
import torch

from hAlgorithm.datasets4d.lidar_depth_dataset import LidarDepthDataset4D
from hAlgorithm.datasets.transforms.transforms import resize_depth_preserve


class IphoneDataset4D(LidarDepthDataset4D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, data_info):
        data_batch = super().load_data(data_info=data_info)

        curr_confidence_path = os.path.join(self.depth_root, data_info["confidence"])
        confidence = self.read_file(curr_confidence_path, image=None, const=0, dtype=np.float32)
        curr_depth_mask = data_batch["curr_depth_mask"]

        if confidence.shape[0:2] != curr_depth_mask.shape[0:2]:
            confidence = resize_depth_preserve(confidence, curr_depth_mask.shape[0:2])

        confidence_mask = confidence == 2
        curr_depth_mask = (curr_depth_mask * confidence_mask).astype(int)
        data_batch["curr_depth_mask"] = curr_depth_mask

        return data_batch


if __name__ == "__main__":

    dataset = IphoneDataset4D(
        phase="test",
        name="iphone",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/iphone_test/test_mf_meeting_maxwell.json",
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=672,
                height=448,
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
                type="hAlgorithm.datasets.transforms.transforms.ResizeSR",
                width=672,
                height=448,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        depth_scale=1.0,
        min_depth=1e-3,
        max_depth=25.0,
        with_pointmap=True,
        debug=True,
        mv_frame_num=8,
        mf_frame_step=10,
        mf_view_ids=[0],
        mv_view_num=1,
    )

    for index in range(len(dataset)):
        print(index)
        dataset.__getitem__(index)
