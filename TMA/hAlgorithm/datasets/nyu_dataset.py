import os
import sys

sys.path.append(os.getcwd())

import torch

from hAlgorithm.datasets.base_dataset import BaseDataset


class NYUDataset(BaseDataset):
    def __init__(
        self,
        name: str = "nyu",
        eigen_valid_mask: bool = True,
        depth_scale: float = 1000.0,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            name=name,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            **kwargs,
        )
        self.eigen_valid_mask = eigen_valid_mask

    def get_valid_mask(self, valid_mask):
        assert len(valid_mask.shape) == 3

        _, height, width = valid_mask.shape
        eval_mask = torch.zeros_like(valid_mask)

        eval_mask[:, 45:471, 41:601] = 1

        return eval_mask

    def process_depth(self, depth, depth_mask, intrinsics, sky_mask):
        """
        Process the depth map, including normalization, inverse depth calculation, and point cloud generation.

        Args:
            depth (torch.Tensor): The raw depth map as a tensor.
            depth_mask (torch.Tensor or None): A boolean mask indicating valid depth values.
            intrinsics (torch.Tensor): Intrinsic camera parameters for point cloud generation.
            sky_mask (torch.Tensor or None): A boolean mask indicating the sky region to be excluded from processing.

        Returns:
            tuple: A tuple containing the normalized depth map, normalized inverse depth map,
                the mask for valid inverse depth values, and the point cloud map if enabled.
        """
        depth_output = super().process_depth(depth, depth_mask, intrinsics, sky_mask)

        if self.eigen_valid_mask is not None:
            depth_mask = depth_output["depth_mask"]
            eval_mask = self.get_valid_mask(depth_mask)
            depth_output["depth_mask"] = torch.logical_and(depth_mask, eval_mask)
            depth_output["inv_depth_mask"] = torch.logical_and(
                depth_output["inv_depth_mask"], eval_mask
            )
            if self.sparse_depth_ratio > 0:
                (depth_output["sparse_depth_mask"],) = torch.logical_and(
                    depth_output["sparse_depth_mask"], eval_mask
                )

        return depth_output


if __name__ == "__main__":

    dataset = NYUDataset(
        phase="train",
        name="nyu",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/NYUV2/test.json",
        sampling_strategy="all",
        train_transforms=[
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
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        depth_scale=1000.0,
        min_depth=1e-3,
        max_depth=10.0,
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
        eigen_valid_mask=True,
    )

    for index in range(50):
        print(index)
        dataset.__getitem__(index)
