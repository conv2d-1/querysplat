import os
import sys

sys.path.append(os.getcwd())

import torch

from hAlgorithm.datasets.base_dataset import BaseDataset


class KITTIDataset(BaseDataset):
    def __init__(
        self,
        name: str = "kitti",
        valid_mask_crop: str = None,
        depth_scale: float = 256.0,
        min_depth: float = 1e-5,
        max_depth: float = 80.0,
        **kwargs,
    ):
        super().__init__(
            name=name,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            **kwargs,
        )
        self.valid_mask_crop = valid_mask_crop

    def get_valid_mask(self, valid_mask):
        # reference: https://github.com/cleinc/bts/blob/master/pytorch/bts_eval.py

        assert len(valid_mask.shape) == 3

        _, height, width = valid_mask.shape
        eval_mask = torch.zeros_like(valid_mask)

        if "garg" == self.valid_mask_crop:
            eval_mask[
                :,
                int(0.40810811 * height) : int(0.99189189 * height),
                int(0.03594771 * width) : int(0.96405229 * width),
            ] = 1
        elif "eigen" == self.valid_mask_crop:
            eval_mask[
                :,
                int(0.3324324 * height) : int(0.91351351 * height),
                int(0.0359477 * width) : int(0.96405229 * width),
            ] = 1

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
            tuple: A Dict containing the normalized depth map, normalized inverse depth map,
                the mask for valid inverse depth values, and the point cloud map if enabled.
        """
        depth_output = super().process_depth(depth, depth_mask, intrinsics, sky_mask)

        if self.valid_mask_crop is not None:
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

    dataset = KITTIDataset(
        phase="train",
        name="kitti",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/Kitti/test.json",
        sampling_strategy="all",
        train_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.KittiBenchmarkCrop"),
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
            dict(type="hAlgorithm.datasets.transforms.transforms.KittiBenchmarkCrop"),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        depth_scale=256.0,
        min_depth=1e-5,
        max_depth=80.0,
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
        valid_mask_crop="eigen",  # valid_mask_crop for test,
    )

    for index in range(50):
        print(index)
        dataset.__getitem__(index)
