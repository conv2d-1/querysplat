import os
import sys

sys.path.append(os.getcwd())

import logging

import numpy as np
import torch

from hAlgorithm.datasets.base_dataset import BaseDataset
from hAlgorithm.datasets.transforms.edge_filter import edge_filter


class LidarDepthDataset(BaseDataset):

    def __init__(
        self,
        name: str = "lidardepth",
        lidar_name="lidar_depth",
        depth_scale: float = 1.0,
        min_depth: float = 1e-3,
        max_depth: float = 200.0,
        confidence_thresh: float = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            lidar_name=lidar_name,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            **kwargs,
        )
        self.confidence_thresh = confidence_thresh

        if self.sparse_pattern is not None:
            if hasattr(self.sparse_pattern, "sparse_ratio"):
                self.sparse_pattern.sparse_ratio = 1.0
            if hasattr(self.sparse_pattern, "sparse_nums"):
                self.sparse_pattern.sparse_nums = None
            if hasattr(self.sparse_pattern, "is_lidar"):
                self.sparse_pattern.is_lidar = True

        if self.sparse_pattern_v2 is not None:
            if hasattr(self.sparse_pattern_v2, "sparse_ratio"):
                self.sparse_pattern_v2.sparse_ratio = 1.0
            if hasattr(self.sparse_pattern_v2, "sparse_nums"):
                self.sparse_pattern_v2.sparse_nums = None
            if hasattr(self.sparse_pattern_v2, "is_lidar"):
                self.sparse_pattern_v2.is_lidar = True

        if self.sparse_pattern_v3 is not None:
            if hasattr(self.sparse_pattern_v3, "sparse_ratio"):
                self.sparse_pattern_v3.sparse_ratio = 1.0
            if hasattr(self.sparse_pattern_v3, "sparse_nums"):
                self.sparse_pattern_v3.sparse_nums = None
            if hasattr(self.sparse_pattern_v3, "is_lidar"):
                self.sparse_pattern_v3.is_lidar = True

        for transform in self.data_transforms.transforms:
            if transform.__class__.__name__ in [
                "Resize",
                "ResizeSR",
                "ResizeKeepRatio",
                "ResizePatch",
            ]:
                transform.is_lidar = True

    def process_depth(
        self,
        depth,
        depth_mask,
        intrinsics,
        sparse_pointmap=None,
        sparse_pointmap_mask=None,
    ):
        """
        Process the depth map, including normalization, inverse depth calculation, and point cloud generation.

        Args:
            depth (torch.Tensor): The raw depth map as a tensor.
            depth_mask (torch.Tensor or None): A boolean mask indicating valid depth values.
            intrinsics (torch.Tensor): Intrinsics camera parameters for point cloud generation.

        Returns:
            tuple: A Dict containing the normalized depth map, normalized inverse depth map,
                the mask for valid inverse depth values, and the point cloud map if enabled.
        """
        # Initialize output variables to None; they will be set only if depth is not None.
        depth_norm = inv_depth_norm = inv_depth_mask = pointmap = None
        sparse_depth = sparse_depth_norm = sparse_depth_mask = None
        sparse_depth_min = sparse_depth_max = None
        sparse_dense_pointmap = None
        sparse_pointmap_center = sparse_pointmap_max = sparse_pointmap_min = (
            sparse_pointmap_max_range
        ) = None
        edge_mask = None
        sparse_dense_abs_diffmap = None

        # Only proceed if the depth map is provided.
        if depth is not None:
            depth_mask = depth_mask.bool()

            if not self.only_pointmap:
                # Normalize the depth map using the provided normalize_depth function and the updated depth_mask.
                if self.normalize_depth is not None:
                    depth_norm = self.normalize_depth(depth, depth_mask)

            # If point cloud generation is enabled, create a point cloud map using the depth, intrinsics parameters, and depth mask.
            if self.with_pointmap or self.only_pointmap:
                pointmap = self.load_pointmap(depth, intrinsics)

                assert sparse_pointmap is not None
                if (
                    self.sparse_depth_ratio > 0 and self.sparse_depth_ratio < 1.0
                ) or self.sparse_depth_nums > 0:
                    sparse_pointmap, sparse_pointmap_mask = self.load_sparse_depth(
                        sparse_pointmap, sparse_pointmap_mask
                    )
                    sparse_pointmap_mask = sparse_pointmap_mask[0:1, ...]

                sparse_pointmap_center, sparse_pointmap_max_range = self.point_normalize(
                    sparse_pointmap, sparse_pointmap_mask
                )

                if self.interpolate_version == "v1":
                    sparse_dense_pointmap = self.depth_interpolate_torch(
                        sparse_pointmap, depth_mask
                    )
                elif self.interpolate_version == "v2":
                    sparse_dense_pointmap = self.depth_interpolate_torch_v2(
                        sparse_pointmap, sparse_pointmap_mask.squeeze(0)
                    )
                elif self.interpolate_version == "v3":
                    sparse_dense_pointmap = self.depth_interpolate_torch_v3(
                        sparse_pointmap, sparse_pointmap_mask.squeeze(0), intrinsics=intrinsics
                    )
                if sparse_dense_pointmap is not None and torch.isnan(sparse_dense_pointmap.sum()):
                    logging.warning(
                        "sparse_dense_pointmap contains NaN values, no valid points found."
                    )
                    raise ValueError("No valid points in the sparse_dense_pointmap.")

            if sparse_dense_pointmap is not None and pointmap is not None:
                sparse_dense_abs_diffmap = torch.abs(pointmap - sparse_dense_pointmap)

            if self.with_depth_edge_mask:
                edge_mask = edge_filter(depth, valid_mask=depth_mask, times=0.05)

        depth_output = {
            "depth": depth,
            "depth_norm": depth_norm,
            "depth_mask": depth_mask,
            "inv_depth_norm": inv_depth_norm,
            "inv_depth_mask": inv_depth_mask,
            "sparse_depth": sparse_depth,
            "sparse_depth_norm": sparse_depth_norm,
            "sparse_depth_mask": sparse_depth_mask,
            "sparse_depth_max": sparse_depth_max,
            "sparse_depth_min": sparse_depth_min,
            "pointmap": pointmap,
            "sparse_pointmap": sparse_pointmap,
            "sparse_pointmap_mask": sparse_pointmap_mask,
            "sparse_pointmap_center": sparse_pointmap_center,
            "sparse_pointmap_max": sparse_pointmap_max,
            "sparse_pointmap_min": sparse_pointmap_min,
            "sparse_pointmap_max_range": sparse_pointmap_max_range,
            "edge_mask": edge_mask,
            "sparse_dense_pointmap": sparse_dense_pointmap,
            "sparse_dense_abs_diffmap": sparse_dense_abs_diffmap,
        }
        return depth_output


if __name__ == "__main__":

    dataset = LidarDepthDataset(
        phase="test",
        name="lidardepth",
        mf_to_sf=True,
        # mf_scene= "20240829_152118",
        # mf_frame_id= 2,
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        # data_path="/mnt/netdata/Team/AI/datasets/TMD/Davinci/train_mf.json",
        # data_path="/mnt/netdata/Team/AI/datasets/TMD/Ha2RGBD/test_indoor.json",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/iphone_test/train_mf.json",
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
        depth_scale=1.0,
        min_depth=1e-5,
        max_depth=60.0,
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
        recalculate_normal=False,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=True,
        confidence_thresh=2.0,
    )

    for index in range(min(10, len(dataset))):
        print(index)
        dataset.__getitem__(index)
