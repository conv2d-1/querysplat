import os
import sys

sys.path.append(os.getcwd())

import logging

import numpy as np
import torch

from hAlgorithm.datasets.transforms.edge_filter import edge_filter
from hAlgorithm.datasets_mv.base_dataset import BaseDatasetMV


class LidarDepthDatasetMV(BaseDatasetMV):

    def __init__(
        self,
        name: str = "lidardepth",
        lidar_name="lidar_depth",
        reset_sparse_pattern=True,
        **kwargs,
    ):
        super().__init__(
            name=name,
            lidar_name=lidar_name,
            **kwargs,
        )

        self.reset_sparse_pattern = reset_sparse_pattern

        if self.reset_sparse_pattern and self.sparse_pattern_v3 is not None:
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
        **kwargs
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

                if sparse_pointmap is not None:
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
                else:
                    # NOTE: 沿用 sparse_pointmap_max_range
                    sparse_pointmap_center, sparse_pointmap_max_range = self.point_normalize(
                        pointmap, depth_mask
                    )

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

    import yaml
    import random

    from hAlgorithm.utils import instantiate_from_config
    path = "hAlgorithm/configs/mv_v1.0/dataset_configs_real/test_dl3dv_1080p.yaml"

    with open(path, "r") as file:
        yaml_data = yaml.safe_load(file)
    dataset_configs = yaml_data.get("datasets", [])

    for config in dataset_configs:
        # if config["name"] != "scpp":
        #     continue

        base_config = dict(
            phase="test",
            seed=0,
            test_transforms=[
                dict(
                    type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                    max_size=504,
                    patch_size=14,
                    # is_lidar=True,
                ),
                dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
                dict(
                    type="hAlgorithm.datasets.transforms.transforms.Normalize",
                    mean=[127.5, 127.5, 127.5],
                    std=[127.5, 127.5, 127.5],
                    ),
            ],
            min_depth=1e-3,
            max_depth=1000.0,
            sparse_pattern=None,
            mf_to_sf=False,
            debug=True,
        )
        config.update(base_config)
        # config["clip_sampler"]["shuffle"] = False

        dataset = instantiate_from_config(config)
        print(config["name"], len(dataset))
        data = dataset[0]
        indices = random.sample(range(len(dataset)), k=min(5, len(dataset)))
        indices = [0, 1, 2]

        for index in indices:
            print(index)
            dataset.__getitem__(index)