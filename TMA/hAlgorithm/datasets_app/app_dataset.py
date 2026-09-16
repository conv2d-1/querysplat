import os
import sys

sys.path.append(os.getcwd())


from hAlgorithm.datasets.base_dataset import BaseDataset


class APPDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_data_infos(self, data_infos=None):
        # Load the dataset configuration from the JSON file
        if data_infos is None:
            cur_data_infos = self.load_json_file(self.data_path)
        else:
            cur_data_infos = data_infos
        # Parse the sampling strategy to determine how many samples to use
        sampling_number = self._parse_sampling_strategy(self.sampling_strategy, len(cur_data_infos))

        # Apply the sampling strategy if a specific number of samples is requested
        if sampling_number is not None:
            if self.sampling_strategy.startswith("first"):
                cur_data_infos = cur_data_infos[:sampling_number]
            elif self.sampling_strategy.startswith("end"):
                cur_data_infos = cur_data_infos[-sampling_number:]
            elif self.sampling_strategy.startswith("index"):
                cur_data_infos = cur_data_infos[sampling_number : sampling_number + 1]

        self.data_infos = cur_data_infos


class APPLidarDepthDataset(APPDataset):

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
