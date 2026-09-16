import os
import sys

sys.path.append(os.getcwd())

import json
import logging
import random
import re
import traceback
from math import ceil
from typing import Dict, List

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch_geometric.nn import knn_interpolate

from hAlgorithm.datasets.transforms.edge_filter import edge_filter
from hAlgorithm.datasets.transforms.transforms import Compose, resize_depth_preserve
from hAlgorithm.utils import colorize_depth_maps, instantiate_from_config
from hAlgorithm.utils.sem_utils import SEM_LABEL, remap_sem_label
from hAlgorithm.modules.utils.ray import get_rays_in_camera_frame


class BaseDataset(Dataset):
    semantic_labels = None

    def __init__(
        self,
        phase: str,
        data_root: str,
        data_path: str,
        name: str = "base",
        seed: int = 0,
        sampling_strategy: str = "all",
        depth_root: str = None,
        mask_root: str = None,
        sem_root: str = None,
        norm_root: str = None,
        lidar_root: str = None,
        disparity_root: str = None,
        rbg_name: str = "rgb",
        depth_name: str = "depth",
        mask_name: str = "mask",
        sem_name: str = "sem_id",
        normal_name: str = "normal",
        lidar_name: str = "lidar",
        eval_mask_name: str = "eval_mask",
        disparity_name: str = "disparity",
        intrinsics_name: str = "cam_in",
        extrinsics_name: str = "extrinsics",
        prompt_extrinsics_name: str = None,
        invalid_mask_name: str = None,
        depth_invalid_sem_names: List = None,
        train_transforms: List[Dict] = [],
        test_transforms: List[Dict] = [],
        depth_scale: float = None,
        min_depth: float = None,
        max_depth: float = None,
        normalize_depth: Dict = None,
        with_pointmap: bool = False,
        only_pointmap: bool = False,
        sparse_depth_ratio: float = 0,
        sparse_depth_nums: int = 0,
        sparse_pattern: Dict = None,
        sparse_pattern_v2: Dict = None,
        sparse_pattern_v3: Dict = None,
        sparse_pattern_ratio: float = None,
        recalculate_normal: bool = False,
        normal_transform: Dict = None,
        size_pool: List[List] = None,
        with_rgb_edge_mask: bool = False,
        with_depth_edge_mask: bool = False,
        point_normalize_mode: str = "distance",
        debug: bool = False,
        interpolate_k: int = 0,
        interpolate_version: str = None,
        sparse_width: int = None,
        sparse_height: int = None,
        sparse_width_ratio: float = None,
        sparse_height_ratio: float = None,
        sparse_max_size: int = None,
        sparse_patch_size: int = None,
        quantile_filter: float = 0,
        sf_to_mf: bool = False,
        hdf5: bool = False,
        normal_hdf5: str = None,
        pre_crop_edge: List[int] = None,
        pre_crop_size: List[int] = None,
        with_curr_sift_mask: bool = False,
        with_inf_mask: bool = False,
        inf_mask_min: float = None,
        inf_mask_max: float = None,
        inf_mask_sem_names: List = None,
        inf_mask_depth_nan: bool = False,
        with_image_raw: bool = False,
        with_depth_raw: bool = True,
        with_pointmap_raw: bool = False,
        with_ray_dirs: bool = False,
        with_normal_raw: bool = False,
        ray_dirs_normal_to_unit_sphere: bool = False,
        max_scale_noise: float = None,
        custom_scale: float = None,
        extrinsics_noise_prob: float = None,
        extrinsics_noise_seed: int = None,
        extrinsics_noise_mean: float = 0,
        extrinsics_noise_std: float = 0, 
        extrinsics_rot_noise_std: float = 0,
        align_corners: bool = False,
        **kwargs,
    ):
        """
        Initializes a new instance of the BaseDataset class.

        Args:
            - phase (str): The phase of the dataset, typically 'train', 'val', or 'test'.
                            Determines which set of transformations to apply.
            - name (str): A string identifier for the dataset, useful for logging or multi-dataset setups.
            - data_root (str): The root directory where the dataset is stored. All other paths are relative to this.
            - data_path (str): The path to the actual data files, relative to `data_root`.
            - seed (int): The seed for random number generation. Defaults to 0.
            - sampling_strategy (str):
                The strategy used to sample data from the dataset. This parameter controls how many and which samples
                are included in the final dataset. The following strategies are supported:

                - `"all"`: Use all available samples in the dataset.
                - `"first:<N>"`: Use the first N samples from the dataset. For example, `"first:100"` will use the first 100 samples.
                - `"end:<N>"`: Use the last N samples from the dataset. For example, `"end:100"` will use the last 100 samples.
                - `"index:<N>"`: Use exactly one sample from the dataset. For example, `"index:1"` will use the sample with index 1.

                If no specific strategy is provided, the default behavior is to use all samples (`"all"`).
                Note: When using `"random:<N>"`, the random seed can be controlled via the `seed` parameter to ensure reproducibility.
            - depth_root (str): The root directory for depth images, if applicable. Defaults to None.
            - mask_root (str): The root directory for mask images, if applicable. Defaults to None.
            - sem_root (str): The root directory for semantic segmentation labels, if applicable. Defaults to None.
            - norm_root (str): The root directory for normal maps, if applicable. Defaults to None.
            - lidar_root (str): The root directory for LiDAR point cloud data, if applicable. Defaults to None.
            - train_transforms (List[Dict]): A list of transformation dictionaries to apply during the training phase. Defaults to an empty list.
            - test_transforms (List[Dict]): A list of transformation dictionaries to apply during the testing phase. Defaults to an empty list.
            - depth_scale (float): A scaling factor to apply to depth values, useful for converting between different units or scales. Defaults to None.
            - min_depth (float): The minimum valid depth value for filtering or normalization. Defaults to None.
            - max_depth (float): The maximum valid depth value for filtering or normalization. Defaults to None.
            - normalize_depth (Dict): A dictionary containing parameters for normalizing depth values, such as mean and standard deviation. Defaults to None.
            - with_pointmap (bool): Whether to include point cloud maps in the dataset. If True, point clouds will be generated from depth maps. Defaults to False.
            - sparse_depth_ratio (float): The ratio of points to keep when generating sparse depth maps. A value of 0.5 would retain 50% of the points. Defaults to 0.
            - depth_invalid_sem_names (List): List of semantic class names that indicate invalid depth regions. Defaults to None.
            - recalculate_normal (bool): Whether to recalculate normal maps from depth information. Useful if precomputed normals are not available or need updating. Defaults to False.
            - normal_transform (Dict): A dictionary containing parameters for transforming normal maps, such as resizing or normalization. Defaults to None.
            - size_pool (List[List], optional): List of image sizes to randomly select from. Default is None.
            - debug (bool): Whether to enable debug mode, which might include additional checks or verbose logging. Defaults to False.
            - pre_crop_edge (List[int, int]): Crop the RGB and depth after loading files, to avoid black edge of some datasets. Defaults to None.
            **kwargs: Additional keyword arguments that can be used by subclasses or for future expansion.
        """
        super().__init__()

        self.phase = phase
        self.name = name
        self.debug = debug
        self.seed = seed

        self.data_root = data_root
        self.data_path = data_path
        self.sampling_strategy = sampling_strategy
        self.quantile_filter = quantile_filter

        # Set default roots if not provided
        self.depth_root = depth_root if depth_root is not None else self.data_root
        self.mask_root = mask_root if mask_root is not None else self.depth_root
        self.sem_root = sem_root if sem_root is not None else self.data_root
        self.norm_root = norm_root if norm_root is not None else self.data_root
        self.lidar_root = lidar_root if lidar_root is not None else self.data_root
        self.disparity_root = disparity_root if disparity_root is not None else self.data_root

        self.rbg_name = rbg_name if rbg_name not in ["None", "none"] else None
        self.depth_name = depth_name if depth_name not in ["None", "none"] else None
        self.mask_name = mask_name if mask_name not in ["None", "none"] else None
        self.sem_name = sem_name if sem_name not in ["None", "none"] else None
        self.normal_name = normal_name if normal_name not in ["None", "none"] else None
        self.lidar_name = lidar_name if lidar_name not in ["None", "none"] else None
        self.eval_mask_name = eval_mask_name if eval_mask_name not in ["None", "none"] else None
        self.disparity_name = disparity_name if disparity_name not in ["None", "none"] else None
        self.intrinsics_name = intrinsics_name if intrinsics_name not in ["None", "none"] else None
        self.extrinsics_name = extrinsics_name if extrinsics_name not in ["None", "none"] else None
        self.prompt_extrinsics_name = prompt_extrinsics_name if prompt_extrinsics_name not in ["None", "none"] else None
        self.normal_hdf5 = normal_hdf5 if normal_hdf5 not in ["None", "none"] else None
        self.invalid_mask_name = invalid_mask_name if invalid_mask_name not in ["None", "none"] else None
        self.hdf5 = hdf5

        # Initialize transformations
        self.size_pool = size_pool

        if self.phase == "train":
            self.data_transforms = Compose(train_transforms)
        else:
            if test_transforms is not None and len(test_transforms) > 0:
                self.data_transforms = Compose(test_transforms)
            else:
                self.data_transforms = Compose(train_transforms)

        # Depth-related parameters
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth
        assert (
            self.min_depth is None or self.min_depth >= 1e-6
        ), "min_depth must be greater than or equal to 1e-6"
        self.normalize_depth = instantiate_from_config(normalize_depth) if normalize_depth else None
        self.with_pointmap = with_pointmap
        self.only_pointmap = only_pointmap
        self.point_normalize_mode = point_normalize_mode
        self.max_scale_noise = max_scale_noise
        self.sparse_depth_ratio = sparse_depth_ratio
        self.sparse_depth_nums = sparse_depth_nums
        self.sparse_pattern_v3 = instantiate_from_config(sparse_pattern_v3)
        self.sparse_pattern_ratio = sparse_pattern_ratio
        self.sparse_width = sparse_width
        self.sparse_height = sparse_height
        self.sparse_width_ratio = sparse_width_ratio
        self.sparse_height_ratio = sparse_height_ratio
        self.sparse_max_size = sparse_max_size
        self.sparse_patch_size = sparse_patch_size
        self.with_rgb_edge_mask = with_rgb_edge_mask
        self.with_depth_edge_mask = with_depth_edge_mask
        self.with_curr_sift_mask = with_curr_sift_mask
        if self.with_curr_sift_mask:
            self.sift_detector = cv2.SIFT.create()
        self.sf_to_mf = sf_to_mf
        self.with_image_raw = with_image_raw
        self.with_depth_raw = with_depth_raw
        self.with_pointmap_raw = with_pointmap_raw
        self.with_normal_raw = with_normal_raw
        # Segmentation-related parameters
        self.depth_invalid_sem_names = depth_invalid_sem_names

        # inf mask
        self.with_inf_mask = with_inf_mask
        self.inf_mask_min = inf_mask_min
        self.inf_mask_max = inf_mask_max
        self.inf_mask_sem_names = inf_mask_sem_names
        self.inf_mask_depth_nan = inf_mask_depth_nan

        # Noraml-related parameters
        self.recalculate_normal = recalculate_normal
        self.normal_transform = (
            instantiate_from_config(normal_transform) if normal_transform else None
        )
        assert (not self.recalculate_normal) or (
            self.recalculate_normal and self.normal_transform is not None
        )
        # Sparse depth interpolate parames
        self.interpolate_k = interpolate_k
        self.interpolate_version = interpolate_version

        self.pre_crop_edge = pre_crop_edge
        self.pre_crop_size = pre_crop_size

        # ray dirs
        self.with_ray_dirs = with_ray_dirs
        self.ray_dirs_normal_to_unit_sphere = ray_dirs_normal_to_unit_sphere

        self.custom_scale = custom_scale

        # prompt extrinsics noise
        self.extrinsics_noise_prob = extrinsics_noise_prob
        self.extrinsics_noise_seed = extrinsics_noise_seed
        self.extrinsics_noise_mean = extrinsics_noise_mean
        self.extrinsics_noise_std = extrinsics_noise_std
        self.extrinsics_rot_noise_std = extrinsics_rot_noise_std

        self.align_corners = align_corners

        # Loads and processes the dataset information from a JSON file
        self.get_data_infos()

    def get_data_infos(self):
        """
        Loads and processes the dataset information from a JSON file.

        This method reads the dataset configuration from the specified `data_path` and applies the sampling strategy
        to filter the data. The resulting data information is stored in `self.data_infos`.

        Steps:
        1. Load the JSON file containing the dataset information.
        2. Apply the specified sampling strategy (`sampling_strategy`) to select a subset of the data.
        3. Store the processed data information in `self.data_infos`.
        """
        # Load the dataset configuration from the JSON file
        cur_data_infos = self.load_json_file(self.data_path)

        if isinstance(self.sampling_strategy, str):
            # Parse the sampling strategy to determine how many samples to use
            sampling_number = self._parse_sampling_strategy(
                self.sampling_strategy, len(cur_data_infos)
            )

            # Apply the sampling strategy if a specific number of samples is requested
            if sampling_number is not None:
                if self.sampling_strategy.startswith("first"):
                    cur_data_infos = cur_data_infos[:sampling_number]
                elif self.sampling_strategy.startswith("end"):
                    cur_data_infos = cur_data_infos[-sampling_number:]
                elif self.sampling_strategy.startswith("index"):
                    cur_data_infos = cur_data_infos[sampling_number : sampling_number + 1]
        elif isinstance(self.sampling_strategy, (list, tuple)):
            cur_data_infos = [cur_data_infos[int(i)] for i in self.sampling_strategy]
        else:
            raise NotImplementedError(
                f"{type(self.sampling_strategy)} sampling_strategy {self.sampling_strategy} is not implemented"
            )

        self.data_infos = cur_data_infos

    @staticmethod
    def _parse_sampling_strategy(strategy: str, total_samples: int):
        """Parse the sampling strategy string and return the number of samples to select."""
        if ":" in strategy:
            strategy, num_str = strategy.split(":")
            if "%" in num_str:
                return ceil(int(num_str.strip("%")) * total_samples / 100)
            else:
                return int(num_str)
        return None

    def load_json_file(self, json_path):
        """Load data from a JSON or JSONL file."""
        if isinstance(json_path, (list, tuple)):
            total_data = []
            for json_path_i in json_path:
                total_data.extend(self.load_json_file(json_path_i))
            return total_data
        elif json_path.endswith(".jsonl"):
            with open(json_path, "r") as json_file:
                return [
                    (
                        json.loads(line.strip())["files"]
                        if "files" in json.loads(line.strip())
                        else json.loads(line.strip())
                    )
                    for line in json_file
                ]
        elif json_path.endswith(".json"):
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
                if "files" in data:
                    data = data["files"]
                return data
        elif json_path.endswith(".parquet"):
            data = pd.read_parquet(json_path)
            if "files" in data:
                data = data["files"]
            return data
        else:
            raise ValueError(f"Unsupported file type: {json_path}")

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx) -> dict:
        return self.getitem_single_frame(idx)

    def info_rbg_path(self, data_info):
        if self.rbg_name in data_info:
            logging.info(
                f"{self.rbg_name} {os.path.join(self.data_root, data_info[self.rbg_name])}"
            )
        elif self.hdf5:
            logging.info(f"hdf5 {os.path.join(self.data_root, data_info['hdf5'])}")

    def getitem_single_frame(self, idx):
        if self.phase == "test":
            return self.get_data_for_test(idx)
        else:
            try:
                return self.get_data_for_trainval(idx)
            except Exception as e:
                traceback.print_exc()
                logging.error(
                    f"Dataset:{self.name} getitem error, index update {idx}, got exception {e}"
                )
                if isinstance(idx, (list, tuple)):
                    logging.error(self.data_infos[idx[0]])
                    if self.phase == "train":
                        idx[0] = random.randint(0, len(self.data_infos) - 1)
                    else:
                        idx[0] = (idx[0] + 1) % len(self.data_infos)
                else:
                    logging.error(self.data_infos[idx])
                    if self.phase == "train":
                        idx = random.randint(0, len(self.data_infos) - 1)
                    else:
                        idx = (idx + 1) % len(self.data_infos)

                return self.getitem_single_frame(idx)

    def get_data_for_trainval(
        self, idx, data_info=None, data_batch=None, transform_info=None, mf_debug=False, trajectory_noise=None,
    ):
        if data_info is None or data_batch is None:
            if isinstance(idx, (list, tuple)):
                idx, others = idx
                self.update_data_transforms(**others)

            # Load the data information and batch for the given index
            data_info = self.data_infos[idx]
            data_batch = self.load_data(data_info)
            mv_flag = False
        else:
            mv_flag = True

        if self.debug:
            self.info_rbg_path(data_info)

        (
            curr_rgb,
            curr_intrinsics,
            curr_extrinsics,
            curr_prompt_extrinsics,
            curr_depth,
            curr_depth_mask,
            curr_sem,
            curr_normal,
            curr_prompt,
        ) = (
            data_batch["curr_rgb"],
            data_batch["curr_intrinsics"],
            data_batch["curr_extrinsics"],
            data_batch["curr_prompt_extrinsics"],
            data_batch["curr_depth"],
            data_batch["curr_depth_mask"],
            data_batch["curr_sem"],
            data_batch["curr_normal"],
            data_batch["curr_prompt"],
        )
        curr_depth, curr_depth_mask = self.depth_filter(
            curr_depth, depth_mask=curr_depth_mask, sem_label=curr_sem
        )
        if curr_depth_mask is not None and curr_depth_mask.sum() == 0:
            self.info_rbg_path(data_info)
            logging.warning("depth_mask is empty, no valid points found.")
            raise ValueError("depth_mask is empty, no valid points found.")

        curr_eval_mask = data_batch.get("curr_eval_mask", None)
        curr_disp = data_batch.get("curr_disp", None)
        curr_inf_mask = data_batch.get("curr_inf_mask", None)
        curr_invalid_mask = data_batch.get("curr_invalid_mask", None)

        transform_info = transform_info if transform_info is not None else dict()
        transform_info["name"] = self.name
        transform_info["other_labels"] = [
            "sem", "eval_mask", "disp", "inf_mask", "invalid_mask"
        ]

        # Apply data augmentation transforms
        (
            image,
            intrinsics,
            depth,
            depth_mask,
            normal,
            other_labels,
            transform_info,
        ) = self.data_transforms(
            image=curr_rgb,
            intrinsics=curr_intrinsics.copy() if curr_intrinsics is not None else None,
            depth=curr_depth,
            depth_mask=curr_depth_mask,
            normal=curr_normal.astype(np.float32) if curr_normal is not None else None,
            other_labels=[curr_sem, curr_eval_mask, curr_disp, curr_inf_mask, curr_invalid_mask],
            transform_info=transform_info,
        )

        # Process the semantic label and sky mask
        sem_label = other_labels[0]
        eval_mask = other_labels[1]
        disp = other_labels[2]
        inf_mask = other_labels[3]
        invalid_mask = other_labels[4]

        disp, disp_valid = self.process_disp(disp, image, curr_rgb)

        normal_mask = None
        if normal is not None:
            normal_mask = torch.logical_or(
                torch.isfinite(normal).any(dim=0), (normal == 0).all(dim=0)
            )
            normal = torch.nan_to_num(normal, 0, 0, 0)
            normal[:, ~normal_mask] = 0.0
            normal[:, normal_mask] = torch.nn.functional.normalize(normal[:, normal_mask], dim=0)

        if eval_mask is not None:
            eval_mask = eval_mask.bool()

        if inf_mask is not None:
            inf_mask = inf_mask.bool()

        if invalid_mask is not None:
            invalid_mask = invalid_mask.bool()

        if self.with_curr_sift_mask:
            if self.phase != "train":
                curr_sift_mask, curr_sift_eval_mask = self.load_sift_mask(curr_rgb, curr_depth)
            else:
                curr_sift_mask = curr_sift_eval_mask = None
            sift_mask_, sift_eval_mask_ = self.load_sift_mask(
                image.permute(1, 2, 0).numpy(), depth.squeeze().numpy()
            )
            sift_mask = torch.from_numpy(sift_mask_).unsqueeze(0).bool()
            sift_eval_mask = torch.from_numpy(sift_eval_mask_).unsqueeze(0).bool()
        else:
            curr_sift_mask = curr_sift_eval_mask = None
            sift_mask = sift_eval_mask = None

        # A boolean mask indicating the invalid depth regions
        depth_invalid_sem_mask = self.process_sem_mask(sem_label)

        # Create the intrinsics and extrinsics matrix from the parameters
        intrinsics_mat = (
            self.create_intrinsics_matrix(intrinsics) if intrinsics is not None else None
        )
        extrinsics_mat = (
            self.create_extrinsics_matrix(curr_extrinsics) if curr_extrinsics is not None else None
        )
        prompt_extrinsics_mat = self.create_extrinsics_matrix(curr_prompt_extrinsics) if curr_prompt_extrinsics is not None else None

        # Extracts sparse depth points and their corresponding masks from the origin depthmap
        if self.sparse_pattern_ratio is None or random.random() <= self.sparse_pattern_ratio:
            if curr_prompt is not None:
                if curr_prompt.shape[0:2] != curr_rgb.shape[0:2]:
                    curr_prompt = resize_depth_preserve(curr_prompt, curr_rgb.shape[0:2])
                curr_prompt_mask = (curr_prompt > self.min_depth) & (curr_prompt < self.max_depth)
                if (
                    curr_depth_mask is not None
                    and curr_prompt_mask.shape[0:2] == curr_depth_mask.shape[0:2]
                ):
                    curr_prompt_mask = curr_prompt_mask & curr_depth_mask
                curr_prompt_mask = curr_prompt_mask.astype(curr_depth_mask.dtype)
                sparse_pointmap, sparse_pointmap_mask = self.get_sparse_depth_with_pattern(
                    curr_rgb=curr_rgb,
                    curr_depth=curr_prompt,
                    curr_depth_mask=curr_prompt_mask,
                    curr_intrinsics=curr_intrinsics,
                    rgb=image,
                    depth=depth,
                    depth_mask=depth_mask,
                    intrinsics_mat=intrinsics_mat,
                    transform_info=transform_info,
                    data_idx=idx,
                )
            elif curr_depth is not None:
                sparse_pointmap, sparse_pointmap_mask = self.get_sparse_depth_with_pattern(
                    curr_rgb=curr_rgb,
                    curr_depth=curr_depth,
                    curr_depth_mask=curr_depth_mask,
                    curr_intrinsics=curr_intrinsics,
                    rgb=image,
                    depth=depth,
                    depth_mask=depth_mask,
                    intrinsics_mat=intrinsics_mat,
                    transform_info=transform_info,
                    data_idx=idx,
                )
            else:
                sparse_pointmap = sparse_pointmap_mask = None

            # Process the depth data
            depth_output = self.process_depth(
                depth,
                depth_mask,
                intrinsics_mat,
                sparse_pointmap=sparse_pointmap,
                sparse_pointmap_mask=sparse_pointmap_mask,
            )
        else:
            # Process the depth data
            depth_output = self.process_depth(
                depth,
                depth_mask,
                intrinsics_mat,
                sparse_pointmap=None,
                sparse_pointmap_mask=None,
            )
            depth_output["sparse_pointmap"] = depth.new_zeros([3, *depth.shape[-2:]])
            depth_output["sparse_pointmap_mask"] = torch.zeros_like(depth).bool()

        if depth_output["sparse_pointmap_max_range"] is None and self.custom_scale is not None:
            depth_output["sparse_pointmap_max_range"] = torch.tensor(self.custom_scale).float()

        # Construct the final data dictionary
        data_dict = {
            "image": image,
            "intrinsics": intrinsics_mat,
            "extrinsics": extrinsics_mat,
            "prompt_extrinsics": prompt_extrinsics_mat,
            "sem_label": sem_label,
            "eval_mask": eval_mask,
            "sift_mask": sift_mask,
            "sift_eval_mask": sift_eval_mask,
            "depth_invalid_sem_mask": depth_invalid_sem_mask,
            "normal": normal,
            "normal_mask": normal_mask,
            "disp": disp,
            "disp_valid": disp_valid,
            "inf_mask": inf_mask,
            "invalid_mask": invalid_mask,
            **depth_output,
        }

        if self.with_ray_dirs:
            image_h, image_w = image.shape[-2:]
            cam_ray_dirs = get_rays_in_camera_frame(
                intrinsics_mat, image_h, image_w, self.ray_dirs_normal_to_unit_sphere
            )
            data_dict["cam_ray_dirs"] = cam_ray_dirs

        # Apply flips to depth/mask/intrinsics for train mode (shared by depth_raw and pointmap_raw)
        if curr_depth is not None and (self.with_depth_raw or self.with_pointmap_raw):
            h, w = curr_depth.shape[-2:]
            if transform_info.get("horizontal_flip", False):
                curr_depth = np.fliplr(curr_depth).copy()
                curr_depth_mask = np.fliplr(curr_depth_mask).copy()
                curr_intrinsics = curr_intrinsics.copy()
                curr_intrinsics[2] = w - 1 - curr_intrinsics[2]
            if transform_info.get("vertical_flip", False):
                curr_depth = np.flipud(curr_depth).copy()
                curr_depth_mask = np.flipud(curr_depth_mask).copy()
                curr_intrinsics = curr_intrinsics.copy()
                curr_intrinsics[3] = h - 1 - curr_intrinsics[3]

        # Add raw depth and mask if available (now supports train with flip augmentation)
        if self.with_depth_raw and curr_depth is not None:
            data_dict["depth_raw"] = torch.from_numpy(np.ascontiguousarray(curr_depth)).float()
            data_dict["depth_raw_mask"] = torch.from_numpy(np.ascontiguousarray(curr_depth_mask)).bool()
            data_dict["intrinsics_raw"] = self.create_intrinsics_matrix(curr_intrinsics)

        if self.with_normal_raw and curr_normal is not None:
            if self.phase == "train" and transform_info.get("horizontal_flip", False):
                curr_normal = cv2.flip(curr_normal, 1)
                curr_normal[:, :, 0] = -curr_normal[:, :, 0]
            if self.phase == "train" and transform_info.get("vertical_flip", False):
                curr_normal = cv2.flip(curr_normal, 0)
                curr_normal[:, :, 1] = -curr_normal[:, :, 1]

            normal_raw = torch.from_numpy(curr_normal).float().permute(2, 0, 1)  # H,W,3 -> 3,H,W
            normal_raw_mask = torch.logical_or(
                torch.isfinite(normal_raw).any(dim=0), (normal_raw == 0).all(dim=0)
            )
            normal_raw = torch.nan_to_num(normal_raw, 0, 0, 0)
            normal_raw[:, ~normal_raw_mask] = 0.0
            normal_raw[:, normal_raw_mask] = torch.nn.functional.normalize(
                normal_raw[:, normal_raw_mask], dim=0
            )
            data_dict["normal_raw"] = normal_raw
            data_dict["normal_raw_mask"] = normal_raw_mask

        if self.with_pointmap_raw and curr_depth is not None:
            curr_intrinsics_mat = self.create_intrinsics_matrix(curr_intrinsics)
            curr_pointmap = self.load_pointmap(curr_depth, curr_intrinsics_mat)
            data_dict["pointmap_raw"] = curr_pointmap.float()
            if "depth_raw_mask" not in data_dict:
                data_dict["depth_raw_mask"] = torch.from_numpy(np.ascontiguousarray(curr_depth_mask)).bool()
        
        # Add raw depth and mask if available
        if curr_eval_mask is not None and self.phase != "train":
            data_dict["eval_raw_mask"] = torch.from_numpy(curr_eval_mask).bool()

        if curr_sift_eval_mask is not None and self.phase != "train":
            data_dict["sift_eval_raw_mask"] = torch.from_numpy(curr_sift_eval_mask).bool()
            data_dict["sift_raw_mask"] = torch.from_numpy(curr_sift_mask).bool()

        # Add raw depth and mask if available
        if curr_disp is not None and self.phase != "train":
            data_dict["disp_raw"] = torch.from_numpy(curr_disp)

        if curr_inf_mask is not None and self.phase != "train":
            data_dict["inf_raw_mask"] = torch.from_numpy(curr_inf_mask).bool()

        if "image_show" in transform_info and self.phase != "train":
            image_show = transform_info["image_show"]
            if not isinstance(image_show, torch.Tensor):
                image_show = torch.from_numpy(transform_info["image_show"]).float()
            data_dict["image_show"] = image_show

        if self.with_image_raw:
            data_dict["image_raw"] = torch.from_numpy(curr_rgb).float()

            if self.phase == "train" and transform_info.get("horizontal_flip", False):
                data_dict["image_raw"] = torch.flip(data_dict["image_raw"], dims=[1])

            if self.phase == "train" and transform_info.get("vertical_flip", False):
                data_dict["image_raw"] = torch.flip(data_dict["image_raw"], dims=[0])
        
        if "image_backup" in transform_info:
            data_dict["image_backup"] = transform_info["image_backup"]

            if self.with_rgb_edge_mask:
                tmp_data = data_dict["image_backup"].mean(0, keepdim=True)
                edge_mask = edge_filter(tmp_data, valid_mask=data_dict["depth_mask"], times=0.3)
                
                if data_dict.get("edge_mask", None) is not None:
                    data_dict["edge_mask"] = edge_mask | data_dict["edge_mask"]
                else:
                    data_dict["edge_mask"] = edge_mask

        if trajectory_noise is None or trajectory_noise is True:
            if self.extrinsics_noise_prob is not None and self.extrinsics_noise_prob > 0:
                if self.extrinsics_noise_seed is not None:
                    random.seed(self.extrinsics_noise_seed)
                    np.random.seed(self.extrinsics_noise_seed)
                    torch.manual_seed(self.extrinsics_noise_seed)
                if (self.extrinsics_noise_prob >= 1.0) or (random.random() <= self.extrinsics_noise_prob):
                    from hAlgorithm.datasets.patterns.pattern_transform import RandomProjectNoise
                    extrinsics_noise = RandomProjectNoise.generate_random_transform(
                        None,
                        noise_mean=self.extrinsics_noise_mean, 
                        noise_std=self.extrinsics_noise_std, 
                        rot_noise_std=self.extrinsics_rot_noise_std
                    )
                    data_dict["extrinsics_noise"] = extrinsics_noise
                else:
                    data_dict["extrinsics_noise"] = torch.eye(4).float()

        # Add metadata
        data_dict["meta_data"] = {
            "name": self.name,
            "data_path": self.data_path,
            "data_root": self.data_root,
            "data_idx": idx,
            "data_info": {k: v for k, v in data_info.items() if k not in ["sem_sky", "normal"]},
            "depth_scale": self.depth_scale,
            "input_width": image.shape[2],
            "input_height": image.shape[1],
            "origin_width": curr_rgb.shape[1],
            "origin_height": curr_rgb.shape[0],
        }

        # NOTE: save output with kosmo format, only do it with sf
        if (not mv_flag) and getattr(self, "kosmo_scene_infos", None) is not None and len(self.kosmo_scene_infos) > 0:
            scene = data_dict["meta_data"]['data_info']['scene']
            data_dict["meta_data"]["kosmo_scene_infos"] = self.kosmo_scene_infos[scene]

            frame_id = int(data_dict['meta_data']['data_info']['frame_id'])
            view_id = int(data_dict['meta_data']['data_info']['view_id'])
            kosmo_frames = dict()
            if frame_id not in kosmo_frames:
                kosmo_frames[frame_id] = dict()
            kosmo_frames[frame_id][view_id] = self.kosmo_frames[frame_id][view_id]
            data_dict["meta_data"]["kosmo_frames"] = kosmo_frames

        if "depth_scale" not in data_dict["meta_data"]["data_info"]:
            data_dict["meta_data"]["data_info"]["depth_scale"] = self.depth_scale

        if "rgb" not in data_dict["meta_data"]["data_info"]:
            if self.rbg_name in data_dict["meta_data"]["data_info"]:
                data_dict["meta_data"]["data_info"]["rgb"] = data_dict["meta_data"]["data_info"][
                    self.rbg_name
                ]
            else:
                data_dict["meta_data"]["data_info"]["rgb"] = data_dict["meta_data"]["data_info"][
                    "hdf5"
                ]

        # Remove entries with None values to clean up the dictionary
        data_dict = {k: v for k, v in data_dict.items() if v is not None}

        # Optionally debug the data dictionary
        if self.debug and not mf_debug:
            self.debug_data_dict(idx, data_dict)

        if self.sf_to_mf:
            from torch.utils.data._utils.collate import default_collate

            for key in data_dict.keys():
                if key != "meta_data":
                    data_dict[key] = default_collate([data_dict[key]])
                else:
                    data_dict["meta_data"]["views"] = 1
                    data_dict["meta_data"]["frames"] = 1

            extrinsics_mat = torch.eye(4).float()
            data_dict["extrinsics"] = default_collate([extrinsics_mat])
            data_dict["extrinsics_reff"] = default_collate([extrinsics_mat])

            if "pointmap" in data_dict:
                data_dict["pointmap_reff"] = data_dict["pointmap"]

        return data_dict

    def get_data_for_test(self, idx: int):
        return self.get_data_for_trainval(idx)

    def update_data_transforms(self, **kwargs):
        # """
        # Updates the parameters of data transformations (e.g., RandomCrop, Resize) based on the newly selected image size.
        # """
        # if self.phase == "train" and self.size_pool is not None:
        #     for transform in self.data_transforms.transforms:
        #         if transform.__class__.__name__ == "RandomCrop":
        #             transform.crop_h = new_size[0]
        #             transform.crop_w = new_size[1]
        #         if transform.__class__.__name__ == "Resize":
        #             transform.height = new_size[0]
        #             transform.width = new_size[1]
        # TODO: delete update_data_transforms
        
        max_size = kwargs.get("max_size")
        if self.phase == "train" and max_size is not None:
            update_flag = False
            for transform in self.data_transforms.transforms:
                if hasattr(transform, "update_max_size"):
                    transform.update_max_size(max_size=max_size)
                    update_flag = True
            assert update_flag
        return None

    def load_data_path(self, data_info):
        curr_rgb_path = os.path.join(self.data_root, data_info[self.rbg_name])
        curr_depth_path = (
            os.path.join(self.depth_root, data_info[self.depth_name])
            if data_info.get(self.depth_name, None) is not None
            else None
        )
        curr_depth_mask_path = (
            os.path.join(self.mask_root, data_info[self.mask_name])
            if data_info.get(self.mask_name, None) is not None
            else None
        )
        curr_eval_mask_path = (
            os.path.join(self.mask_root, data_info[self.eval_mask_name])
            if data_info.get(self.eval_mask_name, None) is not None
            else None
        )
        curr_sem_path = (
            os.path.join(self.sem_root, data_info[self.sem_name])
            if data_info.get(self.sem_name, None) is not None
            else None
        )
        curr_norm_path = (
            os.path.join(self.norm_root, data_info[self.normal_name])
            if data_info.get(self.normal_name, None) is not None
            else None
        )
        curr_lidar_path = (
            os.path.join(self.lidar_root, data_info[self.lidar_name])
            if data_info.get(self.lidar_name, None) is not None
            else None
        )
        curr_disp_path = (
            os.path.join(self.disparity_root, data_info[self.disparity_name])
            if data_info.get(self.disparity_name, None) is not None
            else None
        )
        curr_invalid_mask_path = (
            os.path.join(self.mask_root, data_info[self.invalid_mask_name])
            if data_info.get(self.invalid_mask_name, None) is not None
            else None
        )
        data_path = dict(
            rgb_path=curr_rgb_path,
            depth_path=curr_depth_path,
            depth_mask_path=curr_depth_mask_path,
            sem_path=curr_sem_path,
            normal_path=curr_norm_path,
            lidar_path=curr_lidar_path,
            eval_mask_path=curr_eval_mask_path,
            disp_path=curr_disp_path,
            invalid_mask_path=curr_invalid_mask_path,
        )
        return data_path

    def load_data(self, data_info):
        curr_rgb = curr_depth = curr_depth_mask = None
        curr_sem = curr_normal = curr_disp = None
        curr_prompt = curr_eval_mask = curr_inf_mask = None
        curr_invalid_mask = None

        # 获取相机内参矩阵（Intrinsics）
        curr_intrinsics = data_info.get(self.intrinsics_name, None)
        # 获取相机外参矩阵（Extrinsics）
        curr_extrinsics = data_info.get(self.extrinsics_name, None)
        curr_prompt_extrinsics = data_info.get(self.prompt_extrinsics_name, None)

        # 根据传入的data_info加载数据路径
        if self.hdf5 or "hdf5" in data_info:
            hdf5_path = os.path.join(self.data_root, data_info["hdf5"])
            hdf5_id = data_info.get("hdf5_id", None)

            def get_h5_data(f, name):
                return f[name][hdf5_id] if hdf5_id is not None else f[name][:]

            with h5py.File(hdf5_path, "r") as f:
                if self.rbg_name is not None and self.rbg_name in f:
                    curr_rgb = get_h5_data(f, self.rbg_name)
                if self.depth_name is not None and self.depth_name in f:
                    curr_depth = get_h5_data(f, self.depth_name)
                    if curr_depth.dtype != np.float32:
                        curr_depth = curr_depth.astype(np.float32)
                    if len(curr_depth.shape) == 3:
                        curr_depth = curr_depth[..., -1]
                if self.mask_name is not None and self.mask_name in f:
                    curr_depth_mask = get_h5_data(f, self.mask_name)
                    if len(curr_depth_mask.shape) == 3:
                        curr_depth_mask = curr_depth_mask[..., -1]
                if self.eval_mask_name is not None and self.eval_mask_name in f:
                    curr_eval_mask = get_h5_data(f, self.eval_mask_name)
                    if len(curr_eval_mask.shape) == 3:
                        curr_eval_mask = curr_eval_mask[..., -1]
                if self.sem_name is not None and self.sem_name in f:
                    curr_sem = get_h5_data(f, self.sem_name)
                    curr_sem = self.remap_sem_label(curr_sem, data_info)
                if self.normal_name is not None and self.normal_name in f:
                    curr_normal = get_h5_data(f, self.normal_name)
                if self.lidar_name is not None and self.lidar_name in f:
                    curr_prompt = get_h5_data(f, self.lidar_name)
                    if curr_prompt.dtype != np.float32:
                        curr_prompt = curr_prompt.astype(np.float32)
                    if len(curr_prompt.shape) == 3:
                        curr_prompt = curr_prompt[..., -1]
                    if curr_depth is None:
                        curr_depth = curr_prompt.copy()
                if self.disparity_name is not None and self.disparity_name in f:
                    curr_disp = get_h5_data(f, self.disparity_name)

                if "intrinsics" in f:
                    curr_intrinsics = get_h5_data(f, "intrinsics").tolist()

                if "extrinsics" in f:
                    curr_extrinsics = get_h5_data(f, "extrinsics")

                if "depth_scale" in f:
                    self.depth_scale = f["depth_scale"][()]
                else:
                    self.depth_scale = 1.0
            
            if self.normal_hdf5 is not None and self.normal_hdf5 in data_info:
                normal_hdf5_path = os.path.join(self.data_root, data_info[self.normal_hdf5])
                with h5py.File(normal_hdf5_path, "r") as f_normal:
                    if self.normal_name in f_normal:
                        curr_normal = get_h5_data(f_normal, self.normal_name)

            if curr_normal is not None and curr_normal.shape[-1] != 3:
                curr_normal = curr_normal.transpose(1, 2, 0) # 3,H,W -> H,W,3

            if curr_depth is not None:
                curr_depth = curr_depth / self.depth_scale
            if self.with_inf_mask:
                curr_inf_mask = self.load_inf_mask(
                    curr_depth, curr_sem, curr_rgb, data_info
                ).astype(np.float_)
            if curr_depth is not None:
                if curr_depth_mask is None:
                    curr_depth_mask = (~np.isnan(curr_depth)).astype(int)
                else:
                    curr_depth_mask = curr_depth_mask * (~np.isnan(curr_depth)).astype(int)
                curr_depth[~curr_depth_mask.astype(bool)] = 0

            if curr_prompt is not None:
                curr_prompt = curr_prompt / self.depth_scale

        else:
            data_path = self.load_data_path(data_info)

            # 加载RGB图像，形状为[h, w, 3]，即高度、宽度和RGB通道
            if data_path["rgb_path"] is not None:
                curr_rgb = self.read_image(data_path["rgb_path"]).copy()

            # 加载深度图（Depth Map），如果路径存在则加载，否则为None
            if data_path["depth_path"] is not None:
                curr_depth = self.load_depth(
                    data_path["depth_path"], image=None, const=None, dtype=np.float32
                )
                if len(curr_depth.shape) == 3:
                    curr_depth = curr_depth[..., -1]
                if len(curr_depth.shape) == 1:
                    curr_depth = curr_depth.reshape(*curr_rgb.shape[:2])

            # 加载深度提示（Depth Prompt），通常来自激光雷达（LiDAR）数据
            if data_path["lidar_path"] is not None:
                curr_prompt = self.load_depth(
                    data_path["lidar_path"], image=None, const=None, dtype=np.float32
                )
                if len(curr_prompt.shape) == 3:
                    curr_prompt = curr_prompt[..., -1]
                # 如果深度图为空但深度提示存在，则将深度提示复制为深度图
                if curr_depth is None:
                    curr_depth = curr_prompt.copy()

            if curr_intrinsics is not None and isinstance(curr_intrinsics, str):
                if curr_intrinsics.endswith(".npy"):
                    curr_intrinsics = np.load(curr_intrinsics).reshape(3, 3)
                else:
                    with open(curr_intrinsics, "r") as f:
                        curr_intrinsics = np.array(json.load(f)).reshape(3, 3)
                curr_intrinsics = curr_intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]].tolist()

            if curr_extrinsics is not None and isinstance(curr_extrinsics, str):
                if curr_extrinsics.endswith(".npy"):
                    curr_extrinsics = np.load(curr_extrinsics).reshape(4, 4)
                else:
                    with open(curr_extrinsics, "r") as f:
                        curr_extrinsics = np.array(json.load(f)).reshape(4, 4)

            # 根据数据信息中的深度缩放因子（depth_scale）对深度图进行缩放
            if "depth_scale" in data_info and self.depth_scale != data_info["depth_scale"]:
                self.depth_scale = data_info["depth_scale"]
            elif "depth_scale" not in data_info and self.depth_scale is None:
                self.depth_scale = 1

            # 如果深度提示存在，同样对其进行缩放
            if curr_depth is not None:
                curr_depth = curr_depth / self.depth_scale
            if curr_prompt is not None:
                curr_prompt = curr_prompt / self.depth_scale

            # 加载语义标签（Semantic Labels），默认值为-1
            if data_path["sem_path"] is not None:
                curr_sem = self.read_file(
                    data_path["sem_path"], image=curr_rgb[..., 0], const=-1, dtype=int
                )
                curr_sem = self.remap_sem_label(curr_sem, data_info)
            # 加载inf mask
            if self.with_inf_mask:
                curr_inf_mask = self.load_inf_mask(
                    curr_depth, curr_sem, curr_rgb, data_info
                ).astype(np.float_)

            # 加载深度掩码（Depth Mask），用于标记深度图中的有效区域
            if curr_depth is not None:
                curr_depth_mask = self.read_file(
                    data_path["depth_mask_path"], image=curr_depth, const=1, dtype=int
                )
                if len(curr_depth_mask.shape) == 3:
                    curr_depth_mask = curr_depth_mask[..., -1]
                # 如果深度图存在，更新深度掩码以移除无效深度值（NaN值），并设置无效区域的深度值为0
                curr_depth_mask = (
                    curr_depth_mask
                    * (~np.isnan(curr_depth)).astype(int)
                    * (~np.isinf(curr_depth)).astype(int)
                )
                curr_depth[~curr_depth_mask.astype(bool)] = 0

            # 加载法向量图（Normal Map）
            if self.recalculate_normal:
                assert curr_depth is not None and curr_intrinsics is not None
                curr_normal = self.normal_transform(curr_intrinsics, curr_depth)
            elif data_path["normal_path"] is not None:
                curr_normal = self.load_normal(
                    data_path["normal_path"], image=curr_rgb, const=0, dtype=np.float32
                )
                if curr_normal.shape[-1] != 3:
                    curr_normal = curr_normal.transpose(1, 2, 0) # 3,H,W -> H,W,3

            # 加载评估掩码（Evaluation Mask），用于标记需要评估的区域
            if data_path["eval_mask_path"] is not None:
                curr_eval_mask = self.read_file(
                    data_path["eval_mask_path"], image=curr_depth, const=1, dtype=int
                )
                if len(curr_eval_mask.shape) == 3:
                    curr_eval_mask = curr_eval_mask[..., -1]

            # load invalid mask
            if data_path["invalid_mask_path"] is not None:
                curr_invalid_mask = self.read_file(
                    data_path["invalid_mask_path"], image=curr_depth, const=1, dtype=bool
                ).astype(np.float_)

            # 加载视差图（Disparity)
            if data_path["disp_path"] is not None:
                curr_disp = self.load_disparity(
                    data_path["disp_path"], image=None, const=None, dtype=int
                )

        # 预处理输入， crop
        if self.pre_crop_edge is not None:
            crop_x, crop_y = self.pre_crop_edge
        elif self.pre_crop_size is not None:
            h, w = curr_rgb.shape[:2]
            croph, cropw = self.pre_crop_size
            crop_x = int((w - cropw) / 2)
            crop_y = int((h - croph) / 2)
        else:
            crop_x = crop_y = 0

        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)

        if crop_x > 0 or crop_y > 0:
            if curr_intrinsics is not None:
                curr_intrinsics[2] -= crop_x
                curr_intrinsics[3] -= crop_y

            def crop_edge(data):
                if data is None:
                    return data
                if crop_y > 0:
                    data = data[crop_y:-crop_y]
                if crop_x > 0:
                    data = data[:, crop_x:-crop_x]
                return data
                # return data if data is None else data[crop_y:-crop_y, crop_x:-crop_x]

            curr_rgb = crop_edge(curr_rgb)
            curr_depth = crop_edge(curr_depth)
            curr_depth_mask = crop_edge(curr_depth_mask)
            curr_sem = crop_edge(curr_sem)
            curr_normal = crop_edge(curr_normal)
            curr_eval_mask = crop_edge(curr_eval_mask)
            curr_invalid_mask = crop_edge(curr_invalid_mask)
            curr_prompt = crop_edge(curr_prompt)
            curr_disp = crop_edge(curr_disp)
            curr_inf_mask = crop_edge(curr_inf_mask)

        data_batch = dict(
            curr_intrinsics=curr_intrinsics,
            curr_extrinsics=curr_extrinsics,
            curr_prompt_extrinsics=curr_prompt_extrinsics,
            curr_rgb=curr_rgb,
            curr_depth=curr_depth,
            curr_depth_mask=curr_depth_mask,
            curr_sem=curr_sem,
            curr_normal=curr_normal,
            curr_eval_mask=curr_eval_mask,
            curr_invalid_mask=curr_invalid_mask,
            curr_prompt=curr_prompt,
            curr_disp=curr_disp,
            curr_inf_mask=curr_inf_mask,
        )
        return data_batch

    def load_inf_mask(self, curr_depth, curr_sem, curr_rgb, data_info, **kwargs):
        curr_inf_mask = np.zeros(curr_rgb.shape[:2], dtype=np.bool_)
        if curr_depth is not None:
            if self.inf_mask_min is not None:
                curr_inf_mask = np.logical_or(curr_inf_mask, curr_depth < self.inf_mask_min)
            if self.inf_mask_max is not None:
                curr_inf_mask = np.logical_or(curr_inf_mask, curr_depth > self.inf_mask_max)
            if self.inf_mask_depth_nan:
                depth_mask = np.logical_or(np.isnan(curr_depth), np.isinf(curr_depth))
                curr_inf_mask = np.logical_or(curr_inf_mask, depth_mask)
        if (
            self.inf_mask_sem_names is not None
            and len(self.inf_mask_sem_names) > 0
            and curr_sem is not None
        ):
            sem_mask = self.process_sem_mask(curr_sem, self.inf_mask_sem_names)
            curr_inf_mask = np.logical_or(curr_inf_mask, sem_mask)
        return curr_inf_mask

    def load_sift_mask(self, curr_rgb, curr_depth, uv_range=3, d_delta=0.1):
        target = curr_depth.squeeze()
        height, width = curr_rgb.shape[:2]
        gray = cv2.cvtColor(curr_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        detector = self.sift_detector
        keypoints = detector.detect(gray)
        mask = np.zeros([height, width])
        mask_pt = np.zeros([height, width])

        for keypoint in keypoints:
            x = round(keypoint.pt[1])
            y = round(keypoint.pt[0])

            x0 = max(x - uv_range, 0)
            x1 = min(x + uv_range, height - 1)
            y0 = max(y - uv_range, 0)
            y1 = min(y + uv_range, width - 1)

            depth_this_pt = target[x, y]
            if depth_this_pt < 1e-3:
                continue
            mask_this_pt = (target[x0:x1, y0:y1] / depth_this_pt - 1) < d_delta
            mask[x0:x1, y0:y1] = mask_this_pt
            mask_pt[x, y] = 1

        return mask_pt, mask

    def load_normal(self, normal_path, image, const, dtype):
        return self.read_file(normal_path, image=image, const=const, dtype=dtype)

    def load_depth(self, depth_path, image, const, dtype):
        return self.read_file(depth_path, image=image, const=const, dtype=dtype)

    def load_disparity(self, disp_path, image, const, dtype):
        return self.read_file(disp_path, image=image, const=const, dtype=dtype)

    def read_image(self, image_path):
        """
        Reads an image from the specified file path and returns it as a NumPy array.

        Args:
            image_path (str or None): The file path to the image. If None, returns None.

        Returns:
            np.ndarray or None: The image as a NumPy array in RGB format, or None if image_path is None.

        Raises:
            RuntimeError: If the file extension is not supported.
        """
        # Return None if the image_path is None.
        if image_path is None:
            return None

        # Extract the file extension from the image_path.
        data_type = os.path.splitext(image_path)[-1].lower()

        # List of supported image file types.
        img_file_type = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]

        # Handle different file types.
        if data_type in img_file_type:
            # Open the image using PIL and convert it to RGB format.
            data = Image.open(image_path).convert("RGB")  # [H, W, rgb]
            # Convert the PIL Image to a NumPy array.
            data = np.asarray(data)
        elif data_type in [".hdf5", ".h5"]:
            # Open the HDF5 file and read the dataset.
            with h5py.File(image_path, "r") as f:
                data = np.array(f["dataset"])
        else:
            # Raise an error if the file type is not supported.
            raise RuntimeError(f"File type {data_type} is not supported in the current version.")

        return data

    def read_file(self, file_path, image=None, const=0, dtype=np.float32):
        """
        Reads data from the specified file path or creates a constant-valued array based on the given image dimensions.

        Args:
            file_path (str or None): The file path to the data file. If None, uses the dimensions from the provided image.
            image (np.ndarray or None): An optional image to infer dimensions if file_path is None.
            const (float): The constant value to fill the array if file_path is None.
            dtype (type): The data type for the output array.

        Returns:
            np.ndarray or None: The data as a NumPy array, or None if both file_path and image are None.

        Raises:
            RuntimeError: If the file extension is not supported.
        """
        # Return None if both file_path and image are None.
        if file_path is None and image is None:
            return None

        # Create a constant-valued array based on the image dimensions if file_path is None.
        if file_path is None or not os.path.exists(file_path):
            data = np.zeros(image.shape, dtype=dtype) + const
        else:
            # Extract the file extension from the file_path.
            data_type = os.path.splitext(file_path)[-1].lower()

            # List of supported image file types.
            img_file_type = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]

            # Handle different file types.
            if data_type in img_file_type:
                # Open the image using PIL and convert it to a NumPy array.
                data = Image.open(file_path)
                data = np.asarray(data)
            elif data_type in [".npz", ".npy"]:
                # Load the data from a .npz or .npy file.
                data = np.load(file_path)
                # If it's a .npz file, extract the first array (assuming single array storage).
                if isinstance(data, np.lib.npyio.NpzFile):
                    data = data[data.files[0]]
            elif data_type in [".hdf5", ".h5"]:
                # Open the HDF5 file and read the dataset.
                with h5py.File(file_path, "r") as f:
                    data = np.array(f["dataset"])
            elif data_type in [".pfm"]:
                data = self.load_pfm(file_path)
            elif data_type == ".exr":
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                data = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            elif data_type in [".mat"]:
                data = loadmat(file_path)
                keys = list(data.keys())
                for key in ["__header__", "__version__", "__globals__"]:
                    if key in keys:
                        keys.remove(key)
                data = data[keys[0]].squeeze()
                assert isinstance(data, np.ndarray)
            elif data_type in [".dpt"]:
                with open(file_path, "rb") as f:
                    _ = np.fromfile(f, dtype=np.float32, count=1)[0]
                    width = np.fromfile(f, dtype=np.int32, count=1)[0]
                    height = np.fromfile(f, dtype=np.int32, count=1)[0]
                    size = width * height
                    assert (
                        width > 0 and height > 0 and size > 1 and size < 100000000
                    ), " depth_read:: Wrong input size (width = {0}, height = {1}).".format(
                        width, height
                    )
                    data = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
            elif data_type in [".float3"]:
                with open(file_path, "rb") as f:
                    if (f.readline().decode("utf-8")) != "float\n":
                        raise Exception("float file %s did not contain <float> keyword" % file_path)

                    dim = int(f.readline())

                    dims = []
                    count = 1
                    for i in range(0, dim):
                        d = int(f.readline())
                        dims.append(d)
                        count *= d

                    dims = list(reversed(dims))
                    data = np.fromfile(f, np.float32, count).reshape(dims)
            else:
                # Raise an error if the file type is not supported.
                raise RuntimeError(
                    f"File type {data_type} is not supported in the current version."
                )

        # Ensure the data is of the specified data type.
        data = data.astype(dtype)

        return data

    def load_pfm(self, depth_path):
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        with open(depth_path, "rb") as file:
            header = file.readline().decode("UTF-8").rstrip()
            if header == "PF":
                color = True
            elif header == "Pf":
                color = False
            else:
                raise Exception("Not a PFM file.")
            dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("UTF-8"))
            if dim_match:
                width, height = map(int, dim_match.groups())
            else:
                raise Exception("Malformed PFM header.")
            # scale = float(file.readline().rstrip())
            scale = float((file.readline()).decode("UTF-8").rstrip())
            if scale < 0:  # little-endian
                data_type = "<f"
            else:
                data_type = ">f"  # big-endian
            data_string = file.read()
            data = np.fromstring(data_string, data_type)
            shape = (height, width, 3) if color else (height, width)
            data = np.reshape(data, shape)
            data = cv2.flip(data, 0)

        return data

    def remap_sem_label(self, curr_sem, data_info):
        return remap_sem_label(curr_sem, self.semantic_labels)

    def process_sem_mask(self, sem_label: torch.Tensor, invalid_names=None, semantic_labels=None):
        """
        Generates a boolean mask indicating the invalid depth regions based on the provided semantic label map.
        """
        if invalid_names is None:
            invalid_names = self.depth_invalid_sem_names
        if semantic_labels is None:
            semantic_labels = SEM_LABEL

        if (
            semantic_labels is None
            or sem_label is None
            or invalid_names is None
            or len(invalid_names) == 0
        ):
            return None

        # Convert the list of invalid semantic class names to their corresponding labels
        invalid_labels = [
            semantic_labels[name] for name in invalid_names if name in semantic_labels
        ]

        if len(invalid_labels) == 0:
            return None

        # Initialize the mask as False (valid depth)
        if isinstance(sem_label, torch.Tensor):
            sem_mask = torch.zeros_like(sem_label, dtype=torch.bool)
        else:
            sem_mask = np.zeros_like(sem_label, dtype=np.bool_)
        # Generate the mask by checking if each pixel belongs to any of the invalid labels
        for label in invalid_labels:
            sem_mask |= sem_label == label

        if len(sem_label.shape) > 2:
            if isinstance(sem_label, torch.Tensor):
                sem_mask = torch.all(sem_mask, dim=-1)
            else:
                sem_mask = np.all(sem_mask, axis=-1)
        return sem_mask

    def process_disp(self, disp, image, curr_rgb):
        if disp is not None:
            disp_scale = image.shape[-1] / curr_rgb.shape[1]
            disp = disp * disp_scale
            disp = disp / disp.shape[-1]
            disp_valid = disp > 1e-3
        else:
            disp_valid = None
        return disp, disp_valid

    def create_intrinsics_matrix(self, intrinsics):
        """Create an intrinsics matrix from a list of parameters."""
        intrinsics_mat = torch.zeros((3, 3)).float()
        intrinsics_mat[0, 0] = intrinsics[0]
        intrinsics_mat[1, 1] = intrinsics[1]
        intrinsics_mat[0, 2] = intrinsics[2]
        intrinsics_mat[1, 2] = intrinsics[3]
        intrinsics_mat[2, 2] = 1.0
        return intrinsics_mat

    def create_extrinsics_matrix(self, extrinsics):
        """Create an extrinsics matrix from a list of parameters."""
        extrinsics_mat = torch.tensor(extrinsics).float().reshape(4, 4)
        return extrinsics_mat

    def depth_filter(self, depth, depth_mask, sem_label=None):
        # Create a mask for valid depth values based on min_depth and max_depth thresholds.
        if depth is None:
            return None, None

        depth_valid_mask = (depth > self.min_depth) & (depth < self.max_depth)

        if sem_label is not None:
            sem_mask = self.process_sem_mask(sem_label)
            if sem_mask is not None:
                depth_valid_mask = depth_valid_mask & (~sem_mask)

        depth_valid_mask = depth_valid_mask.astype(int)

        # Combine the provided depth_mask with the valid depth mask if depth_mask is given,
        # otherwise use the valid depth mask alone.
        if depth_mask is not None:
            depth_mask = depth_mask * depth_valid_mask
        else:
            depth_mask = depth_valid_mask

        if self.quantile_filter > 0:
            depth_quantile_filter = float(
                torch.quantile(
                    torch.from_numpy(depth[depth_mask.astype(bool)]), self.quantile_filter
                )
            )
            depth_mask = depth_mask * (depth <= depth_quantile_filter)

        depth[depth_mask == 0] = 0

        return depth, depth_mask

    def get_sparse_depth_with_pattern(
        self,
        curr_rgb,
        curr_depth,
        curr_depth_mask,
        curr_intrinsics,
        rgb,
        depth,
        depth_mask,
        intrinsics_mat,
        transform_info,
        data_idx,
        **kwargs,
    ):
        # Extracts sparse depth points and their corresponding masks from the origin depthmap
        if self.sparse_pattern_v3 is not None:
            sparse_pointmap, sparse_pointmap_mask = self.get_sparse_depth_pattern(
                self.sparse_pattern_v3,
                curr_rgb,
                curr_depth,
                curr_depth_mask,
                curr_intrinsics,
                transform_info,
                intrinsics_mat.numpy(),
                H=depth.shape[1] if depth is not None else rgb.shape[1],
                W=depth.shape[2] if depth is not None else rgb.shape[2],
                data_idx=data_idx,
            )
        else:
            return None, None

        if sparse_pointmap_mask is not None:
            if sparse_pointmap_mask.sum() == 0:
                logging.warning("sparse_pointmap_mask is empty, no valid points found.")
                raise ValueError("No valid points in the sparse point map.")

        return sparse_pointmap, sparse_pointmap_mask

    def get_sparse_depth_pattern(
        self,
        sparse_pattern,
        curr_rgb,
        curr_depth,
        curr_depth_mask,
        curr_intrinsics,
        transform_info,
        new_intrinsics_mat,
        H,
        W,
        data_idx,
        **kwargs,
    ):
        """
        Generates a sparse depth pattern based on the current depth map, intrinsics matrix, and transformation information.

        Args:
            curr_depth (numpy.ndarray): The current depth map of shape (H, W).
            curr_depth_mask (numpy.ndarray): A binary mask indicating valid depth values of shape (H, W).
            curr_intrinsics (list or numpy.ndarray): intrinsics camera parameters [fx, fy, cx, cy].
            transform_info (dict): Transformation information including horizontal flip flag.
            new_intrinsics_mat (numpy.ndarray): New intrinsics matrix after transformation.
            H (int): Height of the target image.
            W (int): Width of the target image.

        Returns:
            tuple: A tuple containing two elements:
                - sparse_pointmap (numpy.ndarray): A tensor of shape (H, W, 3) with sparse 3D points.
                - sparse_pointmap_mask (numpy.ndarray): A binary mask of shape (H, W) indicating valid sparse points.
        """

        if sparse_pattern is None:
            return None, None

        if transform_info.get("horizontal_flip", False):
            curr_depth = cv2.flip(curr_depth, 1)
            curr_depth_mask = cv2.flip(curr_depth_mask, 1)
            h, w = curr_rgb.shape[:2]
            curr_intrinsics[2] = w - 1 -curr_intrinsics[2]

        if transform_info.get("vertical_flip", False):
            curr_depth = cv2.flip(curr_depth, 0)
            curr_depth_mask = cv2.flip(curr_depth_mask, 0)
            h, w = curr_rgb.shape[:2]
            curr_intrinsics[3] = h - 1 -curr_intrinsics[3]

        curr_intrinsics_mat = self.create_intrinsics_matrix(curr_intrinsics)

        curr_pointmap = self.load_pointmap(curr_depth, curr_intrinsics_mat)

        sparse_intrinsics = new_intrinsics_mat.copy()

        if self.sparse_max_size is not None:
            if W < self.sparse_max_size and H < self.sparse_max_size:
                new_width = int(W / self.sparse_patch_size) * self.sparse_patch_size
                new_height = int(H / self.sparse_patch_size) * self.sparse_patch_size
            elif W >= H:
                new_width = self.sparse_max_size
                new_height = (
                    round(H * (new_width / W) / self.sparse_patch_size) * self.sparse_patch_size
                )
            else:
                new_height = self.sparse_max_size
                new_width = (
                    round(W * (new_height / H) / self.sparse_patch_size) * self.sparse_patch_size
                )
            ratio_h, ratio_w = new_height / H, new_width / W
        elif self.sparse_width_ratio is not None and self.sparse_height_ratio is not None:
            ratio_h, ratio_w = self.sparse_height_ratio, self.sparse_width_ratio
            new_height, new_width = int(ratio_h * H), int(ratio_w * W)
        elif self.sparse_width is not None and self.sparse_height is not None:
            new_height, new_width = self.sparse_height, self.sparse_width
            ratio_h, ratio_w = 1.0 * self.sparse_height / H, 1.0 * self.sparse_width / W
        else:
            new_height, new_width = H, W
            ratio_h = ratio_w = 1.0

        sparse_intrinsics[0, :] = sparse_intrinsics[0, :] * ratio_w
        sparse_intrinsics[1, :] = sparse_intrinsics[1, :] * ratio_h

        # Get sparse depth pattern using the sparse pattern object
        sparse_pointmap, sparse_pointmap_mask = sparse_pattern.get_sparse_depth(
            curr_pointmap,
            curr_depth_mask,
            sparse_intrinsics,
            new_height,
            new_width,
            rgb=curr_rgb,
            data_idx=data_idx,
        )
        return sparse_pointmap, sparse_pointmap_mask

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

                # If sparse depth sampling is enabled, generate a sparse depth map by selecting a certain percentage of points randomly.
                if (
                    self.sparse_depth_ratio > 0
                    or self.sparse_depth_nums > 0
                    or sparse_pointmap is not None
                ):
                    # assert sparse_pointmap is not None, "use get_sparse_depth_pattern"

                    if sparse_pointmap is None:
                        sparse_depth, sparse_depth_mask = self.load_sparse_depth(depth, depth_mask)
                    else:
                        sparse_depth = sparse_pointmap[2:3, :, :]
                        sparse_depth_mask = sparse_pointmap_mask

                    # Normalize the sparse depth using the same normalize_depth function.
                    if self.normalize_depth is not None:
                        sparse_depth_norm = self.normalize_depth(sparse_depth, sparse_depth_mask)
                        sparse_depth_max, sparse_depth_min = (
                            sparse_depth[sparse_depth_mask].max(),
                            sparse_depth[sparse_depth_mask].min(),
                        )

            # If point cloud generation is enabled, create a point cloud map using the depth, intrinsics parameters, and depth mask.
            if self.with_pointmap or self.only_pointmap:
                pointmap = self.load_pointmap(depth, intrinsics)

                if (
                    self.sparse_depth_ratio > 0
                    or self.sparse_depth_nums > 0
                    or sparse_pointmap is not None
                ):
                    # assert sparse_pointmap is not None, "use get_sparse_depth_pattern"

                    if sparse_pointmap is None:
                        sparse_pointmap, sparse_pointmap_mask = self.load_sparse_depth(
                            pointmap, depth_mask
                        )
                        sparse_pointmap_mask = sparse_pointmap_mask[0:1, ...]

                    sparse_pointmap_center, sparse_pointmap_max_range = self.point_normalize(
                        sparse_pointmap, sparse_pointmap_mask
                    )

                    if self.interpolate_version == "v1":
                        sparse_dense_pointmap = self.depth_interpolate_torch(
                            sparse_pointmap, sparse_pointmap_mask
                        )
                    elif self.interpolate_version == "v2":
                        sparse_dense_pointmap = self.depth_interpolate_torch_v2(
                            sparse_pointmap, sparse_pointmap_mask.squeeze(0)
                        )
                    elif self.interpolate_version == "v3":
                        sparse_dense_pointmap = self.depth_interpolate_torch_v3(
                            sparse_pointmap, sparse_pointmap_mask.squeeze(0), intrinsics=intrinsics
                        )
                    if sparse_dense_pointmap is not None and torch.isnan(
                        sparse_dense_pointmap.sum()
                    ):
                        logging.warning(
                            "sparse_dense_pointmap contains NaN values, no valid points found."
                        )
                        raise ValueError("No valid points in the sparse_dense_pointmap.")
                else:
                    # NOTE: 沿用 sparse_pointmap_max_range
                    sparse_pointmap_center, sparse_pointmap_max_range = self.point_normalize(
                        pointmap, depth_mask
                    )

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
            # "sparse_pointmap_center": sparse_pointmap_center,
            # "sparse_pointmap_max": sparse_pointmap_max,
            # "sparse_pointmap_min": sparse_pointmap_min,
            "sparse_pointmap_max_range": sparse_pointmap_max_range,
            "edge_mask": edge_mask,
            "sparse_dense_pointmap": sparse_dense_pointmap,
            "sparse_dense_abs_diffmap": sparse_dense_abs_diffmap,
        }
        return depth_output

    def depth2invdepth(self, depth, valid_mask):
        """
        Converts a depth map to an inverse depth map and generates a mask for valid inverse depth values.

        Args:
            depth (torch.Tensor): The raw depth map as a tensor.
            valid_mask (torch.Tensor): A boolean mask indicating valid depth values.

        Returns:
            tuple: A tuple containing the inverse depth map and its corresponding validity mask.
        """
        # Compute the inverse depth, ensuring that values are clipped between min_depth and max_depth.
        inv_depth = 1.0 / torch.clip(depth, self.min_depth, self.max_depth)

        # Set invalid inverse depth values (negative) to -1.0.
        inv_depth[inv_depth < 0] = -1.0

        # Set inverse depth values outside the valid mask to -1.0.
        inv_depth[~valid_mask] = -1.0

        # Create a new mask for valid inverse depth values, where inverse depth is greater than 1e-6.
        inv_depth_mask = valid_mask & (inv_depth > 1e-6)

        return inv_depth, inv_depth_mask

    @staticmethod
    def center_and_normalize_point_cloud(points, valid_mask):
        """
        Centers and normalizes a point cloud based on the valid points.

        Args:
            points (np.ndarray): The point cloud as a NumPy array.
            valid_mask (np.ndarray): A boolean mask indicating valid points in the point cloud.

        Returns:
            np.ndarray: The centered and normalized point cloud.
        """
        # Calculate the mean of the valid points to find the center.
        center = np.mean(points[valid_mask], axis=0)

        # Center the point cloud by subtracting the mean from all points.
        centered_points = points - center

        # Calculate the maximum range of the centered points.
        max_range = np.max(np.linalg.norm(centered_points[valid_mask], axis=1))

        return center, max_range

    @staticmethod
    def min_and_normalize_point_cloud(points, valid_mask):
        min_val = torch.quantile(points.reshape(-1, 3)[valid_mask], 0.0, dim=0)
        max_val = torch.quantile(points.reshape(-1, 3)[valid_mask], 1.0, dim=0)

        scale_norm = torch.norm(max_val - min_val)
        scale_norm = torch.clamp(scale_norm, min=1.0)
        return min_val, scale_norm

    def load_pointmap(self, depth, intrinsics):
        """
        Generates a point cloud map from a depth map and intrinsics camera parameters.

        Args:
            depth (torch.Tensor or np.ndarray): The depth map as a tensor or NumPy array.
            intrinsics (torch.Tensor or np.ndarray): The intrinsics camera parameters as a tensor or NumPy array.

        Returns:
            torch.Tensor: The generated point cloud map as a tensor.
        """
        # Convert tensors to NumPy arrays if necessary.
        if isinstance(depth, torch.Tensor):
            depth = depth.squeeze(0).numpy()  # [ 1, h, w] -> [h, w]
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.numpy()

        height, width = depth.shape

        # Create a grid of pixel coordinates (u, v).
        if self.align_corners:
            u, v = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
        else:
            u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5, indexing="xy")
        uv = np.stack([u, v], axis=-1)

        # Convert the pixel coordinates to homogeneous coordinates.
        uv_homogeneous = np.concatenate([uv, np.ones([height, width, 1])], axis=-1).reshape(-1, 3)

        # Invert the intrinsics matrix to convert from image coordinates to world coordinates.
        K_inv = np.linalg.inv(intrinsics)

        # Compute the direction vectors from the camera to each pixel.
        directions = uv_homogeneous @ K_inv.T

        # Multiply the direction vectors by the depth values to get the 3D points.
        points = depth.reshape(-1, 1) * directions

        # Center and normalize the point cloud based on the valid points.
        # points = self.center_and_normalize_point_cloud(points, valid_mask.reshape(-1))

        # Reshape the point cloud to match the original depth map dimensions and transpose axes.
        points = points.reshape(height, width, 3).transpose(2, 0, 1)

        # Clip the point cloud values to be within [-1.0, 1.0].
        # points = np.clip(points, a_min=-1.0, a_max=1.0)

        # Convert the point cloud back to a PyTorch tensor.
        points = torch.from_numpy(points).float()

        return points

    def load_sparse_depth(self, depth, valid_mask):
        """
        Randomly selects a certain ratio of valid points from the given depth map and valid mask to generate a sparse depth map.

        Args:
            depth (torch.Tensor): The original depth map.
            valid_mask (torch.Tensor): A boolean mask indicating which points are valid.

        Returns:
            tuple: A tuple containing the sparsified depth map and the newly generated selection mask.
        """
        # Clone the original depth map to create a sparse depth map
        sparse_depth = depth.clone()

        # Get the indices of all valid points using the valid mask
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)

        # Count the number of valid points
        num_valid_points = len(valid_indices[0])

        # Determine the number of points to select based on a threshold
        if num_valid_points < 500:
            num_to_select = num_valid_points  # Select all points if there are fewer than 500
        elif self.sparse_depth_nums > 0:
            num_to_select = min(self.sparse_depth_nums, num_valid_points)
        else:
            num_to_select = int(
                num_valid_points * self.sparse_depth_ratio
            )  # Otherwise, select a ratio of points

        # Set a manual seed for reproducibility of random operations
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Randomly permute the indices of valid points and select the desired number of points
        random_indices = torch.randperm(num_valid_points)[:num_to_select]

        # Use the random indices to select the corresponding valid indices
        selected_valid_indices = tuple(idx[random_indices] for idx in valid_indices)

        # Create a mask that is True only at the locations of the selected valid points
        selected_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        selected_mask[selected_valid_indices] = True

        # Zero out the values in the sparse depth map where the selected mask is False
        # Note: There seems to be a typo in the original code. It should be 'sparse_depth' instead of 'sparse_depth_norm'
        if selected_mask.shape[0] != sparse_depth.shape[0]:
            selected_mask = selected_mask.repeat(sparse_depth.shape[0], 1, 1)
        sparse_depth[~selected_mask] = 0

        # Return the sparse depth map and the selection mask
        return sparse_depth, selected_mask

    def point_normalize(self, sparse_pointmap, sparse_pointmap_mask, min_range=0.1):
        """
        对稀疏点云进行归一化处理，支持多种归一化模式。

        Args:
            sparse_pointmap (torch.Tensor): 形状为 (3, H, W) 的点云坐标张量，其中 3 是 (x, y, z)。
            sparse_pointmap_mask (torch.Tensor): 形状为 (1, H, W) 或 (H, W) 的掩码, True 表示有效点。

        Returns:
            sparse_pointmap_center (torch.Tensor or None): 形状为 (3,) 的中心点(仅在 center 模式下返回)。
            sparse_pointmap_max_range (torch.Tensor or None): 形状为 () 的归一化尺度(0维标量张量)。
        """
        sparse_pointmap_center = sparse_pointmap_max_range = None

        if self.point_normalize_mode == "center":
            (
                sparse_pointmap_center,
                sparse_pointmap_max_range,
            ) = self.center_and_normalize_point_cloud(
                sparse_pointmap.reshape(3, -1).permute(1, 0).numpy(),
                sparse_pointmap_mask.reshape(-1).numpy(),
            )
            sparse_pointmap_center = torch.from_numpy(sparse_pointmap_center)
            sparse_pointmap_max_range = torch.tensor(sparse_pointmap_max_range)
        elif self.point_normalize_mode == "distance":
            sparse_pointmap_max_range = np.max(
                np.linalg.norm(
                    sparse_pointmap.reshape(3, -1).permute(1, 0).numpy(),
                    axis=1,
                )
            )
            sparse_pointmap_max_range = torch.tensor(sparse_pointmap_max_range).float()
        elif self.point_normalize_mode.startswith("distance_quantile"):
            sparse_pointmap_max_range = torch.quantile(
                torch.norm(
                    sparse_pointmap[:, sparse_pointmap_mask.squeeze(0)],
                    dim=0,
                ),
                float(self.point_normalize_mode.split("distance_quantile_")[-1]),
            ).float()
        elif self.point_normalize_mode == "distance_mean":
            sparse_pointmap_max_range = torch.mean(
                torch.norm(
                    sparse_pointmap[:, sparse_pointmap_mask.squeeze(0)],
                    dim=0,
                )
            ).float()
        elif self.point_normalize_mode.startswith("depth_quantile"):
            sparse_pointmap_max_range = torch.quantile(
                sparse_pointmap[-1, sparse_pointmap_mask.squeeze(0)],
                float(self.point_normalize_mode.split("depth_quantile_")[-1]),
            )

        if (
            self.phase == "train"
            and self.max_scale_noise is not None
            and self.max_scale_noise > 0
            and sparse_pointmap_max_range is not None
        ):
            noise = 1.0 - self.max_scale_noise + 2 * self.max_scale_noise * torch.rand(1)[0]
            sparse_pointmap_max_range = sparse_pointmap_max_range * noise

        if sparse_pointmap_max_range is not None and min_range is not None and sparse_pointmap_max_range < min_range:
            raise Exception(f"sparse pointmap_max_range is too small, {sparse_pointmap_max_range}")

        return sparse_pointmap_center, sparse_pointmap_max_range

    def depth_interpolate_torch(self, depthmap, valid_mask):
        """
        depthmap: [3, H, W]
        valid_mask: [1, H, W]
        return: [3, H, W]
        """
        depthmap = depthmap * valid_mask
        # depthmap[~valid_mask] = 0
        h, w = depthmap.shape[-2:]
        sparse_mask = depthmap[2, ...] > 0  # [H, W]
        valid_coords = (
            (sparse_mask > 0).nonzero(as_tuple=False).float()
        )  # 非零点的坐标 (i, j), [n, 2]
        valid_depths = depthmap[:, sparse_mask].T  # [n, 3]
        all_coords = torch.stack(
            torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 2)
        assert valid_depths.shape[0] > 3  # valid_depths num should be larger than 3
        interpolated_depths = knn_interpolate(
            x=valid_depths,  # 有效点的深度值
            pos_x=valid_coords,  # 有效点的坐标
            pos_y=all_coords,  # 目标插值的像素位置
            k=self.interpolate_k,  # k近邻数
        )
        interpolated_depths = interpolated_depths.view(h, w, 3).permute(2, 0, 1)
        return interpolated_depths

    def depth_interpolate_torch_v2(self, depthmap, valid_mask):
        """
        depthmap: [3, H, W]
        valid_mask: [H, W]
        return: [3, H, W]
        """
        h, w = depthmap.shape[-2:]
        valid_coords = valid_mask.nonzero(as_tuple=False).float()  # 非零点的坐标 (i, j), [n, 2]
        valid_depths = depthmap[:, valid_mask].T  # [n, 3]
        all_coords = torch.stack(
            torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 2)
        assert valid_depths.shape[0] > 3  # valid_depths num should be larger than 3
        interpolated_depths = knn_interpolate(
            x=valid_depths,  # 有效点的深度值
            pos_x=valid_coords,  # 有效点的坐标
            pos_y=all_coords,  # 目标插值的像素位置
            k=self.interpolate_k,  # k近邻数
        )
        interpolated_depths = interpolated_depths.view(h, w, 3).permute(2, 0, 1)
        interpolated_depths[:, valid_mask] = valid_depths.T
        return interpolated_depths

    def depth_interpolate_torch_v3(self, depthmap, valid_mask, intrinsics):
        """
        depthmap: [3, H, W]
        valid_mask: [H, W]
        return: [3, H, W]
        """
        h, w = depthmap.shape[-2:]
        valid_coords = valid_mask.nonzero(as_tuple=False).float()  # 非零点的坐标 (i, j), [n, 2]
        valid_depths = depthmap[2:3, valid_mask].T  # [n, 1]
        all_coords = torch.stack(
            torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 2)
        assert valid_depths.shape[0] > 3  # valid_depths num should be larger than 3
        interpolated_depths = knn_interpolate(
            x=valid_depths,  # 有效点的深度值
            pos_x=valid_coords,  # 有效点的坐标
            pos_y=all_coords,  # 目标插值的像素位置
            k=self.interpolate_k,  # k近邻数
        )
        interpolated_depths = interpolated_depths.view(h, w, 1).permute(2, 0, 1)
        interpolated_depths[:, valid_mask] = valid_depths.T

        interpolated_pointmap = self.load_pointmap(interpolated_depths, intrinsics)

        return interpolated_pointmap

    def debug_data_dict(self, idx, data_dict, prefix=""):
        if isinstance(self.data_path, (list, tuple)):
            data_path = self.data_path[0]
        else:
            data_path = self.data_path
        outdir = os.path.join(
            "./debug/dataset/",
            self.name,
            os.path.splitext(os.path.basename(data_path))[0],
            f"{idx:04d}",
        )
        os.makedirs(outdir, exist_ok=True)

        image = data_dict.get("image", None)
        intrinsics = data_dict.get("intrinsics", None)
        extrinsics = data_dict.get("extrinsics", None)
        depth = data_dict.get("depth", None)
        depth_mask = data_dict.get("depth_mask", None)
        pointmap = data_dict.get("pointmap", None)
        sparse_depth_norm = data_dict.get("sparse_depth_norm", None)
        sparse_depth_mask = data_dict.get("sparse_depth_mask", None)
        sem_label = data_dict.get("sem_label", None)
        normal = data_dict.get("normal", None)
        disparity = data_dict.get("disp", None)

        sparse_pointmap = data_dict.get("sparse_pointmap", None)
        sparse_pointmap_mask = data_dict.get("sparse_pointmap_mask", None)
        sparse_pointmap_center = data_dict.get("sparse_pointmap_center", None)
        sparse_pointmap_max_range = data_dict.get("sparse_pointmap_max_range", None)
        sparse_dense_pointmap = data_dict.get("sparse_dense_pointmap", None)

        edge_mask = data_dict.get("edge_mask", None)
        depth_invalid_sem_mask = data_dict.get("depth_invalid_sem_mask", None)
        sift_eval_mask = data_dict.get("sift_eval_mask", None)
        sift_point_mask = data_dict.get("sift_mask", None)
        inf_mask = data_dict.get("inf_mask", None)
        invalid_mask = data_dict.get("invalid_mask", None)

        if image is not None:
            if isinstance(image, torch.Tensor):
                image = image.numpy().transpose(1, 2, 0)
            save_path = os.path.join(outdir, f"{prefix}image.jpg")
            image = (image + 1.0) / 2.0 * 255.0
            cv2.imwrite(save_path, image[:, :, ::-1].astype(np.uint8))

            if edge_mask is not None:
                if isinstance(edge_mask, torch.Tensor):
                    edge_mask = edge_mask.squeeze(0).numpy()
                edge_mask_image = image.copy()
                edge_mask_image[edge_mask, :] = 0
                save_path = os.path.join(outdir, f"{prefix}edge_mask.jpg")
                cv2.imwrite(save_path, edge_mask_image[:, :, ::-1].astype(np.uint8))

            if depth_mask is not None:
                if isinstance(depth_mask, torch.Tensor):
                    depth_mask = depth_mask.squeeze(0).numpy()
                depth_mask_image = image.copy()
                depth_mask_image[depth_mask, :] *= 0.5
                depth_mask_image[depth_mask, 0] += 127
                save_path = os.path.join(outdir, f"{prefix}depth_mask.jpg")
                cv2.imwrite(save_path, depth_mask_image[:, :, ::-1].astype(np.uint8))

        if inf_mask is not None:
            if isinstance(inf_mask, torch.Tensor):
                inf_mask = inf_mask.squeeze(0).numpy()
            inf_mask_image = image.copy()
            inf_mask_image[inf_mask, :] *= 0.5
            inf_mask_image[inf_mask, 0] += 127
            save_path = os.path.join(outdir, f"{prefix}inf_mask.jpg")
            cv2.imwrite(save_path, inf_mask_image[:, :, ::-1].astype(np.uint8))

        if invalid_mask is not None:
            if isinstance(invalid_mask, torch.Tensor):
                invalid_mask = invalid_mask.squeeze(0).numpy()
            inf_mask_image = image.copy()
            inf_mask_image[invalid_mask, :] *= 0.5
            inf_mask_image[invalid_mask, 0] += 127
            save_path = os.path.join(outdir, f"{prefix}invalid_mask_mask.jpg")
            cv2.imwrite(save_path, inf_mask_image[:, :, ::-1].astype(np.uint8))

        if intrinsics is not None:
            if isinstance(intrinsics, torch.Tensor):
                intrinsics = intrinsics.numpy()
            intrinsics = [
                float(intrinsics[0, 0]),
                float(intrinsics[1, 1]),
                float(intrinsics[0, 2]),
                float(intrinsics[1, 2]),
            ]
            save_path = os.path.join(outdir, f"{prefix}intrinsics.json")
            with open(save_path, "w") as f:
                json.dump(intrinsics, f)

        if depth is not None:
            if isinstance(depth, torch.Tensor):
                depth = depth.squeeze(0).numpy()
            if isinstance(depth_mask, torch.Tensor):
                depth_mask = depth_mask.squeeze(0).numpy()
            depth_colored = colorize_depth_maps(
                depth,
                depth.min(),
                depth.max(),
                cmap="turbo",
                # valid_mask=depth_mask,
            )
            save_path = os.path.join(outdir, f"{prefix}depth.jpg")
            depth_colored.save(save_path)

        if sem_label is not None and (sem_label.ndim == 2 or sem_label.shape[-1] == 1):
            if isinstance(sem_label, torch.Tensor):
                sem_label = sem_label.numpy() + 1  # ignore = -1
            save_path = os.path.join(outdir, f"{prefix}sem.jpg")
            cv2.imwrite(save_path, sem_label.astype(np.uint8))

        if normal is not None:
            if isinstance(normal, torch.Tensor):
                normal = normal.numpy().transpose(1, 2, 0)
            normal = ((normal + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
            save_path = os.path.join(outdir, f"{prefix}normal.jpg")
            normal = cv2.imwrite(save_path, normal)

        pointmap_raw = data_dict.get("pointmap_raw", None)
        image_raw = data_dict.get("image_raw", None)
        depth_raw_mask = data_dict.get("depth_raw_mask", None)
        normal_raw = data_dict.get("normal_raw", None)

        if normal_raw is not None:
            if isinstance(normal_raw, torch.Tensor):
                normal_raw = normal_raw.numpy().transpose(1, 2, 0)
            normal_raw = ((normal_raw + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
            save_path = os.path.join(outdir, f"{prefix}normal_raw.jpg")
            normal_raw = cv2.imwrite(save_path, normal_raw)

        if pointmap_raw is not None:
            import open3d as o3d
            if isinstance(pointmap_raw, torch.Tensor):
                pointmap_raw = pointmap_raw.numpy().transpose(1, 2, 0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointmap_raw[depth_raw_mask, :].reshape(-1, 3))
            if image_raw is not None:
                if isinstance(image_raw, torch.Tensor):
                    image_raw = image_raw.numpy()
                save_path = os.path.join(outdir, f"{prefix}image_raw.jpg")
                # image_raw = (image_raw + 1.0) / 2.0 * 255.0
                cv2.imwrite(save_path, image_raw[:, :, ::-1].astype(np.uint8))
                pcd.colors = o3d.utility.Vector3dVector(
                    image_raw[depth_raw_mask, :].reshape(-1, 3) / 255.0
                )
            save_path = os.path.join(outdir, f"{prefix}pointmap_raw.ply")
            o3d.io.write_point_cloud(save_path, pcd)

        if pointmap is not None:
            import open3d as o3d

            if isinstance(pointmap, torch.Tensor):
                pointmap = pointmap.numpy().transpose(1, 2, 0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointmap[depth_mask, :].reshape(-1, 3))
            if image is not None:
                pcd.colors = o3d.utility.Vector3dVector(
                    image[depth_mask, :].reshape(-1, 3) / 255.0
                )
            save_path = os.path.join(outdir, f"{prefix}pointmap.ply")
            o3d.io.write_point_cloud(save_path, pcd)

            if sift_eval_mask is not None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                    pointmap[sift_point_mask.squeeze(), :].reshape(-1, 3)
                )
                if image is not None:
                    pcd.colors = o3d.utility.Vector3dVector(
                        image[sift_point_mask.squeeze(), :].reshape(-1, 3) / 255.0
                    )
                save_path = os.path.join(outdir, f"{prefix}sift_pointmap.ply")
                o3d.io.write_point_cloud(save_path, pcd)

            if extrinsics is not None:
                extrinsics = extrinsics.cpu().squeeze(0).numpy()
                extrinsics_inv = np.linalg.inv(extrinsics)
                R = extrinsics_inv[:3, :3]
                T = extrinsics_inv[:3, 3]

                pointmap_globla = np.dot(R, pointmap[depth_mask, :].reshape(-1, 3).T).T + T

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pointmap_globla)
                if image is not None:
                    pcd.colors = o3d.utility.Vector3dVector(
                        image[depth_mask, :].reshape(-1, 3) / 255.0
                    )
                save_path = os.path.join(outdir, f"{prefix}pointmap_global.ply")
                o3d.io.write_point_cloud(save_path, pcd)

                save_path = os.path.join(outdir, f"{prefix}extrinsics.json")
                with open(save_path, "w") as f:
                    json.dump(extrinsics.tolist(), f, indent=2)

        if pointmap is not None and sparse_dense_pointmap is not None:
            if isinstance(sparse_dense_pointmap, torch.Tensor):
                sparse_dense_pointmap = sparse_dense_pointmap.numpy().transpose(1, 2, 0)

            if pointmap.size == sparse_dense_pointmap.size:
                l1_diff = np.abs(pointmap - sparse_dense_pointmap).sum(axis=-1)

                l1_diff_colored = colorize_depth_maps(
                    l1_diff, l1_diff.min(), l1_diff.max(), cmap="turbo"
                )
                save_path = os.path.join(outdir, f"{prefix}l1_diff.jpg")
                l1_diff_colored.save(save_path)

                gt_noise_postive = 1.0 - (l1_diff > 0.05).astype(float)
                gt_noise_postive_colored = colorize_depth_maps(gt_noise_postive, 0, 1, cmap="turbo")
                save_path = os.path.join(outdir, f"{prefix}input_diff_mask.jpg")
                gt_noise_postive_colored.save(save_path)

        if sparse_depth_norm is not None:
            if isinstance(sparse_depth_norm, torch.Tensor):
                sparse_depth_norm = sparse_depth_norm.squeeze(0).numpy()
                sparse_depth_mask = sparse_depth_mask.squeeze(0).numpy()
            sparse_depth_colored = colorize_depth_maps(
                sparse_depth_norm,
                self.normalize_depth.norm_min,
                self.normalize_depth.norm_max,
                cmap="turbo",
                valid_mask=sparse_depth_mask,
            )
            save_path = os.path.join(outdir, f"{prefix}sparse_depth.jpg")
            sparse_depth_colored.save(save_path)

        if sparse_pointmap is not None:
            import open3d as o3d

            if isinstance(sparse_pointmap, torch.Tensor):
                sparse_pointmap = sparse_pointmap.squeeze(0).numpy().transpose(1, 2, 0)
                sparse_pointmap_mask = sparse_pointmap_mask.squeeze(0).numpy()
                sparse_pointmap_max_range = sparse_pointmap_max_range.numpy()

                if sparse_pointmap_center is not None:
                    sparse_pointmap_center = sparse_pointmap_center.numpy()
                else:
                    sparse_pointmap_center = np.zeros([1])

            sparse_pointmap_z = sparse_pointmap[:, :, 2]
            sparse_pointmap_z = (sparse_pointmap_z - sparse_pointmap_z.min()) / (
                sparse_pointmap_z.max() - sparse_pointmap_z.min()
            )
            sparse_colored = colorize_depth_maps(
                sparse_pointmap_z,
                0,
                1,
                cmap="turbo",
                valid_mask=None,
            )
            save_path = os.path.join(outdir, f"{prefix}sparse_pointmap.jpg")
            sparse_colored.save(save_path)

            sparse_pointmap = sparse_pointmap[sparse_pointmap_mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sparse_pointmap)
            if image is not None:
                if image.shape[:2] != sparse_pointmap_mask.shape[:2]:
                    tmp_image = cv2.resize(
                        image,
                        dsize=(sparse_pointmap_mask.shape[-1], sparse_pointmap_mask.shape[-2]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    pcd.colors = o3d.utility.Vector3dVector(tmp_image[sparse_pointmap_mask] / 255.0)
                else:
                    pcd.colors = o3d.utility.Vector3dVector(image[sparse_pointmap_mask] / 255.0)
            save_path = os.path.join(outdir, f"{prefix}sparse_pointmap.ply")
            o3d.io.write_point_cloud(save_path, pcd)

            sparse_pointmap = sparse_pointmap / sparse_pointmap_max_range
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sparse_pointmap)
            if image is not None:
                if image.shape[:2] != sparse_pointmap_mask.shape[:2]:
                    pcd.colors = o3d.utility.Vector3dVector(tmp_image[sparse_pointmap_mask] / 255.0)
                else:
                    pcd.colors = o3d.utility.Vector3dVector(image[sparse_pointmap_mask] / 255.0)
            save_path = os.path.join(outdir, f"{prefix}sparse_unit_pointmap.ply")
            o3d.io.write_point_cloud(save_path, pcd)

            print("depth_mask", depth_mask.sum(), "sparse_mask", sparse_pointmap_mask.sum())

        if sparse_dense_pointmap is not None:
            import open3d as o3d

            if isinstance(sparse_dense_pointmap, torch.Tensor):
                sparse_dense_pointmap = sparse_dense_pointmap.numpy().transpose(1, 2, 0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sparse_dense_pointmap.reshape(-1, 3))
            if image is not None:
                pcd.colors = o3d.utility.Vector3dVector(image.reshape(-1, 3) / 255.0)
            save_path = os.path.join(outdir, f"{prefix}sparse_dense_pointmap.ply")
            o3d.io.write_point_cloud(save_path, pcd)

            sparse_dense_pointmap = sparse_dense_pointmap[:, :, 2]
            sparse_dense_pointmap = (sparse_dense_pointmap - sparse_dense_pointmap.min()) / (
                sparse_dense_pointmap.max() - sparse_dense_pointmap.min()
            )

            sparse_dense_colored = colorize_depth_maps(
                sparse_dense_pointmap,
                0,
                1,
                cmap="turbo",
                valid_mask=None,
            )
            save_path = os.path.join(outdir, f"{prefix}sparse_dense_pointmap.jpg")
            sparse_dense_colored.save(save_path)

        if depth_invalid_sem_mask is not None:
            if isinstance(depth_invalid_sem_mask, torch.Tensor):
                depth_invalid_sem_mask = depth_invalid_sem_mask.cpu().float().squeeze(0).numpy()
            sem_mask_color = colorize_depth_maps(
                depth_invalid_sem_mask,
                0,
                1,
                cmap="turbo",
                valid_mask=None,
            )
            save_path = os.path.join(outdir, f"{prefix}depth_invalid_sem_mask.jpg")
            sem_mask_color.save(save_path)

        if disparity is not None:
            if isinstance(disparity, torch.Tensor):
                disparity = disparity.squeeze(0).numpy()
            disparity_colored = colorize_depth_maps(
                disparity,
                disparity.min(),
                disparity.max(),
                cmap="turbo",
                # valid_mask=depth_mask,
            )
            save_path = os.path.join(outdir, f"{prefix}disp.jpg")
            disparity_colored.save(save_path)

        return outdir


if __name__ == "__main__":

    dataset = BaseDataset(
        phase="test",
        name="base",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/Hypersim/test.json",
        sampling_strategy="all",
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=630,
                height=476,
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
        max_depth=60.0,
        debug=True,
        with_pointmap=True,
        sparse_pattern_v3=dict(
            type="hAlgorithm.datasets.patterns.random_patch_pattern.Pattern",
            sparse_ratio=[0.01, 1.0],
            patch_size=14,
            patch_dropout_ratio=[0.5, 0.9],
        ),
    )

    print(len(dataset))

    for index in range(min(10, len(dataset))):
        print(index)
        dataset.__getitem__(index)
