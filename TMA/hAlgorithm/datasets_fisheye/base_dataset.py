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
from hAlgorithm.datasets.base_dataset import BaseDataset
from hAlgorithm.datasets_fisheye.utils.fisheye_blender import load_pointmap_blender_polynomial_fisheye


class BaseDatasetFisheye(BaseDataset):
    semantic_labels = None

    def __init__(
        self,
        camera_type: str = "PINHOLE", # one of "PINHOLE", "FISHEYE_EQUIDISTANT", "FISHEYE_OPENCV"
        fisheye_crop_pixel_normal: int = 0,
        fisheye_crop_pixel_pointmap: int = 0,
        fisheye_edge_value: int = 2,
        fisheye_edge_pixel: int = 5,
        distorts_name: str = "distor_k",
        sensor_size_name: str = "sensor_size",
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
        
        assert camera_type in ["PINHOLE", "FISHEYE_EQUIDISTANT", "FISHEYE_OPENCV", "FISHEYE_BLENDER"]
        self.camera_type = camera_type
        
        self.fisheye_crop_pixel_normal = fisheye_crop_pixel_normal
        self.fisheye_crop_pixel_pointmap = fisheye_crop_pixel_pointmap
        self.fisheye_edge_value = fisheye_edge_value
        self.fisheye_edge_pixel = fisheye_edge_pixel
        self.distorts_name = distorts_name
        self.sensor_size_name = sensor_size_name
        
        super().__init__(**kwargs)

    def is_fisheye_camera(self):
        return self.camera_type in ["FISHEYE_EQUIDISTANT", "FISHEYE_OPENCV", "FISHEYE_BLENDER"]

    def get_fisheye_mask(self, H, W, crop):
        y, x = torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing='ij'
        )
        center_y = (H - 1) / 2.0
        center_x = (W - 1) / 2.0
        a = (W - 1) / 2.0   # x 方向半轴
        b = (H - 1) / 2.0   # y 方向半轴
        a = max(a - crop, 0.0)
        b = max(b - crop, 0.0)
        ellipse_norm = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2
        fisheye_mask = ellipse_norm <= 1.0
        return fisheye_mask


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
        curr_distort_k = data_batch.get("curr_distort_k", None)
        curr_sensor_size = data_batch.get("curr_sensor_size", None)
        
        curr_rgb_fisheye_mask = None
        if self.is_fisheye_camera():
            h, w, _ = curr_rgb.shape
            curr_rgb_fisheye_mask = self.get_fisheye_mask(h, w, crop=self.fisheye_edge_pixel).numpy()
            _edge = self.fisheye_edge_value
            curr_rgb_fisheye_mask = curr_rgb_fisheye_mask | (curr_rgb[..., 0] > _edge) | (curr_rgb[..., 1] > _edge) | (curr_rgb[..., 1] > _edge)
            curr_rgb_fisheye_mask = curr_rgb_fisheye_mask.astype(np.float_)

        transform_info = transform_info if transform_info is not None else dict()
        transform_info["name"] = self.name
        transform_info["other_labels"] = [
            "sem", "eval_mask", "disp", "inf_mask", "invalid_mask", "fisheye_mask"
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
            other_labels=[curr_sem, curr_eval_mask, curr_disp, curr_inf_mask, curr_invalid_mask, curr_rgb_fisheye_mask],
            transform_info=transform_info,
        )

        # Process the semantic label and sky mask
        sem_label = other_labels[0]
        eval_mask = other_labels[1]
        disp = other_labels[2]
        inf_mask = other_labels[3]
        invalid_mask = other_labels[4]
        rgb_fisheye_mask = other_labels[5]

        disp, disp_valid = self.process_disp(disp, image, curr_rgb)

        normal_mask = None
        if normal is not None:
            normal_mask = torch.logical_or(
                torch.isfinite(normal).any(dim=0), (normal == 0).all(dim=0)
            )
            normal = torch.nan_to_num(normal, 0, 0, 0)
            if self.is_fisheye_camera() and self.fisheye_crop_pixel_normal > 0:
                _, H, W = normal.shape
                fisheye_mask = self.get_fisheye_mask(H, W, self.fisheye_crop_pixel_normal)
                normal_mask = torch.logical_and(normal_mask, fisheye_mask)
            normal[:, ~normal_mask] = 0.0
            normal[:, normal_mask] = torch.nn.functional.normalize(normal[:, normal_mask], dim=0)

        if self.is_fisheye_camera() and self.fisheye_crop_pixel_pointmap > 0:
            _, H, W = depth.shape
            fisheye_mask = self.get_fisheye_mask(H, W, self.fisheye_crop_pixel_pointmap)
            depth_mask = torch.logical_and(depth_mask, fisheye_mask)

        if eval_mask is not None:
            eval_mask = eval_mask.bool()

        if inf_mask is not None:
            inf_mask = inf_mask.bool()

        if invalid_mask is not None:
            invalid_mask = invalid_mask > 0
        
        if rgb_fisheye_mask is not None:
            rgb_fisheye_mask = rgb_fisheye_mask.bool()

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
                distort_k_opencv=curr_distort_k,
                sensor_size=curr_sensor_size,
                sparse_pointmap=sparse_pointmap,
                sparse_pointmap_mask=sparse_pointmap_mask,
                origin_hw=(curr_rgb.shape[0], curr_rgb.shape[1]),
            )
        else:
            # Process the depth data
            depth_output = self.process_depth(
                depth,
                depth_mask,
                intrinsics_mat,
                distort_k_opencv=curr_distort_k,
                sensor_size=curr_sensor_size,
                sparse_pointmap=None,
                sparse_pointmap_mask=None,
                origin_hw=(curr_rgb.shape[0], curr_rgb.shape[1]),
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
            "rgb_fisheye_mask": rgb_fisheye_mask,
            **depth_output,
        }

        if self.with_ray_dirs:
            image_h, image_w = image.shape[-2:]
            cam_ray_dirs = get_rays_in_camera_frame(
                intrinsics_mat, image_h, image_w, self.ray_dirs_normal_to_unit_sphere
            )
            data_dict["cam_ray_dirs"] = cam_ray_dirs

        # Apply flips to depth/mask/intrinsics for train mode (shared by depth_raw and pointmap_raw)
        if self.phase == "train" and curr_depth is not None and (self.with_depth_raw or self.with_pointmap_raw):
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

        # Add raw depth, mask, and intrinsics if available (now supports train with flip augmentation)
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
        crop_offset = None
        if self.pre_crop_edge is not None and self.camera_type in ["FISHEYE_BLENDER"]:
            scale_x = image.shape[2] / curr_rgb.shape[1]
            scale_y = image.shape[1] / curr_rgb.shape[0]
            crop_offset = (self.pre_crop_edge[0] * scale_x, self.pre_crop_edge[1] * scale_y)

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
            "camera_type": self.camera_type,
        }
        if curr_distort_k is not None:
            data_dict["meta_data"]["distort_k"] = torch.tensor(curr_distort_k, dtype=torch.float32)
        if curr_sensor_size is not None:
            data_dict["meta_data"]["sensor_size"] = torch.tensor(curr_sensor_size, dtype=torch.float32)
        if crop_offset is not None:
            data_dict["meta_data"]["crop_offset"] = torch.tensor(crop_offset, dtype=torch.float32)

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


    def load_data(self, data_info):
        curr_rgb = curr_depth = curr_depth_mask = None
        curr_sem = curr_normal = curr_disp = None
        curr_prompt = curr_eval_mask = curr_inf_mask = None
        curr_invalid_mask = None

        # 获取相机内参矩阵（Intrinsics）
        curr_intrinsics = data_info.get(self.intrinsics_name, None)
        # 获取相机外参矩阵（Extrinsics）
        curr_extrinsics = data_info.get(self.extrinsics_name, None)
        curr_distort_k = data_info.get(self.distorts_name, None)
        curr_sensor_size = data_info.get(self.sensor_size_name, None)
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
                
                if self.distorts_name is not None and self.distorts_name in f:
                    curr_distort_k = get_h5_data(f, self.distorts_name)

                if self.sensor_size_name is not None and self.sensor_size_name in f:
                    curr_sensor_size = get_h5_data(f, self.sensor_size_name)

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
            rgb_y, rgb_x = curr_rgb.shape[:2]
            if curr_intrinsics is not None:
                curr_intrinsics[2] -= crop_x
                curr_intrinsics[3] -= crop_y

            def crop_edge(data):
                if data is None:
                    return data
                data_y, data_x = data.shape[:2]
                
                if crop_y > 0:
                    _crop_y = int(data_y / rgb_y * crop_y)
                    data = data[_crop_y:-_crop_y]
                if crop_x > 0:
                    _crop_x = int(data_x / rgb_x * crop_x)
                    data = data[:, _crop_x:-_crop_x]
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
            curr_distort_k=curr_distort_k,
            curr_sensor_size=curr_sensor_size,
        )
        return data_batch


    def load_pointmap(self, depth, intrinsics, dist_coeffs=None, sensor_size=None, origin_hw=None):
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
        if dist_coeffs is not None:
            if isinstance(dist_coeffs, torch.Tensor):
                dist_coeffs = dist_coeffs.numpy()
            # Ensure shape (4,)
            dist_coeffs = np.array(dist_coeffs).flatten()
        height, width = depth.shape

        # Create a grid of pixel coordinates (u, v).
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5, indexing="xy")
        uv = np.stack([u, v], axis=-1)
        if self.camera_type in ["FISHEYE_EQUIDISTANT"]:
            # Extract intrinsic parameters
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            # Normalize pixel coordinates to camera plane
            x_cam = (u - cx) / fx
            y_cam = (v - cy) / fy
            # Compute radial distance from optical center
            r = np.sqrt(x_cam**2 + y_cam**2)
            # Equidistant fisheye model: direction vector derived from angle theta = r
            sin_r = np.sin(r)
            cos_r = np.cos(r)
            # Avoid division by zero at the image center
            mask_valid = r > 1e-8
            factor = np.where(mask_valid, sin_r / r, 1.0)
            # Initialize 3D unit direction vectors
            x_3d = x_cam * factor
            y_3d = y_cam * factor
            z_3d = cos_r
            # Normalize to ensure unit length (for numerical stability)
            norm = np.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
            x_3d /= norm
            y_3d /= norm
            z_3d /= norm
            # Scale by depth to get 3D points
            points = np.stack([
                x_3d * depth,
                y_3d * depth,
                z_3d * depth
            ], axis=0)  # Shape: (3, H, W)

        elif self.camera_type in ["FISHEYE_OPENCV"]:
            assert dist_coeffs is not None
            uv_flat = uv.reshape(-1, 1, 2)  # (HW, 1, 2) — required by cv2.fisheye.undistortPoints
            # === 鱼眼模型：使用 OpenCV fisheye 反投影 ===
            # undistortPoints returns normalized coordinates (x, y) = (X/Z, Y/Z)
            undistorted = cv2.fisheye.undistortPoints(uv_flat, intrinsics, dist_coeffs)  # (HW, 1, 2)
            if False:
                undist_hw = undistorted.reshape(height, width, 2)
                breakpoint()
                
            # undistorted = cv2.undistortPoints(uv_flat, intrinsics, dist_coeffs)
            
            xy_normalized = undistorted[:, 0, :]  # (HW, 2)

            # Reconstruct 3D direction: (x, y, 1) then normalize to unit vector
            directions = np.concatenate([xy_normalized, np.ones((xy_normalized.shape[0], 1))], axis=1)
            directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)  # (HW, 3)

            # Multiply the direction vectors by the depth values to get the 3D points.
            points = depth.reshape(-1, 1) * directions

            # Reshape the point cloud to match the original depth map dimensions and transpose axes.
            points = points.reshape(height, width, 3).transpose(2, 0, 1)
        elif self.camera_type in ["FISHEYE_BLENDER"]:
            assert dist_coeffs is not None
            assert sensor_size is not None, "sensor_size is required for FISHEYE_BLENDER"
            crop_offset = None
            if self.pre_crop_edge is not None and origin_hw is not None:
                origin_h, origin_w = origin_hw
                scale_x = width / origin_w
                scale_y = height / origin_h
                crop_offset = (self.pre_crop_edge[0] * scale_x, self.pre_crop_edge[1] * scale_y)
            return load_pointmap_blender_polynomial_fisheye(
                depth, intrinsics, k_coeffs=dist_coeffs, sensor_size=sensor_size,
                crop_offset=crop_offset,
            )
        else:
            # Convert the pixel coordinates to homogeneous coordinates.
            uv_homogeneous = np.concatenate([uv, np.ones((height, width, 1))], axis=-1).reshape(-1, 3)
            K_inv = np.linalg.inv(intrinsics)
            directions = uv_homogeneous @ K_inv.T  # (HW, 3)

            # Multiply the direction vectors by the depth values to get the 3D points.
            points = depth.reshape(-1, 1) * directions

            # Reshape the point cloud to match the original depth map dimensions and transpose axes.
            points = points.reshape(height, width, 3).transpose(2, 0, 1)

        # Clip the point cloud values to be within [-1.0, 1.0].
        # points = np.clip(points, a_min=-1.0, a_max=1.0)

        # Convert the point cloud back to a PyTorch tensor.
        points = torch.from_numpy(points).float()

        return points


    def process_depth(
        self,
        depth,
        depth_mask,
        intrinsics,
        distort_k_opencv=None,
        sensor_size=None,
        sparse_pointmap=None,
        sparse_pointmap_mask=None,
        origin_hw=None,
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

                # Convert the depth map to an inverse depth map and its corresponding mask.
                # inv_depth, inv_depth_mask = self.depth2invdepth(depth, depth_mask)

                # Normalize the inverse depth map using the same normalize_depth function.
                # if self.normalize_depth is not None:
                #     inv_depth_norm = self.normalize_depth(inv_depth, inv_depth_mask)

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
                pointmap = self.load_pointmap(depth, intrinsics, dist_coeffs=distort_k_opencv, sensor_size=sensor_size, origin_hw=origin_hw)

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

            # if sparse_dense_pointmap is not None and pointmap is not None:
            #     if pointmap.numel() == sparse_dense_pointmap.numel():
            #         sparse_dense_abs_diffmap = torch.abs(pointmap - sparse_dense_pointmap)

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


    def debug_data_dict(self, idx, data_dict, prefix=""):
        super().debug_data_dict(idx, data_dict, prefix)
        
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
        rgb_fisheye_mask = data_dict.get("rgb_fisheye_mask", None)
        
        if image is not None:
            if isinstance(image, torch.Tensor):
                image = image.numpy().transpose(1, 2, 0)
            save_path = os.path.join(outdir, f"{prefix}image.jpg")
            image = (image + 1.0) / 2.0 * 255.0
        
        if rgb_fisheye_mask is not None:
            if isinstance(rgb_fisheye_mask, torch.Tensor):
                rgb_fisheye_mask = rgb_fisheye_mask.squeeze(0).numpy()
            inf_mask_image = image.copy()
            inf_mask_image[rgb_fisheye_mask, :] *= 0.5
            inf_mask_image[rgb_fisheye_mask, 0] += 127
            save_path = os.path.join(outdir, f"{prefix}fisheye_mask.jpg")
            cv2.imwrite(save_path, inf_mask_image[:, :, ::-1].astype(np.uint8))
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
        with_pointmap=True,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=True,
        debug=True,
        # quantile_filter=True,
    )

    print(len(dataset))

    for index in range(min(10, len(dataset))):
        print(index)
        dataset.__getitem__(index)
