import os
import sys

sys.path.append(os.getcwd())

import json
import logging
import random
import traceback
from collections import defaultdict
from math import ceil

import cv2
import h5py
import numpy as np
import torch
from hAlgorithm.datasets.dataloader.collate import default_collate
from tqdm import tqdm

from hAlgorithm.datasets.base_dataset import BaseDataset
from hAlgorithm.datasets_mv.track.vggt_track import (
    build_tracks_by_depth,
    get_depth_inside_flag,
    visualize_tracks_neg_on_images,
    visualize_tracks_on_images,
)
from hAlgorithm.utils import grid_images, instantiate_from_config
from hAlgorithm.modules.utils.track_utils import sample_query_points_uvt
from hAlgorithm.datasets_mv.base_dataset import BaseDatasetMV
from hAlgorithm.datasets.transforms.transforms import resize_depth_preserve


class KosmoDatasetMV(BaseDatasetMV):
    def __init__(
        self,
        depth_confidence_name: str = None,
        depth_confidence_thresh: float = None,
        depth_confidence_ratio: float = None,
        reset_sparse_pattern=True,
        mask_mode:int=None,
        is_lidar=True,
        **kwargs,
    ):
        self.depth_confidence_name = depth_confidence_name if depth_confidence_name not in ["None", "none"] else None
        self.depth_confidence_thresh = depth_confidence_thresh
        self.depth_confidence_ratio = depth_confidence_ratio

        super(KosmoDatasetMV, self).__init__(**kwargs)

        self.reset_sparse_pattern = reset_sparse_pattern

        self.is_lidar = is_lidar
        if self.is_lidar:
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
        
        self.mask_mode = mask_mode
        if self.mask_mode is not None:
            if self.mask_mode == 20251231:
                self.camera_1_mask  = np.ones((1024, 1024))
                for i in range(510, 1024):
                    for j in range(1024-(i - 510 + 1), 1024):
                        self.camera_1_mask[i, j] = 0

                self.camera_3_mask = np.ones((1024, 1024))
                for i in range(510, 1024):
                    for j in range(0, i-510+1):
                        self.camera_3_mask[i, j] = 0

    def load_data(self, data_info):
        data_batch = super().load_data(data_info)

        curr_depth = data_batch["curr_depth"]
        curr_depth_mask = data_batch["curr_depth_mask"]

        curr_depth_conf_path = (
            os.path.join(self.depth_root, data_info[self.depth_confidence_name])
            if data_info.get(self.depth_confidence_name, None) is not None
            else None
        )
        if curr_depth_conf_path is not None and curr_depth is not None:
            curr_depth_conf = np.load(curr_depth_conf_path)

            if self.depth_confidence_thresh is not None:
                curr_depth_mask = curr_depth_mask & (curr_depth_conf > self.depth_confidence_thresh)
                curr_depth[~curr_depth_mask.astype(bool)] = 0
            elif self.depth_confidence_ratio is not None:
                curr_depth_mask = curr_depth_mask & (curr_depth_conf > np.quantile(curr_depth_conf, self.depth_confidence_ratio))
                curr_depth[~curr_depth_mask.astype(bool)] = 0
        
        if self.mask_mode is not None:
            if self.mask_mode == 20251231:
                if data_info["view_id"] == 1:
                    if self.camera_1_mask.shape != curr_depth.shape:
                        self.camera_1_mask = cv2.resize(self.camera_1_mask, (curr_depth.shape[1], curr_depth.shape[0])) == 1
                    curr_depth = curr_depth * self.camera_1_mask.astype(curr_depth.dtype)
                elif data_info["view_id"] == 3:
                    if self.camera_3_mask.shape != curr_depth.shape:
                        self.camera_3_mask = cv2.resize(self.camera_3_mask, (curr_depth.shape[1], curr_depth.shape[0])) == 1
                    curr_depth = curr_depth * self.camera_3_mask.astype(curr_depth.dtype)

        if curr_depth is not None and curr_depth.shape[:2] != data_batch["curr_rgb"].shape[:2]:
            curr_depth = resize_depth_preserve(curr_depth, data_batch["curr_rgb"].shape[:2])
            curr_depth_mask = curr_depth > 0

        data_batch["curr_depth"] = curr_depth
        data_batch["curr_depth_mask"] = curr_depth_mask

        return data_batch



if __name__ == "__main__":

    dataset = KosmoDatasetMV(
        phase="test",
        name="Kosmo",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMA_preprocess/Kosmo_Depth_GT/1_3/20260104_5_part2/0744_新华路建筑3/sv_da3b2_all_251207_bs4_500k_stage_kosmo_20260107-191247/data_hgaussian_v1.0.1_normal_mvfr_v251105_sfm_colmap_v1.4.0_mask_mos_v1.1.2_depth_mvfr_v251105@20260104210000.json",
        mf_to_mv=True,  # BaseDatasetMV
        clip_maxlen=1,  # BaseDatasetMV
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.debug_trajectory.DebugTrajectory",
            view_num=1,
        ),
        test_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=504,
                patch_size=14,
                is_lidar=True,
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
        with_pointmap=True,
        debug=True,
        sparse_pattern=None,
        normalize_cameras=True,
        
        mf_view_ids=[3],

        depth_confidence_name="pred_confidence_mvfr",
        depth_confidence_ratio=0.3,
    
        rbg_name="rgb",
        depth_name="pred_depth_mvfr",
        mask_name=None,
        sem_name=None,
        normal_name=None,
        lidar_name=None,
        eval_mask_name=None,
        disparity_name=None,

        prompt_extrinsics_name="slam_extrinsics",
        extrinsics_name="extrinsics",

        prompt_extrinsics_type="slam",
        extrinsics_type="colmap",

        mask_mode=20251231,
    )

    for index in range(min(1, len(dataset))):
        print(index)
        dataset.__getitem__(index)
