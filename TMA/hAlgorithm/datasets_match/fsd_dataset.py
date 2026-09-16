import os
import sys

sys.path.append(os.getcwd())

import json
import logging
import random
import time
from collections import defaultdict
from math import ceil

import imageio
import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from hAlgorithm.datasets4d.transforms.normalize_cameras import normalize_cameras
from hAlgorithm.datasets_match.base_dataset import MatchingDataset
from hAlgorithm.utils import grid_images, instantiate_from_config


def disparity_to_depth(disparity, intrinsic, baseline):
    """
    将视差图转换为深度图
    Args:
        disparity: 视差图（单位：像素）
        focal_length: 相机焦距（单位：像素）
        baseline: 双目相机基线长度（单位：米）
    Returns:
        depth: 深度图（单位：米）
    """
    # 避免除零错误
    disparity = np.maximum(disparity, 0.001)
    # breakpoint()
    # 深度 = 焦距 * 基线 / 视差
    depth = (intrinsic[0] * baseline) / disparity
    return depth


class FSDDatasetMV(MatchingDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_disparity(self, disp_path, image, const, dtype):
        disp = imageio.imread(disp_path)
        disp = disp.astype(float)
        disp = disp[..., 0] * 255 * 255 + disp[..., 1] * 255 + disp[..., 2]
        disp = disp / 1000.0
        return disp

    def load_mf_data(self, idx):
        seq_list = self.clip_sampler(self.data_infos[idx])
        assert len(seq_list) == 2, "FSD Dataset Only supports view_num=2"
        mf_data, mf_info = [], []

        left_view_info = {}
        for view_info in seq_list:
            if view_info.get("depth", None) is not None:
                left_view_info["depth"] = view_info["depth"]
                view_info["main_view"] = True
            if view_info.get("disparity", None) is not None:
                left_view_info["disparity"] = view_info["disparity"]
                view_info["main_view"] = True

        main_views = []
        other_views = []
        for view in seq_list:
            view.update(left_view_info)
            view["extrinsics"] = np.eye(4)
            if "main_view" not in view:
                view["main_view"] = False
                other_views.append(view)
            else:
                main_views.append(view)

        seq_list = []
        seq_list.extend(main_views)
        seq_list.extend(other_views)

        for view in seq_list:
            single_frame_data = self.load_data(view)
            mf_data.append(single_frame_data)
            mf_info.append(view)

        batch = dict(
            data=mf_data,
            info=mf_info,
        )

        return batch


class FSDDatasetMVRandomBaseline(MatchingDataset):
    def __init__(self, baseline_range=[0.09, 0.11], **kwargs):
        super().__init__(**kwargs)
        self.baseline_range = baseline_range

    def load_disparity(self, disp_path, image, const, dtype):
        disp = imageio.imread(disp_path)
        disp = disp.astype(float)
        disp = disp[..., 0] * 255 * 255 + disp[..., 1] * 255 + disp[..., 2]
        disp = disp / 1000.0
        return disp

    def process_stereo_images(self, left_img, right_img, disparity, intrinsic_left_param, intrinsic_right_param, baseline):
        height, width = left_img.shape[:2]

        # 缩放内参矩阵
        intrinsic_left = intrinsic_left_param.copy()
        intrinsic_right = intrinsic_right_param.copy()

        # 根据图像尺寸缩放内参矩阵
        scale_x = width / (intrinsic_left[2] * 2)
        scale_y = height / (intrinsic_left[3] * 2)

        # 缩放左相机内参
        intrinsic_left[0] *= scale_x  # fx
        intrinsic_left[1] *= scale_y  # fy
        intrinsic_left[2] *= scale_x  # cx
        intrinsic_left[3] *= scale_y  # cy

        # 缩放右相机内参
        intrinsic_right[0] *= scale_x  # fx
        intrinsic_right[1] *= scale_y  # fy
        intrinsic_right[2] *= scale_x  # cx
        intrinsic_right[3] *= scale_y  # cy

        # 将视差转换为左相机深度
        depth_left = disparity_to_depth(disparity, intrinsic_left, baseline)

        # 设置左相机外参为单位阵
        extrinsic_left = np.eye(4)

        # 设置右相机外参 - 相对于左相机平移baseline
        extrinsic_right = np.eye(4)
        extrinsic_right[0, 3] = -baseline  # 在x方向上平移baseline

        # 根据左相机深度和右相机外参计算右相机深度
        # 1. 将左相机深度点投影到世界坐标系
        y, x = np.mgrid[0:height, 0:width]
        X = (x - intrinsic_left[2]) * depth_left / intrinsic_left[0]
        Y = (y - intrinsic_left[3]) * depth_left / intrinsic_left[1]
        Z = depth_left

        # 2. 将世界坐标点转换到右相机坐标系
        points_world = np.stack([X, Y, Z], axis=-1)
        points_right = np.dot(points_world, extrinsic_right[:3, :3].T) + extrinsic_right[:3, 3]

        # 3. 将右相机坐标系中的点投影到右相机图像平面
        X_right = points_right[..., 0]
        Y_right = points_right[..., 1]
        Z_right = points_right[..., 2]

        # 计算右相机图像平面上的像素坐标
        u_right = X_right * intrinsic_right[0] / Z_right + intrinsic_right[2]
        v_right = Y_right * intrinsic_right[1] / Z_right + intrinsic_right[3]

        # 创建右相机深度图
        depth_right = np.zeros_like(depth_left)

        # 将有效的投影点填充到右相机深度图中
        valid_mask = (u_right >= 0) & (u_right < width) & (v_right >= 0) & (v_right < height) & (Z_right > 0)
        u_right_valid = u_right[valid_mask].astype(int)
        v_right_valid = v_right[valid_mask].astype(int)
        Z_right_valid = Z_right[valid_mask]

        # 使用最近邻插值填充深度值
        depth_right[v_right_valid, u_right_valid] = Z_right_valid

        return (
            depth_left,
            depth_right,
            intrinsic_left,
            intrinsic_right,
            extrinsic_left,
            extrinsic_right,
        )

    def load_mf_data(self, idx):
        seq_list = self.clip_sampler(self.data_infos[idx])
        assert len(seq_list) == 2, "FSD Dataset Only supports view_num=2"

        left_view_info, right_view_info = {}, {}
        for view_info in seq_list:
            if view_info.get("disparity", None) is not None:
                view_info["main_view"] = True
                left_view_info = view_info
            else:
                view_info["main_view"] = False
                right_view_info = view_info

        right_view_info["disparity"] = left_view_info["disparity"]
        left_view = self.load_data(left_view_info)
        right_view = self.load_data(right_view_info)

        left_rgb, left_intrinsics = left_view["curr_rgb"], left_view["curr_intrinsics"]
        right_rgb, right_intrinsics = right_view["curr_rgb"], right_view["curr_intrinsics"]
        disparity = left_view["curr_disp"]

        baseline = random.uniform(self.baseline_range[0], self.baseline_range[1])  # unit: meter

        (
            depth_left,
            depth_right,
            left_intrinsics,
            right_intrinsics,
            extrinsic_left_param,
            extrinsic_right_param,
        ) = self.process_stereo_images(left_rgb, right_rgb, disparity, left_intrinsics, right_intrinsics, baseline)

        right_view["curr_depth"] = depth_right
        right_view["curr_intrinsics"] = right_intrinsics
        right_view["curr_extrinsics"] = extrinsic_right_param

        left_view["curr_depth"] = depth_left
        left_view["curr_intrinsics"] = left_intrinsics
        left_view["curr_extrinsics"] = extrinsic_left_param

        # dummy info
        right_view_info["depth"] = ""
        right_view_info["disparity"] = ""

        mf_data, mf_info = [left_view, right_view], [left_view_info, right_view_info]

        batch = dict(
            data=mf_data,
            info=mf_info,
        )

        return batch


if __name__ == "__main__":

    dataset = FSDDatasetMV(
        phase="test",
        name="FSD",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/FSD/test_mf.json",
        # mf_view_ids=[0, 1, 2, 3],
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=518,
                patch_size=14,
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
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=518,
                patch_size=14,
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
        sparse_pattern=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=0,
            sparse_ratio=0.05,
        ),
        mf_to_sf=False,
        mf_to_mv=False,
        clip_shuffle=True,
        normalize_cameras=False,
        normalize_cameras_v3=True,
        sampling_strategy="first:10",
        mf_scene_sampling_strategy="all",
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=2,
            seed=None,
        ),
    )

    for index in range(min(5, len(dataset))):
        print(index)
        dataset.__getitem__(index)
