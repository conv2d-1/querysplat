import os
import sys
import json

sys.path.append(os.getcwd())

import logging

import torch
import numpy as np
import random

from hAlgorithm.datasets.transforms.edge_filter import edge_filter
from hAlgorithm.datasets.dataloader.collate import default_collate
from hAlgorithm.datasets_mv.base_dataset import BaseDatasetMV


class SparseDatasetMV(BaseDatasetMV):
    def __init__(self, sparse_gt_name=None, reproject_error_thresh=None, **kwargs):

        self.sparse_gt_name = sparse_gt_name
        self.depth_scale = 1

        self.reproject_error_thresh = reproject_error_thresh

        super().__init__(**kwargs)

    def load_data(self, data_info):
        curr_rgb_path = os.path.join(self.data_root, data_info[self.rbg_name])
        curr_sparse_gt_path = os.path.join(self.data_root, data_info[self.sparse_gt_name])

        # 加载RGB图像，形状为[h, w, 3]，即高度、宽度和RGB通道
        curr_rgb = self.read_image(curr_rgb_path).copy()
        curr_sparse_gt = np.load(curr_sparse_gt_path).astype(np.float64).copy()

        # 获取相机内参矩阵（Intrinsics）
        curr_intrinsics = data_info.get(self.intrinsics_name, None)
        # 获取相机外参矩阵（Extrinsics）
        curr_extrinsics = data_info.get(self.extrinsics_name, None)

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

        data_batch = dict(
            curr_intrinsics=curr_intrinsics,
            curr_extrinsics=curr_extrinsics,
            curr_rgb=curr_rgb,
            curr_sparse_gt=curr_sparse_gt,
        )
        return data_batch

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
                sparse_pointmap.numpy(),
                sparse_pointmap_mask.reshape(-1).numpy(),
            )
            sparse_pointmap_center = torch.from_numpy(sparse_pointmap_center)
            sparse_pointmap_max_range = torch.tensor(sparse_pointmap_max_range)
        elif self.point_normalize_mode == "distance":
            sparse_pointmap_max_range = np.max(
                np.linalg.norm(
                    sparse_pointmap.numpy(),
                    axis=1,
                )
            )
            sparse_pointmap_max_range = torch.tensor(sparse_pointmap_max_range).float()
        elif self.point_normalize_mode.startswith("distance_quantile"):
            sparse_pointmap_max_range = torch.quantile(
                torch.norm(
                    sparse_pointmap,
                    dim=1,
                ),
                float(self.point_normalize_mode.split("distance_quantile_")[-1]),
            ).float()
        elif self.point_normalize_mode == "distance_mean":
            sparse_pointmap_max_range = torch.mean(
                torch.norm(
                    sparse_pointmap,
                    dim=1,
                )
            ).float()
        elif self.point_normalize_mode.startswith("depth_quantile"):
            sparse_pointmap_max_range = torch.quantile(
                sparse_pointmap[:, -1],
                float(self.point_normalize_mode.split("depth_quantile_")[-1]),
            )

        if self.phase == "train" and self.max_scale_noise is not None and self.max_scale_noise > 0 and sparse_pointmap_max_range is not None:
            noise = 1.0 - self.max_scale_noise + 2 * self.max_scale_noise * torch.rand(1)[0]
            sparse_pointmap_max_range = sparse_pointmap_max_range * noise

        if sparse_pointmap_max_range is not None and min_range is not None and sparse_pointmap_max_range < min_range:
            raise Exception(f"sparse pointmap_max_range is too small, {sparse_pointmap_max_range}")

        return sparse_pointmap_center, sparse_pointmap_max_range

    def get_data_for_trainval(
        self,
        idx,
        data_info=None,
        data_batch=None,
        transform_info=None,
        mf_debug=False,
        trajectory_noise=None,
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
            curr_sparse_gt,
            curr_intrinsics,
            curr_extrinsics,
        ) = (
            data_batch["curr_rgb"],
            data_batch["curr_sparse_gt"],
            data_batch["curr_intrinsics"],
            data_batch["curr_extrinsics"],
        )

        transform_info = transform_info if transform_info is not None else dict()
        transform_info["name"] = self.name
        transform_info["other_labels"] = []

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
            depth=None,
            depth_mask=None,
            normal=None,
            other_labels=[],
            transform_info=transform_info,
        )

        # Create the intrinsics and extrinsics matrix from the parameters
        intrinsics_mat = self.create_intrinsics_matrix(intrinsics) if intrinsics is not None else None
        extrinsics_mat = self.create_extrinsics_matrix(curr_extrinsics) if curr_extrinsics is not None else None

        sparse_gt = torch.from_numpy(curr_sparse_gt.copy()).float()

        origin_height, origin_width = curr_rgb.shape[:2]
        sparse_valid = (
            (sparse_gt[:, 1] >= 0) & (sparse_gt[:, 1] < origin_width)
            & (sparse_gt[:, 2] >= 0) & (sparse_gt[:, 2] < origin_height)
        )

        if (
            self.reproject_error_thresh is not None
            and self.reproject_error_thresh > 0
            and curr_intrinsics is not None
            and curr_extrinsics is not None
        ):
            # 根据内外参将 sparse_gt[:, 6:9] 投影到像素坐标系，与 uv 进行对比，重投影误差大于阈值则过滤
            fx, fy, cx, cy = curr_intrinsics
            K = sparse_gt.new_tensor([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ])
            ext = sparse_gt.new_tensor(curr_extrinsics).reshape(4, 4)
            R_w2c = ext[:3, :3]
            T_w2c = ext[:3, 3]

            glb_points = sparse_gt[:, 6:9]
            cam_from_glb = glb_points @ R_w2c.T + T_w2c
            z_cam = cam_from_glb[:, 2]
            valid_z = z_cam > 1e-6

            proj = cam_from_glb @ K.T
            denom = proj[:, 2:3].clamp(min=1e-6)
            uv_proj = proj[:, :2] / denom

            uv_gt = sparse_gt[:, 1:3]
            reproj_error = torch.norm(uv_proj - uv_gt, dim=1)

            sparse_valid = sparse_valid & valid_z & (reproj_error <= self.reproject_error_thresh)

        # 如果有空，则报错（交由上层 __getitem__ 捕获后重采样）
        if int(sparse_valid.sum()) == 0:
            raise ValueError(
                f"sparse_gt is empty after filtering, "
                f"name={self.name}, idx={idx}, "
                f"reproject_error_thresh={self.reproject_error_thresh}"
            )

        sparse_gt = sparse_gt[sparse_valid]

        sparse_pointmap = sparse_gt[:, 3:6]
        # glb_sparse_pointmap = sparse_gt[:, 6:9]
        sparse_pointmap_mask = sparse_pointmap.new_ones(sparse_pointmap.shape[0], 1)
        _, sparse_pointmap_max_range = self.point_normalize(sparse_pointmap, sparse_pointmap_mask, min_range=0.1)

        # Construct the final data dictionary
        data_dict = {
            "image": image,
            "sparse_gt": sparse_gt,
            "intrinsics": intrinsics_mat,
            "extrinsics": extrinsics_mat,
            "sparse_pointmap_max_range": sparse_pointmap_max_range,
            # "sparse_pointmap": sparse_pointmap,
            # "glb_sparse_pointmap": glb_sparse_pointmap,
            # "sparse_pointmap_mask": sparse_pointmap_mask,
        }

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
                valid_mask = torch.ones_like(tmp_data).bool()
                edge_mask = edge_filter(tmp_data, valid_mask=valid_mask, times=0.3)

                if data_dict.get("edge_mask", None) is not None:
                    data_dict["edge_mask"] = edge_mask | data_dict["edge_mask"]
                else:
                    data_dict["edge_mask"] = edge_mask

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

        if "depth_scale" not in data_dict["meta_data"]["data_info"]:
            data_dict["meta_data"]["data_info"]["depth_scale"] = self.depth_scale

        if "rgb" not in data_dict["meta_data"]["data_info"]:
            if self.rbg_name in data_dict["meta_data"]["data_info"]:
                data_dict["meta_data"]["data_info"]["rgb"] = data_dict["meta_data"]["data_info"][self.rbg_name]
            else:
                data_dict["meta_data"]["data_info"]["rgb"] = data_dict["meta_data"]["data_info"]["hdf5"]

        # Remove entries with None values to clean up the dictionary
        data_dict = {k: v for k, v in data_dict.items() if v is not None}

        # Optionally debug the data dictionary
        if self.debug and not mf_debug:
            self.debug_data_dict(idx, data_dict)

        return data_dict
    
    def normalize_cameras_base_first(self, datas):
        """
        Normalize all camera extrinsics relative to the first camera.
        Args:
            datas (list): A list of dictionaries containing pointmap and extrinsics information.
        """

        if "extrinsics" not in datas[0]:
            return

        # extrinsics 为 world->camera (w2c)
        first_extrinsics = datas[0]["extrinsics"]
        first_extrinsics_inv = first_extrinsics.inverse()  # c2w of the first camera

        # 把世界坐标系下的稀疏点云变换到第一相机坐标系下: p_first = R_w2c1 @ p_world + t_w2c1
        R_first = first_extrinsics[:3, :3]
        T_first = first_extrinsics[:3, 3]

        sparse_pointmap_max_range = []
        for data in datas:
            # Compute new extrinsics relative to the first camera: w2c_new = w2c_curr @ c2w_first
            data["extrinsics_reff"] = data["extrinsics"] @ first_extrinsics_inv

            # sparse_gt[:, 6:9] 是世界坐标系下的稀疏点云, 形状 (N, 3)
            glb_sparse_pointmap = data["sparse_gt"][:, 6:9]
            data["sparse_gt"][:, 6:9] = glb_sparse_pointmap @ R_first.T + T_first

            if "sparse_pointmap_max_range" in data:
                sparse_pointmap_max_range.append(data["sparse_pointmap_max_range"])

        if self.with_global_scale:
            # 使用所有视角第一相机坐标系下的稀疏点云计算全局尺度
            points = torch.cat(
                [data["sparse_gt"][:, 6:9] for data in datas],
                dim=0,
            )
            global_scale = torch.mean(torch.norm(points, dim=-1))

            if (
                self.phase == "train"
                and self.max_scale_noise is not None
                and self.max_scale_noise > 0
            ):
                noise = 1.0 - self.max_scale_noise + 2 * self.max_scale_noise * torch.rand(1)[0]
                global_scale = global_scale * noise

            for data in datas:
                data["sparse_pointmap_max_range"] = global_scale

        elif self.with_global_max_scale:
            points = torch.cat(
                [data["sparse_gt"][:, 6:9] for data in datas],
                dim=0,
            )
            global_scale = torch.max(torch.norm(points, dim=-1))

            if (
                self.phase == "train"
                and self.max_scale_noise is not None
                and self.max_scale_noise > 0
            ):
                noise = 1.0 - self.max_scale_noise + 2 * self.max_scale_noise * torch.rand(1)[0]
                global_scale = global_scale * noise

            for data in datas:
                data["sparse_pointmap_max_range"] = global_scale

        elif self.with_max_mean_scale:
            if len(sparse_pointmap_max_range) > 0:
                sparse_pointmap_max_range = sum(sparse_pointmap_max_range) / len(
                    sparse_pointmap_max_range
                )
                for data in datas:
                    data["sparse_pointmap_max_range"] = sparse_pointmap_max_range

        elif self.with_pose_max_scale:
            c2w_translations = torch.stack([data["extrinsics_reff"].inverse()[:3, 3] for data in datas], dim=0)
            # Compute distance of all pose translations to origin
            norm_factor = c2w_translations.norm(dim=-1).max().clip(min=1e-8)
            for data in datas:
                data["sparse_pointmap_max_range"] = norm_factor

        else:
            if self.local_scale_mode in ['max']:
                if len(sparse_pointmap_max_range) > 0:
                    # Find the maximum range among all sparse pointmaps
                    sparse_pointmap_max_range = max(sparse_pointmap_max_range)
                    for data in datas:
                        data["sparse_pointmap_max_range"] = sparse_pointmap_max_range
            elif self.local_scale_mode in ['keep']:
                pass
            else:
                raise NotImplementedError(f"local scale mode {self.local_scale_mode} Not Supported!")
        
    @staticmethod
    def stack_dicts_list(list_of_dicts):
        """
        使用 default_collate 将一个只包含字典的列表堆叠成一个新的字典。

        参数:
        - list_of_dicts: 包含多个字典的列表，每个字典的键为字符串，值为张量。

        返回:
        - stacked_dict: 堆叠后的字典。
        """
        if not list_of_dicts:
            return {}

        # 检查所有字典的键是否相同
        keys = set(list_of_dicts[0].keys())
        for d in list_of_dicts:
            if set(d.keys()) != keys:
                raise ValueError("所有字典的键必须相同")

        # 收集每个键对应的值
        collected_values = {key: [] for key in keys}
        for d in list_of_dicts:
            for key, value in d.items():
                collected_values[key].append(value)

        # 使用 default_collate 处理每个键对应的值
        # stacked_dict = {key: default_collate(collected_values[key]) for key in keys}
        stacked_dict = dict()
        for key in keys:
            if key == "sparse_gt":
                # 每视角的 sparse_gt 形状为 (N_v, 9), N_v 跨视角不同。
                # 用 0 行填充对齐到 N_max (pid==0 在下游 get_sparse_gt 中被视为无效),
                # 再 stack 成 (S, N_max, 9), 便于 dataloader 之后跨 batch 拼成 list。
                values = collected_values[key]
                N_max = max((v.shape[0] for v in values), default=0)
                C = values[0].shape[-1] if values[0].ndim == 2 else values[0].shape[-1]

                if N_max == 0:
                    stacked_dict[key] = torch.stack(
                        [v.new_zeros(0, C) for v in values], dim=0
                    )
                    continue

                padded = []
                for v in values:
                    N_v = v.shape[0]
                    if N_v < N_max:
                        pad = v.new_zeros(N_max - N_v, C)
                        v = torch.cat([v, pad], dim=0)
                    padded.append(v)
                stacked_dict[key] = torch.stack(padded, dim=0)
                continue

            try:
                stacked_dict[key] = default_collate(collected_values[key])
            except Exception as e:
                if key == "image_raw":
                    stacked_dict[key] = default_collate([collected_values[key][0]])
                else:
                    logging.info(f"{key} is not collated, {e}. !!!!!!!!!!!!!!!!!!!!")

        return stacked_dict


    def get_mf_data_for_trainval(self, idx):
        if isinstance(idx, (list, tuple)):
            idx, others = idx

            view_num = others.get("view_num")
            aspect_ratio = others.get("aspect_ratio")
            max_size = others.get("max_size")

            if self.phase == "train" and max_size is not None:
                update_flag = False
                for transform in self.data_transforms.transforms:
                    if hasattr(transform, "update_max_size"):
                        transform.update_max_size(max_size=max_size)
                        update_flag = True
                assert update_flag

            if self.phase == "train" and aspect_ratio is not None:
                update_flag = False
                for transform in self.data_transforms.transforms:
                    if hasattr(transform, "update_aspect_ratio"):
                        transform.update_aspect_ratio(aspect_ratio=aspect_ratio)
                        update_flag = True
                assert update_flag

            if self.phase == "train" and view_num is not None:
                base_view_num = self.clip_sampler.view_num
                max_view_num = self.clip_sampler.get_max_view_num()
                if max_view_num is not None:
                    view_num = min(view_num, max_view_num)
                self.clip_sampler.view_num = view_num

            mf_data = self.load_mf_data(idx)

            if self.phase == "train" and view_num is not None:
                self.clip_sampler.view_num = base_view_num
        else:
            mf_data = self.load_mf_data(idx)

        assert mf_data is not None

        # mf data transforms
        transform_info = {}
        mf_data_batch_list = []
        image_show_list = []

        trajectory_noise = None
        for i, _ in enumerate(mf_data["data"]):
            data_batch = self.get_data_for_trainval(
                idx,
                data_info=mf_data["info"][i],
                data_batch=mf_data["data"][i],
                transform_info=transform_info,
                mf_debug=self.debug,
                trajectory_noise=trajectory_noise,
            )
            data_batch["meta_data"] = {
                "data_info": data_batch["meta_data"]["data_info"],
                "input_width": data_batch["meta_data"]["input_width"],
                "input_height": data_batch["meta_data"]["input_height"],
                "origin_width": data_batch["meta_data"]["origin_width"],
                "origin_height": data_batch["meta_data"]["origin_height"],
            }

            if self.track_points_nums > 0 or self.with_sift_mask:
                image_show_list.append(transform_info.pop("image_show"))

            if self.debug:
                print(
                    f'frame_id {data_batch["meta_data"]["data_info"]["frame_id"]},',
                    f'view_id {data_batch["meta_data"]["data_info"]["view_id"]},',
                    f'horizontal_flip {transform_info.get("horizontal_flip", None)},',
                    f'RGBCompresion {transform_info.get("RGBCompresion", None)},',
                    f'RandomBlur {transform_info.get("RandomBlur", None)},',
                )

            # Optionally debug the data dictionary
            mf_data_batch_list.append(data_batch)

        if self.normalize_cameras:
            self.normalize_cameras_base_first(mf_data_batch_list)

        view_ids = [view["view_id"] for view in mf_data["info"]]
        same_view_id = all([view_ids[0] == view_id for view_id in view_ids[1:]])

        mf_data_batch = self.stack_dicts_list(mf_data_batch_list)
        mf_data_batch["meta_data"].update(
            {
                "name": self.name,
                "data_path": self.data_path,
                "data_root": self.data_root,
                "data_idx": idx,
                "depth_scale": self.depth_scale,
                "views": 1 if same_view_id else len(view_ids),
                "frames": len(view_ids) if same_view_id else 1,
            }
        )
        mf_data_batch["meta_data"]["input_width"] = mf_data_batch["meta_data"]["input_width"][0]
        mf_data_batch["meta_data"]["input_height"] = mf_data_batch["meta_data"]["input_height"][0]
        mf_data_batch["meta_data"]["origin_width"] = mf_data_batch["meta_data"]["origin_width"][0]
        mf_data_batch["meta_data"]["origin_height"] = mf_data_batch["meta_data"]["origin_height"][0]

        if self.novel_view_nums > 0:
            mf_data_batch["meta_data"]["novel_view_nums"] = self.novel_view_nums

        if self.debug:
            frame_ids = [view["frame_id"] for view in mf_data["info"]]
            self.debug_mf_data_dict(idx, mf_data_batch_list, view_ids, frame_ids)

        for key, val in mf_data_batch.items():
            if isinstance(val, torch.Tensor):
                mf_data_batch[key] = val.contiguous()

        return mf_data_batch

    def debug_mf_data_dict(self, idx, data_batch_list, view_ids, frame_ids):
        """
        Override to additionally visualize ``sparse_gt`` per view and across views.

        sparse_gt 列定义::
            [:, 0]   point3D_id
            [:, 1:3] u, v   (原始 RGB 图像空间)
            [:, 3:6] local xyz (当前相机坐标系)
            [:, 6:9] global xyz (世界坐标系；经 ``normalize_cameras_base_first`` 后为第一相机坐标系)

        生成内容::
            {prefix}sparse_gt_uv.jpg              -- 横向三联: uv_gt | cam_proj | glb_proj
            {prefix}sparse_gt_cam.ply             -- 当前相机坐标系下的稀疏点 (RGB 着色)
            {prefix}sparse_gt_glb.ply             -- 世界/第一相机坐标系下的稀疏点 (RGB 着色)
            {idx:04d}_merge_sparse_gt_glb.ply     -- 所有视角合并后的稀疏点 (RGB 着色)
        """

        super().debug_mf_data_dict(idx, data_batch_list, view_ids, frame_ids)

        import cv2
        import open3d as o3d

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

        def _draw_points(img_rgb, uv, color, radius=2):
            """uv: (M, 2) float, 越界点会被自动跳过. img_rgb: HxWx3 uint8 (RGB)."""
            H, W = img_rgb.shape[:2]
            canvas = np.ascontiguousarray(img_rgb.copy())
            if uv is None or len(uv) == 0:
                return canvas
            uv_int = np.round(uv).astype(np.int32)
            mask = (
                (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) &
                (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
            )
            for u, v in uv_int[mask]:
                cv2.circle(canvas, (int(u), int(v)), radius, color, -1)
            return canvas

        def _sample_rgb(img_rgb, uv):
            """在 image (HxWx3 uint8 RGB) 上的 uv 处采样颜色, 返回 (N, 3) float in [0, 1].

            越界点的颜色返回 [0, 0, 0]."""
            H, W = img_rgb.shape[:2]
            colors = np.zeros((uv.shape[0], 3), dtype=np.float64)
            uv_int = np.round(uv).astype(np.int64)
            mask = (
                (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) &
                (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
            )
            if mask.any():
                colors[mask] = img_rgb[uv_int[mask, 1], uv_int[mask, 0]] / 255.0
            return colors

        all_glb_points, all_glb_rgb = [], []

        for i, data_batch in enumerate(data_batch_list):
            view_id = view_ids[i]
            frame_id = frame_ids[i]
            prefix = f"view{view_id:02d}_frame{frame_id:02d}_"

            sparse_gt = data_batch.get("sparse_gt", None)
            if sparse_gt is None:
                continue
            if isinstance(sparse_gt, torch.Tensor):
                sparse_gt_np = sparse_gt.detach().cpu().numpy()
            else:
                sparse_gt_np = np.asarray(sparse_gt)
            if sparse_gt_np.ndim != 2 or sparse_gt_np.shape[0] == 0 or sparse_gt_np.shape[1] < 9:
                continue

            uv_gt_orig = sparse_gt_np[:, 1:3].astype(np.float64)
            cam_points = sparse_gt_np[:, 3:6].astype(np.float64)
            glb_points = sparse_gt_np[:, 6:9].astype(np.float64)

            image = data_batch.get("image", None)
            intrinsics_mat = data_batch.get("intrinsics", None)
            if image is None or intrinsics_mat is None:
                continue

            if isinstance(image, torch.Tensor):
                image_np = image.detach().cpu().numpy().transpose(1, 2, 0)
            else:
                image_np = np.asarray(image)
            image_np = ((image_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

            if isinstance(intrinsics_mat, torch.Tensor):
                K = intrinsics_mat.detach().cpu().numpy()
            else:
                K = np.asarray(intrinsics_mat)
            K = K[:3, :3].astype(np.float64)

            H, W = image_np.shape[:2]

            # uv_gt 在原图坐标系, 缩放到 resize 后的 image 空间
            meta = data_batch.get("meta_data", {}) or {}
            origin_w = float(meta.get("origin_width", W))
            origin_h = float(meta.get("origin_height", H))
            scale_w = W / max(origin_w, 1.0)
            scale_h = H / max(origin_h, 1.0)
            uv_gt = uv_gt_orig * np.array([scale_w, scale_h])

            # cam 点投影 uv = K @ X_cam / Z_cam
            uv_cam = np.full_like(uv_gt, fill_value=-1.0)
            z_cam = cam_points[:, 2]
            valid_cam = z_cam > 1e-6
            if valid_cam.any():
                proj_cam = (K @ cam_points[valid_cam].T).T
                uv_cam[valid_cam] = proj_cam[:, :2] / proj_cam[:, 2:3]

            # glb 点先用外参变换到当前相机坐标系再投影
            ext = data_batch.get("extrinsics_reff", data_batch.get("extrinsics", None))
            uv_glb = np.full_like(uv_gt, fill_value=-1.0)
            if ext is not None:
                if isinstance(ext, torch.Tensor):
                    ext_np = ext.detach().cpu().numpy()
                else:
                    ext_np = np.asarray(ext)
                R_w2c = ext_np[:3, :3].astype(np.float64)
                T_w2c = ext_np[:3, 3].astype(np.float64)
                cam_from_glb = (R_w2c @ glb_points.T).T + T_w2c
                z_glb = cam_from_glb[:, 2]
                valid_glb = z_glb > 1e-6
                if valid_glb.any():
                    proj_glb = (K @ cam_from_glb[valid_glb].T).T
                    uv_glb[valid_glb] = proj_glb[:, :2] / proj_glb[:, 2:3]

            sub_gt = _draw_points(image_np, uv_gt, color=(0, 255, 0))
            sub_cam = _draw_points(image_np, uv_cam, color=(0, 255, 255))
            sub_glb = _draw_points(image_np, uv_glb, color=(255, 0, 0))
            for sub, label in [(sub_gt, "uv_gt"), (sub_cam, "cam_proj"), (sub_glb, "glb_proj")]:
                cv2.putText(sub, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)
            merged_uv = np.concatenate([sub_gt, sub_cam, sub_glb], axis=1)
            cv2.imwrite(
                os.path.join(outdir, f"{prefix}sparse_gt_uv.jpg"),
                merged_uv[:, :, ::-1],
            )

            # 在 uv_gt 处采样 image 的 RGB 颜色作为点云着色
            rgb_colors = _sample_rgb(image_np, uv_gt)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cam_points)
            pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
            o3d.io.write_point_cloud(
                os.path.join(outdir, f"{prefix}sparse_gt_cam.ply"), pcd
            )

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(glb_points)
            pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
            o3d.io.write_point_cloud(
                os.path.join(outdir, f"{prefix}sparse_gt_glb.ply"), pcd
            )

            all_glb_points.append(glb_points)
            all_glb_rgb.append(rgb_colors)

        if len(all_glb_points) > 0:
            merged_points = np.concatenate(all_glb_points, axis=0)
            merged_rgb = np.concatenate(all_glb_rgb, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged_points)
            pcd.colors = o3d.utility.Vector3dVector(merged_rgb)
            save_path = os.path.join(outdir, f"{idx:04d}_merge_sparse_gt_glb.ply")
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"merge sparse_gt {save_path}")


if __name__ == "__main__":

    import yaml

    from hAlgorithm.utils import instantiate_from_config

    path = "hAlgorithm/configs/mv_v1.0/dataset_configs_5/train_260301_mv_sparse.yaml"

    with open(path, "r") as file:
        yaml_data = yaml.safe_load(file)
    dataset_configs = yaml_data.get("datasets", [])

    for config in dataset_configs:
        base_config = dict(
            phase="test",
            seed=0,
            test_transforms=[
                dict(
                    type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                    max_size=840,
                    patch_size=14,
                    # patch_size=32,
                    # is_lidar=True,
                    backup=True,
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
            normalize_cameras=True,
            with_global_scale=True,
            with_rgb_edge_mask=True,
            debug=True,
        )
        config.update(base_config)
        # config["clip_sampler"]["shuffle"] = False

        dataset = instantiate_from_config(config)
        print(config["name"], len(dataset))

        indices = random.sample(range(len(dataset)), k=min(5, len(dataset)))
        indices = [0, 1, 2]
        # indices = random.sample(range(len(dataset)), k=min(2, len(dataset)))
        # # indices = [0, 1, 2]

        for index in indices:
            print(index)
            dataset.__getitem__(index)
