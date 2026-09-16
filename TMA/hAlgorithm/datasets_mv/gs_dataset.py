import os
import sys

sys.path.append(os.getcwd())

import json
import logging
import traceback

import cv2
import numpy as np
import torch
from tqdm import tqdm

from hAlgorithm.datasets_mv.base_dataset import BaseDatasetMV
from hAlgorithm.utils import colorize_depth_maps
from hAlgorithm.datasets.transforms.transforms import resize_depth_preserve


class GSDataset(BaseDatasetMV):
    """
    GSDataset: A dataset class designed for Gaussian Splatting (GS) based 3D reconstruction tasks.

    This class inherits from BaseDataset4D and provides specialized functionality for handling multi-view,
    multi-frame data with support for camera normalization, sparse depth simulation, and NeRF++-style normalization.
    """

    def __init__(
        self,
        depth_name="null",
        mask_name="null",
        normal_name="null",
        confidence_name="null",
        confidence_thresh=None,
        ignore_frame_ids=None,
        ignore_view_ids=None,
        with_depth=False,
        with_points=True,
        object_mask_root=None,
        object_mask_pattern=None,
        **kwargs,
    ):
        ignore_frame_ids = ignore_frame_ids or []
        self.ignore_view_ids = ignore_view_ids or []

        self.lidar_name = "null"
        self.sem_name = "null"
        self.eval_mask_name = "null"
        self.disparity_name = "null"
        self.confidence_name = confidence_name
        self.confidence_thresh = confidence_thresh

        self.object_mask_root = object_mask_root
        self.object_mask_pattern = object_mask_pattern

        self.sparse_depth_ratio = 0
        self.sparse_depth_nums = 0
        self.sparse_pattern = None
        self.sparse_pattern_v2 = None
        self.sparse_pattern_v3 = None

        self.translate = None
        self.radius = 20

        self.with_depth = with_depth
        self.with_points = with_points and with_depth

        clip_sampler = dict(
            type="hAlgorithm.datasets_mv.clip_sampler.debug_trajectory.DebugTrajectory",
            view_num=1,
        )

        super(GSDataset, self).__init__(
            depth_name=depth_name,
            mask_name=mask_name,
            normal_name=normal_name,
            ignore_frame_ids=ignore_frame_ids,
            clip_sampler=clip_sampler,
            **kwargs,
        )
    
    def get_object_mask(self, data_info):
        mask = None

        rgb_path = os.path.join(self.data_root, data_info[self.rbg_name])
        frame_id = data_info["frame_id"]
        view_id = data_info["view_id"]
        if self.object_mask_pattern is not None:
            if isinstance(self.object_mask_pattern[0], (list, tuple)):
                mask_path = rgb_path
                for mask_pattern in self.object_mask_pattern:
                    mask_path = mask_path.replace(mask_pattern[0], mask_pattern[1])
            else:
                mask_path = rgb_path.replace(self.object_mask_pattern[0], self.object_mask_pattern[1])

            if self.data_root is not None:
                mask_path = os.path.join(self.data_root, mask_path)
            mask_path = os.path.join(os.path.dirname(mask_path), "data.json")

            with open(mask_path, "r") as f:
                mask = json.load(f)[f"frame_id_{frame_id}"][f"view_id_{view_id}"]["rgb_mask"]

            if self.mask_root is not None:
                mask = os.path.join(self.mask_root, mask)

            mask = (cv2.imread(mask, cv2.IMREAD_GRAYSCALE) > 127).astype(int)

        return mask
    
    def load_data(self, data_info):
        data_batch = super().load_data(data_info=data_info)

        if self.confidence_name in data_info:
            curr_confidence_path = os.path.join(self.depth_root, data_info[self.confidence_name])
            confidence = self.read_file(curr_confidence_path, image=None, const=0, dtype=np.float32)
            curr_depth_mask = data_batch["curr_depth_mask"]

            if confidence.shape[0:2] != curr_depth_mask.shape[0:2]:
                confidence = resize_depth_preserve(confidence, curr_depth_mask.shape[0:2])

            confidence_mask = confidence >= self.confidence_thresh
            curr_depth_mask = (curr_depth_mask * confidence_mask).astype(int)
            data_batch["curr_depth_mask"] = curr_depth_mask
        
        object_mask = self.get_object_mask(data_info)
        data_batch["curr_object_mask"] = object_mask

        return data_batch

    def get_data_infos(self):
        self.data_infos = self.load_json_file(self.data_path)
        self.gs_total_datas = []
        for data_infos in tqdm(self.data_infos, total=len(self.data_infos), desc="GS Loading"):
            try:
                data_batch = self.load_data(data_infos)
                data_dict = self.get_data_for_trainval(
                    idx=len(self.gs_total_datas), data_info=data_infos, data_batch=data_batch
                )
                new_data_dict = {}
                for key, val in data_dict.items():
                    if key in ["image", "intrinsics", "extrinsics", "object_mask"]:
                        new_data_dict[key] = val
                    elif self.with_depth and key in ["depth", "depth_mask"]:
                        new_data_dict[key] = val
                    elif self.with_points  and key in ["pointmap"]:
                        new_data_dict[key] = val
                    elif key == "meta_data":
                        new_data_dict["meta_data"] = val
                        new_data_dict["meta_data"]["scene"] = data_infos["scene"]
                        new_data_dict["meta_data"]["frame_id"] = data_infos["frame_id"]
                        new_data_dict["meta_data"]["view_id"] = data_infos["view_id"]
                self.gs_total_datas.append(new_data_dict)
            except Exception as e:
                traceback.print_exc()
                logging.error(e)

        if len(self.gs_total_datas) > 0:
            self.getNerfppNorm()
            logging.info(f"translate, radius: {self.translate, self.radius}")

            for data_dict in self.gs_total_datas:
                data_dict["meta_data"]["translate"] = self.translate
                data_dict["meta_data"]["radius"] = self.radius

        if self.normalize_cameras:
            self.normalize_cameras_base_first(self.gs_total_datas)

        # 训练时 meta_json_split = trainval，包含第一帧
        # 完成 normalize_cameras 后，需要根据 phase 进行过滤
        if self.trainval_meta is not None:
            if self.phase == "vis" and "vis" not in self.trainval_meta:
                meta_json_split = "val"
            else:
                meta_json_split = self.phase

            if meta_json_split != self.meta_json_split:
                scene = self.data_infos[0]["scene"]
                trainval = self.trainval_meta[scene][meta_json_split]
                cur_frame_ids = trainval["frame_ids"]
                cur_view_ids = trainval["view_ids"]
                trainval = set(
                    [(cur_frame_ids[i], cur_view_ids[i]) for i in range(len(cur_frame_ids))]
                )

                fix_gs_total_datas = []
                delete_list = []
                for data_infos in self.gs_total_datas:
                    frame_id, view_id = (
                        data_infos["meta_data"]["frame_id"],
                        data_infos["meta_data"]["view_id"],
                    )
                    if (frame_id, view_id) not in trainval:
                        delete_list.append((frame_id, view_id))
                        continue
                    fix_gs_total_datas.append(data_infos)
                self.gs_total_datas = fix_gs_total_datas
                logging.info(
                    f"fix meta_json_split: {meta_json_split}, datas: {len(self.gs_total_datas)}!!"
                )
                logging.debug(f"delete_list: {delete_list}")

    def normalize_cameras_base_first(self, datas):
        # Get the inverse of the first camera's extrinsics
        first_extrinsics_inv = datas[0]["extrinsics"].inverse()

        for data in datas:
            # Compute new extrinsics relative to the first camera
            data["extrinsics_reff"] = data["extrinsics"] @ first_extrinsics_inv

    def choose_frame_and_view_ids(self, total_datas):
        if len(self.ignore_frame_ids) > 0 or len(self.ignore_view_ids) > 0:
            new_total_datas = []
            for data in total_datas:
                frame_id = data["frame_id"]
                view_id = data["view_id"]

                if frame_id not in self.ignore_frame_ids and view_id not in self.ignore_view_ids:
                    new_total_datas.append(data)

            return new_total_datas
        else:
            return total_datas

    def load_mf_files(self, datas):
        total_datas = super().load_mf_files(datas)
        if isinstance(total_datas[0], (list, tuple)):
            total_datas = [data[0] for data in total_datas]
        return self.choose_frame_and_view_ids(total_datas)

    def load_hdf5_mf_files(self, datas):
        total_datas = super().load_hdf5_mf_files(datas)
        if isinstance(total_datas[0], (list, tuple)):
            total_datas = [data[0] for data in total_datas]
        return self.choose_frame_and_view_ids(total_datas)

    def get_data_for_trainval(
        self, idx, data_info, data_batch, transform_info=None, mf_debug=False
    ):
        if self.debug:
            print("rgb", os.path.join(self.data_root, data_info.get("rgb", "None")))

        (
            curr_rgb,
            curr_intrinsics,
            curr_extrinsics,
            curr_depth,
            curr_depth_mask,
            curr_normal,
            curr_object_mask,
        ) = (
            data_batch["curr_rgb"],
            data_batch["curr_intrinsics"],
            data_batch["curr_extrinsics"],
            data_batch["curr_depth"],
            data_batch["curr_depth_mask"],
            data_batch["curr_normal"],
            data_batch["curr_object_mask"],
        )
        curr_depth, curr_depth_mask = self.depth_filter(
            curr_depth, depth_mask=curr_depth_mask, sem_label=None
        )

        if curr_depth is not None and curr_depth_mask.sum() == 0:
            self.info_rbg_path(data_info)
            logging.warning("depth_mask is empty, no valid points found.")
            raise ValueError("depth_mask is empty, no valid points found.")

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
            intrinsics=curr_intrinsics.copy(),  # NOTE
            depth=curr_depth,
            depth_mask=curr_depth_mask,
            normal=curr_normal,
            other_labels=[curr_object_mask],
            transform_info=transform_info if transform_info is not None else dict(),
        )

        # RGB object mask
        object_mask = other_labels[0]
        if object_mask is not None:
            object_mask = object_mask.bool()

        # Create the intrinsics and extrinsics matrix from the parameters
        intrinsics_mat = (
            self.create_intrinsics_matrix(intrinsics) if intrinsics is not None else None
        )
        extrinsics_mat = (
            self.create_extrinsics_matrix(curr_extrinsics) if curr_extrinsics is not None else None
        )

        if self.with_depth and self.with_points:
            pointmap = self.load_pointmap(depth, intrinsics_mat)
        else:
            pointmap = None

        # Construct the final data dictionary
        data_dict = {
            "image": image,
            "intrinsics": intrinsics_mat,
            "extrinsics": extrinsics_mat,
            "depth": depth,
            "pointmap": pointmap,
            "depth_mask": depth_mask,
            "normal": normal,
            "object_mask": object_mask,
        }

        # Add metadata
        data_dict["meta_data"] = {
            "name": self.name,
            "data_path": self.data_path,
            "data_idx": idx,
            "data_info": {k: v for k, v in data_info.items() if k in [self.rbg_name]},
            "input_width": image.shape[2],
            "input_height": image.shape[1],
            "origin_width": curr_rgb.shape[1],
            "origin_height": curr_rgb.shape[0],
            "frames": 1,
            "views": 1,
        }

        # Remove entries with None values to clean up the dictionary
        data_dict = {k: v for k, v in data_dict.items() if v is not None}

        # Optionally debug the data dictionary
        if self.debug and not mf_debug:
            self.debug_data_dict(idx, data_dict)

        return data_dict

    def get_mf_data_for_trainval(self, idx):
        if isinstance(idx, (list, tuple)):
            idx, others = idx
        data = self.gs_total_datas[idx]

        new_data = dict()
        for key in data:
            if key != "meta_data":
                new_data[key] = data[key].unsqueeze(0)
            else:
                new_data[key] = data[key]

        return new_data

    def getNerfppNorm(self):

        def getWorld2View2(Rt, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
            C2W = np.linalg.inv(Rt)
            cam_center = C2W[:3, 3]
            cam_center = (cam_center + translate) * scale
            C2W[:3, 3] = cam_center
            Rt = np.linalg.inv(C2W)
            return np.float32(Rt)

        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_centers = []

        for cam in self.gs_total_datas:
            W2C = getWorld2View2(cam["extrinsics"])
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1

        translate = -center

        self.translate = translate
        self.radius = radius

    def __len__(self):
        return len(self.gs_total_datas)

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
        # depth_mask = data_dict.get("depth_mask", None)
        normal = data_dict.get("normal", None)

        if image is not None:
            if isinstance(image, torch.Tensor):
                image = image.numpy().transpose(1, 2, 0)
            save_path = os.path.join(outdir, f"{prefix}image.jpg")
            cv2.imwrite(save_path, image[:, :, ::-1].astype(np.uint8))

        if depth is not None:
            if isinstance(depth, torch.Tensor):
                depth = depth.squeeze(0).numpy()
            depth_colored = colorize_depth_maps(
                depth,
                depth.min(),
                depth.max(),
                cmap="turbo",
            )
            save_path = os.path.join(outdir, f"{prefix}depth.jpg")
            depth_colored.save(save_path)

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

        if extrinsics is not None:
            extrinsics = extrinsics.cpu().squeeze(0).numpy()
            save_path = os.path.join(outdir, f"{prefix}extrinsics.json")
            with open(save_path, "w") as f:
                json.dump(extrinsics.tolist(), f, indent=2)

        if normal is not None:
            if isinstance(normal, torch.Tensor):
                normal = normal.numpy().transpose(1, 2, 0)
            normal = ((normal + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
            save_path = os.path.join(outdir, f"{prefix}normal.jpg")
            normal = cv2.imwrite(save_path, normal)

        return outdir


if __name__ == "__main__":

    dataset = GSDataset(
        phase="train",
        name="iphone",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/Zed2FT/data/20250508_davinci/gs_demo/20250508_bike_pose.json",
        ignore_frame_ids=[],
        mf_view_num=1,
        mf_view_ids=[0],
        mf_frame_num=1,
        mf_frame_step=1,
        sampling_strategy="all",
        mf_scene_sampling_strategy="all",
        train_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
        ],
        test_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
        ],
        depth_name="lidar_depth",
        min_depth=0.001,
        max_depth=1000,
        debug=True,
    )

    for index in range(min(1, len(dataset))):
        print(index)
        dataset.__getitem__(index)
