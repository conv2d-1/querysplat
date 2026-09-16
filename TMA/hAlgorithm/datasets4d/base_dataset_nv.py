import os
import sys

sys.path.append(os.getcwd())

import json
import logging
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch

from hAlgorithm.datasets4d.base_dataset import BaseDataset4D


class BaseDataset4DNovel(BaseDataset4D):
    def __init__(
        self,
        extra_ids=None,
        extra_skip_step=None,
        **kwargs,
    ):
        """
        Initializes a new instance of the BaseDataset4D class.

        Args:
            - mf_scene_sampling_strategy (str):
                same as sampling_strategy, used in scene in mf data.
            - mf_to_sf (bool): Convert multi-frame data to single-frame data. Defaults to False.
            - mf_scene (str): Select specific scenes from multi-frame data. Defaults to "".
            - mf_frame_id (int): Select specific frame_id from multi-frame data. Defaults to None.
            - mf_view_ids (list): Select specific view_ids from multi-frame data. Defaults to None.
            - mf_frame_num(int): Number of frames from multi-frame data. Defaults to 1.
            - mf_view_num(int): Number of views from multi-frame data. Defaults to 1.
            - mf_frame_step(int): Step of frames from multi-frame data. Defaults to 1.
            - debug (bool): Whether to enable debug mode, which might include additional checks or verbose logging. Defaults to False.
            **kwargs: Additional keyword arguments that can be used by subclasses or for future expansion.
        """
        self.extra_ids = extra_ids
        self.extra_skip_step = extra_skip_step

        super().__init__(**kwargs)

    def load_json_file(self, json_path):
        """Load data from a JSON file."""
        # logging.info(f"Loading data from {json_path}")
        if json_path.endswith(".json"):
            with open(json_path, "r") as json_file:
                data = json.load(json_file)["mf_files"]
                return self.load_mf_files(data)
        elif json_path.endswith(".parquet"):
            data = pd.read_parquet(json_path)
            if "files" in data:
                data = data["files"]
            elif "mf_files" in data:
                data = self.load_mf_files(data["mf_files"])
            return data
        else:
            raise ValueError(f"Unsupported file type: {json_path}")

    def get_mf_data_for_trainval(self, idx):
        if isinstance(idx, (list, tuple)):
            idx, others = idx
            self.update_data_transforms(**others)

        mf_data = self.load_mf_data(idx)
        if mf_data is None:
            # frame or view not Found.
            if isinstance(idx, (list, tuple)):
                idx[0] = (idx[0] + 1) % len(self.clip_infos)
            else:
                idx = (idx + 1) % len(self.clip_infos)
            logging.info(f"Dataset:{self.name} view or frame not found, index update {idx}.")
            return self.get_mf_data_for_trainval(idx)

        # mf data transforms
        transform_info = {}
        mf_data_batch_list = []
        image_show_list = []
        extra_ids = []
        ids_name = "view_id" if self.mf_view_num > 1 else "frame_id"
        if self.extra_ids is None and self.extra_skip_step is not None:
            view_ids = [int(info[ids_name]) for info in mf_data["info"]]
            extra_ids = sorted(view_ids)[:: self.extra_skip_step]
            # extra_ids = view_ids[::self.extra_skip_step]
        elif self.extra_ids is not None:
            extra_ids = self.extra_ids
        extra_ids_in_batch = []

        for i, _ in enumerate(mf_data["data"]):
            data_batch = self.get_data_for_trainval(
                idx,
                data_info=mf_data["info"][i],
                data_batch=mf_data["data"][i],
                transform_info=transform_info,
                mf_debug=self.debug,
            )
            data_batch["meta_data"] = {
                "data_info": data_batch["meta_data"]["data_info"],
                "input_width": data_batch["meta_data"]["input_width"],
                "input_height": data_batch["meta_data"]["input_height"],
                "origin_width": data_batch["meta_data"]["origin_width"],
                "origin_height": data_batch["meta_data"]["origin_height"],
            }

            if self.track_points_nums > 0:
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

            if int(data_batch["meta_data"]["data_info"][ids_name]) in extra_ids:
                extra_ids_in_batch.append(i)

        # put extra frames at the end
        mf_data_batch_list_extra = []
        mf_data_batch_list_ = []
        for i in range(len(mf_data_batch_list)):
            if i in extra_ids_in_batch:
                mf_data_batch_list_extra.append(mf_data_batch_list[i])
            else:
                mf_data_batch_list_.append(mf_data_batch_list[i])

        extra_ids_in_batch = [len(mf_data_batch_list_) + i for i in range(len(extra_ids_in_batch))]
        mf_data_batch_list = mf_data_batch_list_ + mf_data_batch_list_extra

        if self.normalize_cameras:
            self.normalize_cameras_base_first(mf_data_batch_list)

        if self.track_points_nums > 0:
            tracks, track_vis_masks, track_pos_masks = self.create_track_points(
                idx, mf_data_batch_list, image_show_list
            )

        if self.debug:
            self.debug_mf_data_dict(idx, mf_data_batch_list)

        mf_data_batch = self.stack_dicts_list(mf_data_batch_list)
        mf_data_batch["meta_data"].update(
            {
                "name": self.name,
                "data_path": self.data_path,
                "data_idx": idx,
                "depth_scale": self.depth_scale,
                "views": self.mf_view_num,
                "frames": self.mf_frame_num,
                "extra_ids": torch.tensor(extra_ids_in_batch),
            }
        )
        mf_data_batch["meta_data"]["input_width"] = mf_data_batch["meta_data"]["input_width"][0]
        mf_data_batch["meta_data"]["input_height"] = mf_data_batch["meta_data"]["input_height"][0]
        mf_data_batch["meta_data"]["origin_width"] = mf_data_batch["meta_data"]["origin_width"][0]
        mf_data_batch["meta_data"]["origin_height"] = mf_data_batch["meta_data"]["origin_height"][0]

        if self.track_points_nums > 0:
            mf_data_batch["track_query_points"] = tracks
            mf_data_batch["track_vis"] = track_vis_masks
            mf_data_batch["track_pos_masks"] = track_pos_masks

        return mf_data_batch

    def load_mf_data(self, idx):
        seq_list = self.clip_infos[idx]

        if self.clip_shuffle:
            shuffle_idxs = list(range(len(seq_list)))
            random.shuffle(shuffle_idxs)
            seq_list = [seq_list[i] for i in shuffle_idxs]

        mf_data, mf_info = [], []
        for scene, frame_id, view_id, idx in seq_list:
            f_info = self.data_infos[idx]
            # for parquet
            for k, v in f_info.items():
                if isinstance(v, np.ndarray):
                    f_info[k] = v.tolist()
            single_frame_data = self.load_data(f_info)
            if self.check_data_extrinsics(single_frame_data):
                mf_data.append(single_frame_data)
                f_info["frame_id"] = frame_id
                f_info["scene"] = scene
                mf_info.append(f_info)
            else:
                logging.info(f"Dataset:{self.name} invalid extrinsics found.")
                return None

        batch = dict(
            data=mf_data,
            info=mf_info,
        )

        return batch


if __name__ == "__main__":

    dataset = BaseDataset4DNovel(
        phase="test",
        name="Kubric",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/DL3DV-10K/train_mf_1080p.json",
        depth_name="sfm_depth",
        mf_frame_num=5,
        mf_view_num=1,
        mf_frame_step=30,
        # mf_frame_id=[0, 1, 2, 3],
        # extra_ids=[0, 1, 2, 3],
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=518,
                patch_size=14,
                is_lidar=True,
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
        min_depth=1e-3,
        max_depth=1000.0,
        with_pointmap=True,
        debug=True,
        sparse_pattern=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=0,
            sparse_ratio=1.0,
        ),
        mf_to_sf=False,
        clip_shuffle=True,
        normalize_cameras=True,
    )

    for index in range(min(1, len(dataset))):
        print(index)
        dataset.__getitem__(index)
