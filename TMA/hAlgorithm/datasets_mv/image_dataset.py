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
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from hAlgorithm.datasets.image_dataset import ImageDataset
from hAlgorithm.datasets_mv.track.vggt_track import (
    build_tracks_by_depth,
    get_depth_inside_flag,
    visualize_tracks_neg_on_images,
    visualize_tracks_on_images,
)
from hAlgorithm.utils import grid_images, instantiate_from_config


class ImageDatasetMV(ImageDataset):
    def __init__(
        self,
        mf_scene_sampling_strategy: str = "all",
        clip_sampler: dict = None,
        clip_maxlen: int = None,
        clip_step: int = None,
        mf_to_sf: bool = False,
        mf_to_mv: bool = False,
        mf_scene: str = "",
        mf_view_ids: list = None,
        mf_frame_ids: list = None,
        ignore_frame_ids: list = None,
        mf_frame_nums: int = None,
        ignore_frame_nums: int = None,
        normalize_cameras: bool = False,
        with_global_scale: bool = False,
        aspect_ratio_range: list = None,
        track_points_nums: int = 0,
        track_neg_ratio: float = 0,
        track_pos_dis_thres: float = -1,
        track_pos_rel_thres: float = 0.05,
        track_seed: int = None,
        with_sift_mask: float = False,
        sift_track_nums: int = 0,
        novel_view_nums: int = 0,
        info_frame_ids: bool = False,
        meta_json: str = None,
        meta_json_split: str = None,
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
            - debug (bool): Whether to enable debug mode, which might include additional checks or verbose logging. Defaults to False.
            **kwargs: Additional keyword arguments that can be used by subclasses or for future expansion.
        """
        self.mf_data = False
        self.mf_to_sf = mf_to_sf
        self.mf_to_mv = mf_to_mv
        self.mf_scene = mf_scene
        self.mf_view_ids = set(mf_view_ids) if mf_view_ids is not None else None
        self.mf_frame_ids = mf_frame_ids
        self.ignore_frame_ids = ignore_frame_ids
        self.mf_frame_nums = mf_frame_nums
        self.ignore_frame_nums = ignore_frame_nums
        self.mf_scene_sampling_strategy = mf_scene_sampling_strategy

        if meta_json is not None:
            with open(meta_json, "r") as f:
                self.trainval_meta = json.load(f)
            self.meta_json_split = meta_json_split or kwargs["phase"]
        else:
            self.trainval_meta = None

        self.normalize_cameras = normalize_cameras
        self.with_global_scale = with_global_scale
        self.aspect_ratio_range = aspect_ratio_range
        self.track_points_nums = track_points_nums
        self.track_neg_ratio = track_neg_ratio
        self.track_pos_dis_thres = track_pos_dis_thres
        self.track_pos_rel_thres = track_pos_rel_thres
        self.track_seed = track_seed

        self.clip_sampler = instantiate_from_config(clip_sampler)
        self.clip_maxlen = clip_maxlen
        self.clip_step = clip_step

        self.with_sift_mask = with_sift_mask
        self.sift_track_nums = sift_track_nums

        if self.with_sift_mask:
            self.sift_detector = cv2.SIFT.create()

        if self.clip_maxlen is None and self.clip_step is not None:
            if isinstance(self.clip_sampler.view_num, int):
                self.clip_maxlen = self.clip_step * self.clip_sampler.view_num
            else:
                self.clip_maxlen = self.clip_step * max(self.clip_sampler.view_num)

        self.novel_view_nums = novel_view_nums
        if self.clip_sampler is not None and self.novel_view_nums is not None:
            if isinstance(self.clip_sampler.view_num, int):
                self.clip_sampler.view_num = self.clip_sampler.view_num + self.novel_view_nums
            elif self.clip_sampler.view_num is not None:
                self.clip_sampler.view_num = [
                    num + self.novel_view_nums for num in self.clip_sampler.view_num
                ]

        self.info_frame_ids = info_frame_ids

        super().__init__(**kwargs)

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
        if not self.mf_to_sf and not self.sf_to_mf:
            self.data_infos = self.load_json_file(self.data_path)
        else:
            super().get_data_infos()

    def load_json_file(self, json_path):
        """Load data from a JSON file."""
        if self.sf_to_mf:
            return super().load_json_file(json_path)

        if isinstance(json_path, (list, tuple)):
            total_data = dict()
            frame_view_ids = dict()
            for i, json_path_i in enumerate(json_path):
                # datas
                with open(json_path_i, "r") as json_file:
                    data = json.load(json_file)["mf_files"]
                data = {f"{i}_{scene}": data[scene] for scene in data}
                total_data.update(data)

                # frame and view ids
                ids_path = json_path_i[:-4] + "metadata.json"
                if os.path.exists(ids_path):
                    with open(ids_path, "r") as json_file:
                        data = json.load(json_file)
                    data = {f"{i}_{scene}": data[scene] for scene in data}
                    frame_view_ids.update(data)

            if self.hdf5:
                return self.load_hdf5_mf_files(total_data, frame_view_ids)
            else:
                return self.load_mf_files(total_data)
        elif json_path.endswith(".json"):
            with open(json_path, "r") as json_file:
                data = json.load(json_file)["mf_files"]

                # frame and view ids
                ids_path = json_path[:-4] + "metadata.json"
                if os.path.exists(ids_path):
                    with open(ids_path, "r") as json_file:
                        frame_view_ids = json.load(json_file)
                else:
                    frame_view_ids = dict()

                if self.hdf5:
                    return self.load_hdf5_mf_files(data, frame_view_ids)
                else:
                    return self.load_mf_files(data)
        else:
            raise ValueError(f"Unsupported file type: {json_path}")

    def scene_sampling(self, datas):
        data_infos = list(datas.keys())
        if isinstance(self.mf_scene_sampling_strategy, str):
            # Parse the sampling strategy to determine how many samples to use
            scene_sampling_number = self._parse_sampling_strategy(
                self.mf_scene_sampling_strategy, len(data_infos)
            )
            # Apply the sampling strategy if a specific number of samples is requested
            if scene_sampling_number is not None:
                if self.mf_scene_sampling_strategy.startswith("first"):
                    data_infos = data_infos[:scene_sampling_number]
                elif self.mf_scene_sampling_strategy.startswith("end"):
                    data_infos = data_infos[-scene_sampling_number:]
                elif self.mf_scene_sampling_strategy.startswith("index"):
                    data_infos = data_infos[scene_sampling_number : scene_sampling_number + 1]
        elif isinstance(self.mf_scene_sampling_strategy, (list, tuple)):
            data_infos = [
                data_infos[int(i)]
                for i in self.mf_scene_sampling_strategy
                if len(data_infos) > int(i)
            ]
        else:
            raise NotImplementedError(
                f"mf_scene_sampling_strategy {self.mf_scene_sampling_strategy} is not implemented"
            )
        return data_infos

    def load_mf_files(self, datas):
        """
        Load multi-frame data from the provided dictionary and populate mf_infos.

        :param datas: A dictionary containing scene information. Each scene contains a list of frames, each frame contains views.
        :return: A list of all view data.
        """
        self.mf_data = True
        if (self.clip_sampler is not None) and (not self.mf_to_sf):
            max_view_num = self.clip_sampler.get_max_view_num()
            min_view_num = self.clip_sampler.get_min_view_num()
            if self.clip_maxlen is not None and max_view_num is not None and self.clip_maxlen > 0:
                self.clip_maxlen = max(self.clip_maxlen, max_view_num)
        else:
            self.clip_maxlen = max_view_num = min_view_num = None

        total_datas = []

        data_infos = self.scene_sampling(datas)

        self.mf_data_scene = len(data_infos)
        self.mf_data_scene_mf = []
        self.mf_data_scene_frames = []
        self.mf_data_scene_views = []
        info_frame_ids_str = ""

        for scene in tqdm(data_infos, total=len(data_infos), desc=f"{self.name}, Loading"):
            cur_total_datas = []

            frames = datas[scene]
            if len(self.mf_scene) > 0 and scene != self.mf_scene:
                continue

            if self.mf_frame_nums is not None:
                mf_frame_step = len(frames) // self.mf_frame_nums
                frames = [frames[i * mf_frame_step] for i in range(self.mf_frame_nums)]
            elif self.ignore_frame_nums is not None:
                ignore_frame_step = len(frames) // self.mf_frame_nums
                ignore_frames_ids = set(
                    [
                        frames[i * ignore_frame_step]["frame_id"]
                        for i in range(self.ignore_frame_nums)
                    ]
                )
                frames = [frame for frame in frames if frame["frame_id"] not in ignore_frames_ids]
            elif self.trainval_meta is not None:
                if scene not in self.trainval_meta:
                    continue
                if self.meta_json_split == "trainval":
                    train = self.trainval_meta[scene]["train"]
                    val = self.trainval_meta[scene]["val"]
                    cur_frame_ids = train["frame_ids"] + val["frame_ids"]
                    cur_view_ids = train["view_ids"] + val["view_ids"]
                else:
                    trainval = self.trainval_meta[scene][self.meta_json_split]
                    cur_frame_ids = trainval["frame_ids"]
                    cur_view_ids = trainval["view_ids"]
                trainval = set([(cur_frame_ids[i], cur_view_ids[i]) for i in range(len(cur_frame_ids))])

            frame_ids_info = []
            tmp_infos = defaultdict(list)
            for frame in frames:
                frame_id = frame["frame_id"]
                views = frame["views"]
                if self.ignore_frame_ids is not None and frame_id in self.ignore_frame_ids:
                    continue
                if self.mf_frame_ids is not None and frame_id not in self.mf_frame_ids:
                    continue
                if self.mf_view_ids is not None:
                    views = [view for view in views if view["view_id"] in self.mf_view_ids]
                if self.trainval_meta is not None:
                    views = [view for view in views if (frame_id, view["view_id"]) in trainval]

                for view in views:
                    view["scene"] = scene
                    view["frame_id"] = frame_id
                    if self.mf_to_mv:
                        tmp_infos[view["view_id"]].append(view)

                if (not self.mf_to_mv) and ((max_view_num is None) or (len(views) >= max_view_num)):
                    cur_total_datas.append(views)

                if len(views) > 0 and self.info_frame_ids:
                    frame_ids_info.append(frame_id)

            if self.info_frame_ids:
                info_frame_ids_str += f"\nmf_scene: {scene}\nmf_frame_ids: {frame_ids_info}"

            self.mf_data_scene_frames.append(len(frames))
            self.mf_data_scene_views.append(len(views))

            if self.mf_to_mv:
                scene_mf_nums = 0
                for view_id, views in tmp_infos.items():
                    if self.clip_maxlen is not None:
                        split_views = [
                            views[i : i + self.clip_maxlen]
                            for i in range(0, len(views), self.clip_maxlen)
                            if (max_view_num is None)
                            or (len(views[i : i + self.clip_maxlen]) >= max_view_num)
                        ]
                        cur_total_datas.extend(split_views)
                        scene_mf_nums += len(split_views)
                    else:
                        cur_total_datas.append(views)
                        scene_mf_nums += 1

                self.mf_data_scene_mf.append(scene_mf_nums)

            if not self.mf_to_sf:
                if isinstance(self.sampling_strategy, str):
                    sampling_number = self._parse_sampling_strategy(
                        self.sampling_strategy, len(cur_total_datas)
                    )
                    if self.sampling_strategy.startswith("first"):
                        cur_total_datas = cur_total_datas[:sampling_number]
                    elif self.sampling_strategy.startswith("end"):
                        cur_total_datas = cur_total_datas[-sampling_number:]
                    elif self.sampling_strategy.startswith("index"):
                        cur_total_datas = cur_total_datas[sampling_number : sampling_number + 1]
                elif isinstance(self.sampling_strategy, (list, tuple)):
                    cur_total_datas = [
                        cur_total_datas[int(i)]
                        for i in self.sampling_strategy
                        if len(cur_total_datas) > int(i)
                    ]
                else:
                    raise NotImplementedError(
                        f"sampling_strategy {self.sampling_strategy} is not implemented"
                    )

                total_datas.extend(cur_total_datas)
            else:
                total_datas = total_datas + [view for views in cur_total_datas for view in views]

        if self.info_frame_ids:
            logging.info(info_frame_ids_str)

        return total_datas

    def load_hdf5_mf_files(self, datas, frame_view_ids=None):
        """
        Load multi-frame data from the provided dictionary and populate mf_infos.

        :param datas: A dictionary containing scene information. Each scene contains a list of frames, each frame contains views.
        :return: A list of all view data.
        """
        self.mf_data = True
        if (self.clip_sampler is not None) and (not self.mf_to_sf):
            max_view_num = self.clip_sampler.get_max_view_num()
            min_view_num = self.clip_sampler.get_min_view_num()
            if self.clip_maxlen is not None and max_view_num is not None and self.clip_maxlen > 0:
                self.clip_maxlen = max(self.clip_maxlen, max_view_num)
        else:
            self.clip_maxlen = max_view_num = min_view_num = None

        total_datas = []

        data_infos = self.scene_sampling(datas)

        self.mf_data_scene = len(data_infos)
        self.mf_data_scene_mf = []
        self.mf_data_scene_frames = []
        self.mf_data_scene_views = []
        info_frame_ids_str = ""

        for scene in tqdm(data_infos, total=len(data_infos), desc=f"{self.name}, Loading"):
            if len(self.mf_scene) > 0 and scene != self.mf_scene:
                continue

            hdf5_path = os.path.join(self.data_root, datas[scene]["hdf5"])

            if frame_view_ids is not None and scene in frame_view_ids:
                frame_ids = frame_view_ids[scene]["frame_ids"]
                view_ids = frame_view_ids[scene]["view_ids"]
            else:
                try:
                    with h5py.File(hdf5_path, "r") as f:
                        frame_ids = f["frame_idx"][:]
                        view_ids = f["view_idx"][:]
                except Exception as e:
                    logging.warning(f"{hdf5_path}: {e}")

            num = len(frame_ids)
            hdf5_ids = np.arange(num)
            mask = np.ones(num).astype(bool)

            if self.mf_view_ids is not None:
                for mf_view_id in self.mf_view_ids:
                    mask = mask & (view_ids == mf_view_id)

            hdf5_ids = hdf5_ids[mask]

            cur_total_datas = []
            tmp_infos_mf = defaultdict(list)
            tmp_infos_mv = defaultdict(list)

            if self.mf_frame_nums is not None:
                mf_frame_step = len(hdf5_ids) // self.mf_frame_nums
                hdf5_ids = [hdf5_ids[i * mf_frame_step] for i in range(self.mf_frame_nums)]
            elif self.ignore_frame_nums is not None:
                ignore_frame_step = len(hdf5_ids) // self.ignore_frame_nums
                ignore_hdf5_ids = set(
                    [hdf5_ids[i * ignore_frame_step] for i in range(self.ignore_frame_nums)]
                )
                hdf5_ids = [hdf5_id for hdf5_id in hdf5_ids if hdf5_id not in ignore_hdf5_ids]
            elif self.trainval_meta is not None:
                if scene not in self.trainval_meta:
                    continue
                if self.meta_json_split == "trainval":
                    train = self.trainval_meta[scene]["train"]
                    val = self.trainval_meta[scene]["val"]
                    cur_frame_ids = train["frame_ids"] + val["frame_ids"]
                    cur_view_ids = train["view_ids"] + val["view_ids"]
                else:
                    trainval = self.trainval_meta[scene][self.meta_json_split]
                    cur_frame_ids = trainval["frame_ids"]
                    cur_view_ids = trainval["view_ids"]
                trainval = set([(cur_frame_ids[i], cur_view_ids[i]) for i in range(len(cur_frame_ids))])

            for hdf5_id in hdf5_ids:
                frame_id = frame_ids[hdf5_id]
                view_id = view_ids[hdf5_id]
                if self.ignore_frame_ids is not None and frame_id in self.ignore_frame_ids:
                    continue
                if self.mf_frame_ids is not None and frame_id not in self.mf_frame_ids:
                    continue
                if self.trainval_meta is not None:
                    if (frame_id, view_id) not in trainval:
                        continue
                view = dict(
                    hdf5=datas[scene]["hdf5"],
                    hdf5_id=hdf5_id,
                    frame_id=frame_id,
                    view_id=view_id,
                    scene=scene,
                )
                tmp_infos_mf[frame_id].append(view)
                tmp_infos_mv[view_id].append(view)

            if self.info_frame_ids:
                info_frame_ids_str += (
                    f"\nmf_scene: {scene}\nmf_frame_ids: {list(tmp_infos_mf.keys())}"
                )

            self.mf_data_scene_frames.append(len(tmp_infos_mf))
            self.mf_data_scene_views.append(len(tmp_infos_mv))

            for frame_id, views in tmp_infos_mf.items():
                if (not self.mf_to_mv) and ((max_view_num is None) or (len(views) >= max_view_num)):
                    cur_total_datas.append(views)

            if self.mf_to_mv:
                scene_mf_nums = 0
                for view_id, views in tmp_infos_mv.items():
                    if self.clip_maxlen is not None:
                        split_views = [
                            views[i : i + self.clip_maxlen]
                            for i in range(0, len(views), self.clip_maxlen)
                            if (max_view_num is None)
                            or (len(views[i : i + self.clip_maxlen]) >= max_view_num)
                        ]
                        cur_total_datas.extend(split_views)
                        scene_mf_nums += len(split_views)
                    else:
                        cur_total_datas.append(views)
                        scene_mf_nums += 1

                self.mf_data_scene_mf.append(scene_mf_nums)

            if not self.mf_to_sf:
                if isinstance(self.sampling_strategy, str):
                    sampling_number = self._parse_sampling_strategy(
                        self.sampling_strategy, len(cur_total_datas)
                    )
                    if self.sampling_strategy.startswith("first"):
                        cur_total_datas = cur_total_datas[:sampling_number]
                    elif self.sampling_strategy.startswith("end"):
                        cur_total_datas = cur_total_datas[-sampling_number:]
                    elif self.sampling_strategy.startswith("index"):
                        cur_total_datas = cur_total_datas[sampling_number : sampling_number + 1]
                elif isinstance(self.sampling_strategy, (list, tuple)):
                    cur_total_datas = [
                        cur_total_datas[int(i)]
                        for i in self.sampling_strategy
                        if len(cur_total_datas) > int(i)
                    ]
                else:
                    raise NotImplementedError(
                        f"sampling_strategy {self.sampling_strategy} is not implemented"
                    )

                total_datas.extend(cur_total_datas)
            else:
                total_datas = total_datas + [view for views in cur_total_datas for view in views]

        if self.info_frame_ids:
            logging.info(info_frame_ids_str)

        return total_datas

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

    def __getitem__(self, idx) -> dict:
        if self.mf_data and not self.mf_to_sf and not self.sf_to_mf:
            return self.getitem_multi_frame(idx)
        else:
            return super().__getitem__(idx)

    def getitem_multi_frame(self, idx):
        if self.phase == "test":
            return self.get_mf_data_for_test(idx)
        else:
            if self.debug:
                return self.get_mf_data_for_trainval(idx)
            else:
                try:
                    return self.get_mf_data_for_trainval(idx)
                except Exception as e:
                    traceback.print_exc()
                    logging.error(
                        f"Dataset:{self.name} getitem error, index update {idx}, got exception {e}"
                    )

                    if isinstance(idx, (list, tuple)):
                        if self.phase == "train":
                            idx[0] = random.randint(0, len(self.data_infos))
                        else:
                            idx[0] = (idx[0] + 1) % len(self.data_infos)
                    else:
                        if self.phase == "train":
                            idx = random.randint(0, len(self.data_infos))
                        else:
                            idx = (idx + 1) % len(self.data_infos)
                    return self.getitem_multi_frame(idx)

    def get_mf_data_for_trainval(self, idx):
        if isinstance(idx, (list, tuple)):
            idx, others = idx
            self.update_data_transforms(**others)

        mf_data = self.load_mf_data(idx)
        assert mf_data is not None

        # mf data transforms
        transform_info = {}
        mf_data_batch_list = []
        image_show_list = []

        # if self.phase == "train" and self.aspect_ratio_range is not None:
        #     random_aspect_ratio = random.uniform(
        #         min(self.aspect_ratio_range), max(self.aspect_ratio_range)
        #     )
        #     for transform in self.data_transforms.transforms:
        #         if hasattr(transform, "update_aspect_ratio"):
        #             transform.update_aspect_ratio(aspect_ratio=random_aspect_ratio)

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

        if self.track_points_nums > 0:
            tracks, track_vis_masks, track_pos_masks = self.create_track_points(
                idx, mf_data_batch_list, image_show_list
            )

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

        if self.track_points_nums > 0:
            mf_data_batch["track_query_points"] = tracks
            mf_data_batch["track_vis"] = track_vis_masks
            mf_data_batch["track_pos_masks"] = track_pos_masks

        if self.with_sift_mask:
            (
                sift_mask,
                sift_track_points,
                sift_track_points_cam,
                sift_track_points_uv,
                sift_track_mask,
                sift_track_vis,
            ) = self.create_sift_mask(idx, mf_data_batch_list, image_show_list)
            mf_data_batch["sift_mask"] = sift_mask

            if self.sift_track_nums > 0:
                mf_data_batch["sift_track_points"] = sift_track_points
                mf_data_batch["sift_track_points_cam"] = sift_track_points_cam
                mf_data_batch["sift_track_points_uv"] = sift_track_points_uv
                mf_data_batch["sift_track_mask"] = sift_track_mask
                mf_data_batch["sift_track_vis"] = sift_track_vis

        if self.debug:
            frame_ids = [view["frame_id"] for view in mf_data["info"]]
            self.debug_mf_data_dict(idx, mf_data_batch_list, view_ids, frame_ids)

        return mf_data_batch

    def get_mf_data_for_test(self, idx):
        return self.get_mf_data_for_trainval(idx)

    def load_mf_data(self, idx):
        seq_list = self.clip_sampler(self.data_infos[idx])

        mf_data, mf_info = [], []
        for view in seq_list:
            single_frame_data = self.load_data(view)
            if self.check_data_extrinsics(single_frame_data):
                mf_data.append(single_frame_data)
                mf_info.append(view)
            else:
                logging.info(f"Dataset:{self.name} invalid extrinsics found.")
                return None

        batch = dict(
            data=mf_data,
            info=mf_info,
        )

        return batch

    def check_data_extrinsics(self, data_dict, eps=5e-3):
        extrinsics = np.array(data_dict["curr_extrinsics"])
        if np.isnan(extrinsics).sum() > 0 or np.isinf(extrinsics).sum() > 0:
            return False
        last_corner = extrinsics[-1][-1]
        if int(last_corner + eps) != 1:
            return False
        return True

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
        stacked_dict = {key: default_collate(collected_values[key]) for key in keys}

        return stacked_dict

    def normalize_cameras_base_first(self, datas):
        """
        Normalize all camera extrinsics relative to the first camera.
        Args:
            datas (list): A list of dictionaries containing pointmap and extrinsics information.
        """

        def fix(pointmap, extrinsics):
            h, w = pointmap.shape[1], pointmap.shape[2]
            pointmap = pointmap.reshape(3, -1)
            extrinsics_inv = extrinsics.inverse()
            R = extrinsics_inv[:3, :3]
            T = extrinsics_inv[:3, 3]
            pointmap_ref = (R @ pointmap + T.unsqueeze(-1)).reshape(3, h, w)
            return pointmap_ref

        # Get the inverse of the first camera's extrinsics
        first_extrinsics_inv = datas[0]["extrinsics"].inverse()

        sparse_pointmap_max_range = []
        for data in datas:
            # Compute new extrinsics relative to the first camera
            data["extrinsics_reff"] = data["extrinsics"] @ first_extrinsics_inv
            # Apply transformation to the pointmap
            data["pointmap_reff"] = fix(data["pointmap"], data["extrinsics_reff"])

            if "sparse_pointmap_max_range" in data:
                sparse_pointmap_max_range.append(data["sparse_pointmap_max_range"])

        if self.with_global_scale:
            points = np.concatenate(
                [data["pointmap_reff"].reshape(3, -1).permute(1, 0).numpy() for data in datas],
                axis=0,
            )
            global_scale = np.mean(np.linalg.norm(points, axis=1))
            for data in datas:
                data["sparse_pointmap_max_range"] = global_scale
        else:
            # Find the maximum range among all sparse pointmaps
            if len(sparse_pointmap_max_range) > 0:
                sparse_pointmap_max_range = max(sparse_pointmap_max_range)
                for data in datas:
                    data["sparse_pointmap_max_range"] = sparse_pointmap_max_range

    def create_track_points(self, idx, data_batch_list, image_show_list):
        extrinsics = []
        intrinsics = []
        depths = []
        point_masks = []
        points = []
        for data in data_batch_list:
            extrinsics.append(data["extrinsics"])
            intrinsics.append(data["intrinsics"])
            depths.append(data["depth"])
            point_masks.append(data["depth_mask"])
            points.append(data["pointmap"])

        extrinsics = torch.stack(extrinsics, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        depths = torch.stack(depths, dim=0)
        point_masks = torch.stack(point_masks, dim=0)
        points = torch.stack(points, dim=0)

        N, _, H, W = points.shape
        world_points = extrinsics.inverse() @ torch.cat(
            [points, points.new_ones(N, 1, H, W)], dim=1
        ).reshape(N, 4, -1)
        world_points = world_points.permute(0, 2, 1)[..., :3].reshape(N, H, W, 3)

        final_tracks, final_vis_masks, final_pos_masks = build_tracks_by_depth(
            extrinsics,
            intrinsics,
            world_points,
            depths if depths.ndim == 3 else depths[:, 0],
            point_masks if point_masks.ndim == 3 else point_masks[:, 0],
            points=points if self.track_pos_dis_thres >= 0 else None,
            pos_dis_thres=self.track_pos_dis_thres,
            pos_rel_thres=self.track_pos_rel_thres,
            neg_epipolar_thres=5,
            boundary_thres=4,
            target_track_num=self.track_points_nums,
            neg_ratio=self.track_neg_ratio,
            neg_sample_size_ratio=0.5,
            images=None,
            seed=self.track_seed,
            debug=self.debug,
        )
        # final_tracks [N, 128, 2]
        # final_vis_masks [N, 128]
        # final_pos_masks [128]
        # uv_int = final_tracks.floor().long().clone()

        if self.debug:
            path = f"./debug/track_points/{self.name}/{idx:06d}/"
            os.makedirs(path, exist_ok=True)
            visualize_tracks_on_images(
                torch.stack([img[[2, 1, 0]] for img in image_show_list])[None],
                final_tracks[None],
                track_vis_mask=final_vis_masks[None],
                out_dir=path,
                image_format="CHW",  # "CHW" or "HWC"
                # normalize_mode="[0,1]",
                normalize_mode="",
                cmap_name="hsv",
            )

            visualize_tracks_neg_on_images(
                torch.stack([img[[2, 1, 0]] for img in image_show_list])[None],
                final_tracks[None],
                track_pos_mask=final_pos_masks[None],
                out_dir=path,
                image_format="CHW",  # "CHW" or "HWC"
                # normalize_mode="[0,1]",
                normalize_mode="",
                cmap_name="hsv",
            )

        return final_tracks, final_vis_masks, final_pos_masks

    def create_sift_mask(self, idx, mf_data_batch_list, image_show_list):
        """
        从图像中提取 SIFT 关键点，并根据 depth_mask 创建 sift_mask。
        如果启用了 sift_track_nums，则还会进行多帧间的 SIFT 点追踪。

        参数:
            idx (int): 当前 batch 的索引，用于 debug 输出文件名。
            mf_data_batch_list (list of dict): 多视角数据字典列表，每个元素包含相机参数、深度图等信息。
            image_show_list (list of tensor): RGB 图像列表（tensor 格式），用于可视化。

        返回:
            tuple: 包含：
                - sift_mask_list: 每张图像的 SIFT 点掩码（torch.bool）
                - world_points: 第一帧中选中的 SIFT 点对应的 3D 坐标
                - points_cam: 所有帧中对应点在相机坐标系下的坐标
                - points_uv: 所有帧中对应点在图像上的像素坐标
                - valid_mask: 这些点是否在图像边界内且有效
        """

        sift_mask_list = []
        first_sift_points_2d = []
        for ci, rgb in enumerate(image_show_list):
            rgb = rgb.permute(1, 2, 0).numpy().astype(np.uint8)
            height, width = rgb.shape[:2]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            # 使用 SIFT 检测关键点
            keypoints = self.sift_detector.detect(gray)

            depth_mask = mf_data_batch_list[ci]["depth_mask"].squeeze().bool()
            mask = torch.zeros([height, width], dtype=torch.bool)

            # 遍历所有 SIFT 点，判断其是否在 depth_mask 内
            for keypoint in keypoints:
                x = round(keypoint.pt[1])
                y = round(keypoint.pt[0])

                if depth_mask[x, y]:
                    mask[x, y] = 1
                    if ci == 0 and self.sift_track_nums > 0:
                        first_sift_points_2d.append([keypoint.pt[0], keypoint.pt[1]])

            sift_mask_list.append(mask)

            if self.debug:
                x, y = np.where(mask)
                show_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for i in range(len(x)):
                    cv2.circle(
                        show_img, (int(y[i]), int(x[i])), radius=1, color=(0, 0, 255), thickness=-1
                    )

                path = f"./debug/sift_mask/{self.name}/{idx:06d}_{ci:03d}.jpg"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cv2.imwrite(path, show_img)

        # 如果需要追踪 SIFT 点
        if self.sift_track_nums > 0:

            # 提取每帧的外参矩阵和内参矩阵
            extrinsics_reff = torch.stack([info["extrinsics_reff"] for info in mf_data_batch_list])
            intrinsics = torch.stack([info["intrinsics"] for info in mf_data_batch_list])

            first_sift_points_2d = torch.tensor(first_sift_points_2d)

            # 控制最多追踪的点数
            if first_sift_points_2d.shape[0] > self.sift_track_nums:
                cur_sift_track_nums = self.sift_track_nums
                if self.phase == "train":
                    select = list(range(first_sift_points_2d.shape[0]))
                    random.shuffle(select)
                    first_sift_points_2d = first_sift_points_2d[select[: self.sift_track_nums]]
                else:
                    first_sift_points_2d = first_sift_points_2d[: self.sift_track_nums]
            else:
                cur_sift_track_nums = first_sift_points_2d.shape[0]
                first_sift_points_2d = torch.cat(
                    [
                        first_sift_points_2d,
                        first_sift_points_2d.new_zeros(
                            [self.sift_track_nums - cur_sift_track_nums, 2]
                        ),
                    ]
                )

            # 获取这些点在世界坐标系下的 3D 坐标
            world_points = mf_data_batch_list[0]["pointmap_reff"][
                :, first_sift_points_2d[:, 1].long(), first_sift_points_2d[:, 0].long()
            ].permute(1, 0)
            # 转换为齐次坐标 [x, y, z, 1]
            world_points_hm = torch.cat([world_points, torch.ones(world_points.shape[0], 1)], dim=1)
            world_points_hm = world_points_hm.unsqueeze(0).expand(extrinsics_reff.shape[0], -1, -1)

            # 投影到相机坐标系下
            points_cam = (extrinsics_reff @ world_points_hm.permute(0, 2, 1))[:, :3]
            # 投影到图像平面
            points_uv = intrinsics @ points_cam
            points_uv = points_uv[:, :2] / points_uv[:, 2:3]
            points_uv = points_uv.permute(0, 2, 1)

            # 判断投影点是否在图像范围内
            valid_mask = (
                (points_uv[..., 0] >= 0)
                & (points_uv[..., 0] < width - 0.5)
                & (points_uv[..., 1] >= 0)
                & (points_uv[..., 1] < height - 0.5)
            )
            valid_mask[:, cur_sift_track_nums:] = 0

            depth_inside_flag = None
            depths = torch.stack([info["depth"].squeeze() for info in mf_data_batch_list])

            S = len(mf_data_batch_list)
            batch_indices = torch.arange(S).view(S, 1).expand(-1, points_uv.shape[1])

            # 四舍五入并尝试多个偏移量，判断跟踪点是否可见
            for shift in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                cur_uv_int = points_uv.long() + torch.tensor(shift)
                uv_int_mask = (
                    (cur_uv_int[..., 0] >= 0)
                    & (cur_uv_int[..., 0] < width)
                    & (cur_uv_int[..., 1] >= 0)
                    & (cur_uv_int[..., 1] < height)
                )
                cur_uv_int = cur_uv_int * uv_int_mask.unsqueeze(-1)
                cur_depth_inside_flag = get_depth_inside_flag(
                    depths, batch_indices, cur_uv_int, points_cam[:, 2], 0.05
                )
                cur_depth_inside_flag = cur_depth_inside_flag & uv_int_mask
                if depth_inside_flag is None:
                    depth_inside_flag = cur_depth_inside_flag
                else:
                    depth_inside_flag = torch.logical_or(depth_inside_flag, cur_depth_inside_flag)

            # 最终跟踪点的可见类别和深度合法性共同决定
            depth_inside_flag = depth_inside_flag & valid_mask

            if self.debug:
                from hAlgorithm.datasets_mv.track.vggt_track import get_track_colors_by_position

                track_colors_rgb = get_track_colors_by_position(
                    points_uv,  # shape (S, N, 2)
                    vis_mask_b=None,
                    image_width=width,
                    image_height=height,
                    cmap_name="hsv",
                )

                show_img_list = []
                for ci, rgb in enumerate(image_show_list):
                    rgb = rgb.permute(1, 2, 0).numpy().astype(np.uint8)
                    show_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    uv = points_uv[ci][valid_mask[ci]]
                    vis = depth_inside_flag[ci][valid_mask[ci]]
                    cur_rgb = track_colors_rgb[valid_mask[ci].numpy()]
                    for i in range(len(uv)):
                        cv2.circle(
                            show_img,
                            (int(uv[i, 0]), int(uv[i, 1])),
                            radius=2,
                            color=cur_rgb[i].tolist() if vis[i] else (0, 0, 0),
                            thickness=-1,
                        )

                    show_img_list.append(show_img)

                    path = f"./debug/sift_mask/{self.name}/{idx:06d}_{ci:03d}_track.jpg"
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    cv2.imwrite(path, show_img)

                show_img_list = np.concatenate(show_img_list, axis=1)
                path = f"./debug/sift_mask/{self.name}/{idx:06d}_track.jpg"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cv2.imwrite(path, show_img_list)
                print("sift track", path)

            return (
                torch.stack(sift_mask_list, dim=0),
                world_points,
                points_cam.permute(0, 2, 1),
                points_uv,
                valid_mask,
                depth_inside_flag,
            )
        else:
            return torch.stack(sift_mask_list, dim=0), None, None, None, None, None

    def debug_mf_data_dict(self, idx, data_batch_list, view_ids, frame_ids):
        """
        Debug function to save intermediate results for visualization.
        Args:
            idx (int): Index identifier for saving results.
            data_batch_list (list): List of dictionaries containing batched data.
        """
        paths = []
        for i, data_batch in enumerate(data_batch_list):
            view_id = view_ids[i]
            frame_id = frame_ids[i]
            prefix = f"view{view_id:02d}_frame{frame_id:02d}_"

            outdir = self.debug_data_dict(idx, data_batch, prefix=prefix)

            pointmap_reff = data_batch.get("pointmap_reff", None)
            extrinsics_reff = data_batch.get("extrinsics_reff", None)

            if pointmap_reff is not None:
                import open3d as o3d

                if isinstance(pointmap_reff, torch.Tensor):
                    pointmap_reff = pointmap_reff.numpy().transpose(1, 2, 0)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pointmap_reff.reshape(-1, 3))
                save_path = os.path.join(outdir, f"{prefix}pointmap_reff.ply")
                o3d.io.write_point_cloud(save_path, pcd)

            if extrinsics_reff is not None:
                extrinsics_reff = extrinsics_reff.cpu().squeeze(0).numpy()
                save_path = os.path.join(outdir, f"{prefix}extrinsics_reff.json")
                with open(save_path, "w") as f:
                    json.dump(extrinsics_reff.tolist(), f, indent=2)

            with open(os.path.join(outdir, "rgb.list"), "a") as f:
                f.write(prefix + ":" + data_batch["meta_data"]["data_info"]["rgb"] + "\n")

            rgb_path = os.path.join(outdir, f"{prefix}image.jpg")
            paths.append(rgb_path)

        # Combine images into a grid and save
        save_path = os.path.join(os.path.dirname(outdir), f"{idx:04d}_merge_image.jpg")
        grid_images(save_path=save_path, paths=paths, col=4)
        print(f"grid {save_path}")


if __name__ == "__main__":

    dataset = BaseDatasetMV(
        phase="test",
        name="hypersim",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/Hypersim_new/total_mf.json",
        track_points_nums=0,
        track_neg_ratio=0.0,
        mf_to_mv=True,  # BaseDatasetMV
        clip_maxlen=10,  # BaseDatasetMV
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=4,
            seed=None,
        ),
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=518,
                patch_size=14,
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
        min_depth=1e-3,
        max_depth=25.0,
        with_pointmap=True,
        debug=True,
        sparse_pattern=None,
        normalize_cameras=True,
        with_sift_mask=True,
        sift_track_nums=8,
    )

    for index in range(min(1, len(dataset))):
        print(index)
        dataset.__getitem__(index)
