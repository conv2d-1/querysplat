import os
import sys

sys.path.append(os.getcwd())

import json
import logging
import random
from collections import defaultdict
from math import ceil

import cv2
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

# from hAlgorithm.datasets4d.transforms.normalize_cameras import normalize_cameras
from hAlgorithm.datasets.base_dataset import BaseDataset
from hAlgorithm.utils import grid_images, instantiate_from_config

# from pytorch3d.renderer import PerspectiveCameras



class BaseDataset4D(BaseDataset):
    def __init__(
        self,
        mf_scene_sampling_strategy: str = "all",
        mf_to_sf: bool = False,
        mf_scene: str = "",
        mf_frame_id: int = None,
        mf_view_ids: list = None,
        max_frame_id: int = None,
        min_frame_id: int = None,
        clip_sampler: dict = None,
        mf_offline_mode: bool = False,
        clip_cache_path: str = None,
        clip_shuffle: bool = False,
        normalize_cameras: bool = False,
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
        self.mf_data = False
        self.mf_to_sf = mf_to_sf
        self.mf_scene = mf_scene
        self.mf_frame_id = mf_frame_id
        self.max_frame_id = max_frame_id
        self.min_frame_id = min_frame_id
        self.mf_view_ids = set(mf_view_ids) if mf_view_ids is not None else None
        self.mf_scene_sampling_strategy = mf_scene_sampling_strategy
        self.mf_offline_mode = mf_offline_mode
        self.clip_cache_path = clip_cache_path
        self.clip_sampler = instantiate_from_config(clip_sampler)
        self.clip_shuffle = clip_shuffle
        self.normalize_cameras = normalize_cameras

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
        if not self.mf_to_sf:
            self.data_infos = self.load_json_file(self.data_path)
            self.build_multi_views_and_frames()
        else:
            super().get_data_infos()

    def load_json_file(self, json_path):
        """Load data from a JSON file."""
        if json_path.endswith(".json"):
            clip_cach_info = None
            if self.clip_cache_path is not None and (not self.mf_to_sf):

                cach_name, cach_f, cach_v = self.check_cach(self.clip_cache_path)

                sample_clip_flen, sample_clipvlen = self.clip_sampler.get_sample_fv()

                assert sample_clip_flen <= cach_f and sample_clipvlen <= cach_v
                if cach_name is None:
                    raise FileNotFoundError(
                        f"Cach File Not Found for clip f:{sample_clip_flen} v:{sample_clipvlen}"
                    )

                logging.info(f"Loaded cach {cach_name}, f:{cach_f} v:{cach_v}.")
                clip_seqs = np.load(self.clip_cache_path)
                clip_cach_info = (clip_seqs, cach_f, cach_v)

            with open(json_path, "r") as json_file:
                data = json.load(json_file)["mf_files"]
                return self.load_mf_files(data, clip_cach_info)
        else:
            raise ValueError(f"Unsupported file type: {json_path}")

    def check_cach(self, cach_path):
        cach_name = os.path.basename(cach_path)
        cach_f, cach_v = (
            cach_name.split(".")[0].split("_clip_cache_f")[1].split("_mp")[0].split("_v")
        )
        return cach_name, int(cach_f), int(cach_v)

    def load_mf_files(self, datas, clip_cach_info=None):
        """
        Load multi-frame data from the provided dictionary and populate mf_infos.

        :param datas: A dictionary containing scene information. Each scene contains a list of frames, each frame contains views.
        :return: A list of all view data.
        """
        self.mf_data = True

        if self.mf_to_sf:
            return self.load_mf_filts_to_sf(datas)

        if clip_cach_info:
            clip_seqs_cach, cach_f, cach_v = clip_cach_info

        total_datas = []
        self.clip_infos = []
        self.mf_infos = []

        scene_infos = list(datas.keys())
        # Parse the sampling strategy to determine how many samples to use
        scene_sampling_number = self._parse_sampling_strategy(
            self.mf_scene_sampling_strategy, len(scene_infos)
        )
        # Apply the sampling strategy if a specific number of samples is requested
        if scene_sampling_number is not None:
            if self.mf_scene_sampling_strategy.startswith("first"):
                sample_scene_infos = scene_infos[:scene_sampling_number]
            elif self.mf_scene_sampling_strategy.startswith("end"):
                sample_scene_infos = scene_infos[-scene_sampling_number:]
            elif self.mf_scene_sampling_strategy.startswith("index"):
                sample_scene_infos = scene_infos[scene_sampling_number : scene_sampling_number + 1]
        else:
            sample_scene_infos = scene_infos

        for scene in scene_infos:
            frames = datas[scene]
            if len(self.mf_scene) > 0 and scene != self.mf_scene:
                continue

            if self.mf_frame_id is not None:
                for i, frame in enumerate(frames):
                    if self.mf_frame_id == frame["frame_id"]:
                        frames = [frames[i]]
                        break

            for frame_i, frame in enumerate(frames):
                frame_id = frame["frame_id"]
                views = frame["views"]
                if self.mf_view_ids is not None:
                    views = [view for view in frame["views"] if view["view_id"] in self.mf_view_ids]
                    frames[frame_i]["views"] = views

                for view in views:
                    view["scene"] = scene
                    view["frame_id"] = frame_id

                total_datas.extend(views)
                self.mf_infos.extend([[scene, frame_id, view["view_id"]] for view in views])

        for scene in sample_scene_infos:
            if len(datas[scene]) < cach_f * cach_v:
                continue
            self.clip_f = None
            if clip_cach_info:
                clip_seqs = []
                if len(frames) >= cach_f:
                    clips = clip_seqs_cach[scene].tolist()
                    sample_clips = []
                    for i, clip in enumerate(clips):
                        sample_clip, clip_f = self.clip_sampler.presample_clip(clip, cach_f)
                        if self.clip_f is None:
                            self.clip_f = clip_f
                        for j, _ in enumerate(sample_clip):
                            sample_clip[j][-3:] = [int(item) for item in sample_clip[j][-3:]]
                        sample_clips.append(sample_clip)
                    clip_seqs.extend(sample_clips)
            else:
                raise NotImplementedError()

            if len(clip_seqs) > 0:
                clip_sampling_number = self._parse_sampling_strategy(
                    self.sampling_strategy, len(clip_seqs)
                )
                if clip_sampling_number is not None:
                    if self.sampling_strategy.startswith("first"):
                        clip_seqs = clip_seqs[:clip_sampling_number]
                    elif self.sampling_strategy.startswith("end"):
                        clip_seqs = clip_seqs[-clip_sampling_number:]
                    elif self.sampling_strategy.startswith("index"):
                        clip_seqs = clip_seqs[clip_sampling_number : clip_sampling_number + 1]
                self.clip_infos.extend(clip_seqs)

        return total_datas

    def load_mf_filts_to_sf(self, datas):
        total_datas = []
        self.mf_infos = []

        scene_infos = list(datas.keys())
        # Parse the sampling strategy to determine how many samples to use
        scene_sampling_number = self._parse_sampling_strategy(
            self.mf_scene_sampling_strategy, len(scene_infos)
        )
        # Apply the sampling strategy if a specific number of samples is requested
        if scene_sampling_number is not None:
            if self.mf_scene_sampling_strategy.startswith("first"):
                sample_scene_infos = scene_infos[:scene_sampling_number]
            elif self.mf_scene_sampling_strategy.startswith("end"):
                sample_scene_infos = scene_infos[-scene_sampling_number:]
            elif self.mf_scene_sampling_strategy.startswith("index"):
                sample_scene_infos = scene_infos[scene_sampling_number : scene_sampling_number + 1]
        else:
            sample_scene_infos = scene_infos

        for scene in sample_scene_infos:
            frames = datas[scene]
            if len(self.mf_scene) > 0 and scene != self.mf_scene:
                continue

            if self.mf_frame_id is not None:
                for i, frame in enumerate(frames):
                    if self.mf_frame_id == frame["frame_id"]:
                        frames = [frames[i]]
                        break

            for frame_i, frame in enumerate(frames):
                frame_id = frame["frame_id"]
                views = frame["views"]
                if self.mf_frame_id is not None and frame_id != self.mf_frame_id:
                    continue
                if self.max_frame_id is not None and frame_id > self.max_frame_id:
                    continue
                if self.min_frame_id is not None and frame_id < self.min_frame_id:
                    continue
                if self.mf_view_ids is not None:
                    views = [view for view in frame["views"] if view["view_id"] in self.mf_view_ids]
                    frames[frame_i]["views"] = views

                for view in views:
                    view["scene"] = scene
                    view["frame_id"] = frame_id

                total_datas.extend(views)
                self.mf_infos.extend([[scene, frame_id, view["view_id"]] for view in views])

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

    def build_multi_views_and_frames(self):
        """
        Build the multi_views and multi_frames dictionaries by categorizing all view and frame information by scene and frame/view ID.
        """
        self.multi_views = defaultdict(list)
        self.multi_frames = defaultdict(list)

        for idx, info in enumerate(self.mf_infos):
            scene, frame_id, view_id = info
            view_key = self.get_multi_view_key(scene, frame_id)
            self.multi_views[view_key].append([view_id, idx])

            frame_key = self.get_multi_frame_key(scene, view_id)
            self.multi_frames[frame_key].append([frame_id, idx])

    @staticmethod
    def get_multi_view_key(scene, frame_id):
        """
        Generate a unique key for storing and retrieving multi-view data based on scene and frame ID.
        """
        return f"scene:{scene}_frame_id:{frame_id}"

    @staticmethod
    def get_multi_frame_key(scene, view_id):
        """
        Generate a unique key for storing and retrieving multi-frame data based on scene and view ID.
        """
        return f"scene:{scene}_view_id:{view_id}"

    def get_multi_views(self, idx):
        """
        Retrieve the corresponding view IDs and index list based on the given index.

        :param idx: Index of the view information
        :return: Tuple containing view IDs list and index list
        """
        scene, frame_id, view_id = self.mf_infos[idx]
        key = self.get_multi_view_key(scene, frame_id)
        views = self.multi_views[key]
        view_ids = [view[0] for view in views]
        idx_list = [view[1] for view in views]

        return view_ids, idx_list

    def get_multi_frames(self, idx):
        """
        Retrieve the corresponding frame IDs and index list based on the given index.

        :param idx: Index of the view information
        :return: Tuple containing frame IDs list and index list
        """
        scene, frame_id, view_id = self.mf_infos[idx]
        key = self.get_multi_frame_key(scene, view_id)

        frames = self.multi_frames[key]
        frame_ids = [frame[0] for frame in frames]
        idx_list = [frame[1] for frame in frames]

        return frame_ids, idx_list

    def __len__(self):
        if self.mf_data and not self.mf_to_sf:
            return len(self.clip_infos)
        else:
            return super().__len__()

    def __getitem__(self, idx) -> dict:
        if self.mf_data and not self.mf_to_sf:
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
                    if isinstance(idx, (list, tuple)):
                        if self.phase == "train":
                            idx[0] = random.randint(0, len(self.clip_infos))
                        else:
                            idx[0] = (idx[0] + 1) % len(self.clip_infos)
                    else:
                        if self.phase == "train":
                            idx = random.randint(0, len(self.clip_infos))
                        else:
                            idx = (idx + 1) % len(self.clip_infos)
                    logging.warning(
                        f"Dataset:{self.name} getitem error, index update {idx}, got exception {e}"
                    )
                    return self.getitem_multi_frame(idx)

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
        # print([f['rgb'] for f in mf_data["info"]])
        # mf data transforms
        transform_info = {}
        mf_data_batch_list = []
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

            if self.debug:
                print(
                    f'frame_id {data_batch["meta_data"]["data_info"]["frame_id"]},',
                    f'view_id {data_batch["meta_data"]["data_info"]["view_id"]},',
                    f'horizontal_flip {transform_info.get("horizontal_flip", None)},',
                    f'RGBCompresion {transform_info.get("RGBCompresion", None)},',
                    f'RandomBlur {transform_info.get("RandomBlur", None)},',
                )

            # Optionally debug the data dictionary
            # if self.debug:
            #     view_id = int(i / self.clip_sampler.frame_num)
            #     frame_id = int(i % self.clip_sampler.frame_num)
            #     self.debug_mf_data_dict(idx, view_id, frame_id, data_batch)

            mf_data_batch_list.append(data_batch)

        if self.normalize_cameras:
            self.normalize_cameras_base_first(mf_data_batch_list)

        if self.debug:
            self.debug_mf_data_dict(idx, mf_data_batch_list)

        mf_data_batch = self.stack_dicts_list(mf_data_batch_list)
        mf_data_batch["meta_data"].update(
            {
                "name": self.name,
                "data_path": self.data_path,
                "data_idx": idx,
                "depth_scale": self.depth_scale,
                "views": self.clip_sampler.view_num,
                "frames": self.clip_sampler.frame_num,
            }
        )
        mf_data_batch["meta_data"]["input_width"] = mf_data_batch["meta_data"]["input_width"][0]
        mf_data_batch["meta_data"]["input_height"] = mf_data_batch["meta_data"]["input_height"][0]
        mf_data_batch["meta_data"]["origin_width"] = mf_data_batch["meta_data"]["origin_width"][0]
        mf_data_batch["meta_data"]["origin_height"] = mf_data_batch["meta_data"]["origin_height"][0]

        return mf_data_batch

    def get_mf_data_for_test(self, idx):
        return self.get_mf_data_for_trainval(idx)

    def load_mf_data(self, idx):
        seq_list = self.clip_infos[idx]
        seq_list = self.clip_sampler(seq_list, self.clip_f)

        if self.clip_shuffle:
            shuffle_idxs = list(range(len(seq_list)))
            random.shuffle(shuffle_idxs)
            seq_list = [seq_list[i] for i in shuffle_idxs]

        mf_data, mf_info = [], []
        for scene, frame_id, view_id, idx in seq_list:
            f_info = self.data_infos[idx]
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

            sparse_pointmap_max_range.append(data["sparse_pointmap_max_range"])

        # Find the maximum range among all sparse pointmaps
        sparse_pointmap_max_range = max(sparse_pointmap_max_range)
        for data in datas:
            data["sparse_pointmap_max_range"] = sparse_pointmap_max_range

    def debug_mf_data_dict(self, idx, data_batch_list):
        """
        Debug function to save intermediate results for visualization.
        Args:
            idx (int): Index identifier for saving results.
            data_batch_list (list): List of dictionaries containing batched data.
        """
        paths = []
        for i, data_batch in enumerate(data_batch_list):
            view_id = int(i / self.clip_sampler.frame_num)
            frame_id = int(i % self.clip_sampler.frame_num)
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


if __name__ == "__main__":

    dataset = BaseDataset4D(
        phase="test",
        name="Habitat-Sim-v2-part2",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/Habitat-Sim-v2/train_mf_v2_0331_part2.json",
        clip_cache_path="/mnt/netdata/Team/AI/datasets/TMD/Habitat-Sim-v2/clip_cache/train_mf_v2_0331_part2_clip_cache_f0005_v0001_mp.npz",
        clip_sampler=dict(
            type="hAlgorithm.datasets4d.clip_sampler.step_sampler.StepSampler",
            frame_num=4,
        ),
        sampling_strategy="first:2",
        mf_scene_sampling_strategy="end:5",
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
        depth_scale=1000.0,
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
        clip_shuffle=True,
        normalize_cameras=True,
    )

    for index in range(min(10, len(dataset))):
        print(index)
        dataset.__getitem__(index)
