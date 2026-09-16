import os
import sys

sys.path.append(os.getcwd())

import json
import logging
import time

import numpy as np
from tqdm import tqdm

from hAlgorithm.datasets.transforms.transforms import resize_depth_preserve
from hAlgorithm.datasets_app.app_dataset import APPDataset


class APPDataset4D(APPDataset):
    def __init__(
        self,
        mf_scene_sampling_strategy: str = "all",
        mf_to_sf: bool = False,
        mf_scene: str = "",
        mf_frame_id: int = None,
        mf_frame_ids: list = None,
        mf_view_ids: list = None,
        max_frame_id: int = None,
        min_frame_id: int = None,
        mf_frame_num: int = 1,
        mf_view_num: int = 1,
        mf_frame_step: int = 1,
        mf_offline_mode: bool = False,
        clip_shuffle: bool = False,
        normalize_cameras: bool = False,
        with_extrinsics_scale: bool = False,
        **kwargs,
    ):
        self.mf_data = None
        self.mf_to_sf = mf_to_sf
        self.mf_scene = mf_scene
        self.mf_frame_id = mf_frame_id
        self.mf_frame_ids = set(mf_frame_ids) if mf_frame_ids is not None else None
        self.max_frame_id = max_frame_id
        self.min_frame_id = min_frame_id
        self.mf_view_ids = set(mf_view_ids) if mf_view_ids is not None else None
        self.mf_frame_num = mf_frame_num
        self.mf_view_num = mf_view_num
        self.mf_frame_step = mf_frame_step
        self.mf_scene_sampling_strategy = mf_scene_sampling_strategy
        self.mf_offline_mode = mf_offline_mode
        self.clip_shuffle = clip_shuffle
        self.normalize_cameras = normalize_cameras
        self.with_extrinsics_scale = with_extrinsics_scale

        assert self.mf_to_sf, "APPDataset4D only support mf_to_sf"

        super().__init__(**kwargs)

    def load_json_file(self, json_path):
        """Load data from a JSON file."""
        if isinstance(json_path, (list, tuple)):
            total_data = dict()
            for json_path_i in json_path:
                with open(json_path_i, "r") as json_file:
                    t0 = time.time()
                    data = json.load(json_file)["mf_files"]
                    t1 = time.time()
                    logging.info(f"load file {json_path_i} time: {t1 - t0:.2f}s")
                total_data.update(data)
            return self.load_mf_files(total_data)
        elif json_path.endswith(".json"):
            with open(json_path, "r") as json_file:
                t0 = time.time()
                data = json.load(json_file)["mf_files"]
                t1 = time.time()
                logging.info(f"load file {json_path} time: {t1 - t0:.2f}s")
                return self.load_mf_files(data)
        else:
            raise ValueError(f"Unsupported file type: {json_path}")

    def load_mf_files(self, datas=None, scene_index=None):
        """
        Load multi-frame data from the provided dictionary and populate mf_infos.

        :param datas: A dictionary containing scene information. Each scene contains a list of frames, each frame contains views.
        :return: A list of all view data.
        """
        if datas is None:
            datas = self.mf_data
        else:
            self.mf_data = datas

        total_datas = []
        # self.clip_infos = []
        self.mf_infos = []

        scene_infos = list(datas.keys())

        if self.mf_scene is None or len(self.mf_scene) == 0:
            # Parse the sampling strategy to determine how many samples to use
            scene_sampling_number = self._parse_sampling_strategy(
                self.mf_scene_sampling_strategy, len(scene_infos)
            )
            # Apply the sampling strategy if a specific number of samples is requested
            if scene_sampling_number is not None:
                if self.mf_scene_sampling_strategy.startswith("first"):
                    scene_infos = scene_infos[:scene_sampling_number]
                elif self.mf_scene_sampling_strategy.startswith("end"):
                    scene_infos = scene_infos[-scene_sampling_number:]
                elif self.mf_scene_sampling_strategy.startswith("index"):
                    scene_infos = scene_infos[scene_sampling_number : scene_sampling_number + 1]

            if scene_index is not None:
                scene_infos = [scene_infos[scene_index]]
        else:
            scene_infos = [self.mf_scene]

        for scene in tqdm(scene_infos, total=len(scene_infos), desc=f"{self.name}, Loading"):
            frames = datas[scene]

            if self.mf_frame_id is not None:
                for i, frame in enumerate(frames):
                    if self.mf_frame_id == frame["frame_id"]:
                        frames = [frames[i]]
                        break

            frame_idxes = []
            for frame_i, frame in enumerate(frames):
                frame_id = frame["frame_id"]
                views = frame["views"]
                if self.mf_frame_id is not None and frame_id != self.mf_frame_id:
                    continue
                if self.mf_frame_ids is not None and frame_id not in self.mf_frame_ids:
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
                cur_id = len(self.mf_infos)
                self.mf_infos.extend([[scene, frame_id, view["view_id"]] for view in views])
                frame_idxes.append([cur_id + i for i, _ in enumerate(views)])

        return total_datas


class APPLidarDepthDataset4D(APPDataset4D):

    def __init__(
        self,
        name: str = "lidardepth",
        depth_name: str = "depth",
        lidar_name: str = "lidar_depth",
        depth_scale: float = 1.0,
        min_depth: float = 1e-3,
        max_depth: float = 200.0,
        confidence_thresh: float = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            depth_name=depth_name,
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


class APPIphoneDataset4D(APPLidarDepthDataset4D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, data_info):
        data_batch = super().load_data(data_info=data_info)

        curr_confidence_path = os.path.join(self.depth_root, data_info["confidence"])
        confidence = self.read_file(curr_confidence_path, image=None, const=0, dtype=np.float32)
        curr_depth_mask = data_batch["curr_depth_mask"]

        if confidence.shape[0:2] != curr_depth_mask.shape[0:2]:
            confidence = resize_depth_preserve(confidence, curr_depth_mask.shape[0:2])

        confidence_mask = confidence == 2
        curr_depth_mask = (curr_depth_mask * confidence_mask).astype(int)
        data_batch["curr_depth_mask"] = curr_depth_mask

        return data_batch


if __name__ == "__main__":

    dataset = APPIphoneDataset4D(
        phase="test",
        name="iphone",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/iphone_data/test_mf_demo_0415_filter.json",
        sampling_strategy="all",
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=952,
                patch_size=14,
                is_lidar=False,
                low_resolution=True,
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
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=952,
                patch_size=14,
                is_lidar=False,
                low_resolution=True,
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
        sparse_depth_ratio=0.2,
        depth_invalid_sem_names=None,
        recalculate_normal=False,
        debug=True,
        sparse_max_size=546,
        sparse_patch_size=14,
        sparse_pattern=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=0,
            sparse_nums=6000,
        ),
        mf_to_sf=True,
        mf_view_ids=[0],
        max_frame_id=10,
    )

    print(len(dataset))

    for index in range(min(10, len(dataset))):
        print(index)
        dataset.__getitem__(index)
