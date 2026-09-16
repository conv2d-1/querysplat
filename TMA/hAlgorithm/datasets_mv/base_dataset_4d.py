import os
import sys

sys.path.append(os.getcwd())

import logging
from collections import defaultdict

import json
import h5py
import numpy as np
import torch
from tqdm import tqdm

from hAlgorithm.utils import grid_images
from hAlgorithm.datasets_mv.base_dataset import BaseDatasetMV


class BaseDataset4D(BaseDatasetMV):
    def __init__(self, replacement=True, **kwargs):
        kwargs["mf_to_sf"] = False
        kwargs["mf_to_mv"] = False
        super().__init__(**kwargs)

        self.replacement = replacement

    def load_mf_files(self, datas):
        """
        Load multi-frame data from the provided dictionary and populate mf_infos.

        :param datas: A dictionary containing scene information. Each scene contains a list of frames, each frame contains views.
        :return: A list of all view data.
        """
        self.mf_data = True
        if self.clip_sampler is not None:
            max_view_num = self.clip_sampler.get_max_view_num()
            max_frame_num = self.clip_sampler.get_max_frame_num()
        else:
            max_view_num = max_frame_num = None

        total_datas = []

        data_infos = self.scene_sampling(datas)

        self.mf_data_scene = len(data_infos)
        self.mf_data_scene_mf = []
        self.mf_data_scene_frames = []
        self.mf_data_scene_views = []

        for scene in tqdm(data_infos, total=len(data_infos), desc=f"{self.name}, Loading"):
            if self.mf_scene is not None and len(self.mf_scene) > 0 and scene != self.mf_scene:
                continue

            frames = datas[scene]

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
                trainval = set(
                    [(cur_frame_ids[i], cur_view_ids[i]) for i in range(len(cur_frame_ids))]
                )

            clip_4d = defaultdict(list)
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

                if max_view_num is None or len(views) >= max_view_num:
                    assert frame_id not in clip_4d
                    clip_4d[frame_id] = sorted(views, key=lambda x: x["view_id"])

            if max_frame_num is None or len(clip_4d.keys()) >= max_frame_num:
                total_datas.append(clip_4d)

            if isinstance(self.sampling_strategy, str):
                sampling_number = self._parse_sampling_strategy(
                    self.sampling_strategy, len(total_datas)
                )
                if self.sampling_strategy.startswith("first"):
                    total_datas = total_datas[:sampling_number]
                elif self.sampling_strategy.startswith("end"):
                    total_datas = total_datas[-sampling_number:]
                elif self.sampling_strategy.startswith("index"):
                    total_datas = total_datas[sampling_number : sampling_number + 1]
            elif isinstance(self.sampling_strategy, (list, tuple)):
                total_datas = [
                    total_datas[int(i)] for i in self.sampling_strategy if len(total_datas) > int(i)
                ]
            else:
                raise NotImplementedError(
                    f"sampling_strategy {self.sampling_strategy} is not implemented"
                )

        return total_datas

    def load_hdf5_mf_files(self, datas, frame_view_ids=None):
        """
        Load multi-frame data from the provided dictionary and populate mf_infos.

        :param datas: A dictionary containing scene information. Each scene contains a list of frames, each frame contains views.
        :return: A list of all view data.
        """
        self.mf_data = True
        if self.clip_sampler is not None:
            max_view_num = self.clip_sampler.get_max_view_num()
            max_frame_num = self.clip_sampler.get_max_frame_num()
        else:
            max_view_num = max_frame_num = None

        total_datas = []

        data_infos = self.scene_sampling(datas)

        self.mf_data_scene = len(data_infos)
        self.mf_data_scene_mf = []
        self.mf_data_scene_frames = []
        self.mf_data_scene_views = []

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

            clip_4d = defaultdict(list)
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
                trainval = set(
                    [(cur_frame_ids[i], cur_view_ids[i]) for i in range(len(cur_frame_ids))]
                )

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
                clip_4d[frame_id].append(view)

            frame_ids = list(clip_4d.keys())
            if max_frame_num is None or len(frame_ids) >= max_frame_num:
                for frame_id in frame_ids:
                    if max_view_num is not None and len(clip_4d[frame_id]) < max_view_num:
                        clip_4d.pop(frame_id)

            if max_frame_num is None or len(clip_4d.keys()) >= max_frame_num:
                total_datas.append(clip_4d)

            if isinstance(self.sampling_strategy, str):
                sampling_number = self._parse_sampling_strategy(
                    self.sampling_strategy, len(total_datas)
                )
                if self.sampling_strategy.startswith("first"):
                    total_datas = total_datas[:sampling_number]
                elif self.sampling_strategy.startswith("end"):
                    total_datas = total_datas[-sampling_number:]
                elif self.sampling_strategy.startswith("index"):
                    total_datas = total_datas[sampling_number : sampling_number + 1]
            elif isinstance(self.sampling_strategy, (list, tuple)):
                total_datas = [
                    total_datas[int(i)] for i in self.sampling_strategy if len(total_datas) > int(i)
                ]
            else:
                raise NotImplementedError(
                    f"sampling_strategy {self.sampling_strategy} is not implemented"
                )

        return total_datas

    def get_mf_data_for_trainval(self, idx):
        if isinstance(idx, (list, tuple)):
            idx, others = idx
            self.update_data_transforms(**others)

        mf_data = self.load_mf_data(idx)
        assert mf_data is not None

        # mf data transforms
        transform_info = {}
        mf_data_batch_list = []
        mf_data_info = []
        input_width = input_height = origin_width = origin_height = None

        for i, mv_data in enumerate(mf_data["data"]):
            mv_data_batch_list = []
            mv_data_info = []
            image_show_list = []
            for j in range(len(mv_data)):
                data_batch = self.get_data_for_trainval(
                    idx,
                    data_info=mf_data["info"][i][j],
                    data_batch=mf_data["data"][i][j],
                    transform_info=transform_info,
                    mf_debug=self.debug,
                )
                meta_data = data_batch.pop("meta_data")
                data_info = meta_data.pop("data_info")

                assert input_width is None or input_width == meta_data["input_width"]
                assert input_height is None or input_height == meta_data["input_height"]
                assert origin_width is None or origin_width == meta_data["origin_width"]
                assert origin_height is None or origin_height == meta_data["origin_height"]

                input_width = meta_data["input_width"]
                input_height = meta_data["input_height"]
                origin_width = meta_data["origin_width"]
                origin_height = meta_data["origin_height"]

                if self.track_points_nums > 0 or self.with_sift_mask:
                    image_show_list.append(transform_info.pop("image_show"))

                if self.debug:
                    print(
                        f'frame_id {data_info["frame_id"]},',
                        f'view_id {data_info["view_id"]},',
                        f'horizontal_flip {transform_info.get("horizontal_flip", None)},',
                        f'RGBCompresion {transform_info.get("RGBCompresion", None)},',
                        f'RandomBlur {transform_info.get("RandomBlur", None)},',
                    )

                mv_data_batch_list.append(data_batch)
                mv_data_info.append(data_info)

            if self.normalize_cameras:
                self.normalize_cameras_base_first(mv_data_batch_list)

            if self.track_points_nums > 0:
                tracks, track_vis_masks, track_pos_masks = self.create_track_points(
                    idx, mv_data_batch_list, image_show_list
                )

            mv_data_batch = self.stack_dicts_list(mv_data_batch_list)

            if self.track_points_nums > 0:
                mv_data_batch["track_query_points"] = tracks
                mv_data_batch["track_vis"] = track_vis_masks
                mv_data_batch["track_pos_masks"] = track_pos_masks

            if self.with_sift_mask:
                (
                    sift_mask,
                    sift_track_points,
                    sift_track_points_cam,
                    sift_track_points_uv,
                    sift_track_mask,
                    sift_track_vis,
                ) = self.create_sift_mask(idx, mv_data_batch_list, image_show_list)
                mv_data_batch["sift_mask"] = sift_mask

                if self.sift_track_nums > 0:
                    mv_data_batch["sift_track_points"] = sift_track_points
                    mv_data_batch["sift_track_points_cam"] = sift_track_points_cam
                    mv_data_batch["sift_track_points_uv"] = sift_track_points_uv
                    mv_data_batch["sift_track_mask"] = sift_track_mask
                    mv_data_batch["sift_track_vis"] = sift_track_vis

            mf_data_batch_list.append(mv_data_batch)
            mf_data_info.append(mv_data_info)


        mf_data_batch = dict()
        for key in mv_data_batch.keys():
            mf_data_batch[key] = torch.cat(
                [mf_data_batch[key] for mf_data_batch in mf_data_batch_list], dim=0
            )
        
        if self.replacement and "sparse_pointmap_max_range" in mf_data_batch:
            # NOTE: To enforce temporal consistency
            # Replace the scale of all views at each timestep with those from the first timestep
            mf_data_batch["sparse_pointmap_max_range"][:] = mf_data_batch["sparse_pointmap_max_range"][0]

        assert input_width is not None
        assert input_height is not None
        assert origin_width is not None
        assert origin_height is not None

        mf_data_batch["meta_data"] = {
            "name": self.name,
            "data_path": self.data_path,
            "data_root": self.data_root,
            "data_idx": idx,
            "depth_scale": self.depth_scale,
            "frames": len(mf_data["data"]),
            "views": len(mf_data["data"][0]),
            "input_width": input_width,
            "input_height": input_height,
            "origin_width": origin_width,
            "origin_height": origin_height,
        }

        data_info = dict()
        for key, val in mf_data_info[0][0].items():
            data_info[key] = []
            for fi in range(mf_data_batch["meta_data"]["frames"]):
                data_info[key].append([])
                for vi in range(mf_data_batch["meta_data"]["views"]):
                    data_info[key][-1].append(mf_data_info[fi][vi][key])

            if not isinstance(val, str):
                data_info[key] = torch.tensor(data_info[key])

        mf_data_batch["meta_data"]["data_info"] = data_info

        if self.debug:
            self.debug_4d_data_dict(idx, mf_data_batch)

        return mf_data_batch

    def load_mf_data(self, idx):
        seq_list = self.clip_sampler(self.data_infos[idx])

        mf_data, mf_info = [], []
        for frame in seq_list:
            mv_data, mv_info = [], []
            for view in frame:
                single_frame_data = self.load_data(view)
                if self.check_data_extrinsics(single_frame_data):
                    mv_data.append(single_frame_data)
                    mv_info.append(view)
                else:
                    logging.info(f"Dataset:{self.name} invalid extrinsics found.")
                    return None

            mf_data.append(mv_data)
            mf_info.append(mv_info)

        batch = dict(
            data=mf_data,
            info=mf_info,
        )

        return batch

    def debug_4d_data_dict(self, idx, data_batch):
        """
        Debug function to save intermediate results for visualization.
        Args:
            idx (int): Index identifier for saving results.
            data_batch (list): List of dictionaries containing batched data.
        """
        paths = []

        meta_data = data_batch["meta_data"]
        data_info = meta_data["data_info"]

        frame_ids = data_info["frame_id"]
        view_ids = data_info["view_id"]

        frames = meta_data["frames"]
        views = meta_data["views"]

        for fi in range(frames):
            for vi in range(views):
                frame_id = frame_ids[fi, vi]
                view_id = view_ids[fi, vi]

                prefix = f"frame{frame_id:03d}_view{view_id:03d}_"
                
                single_data_batch = dict()
                for key, val in data_batch.items():
                    if key != "meta_data":
                        single_data_batch[key] = val.reshape(frames, views, *val.shape[1:])[fi, vi]
                outdir = self.debug_data_dict(idx, single_data_batch, prefix=prefix)

                pointmap_reff = single_data_batch.get("pointmap_reff", None)
                extrinsics_reff = single_data_batch.get("extrinsics_reff", None)

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
                    f.write(prefix + ":" + data_info[self.rbg_name][fi][vi] + "\n")

                rgb_path = os.path.join(outdir, f"{prefix}image.jpg")
                paths.append(rgb_path)

        # Combine images into a grid and save
        save_path = os.path.join(os.path.dirname(outdir), f"{idx:04d}_merge_image.jpg")
        grid_images(save_path=save_path, paths=paths, col=views)
        print(f"grid {save_path}")



if __name__ == "__main__":

    dataset = BaseDataset4D(
        phase="test",
        name="kubric4d",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        # data_path="/mnt/netdata/Team/AI/datasets/TMD/Kubric-4D/train_mf.json",
        data_path="flame_salmon_STG.json",
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler_4d.debug_sampler.Sampler4D",
            frame_num=5,
            view_num=21,
            frame_step=1,
            debug=False,
        ),
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
        max_depth=30.0,
        with_pointmap=False,
        debug=True,
        sparse_pattern=None,
        normalize_cameras=True,
    )

    for index in range(min(1, len(dataset))):
        print(index)
        dataset.__getitem__(index)
