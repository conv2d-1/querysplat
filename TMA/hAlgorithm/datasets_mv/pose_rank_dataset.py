import os
import sys

sys.path.append(os.getcwd())

import logging
from collections import defaultdict

import h5py
import numpy as np
from tqdm import tqdm

from hAlgorithm.datasets_mv.base_dataset import BaseDatasetMV
from hAlgorithm.datasets_mv.pose_rank.vggt import compute_ranking


class PoseRankDatasetMV(BaseDatasetMV):
    def __init__(self, rank_lambda_t=1.0, rank_t_normalize=True, **kwargs):

        self.rank_lambda_t = rank_lambda_t
        self.rank_t_normalize = rank_t_normalize

        super().__init__(**kwargs)

    def load_mf_files(self, datas):
        """
        Load multi-frame data from the provided dictionary and populate mf_infos.

        :param datas: A dictionary containing scene information. Each scene contains a list of frames, each frame contains views.
        :return: A list of all view data.
        """
        self.mf_data = True
        if self.clip_sampler is not None:
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

        for scene in tqdm(data_infos, total=len(data_infos), desc=f"{self.name}, Loading"):
            cur_total_datas = []

            frames = datas[scene]
            if len(self.mf_scene) > 0 and scene != self.mf_scene:
                continue

            tmp_infos = defaultdict(list)
            for frame in frames:
                frame_id = frame["frame_id"]
                views = frame["views"]
                if self.mf_view_ids is not None:
                    views = [view for view in frame["views"] if view["view_id"] in self.mf_view_ids]

                for view in views:
                    view["scene"] = scene
                    view["frame_id"] = frame_id
                    if self.mf_to_mv:
                        tmp_infos[view["view_id"]].append(view)

                if (not self.mf_to_mv) and ((max_view_num is None) or (len(views) >= max_view_num)):
                    cur_total_datas.append(views)

            self.mf_data_scene_frames.append(len(frames))
            self.mf_data_scene_views.append(len(views))

            if self.mf_to_mv and not self.mf_to_sf:
                scene_mf_nums = 0
                for view_id, views in tmp_infos.items():
                    rank = compute_ranking(
                        np.linalg.inv(np.stack([view["extrinsics"] for view in views], axis=0)),
                        lambda_t=self.rank_lambda_t,
                        normalize=self.rank_t_normalize,
                        batched=True,
                    )
                    viewss = [
                        [views[j] for j in rank[i][: self.clip_maxlen]] for i in range(len(rank))
                    ]
                    cur_total_datas.extend(viewss)
                    scene_mf_nums += len(viewss)

                self.mf_data_scene_mf.append(scene_mf_nums)
            elif self.mf_to_sf:
                for view_id, views in tmp_infos.items():
                    cur_total_datas.append(views)

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
            elif self.mf_sampling_strategy is not None and self.mf_sampling_strategy != "all":
                # NOTE: mf_to_sf
                if isinstance(self.mf_sampling_strategy, str):
                    sampling_number = self._parse_sampling_strategy(
                        self.mf_sampling_strategy, len(cur_total_datas)
                    )
                    if self.mf_sampling_strategy.startswith("first"):
                        total_datas += [view for views in cur_total_datas for view in views[:sampling_number]]
                    elif self.mf_sampling_strategy.startswith("end"):
                        total_datas += [view for views in cur_total_datas for view in views[-sampling_number:]]
                    elif self.mf_sampling_strategy.startswith("index"):
                        total_datas += [view for views in cur_total_datas for view in views[sampling_number : sampling_number + 1]]
                elif isinstance(self.mf_sampling_strategy, (list, tuple)):
                    total_datas += [views[i] for views in cur_total_datas for i in self.mf_sampling_strategy if i < len(views)]
                else:
                    raise NotImplementedError(
                        f"mf_sampling_strategy {self.mf_sampling_strategy} is not implemented"
                    )
            else:
                total_datas += [view for views in cur_total_datas for view in views]

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

        for scene in tqdm(data_infos, total=len(data_infos), desc=f"{self.name}, Loading"):

            if len(self.mf_scene) > 0 and scene != self.mf_scene:
                continue

            hdf5_path = os.path.join(self.data_root, datas[scene]["hdf5"])
            try:
                with h5py.File(hdf5_path, "r") as f:
                    frame_ids = f["frame_idx"][:]
                    view_ids = f["view_idx"][:]
                    extrinsics_list = f["extrinsics"][:]
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

            for hdf5_id in hdf5_ids:
                frame_id = frame_ids[hdf5_id]
                view_id = view_ids[hdf5_id]
                view = dict(
                    hdf5=datas[scene]["hdf5"],
                    hdf5_id=hdf5_id,
                    frame_id=frame_id,
                    view_id=view_id,
                    scene=scene,
                    extrinsics=extrinsics_list[hdf5_id],
                )
                tmp_infos_mf[frame_id].append(view)
                tmp_infos_mv[view_id].append(view)

            self.mf_data_scene_frames.append(len(tmp_infos_mf))
            self.mf_data_scene_views.append(len(tmp_infos_mv))

            for frame_id, views in tmp_infos_mf.items():
                if (not self.mf_to_mv) and ((max_view_num is None) or (len(views) >= max_view_num)):
                    cur_total_datas.append(views)

            if self.mf_to_mv and not self.mf_to_sf:
                scene_mf_nums = 0
                for view_id, views in tmp_infos_mv.items():
                    rank = compute_ranking(
                        np.linalg.inv(np.stack([view["extrinsics"] for view in views], axis=0)),
                        lambda_t=self.rank_lambda_t,
                        normalize=self.rank_t_normalize,
                        batched=True,
                    )
                    viewss = [
                        [views[j] for j in rank[i][: self.clip_maxlen]] for i in range(len(rank))
                    ]
                    cur_total_datas.extend(viewss)
                    scene_mf_nums += len(viewss)

                self.mf_data_scene_mf.append(scene_mf_nums)
            elif self.mf_to_sf:
                for view_id, views in tmp_infos_mf.items():
                    cur_total_datas.append(views)

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
            elif self.mf_sampling_strategy is not None and self.mf_sampling_strategy != "all":
                # NOTE: mf_to_sf
                if isinstance(self.mf_sampling_strategy, str):
                    sampling_number = self._parse_sampling_strategy(
                        self.mf_sampling_strategy, len(cur_total_datas)
                    )
                    if self.mf_sampling_strategy.startswith("first"):
                        total_datas += [view for views in cur_total_datas for view in views[:sampling_number]]
                    elif self.mf_sampling_strategy.startswith("end"):
                        total_datas += [view for views in cur_total_datas for view in views[-sampling_number:]]
                    elif self.mf_sampling_strategy.startswith("index"):
                        total_datas += [view for views in cur_total_datas for view in views[sampling_number : sampling_number + 1]]
                elif isinstance(self.mf_sampling_strategy, (list, tuple)):
                    total_datas += [views[i] for views in cur_total_datas for i in self.mf_sampling_strategy if i < len(views)]
                else:
                    raise NotImplementedError(
                        f"mf_sampling_strategy {self.mf_sampling_strategy} is not implemented"
                    )
            else:
                total_datas += [view for views in cur_total_datas for view in views]

        return total_datas
