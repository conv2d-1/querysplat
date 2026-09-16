import os
import sys

sys.path.append(os.getcwd())

import json
import logging
import random

import numpy as np
import torch
from collections import defaultdict
from hAlgorithm.datasets_match.base_dataset import MatchingDataset
import h5py

import torch.nn.functional as F
from hAlgorithm.modules.models2.external.romav2.utils.utils import get_gt_warp as romav1_depth_to_warp
from hAlgorithm.modules.models2.external.romav2.utils.utils import tensor_to_pil
from hAlgorithm.datasets.transforms.transforms import resize_depth_preserve, ResizePatch
from hAlgorithm.datasets.transforms.edge_filter import edge_filter
from tqdm import tqdm

class PairMatchingDataset(MatchingDataset):
    def __init__(
        self, 
        pair_path : str = None, 
        overlap_path : str = None,
        overlap_conditions : list [tuple] = None,
        scene_pair_sampling_strategy : str | list = None,
        pair_sampling_strategy : str = None, 
        mvs_mode : bool = False,
        match_scales : list = [1],
        with_warp : bool = True,
        **kwargs
    ):
        self.pair_path = pair_path
        self.overlap_path = overlap_path
        self.overlap_conditions = overlap_conditions

        self.pair_sampling_strategy = pair_sampling_strategy
        self.scene_pair_sampling_strategy = scene_pair_sampling_strategy
        kwargs["clip_sampler"] = None
        super(PairMatchingDataset, self).__init__(**kwargs)
        self.match_scales = match_scales
        self.with_warp = with_warp
        
        # Fixed resize method for MVS dataset. They have different image sizes
        if mvs_mode:
            new_transforms = []
            for transform in self.data_transforms.transforms:
                cls_name = transform.__class__.__name__
                if cls_name in [
                    "Resize",
                    "ResizeSR",
                    "ResizeKeepRatio",
                    "ResizePatch",
                ]:
                    is_lidar = getattr(transform, "is_lidar", False)
                    patch_size = getattr(transform, "patch_size", None)
                    if cls_name in ["Resize", "ResizeSR", "ResizePatch"]:
                        width = transform.width
                        height = transform.height
                    elif cls_name in ["ResizeKeepRatio"]:
                        width = transform.max_size
                        height = transform.max_size
                    logging.info(f"MVS Mode set to size w={width}, h={height}")
                    new_transforms.append(ResizePatch(width=width, height=height, low_resolution=False, is_lidar=is_lidar, patch_size=patch_size))
                else:
                    new_transforms.append(transform)
            self.data_transforms.transforms = new_transforms
            self.with_depth_raw = False

    def get_data_infos(self):
        super().get_data_infos()
        
        scene_frame_view_to_idx = defaultdict(lambda: defaultdict(dict))
        for i, info in enumerate(self.data_infos):
            info = info[0]
            scene_frame_view_to_idx[info["scene"]][info["frame_id"]][info["view_id"]] = i
        
        # build data pairs
        data_pairs_list = []
        if self.pair_path is not None:
            if isinstance(self.pair_path, str):
                data_pairs_list.append(self.pair_path)
            elif isinstance(self.pair_path, (list, tuple)):
                for file in self.pair_path:
                    data_pairs_list.append(file)
            else:
                raise NotImplementedError()

        generated_pairs_list = []
        if self.overlap_path is not None:
            assert self.overlap_conditions is not None
            if self.overlap_path.endswith(".h5") or self.overlap_path.endswith(".hdf5"):
                with h5py.File(self.overlap_path, 'r') as f:
                    scene_names = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
                    total_candidates = 0
                    cur_generated_pairs = {}
                    for scene in tqdm(scene_names, total=len(scene_names), desc="Generating Pairs"):
                        scene_pairs = []
                        overlap_matrix = f[scene]["overlap_matrix"][:]
                        frameview_idx = f[scene]["frameview_idx"][:]
                        total_candidates += int(frameview_idx.shape[0]*frameview_idx.shape[0])
                        for min_overlap, max_overlap in self.overlap_conditions:
                            valid_overlap = np.logical_and(overlap_matrix > min_overlap, overlap_matrix < max_overlap)
                            img0s, img1s = np.where(valid_overlap)
                            for img0, img1 in zip(img0s, img1s):
                                img0_info = tuple(frameview_idx[img0])
                                img1_info = tuple(frameview_idx[img1])
                                scene_pairs.append((img0_info, img1_info))
                        cur_generated_pairs[scene] = scene_pairs
                generated_pairs_list.append(cur_generated_pairs)
                if self.debug:
                    pairs_per_scene = [len(p) for p in cur_generated_pairs.values()]
                    print(f"Total {sum(pairs_per_scene)} Pairs From {total_candidates} Candidates. Mean {int(sum(pairs_per_scene)/len(scene_names))} Max {max(pairs_per_scene)} Min {min(pairs_per_scene)}")
            else:
                raise NotImplementedError(f"Only supports h5 data for overlap")

        def pair_to_idx(scene, f, v):
            if scene in scene_frame_view_to_idx:
                if f in scene_frame_view_to_idx[scene]:
                    if v in scene_frame_view_to_idx[scene][f]:
                        return scene_frame_view_to_idx[scene][f][v]
            return None

        total_pairs_datas = [] # list of pairs dict , containing scene: {[[(f_id, v_id), (f_id, v_id)], ...]}
        for pair_file in data_pairs_list:
            if pair_file.endswith(".json"):
                with open(pair_file, 'r') as f:
                    cur_pairs = json.load(f)["image_pairs"]
                    total_pairs_datas.append(cur_pairs)
        total_pairs_datas.extend(generated_pairs_list)

        total_image_pairs = []
        for cur_pairs in total_pairs_datas:
            for scene, pairs in cur_pairs.items():
                if (
                    self.mf_scene_delete is not None
                    and len(self.mf_scene_delete) > 0
                    and scene in self.mf_scene_delete
                ):
                    if self.debug:
                        print(f"Skipping deleted scene {scene}")
                    continue
                scene_pairs = []
                for pair in pairs:
                    (f0, v0), (f1, v1) = pair
                    id0 = pair_to_idx(scene, f0, v0)
                    id1 = pair_to_idx(scene, f1, v1)
                    if id0 is not None and id1 is not None:
                        scene_pairs.append((id0, id1))
                #sampling scene pairs
                if self.scene_pair_sampling_strategy is not None:
                    if isinstance(self.scene_pair_sampling_strategy, (list, tuple)):
                        scene_pairs = [scene_pairs[sam_i] for sam_i in self.scene_pair_sampling_strategy if sam_i < len(scene_pairs)]
                    else:
                        pair_sampling_number = self._parse_sampling_strategy(self.scene_pair_sampling_strategy, len(scene_pairs))
                        if pair_sampling_number is not None:
                            if self.scene_pair_sampling_strategy.startswith("first"):
                                scene_pairs = scene_pairs[:pair_sampling_number]
                            elif self.scene_pair_sampling_strategy.startswith("end"):
                                scene_pairs = scene_pairs[-pair_sampling_number:]
                total_image_pairs.extend(scene_pairs)

        if self.pair_sampling_strategy is not None:
            pair_sampling_number = self._parse_sampling_strategy(self.pair_sampling_strategy, len(total_image_pairs))
            if pair_sampling_number is not None:
                if self.pair_sampling_strategy.startswith("first"):
                    total_image_pairs = total_image_pairs[:pair_sampling_number]
                elif self.pair_sampling_strategy.startswith("end"):
                    total_image_pairs = total_image_pairs[-pair_sampling_number:]
        self.data_pairs = total_image_pairs

    def __len__(self):
        return len(self.data_pairs)

    def load_depth(self, depth_path, image, const, dtype):
        if depth_path.endswith(".hdf5") or depth_path.endswith(".h5"):
            # Open the HDF5 file and read the dataset.
            return np.array(h5py.File(depth_path, "r")["depth"], dtype=dtype)
        else:
            return self.read_file(depth_path, image=image, const=const, dtype=dtype)

    def load_mf_data(self, idx):
        seq_list = [self.data_infos[pair][0] for pair in self.data_pairs[idx]]

        mf_data, mf_info = [], []
        for view in seq_list:
            single_frame_data = self.load_data(view)
            if self.check_data_extrinsics(single_frame_data):
                mf_data.append(single_frame_data)
                mf_info.append(view)
            else:
                logging.info(f"Dataset:{self.name} invalid extrinsics found.")
                return None

        return dict(data=mf_data, info=mf_info)

    def get_data_for_trainval(self, idx, data_info, data_batch, transform_info=None, mf_debug=False, **kwargs):
        if self.debug:
            self.info_rbg_path(data_info)

        (
            curr_rgb,
            curr_intrinsics,
            curr_extrinsics,
            curr_depth,
            curr_depth_mask,
            curr_prompt,
        ) = (
            data_batch["curr_rgb"],
            data_batch["curr_intrinsics"],
            data_batch["curr_extrinsics"],
            data_batch["curr_depth"],
            data_batch["curr_depth_mask"],
            data_batch["curr_prompt"],
        )
        curr_depth, curr_depth_mask = self.depth_filter(curr_depth, depth_mask=curr_depth_mask, sem_label=None)
        if curr_depth_mask is not None and curr_depth_mask.sum() == 0:
            self.info_rbg_path(data_info)
            logging.warning("depth_mask is empty, no valid points found.")
            raise ValueError("depth_mask is empty, no valid points found.")

        transform_info = transform_info if transform_info is not None else dict()
        transform_info["name"] = self.name

        # Apply data augmentation transforms
        (
            image,
            intrinsics,
            depth,
            depth_mask,
            _,
            _,
            transform_info,
        ) = self.data_transforms(
            image=curr_rgb,
            intrinsics=curr_intrinsics.copy() if curr_intrinsics is not None else None,
            depth=curr_depth,
            depth_mask=curr_depth_mask,
            normal=None,
            other_labels=[],
            transform_info=transform_info,
        )

        # Create the intrinsics and extrinsics matrix from the parameters
        intrinsics_mat = self.create_intrinsics_matrix(intrinsics) if intrinsics is not None else None
        extrinsics_mat = self.create_extrinsics_matrix(curr_extrinsics) if curr_extrinsics is not None else None

        if self.with_depth_raw and curr_depth is not None:
            h, w = curr_depth.shape[-2:]
            if transform_info.get("horizontal_flip", False):
                curr_depth = np.fliplr(curr_depth).copy()
                curr_depth_mask = np.fliplr(curr_depth_mask).copy()
                curr_intrinsics = curr_intrinsics.copy()
                curr_intrinsics[2] = w - 1 - curr_intrinsics[2]
            if transform_info.get("vertical_flip", False):
                curr_depth = np.flipud(curr_depth).copy()
                curr_depth_mask = np.flipud(curr_depth_mask).copy()
                curr_intrinsics = curr_intrinsics.copy()
                curr_intrinsics[3] = h - 1 - curr_intrinsics[3]

        # Extracts sparse depth points and their corresponding masks from the origin depthmap
        if curr_prompt is not None:
            if curr_prompt.shape[0:2] != curr_rgb.shape[0:2]:
                curr_prompt = resize_depth_preserve(curr_prompt, curr_rgb.shape[0:2])
            curr_prompt_mask = (curr_prompt > self.min_depth) & (curr_prompt < self.max_depth)
            if (
                curr_depth_mask is not None
                and curr_prompt_mask.shape[0:2] == curr_depth_mask.shape[0:2]
            ):
                curr_prompt_mask = curr_prompt_mask & curr_depth_mask
            curr_prompt_mask = curr_prompt_mask.astype(curr_depth_mask.dtype)
            sparse_pointmap, sparse_pointmap_mask = self.get_sparse_depth_with_pattern(
                curr_rgb=curr_rgb,
                curr_depth=curr_prompt,
                curr_depth_mask=curr_prompt_mask,
                curr_intrinsics=curr_intrinsics,
                rgb=image,
                depth=depth,
                depth_mask=depth_mask,
                intrinsics_mat=intrinsics_mat,
                transform_info=transform_info,
                data_idx=idx,
            )
        elif curr_depth is not None:
            sparse_pointmap, sparse_pointmap_mask = self.get_sparse_depth_with_pattern(
                curr_rgb=curr_rgb,
                curr_depth=curr_depth,
                curr_depth_mask=curr_depth_mask,
                curr_intrinsics=curr_intrinsics,
                rgb=image,
                depth=depth,
                depth_mask=depth_mask,
                intrinsics_mat=intrinsics_mat,
                transform_info=transform_info,
                data_idx=idx,
            )
        else:
            sparse_pointmap = sparse_pointmap_mask = None
        # Process the depth data
        depth_output = self.process_depth(
            depth,
            depth_mask,
            intrinsics_mat,
            sparse_pointmap=sparse_pointmap,
            sparse_pointmap_mask=sparse_pointmap_mask,
        )
        if depth_output["sparse_pointmap_max_range"] is None and self.custom_scale is not None:
            depth_output["sparse_pointmap_max_range"] = torch.tensor(self.custom_scale).float()
        
        
        # Construct the final data dictionary
        data_dict = {
            "image": image,
            "intrinsics": intrinsics_mat,
            "extrinsics": extrinsics_mat,
            "depth": depth,
            "depth_mask": depth_mask,
            **depth_output,
        }

        if self.with_depth_raw and curr_depth is not None:
            data_dict["depth_raw"] = torch.from_numpy(np.ascontiguousarray(curr_depth)).float()
            data_dict["depth_raw_mask"] = torch.from_numpy(np.ascontiguousarray(curr_depth_mask)).bool()
            data_dict["intrinsics_raw"] = self.create_intrinsics_matrix(curr_intrinsics)

        if "image_show" in transform_info and self.phase != "train":
            image_show = transform_info["image_show"]
            if not isinstance(image_show, torch.Tensor):
                image_show = torch.from_numpy(transform_info["image_show"]).float()
            data_dict["image_show"] = image_show

        if "image_backup" in transform_info:
            data_dict["image_backup"] = transform_info["image_backup"]

            if self.with_rgb_edge_mask:
                tmp_data = data_dict["image_backup"].mean(0, keepdim=True)
                rgb_edge_mask = edge_filter(tmp_data, valid_mask=data_dict["depth_mask"], times=0.3)

                if data_dict.get("edge_mask", None) is not None:
                    data_dict["edge_mask"] = rgb_edge_mask | data_dict["edge_mask"]
                else:
                    data_dict["edge_mask"] = rgb_edge_mask

        curr_intrinsics_mat = self.create_intrinsics_matrix(curr_intrinsics)
        data_dict["intrinsics_raw"] = curr_intrinsics_mat

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

    def get_mf_data_for_trainval(self, idx):
        mf_data_batch = super(MatchingDataset, self).get_mf_data_for_trainval(idx)

        if ("depth" in mf_data_batch or "depth_raw" in mf_data_batch) and self.with_warp:
            mf_data_batch = self.get_gt_warp(idx=idx, data=mf_data_batch)

        kept_keys = {
            "meta_data", "image", "warp", "warp_mask", "intrinsics", "extrinsics", "extrinsics_reff", "image_show",
            "sparse_pointmap", "sparse_pointmap_max_range", "sparse_pointmap_mask", "depth", "pointmap", "depth_raw", "intrinsics_raw",
            "warp_scales", "warp_mask_scales",
        }
        kept_keys.update({f"warp_{scale}" for scale in self.match_scales})
        kept_keys.update({f"warp_mask_{scale}" for scale in self.match_scales})

        if self.with_depth_raw:
            kept_keys.update({"depth_raw", "depth_raw_mask"})

        if self.with_rgb_edge_mask or self.with_depth_edge_mask:
            kept_keys.add("edge_mask")

        keys = set(list(mf_data_batch.keys())) - kept_keys

        for key in keys:
            mf_data_batch.pop(key)

        for key, val in mf_data_batch.items():
            if isinstance(val, torch.Tensor):
                mf_data_batch[key] = val.contiguous()

        return mf_data_batch

    def get_gt_warp(self, idx, data):
        if isinstance(idx, (list, tuple)):
            idx = idx[0]
        N, _, H, W = data["image"].shape
        assert N == 2

        if "depth_raw" in data:
            depth1 = data["depth_raw"][0].clone()
            depth2 = data["depth_raw"][1].clone()
            K1 = data["intrinsics_raw"][0].clone()
            K2 = data["intrinsics_raw"][1].clone()
        else:
            depth1 = data["depth"][0].clone()
            depth2 = data["depth"][1].clone()
            K1 = data["intrinsics"][0].clone()
            K2 = data["intrinsics"][1].clone()
        if depth1.ndim == 2 or depth2.ndim == 2:
            depth1 = depth1[None]
            depth2 = depth2[None]

        # read and compute relative poses
        T1 = data["extrinsics"][0].clone()
        T2 = data["extrinsics"][1].clone()
        T_1to2 = (T2 @ T1.inverse())[None]

        warp_scales = {}
        warp_mask_scales = {}
        for scale in self.match_scales:
            h1, w1 = int(H / scale), int(W / scale)
            warp_s, warp_mask_s = romav1_depth_to_warp(
                depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode='bilinear',
                H=h1, W=w1,
            )
            warp_s = warp_s.to(depth1.dtype)
            warp_mask_s = warp_mask_s.float()
            warp_scales[scale] = warp_s
            warp_mask_scales[scale] = warp_mask_s

        data["warp_scales"] = warp_scales
        data["warp_mask_scales"] = warp_mask_scales
        data["meta_data"]["pair_idx"] = [(0, 1)]

        if 1 in warp_scales:
            data["warp"] = warp_scales[1]
            data["warp_mask"] = warp_mask_scales[1]
        
        # Plot for debug purposes
        if self.debug and True:
            warp = data.get("warp")
            warp_mask = data.get("warp_mask")
            if warp is None or warp_mask is None:
                return data

            device = data["image"].device
            x1 = data["image"][0]
            x2 = data["image"][1]

            outdir = os.path.join("./debug/dataset/", self.name, "warp", f"{idx:04d}")
            os.makedirs(outdir, exist_ok=True)
            warp_im = F.grid_sample(
                x2[None], warp, mode="bilinear", align_corners=False
            )[0]
            white_im = torch.ones((H, W), device=device)
            warp_vis = warp_mask * warp_im + (1 - warp_mask) * white_im
            vis_im = torch.cat([x1, warp_vis, x2], dim=2)
            path = os.path.join(outdir, f"warp_AB.png")
            tensor_to_pil(vis_im).save(path)
        return data

if __name__ == "__main__":

    import random
    import sys

    import yaml

    from hAlgorithm.utils import instantiate_from_config

    # path = "hAlgorithm/configs/mv_v1.0/dataset_configs_matching/train_251106.yaml"
    # path = "hAlgorithm/configs/mv_v1.0/dataset_configs_matching/debug_251106.yaml"
    # path = "hAlgorithm/configs/mv_v1.0/dataset_configs_matching/test_251106.yaml"
    # path = "hAlgorithm/configs/mv_v1.0/dataset_configs_matching/vis_251106.yaml"
    path = "hAlgorithm/configs/mv_v1.0/dataset_configs_matching_pair/train_260104.yaml"
    path = "hAlgorithm/configs/mv_v1.0/dataset_configs_matching_pair/test_260104.yaml"
    

    with open(path, "r") as file:
        yaml_data = yaml.safe_load(file)
    dataset_configs = yaml_data.get("datasets", [])

    for config in dataset_configs:
        # if config["name"] != "scpp":
        #     continue

        base_config = dict(
            phase="test",
            seed=0,
            test_transforms=[
                dict(
                    type="hAlgorithm.datasets.transforms.transforms.Resize",
                    width=672,
                    height=672,
                    # patch_size=32,
                    # is_lidar=True,
                ),
                dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
                dict(
                    type="hAlgorithm.datasets.transforms.transforms.Normalize",
                    mean=[0.0, 0.0, 0.0],
                    std=[255.0, 255.0, 255.0],
                ),
            ],
            min_depth=1e-3,
            max_depth=1000.0,
            sparse_pattern=None,
            mf_to_sf=False,
            debug=True,
        )
        config.update(base_config)
        # config["clip_sampler"]["shuffle"] = False

        dataset = instantiate_from_config(config)
        print(config["name"], len(dataset))
        data = dataset[0]
        indices = random.sample(range(len(dataset)), k=min(10, len(dataset)))

        # indices = random.sample(range(len(dataset)), k=min(2, len(dataset)))
        # # indices = [0, 1, 2]

        for index in indices:
            print(index)
            dataset.__getitem__(index)
        # try:
        #     for index in indices:
        #         print(index)
        #         dataset.__getitem__(index)
        # except Exception as e:
        #     print(e)
