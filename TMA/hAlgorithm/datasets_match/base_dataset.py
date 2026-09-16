import os
import sys

sys.path.append(os.getcwd())

import logging
import random

import numpy as np
import torch

from hAlgorithm.datasets_mv.base_dataset import BaseDatasetMV
from hAlgorithm.modules.models2.external.xfeat.dataset.megadepth_warper import create_meshgrid, warp_kpts


class MatchingDataset(BaseDatasetMV):
    def __init__(self, pair_mode=None, downsample=8, **kwargs):
        self.pair_mode = pair_mode
        self.downsample = int(downsample)

        super(MatchingDataset, self).__init__(**kwargs)

    def load_mf_data(self, idx):
        seq_list = self.clip_sampler(self.data_infos[idx])

        if self.pair_mode == "first":
            seq_list = [seq_list[0], seq_list[1]]
        elif self.pair_mode == "max":
            seq_list = [seq_list[0], seq_list[-1]]
        elif self.pair_mode == "random":
            idx1, idx2 = random.sample(range(len(seq_list)), 2)
            seq_list = [seq_list[idx1], seq_list[idx2]]
        elif self.pair_mode is None:
            # NOTE: 1 match n-1
            pass
        else:
            raise NotImplementedError

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
        ) = (
            data_batch["curr_rgb"],
            data_batch["curr_intrinsics"],
            data_batch["curr_extrinsics"],
            data_batch["curr_depth"],
            data_batch["curr_depth_mask"],
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
            _,
            _,
            _,
            _,
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
        intrinsics_mat_raw = self.create_intrinsics_matrix(curr_intrinsics) if curr_intrinsics is not None else None

        intrinsics_mat = self.create_intrinsics_matrix(intrinsics) if intrinsics is not None else None
        extrinsics_mat = self.create_extrinsics_matrix(curr_extrinsics) if curr_extrinsics is not None else None

        # Construct the final data dictionary
        data_dict = {
            "image": image,
            "intrinsics_mat_raw": intrinsics_mat_raw,
            "intrinsics": intrinsics_mat,
            "extrinsics": extrinsics_mat,
        }

        # Add raw depth and mask if available
        if curr_depth is not None:
            data_dict["depth_raw"] = torch.from_numpy(curr_depth).float()
            data_dict["depth_raw_mask"] = torch.from_numpy(curr_depth_mask).bool()

        if "image_show" in transform_info and self.phase != "train":
            image_show = transform_info["image_show"]
            if not isinstance(image_show, torch.Tensor):
                image_show = torch.from_numpy(transform_info["image_show"]).float()
            data_dict["image_show"] = image_show

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
        mf_data_batch = super().get_mf_data_for_trainval(idx)

        if "depth_raw" in mf_data_batch:
            mf_data_batch = self.spvs_coarse(idx=idx, data=mf_data_batch, scale=self.downsample)

        keys = set(list(mf_data_batch.keys())) - set(["meta_data", "image", "lut_mat21", "intrinsics", "extrinsics", "extrinsics_reff", "image_show"])
        for key in keys:
            mf_data_batch.pop(key)

        return mf_data_batch

    def spvs_coarse(self, idx, data, scale=8):
        """Supervise correspondences with dense depth & camera poses."""

        # 1. misc
        device = data["image"].device
        N, _, H, W = data["image"].shape

        # read intrinsics of original size
        K0 = data["intrinsics_mat_raw"][0:1].clone().repeat(N - 1, 1, 1)
        K1 = data["intrinsics_mat_raw"][1:].clone()

        # read and compute relative poses
        T0 = data["extrinsics"][0:1].clone().repeat(N - 1, 1, 1)
        T1 = data["extrinsics"][1:].clone()
        T_0to1 = T1 @ T0.inverse()
        T_1to0 = T_0to1.inverse()

        depth0 = data["depth_raw"][0:1].clone().repeat(N - 1, 1, 1)
        depth1 = data["depth_raw"][1:].clone()

        depth0_mask = data["depth_raw_mask"][0:1].clone().repeat(N - 1, 1, 1)
        depth1_mask = data["depth_raw_mask"][1:].clone()

        # scale = 8
        origin_width = data["meta_data"]["origin_width"].item()
        input_width = data["meta_data"]["input_width"].item()
        origin_height = data["meta_data"]["origin_height"].item()
        input_height = data["meta_data"]["input_height"].item()
        image_scale = torch.tensor([origin_width / input_width, origin_height / input_height], dtype=torch.float).unsqueeze(0).unsqueeze(1).repeat(N - 1, 1, 1)

        scale0 = scale1 = image_scale
        h0, w0, h1, w1 = map(lambda x: x // scale, [H, W, H, W])

        # 2. warp grids
        # create kpts in meshgrid and resize them to image resolution
        # NOTE: original size
        grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(N - 1, 1, 1)  # [N - 1, hw, 2]
        grid_pt1_i = grid_pt1_c * scale1 * scale

        # warp kpts bi-directionally and check reproj error
        valid_m1 = torch.stack([depth1_mask[i, grid_pt1_i[i, :, 1].long(), grid_pt1_i[i, :, 0].long()] for i in range(N - 1)], dim=0)
        grid_pt1_i = grid_pt1_i * valid_m1.float().unsqueeze(-1)
        nonzero_m1, w_pt1_i = warp_kpts(grid_pt1_i, depth1, depth0, T_1to0[:, :3, :4], K1, K0)

        w_pt1_i = w_pt1_i * nonzero_m1.float().unsqueeze(-1)
        valid_m2 = torch.stack([depth0_mask[i, w_pt1_i[i, :, 1].long(), w_pt1_i[i, :, 0].long()] for i in range(N - 1)], dim=0)
        w_pt1_i = w_pt1_i * valid_m2.float().unsqueeze(-1)
        nonzero_m2, w_pt1_og = warp_kpts(w_pt1_i, depth0, depth1, T_0to1[:, :3, :4], K0, K1)

        dist = torch.linalg.norm(grid_pt1_i - w_pt1_og, dim=-1)
        mask_mutual = (dist < 1.5) & nonzero_m1 & nonzero_m2

        # NOTE: input size
        batched_corrs = [torch.cat([w_pt1_i[i, mask_mutual[i]] / scale0[i], grid_pt1_i[i, mask_mutual[i]] / scale1[i]], dim=-1) for i in range(len(mask_mutual))]

        # Remove repeated correspondences - this is important for network convergence
        corrs = []
        lut_mat21_list = []
        for pts in batched_corrs:
            lut_mat12 = torch.ones((h1, w1, 4), device=device, dtype=torch.float32) * -1
            lut_mat21 = torch.clone(lut_mat12)
            src_pts = pts[:, :2] / scale
            tgt_pts = pts[:, 2:] / scale

            src_pts_long = src_pts.long()
            mask = (src_pts_long[:, 0] < w1) & (src_pts_long[:, 1] < h1)
            src_pts = src_pts[mask]
            tgt_pts = tgt_pts[mask]
            src_pts_long = src_pts_long[mask]

            lut_mat12[src_pts_long[:, 1], src_pts_long[:, 0]] = torch.cat([src_pts, tgt_pts], dim=1)
            mask_valid12 = torch.all(lut_mat12 >= 0, dim=-1)
            points = lut_mat12[mask_valid12]

            # Target-src check
            src_pts, tgt_pts = points[:, :2], points[:, 2:]
            lut_mat21[tgt_pts[:, 1].long(), tgt_pts[:, 0].long()] = torch.cat([src_pts, tgt_pts], dim=1)

            if self.debug:
                mask_valid21 = torch.all(lut_mat21 >= 0, dim=-1)
                points = lut_mat21[mask_valid21]
                corrs.append(points)

            lut_mat21_list.append(lut_mat21)

        data["lut_mat21"] = torch.stack(lut_mat21_list, dim=0)

        # Plot for debug purposes
        if self.debug:
            outdir = os.path.join("./debug/dataset/", self.name, "matching", f"{idx:04d}")
            os.makedirs(outdir, exist_ok=True)

            from hAlgorithm.modules.models2.external.xfeat.training.utils import plot_corrs

            for i in range(len(corrs)):
                path = os.path.join(outdir, f"{i:02d}.png")
                print(f"{i}, nums {len(corrs[i])}, {path}")
                plot_corrs(data["image"][0], data["image"][i + 1], corrs[i][:, :2] * scale, corrs[i][:, 2:] * scale, path=path)

        return data

    def normalize_cameras_base_first(self, datas):
        """
        Normalize all camera extrinsics relative to the first camera.
        Args:
            datas (list): A list of dictionaries containing pointmap and extrinsics information.
        """

        # Get the inverse of the first camera's extrinsics
        if "extrinsics" in datas[0]:
            first_extrinsics_inv = datas[0]["extrinsics"].inverse()

            for data in datas:
                # Compute new extrinsics relative to the first camera
                data["extrinsics_reff"] = data["extrinsics"] @ first_extrinsics_inv


if __name__ == "__main__":

    import random
    import sys

    import yaml

    from hAlgorithm.utils import instantiate_from_config

    # path = "hAlgorithm/configs/mv_v1.0/dataset_configs_matching/train_251106.yaml"
    # path = "hAlgorithm/configs/mv_v1.0/dataset_configs_matching/debug_251106.yaml"
    # path = "hAlgorithm/configs/mv_v1.0/dataset_configs_matching/test_251106.yaml"
    path = "hAlgorithm/configs/mv_v1.0/dataset_configs_matching/vis_251106.yaml"

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
                    type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                    max_size=518,
                    patch_size=32,
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
        config["clip_sampler"]["shuffle"] = False

        dataset = instantiate_from_config(config)
        print(config["name"], len(dataset))

        # indices = random.sample(range(len(dataset)), k=min(2, len(dataset)))
        # # indices = [0, 1, 2]

        # try:
        #     for index in indices:
        #         print(index)
        #         dataset.__getitem__(index)
        # except Exception as e:
        #     print(e)
