import os
import sys
import torch

from hAlgorithm.datasets.base_dataset import BaseDataset

class ImageDataset(BaseDataset):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

    def get_data_for_trainval(
        self, idx, data_info=None, data_batch=None, transform_info=None, mf_debug=False
    ):
        if data_info is None or data_batch is None:
            if isinstance(idx, (list, tuple)):
                idx, others = idx
                self.update_data_transforms(**others)

            # Load the data information and batch for the given index
            data_info = self.data_infos[idx]
            data_batch = self.load_data(data_info)

        if self.debug:
            self.info_rbg_path(data_info)
        (
            curr_rgb,
            curr_intrinsics,
            curr_extrinsics,
        ) = (
            data_batch["curr_rgb"],
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
            intrinsics=curr_intrinsics.copy() if curr_intrinsics is not None else None,  # NOTE
            depth=None,
            depth_mask=None,
            normal=None,
            other_labels=[],
            transform_info=transform_info,
        )
        # Create the intrinsics and extrinsics matrix from the parameters
        intrinsics_mat = (
            self.create_intrinsics_matrix(intrinsics) if intrinsics is not None else None
        )
        extrinsics_mat = (
            self.create_extrinsics_matrix(curr_extrinsics) if curr_extrinsics is not None else None
        )
        # Construct the final data dictionary
        data_dict = {
            "image": image,
            "intrinsics": intrinsics_mat,
            "extrinsics": extrinsics_mat,
        }
        
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
        if "rgb" not in data_dict["meta_data"]["data_info"]:
            if self.rbg_name in data_dict["meta_data"]["data_info"]:
                data_dict["meta_data"]["data_info"]["rgb"] = data_dict["meta_data"]["data_info"][
                    self.rbg_name
                ]
            else:
                data_dict["meta_data"]["data_info"]["rgb"] = data_dict["meta_data"]["data_info"][
                    "hdf5"
                ]
        if "depth_scale" not in data_dict["meta_data"]["data_info"]:
            data_dict["meta_data"]["data_info"]["depth_scale"] = 1.0
        # Remove entries with None values to clean up the dictionary
        data_dict = {k: v for k, v in data_dict.items() if v is not None}
        return data_dict