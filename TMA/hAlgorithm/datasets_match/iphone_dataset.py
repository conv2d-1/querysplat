import os
import sys

sys.path.append(os.getcwd())

import numpy as np

from hAlgorithm.datasets.transforms.transforms import resize_depth_preserve
from hAlgorithm.datasets_match.lidar_depth_dataset import LidarDepthDataset


class IphoneDataset(LidarDepthDataset):

    def __init__(self, confidence=2, confidence_name="confidence", delete_camera=False, **kwargs):
        super().__init__(**kwargs)

        self.confidence = confidence
        self.confidence_name = confidence_name

        self.delete_camera = delete_camera

    def load_data(self, data_info):
        data_batch = super().load_data(data_info=data_info)

        if self.confidence_name in data_info:
            curr_confidence_path = os.path.join(self.depth_root, data_info[self.confidence_name])
            confidence = self.read_file(curr_confidence_path, image=None, const=0, dtype=np.float32)
            curr_depth_mask = data_batch["curr_depth_mask"]

            if confidence.shape[0:2] != curr_depth_mask.shape[0:2]:
                confidence = resize_depth_preserve(confidence, curr_depth_mask.shape[0:2])

            confidence_mask = confidence >= self.confidence
            curr_depth_mask = (curr_depth_mask * confidence_mask).astype(int)
            data_batch["curr_depth_mask"] = curr_depth_mask

        curr_rgb = data_batch["curr_rgb"]
        curr_depth = data_batch["curr_depth"]
        curr_depth_mask = data_batch["curr_depth_mask"]
        curr_prompt = data_batch["curr_prompt"]

        if curr_depth is not None and curr_depth.shape[:2] != curr_rgb.shape[:2]:
            data_batch["curr_depth"] = resize_depth_preserve(curr_depth, curr_rgb.shape[0:2])
        if curr_depth_mask is not None and curr_depth_mask.shape[:2] != curr_rgb.shape[:2]:
            curr_depth_mask = resize_depth_preserve(curr_depth_mask, curr_rgb.shape[0:2])
            data_batch["curr_depth_mask"] = curr_depth_mask.astype(int)
        if curr_prompt is not None and curr_prompt.shape[:2] != curr_rgb.shape[:2]:
            data_batch["curr_prompt"] = resize_depth_preserve(curr_prompt, curr_rgb.shape[0:2])

        return data_batch

    def get_mf_data_for_trainval(self, idx):
        data = super().get_mf_data_for_trainval(idx)
        if self.delete_camera:
            data.pop("intrinsics", None)
            data.pop("extrinsics", None)
            data.pop("extrinsics_reff", None)
        return data
