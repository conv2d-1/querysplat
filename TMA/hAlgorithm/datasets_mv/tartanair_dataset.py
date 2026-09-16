import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from hAlgorithm.datasets_mv.base_dataset import BaseDatasetMV
from hAlgorithm.utils.sem_utils import SEM_LABEL, remap_sem_label


class TartanAirDatasetMV(BaseDatasetMV):
    semantic_labels = dict(sky=-1)

    scene_sky_labels = dict(
        # abandonedfactory_night_Easy=196,
        # abandonedfactory_night_Hard=196,
        # abandonedfactory_Easy=192,
        # abandonedfactory_Hard=192,
        abandonedfactory=196,
        amusement=182,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.depth_scale = 1.0

    def remap_sem_label(self, curr_sem, data_info):
        scene = data_info["scene"]
        semantic_labels = {}
        for scene_k, sky_label in self.scene_sky_labels.items():
            if scene_k in scene:
                semantic_labels["sky"] = sky_label
        return remap_sem_label(curr_sem, semantic_labels)

    def load_inf_mask(self, curr_depth, curr_sem, curr_rgb, data_info, **kwargs):
        scene = data_info["scene"]
        semantic_labels = {}
        for scene_k, sky_label in self.scene_sky_labels.items():
            if scene_k in scene:
                semantic_labels["sky"] = sky_label

        curr_inf_mask = np.zeros(curr_rgb.shape[:2], dtype=np.bool_)
        if curr_depth is not None:
            if self.inf_mask_min is not None:
                curr_inf_mask = np.logical_or(curr_inf_mask, curr_depth < self.inf_mask_min)
            if self.inf_mask_max is not None:
                curr_inf_mask = np.logical_or(curr_inf_mask, curr_depth > self.inf_mask_max)
            if self.inf_mask_depth_nan:
                depth_mask = np.logical_or(np.isnan(curr_depth), np.isinf(curr_depth))
                curr_inf_mask = np.logical_or(curr_inf_mask, depth_mask)
        if self.inf_mask_sem_names is not None and len(self.inf_mask_sem_names) > 0 and curr_sem is not None and "sky" in semantic_labels:
            sem_mask = self.process_sem_mask(curr_sem, self.inf_mask_sem_names, semantic_labels)
            curr_inf_mask = np.logical_or(curr_inf_mask, sem_mask)
        return curr_inf_mask
