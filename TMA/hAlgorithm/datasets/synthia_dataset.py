import os
import sys

sys.path.append(os.getcwd())

import cv2
import h5py
import numpy as np
from PIL import Image

from hAlgorithm.datasets.base_dataset import BaseDataset


class SynthiaDataset(BaseDataset):
    semantic_labels = dict(
        sky=(11, 0, 0, 255),
    )

    def __init__(
        self,
        name: str = "synthia",
        sem_name: str = "sem",
        depth_invalid_sem_names: list = ["sky"],
        depth_scale: float = 1.0,
        min_depth: float = 1e-5,
        max_depth: float = 100.0,
        **kwargs,
    ):
        super().__init__(
            name=name,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            sem_name=sem_name,
            depth_invalid_sem_names=depth_invalid_sem_names,
            **kwargs,
        )

    def load_depth(self, depth_path, image, const, dtype):
        # Return None if both file_path and image are None.
        if depth_path is None and image is None:
            return None

        # Create a constant-valued array based on the image dimensions if file_path is None.
        if depth_path is None or not os.path.exists(depth_path):
            data = np.zeros(image.shape, dtype=dtype) + const
        else:
            image_fp = cv2.imread(depth_path).astype(np.float32)
            data = (
                5000
                * (image_fp[..., 2] + image_fp[..., 1] * 256 + image_fp[..., 0] * 256 * 256)
                / (256 * 256 * 256 - 1)
            )
        return data
