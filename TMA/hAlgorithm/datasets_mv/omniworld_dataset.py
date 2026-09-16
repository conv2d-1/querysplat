import os
import sys

sys.path.append(os.getcwd())

import imageio
import numpy as np
import torch
from hAlgorithm.datasets_mv.base_dataset import BaseDatasetMV


class OmniWorldDatasetMV(BaseDatasetMV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.min_depth = 0.0001

    def load_depth(self, depth_path, image, const, dtype):
        """
        Returns
        -------
        depthmap : (H, W) float32
        valid   : (H, W) bool      True for reliable pixels
        """

        depthmap = imageio.v2.imread(depth_path).astype(np.float32) / 65535.0
        near_mask = depthmap < 0.0015   # 1. too close
        far_mask = depthmap > (65500.0 / 65535.0) # 2. filter sky
        far_mask = depthmap > np.percentile(depthmap[~far_mask], 95) # 3. filter far area (optional)
        near, far = 1., 1000.
        depthmap = depthmap / (far - depthmap * (far - near)) / 0.004
        # depthmap = (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min()) * 10 + 1.0
        valid = ~(near_mask | far_mask)
        depthmap[~valid] = 0

        return depthmap #, valid

    def point_normalize(self, sparse_pointmap, sparse_pointmap_mask):
        return super().point_normalize(sparse_pointmap, sparse_pointmap_mask, min_range=None)


if __name__ == "__main__":
    import yaml
    from hAlgorithm.utils import instantiate_from_config

    path = "hAlgorithm/configs/mv_v1.0/dataset_configs_5/vis_260301_mv.yaml"

    with open(path, "r") as file:
        yaml_data = yaml.safe_load(file)
    dataset_configs = yaml_data.get("datasets", [])

    for config in dataset_configs:
        if config["name"] != "omniworld":
            continue

        base_config = dict(
            phase="test",
            seed=0,
            test_transforms=[
                dict(
                    type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                    max_size=504,
                    patch_size=14,
                ),
                dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
                dict(
                    type="hAlgorithm.datasets.transforms.transforms.Normalize",
                    mean=[127.5, 127.5, 127.5],
                    std=[127.5, 127.5, 127.5],
                    ),
            ],
            min_depth=1e-4,
            max_depth=1000.0,
            sparse_pattern=None,
            debug=True,
            only_pointmap=False,
            with_pointmap=True,
            sampling_strategy="all",
            mf_scene_sampling_strategy="all",
        )
        config.update(base_config)
        # config["clip_sampler"]["shuffle"] = False

        dataset = instantiate_from_config(config)
        print(config["name"], len(dataset))
        indices = [4600]
        # indices = range(0, len(dataset), 100)

        # nums = [len(infos) for infos in dataset.data_infos]
        
        for index in indices:
            print(index)
            dataset.__getitem__(index)