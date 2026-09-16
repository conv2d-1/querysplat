import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch

from hAlgorithm.datasets.base_dataset import BaseDataset
from hAlgorithm.utils.mae_mask import decoder_mask_map_generator, encoder_mask_map_generator


class MAEDataset(BaseDataset):
    def __init__(
        self,
        patch_size: int,
        mask_ratio: float,
        decoder_mask_ratio: float = 0.5,
        mask_type: str = "tube",
        decoder_mask_type: str = "run_cell",
        name: str = "mae",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.patch_size = patch_size

        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

        self.decoder_mask_ratio = decoder_mask_ratio
        self.decoder_mask_type = decoder_mask_type

    def get_data_for_trainval(
        self, idx, data_info=None, data_batch=None, transform_info=None, mf_debug=False
    ):
        data_dict = super().get_data_for_trainval(
            idx,
            data_info=data_info,
            data_batch=data_batch,
            transform_info=transform_info,
            mf_debug=mf_debug,
        )

        image = data_dict["image"]
        input_size = (1, image.shape[-2] // self.patch_size, image.shape[-1] // self.patch_size)

        encoder_mask = encoder_mask_map_generator(
            input_size, mask_ratio=self.mask_ratio, mask_type=self.mask_type
        )
        data_dict["encoder_mask"] = torch.from_numpy(encoder_mask[0])

        if self.decoder_mask_ratio > 0.0:
            decoder_mask = decoder_mask_map_generator(
                input_size, mask_ratio=self.decoder_mask_ratio, mask_type=self.decoder_mask_type
            )
            data_dict["decoder_mask"] = torch.from_numpy(decoder_mask[:, 0])

        return data_dict


if __name__ == "__main__":

    dataset = MAEDataset(
        phase="train",
        name="void",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/VOID/test.json",
        sampling_strategy="random:10",
        train_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
            dict(type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip", prob=0.5),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
            # x / 255 * 2 - 1
        ],
        test_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        debug=True,
    )

    from hAlgorithm.utils.vis_util import visualize_batch

    for index in range(3):
        print(index)
        batch = dataset.__getitem__(index)
        visualize_batch(batch)
