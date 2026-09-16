import os
import sys

sys.path.append(os.getcwd())

import torch

from hAlgorithm.datasets_mv.base_dataset import BaseDatasetMV
from hAlgorithm.utils.mae_mask import decoder_mask_map_generator, encoder_mask_map_generator


class MAEDatasetMV(BaseDatasetMV):
    def __init__(
        self,
        tubelet_size: int,
        patch_size: int,
        mask_ratio: float,
        decoder_mask_ratio: float = 0.5,
        mask_type: str = "tube",
        decoder_mask_type: str = "run_cell",
        name: str = "mae",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.tubelet_size = tubelet_size
        self.patch_size = patch_size

        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

        self.decoder_mask_ratio = decoder_mask_ratio
        self.decoder_mask_type = decoder_mask_type

        if self.phase != "train":
            self.clip_sampler.seed = 0

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
        input_size = (
            len(image) // self.tubelet_size,
            image.shape[-2] // self.patch_size,
            image.shape[-1] // self.patch_size,
        )

        encoder_mask = encoder_mask_map_generator(
            input_size, mask_ratio=self.mask_ratio, mask_type=self.mask_type
        )
        data_dict["encoder_mask"] = torch.from_numpy(encoder_mask)

        if self.decoder_mask_ratio > 0.0:
            decoder_mask = decoder_mask_map_generator(
                input_size, mask_ratio=self.decoder_mask_ratio, mask_type=self.decoder_mask_type
            )
            data_dict["decoder_mask"] = torch.from_numpy(decoder_mask)

        return data_dict
