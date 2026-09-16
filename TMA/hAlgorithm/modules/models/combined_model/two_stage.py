import logging

import torch
import torch.nn as nn

from hAlgorithm.utils import instantiate_from_config


class TwoStageModel(nn.Module):
    def __init__(self, stage1, stage2, stage1_pretrain=None, stage2_pretrain=None):
        super(TwoStageModel, self).__init__()

        stage1["return_features"] = True
        self.stage1 = instantiate_from_config(stage1)
        self.module_names.append("stage1")

        self.stage2 = instantiate_from_config(stage2)
        self.module_names.append("stage2")

        if stage1_pretrain is not None:
            self.stage1.load_state_dict(
                torch.load(stage1_pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"TwoStageModel, stage1 load pretrain {stage1_pretrain}")

        if stage2_pretrain is not None:
            self.stage2.load_state_dict(
                torch.load(stage2_pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"TwoStageModel, stage2 load pretrain {stage2_pretrain}")

    def forward(self, rgb, **kwargs):
        features_dict, results = self.stage1(rgb, **kwargs)
        results = self.stage2(rgb, features_dict, results, **kwargs)
        return results
