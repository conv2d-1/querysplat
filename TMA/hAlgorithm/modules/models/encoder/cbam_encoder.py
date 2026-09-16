import torch
import torch.nn as nn
from hAlgorithm.modules.models.cbam.cbam import CBAM


class CBAMModule(nn.Module):
    def __init__(self, layer_num, use_dims, input_channel=None, spatial_config=None):
        super().__init__()
        self.layer_num = layer_num
        self.use_dims = use_dims
        self.input_channel = len(self.use_dims) if self.use_dims is not None else input_channel
        self.spatial_config = spatial_config

        layers = []
        for _ in range(self.layer_num):
            layers.append(
                CBAM(
                    gate_channels=input_channel,
                    reduction_ratio=16,
                    pool_types=["avg", "max"],
                    spatial_config=spatial_config,
                )
            )
        assert len(layers) > 0
        self.cbams = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        if self.use_dims is not None:
            return self.cbams(x[:, self.use_dims])
        else:
            return self.cbams(x)
