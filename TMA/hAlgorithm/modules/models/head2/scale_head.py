from ..aggregator.base import MLP
import torch
from ..head.blocks import Exp


class MLPScaleHead(MLP):
    def __init__(
        self,
        in_chans,
        out_dim,
        intermediate_dims,
        layer_idx=None,
        output_act="exp",
        **kwargs,
    ):
        super().__init__(
            in_chans=in_chans, enc_embed_dim=out_dim, intermediate_dims=intermediate_dims, **kwargs
        )
        self.layer_idx = layer_idx
        if output_act in ["relu"]:
            self.output_act = torch.nn.ReLU()
        elif output_act in ["exp"]:
            self.output_act = Exp()
        else:
            self.output_act = torch.nn.Identity()

    def forward(self, patch_features, **kwargs):
        cls_tokens = [x[1] for x in patch_features]
        if self.layer_idx is not None:
            cls_tokens = cls_tokens[self.layer_idx]  # B,C
        else:
            cls_tokens = torch.cat(cls_tokens, dim=1)  # B,N*C
        output = super().forward(cls_tokens)
        output = self.output_act(output)
        output = dict(scale=output)
        return output
