from .base import MLP
import torch


class MLPScaleEncoder(MLP):
    def __init__(
        self,
        in_chans,
        enc_embed_dim,
        intermediate_dims,
        **kwargs,
    ):
        super().__init__(
            in_chans=in_chans,
            enc_embed_dim=enc_embed_dim,
            intermediate_dims=intermediate_dims,
            **kwargs,
        )

    def forward(self, prompt_scale: torch.Tensor, **kwargs):
        prompt_scale = prompt_scale.squeeze(-1).squeeze(-1)  # B,1,1,1 -> B,1
        return super().forward(prompt_scale)
