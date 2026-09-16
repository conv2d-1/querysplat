from .base import AuxResnet, MLP
import torch


class AuxResnetRayEncoder(AuxResnet):
    def __init__(self, layer_num, use_dims, use_bn, **kwargs):
        super().__init__(layer_num=layer_num, use_dims=use_dims, use_bn=use_bn, **kwargs)

    def forward(self, intrinsics, **kwargs):
        return super().forward(intrinsics)


class MLPIntrinsicsEncoder(MLP):
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

    def forward(self, intrinsics: torch.Tensor, **kwargs):
        if intrinsics.ndim == 3:
            intrinsics = intrinsics.flatten(1)  # b,3,3 -> b,9
        return super().forward(intrinsics)
