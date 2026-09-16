import torch
import torch.nn as nn

from hAlgorithm.modules.models2.external.cut3r.dust3r.blocks import (
    Block,
    DecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
    CustomDecoderBlock,
)  # noqa
from hAlgorithm.modules.models2.encoder.croco_encoder import RgbEncoder
from hAlgorithm.modules.models2.external.cut3r.dust3r.patch_embed import get_patch_embed
from hAlgorithm.modules.models2.external.cut3r.croco.models.blocks import PatchEmbed
from functools import partial
from torch.utils.checkpoint import checkpoint

class Cut3rEncoder(RgbEncoder):
    def __init__(
        self,
        ray_enc_depth,
        patch_embed_cls='PatchEmbedDust3R',
        gradient_checkpointing=True,
        **kwargs
    ):
        # NOTE: Must be set before calling init
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        
        super(Cut3rEncoder, self).__init__(**kwargs)
        
        if self.ray_enc_depth > 0:
            self.enc_blocks_ray_map = nn.ModuleList(
                [
                    Block(
                        self.enc_embed_dim,
                        16,
                        4,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        rope=self.rope,
                    )
                    for _ in range(ray_enc_depth)
                ]
            )
            self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)
        
        self.gradient_checkpointing = gradient_checkpointing

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        if self.ray_enc_depth > 0:
            self.patch_embed_ray_map = get_patch_embed(
                self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
            )

    def _encode_image(self, image, true_shape=None) -> torch.Tensor:
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and image.requires_grad:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        return x, pos

    def _encode_ray_map(self, ray_map: torch.Tensor, true_shape=None):
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks_ray_map:
            if self.gradient_checkpointing and ray_map.requires_grad:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm_ray_map(x)
        return x, pos

    def forward(self, rgb:torch.Tensor, meta_data, prompt_ray=None, **kwargs):
        if rgb.ndim == 5:
            rgb = rgb.view(-1, *rgb.shape[2:]).contiguous()
        if prompt_ray is not None and prompt_ray.ndim == 5:
            prompt_ray = prompt_ray.view(-1, *prompt_ray.shape[2:]).contiguous()

        img_tokens, img_pos = self._encode_image(rgb)
        if prompt_ray is not None:
            ray_tokens, ray_pos = self._encode_ray_map(prompt_ray)
            img_tokens += ray_tokens
            img_pos += ray_pos

        return img_tokens, img_pos

