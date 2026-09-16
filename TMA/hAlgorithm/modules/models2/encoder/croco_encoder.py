import torch
import torch.nn as nn
from hAlgorithm.modules.models2.external.cut3r.croco.models.blocks import (
    Block, 
    DecoderBlock, 
    PositionGetter,
    PatchEmbed
    )
from hAlgorithm.modules.models2.external.cut3r.croco.models.pos_embed import RoPE2D
from functools import partial

class RgbEncoder(nn.Module):
    def __init__(
        self,
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        mlp_ratio=4,
        pos_embed="RoPE100",
        img_size=512,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        patch_size=16,
        **kwargs
    ):
        super().__init__()
        if pos_embed.startswith("RoPE"):  # eg RoPE100
            self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(pos_embed[len("RoPE") :])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown pos_embed " + pos_embed)
        
        self._set_patch_embed(img_size, patch_size, enc_embed_dim)
        
        # transformer for the encoder
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_num_heads = enc_num_heads
        self.mlp_ratio = mlp_ratio
        
        self.enc_blocks = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    self.enc_num_heads,
                    self.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    rope=self.rope,
                )
                for i in range(self.enc_depth)
            ]
        )
        self.enc_norm = norm_layer(self.enc_embed_dim)
        self.initialize_weights()

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

    def initialize_weights(self):
        # patch embed
        self.patch_embed._init_weights()
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, **kwargs):
        raise NotImplementedError()
