# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import math
from typing import Callable, List, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from hAlgorithm.modules.models2.external.depth_anything_3.utils.logger import logger

from hAlgorithm.modules.models2.external.depth_anything_3.model.dinov2.layers import LayerScale  # noqa: F401
from hAlgorithm.modules.models2.external.depth_anything_3.model.dinov2.layers import Mlp  # noqa: F401
from hAlgorithm.modules.models2.external.depth_anything_3.model.dinov2.layers import (  # noqa: F401
    Block,
    PatchEmbed,
    PositionGetter,
    RotaryPositionEmbedding2D,
    SwiGLUFFNFused,
)

# logger = logging.getLogger("dinov2")


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1.0,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        rope_freq=100,
        plus_cam_token=False,
        cat_token=True,
        use_checkpoint_start=-1,
        use_checkpoint_end=-1,
        prompt_patch_size=None,
        prompt_in_chans=None,
        prompt_embed_dim=None,
        prompt_zero_adapt=None,
        prompt_start=-1,
        prompt_index=None,

        prompt_ray_in_chans=None,
        prompt_ray_embed_dim=None,
        prompt_ray_zero_adapt=None,
        prompt_ray_start=-1,
        prompt_ray_index=None,

        add_aux_camera_token=False,
        add_aux_camera_token_every=False,
        add_aux_camera_token_index=None,
        start=0,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating
                positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating
                positional embeddings
            block_prompt: (bool) whether to add ray embeddings to the block input
        """
        super().__init__()
        self.patch_start_idx = 1
        norm_layer = nn.LayerNorm
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.start = start
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.use_checkpoint_start = use_checkpoint_start
        self.use_checkpoint_end = use_checkpoint_end
        self.add_aux_camera_token = add_aux_camera_token
        self.add_aux_camera_token_every = add_aux_camera_token_every
        self.add_aux_camera_token_index = add_aux_camera_token_index

        self.prompt_ray_in_chans = prompt_ray_in_chans
        self.prompt_ray_embed_dim = prompt_ray_embed_dim
        self.prompt_ray_zero_adapt = prompt_ray_zero_adapt
        self.prompt_ray_start = prompt_ray_start
        self.prompt_ray_index = prompt_ray_index

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule
        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
            self.position_getter = PositionGetter() if self.rope is not None else None
        else:
            self.rope = None
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                qk_norm=i >= qknorm_start if qknorm_start != -1 else False,
                rope=self.rope if i >= rope_start and rope_start != -1 else None,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)

        self.prompt_start = prompt_start
        self.prompt_index = prompt_index
        self.prompt_patch_size = prompt_patch_size
        self.prompt_in_chans = prompt_in_chans
        self.prompt_embed_dim = prompt_embed_dim
        if prompt_patch_size is not None and prompt_in_chans is not None and prompt_embed_dim is not None:
            self.prompt_patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=prompt_patch_size,
                in_chans=prompt_in_chans,
                embed_dim=prompt_embed_dim,
            )
        else:
            self.prompt_patch_embed = None

        self.prompt_zero_adapt = prompt_zero_adapt
        if self.prompt_zero_adapt is not None:
            if self.prompt_patch_embed is not None:
                self.prompt_zero_adapt = nn.Linear(prompt_embed_dim, prompt_embed_dim, bias=True)
            else:
                self.prompt_zero_adapt = nn.Linear(prompt_in_chans, prompt_embed_dim, bias=True)
            nn.init.zeros_(self.prompt_zero_adapt.weight)
            nn.init.zeros_(self.prompt_zero_adapt.bias)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the
            # interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using
            # both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_cls_token(self, B, S):
        cls_token = self.cls_token.expand(B, S, -1)
        cls_token = cls_token.reshape(B * S, -1, self.embed_dim)
        return cls_token

    def prepare_tokens_with_masks(self, x, masks=None, cls_token=None, prompt_depth=None, prompt_ray=None, **kwargs):
        B, S, nc, w, h = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        cls_token = self.prepare_cls_token(B, S)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
        return x

    def _prepare_rope(self, B, S, H, W, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=device
            )
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(device).to(pos.dtype)
                pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
                pos = torch.cat([pos_special, pos], dim=2)
                pos_nodiff = pos_nodiff + 1
                pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], prompt_depth=None, prompt_ray=None, **kwargs):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x, prompt_depth=prompt_depth, prompt_ray=prompt_ray)
        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        for i, blk in enumerate(self.blocks):
            if i < self.start:
                continue

            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos
            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    # logger.info("Using camera conditions provided by the user")
                    if self.add_aux_camera_token:
                        ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                        src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                        cam_token = torch.cat([ref_token, src_token], dim=1)
                        cam_token += kwargs.get("cam_token")
                    elif self.add_aux_camera_token_index is not None:
                        ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                        src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                        cam_token = torch.cat([ref_token, src_token], dim=1)
                    else:
                        cam_token = kwargs.get("cam_token")
                else:
                    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                    src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                x[:, :, 0] = cam_token

            if self.add_aux_camera_token_index is not None and i in self.add_aux_camera_token_index:
                x[:, :, 0] += kwargs.get("cam_token")

            # prompt depth
            if self.prompt_start != -1 and i == self.prompt_start:
                if self.prompt_patch_embed is not None:
                    prompt_depth = self.prompt_patch_embed(prompt_depth)
                if self.prompt_zero_adapt is not None:
                    prompt_depth = self.prompt_zero_adapt(prompt_depth)
                if prompt_depth.ndim == 3:
                    prompt_depth = prompt_depth.view(B, S, *prompt_depth.shape[1:])
                if self.num_register_tokens > 0:
                    x[:, :, 1 + self.num_register_tokens:] += prompt_depth
                else:
                    x[:, :, 1:] += prompt_depth
            elif self.prompt_index is not None and i in self.prompt_index:
                if self.num_register_tokens > 0:
                    x[:, :, 1 + self.num_register_tokens:] += prompt_depth
                else:
                    x[:, :, 1:] += prompt_depth

            # prompt ray
            if self.prompt_ray_start != -1 and i == self.prompt_ray_start:
                if self.prompt_ray_patch_embed is not None:
                    prompt_ray = self.prompt_ray_patch_embed(prompt_ray)
                if self.prompt_ray_zero_adapt is not None:
                    prompt_ray = self.prompt_ray_zero_adapt(prompt_ray)
                if prompt_ray.ndim == 3:
                    prompt_ray = prompt_ray.view(B, S, *prompt_ray.shape[1:])
                if self.num_register_tokens > 0:
                    x[:, :, 1 + self.num_register_tokens:] += prompt_ray
                else:
                    x[:, :, 1:] += prompt_ray
            elif self.prompt_ray_index is not None and i in self.prompt_ray_index:
                if prompt_ray.ndim == 3:
                    prompt_ray = prompt_ray.view(B, S, *prompt_ray.shape[1:])
                if self.num_register_tokens > 0:
                    x[:, :, 1 + self.num_register_tokens:] += prompt_ray
                else:
                    x[:, :, 1:] += prompt_ray

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(
                    i, x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
                )
                if self.add_aux_camera_token_every:
                    x[:, :, 0] += kwargs.get("cam_token")
            else:
                x = self.process_attention(i, x, blk, "local", pos=l_pos)
                local_x = x

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)
        return output, aux_output

    def process_attention(self, i, x, block, attn_type="global", pos=None, attn_mask=None):
        b, s, n = x.shape[:3]
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        if self.training and self.use_checkpoint_start != -1 and i >= self.use_checkpoint_start:
            if self.use_checkpoint_end == -1 or i <= self.use_checkpoint_end:
                # logging.info(f"Using checkpoint for layer {i}")
                x = checkpoint(block, x, pos, use_reentrant=False)
            else:
                x = block(x, pos=pos, attn_mask=attn_mask)
        else:
            x = block(x, pos=pos, attn_mask=attn_mask)

        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        return x

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        export_feat_layers: List[int] = [],
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        outputs, aux_outputs = self._get_intermediate_layers_not_chunked(
            x, n, export_feat_layers=export_feat_layers, **kwargs
        )
        if outputs[0][1].shape[-1] == self.embed_dim:
            outputs = [self.norm(out[1]) for out in outputs]
        elif outputs[0][1].shape[-1] == (self.embed_dim * 2):
            outputs = [
                torch.cat(
                    [out[1][..., : self.embed_dim], self.norm(out[1][..., self.embed_dim :])],
                    dim=-1,
                )
                for out in outputs
            ]
        else:
            raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")

        # NOTE, TMA pipe, patch_features, pos, patch_start_idx
        return outputs, None, 1 + self.num_register_tokens

    def get_intermediate_layers_onnx(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        export_feat_layers: List[int] = [],
        prompt_depth=None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        # outputs, aux_outputs = self._get_intermediate_layers_not_chunked(
        #     x, n, export_feat_layers=export_feat_layers, **kwargs
        # )
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x, prompt_depth=prompt_depth)
        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = kwargs.get("pos", None), kwargs.get("pos_nodiff", None)

        for i, blk in enumerate(self.blocks):
            if i < self.start:
                continue

            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos
            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    # logger.info("Using camera conditions provided by the user")
                    if self.add_aux_camera_token:
                        ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                        src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                        cam_token = torch.cat([ref_token, src_token], dim=1)
                        cam_token += kwargs.get("cam_token")
                    else:
                        cam_token = kwargs.get("cam_token")
                else:
                    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                    src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                x[:, :, 0] = cam_token

            if self.prompt_start != -1 and i == self.prompt_start:
                if self.prompt_patch_embed is not None:
                    prompt_depth = self.prompt_patch_embed(prompt_depth)
                if self.prompt_zero_adapt is not None:
                    prompt_depth = self.prompt_zero_adapt(prompt_depth)
                if prompt_depth.ndim == 3:
                    prompt_depth = prompt_depth.view(B, S, *prompt_depth.shape[1:])
                if self.num_register_tokens > 0:
                    x[:, :, 1 + self.num_register_tokens:] += prompt_depth
                else:
                    x[:, :, 1:] += prompt_depth
            elif self.prompt_index is not None and i in self.prompt_index:
                if self.num_register_tokens > 0:
                    x[:, :, 1 + self.num_register_tokens:] += prompt_depth
                else:
                    x[:, :, 1:] += prompt_depth

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(
                    i, x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
                )
                if self.add_aux_camera_token_every:
                    x[:, :, 0] += kwargs.get("cam_token")
            else:
                x = self.process_attention(i, x, blk, "local", pos=l_pos)
                local_x = x

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)

        # camera_tokens = [out[0] for out in outputs]
        if output[0][1].shape[-1] == self.embed_dim:
            output = [self.norm(out[1]) for out in output]
        elif output[0][1].shape[-1] == (self.embed_dim * 2):
            output = [
                torch.cat(
                    [out[1][..., : self.embed_dim], self.norm(out[1][..., self.embed_dim :])],
                    dim=-1,
                )
                for out in output
            ]
        else:
            raise ValueError(f"Invalid output shape: {output[0][1].shape}")
        
        # outputs = [out[..., 1 + self.num_register_tokens :, :] for out in outputs]

        # aux_outputs = [self.norm(out) for out in aux_outputs]
        # aux_outputs = [out[..., 1 + self.num_register_tokens :, :] for out in aux_outputs]
        
        # return tuple(zip(outputs, camera_tokens)), aux_outputs

        # NOTE, TMA pipe, patch_features, pos, patch_start_idx 
        return output, None, 1 + self.num_register_tokens

def vit_small(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, depth=24, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=depth,
        num_heads=16,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, depth=40, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=depth,
        num_heads=24,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


class DinoVisionTransformerWithTimeToken(DinoVisionTransformer):
    """
    Extension of DinoVisionTransformer that supports MoVieS-style time token injection
    and Plücker ray embedding (prompt_ray_in_world).

    Sequence layout inside the backbone:
        [cls | registers (optional) | time_token (optional) | patch_tokens ...]

    patch_start_idx is bumped from ``1 + num_register_tokens`` to
    ``1 + num_register_tokens + 1`` when a time_token is provided, so that
    downstream heads index patch tokens correctly.
    """

    def prepare_tokens_with_masks(self, x, masks=None, cls_token=None, prompt_depth=None, time_token=None, **kwargs):
        B, S, nc, w, h = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        cls_token = self.prepare_cls_token(B, S)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        # MoVieS-style: concat time_token as an independent slot after cls+registers,
        # before patch tokens.  time_token shape: [B*S, 1, C]
        if time_token is not None:
            num_special = 1 + self.num_register_tokens
            x = torch.cat(
                (x[:, :num_special], time_token, x[:, num_special:]),
                dim=1,
            )
        x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
        return x

    def _prepare_rope(self, B, S, H, W, device, patch_start_idx=None):
        if patch_start_idx is None:
            patch_start_idx = self.patch_start_idx
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=device
            )
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            if patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(B * S, patch_start_idx, 2).to(device).to(pos.dtype)
                pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
                pos = torch.cat([pos_special, pos], dim=2)
                pos_nodiff = pos_nodiff + 1
                pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], prompt_depth=None, time_token=None, **kwargs):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x, prompt_depth=prompt_depth, time_token=time_token)
        # Bump patch_start_idx by 1 when a time_token slot was prepended.
        dyn_patch_start_idx = 1 + self.num_register_tokens + (1 if time_token is not None else 0)

        prompt_ray_in_world = kwargs.get("prompt_ray_in_world", None)
        if prompt_ray_in_world is not None:
            if prompt_ray_in_world.ndim == 4:
                prompt_ray_in_world = prompt_ray_in_world.flatten(2).transpose(1, 2)
            if prompt_ray_in_world.ndim == 3:
                prompt_ray_in_world = rearrange(prompt_ray_in_world, "(b s) n c -> b s n c", b=B, s=S)
            x[:, :, dyn_patch_start_idx:] = x[:, :, dyn_patch_start_idx:] + prompt_ray_in_world

        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device, patch_start_idx=dyn_patch_start_idx)

        for i, blk in enumerate(self.blocks):
            if i < self.start:
                continue

            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos
            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    if self.add_aux_camera_token:
                        ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                        src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                        cam_token = torch.cat([ref_token, src_token], dim=1)
                        cam_token += kwargs.get("cam_token")
                    elif self.add_aux_camera_token_index is not None:
                        ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                        src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                        cam_token = torch.cat([ref_token, src_token], dim=1)
                    else:
                        cam_token = kwargs.get("cam_token")
                else:
                    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                    src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                x[:, :, 0] = cam_token

            if self.add_aux_camera_token_index is not None and i in self.add_aux_camera_token_index:
                x[:, :, 0] += kwargs.get("cam_token")

            if self.prompt_start != -1 and i == self.prompt_start:
                if self.prompt_patch_embed is not None:
                    prompt_depth = self.prompt_patch_embed(prompt_depth)
                if self.prompt_zero_adapt is not None:
                    prompt_depth = self.prompt_zero_adapt(prompt_depth)
                if prompt_depth.ndim == 3:
                    prompt_depth = prompt_depth.view(B, S, *prompt_depth.shape[1:])
                if self.num_register_tokens > 0:
                    x[:, :, 1 + self.num_register_tokens:] += prompt_depth
                else:
                    x[:, :, 1:] += prompt_depth
            elif self.prompt_index is not None and i in self.prompt_index:
                if self.num_register_tokens > 0:
                    x[:, :, 1 + self.num_register_tokens:] += prompt_depth
                else:
                    x[:, :, 1:] += prompt_depth

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(
                    i, x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
                )
                if self.add_aux_camera_token_every:
                    x[:, :, 0] += kwargs.get("cam_token")
            else:
                x = self.process_attention(i, x, blk, "local", pos=l_pos)
                local_x = x

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)
        return output, aux_output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        export_feat_layers: List[int] = [],
        time_token=None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        outputs, aux_outputs = self._get_intermediate_layers_not_chunked(
            x, n, export_feat_layers=export_feat_layers, time_token=time_token, **kwargs
        )
        if outputs[0][1].shape[-1] == self.embed_dim:
            outputs = [self.norm(out[1]) for out in outputs]
        elif outputs[0][1].shape[-1] == (self.embed_dim * 2):
            outputs = [
                torch.cat(
                    [out[1][..., : self.embed_dim], self.norm(out[1][..., self.embed_dim :])],
                    dim=-1,
                )
                for out in outputs
            ]
        else:
            raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")

        # NOTE, TMA pipe, patch_features, pos, patch_start_idx
        # When a time_token is present, there is one extra special slot before patches.
        patch_start_idx = 1 + self.num_register_tokens + (1 if time_token is not None else 0)
        return outputs, None, patch_start_idx


class DinoV2WithTimeToken(nn.Module):
    """
    Drop-in replacement for DinoV2 that uses DinoVisionTransformerWithTimeToken as its
    backbone, enabling MoVieS-style time token injection and prompt_ray_in_world support.

    All constructor arguments are identical to DinoV2.
    """

    def __init__(
        self,
        name: str,
        out_layers: List[int],
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        start: int = 0,
        use_checkpoint_start: int = -1,
        use_checkpoint_end: int = -1,
        patch_size: int = 14,
        pretrain: str = None,
        pretrained_pretrain: str = None,
        pertrain_strict: bool = True,
        normalize: bool = False,
        prompt_patch_size=None,
        prompt_in_chans=None,
        prompt_embed_dim=None,
        prompt_zero_adapt=None,
        prompt_start=-2,
        prompt_index=None,
        add_aux_camera_token=False,
        add_aux_camera_token_every=False,
        add_aux_camera_token_index=None,
        **kwargs,
    ):
        super().__init__()
        assert name in {"vits", "vitb", "vitl", "vitg"}
        self.name = name
        self.out_layers = out_layers
        self.start = start
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token

        embed_dim_map = {"vits": 384, "vitb": 768, "vitl": 1024, "vitg": 1536}
        num_heads_map = {"vits": 6, "vitb": 12, "vitl": 16, "vitg": 24}
        depth_map = {"vits": 12, "vitb": 12, "vitl": 24, "vitg": 40}
        ffn_layer = "swiglufused" if name == "vitg" else "mlp"

        self.pretrained = DinoVisionTransformerWithTimeToken(
            img_size=518,
            patch_size=patch_size,
            embed_dim=embed_dim_map[name],
            depth=depth_map[name],
            num_heads=num_heads_map[name],
            mlp_ratio=4,
            ffn_layer=ffn_layer,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
            start=start,
            use_checkpoint_start=use_checkpoint_start,
            use_checkpoint_end=use_checkpoint_end,
            prompt_patch_size=prompt_patch_size,
            prompt_in_chans=prompt_in_chans,
            prompt_embed_dim=prompt_embed_dim,
            prompt_zero_adapt=prompt_zero_adapt,
            prompt_start=prompt_start,
            prompt_index=prompt_index,
            add_aux_camera_token=add_aux_camera_token,
            add_aux_camera_token_every=add_aux_camera_token_every,
            add_aux_camera_token_index=add_aux_camera_token_index,
        )

        self.use_checkpoint_start = use_checkpoint_start
        self.use_checkpoint_end = use_checkpoint_end

        if pretrain is not None:
            res = self.load_state_dict(
                torch.load(pretrain, map_location="cpu", weights_only=False),
                strict=pertrain_strict,
            )
            logging.info(f"DinoV2WithTimeToken, load pretrain {pretrain}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")
        if pretrained_pretrain is not None:
            res = self.pretrained.load_state_dict(
                torch.load(pretrained_pretrain, map_location="cpu", weights_only=False),
                strict=pertrain_strict,
            )
            logging.info(f"DinoV2WithTimeToken-DinoVisionTransformerWithTimeToken, load pretrain {pretrained_pretrain}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")

        self.normalize = normalize
        if self.normalize:
            self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, **kwargs):
        if self.normalize:
            x = ((x + 1) * 0.5 - self._mean) / self._std

        return self.pretrained.get_intermediate_layers(
            x,
            self.out_layers,
            **kwargs,
        )


class DinoV2(nn.Module):
    def __init__(
        self,
        name: str,
        out_layers: List[int],
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        start: int = 0,
        use_checkpoint_start: int = -1,
        use_checkpoint_end: int = -1,
        patch_size: int = 14,
        pretrain: str = None,
        pretrained_pretrain: str = None,
        pertrain_strict: bool = True,
        normalize: bool = False,
        prompt_patch_size=None,
        prompt_in_chans=None,
        prompt_embed_dim=None,
        prompt_zero_adapt=None,
        prompt_start=-2,
        prompt_index=None,
        add_aux_camera_token=False,
        add_aux_camera_token_every=False,
        add_aux_camera_token_index=None,
        **kwargs,
    ):
        super().__init__()
        assert name in {"vits", "vitb", "vitl", "vitg"}
        self.name = name
        self.out_layers = out_layers
        self.start = start
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        encoder_map = {
            "vits": vit_small,
            "vitb": vit_base,
            "vitl": vit_large,
            "vitg": vit_giant2,
        }
        encoder_fn = encoder_map[self.name]
        ffn_layer = "swiglufused" if self.name == "vitg" else "mlp"
        self.pretrained = encoder_fn(
            img_size=518,
            patch_size=patch_size,
            ffn_layer=ffn_layer,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
            start=start,
            use_checkpoint_start=use_checkpoint_start,
            use_checkpoint_end=use_checkpoint_end,
            prompt_patch_size=prompt_patch_size,
            prompt_in_chans=prompt_in_chans,
            prompt_embed_dim=prompt_embed_dim,
            prompt_zero_adapt=prompt_zero_adapt,
            prompt_start=prompt_start,
            prompt_index=prompt_index,
            add_aux_camera_token=add_aux_camera_token,
            add_aux_camera_token_every=add_aux_camera_token_every,
            add_aux_camera_token_index=add_aux_camera_token_index,
            **kwargs,
        )

        self.use_checkpoint_start = use_checkpoint_start
        self.use_checkpoint_end = use_checkpoint_end

        if pretrain is not None:
            res = self.load_state_dict(
                torch.load(pretrain, map_location="cpu", weights_only=False),
                strict=pertrain_strict,
            )
            logging.info(f"DinoV2, load pretrain {pretrain}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")
        if pretrained_pretrain is not None:
            res = self.pretrained.load_state_dict(
                torch.load(pretrained_pretrain, map_location="cpu", weights_only=False),
                strict=pertrain_strict,
            )
            logging.info(f"DinoV2-DinoVisionTransformer, load pretrain {pretrained_pretrain}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")
        
        self.normalize = normalize
        if self.normalize:
            self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, x, **kwargs):
        if self.normalize:
            x = ((x + 1) * 0.5 - self._mean) / self._std

        return self.pretrained.get_intermediate_layers(
            x,
            self.out_layers,
            **kwargs,
        )
