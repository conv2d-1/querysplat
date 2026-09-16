import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from hAlgorithm.modules.models.pi3.models.layers.pos_embed import PositionGetter, RoPE2D
from hAlgorithm.modules.models.vggt.layers.block import Block
from hAlgorithm.modules.models.vggt.layers.rope import PositionGetter

from .blocks import slice_expand_and_flatten


class MVEncoder(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        # num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        qk_norm (bool): Whether to apply QK normalization.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=0,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        aa_order=["frame", "global"],
        qk_norm=True,
        init_values=0.01,
        hooks=None,
        use_checkpoint=False,
        pretrain=None,
        pertrain_strict=True,
        chunk_size=None,
        **kwargs,
    ):
        super().__init__()

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.embed_dim = embed_dim
        self.hooks = hooks
        self.use_checkpoint = use_checkpoint
        self.pretrain = pretrain
        self.pertrain_strict = pertrain_strict
        self.chunk_size = chunk_size

        if RoPE2D is None:
            raise ImportError("Cannot find cuRoPE2D")
        self.rope = RoPE2D(freq=100)
        self.position_getter = PositionGetter()

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        if self.num_register_tokens > 0:
            self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        if self.num_register_tokens > 0:
            nn.init.normal_(self.register_token, std=1e-6)

        if pretrain is not None:
            self.load_state_dict(
                torch.load(pretrain, map_location="cpu", weights_only=False),
                strict=pertrain_strict,
            )
            logging.info(f"MVVIT, load pretrain {pretrain}")

    def forward(self, patch_tokens, meta_data, memory_efficient_infer=False, **kwargs):
        patch_w = meta_data["input_width"][0] // self.patch_size
        patch_h = meta_data["input_height"][0] // self.patch_size
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        if isinstance(patch_tokens, (list, tuple)):
            patch_tokens = patch_tokens[-1]

        BS, P, C = patch_tokens.shape
        S = frame_num * view_num
        B = BS // S

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        if self.num_register_tokens > 0:
            register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        if self.num_register_tokens > 0:
            tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        else:
            tokens = torch.cat([camera_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, patch_h, patch_w, device=patch_tokens.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(patch_tokens.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        if memory_efficient_infer:
            del camera_token
            del register_token
            del patch_tokens

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        bar = range(self.depth) if not memory_efficient_infer else tqdm(range(self.depth), desc="mv encoder depth")
        for di in bar:
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(tokens, B, S, P, C, frame_idx, pos=pos, memory_efficient_infer=memory_efficient_infer)
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(tokens, B, S, P, C, global_idx, pos=pos, memory_efficient_infer=memory_efficient_infer)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # concat frame and global intermediates, [B x S x P x 2C]
            if (self.hooks is not None and di in self.hooks) or (self.hooks is None and di == self.depth - 1):
                output_list.append(torch.cat([frame_intermediates, global_intermediates], dim=-1))
        
        if memory_efficient_infer:
            del frame_intermediates
            del global_intermediates

        return output_list, pos, self.patch_start_idx

    def _onnx_prepare_extra_input(self, B, S, patch_h, patch_w):
        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        if self.num_register_tokens > 0:
            register_token = slice_expand_and_flatten(self.register_token, B, S)
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, patch_h, patch_w, device=camera_token.device)
            if self.patch_start_idx > 0:
                # do not use position embedding for special tokens (camera and register tokens)
                # so set pos to 0 for the special tokens
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(pos.device).to(pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)
        return camera_token, register_token, pos

    def onnx_forward(self, patch_tokens, view_num, camera_token, register_token, pos):
        
        if isinstance(patch_tokens, (list, tuple)):
            patch_tokens = patch_tokens[-1]
        
        BS, P, C = patch_tokens.shape
        S = int(view_num)
        B = BS // S
        
        # Concatenate special tokens with patch tokens
        if self.num_register_tokens > 0:
            tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        else:
            tokens = torch.cat([camera_token, patch_tokens], dim=1)
        
        # update P because we added special tokens
        _, P, C = tokens.shape
        
        frame_idx = 0
        global_idx = 0
        output_list = []
        
        for di in range(self.depth):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(tokens, B, S, P, C, frame_idx, pos=pos, memory_efficient_infer=False)
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(tokens, B, S, P, C, global_idx, pos=pos, memory_efficient_infer=False)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # concat frame and global intermediates, [B x S x P x 2C]
            if (self.hooks is not None and di in self.hooks) or (self.hooks is None and di == self.depth - 1):
                output_list.append(torch.cat([frame_intermediates, global_intermediates], dim=-1))
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, memory_efficient_infer=False):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        if self.use_checkpoint:
            tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=False)
        else:
            if self.chunk_size is not None and self.chunk_size > 0:
                for i in range(0, B * S, self.chunk_size):
                    end = min(i + self.chunk_size, B * S)
                    tokens[i:end] = self.frame_blocks[frame_idx](tokens[i:end], pos=pos[i:end] if pos is not None else None)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)

        frame_idx += 1
        if memory_efficient_infer:
            intermediates = tokens.cpu().view(B, S, P, C)
        else:
            intermediates = tokens.view(B, S, P, C)

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, memory_efficient_infer=False):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        if self.use_checkpoint:
            tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=False)
        else:
            tokens = self.global_blocks[global_idx](tokens, pos=pos)

        global_idx += 1
        if memory_efficient_infer:
            intermediates = tokens.cpu().view(B, S, P, C)
        else:
            intermediates = tokens.view(B, S, P, C)

        return tokens, global_idx, intermediates


class RayMVEncoder(MVEncoder):
    def __init__(self, normalized_shape=None, **kwargs):
        super(RayMVEncoder, self).__init__(**kwargs)

        if normalized_shape is not None:
            self.fusion_norm_layer = nn.LayerNorm(normalized_shape=normalized_shape, eps=1e-6)
        else:
            self.fusion_norm_layer = None

    def forward(self, patch_tokens, prompt_ray=None, prompt_ray_in_world=None, meta_data=None, **kwargs):

        if isinstance(patch_tokens, (list, tuple)):
            patch_tokens = patch_tokens[-1]

        if prompt_ray is not None:
            patch_tokens = patch_tokens + prompt_ray
        
        if prompt_ray_in_world is not None:
            patch_tokens = patch_tokens + prompt_ray_in_world
        
        if self.fusion_norm_layer is not None:
            patch_tokens = self.fusion_norm_layer(patch_tokens)

        return super().forward(patch_tokens=patch_tokens, meta_data=meta_data, **kwargs)


class RayExtraMVEncoder(MVEncoder):
    def __init__(self, normalized_shape=None, **kwargs):
        super(RayExtraMVEncoder, self).__init__(**kwargs)

        if normalized_shape is not None:
            self.fusion_norm_layer = nn.LayerNorm(normalized_shape=normalized_shape, eps=1e-6)
        else:
            self.fusion_norm_layer = None

    def forward(self, patch_tokens, prompt_ray=None, prompt_ray_in_world=None, prompt_extra=None, meta_data=None, **kwargs):

        if isinstance(patch_tokens, (list, tuple)):
            patch_tokens = patch_tokens[-1]

        if prompt_ray is not None:
            patch_tokens = patch_tokens + prompt_ray
        
        if prompt_ray_in_world is not None:
            patch_tokens = patch_tokens + prompt_ray_in_world

        if prompt_extra is not None:
            for prompt in prompt_extra:
                patch_tokens = patch_tokens + prompt.unsqueeze(1)
        
        if self.fusion_norm_layer is not None:
            patch_tokens = self.fusion_norm_layer(patch_tokens)

        return super().forward(patch_tokens=patch_tokens, meta_data=meta_data, **kwargs)


class FullMVEncoder(MVEncoder):
    def __init__(self, normalized_shape=None, **kwargs):
        super(RayExtraMVEncoder, self).__init__(**kwargs)

        if normalized_shape is not None:
            self.fusion_norm_layer = nn.LayerNorm(normalized_shape=normalized_shape, eps=1e-6)
        else:
            self.fusion_norm_layer = None

    def forward(self, patch_tokens, prompt_depth=None, prompt_ray=None, prompt_ray_in_world=None, prompt_extra=None, meta_data=None, **kwargs):

        if isinstance(patch_tokens, (list, tuple)):
            patch_tokens = patch_tokens[-1]
        
        if prompt_depth is not None:
            patch_tokens = patch_tokens + prompt_depth

        if prompt_ray is not None:
            patch_tokens = patch_tokens + prompt_ray
        
        if prompt_ray_in_world is not None:
            patch_tokens = patch_tokens + prompt_ray_in_world

        if prompt_extra is not None:
            for prompt in prompt_extra:
                patch_tokens = patch_tokens + prompt.unsqueeze(1)
        
        if self.fusion_norm_layer is not None:
            patch_tokens = self.fusion_norm_layer(patch_tokens)

        return super().forward(patch_tokens=patch_tokens, meta_data=meta_data, **kwargs)
