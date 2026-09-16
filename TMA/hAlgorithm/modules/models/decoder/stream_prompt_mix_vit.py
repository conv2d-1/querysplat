import logging

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from hAlgorithm.modules.models.facebookresearch_dinov2_main.dinov2.layers import PatchEmbed
from hAlgorithm.modules.models.streamvggt.layers.block import Block
from hAlgorithm.modules.models.streamvggt.layers.rope import (
    PositionGetter,
    RotaryPositionEmbedding2D,
)

from .blocks import slice_expand_and_flatten


class MixVIT(nn.Module):
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
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        prompt_in_chans,
        prompt_embed_dim,
        fuse_embed_dim=0,
        patch_size=14,
        prompt_patch_size=None,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=0,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        aa_order=["frame", "global", "stream_global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        hooks=None,
        use_checkpoint=False,
        output_frame=False,
        output_global=False,
        pretrain=None,
        pertrain_strict=True,
        debug=False,
    ):
        super().__init__()

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.prompt_patch_size = prompt_patch_size or self.patch_size
        self.aa_block_size = aa_block_size
        self.num_register_tokens = num_register_tokens
        self.embed_dim = embed_dim

        self.prompt_in_chans = prompt_in_chans
        self.prompt_embed_dim = prompt_embed_dim
        self.fuse_embed_dim = fuse_embed_dim

        if fuse_embed_dim == 0:
            fuse_embed_dim = self.embed_dim + self.prompt_embed_dim

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=fuse_embed_dim,
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
                    dim=fuse_embed_dim,
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

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(
                f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})"
            )

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, fuse_embed_dim))
        if self.num_register_tokens > 0:
            self.register_token = nn.Parameter(
                torch.randn(1, 2, num_register_tokens, fuse_embed_dim)
            )

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        if self.num_register_tokens > 0:
            nn.init.normal_(self.register_token, std=1e-6)

        self.hooks = hooks
        self.use_checkpoint = use_checkpoint
        self.debug = debug
        self.output_frame = output_frame
        self.output_global = output_global
        self.pretrain = pretrain

        if self.pretrain is not None:
            self.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=pertrain_strict,
            )
            logging.info(f"MixVIT, load pretrain {self.pretrain}")

        if self.prompt_in_chans > 0:
            self.prompt_patch_embed = PatchEmbed(
                img_size=224,
                patch_size=self.prompt_patch_size,
                in_chans=self.prompt_in_chans,
                embed_dim=self.prompt_embed_dim,
            )
        if self.fuse_embed_dim > 0:
            self.fuse_project = nn.Linear(
                self.embed_dim + self.prompt_embed_dim, self.fuse_embed_dim, bias=True
            )

    def forward(
        self,
        patch_tokens,
        prompt_features,
        meta_data,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        patch_w = meta_data["input_width"][0] // self.patch_size
        patch_h = meta_data["input_height"][0] // self.patch_size
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        if isinstance(patch_tokens, (list, tuple)):
            patch_tokens = patch_tokens[-1]

        BS, P, C = patch_tokens.shape
        S = frame_num * view_num
        B = BS // S

        if use_cache and past_key_values[0] is not None:
            _, _, S_true, _, _ = past_key_values[0][0].shape
            S_true += 1
        else:
            S_true = S

        if use_cache and S > 1:
            logging.error(f"Use KV cache expects S=1, got S={S}")

        # Expand camera and register tokens to match batch size and sequence length
        if use_cache:
            camera_token_full = slice_expand_and_flatten(self.camera_token, B, S_true)
            camera_token = camera_token_full[-1:, :, :]
            if self.num_register_tokens > 0:
                register_token_full = slice_expand_and_flatten(self.register_token, B, S_true)
                register_token = register_token_full[-1:, :, :]
        else:
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            if self.num_register_tokens > 0:
                register_token = slice_expand_and_flatten(self.register_token, B, S)

        if self.prompt_in_chans > 0 and prompt_features is not None:
            prompt_tokens = self.prompt_patch_embed(prompt_features)
            patch_tokens = torch.cat([patch_tokens, prompt_tokens], dim=-1)
            if self.fuse_embed_dim > 0:
                patch_tokens = self.fuse_project(patch_tokens)

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
            pos_special = (
                torch.zeros(B * S, self.patch_start_idx, 2).to(patch_tokens.device).to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        camera_output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                elif attn_type == "stream_global":
                    if use_cache:
                        # assert past_key_values is not None
                        # if past_key_values[global_idx] is None:
                        #     print("past_key_values", global_idx, past_key_values[global_idx])
                        # else:
                        #     print("past_key_values", global_idx, past_key_values[global_idx][0].shape, past_key_values[global_idx][1].shape)
                        tokens, global_idx, global_intermediates, block_kv = (
                            self._process_stream_global_attention(
                                tokens,
                                B,
                                S,
                                P,
                                C,
                                global_idx,
                                pos=pos,
                                past_key_values_block=past_key_values[global_idx],
                                use_cache=True,
                            )
                        )
                        past_key_values[global_idx - 1] = block_kv
                    else:
                        tokens, global_idx, global_intermediates = (
                            self._process_stream_global_attention(
                                tokens, B, S, P, C, global_idx, pos=pos
                            )
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(global_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                if self.output_frame:
                    concat_inter = frame_intermediates[i]
                elif self.output_global:
                    concat_inter = global_intermediates[i]
                else:
                    concat_inter = torch.cat(
                        [frame_intermediates[i], global_intermediates[i]], dim=-1
                    )

                output_list.append(concat_inter[:, :, self.patch_start_idx :])
                camera_output_list.append(concat_inter[:, :, : self.patch_start_idx])

        # del concat_inter
        # del frame_intermediates
        # del global_intermediates

        if self.debug:
            self.vis_vit_features(
                [output_list[hook] for hook in self.hooks], patch_h, patch_w, meta_data
            )

        if self.hooks is not None:
            return [output_list[hook] for hook in self.hooks], [
                camera_output_list[hook] for hook in self.hooks
            ]
        else:
            return output_list, camera_output_list

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=False)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                tokens = checkpoint(
                    self.global_blocks[global_idx], tokens, pos, use_reentrant=False
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    def _process_stream_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        past_key_values_block=None,
        use_cache=False,
    ):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """

        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)

        intermediates = []

        for _ in range(self.aa_block_size):
            if use_cache:
                tokens, block_kv = self.global_blocks[global_idx](
                    tokens,
                    pos=pos,
                    attn_mask=None,
                    past_key_values=past_key_values_block,
                    use_cache=True,
                )
            else:
                L = S * P
                frame_ids = torch.arange(L, device=tokens.device) // P  # [0,0,...,1,1,...,S-1]
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = future_frame.to(tokens.dtype) * torch.finfo(tokens.dtype).min

                tokens = self.global_blocks[global_idx](tokens, pos=pos, attn_mask=attn_mask)

            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        if use_cache:
            return tokens, global_idx, intermediates, block_kv

        return tokens, global_idx, intermediates
