import logging

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from hAlgorithm.modules.models.facebookresearch_dinov2_main.dinov2.layers import PatchEmbed
from hAlgorithm.modules.models.pi3.models.layers.pos_embed import PositionGetter, RoPE2D
from hAlgorithm.modules.models.vggt.layers.block import Block
from hAlgorithm.modules.models.vggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D


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
        aa_order=["frame", "global", "video"],
        aa_block_size=1,
        qk_norm=True,
        pos_type="rope100",
        init_values=0.01,
        video_init_values=0,
        hooks=None,
        use_checkpoint=False,
        pretrain=None,
        pertrain_strict=True,
        replacement=False,
        debug=False,
        freeze_frame=False,
        freeze_global=False,
    ):
        super().__init__()

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.prompt_patch_size = prompt_patch_size or self.patch_size
        self.aa_block_size = aa_block_size
        self.num_register_tokens = num_register_tokens
        self.embed_dim = embed_dim
        self.hooks = hooks

        self.prompt_in_chans = prompt_in_chans
        self.prompt_embed_dim = prompt_embed_dim
        self.fuse_embed_dim = fuse_embed_dim

        if fuse_embed_dim == 0:
            fuse_embed_dim = self.embed_dim + self.prompt_embed_dim

        # NOTE: To enforce temporal consistency
        # Replace the camera tokens of all views at each timestep with those from the first timestep
        self.replacement = replacement

        # Initialize rotary position embedding if frequency > 0
        # self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        # self.position_getter = PositionGetter() if self.rope is not None else None

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else "none"
        self.rope = None
        if self.pos_type.startswith("rope"):  # eg rope100
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(self.pos_type[len("rope") :])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        if "frame" in aa_order:
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

        if "global" in aa_order:
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

        if "video" in aa_order:
            self.video_blocks = nn.ModuleList(
                [
                    block_fn(
                        dim=fuse_embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=video_init_values,
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

        self.use_checkpoint = use_checkpoint
        self.debug = debug
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

        self.freeze_frame = freeze_frame
        self.freeze_global = freeze_global

    @staticmethod
    def slice_expand_and_flatten(token_tensor, B, F, V):
        """
        Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
        1) Uses the first position (index=0) for the first frame only
        2) Uses the second position (index=1) for all remaining frames (S-1 frames)
        3) Expands both to match batch size B
        4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
        followed by (S-1) second-position tokens
        5) Flattens to (B*S, X, C) for processing

        Returns:
            torch.Tensor: Processed tokens with shape (B*S, X, C)
        """

        # Slice out the "query" tokens => shape (1, 1, ...)
        query = token_tensor[:, None, 0:1, ...].expand(B, F, 1, *token_tensor.shape[2:])
        # Slice out the "other" tokens => shape (1, S-1, ...)
        others = token_tensor[:, None, 1:, ...].expand(B, F, V - 1, *token_tensor.shape[2:])
        # Concatenate => shape (B, S, ...)
        combined = torch.cat([query, others], dim=2)

        # Finally flatten => shape (B*S, ...)
        combined = combined.view(B * F * V, *combined.shape[3:])
        return combined

    def forward(self, patch_tokens, prompt_features, meta_data, **kwargs):
        patch_w = meta_data["input_width"][0] // self.patch_size
        patch_h = meta_data["input_height"][0] // self.patch_size
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        if isinstance(patch_tokens, (list, tuple)):
            patch_tokens = patch_tokens[-1]

        BS, P, C = patch_tokens.shape
        F, V = frame_num, view_num
        S = F * V
        B = BS // S

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = self.slice_expand_and_flatten(self.camera_token, B, F, V)
        if self.num_register_tokens > 0:
            register_token = self.slice_expand_and_flatten(self.register_token, B, F, V)

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
        video_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    if self.freeze_frame:
                        with torch.no_grad():
                            tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                                tokens, B, S, F, V, P, C, frame_idx, pos=pos
                            )
                    else:
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, S, F, V, P, C, frame_idx, pos=pos
                        )
                elif attn_type == "global":
                    if self.freeze_global:
                        with torch.no_grad():
                            tokens, global_idx, global_intermediates = (
                                self._process_global_attention(
                                    tokens, B, S, F, V, P, C, global_idx, pos=pos
                                )
                            )
                    else:
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, F, V, P, C, global_idx, pos=pos
                        )
                elif attn_type == "video":
                    tokens, video_idx, global_intermediates = self._process_video_attention(
                        tokens, B, S, F, V, P, C, video_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(global_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)

                output_list.append(concat_inter)

        # del concat_inter
        # del frame_intermediates
        # del global_intermediates

        if self.hooks is None:
            return output_list[-1], pos, self.patch_start_idx
        else:
            return [output_list[hook] for hook in self.hooks], pos, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, F, V, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        assert S == F * V
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

            if self.replacement and F > 1:
                tokens = tokens.view(B, F, V, P, C)
                tokens[:, 1:, :, 0, :] = tokens[:, 0:1, :, 0, :]
                tokens = tokens.view(B * S, P, C)

            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, F, V, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        assert S == F * V
        if tokens.shape != (B * F, V * P, C):
            tokens = tokens.view(B, S, P, C).view(B * F, V * P, C)

        if pos is not None and pos.shape != (B * F, V * P, 2):
            pos = pos.view(B, S, P, 2).view(B * F, V * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                tokens = checkpoint(
                    self.global_blocks[global_idx], tokens, pos, use_reentrant=False
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)

            if self.replacement and F > 1:
                tokens = tokens.view(B, F, V, P, C)
                tokens[:, 1:, :, 0, :] = tokens[:, 0:1, :, 0, :]
                tokens = tokens.view(B * S, P, C)

            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    def _process_video_attention(self, tokens, B, S, F, V, P, C, video_idx, pos=None):
        """
        Process video attention blocks. We keep tokens in shape (B, S*P, C).
        """
        assert S == F * V
        tokens = (
            tokens.view(B, S, P, C)
            .view(B, F, V, P, C)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(B * V, F * P, C)
        )

        if pos is not None:
            pos = (
                pos.view(B, S, P, 2)
                .view(B, F, V, P, 2)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
                .view(B * V, F * P, 2)
            )

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                tokens = checkpoint(self.video_blocks[video_idx], tokens, pos, use_reentrant=False)
            else:
                tokens = self.video_blocks[video_idx](tokens, pos=pos)

            if self.replacement and F > 1:
                tokens = tokens.view(B, V, F, P, C).permute(0, 2, 1, 3, 4).contiguous()
                tokens[:, 1:, :, 0, :] = tokens[:, 0:1, :, 0, :]
                tokens = tokens.view(B * S, P, C)
            else:
                tokens = (
                    tokens.view(B, V, F, P, C).permute(0, 2, 1, 3, 4).contiguous().view(B * S, P, C)
                )

            video_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, video_idx, intermediates
