import logging

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from hAlgorithm.modules.models.facebookresearch_dinov2_main.dinov2.layers import PatchEmbed
from hAlgorithm.modules.models.pi3.models.layers.pos_embed import PositionGetter, RoPE2D
from hAlgorithm.modules.models.vggt.layers.block import Block
from hAlgorithm.modules.models.vggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D

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
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        pos_type="rope100",
        init_values=0.01,
        hooks=None,
        use_checkpoint=False,
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
        self.hooks = hooks

        self.prompt_in_chans = prompt_in_chans
        self.prompt_embed_dim = prompt_embed_dim
        self.fuse_embed_dim = fuse_embed_dim

        if fuse_embed_dim == 0:
            fuse_embed_dim = self.embed_dim + self.prompt_embed_dim

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

    def forward(self, patch_tokens, prompt_features, meta_data, **kwargs):
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


class RayMixVIT(MixVIT):
    def __init__(self, ray_add=False, **kwargs):

        self.ray_add = ray_add

        if self.ray_add:
            kwargs["fuse_embed_dim"] = kwargs["embed_dim"]
            kwargs["prompt_embed_dim"] = kwargs["embed_dim"]

        super(RayMixVIT, self).__init__(**kwargs)

        if self.ray_add:
            self.fuse_project = None

        self.cache = dict()

    def compute_rays(self, fxfycxcy, h, w, device="cuda"):
        """
        Args:
            fxfycxcy (torch.tensor): [b, v, 4]
            h (int): height of the image
            w (int): width of the image
        Returns:
            ray_o (torch.tensor): [b, v, 3, h, w]
            ray_d (torch.tensor): [b, v, 3, h, w]
        """

        b, v = fxfycxcy.shape[:2]
        fxfycxcy = fxfycxcy.reshape(b * v, 4)

        cache_key = f"{h}_{w}"
        if cache_key in self.cache:
            y, x = self.cache[cache_key]
        else:
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            y, x = y.to(device), x.to(device)
            self.cache[cache_key] = [y, x]

        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_d

    def forward(self, patch_tokens, prompt_features, meta_data, intrinsics=None, **kwargs):

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

        with torch.no_grad():
            ray_d = self.compute_rays(
                fxfycxcy=intrinsics[:, :, [0, 1, 0, 1], [0, 1, 2, 2]],
                h=patch_h * self.patch_size,
                w=patch_w * self.patch_size,
            )
            ray_d = ray_d.view(-1, *ray_d.shape[-3:])

        ray_tokens = self.prompt_patch_embed(ray_d)

        if self.ray_add:
            patch_tokens = patch_tokens + ray_tokens
        else:
            patch_tokens = torch.cat([patch_tokens, ray_tokens], dim=-1)
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
