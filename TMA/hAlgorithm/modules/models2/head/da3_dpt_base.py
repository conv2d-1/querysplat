from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
from addict import Dict
import logging
from einops import rearrange

from hAlgorithm.modules.models2.external.depth_anything_3.model.dpt import _make_fusion_block, _make_scratch
from hAlgorithm.modules.models2.external.depth_anything_3.model.utils.head_utils import (
    Permute,
    create_uv_grid,
    custom_interpolate,
    position_grid_to_embed,
)


class DPT(nn.Module):
    """
    DPT for dense prediction (main head + optional sky head, sky always 1 channel).

    Returns:
      - Main head:
        * If output_dim>1: { head_name, f"{head_name}_conf" }
        * If output_dim==1: { head_name }
      - Sky head (if use_sky_head=True): { sky_name }  # [B, S, 1, H/down_ratio, W/down_ratio]
    """

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 1,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = False,
        down_ratio: int = 1,
        head_name: str = "depth",
        # ---- sky head (fixed 1 channel) ----
        use_sky_head: bool = True,
        sky_name: str = "sky",
        sky_activation: str = "relu",  # 'sigmoid' / 'relu' / 'linear'
        use_ln_for_heads: bool = False,  # If needed, apply LayerNorm on intermediate features of both heads
        norm_type: str = "idt",  # use to match legacy GS-DPT head, "idt" / "layer"
        fusion_block_inplace: bool = False,
        chunk_size: int=None,
        pretrain: str = None,
        pretrain_strict: bool = True,
        use_conf_head: bool = False,
    ) -> None:
        super().__init__()

        # -------------------- configuration --------------------
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio

        # Names
        self.head_main = head_name
        self.sky_name = sky_name

        # Main head: output dimension and confidence switch
        self.out_dim = output_dim
        self.has_conf = output_dim > 1
        self.use_conf_head = use_conf_head

        # Sky head parameters (always 1 channel)
        self.use_sky_head = use_sky_head
        self.sky_activation = sky_activation

        # Fixed 4 intermediate outputs
        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)

        # -------------------- token pre-norm + per-stage projection --------------------
        if norm_type == "layer":
            self.norm = nn.LayerNorm(dim_in)
        elif norm_type == "idt":
            self.norm = nn.Identity()
        else:
            raise Exception(f"Unknown norm_type {norm_type}, should be 'layer' or 'idt'.")
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # -------------------- Spatial re-size (align to common scale before fusion) --------------------
        # Design consistent with original: relative to patch grid (x4, x2, x1, /2)
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        # -------------------- scratch: stage adapters + main fusion chain --------------------
        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Main fusion chain
        self.scratch.refinenet1 = _make_fusion_block(features, inplace=fusion_block_inplace)
        self.scratch.refinenet2 = _make_fusion_block(features, inplace=fusion_block_inplace)
        self.scratch.refinenet3 = _make_fusion_block(features, inplace=fusion_block_inplace)
        self.scratch.refinenet4 = _make_fusion_block(
            features, has_residual=False, inplace=fusion_block_inplace
        )

        # Heads (shared neck1; then split into two heads)
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )

        ln_seq = (
            [Permute((0, 2, 3, 1)), nn.LayerNorm(head_features_2), Permute((0, 3, 1, 2))]
            if use_ln_for_heads
            else []
        )

        # Main head
        if self.out_dim >= 1:
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                *ln_seq,
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )
        if self.use_conf_head:
            self.scratch.output_conv2_conf = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                *ln_seq,
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            )

        # Sky head (fixed 1 channel)
        if self.use_sky_head:
            self.scratch.sky_output_conv2 = nn.Sequential(
                nn.Conv2d(
                    head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1
                ),
                *ln_seq,
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            )
        
        self.pretrain = pretrain
        self.pretrain_strict = pretrain_strict

        if self.pretrain is not None:
            self.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=self.pretrain_strict,
            )
            logging.info(f"DPT, load pretrain {self.pretrain}")

    # -------------------------------------------------------------------------
    # Public forward (supports frame chunking to save memory)
    # -------------------------------------------------------------------------
    def forward(
        self,
        feats,
        prompt_depth=None,
        patch_h=None,
        patch_w=None,
        meta_data=None,
        patch_start_idx=None,
        **kwargs,
    ):
        """
        Args:
            feats: List of 4 entries, each entry is a tensor like [B, S, T, C] (or the 0th element of tuple/list is that tensor).
            H, W:  Original image dimensions
            patch_start_idx: Starting index of patch tokens in sequence (for cropping non-patch tokens)
            chunk_size:      Chunk size along time dimension S

        Returns:
            Dict[str, Tensor]
        """
        if patch_h is None or patch_w is None:
            patch_w = meta_data["input_width"][0] // self.patch_size
            patch_h = meta_data["input_height"][0] // self.patch_size

        if feats[0].ndim == 3:
            frame_num = meta_data["frames"][0]
            view_num = meta_data["views"][0]
            S = frame_num * view_num
            B, N, C = feats[0].shape
            B = B // S
        elif feats[0].ndim == 4:
            B, S, N, C = feats[0].shape
            feats = [feat.reshape(B * S, N, C) for feat in feats]
        else:
            raise NotImplementedError
        
        # update image info, used by the GS-DPT head
        extra_kwargs = {}
        if "images" in kwargs:
            extra_kwargs.update({"images": rearrange(kwargs["images"], "B S ... -> (B S) ...")})

        if self.training or self.chunk_size is None or self.chunk_size >= S:
            out_dict = self._forward_impl(feats, patch_h, patch_w, patch_start_idx, **extra_kwargs)
            out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
            return Dict(out_dict)

        out_dicts = []
        for s0 in range(0, S, self.chunk_size):
            s1 = min(s0 + self.chunk_size, S)
            kw = {}
            if "images" in extra_kwargs:
                kw.update({"images": extra_kwargs["images"][s0:s1]})
            out_dicts.append(
                self._forward_impl([f[s0:s1] for f in feats], patch_h, patch_w, patch_start_idx, **kw)
            )
        out_dict = {k: torch.cat([od[k] for od in out_dicts], dim=0) for k in out_dicts[0].keys()}
        out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
        return Dict(out_dict)

    # -------------------------------------------------------------------------
    # Internal forward (single chunk)
    # -------------------------------------------------------------------------
    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        ph: int,
        pw: int,
        patch_start_idx: int,
    ):
        B, _, C = feats[0].shape
        H, W = ph * self.patch_size, pw * self.patch_size
        # ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]  # [B*S, N_patch, C]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)  # [B*S, C, ph, pw]

            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)  # Align scale
            resized_feats.append(x)

        # 2) Fusion pyramid (main branch only)
        fused = self._fuse(resized_feats)

        # 3) Upsample to target resolution, optionally add position encoding again
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused = self.scratch.output_conv1(fused)
        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused = self._add_pos_embed(fused, W, H)

        # 4) Shared neck1
        feat = fused

        # 5) Main head: logits -> activation
        outs = {}
        if self.out_dim >= 1:
            main_logits = self.scratch.output_conv2(feat)
            if self.has_conf:
                fmap = main_logits.permute(0, 2, 3, 1)
                pred = self._apply_activation_single(fmap[..., :-1], self.activation)
                conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)
                outs[self.head_main] = pred.squeeze(1)
                outs["confidence"] = conf.squeeze(1)
            else:
                outs[self.head_main] = self._apply_activation_single(
                    main_logits, self.activation
                ).squeeze(1)

        if self.use_conf_head:
            main_logits = self.scratch.output_conv2_conf(feat)
            outs["confidence"] = self._apply_activation_single(
                main_logits, self.conf_activation
            ).squeeze(1)

        # 6) Sky head (fixed 1 channel)
        if self.use_sky_head:
            sky_logits = self.scratch.sky_output_conv2(feat)
            outs[self.sky_name] = self._apply_sky_activation(sky_logits).squeeze(1)

        return outs

    # -------------------------------------------------------------------------
    # Subroutines
    # -------------------------------------------------------------------------
    def _fuse(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        4-layer top-down fusion, returns finest scale features (after fusion, before neck1).
        """
        l1, l2, l3, l4 = feats

        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        # 4 -> 3 -> 2 -> 1
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)
        return out

    def _apply_activation_single(
        self, x: torch.Tensor, activation: str = "linear"
    ) -> torch.Tensor:
        """
        Apply activation to single channel output, maintaining semantic consistency with value branch in multi-channel case.
        Supports: exp / relu / sigmoid / softplus / tanh / linear / expp1
        """
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "expm1":
            return torch.expm1(x)
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        # Default linear
        return x

    def _apply_sky_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sky head activation (fixed 1 channel):
          * 'sigmoid' -> Sigmoid probability map
          * 'relu'    -> ReLU positive domain output
          * 'linear'  -> Original value (logits)
        """
        act = (
            self.sky_activation.lower()
            if isinstance(self.sky_activation, str)
            else self.sky_activation
        )
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "relu":
            return torch.relu(x)
        # 'linear'
        return x

    def _add_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Simple UV position encoding directly added to feature map."""
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe
