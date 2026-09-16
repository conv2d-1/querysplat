# flake8: noqa E501
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict as TyDict
from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
from addict import Dict
from einops import rearrange

from hAlgorithm.modules.models2.external.depth_anything_3.model.utils.head_utils import (
    Permute,
    create_uv_grid,
    custom_interpolate,
    position_grid_to_embed,
)
from hAlgorithm.utils import instantiate_from_config


# -----------------------------------------------------------------------------
# Building blocks (preserved, consistent with original)
# -----------------------------------------------------------------------------
def _make_fusion_block(
    features: int,
    size: Tuple[int, int] = None,
    has_residual: bool = True,
    groups: int = 1,
    inplace: bool = False,
    prompt_inchannel=None,
    prompt_fusion_mode=None,
    out_conv_cfg=None,
) -> nn.Module:
    return FeatureFusionBlock(
        features=features,
        activation=nn.ReLU(inplace=inplace),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
        prompt_inchannel=prompt_inchannel,
        prompt_fusion_mode=prompt_fusion_mode,
        out_conv_cfg=out_conv_cfg,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    # Optional expansion by stage
    c1 = out_shape
    c2 = out_shape * (2 if expand else 1)
    c3 = out_shape * (4 if expand else 1)
    c4 = out_shape * (8 if expand else 1)

    scratch.layer1_rn = nn.Conv2d(in_shape[0], c1, 3, 1, 1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], c2, 3, 1, 1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], c3, 3, 1, 1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], c4, 3, 1, 1, bias=False, groups=groups)
    return scratch


class ResidualConvUnit(nn.Module):
    """Lightweight residual convolution block for fusion"""

    def __init__(self, features: int, activation: nn.Module, bn: bool, groups: int = 1) -> None:
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FeatureFusionBlock(nn.Module):
    """Top-down fusion block: (optional) residual merge + upsampling + 1x1 contraction"""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: Tuple[int, int] = None,
        has_residual: bool = True,
        groups: int = 1,
        prompt_inchannel=None,
        prompt_fusion_mode=None,
        out_conv_cfg=None,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.has_residual = has_residual
        self.out_conv_cfg = out_conv_cfg

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=groups) if has_residual else None
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=groups)

        out_features = (features // 2) if expand else features

        if self.out_conv_cfg is not None:
            self.out_conv_cfg["input_channel"] = features
            self.out_conv_pre = instantiate_from_config(self.out_conv_cfg)

        self.out_conv = nn.Conv2d(features, out_features, 1, 1, 0, bias=True, groups=groups)
        self.skip_add = nn.quantized.FloatFunctional()

        self.prompt_inchannel = prompt_inchannel
        self.prompt_fusion_mode = prompt_fusion_mode
        if self.prompt_inchannel is not None and self.prompt_inchannel > 0:
            self.resConfUnit_depth = nn.Sequential(
                nn.Conv2d(
                    self.prompt_inchannel,
                    features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    groups=1,
                ),
                activation,
                nn.Conv2d(
                    features,
                    features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    groups=1,
                ),
                activation,
                zero_module(
                    nn.Conv2d(
                        features,
                        features,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        groups=1,
                    )
                ),
            )

            if self.prompt_fusion_mode == "add":
                self.resFusion_depth = nn.Identity()
            elif self.prompt_fusion_mode == "cat":
                self.resFusion_depth = nn.Conv2d(
                    features * 2,
                    features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                    groups=1,
                )
            else:
                raise NotImplementedError

    def forward(self, *xs: torch.Tensor, size: Tuple[int, int] = None, prompt_depth=None) -> torch.Tensor:  # type: ignore[override]
        """
        xs:
          - xs[0]: Top branch input
          - xs[1]: Lateral input (can do residual addition with top branch)
        """
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = self.skip_add.add(y, self.resConfUnit1(xs[1]))

        y = self.resConfUnit2(y)

        if prompt_depth is not None and self.prompt_inchannel is not None and self.prompt_inchannel > 0:
            prompt_depth = torch.nn.functional.interpolate(prompt_depth, y.shape[2:], mode="bilinear", align_corners=False)
            res = self.resConfUnit_depth(prompt_depth)
            if self.prompt_fusion_mode == "add":
                y = self.skip_add.add(y, res)
            elif self.prompt_fusion_mode == "cat":
                y = self.skip_add.cat([y, res], dim=1)
                y = self.resFusion_depth(y)
            else:
                raise NotImplementedError

        # Upsampling
        if (size is None) and (self.size is None):
            up_kwargs = {"scale_factor": 2}
        elif size is None:
            up_kwargs = {"size": self.size}
        else:
            up_kwargs = {"size": size}

        y = custom_interpolate(y, **up_kwargs, mode="bilinear", align_corners=self.align_corners)
        if self.out_conv_cfg is not None:
            y = self.out_conv_pre(y)
        y = self.out_conv(y)
        return y


class DPT(nn.Module):
    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        # head_names: Tuple[str, str] = ("depth", "ray"),
        chunk_size: int = None,
        pretrain: str = None,
        pretrain_strict: bool = True,
        prompt_inchannel=None,
        prompt_flag=None,
        prompt_fusion_mode=None,
        pred_confidence=False,
        pred_normal=False,
        pred_invalid_mask=False,
        interp_refinenet_cfg=None,
    ) -> None:
        super().__init__()

        # -------------------- configuration --------------------
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio

        self.prompt_inchannel = prompt_inchannel
        self.prompt_flag = prompt_flag if prompt_flag is not None else [True, True, True, True]
        self.prompt_fusion_mode = prompt_fusion_mode

        self.interp_refinenet_cfg = interp_refinenet_cfg
        if self.interp_refinenet_cfg is not None:
            self.interp_refinenet_cfg["input_channel"] = features // 2
            self.interp_refinenet = instantiate_from_config(self.interp_refinenet_cfg)

        # Always expect 4 scales; enforce intermediate idx = (0, 1, 2, 3)
        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)

        # -------------------- token pre-norm + per-stage projection --------------------
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels])

        # -------------------- spatial re-sizers (align to common scale before fusion) --------------------
        # design: stage strides (x4, x2, x1, /2) relative to patch grid to align to a common pivot scale
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        # -------------------- scratch: stage adapters + fusion (main & aux are separate) --------------------
        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Main fusion chain (independent)
        self.scratch.refinenet1 = _make_fusion_block(features, prompt_inchannel=self.prompt_inchannel if self.prompt_flag[0] else 0, prompt_fusion_mode=self.prompt_fusion_mode, out_conv_cfg=self.interp_refinenet_cfg)
        self.scratch.refinenet2 = _make_fusion_block(features, prompt_inchannel=self.prompt_inchannel if self.prompt_flag[0] else 0, prompt_fusion_mode=self.prompt_fusion_mode, out_conv_cfg=self.interp_refinenet_cfg)
        self.scratch.refinenet3 = _make_fusion_block(features, prompt_inchannel=self.prompt_inchannel if self.prompt_flag[0] else 0, prompt_fusion_mode=self.prompt_fusion_mode, out_conv_cfg=self.interp_refinenet_cfg)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False, prompt_inchannel=self.prompt_inchannel if self.prompt_flag[0] else 0, prompt_fusion_mode=self.prompt_fusion_mode, out_conv_cfg=self.interp_refinenet_cfg)

        # Primary head neck + head (independent)
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

        self.pred_confidence = pred_confidence
        if self.pred_confidence:
            self.conf_conv = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            )

        self.pred_normal = pred_normal
        if self.pred_normal:
            self.normal_conv = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, 3, kernel_size=1, stride=1, padding=0),
            )

        self.pred_invalid_mask = pred_invalid_mask
        if self.pred_invalid_mask:
            self.invalid_mask_conv = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
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
            aggregated_tokens_list: List of 4 tensors [B, S, T, C] from transformer.
            images:                [B, S, 3, H, W], in [0, 1].
            patch_start_idx:       Patch-token start in the token sequence (to drop non-patch tokens).
            frames_chunk_size:     Optional chunking along S for memory.

        Returns:
            Dict[str, Tensor] with keys based on `head_names`, e.g.:
                self.head_main, f"{self.head_main}_conf",
                self.head_aux,  f"{self.head_aux}_conf"
            Shapes:
              main:    [B, S, out_dim, H/down_ratio, W/down_ratio]
              main_cf: [B, S, 1,       H/down_ratio, W/down_ratio]
              aux:     [B, S, 7,       H/down_ratio, W/down_ratio]
              aux_cf:  [B, S, 1,       H/down_ratio, W/down_ratio]
        """
        if patch_h is None or patch_w is None:
            patch_w = meta_data["input_width"][0] // self.patch_size
            patch_h = meta_data["input_height"][0] // self.patch_size

        B, S, N, C = feats[0].shape
        feats = [feat.reshape(B * S, N, C) for feat in feats]

        if self.training or self.chunk_size is None or self.chunk_size >= S:
            out_dict = self._forward_impl(feats, patch_h, patch_w, patch_start_idx, prompt_depth=prompt_depth)
            out_dict = {k: v.reshape(B, S, *v.shape[1:]) for k, v in out_dict.items()}
            return Dict(out_dict)

        out_dicts = []
        for s0 in range(0, S, self.chunk_size):
            s1 = min(s0 + self.chunk_size, S)
            out_dict = self._forward_impl(
                [feat[s0:s1] for feat in feats],
                patch_h,
                patch_w,
                patch_start_idx,
                prompt_depth=prompt_depth,
            )
            out_dicts.append(out_dict)

        out_dict = {k: torch.cat([out_dict[k] for out_dict in out_dicts], dim=0) for k in out_dicts[0].keys()}
        out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
        return Dict(out_dict)

    @torch.no_grad()
    def onnx_forward(
        self,
        feats,
        patch_h,
        patch_w,
        prompt_depth=None,
        patch_start_idx=None,
        **kwargs,
    ):
        B, S, N, C = feats[0].shape
        feats = [feat.reshape(B * S, N, C) for feat in feats]
        chunk_size = self.chunk_size if self.chunk_size is not None else 1
        out_dicts = []
        for s0 in range(0, S, chunk_size):
            s1 = min(s0 + chunk_size, S)
            
            cur_feats = [feat[s0:s1] for feat in feats]
            ph = patch_h
            pw = patch_w
            B, _, C = cur_feats[0].shape
            H, W = ph * self.patch_size, pw * self.patch_size
            resized_feats = []
            for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
                x = cur_feats[take_idx][:, patch_start_idx:]
                x = self.norm(x)
                x = x.permute(0, 2, 1).reshape(B, C, ph, pw)  # [B*S, C, ph, pw]

                x = self.projects[stage_idx](x)
                if self.pos_embed:
                    pe = kwargs.get(f"pos_stage{stage_idx}")
                    x = x + pe
                x = self.resize_layers[stage_idx](x)  # align scales
                resized_feats.append(x)

            # 2) Fuse pyramid (main & aux are completely independent)
            fused_main = self._fuse(resized_feats, prompt_depth=prompt_depth)

            # 3) Upsample to target resolution and (optional) add pos-embed again
            h_out = int(ph * self.patch_size / self.down_ratio)
            w_out = int(pw * self.patch_size / self.down_ratio)

            fused_main = custom_interpolate(fused_main, (h_out, w_out), mode="bilinear", align_corners=True)
            if self.pos_embed:
                pe = kwargs.get(f"pos_stage{len(self.intermediate_layer_idx)}")
                fused_main = fused_main + pe

            if self.interp_refinenet_cfg is not None:
                fused_main = self.interp_refinenet(fused_main)

            # Primary head: conv1 -> conv2 -> activate
            # fused_main = self.scratch.output_conv1(fused_main)
            main_logits = self.scratch.output_conv2(fused_main)
            fmap = main_logits.permute(0, 2, 3, 1)
            main_pred = self._apply_activation_single(fmap[..., :-1], self.activation)
            main_conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)

            cur_out = dict(
                depth=main_pred.squeeze(-1).unsqueeze(-3),
                confidence=main_conf.unsqueeze(-3),
            )

            if self.pred_normal:
                normal = self.normal_conv(fused_main)
                cur_out["normal"] = normal

            if self.pred_invalid_mask:
                invalid_mask = self.invalid_mask_conv(fused_main)
                cur_out["invalid_mask"] = invalid_mask
            
            out_dicts.append(cur_out)

        out_dict = {k: torch.cat([out_dict[k] for out_dict in out_dicts], dim=0) for k in out_dicts[0].keys()}
        out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
        return Dict(out_dict)

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        ph: int,
        pw: int,
        patch_start_idx: int,
        prompt_depth=None,
    ) -> Dict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        H, W = ph * self.patch_size, pw * self.patch_size

        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)  # [B*S, C, ph, pw]

            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)  # align scales
            resized_feats.append(x)

        # 2) Fuse pyramid (main & aux are completely independent)
        fused_main = self._fuse(resized_feats, prompt_depth=prompt_depth)

        # 3) Upsample to target resolution and (optional) add pos-embed again
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused_main = custom_interpolate(fused_main, (h_out, w_out), mode="bilinear", align_corners=True)
        
        if self.interp_refinenet_cfg is not None:
            fused_main = self.interp_refinenet(fused_main)
        
        if self.pos_embed:
            fused_main = self._add_pos_embed(fused_main, W, H)

        # Primary head: conv1 -> conv2 -> activate
        # fused_main = self.scratch.output_conv1(fused_main)
        main_logits = self.scratch.output_conv2(fused_main)
        fmap = main_logits.permute(0, 2, 3, 1)
        main_pred = self._apply_activation_single(fmap[..., :-1], self.activation)
        main_conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)

        results = dict(
            depth=main_pred.squeeze(-1).unsqueeze(-3),
            confidence=main_conf.unsqueeze(-3),
        )

        if self.pred_normal:
            normal = self.normal_conv(fused_main)
            results["normal"] = normal

        if self.pred_invalid_mask:
            invalid_mask = self.invalid_mask_conv(fused_main)
            results["invalid_mask"] = invalid_mask

        return results

    def _fuse(self, feats: List[torch.Tensor], prompt_depth=None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Feature pyramid fusion.
        Returns:
            fused_main: Tensor at finest scale (after refinenet1)
            aux_pyr:    List of aux tensors at each level (pre out_conv1_aux)
        """
        l1, l2, l3, l4 = feats

        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        # level 4 -> 3
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:], prompt_depth=prompt_depth if self.prompt_flag[3] else None)
        # level 3 -> 2
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:], prompt_depth=prompt_depth if self.prompt_flag[2] else None)
        # level 2 -> 1
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:], prompt_depth=prompt_depth if self.prompt_flag[1] else None)
        # level 1 (final)
        out = self.scratch.refinenet1(out, l1_rn, prompt_depth=prompt_depth if self.prompt_flag[0] else None)
        out = self.scratch.output_conv1(out)

        return out

    def _prepare_pos_embed(self, pw, ph, channel, x, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, channel) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(1, -1, -1, -1)
        return pe

    def _add_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Simple UV positional embedding added to feature maps."""
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe

    def _apply_activation_single(self, x: torch.Tensor, activation: str = "linear") -> torch.Tensor:
        """
        Apply activation to single channel output, maintaining semantic consistency with value branch in multi-channel case.
        Supports: exp / relu / sigmoid / softplus / tanh / linear / expp1
        """
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expm1":
            return torch.expm1(x)
        if act == "expp1":
            return torch.exp(x) + 1
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


class MonoDPT(nn.Module):
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
        # head_name: str = "depth",
        # ---- sky head (fixed 1 channel) ----
        use_sky_head: bool = True,
        # sky_name: str = "sky",
        # sky_activation: str = "relu",  # 'sigmoid' / 'relu' / 'linear'
        use_ln_for_heads: bool = False,  # If needed, apply LayerNorm on intermediate features of both heads
        norm_type: str = "idt",  # use to match legacy GS-DPT head, "idt" / "layer"
        fusion_block_inplace: bool = False,
        chunk_size=None,
        pred_confidence=False,
        pred_normal=False,
        pred_invalid_mask=False,
        pretrain=None,
        pretrain_strict=True,
    ) -> None:
        super().__init__()

        """
        DPT for dense prediction (main head + optional sky head, sky always 1 channel).

        Returns:
        - Main head:
            * If output_dim>1: { head_name, f"{head_name}_conf" }
            * If output_dim==1: { head_name }
        - Sky head (if use_sky_head=True): { sky_name }  # [B, S, 1, H/down_ratio, W/down_ratio]
        """


        # -------------------- configuration --------------------
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.chunk_size = chunk_size

        # Names
        # self.head_main = head_name
        # self.sky_name = sky_name

        # Main head: output dimension and confidence switch
        self.out_dim = output_dim
        # self.has_conf = output_dim > 1

        # Sky head parameters (always 1 channel)
        # self.use_sky_head = use_sky_head
        # self.sky_activation = sky_activation

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
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            *ln_seq,
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

        # Sky head (fixed 1 channel)
        self.pred_confidence = pred_confidence
        if self.pred_confidence:
            self.scratch.conf_output_conv2 = nn.Sequential(
                nn.Conv2d(
                    head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1
                ),
                *ln_seq,
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            )
        
        self.pred_normal = pred_normal
        if self.pred_normal:
            self.scratch.normal_output_conv2 = nn.Sequential(
                nn.Conv2d(
                    head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1
                ),
                *ln_seq,
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, 3, kernel_size=1, stride=1, padding=0),
            )
        
        self.pred_invalid_mask = pred_invalid_mask
        if self.pred_invalid_mask:
            self.scratch.invalid_mask_output_conv2 = nn.Sequential(
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

        B, S, N, C = feats[0].shape
        feats = [feat.reshape(B * S, N, C) for feat in feats]

        # update image info, used by the GS-DPT head
        extra_kwargs = {}
        if "images" in kwargs:
            extra_kwargs.update({"images": rearrange(kwargs["images"], "B S ... -> (B S) ...")})

        if self.chunk_size is None or self.chunk_size >= S:
            out_dict = self._forward_impl(feats, patch_h, patch_w, patch_start_idx, **extra_kwargs)
            out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
            return Dict(out_dict)

        out_dicts: List[TyDict[str, torch.Tensor]] = []
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
    ) -> TyDict[str, torch.Tensor]:
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
        main_logits = self.scratch.output_conv2(feat)
        outs: TyDict[str, torch.Tensor] = {}

        outs["depth"] = self._apply_activation_single(main_logits, self.activation)

        # 6) Sky head (fixed 1 channel)
        if self.pred_confidence:
            outs["confidence"] = self.scratch.conf_output_conv2(feat)
            # outs[self.sky_name] = self._apply_sky_activation(sky_logits).squeeze(1)

        if self.pred_normal:
            outs["normal"] = self.scratch.normal_output_conv2(feat)

        if self.pred_invalid_mask:
            outs["invalid_mask"] = self.scratch.invalid_mask_output_conv2(feat)
        

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