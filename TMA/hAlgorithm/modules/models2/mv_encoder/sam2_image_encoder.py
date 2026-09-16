"""Frozen SAM2 image encoder (bottleneck FPN features only).

Vendored SAM2 lives under ``hAlgorithm.modules.models2.external.sam2_vendor``.
This module is only used when explicitly configured; existing pipelines are unchanged.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vendored SAM2 uses top-level ``sam2.*`` imports; register the repo root once.
_SAM2_ROOT = Path(__file__).resolve().parents[1] / "external" / "sam2_vendor" / "sam2"
if str(_SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM2_ROOT))

from hAlgorithm.modules.models2.external.sam2_vendor.sam2.sam2.modeling.backbones.hieradet import Hiera
from hAlgorithm.modules.models2.external.sam2_vendor.sam2.sam2.modeling.backbones.image_encoder import (
    FpnNeck,
    ImageEncoder,
)
from hAlgorithm.modules.models2.external.sam2_vendor.sam2.sam2.modeling.position_encoding import (
    PositionEmbeddingSine,
)

logger = logging.getLogger(__name__)

# ImageNet normalization used by SAM2 (differs from WFM's 127.5 normalize).
_SAM2_MEAN = (0.485, 0.456, 0.406)
_SAM2_STD = (0.229, 0.224, 0.225)
# Hiera PatchEmbed: kernel=7, stride=4, padding=3 → grid = (size - 1) // 4 + 1.
# ``_get_pos_embed`` tiles ``window_pos_embed_window`` (8×8) over that grid.
_SAM2_PATCH_STRIDE = 4
_SAM2_WINDOW_TILE = 8


def _sam2_compatible_image_size(image_size: int) -> int:
    """Snap side length so Hiera patch grid is divisible by the window tile size."""
    patch_grid = (int(image_size) - 1) // _SAM2_PATCH_STRIDE + 1
    if patch_grid % _SAM2_WINDOW_TILE == 0:
        return int(image_size)
    aligned_grid = ((patch_grid + _SAM2_WINDOW_TILE - 1) // _SAM2_WINDOW_TILE) * _SAM2_WINDOW_TILE
    return (aligned_grid - 1) * _SAM2_PATCH_STRIDE + 1


def _build_hiera_large_image_encoder() -> ImageEncoder:
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    )
    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=256,
        backbone_channel_list=[1152, 576, 288, 144],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )
    trunk = Hiera(
        embed_dim=144,
        num_heads=2,
        stages=[2, 6, 36, 4],
        global_att_blocks=[23, 33, 43],
        window_pos_embed_bkg_spatial_size=[7, 7],
        window_spec=[8, 4, 16, 8],
    )
    return ImageEncoder(scalp=1, trunk=trunk, neck=neck)


class SAM2ImageEncoder(nn.Module):
    """Extract SAM2 bottleneck features ``(B, 256, H', W')`` from RGB.

    Args:
        ckpt_path: Checkpoint containing ``image_encoder`` weights (SAM2.1 Hiera-L).
        image_size: Square resize side length before SAM2 encoding.
        freeze: If True, keep weights frozen and force eval mode.
        wfm_normalized_input: If True, input is WFM-normalized ``x/127.5-1`` and will be
            denormalized to ``[0, 1]`` before applying ImageNet stats.
    """

    out_channels = 256

    def __init__(
        self,
        ckpt_path: str,
        image_size: int = 504,
        freeze: bool = True,
        wfm_normalized_input: bool = True,
    ) -> None:
        super().__init__()
        requested_size = int(image_size)
        self.image_size = _sam2_compatible_image_size(requested_size)
        if self.image_size != requested_size:
            logger.warning(
                "SAM2ImageEncoder adjusted image_size %d -> %d "
                "(Hiera patch grid must be divisible by %d)",
                requested_size,
                self.image_size,
                _SAM2_WINDOW_TILE,
            )
        self.freeze = bool(freeze)
        self.wfm_normalized_input = bool(wfm_normalized_input)

        self.image_encoder = _build_hiera_large_image_encoder()
        self._load_ckpt(ckpt_path)

        if self.freeze:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

        mean = torch.tensor(_SAM2_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_SAM2_STD).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def _load_ckpt(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        enc_sd = {k[len("image_encoder."):]: v for k, v in state.items() if k.startswith("image_encoder.")}
        if not enc_sd:
            enc_sd = {k: v for k, v in state.items() if "image_encoder" in k}
        missing, unexpected = self.image_encoder.load_state_dict(enc_sd, strict=False)
        logger.info(
            "SAM2ImageEncoder loaded %s (%d keys, missing=%d, unexpected=%d)",
            ckpt_path,
            len(enc_sd),
            len(missing),
            len(unexpected),
        )

    def _preprocess_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """``rgb``: ``(B, 3, H, W)`` in WFM or raw ``[0,1]`` layout."""
        if self.wfm_normalized_input:
            rgb = rgb * 127.5 + 127.5
            rgb = rgb / 255.0
        rgb = rgb.clamp(0.0, 1.0)
        if rgb.shape[-1] != self.image_size or rgb.shape[-2] != self.image_size:
            rgb = F.interpolate(
                rgb,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (rgb - self._mean) / self._std

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return bottleneck map ``(B, 256, H', W')``."""
        if self.freeze:
            self.eval()
        x = self._preprocess_rgb(rgb)
        backbone_out = self.image_encoder(x)
        return backbone_out["backbone_fpn"][-1]
