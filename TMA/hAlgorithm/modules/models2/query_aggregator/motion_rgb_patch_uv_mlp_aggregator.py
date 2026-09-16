"""Motion MLP aggregator with source RGB patch and direct UV encodings.

Extends ``MotionAggregatorMLP`` by augmenting the dual-frame feature stream
with:
  1. a source-frame RGB patch embedding sampled at each query UV, and
  2. a Fourier embedding of the query UV itself.

The final projection remains ``[B, Q, out_dim]`` so it can reuse the same
shared ``mlp_head.Head`` as the existing motion-only setup.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models2.query_aggregator.motion_mlp_aggregator import (
    MotionAggregatorMLP,
)

logger = logging.getLogger(__name__)


class _FourierFeatureEncoder(nn.Module):
    """Random Fourier feature encoder for low-dimensional continuous inputs."""

    def __init__(self, input_dim: int, out_dim: int, scale: float = 10.0):
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError(f"out_dim must be even, got {out_dim}")
        self.register_buffer("freqs", torch.randn(input_dim, out_dim // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.freqs.to(device=x.device, dtype=x.dtype)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MotionAggregatorRGBPatchUVMLP(MotionAggregatorMLP):
    """Motion aggregator with RGB patch and direct UV query cues."""

    AGGREGATOR_TYPE: str = "motion"

    def __init__(
        self,
        rgb_patch_size: int = 9,
        rgb_patch_dim: int = 256,
        direct_uv_dim: int = 128,
        direct_uv_scale: float = 10.0,
        use_encoder_cam_tokens: bool = False,
        cam_token_proj_dim: int = 256,
        detach_encoder_cam_tokens: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if rgb_patch_size % 2 != 1:
            raise ValueError(f"rgb_patch_size must be odd, got {rgb_patch_size}")

        self.rgb_patch_size = rgb_patch_size
        self.rgb_patch_dim = rgb_patch_dim
        self.direct_uv_dim = direct_uv_dim
        self.use_encoder_cam_tokens = use_encoder_cam_tokens
        self.cam_token_proj_dim = cam_token_proj_dim
        self.detach_encoder_cam_tokens = detach_encoder_cam_tokens
        self._warned_missing_rgb = False
        self._warned_missing_cam_tokens = False

        patch_dim = 3 * rgb_patch_size * rgb_patch_size
        self.rgb_patch_encoder = nn.Sequential(
            nn.Linear(patch_dim, rgb_patch_dim),
            nn.LayerNorm(rgb_patch_dim),
            nn.GELU(),
            nn.Linear(rgb_patch_dim, rgb_patch_dim),
        )

        self.direct_uv_encoder = (
            _FourierFeatureEncoder(input_dim=2, out_dim=direct_uv_dim, scale=direct_uv_scale)
            if direct_uv_dim > 0 else None
        )

        cam_ch = self.in_chans[-1]
        self.cam_pair_proj: Optional[nn.Module] = None
        if use_encoder_cam_tokens:
            self.cam_pair_proj = nn.Sequential(
                nn.LayerNorm(cam_ch * 2),
                nn.Linear(cam_ch * 2, cam_token_proj_dim),
                nn.GELU(),
            )

        fused_dim = self.embed_dims[-1]
        cam_extra = cam_token_proj_dim if use_encoder_cam_tokens else 0
        fusion_in = (
            fused_dim + fused_dim + self.time_fourier_dim + rgb_patch_dim
            + max(direct_uv_dim, 0) + cam_extra
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, self.out_dim),
            nn.GELU(),
        )

    @staticmethod
    def _gather_cam_pair(
        cam_bn_or_bn: torch.Tensor,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        B: int,
        N: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Index per-view camera-slot embeddings for source/target frames.

        Args:
            cam_bn_or_bn: ``[B, N, C]`` or ``[B * N, C]`` (last hook).
            src_idx / tgt_idx: ``[B, Q]`` int64 frame indices.
        """
        if cam_bn_or_bn.ndim == 2:
            cam = cam_bn_or_bn.view(B, N, -1)
        else:
            cam = cam_bn_or_bn
        Q = src_idx.shape[1]
        b_idx = torch.arange(B, device=cam.device, dtype=src_idx.dtype).unsqueeze(1).expand(-1, Q)
        cam_src = cam[b_idx, src_idx]
        cam_tgt = cam[b_idx, tgt_idx]
        return cam_src, cam_tgt

    def _collect_motion_features(
        self,
        pyramid: List[torch.Tensor],
        motion_queries: dict,
        B: int,
        N: int,
        cam_tokens: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        out = super()._collect_motion_features(
            pyramid, motion_queries, B, N, cam_tokens=cam_tokens,
        )
        if not self.use_encoder_cam_tokens:
            return out

        src_idx = out["src_idx"]
        tgt_idx = out["tgt_idx"]
        if cam_tokens is None or len(cam_tokens) == 0:
            if not self._warned_missing_cam_tokens:
                logger.warning(
                    "[MotionAggregatorRGBPatchUVMLP] use_encoder_cam_tokens=True but "
                    "cam_tokens is None; using zeros for camera-slot features."
                )
                self._warned_missing_cam_tokens = True
            C = self.in_chans[-1]
            z = out["src_fused"].new_zeros(out["src_fused"].shape[0], out["src_fused"].shape[1], C)
            out["cam_src"] = z
            out["cam_tgt"] = z
            return out

        cam_last = cam_tokens[-1]
        cam_src, cam_tgt = self._gather_cam_pair(cam_last, src_idx, tgt_idx, B, N)
        cam_src = cam_src.to(out["src_fused"].dtype)
        cam_tgt = cam_tgt.to(out["src_fused"].dtype)
        if self.detach_encoder_cam_tokens:
            cam_src = cam_src.detach()
            cam_tgt = cam_tgt.detach()
        out["cam_src"] = cam_src
        out["cam_tgt"] = cam_tgt
        return out

    def _sample_rgb_patches(
        self,
        rgb_images: torch.Tensor,
        uv: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Sample source-frame RGB patches at the query UVs.

        Args:
            rgb_images: ``[B, N, 3, H, W]`` normalized RGB frames.
            uv: ``[B, Q, 2]`` normalized query coordinates in [0, 1].
            frame_indices: ``[B, Q]`` source frame index per query.
        Returns:
            ``[B, Q, 3 * patch_size^2]`` flattened RGB patches.
        """
        B, N, C, H, W = rgb_images.shape
        if C != 3:
            raise ValueError(f"Expected 3-channel RGB images, got C={C}")

        Q = uv.shape[1]
        patch_area = self.rgb_patch_size * self.rgb_patch_size

        center_grid = uv * 2 - 1
        half = self.rgb_patch_size // 2
        dx = torch.arange(-half, half + 1, device=uv.device, dtype=uv.dtype) * (2.0 / max(W - 1, 1))
        dy = torch.arange(-half, half + 1, device=uv.device, dtype=uv.dtype) * (2.0 / max(H - 1, 1))
        off_y, off_x = torch.meshgrid(dy, dx, indexing="ij")
        offsets = torch.stack([off_x, off_y], dim=-1).reshape(1, 1, patch_area, 2)
        patch_grid = center_grid.unsqueeze(2) + offsets

        frame_indices = frame_indices.clamp(0, N - 1)
        if N == 1:
            norm_t = torch.zeros_like(patch_grid[..., 0])
        else:
            norm_t = (
                2.0 * frame_indices.to(dtype=uv.dtype).unsqueeze(-1).expand(-1, -1, patch_area)
                / float(N - 1) - 1.0
            )

        # Treat frames as the depth axis of a 3D volume so all query/frame pairs
        # are sampled in one CUDA kernel.
        grid_3d = torch.stack(
            [patch_grid[..., 0], patch_grid[..., 1], norm_t], dim=-1
        ).reshape(B, Q, patch_area, 1, 3)
        rgb_volume = rgb_images.permute(0, 2, 1, 3, 4)
        sampled = F.grid_sample(
            rgb_volume,
            grid_3d.to(rgb_volume.dtype),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return (
            sampled.squeeze(-1)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(B, Q, C * patch_area)
        )

    def _build_output_input(
        self,
        motion_features: dict,
        motion_queries: Optional[dict] = None,
        rgb_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src_fused = motion_features["src_fused"]
        tgt_fused = motion_features["tgt_fused"]
        time_emb = motion_features["time_emb"]
        uv = motion_features["uv"]
        src_idx = motion_features["src_idx"]

        parts = [
            src_fused,
            (tgt_fused - src_fused).to(src_fused.dtype),
            time_emb.to(src_fused.dtype),
        ]

        if rgb_images is None:
            if not self._warned_missing_rgb:
                logger.warning(
                    "[MotionAggregatorRGBPatchUVMLP] rgb_images is None; RGB patch features will be zero."
                )
                self._warned_missing_rgb = True
            rgb_feat = src_fused.new_zeros(src_fused.shape[0], src_fused.shape[1], self.rgb_patch_dim)
        else:
            rgb_patch = self._sample_rgb_patches(rgb_images, uv, src_idx)
            rgb_feat = self.rgb_patch_encoder(rgb_patch.to(src_fused.dtype))
        parts.append(rgb_feat.to(src_fused.dtype))

        if self.direct_uv_encoder is not None:
            uv_feat = self.direct_uv_encoder(uv.to(src_fused.dtype))
            parts.append(uv_feat.to(src_fused.dtype))

        if self.use_encoder_cam_tokens and self.cam_pair_proj is not None:
            cam_src = motion_features.get("cam_src")
            cam_tgt = motion_features.get("cam_tgt")
            if cam_src is None or cam_tgt is None:
                C = self.in_chans[-1]
                cam_src = src_fused.new_zeros(*src_fused.shape[:2], C)
                cam_tgt = cam_src
            pair = torch.cat([cam_src, cam_tgt], dim=-1).to(src_fused.dtype)
            parts.append(self.cam_pair_proj(pair))

        return torch.cat(parts, dim=-1)
