"""GMOS-style DA3 + SAM2 feature fusion for pair pyramids."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TokenProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, p_drop: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.drop(self.act(self.proj(x))))
        return x


class SamDa3PyramidFusion(nn.Module):
    """Fuse SAM2 bottleneck maps into DA3 pair pyramids at selected scales.

    For each fused pyramid level ``i``:
        merged = merge_proj( concat( da3_i, interp(sam_proj(sam)) ) )

    Args:
        da3_dims: Channel width per pyramid level (matches ``embed_dims``).
        sam_dim: SAM2 bottleneck channels (256).
        fuse_layer_indices: Which pyramid indices to fuse. ``None`` → last level only.
        fuse_tgt: Fuse target-view pyramids (cross-attn memory path).
        fuse_src: Fuse source-view pyramids (optional).
    """

    def __init__(
        self,
        da3_dims: Sequence[int],
        sam_dim: int = 256,
        p_drop: float = 0.1,
        fuse_layer_indices: Optional[Sequence[int]] = None,
        fuse_tgt: bool = True,
        fuse_src: bool = False,
    ) -> None:
        super().__init__()
        self.da3_dims = list(da3_dims)
        self.fuse_tgt = bool(fuse_tgt)
        self.fuse_src = bool(fuse_src)

        if fuse_layer_indices is None:
            fuse_layer_indices = [len(self.da3_dims) - 1]
        self.fuse_layer_indices = sorted(set(int(i) for i in fuse_layer_indices))

        self.sam_projector = _TokenProjector(sam_dim, sam_dim, p_drop=p_drop)
        self.merge_projectors = nn.ModuleDict()
        for idx in self.fuse_layer_indices:
            out_c = self.da3_dims[idx]
            self.merge_projectors[str(idx)] = _TokenProjector(out_c + sam_dim, out_c, p_drop=p_drop)

    def _fuse_level(self, da3_map: torch.Tensor, sam_map: torch.Tensor, level_idx: int) -> torch.Tensor:
        """``da3_map``: ``(B, C, H, W)``, ``sam_map``: ``(B, 256, Hs, Ws)``."""
        sam = F.interpolate(
            sam_map,
            size=da3_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        sam = self.sam_projector(sam.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        merged = torch.cat([da3_map, sam], dim=1)
        merged = self.merge_projectors[str(level_idx)](merged.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return merged

    def fuse_pyramids(
        self,
        pyramids: List[torch.Tensor],
        sam_maps: torch.Tensor,
    ) -> List[torch.Tensor]:
        out = list(pyramids)
        for idx in self.fuse_layer_indices:
            out[idx] = self._fuse_level(out[idx], sam_maps, idx)
        return out

    def forward(
        self,
        src_pyramids: List[torch.Tensor],
        tgt_pyramids: List[torch.Tensor],
        sam_src: torch.Tensor,
        sam_tgt: torch.Tensor,
    ):
        if self.fuse_src:
            src_pyramids = self.fuse_pyramids(src_pyramids, sam_src)
        if self.fuse_tgt:
            tgt_pyramids = self.fuse_pyramids(tgt_pyramids, sam_tgt)
        return src_pyramids, tgt_pyramids
