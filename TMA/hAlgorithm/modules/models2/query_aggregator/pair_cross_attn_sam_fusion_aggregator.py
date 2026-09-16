"""Pair cross-attention aggregator with optional GMOS-style SAM2 fusion (scheme A).

When ``sam_encoder`` / ``sam_fusion`` are omitted, behaviour is identical to
``unified_query.PairCrossAttnAggregator``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from hAlgorithm.modules.models2.query_aggregator.unified_query import PairCrossAttnAggregator
from hAlgorithm.utils import instantiate_from_config


class PairCrossAttnSamFusionAggregator(PairCrossAttnAggregator):
    """Fuse frozen SAM2 semantics into pair tgt/src pyramids before cross-attention.

    SAM2 is only run on views referenced by ``pair_idx`` (typically 2 views per pair),
    not on the full multi-view tensor.
    """

    def __init__(
        self,
        sam_encoder=None,
        sam_fusion=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sam_encoder = instantiate_from_config(sam_encoder) if sam_encoder else None
        self.sam_fusion = instantiate_from_config(sam_fusion) if sam_fusion else None

    @property
    def sam_fusion_enabled(self) -> bool:
        return self.sam_encoder is not None and self.sam_fusion is not None

    def _encode_pair_view_rgb(
        self,
        rgb: torch.Tensor,
        pair_idx: Sequence[Tuple[int, int]],
        view_indices: Sequence[int],
        B: int,
        N: int,
        num_pair: int,
    ) -> torch.Tensor:
        """Encode selected views; return ``(B * num_pair, C, H', W')`` aligned to pair order."""
        if rgb.ndim == 5:
            rgb_bn = rgb.view(B, N, *rgb.shape[-3:])
        else:
            rgb_bn = rgb.unsqueeze(1)

        flat_imgs = []
        for pair_pos, vidx in enumerate(view_indices):
            for b in range(B):
                flat_imgs.append(rgb_bn[b, vidx])
        batch = torch.stack(flat_imgs, dim=0)

        sam_maps = self.sam_encoder(batch)

        # Reorder to (B, num_pair, C, H, W) then flatten pair dim
        c, h, w = sam_maps.shape[1:]
        sam_maps = sam_maps.view(B, num_pair, c, h, w)
        return sam_maps.view(B * num_pair, c, h, w)

    def _apply_pair_pyramid_fusion(
        self,
        src_pyramids: List[torch.Tensor],
        tgt_pyramids: List[torch.Tensor],
        pair_idx=None,
        meta_data=None,
        B: int = 1,
        N: int = 1,
        num_pair: int = 1,
        rgb: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not self.sam_fusion_enabled or rgb is None or pair_idx is None:
            return src_pyramids, tgt_pyramids

        src_indices = [src for src, _ in pair_idx]
        tgt_indices = [tgt for _, tgt in pair_idx]

        sam_src = self._encode_pair_view_rgb(rgb, pair_idx, src_indices, B, N, num_pair)
        sam_tgt = self._encode_pair_view_rgb(rgb, pair_idx, tgt_indices, B, N, num_pair)

        return self.sam_fusion(src_pyramids, tgt_pyramids, sam_src, sam_tgt)
