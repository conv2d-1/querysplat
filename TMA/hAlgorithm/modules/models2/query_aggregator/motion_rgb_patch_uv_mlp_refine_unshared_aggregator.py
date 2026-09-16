"""Multi-pass MLP aggregator with independent (unshared) parameters per pass.

Extends ``MotionAggregatorRGBPatchUVMLPRefine`` by giving each refinement
pass its own ``out_proj`` and ``flow_head`` instead of sharing them.

  Pass 0 (initial):  sample tgt at src UV → out_proj[0] → flow_head[0] → flow_0
  Pass 1 (refine 1): sample tgt at UV+flow_0 → out_proj[1] → flow_head[1] → flow_1
  ...
  Pass K (final):    sample tgt at UV+flow_{K-1} → out_proj[K] → features

All intermediate flow predictions are stored in ``aux`` under keys
``pass_0_flow_2d``, ``pass_1_flow_2d``, ... so the pipeline can apply
per-pass GT supervision with gamma decay.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from hAlgorithm.modules.models2.query_aggregator.motion_rgb_patch_uv_mlp_aggregator import (
    MotionAggregatorRGBPatchUVMLP,
)

logger = logging.getLogger(__name__)


class MotionAggregatorRGBPatchUVMLPRefineUnshared(MotionAggregatorRGBPatchUVMLP):
    """Multi-pass MLP aggregator with fully independent parameters per pass.

    Each of the ``num_refine_passes`` refinement steps and the initial pass
    gets its own ``out_proj`` (LayerNorm → Linear → GELU) and, for all but
    the last pass, its own ``flow_head`` (Linear → 2-D flow).

    Compared with ``MotionAggregatorRGBPatchUVMLPRefine`` (shared params),
    this class trades parameter efficiency for representational flexibility.

    Args:
        num_refine_passes: Number of refinement passes after the initial pass.
            Total passes = 1 + num_refine_passes.
            Each pass except the last emits an intermediate flow stored in
            ``aux`` for supervised training.
        **kwargs: Forwarded to ``MotionAggregatorRGBPatchUVMLP``.
    """

    AGGREGATOR_TYPE: str = "motion"

    def __init__(self, num_refine_passes: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_refine_passes = num_refine_passes

        total_passes = 1 + num_refine_passes  # initial + refine passes

        # Compute fusion_in to build sibling out_proj instances.
        fused_dim = self.embed_dims[-1]
        cam_extra = self.cam_token_proj_dim if getattr(self, "use_encoder_cam_tokens", False) else 0
        fusion_in = (
            fused_dim
            + fused_dim
            + self.time_fourier_dim
            + self.rgb_patch_dim
            + max(self.direct_uv_dim, 0)
            + cam_extra
        )

        # Pass 0 reuses parent's self.out_proj; passes 1..K get fresh instances.
        self.refine_proj_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(fusion_in),
                    nn.Linear(fusion_in, self.out_dim),
                    nn.GELU(),
                )
                for _ in range(num_refine_passes)
            ]
        )

        # Each pass except the last needs a flow head.
        # passes 0 .. num_refine_passes-1 each predict an intermediate flow.
        self.flow_head_list = nn.ModuleList(
            [nn.Linear(self.out_dim, 2) for _ in range(num_refine_passes)]
        )
        for head in self.flow_head_list:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward_with_pyramid(
        self,
        pyramid: List[torch.Tensor],
        motion_queries: dict,
        B: int,
        N: int,
        rgb_images: Optional[torch.Tensor] = None,
        cam_tokens: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> dict:
        if self.num_refine_passes <= 0:
            out = super().forward_with_pyramid(
                pyramid, motion_queries, B, N,
                rgb_images=rgb_images, cam_tokens=cam_tokens, **kwargs,
            )
            # Wrap plain tensor in dict for consistent return type.
            if isinstance(out, torch.Tensor):
                return {"features": out, "aux": {}}
            return out

        uv = motion_queries["uv"]
        src_idx = motion_queries["src_frame_idx"]
        tgt_idx = motion_queries["tgt_frame_idx"]

        motion_features = self._collect_motion_features(
            pyramid, motion_queries, B, N, cam_tokens=cam_tokens,
        )

        fused = self._build_output_input(motion_features, motion_queries, rgb_images)
        out = self.out_proj(fused)  # pass-0 uses parent's out_proj

        aux = {}
        current_uv = uv

        # ── Refinement passes (each with its own out_proj and flow_head) ─────
        for i in range(self.num_refine_passes):
            flow = self.flow_head_list[i](out)                  # (B, Q, 2)
            aux[f"pass_{i}_flow_2d"] = flow

            refined_uv = (current_uv + flow).clamp(0.0, 1.0)
            current_uv = refined_uv

            tgt_list_r = self._sample_frame_at_uv(pyramid, refined_uv, tgt_idx, B, N)
            tgt_fused_r = self._gated_fusion(tgt_list_r)

            motion_features["tgt_fused"] = tgt_fused_r
            fused = self._build_output_input(motion_features, motion_queries, rgb_images)
            out = self.refine_proj_list[i](fused)               # pass-(i+1) proj

        return {
            "features": out,
            "aux": aux,
        }
