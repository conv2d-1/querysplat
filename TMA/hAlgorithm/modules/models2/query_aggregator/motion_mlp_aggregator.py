"""Motion MLP Aggregator — dual-frame feature sampling for MLP-based motion prediction.

Samples multi-scale features from both the source and target frames at the
same query UV position, then fuses them with a Fourier time embedding so
that a plain MLP Head (``mlp_head.Head``) can predict 3D displacement
without requiring a cross-attention decoder.

Why this works
--------------
The DA3 ViT-G fuse_encoder applies cross-frame multi-view attention from
layer 13 onward.  Consequently, ``patch_features[tgt_frame, uv_src]`` is
not merely the raw appearance at that pixel in the target frame; it already
encodes context from all other frames.  The feature difference
``tgt_feat - src_feat`` therefore acts as an implicit "motion signal" that
an MLP can decode into a 3D displacement.

Output is shape ``[B, Q, out_dim]`` — same as Aggregator2 — so it plugs
directly into the same ``mlp_head.Head`` used for depth/global-points.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.utils import instantiate_from_config


# ─────────────────────────────────────────────────────────────────────────────
# Fourier encoding for the target-frame time index
# ─────────────────────────────────────────────────────────────────────────────

class _FourierEncoder(nn.Module):
    """Random Fourier feature encoder for scalar inputs."""

    def __init__(self, out_dim: int, scale: float = 5.0):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even"
        freqs = torch.randn(1, out_dim // 2) * scale
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return self.freqs.shape[-1] * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., 1]  →  [..., out_dim]"""
        proj = x @ self.freqs
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────────────────────────────────────

class MotionAggregatorMLP(nn.Module):
    """Dual-frame feature aggregator for MLP-based 3D displacement prediction.

    Class attribute ``AGGREGATOR_TYPE = "motion"`` is used by ``MVQueryUnified``
    to dispatch the correct ``forward`` call signature without modifying stable
    existing aggregator classes (e.g. ``Aggregator2``).

    Architecture
    ------------
    1. **Pyramid build** -- project each encoder hook to ``embed_dims[k]`` via
       a linear layer followed by pixel-shuffle upsampling (identical to
       ``Aggregator2``), producing spatial feature maps for all BxN frames.
    2. **Per-query frame sampling** -- sample the pyramid at query UV
       separately for the *source* frame and each *target* frame.  Source
       indices are uniform within a batch; target indices may vary per query
       and are handled by iterating over unique values.
    3. **Gated multi-scale fusion** — fuse shallow→deep scales with a learned
       gate (same structure as ``Aggregator2``).  The *same* fusion weights
       are applied to both source and target feature lists to encourage a
       shared feature space.
    4. **Time embedding** — encode the normalised target-frame time with
       random Fourier features.
    5. **Output projection** — ``LayerNorm → Linear → GELU`` maps
       ``[src_fused ‖ tgt_fused − src_fused ‖ time_emb]`` to ``out_dim``.

    Args:
        patch_size: ViT patch size (pixels), used to infer spatial grid dims.
        in_chans: Per-hook encoder channel count.
        embed_dims: Per-scale projected channel count after pixel-shuffle.
        upsample_scales: Pixel-shuffle factor per scale.
        intermediate_layer_idx: Which encoder hook indices to use.
        mode: Interpolation mode for ``F.grid_sample``.
        interp_refinenet_cfg: Optional per-scale Conv refinement network.
        time_fourier_dim: Output size of the Fourier time encoder.
        time_fourier_scale: Frequency scale of the Fourier encoder.
        out_dim: Final output dimension per query (should match the dense
            aggregator's output so both can share the same MLP Head).
    """

    # Dispatch tag consumed by MVQueryUnified._call_aggregator.
    AGGREGATOR_TYPE: str = "motion"

    def __init__(
        self,
        patch_size: int = 14,
        in_chans: List[int] = None,
        embed_dims: List[int] = None,
        upsample_scales: List[int] = None,
        intermediate_layer_idx: List[int] = None,
        mode: str = "bilinear",
        interp_refinenet_cfg: Optional[dict] = None,
        time_fourier_dim: int = 128,
        time_fourier_scale: float = 5.0,
        out_dim: int = 256,
    ):
        super().__init__()

        in_chans = list(in_chans or [3072, 3072, 3072, 3072])
        embed_dims = list(embed_dims or [256, 256, 256, 256])
        upsample_scales = list(upsample_scales or [4, 2, 1, 1])

        assert len(in_chans) == len(embed_dims) == len(upsample_scales)

        self.patch_size = patch_size
        self.mode = mode
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.upsample_scales = upsample_scales
        self.intermediate_layer_idx = intermediate_layer_idx

        # ── Pyramid projection (identical structure to Aggregator2) ──────
        self.q_projs = nn.ModuleList()
        for ch_in, ch_out, s in zip(in_chans, embed_dims, upsample_scales):
            self.q_projs.append(nn.Linear(ch_in, ch_out * s * s))

        # Optional per-scale spatial refinement (e.g. small ResNet)
        self.interp_refinenet: Optional[nn.ModuleList] = None
        if interp_refinenet_cfg is not None:
            nets = []
            for ch_out in embed_dims:
                cfg = dict(interp_refinenet_cfg)
                cfg["input_channel"] = ch_out
                nets.append(instantiate_from_config(cfg))
            self.interp_refinenet = nn.ModuleList(nets)

        # ── Gated multi-scale fusion (shared for src and tgt) ────────────
        # Each step fuses the output of scale k−1 (h_proj) into scale k.
        self.h_projs = nn.ModuleList()
        self.gates = nn.ParameterList()
        self.ffns = nn.ModuleList()
        for k in range(len(embed_dims) - 1):
            d_out = embed_dims[k + 1]
            self.h_projs.append(nn.Linear(embed_dims[k], d_out))
            self.gates.append(nn.Parameter(torch.zeros(d_out)))
            self.ffns.append(nn.Sequential(
                nn.Linear(d_out, d_out * 4), nn.GELU(), nn.Linear(d_out * 4, d_out),
            ))

        fused_dim = embed_dims[-1]  # output dim after gated fusion

        # ── Fourier time embedding ────────────────────────────────────────
        self.time_encoder = _FourierEncoder(out_dim=time_fourier_dim, scale=time_fourier_scale)
        self.time_fourier_dim = time_fourier_dim

        # ── Final projection: [src ‖ diff ‖ time] → out_dim ─────────────
        fusion_in = fused_dim + fused_dim + time_fourier_dim
        self.out_proj = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, out_dim),
            nn.GELU(),
        )

        self.out_dim = out_dim

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _build_pyramid(
        self,
        x: List[torch.Tensor],
        patch_h: int,
        patch_w: int,
    ) -> List[torch.Tensor]:
        """Project + reshape + pixel-shuffle encoder tokens → spatial maps.

        Args:
            x:       List of ``[B*N, P, C]`` patch-token tensors (one per hook).
            patch_h: Number of patches in height direction.
            patch_w: Number of patches in width direction.

        Returns:
            List of ``[B*N, C_k, H_k, W_k]`` feature maps, one per scale.
        """
        if self.intermediate_layer_idx is not None:
            x = [x[i] for i in self.intermediate_layer_idx]

        pyramid = []
        for k, feats in enumerate(x):
            # Normalise to [B*N, P, C] — the encoder may return 4-D [B, N, P, C].
            if feats.ndim == 4:
                feats = feats.reshape(feats.shape[0] * feats.shape[1], *feats.shape[2:])
            BN = feats.shape[0]
            proj = self.q_projs[k](feats)                        # [BN, P, C_k*s²]
            proj = proj.transpose(-1, -2).view(BN, -1, patch_h, patch_w)
            proj = F.pixel_shuffle(proj, self.upsample_scales[k])  # [BN, C_k, H_k, W_k]
            if self.interp_refinenet is not None:
                proj = self.interp_refinenet[k](proj)
            pyramid.append(proj)
        return pyramid

    def _sample_frame_at_uv(
        self,
        pyramid: List[torch.Tensor],
        uv: torch.Tensor,
        frame_indices: torch.Tensor,
        B: int,
        N: int,
    ) -> List[torch.Tensor]:
        """Sample each pyramid scale at ``uv`` for per-query ``frame_indices``.

        Iterates over unique frame indices and scatter-adds sampled features
        back via a boolean mask.  Works correctly when ``frame_indices``
        varies per query (typical for target-frame sampling).

        Args:
            pyramid:       Output of :meth:`_build_pyramid`.
            uv:            ``[B, Q, 2]`` normalised query coordinates in [0, 1].
            frame_indices: ``[B, Q]`` integer frame indices.
            B, N:          Batch size and number of frames.

        Returns:
            List of ``[B, Q, C_k]`` per-scale sampled features.
        """
        Q = uv.shape[1]
        grid = (uv * 2 - 1).unsqueeze(2)   # [B, Q, 1, 2]
        sampled_list: List[torch.Tensor] = []

        for feat_map in pyramid:                                      # [B*N, C_k, H_k, W_k]
            C_k = feat_map.shape[1]
            feat_4d = feat_map.view(B, N, C_k, feat_map.shape[-2], feat_map.shape[-1])

            unique_frames = frame_indices.reshape(-1).unique()
            result = torch.zeros(B, Q, C_k, device=uv.device, dtype=feat_map.dtype)

            for f in unique_frames:
                f_int = int(f.item())
                if f_int < 0 or f_int >= N:
                    continue
                mask = (frame_indices == f).unsqueeze(-1).to(feat_map.dtype)  # [B, Q, 1]
                frame_feat = feat_4d[:, f_int]                        # [B, C_k, H_k, W_k]
                s = F.grid_sample(
                    frame_feat, grid.to(frame_feat.dtype),
                    mode=self.mode, align_corners=True, padding_mode="border",
                )                                                      # [B, C_k, Q, 1]
                s = s.squeeze(-1).permute(0, 2, 1).contiguous()       # [B, Q, C_k]
                result = result + s * mask

            sampled_list.append(result)
        return sampled_list

    def _gated_fusion(self, sampled_list: List[torch.Tensor]) -> torch.Tensor:
        """Fuse shallow→deep multi-scale features with learned gates.

        Shared weights are used for both source and target lists to encourage
        a consistent feature space.

        Args:
            sampled_list: Per-scale ``[B, Q, C_k]`` features (shallow first).

        Returns:
            ``[B, Q, C_last]`` fused features.
        """
        hidden = sampled_list[0]
        for k in range(1, len(sampled_list)):
            hidden = self.h_projs[k - 1](hidden)
            gate = torch.sigmoid(self.gates[k - 1]).view(1, 1, -1)
            hidden = sampled_list[k] + gate * hidden
            hidden = self.ffns[k - 1](hidden)
        return hidden

    def _encode_time(
        self,
        tgt_time: Optional[torch.Tensor],
        B: int,
        Q: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode target-frame time for per-query motion prediction."""
        if tgt_time is not None:
            t = tgt_time.unsqueeze(-1)
        else:
            t = torch.full((B, Q, 1), 0.5, device=device, dtype=dtype)
        return self.time_encoder(t.to(dtype))

    def _collect_motion_features(
        self,
        pyramid: List[torch.Tensor],
        motion_queries: dict,
        B: int,
        N: int,
        cam_tokens: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        """Build the shared dual-frame motion features used by MLP decoders.

        Args:
            cam_tokens: Optional per-encoder-hook camera-slot features as
                ``[B, N, C]`` tensors (token index 0 before patch tokens were
                stripped). Subclasses may use these for query-conditioned cues.
        """
        uv = motion_queries["uv"]                         # [B, Q, 2]
        src_idx = motion_queries["src_frame_idx"]         # [B, Q]
        tgt_idx = motion_queries["tgt_frame_idx"]         # [B, Q]
        tgt_time = motion_queries.get("tgt_time")         # [B, Q] or None

        src_list = self._sample_frame_at_uv(pyramid, uv, src_idx, B, N)
        src_fused = self._gated_fusion(src_list)          # [B, Q, fused_dim]

        tgt_list = self._sample_frame_at_uv(pyramid, uv, tgt_idx, B, N)
        tgt_fused = self._gated_fusion(tgt_list)          # [B, Q, fused_dim]

        time_emb = self._encode_time(
            tgt_time=tgt_time,
            B=B,
            Q=uv.shape[1],
            device=uv.device,
            dtype=uv.dtype,
        )

        return {
            "uv": uv,
            "src_idx": src_idx,
            "tgt_idx": tgt_idx,
            "tgt_time": tgt_time,
            "src_fused": src_fused,
            "tgt_fused": tgt_fused,
            "time_emb": time_emb,
        }

    def _build_output_input(
        self,
        motion_features: dict,
        motion_queries: Optional[dict] = None,
        rgb_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the final MLP input from shared motion features.

        Subclasses can override this hook to append extra query-conditioned
        inputs such as RGB patches or direct UV encodings.
        """
        src_fused = motion_features["src_fused"]
        tgt_fused = motion_features["tgt_fused"]
        time_emb = motion_features["time_emb"]
        return torch.cat(
            [src_fused, (tgt_fused - src_fused).to(src_fused.dtype), time_emb.to(src_fused.dtype)],
            dim=-1,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward_with_pyramid(
        self,
        pyramid: List[torch.Tensor],
        motion_queries: dict,
        B: int,
        N: int,
        rgb_images: Optional[torch.Tensor] = None,
        cam_tokens: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run aggregation with a pre-built pyramid (skips ``_build_pyramid``).

        Used for cached inference: the encoder is run once, the spatial
        feature pyramid is built once, and this method is called once per
        target frame without re-running the encoder.

        Args:
            pyramid:       Output of :meth:`_build_pyramid` — list of
                           ``[B*N, C_k, H_k, W_k]`` feature maps.
            motion_queries: Same format as :meth:`forward`.
            B, N:          Batch size and number of frames.

        Returns:
            ``[B, Q, out_dim]`` features ready for the shared MLP Head.
        """
        motion_features = self._collect_motion_features(
            pyramid, motion_queries, B, N, cam_tokens=cam_tokens,
        )
        fused = self._build_output_input(
            motion_features,
            motion_queries=motion_queries,
            rgb_images=rgb_images,
        )
        return self.out_proj(fused)

    def forward(
        self,
        x: List[torch.Tensor],
        motion_queries: dict,
        B: int,
        N: int,
        meta_data: dict,
        rgb: Optional[torch.Tensor] = None,
        cam_tokens: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Aggregate dual-frame features for motion MLP prediction.

        Args:
            x:             List of ``[B*N, P, C]`` patch tokens from the encoder
                           (with prefix tokens already stripped).
            motion_queries: Dict containing:
                           - ``uv``: ``[B, Q, 2]`` normalised query UV in [0, 1].
                           - ``src_frame_idx``: ``[B, Q]`` source frame indices.
                           - ``tgt_frame_idx``: ``[B, Q]`` target frame indices.
                           - ``tgt_time``: ``[B, Q]`` normalised target time.
            B, N:          Batch size and number of frames/views.
            meta_data:     Metadata dict (must contain ``input_width/height``).

        Returns:
            ``[B, Q, out_dim]`` features ready for the shared MLP Head.
        """
        patch_h = int(meta_data["input_height"][0]) // self.patch_size
        patch_w = int(meta_data["input_width"][0]) // self.patch_size

        # Build spatial pyramid for all B*N frames.
        pyramid = self._build_pyramid(x, patch_h, patch_w)
        return self.forward_with_pyramid(
            pyramid=pyramid,
            motion_queries=motion_queries,
            B=B,
            N=N,
            rgb_images=rgb,
            cam_tokens=cam_tokens,
        )
