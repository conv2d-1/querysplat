"""Cross-Attention AdaLN Flow Head (v48).

Combines three key design ingredients:

1. **Cross-attention** to target-frame encoder tokens — explicit target awareness.
2. **AdaLN** conditioning by target time at every decoder layer.
3. **2D flow readout** from attention weights, supervised by ``trajs_2d``.

The attention mechanism serves a dual purpose: it locates the correct 2D
correspondence in the target frame (gradient shaped by the ``trajs_2d``
loss) *and* aggregates target features for 3D displacement prediction.
This naturally couples 2D matching and 3D regression through a single,
unified decoder — no separate coarse-matching or local-window stages.

Architecture (three-module pattern, aligned with sv_query)
----------------------------------------------------------
::

    query_feats_aggregator (MotionQueryAggregator):
        Source features ← multi-scale bilinear sample at query UV
        Query = MLP(src_feat ‖ Fourier(UV))

    query_decoder (MotionQueryDecoder):
        Memory = target-frame deepest-scale tokens + Fourier(UV_grid) pos enc
        Decoder: K × [AdaLN(time) → CrossAttn → AdaLN(time) → FFN]
        Output: pred_3d = Linear(decoded)
        Flow:   pred_flow = Σ(attn_weight · mem_uv) − query_uv
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.utils import instantiate_from_config
from .sparse_motion_head import (
    FourierFeatureEncoder,
    MultiScaleImplicitSampler,
    TokenAdaLN,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder building block
# ─────────────────────────────────────────────────────────────────────────────


class AdaLNCrossAttnDecoderLayer(nn.Module):
    """Single decoder layer: AdaLN → CrossAttn → AdaLN → FFN.

    Attention weights can be returned from the last layer for flow readout.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        cond_dim: int = 128,
    ):
        super().__init__()
        self.adaln_attn = TokenAdaLN(d_model, cond_dim)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm_kv = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)

        self.adaln_ffn = TokenAdaLN(d_model, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        time_cond: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            tgt:          [B, Q, D]
            memory:       [B, S, D]
            time_cond:    [B, Q, cond_dim]
            need_weights: return per-head attention weights [B, H, Q, S]
        """
        q_norm = self.adaln_attn(tgt, time_cond)
        mem_norm = self.norm_kv(memory)
        attn_out, attn_weights = self.cross_attn(
            query=q_norm,
            key=mem_norm,
            value=mem_norm,
            need_weights=need_weights,
            average_attn_weights=False,
        )

        if not torch.isfinite(attn_out).all():
            finite = torch.isfinite(attn_out)
            logger.warning(
                "[AdaLNCrossAttn] %d NaN/Inf — zeroing",
                (~finite).sum().item(),
            )
            attn_out = torch.where(
                finite, attn_out, torch.zeros_like(attn_out.detach()),
            )

        tgt = tgt + self.dropout_attn(attn_out)
        tgt = tgt + self.ffn(self.adaln_ffn(tgt, time_cond))
        return tgt, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 of 3: MotionQueryAggregator  (≈ query_feats_aggregator in sv_query)
# ─────────────────────────────────────────────────────────────────────────────


class MotionQueryAggregator(nn.Module):
    """Aggregates multi-scale source-frame features at query UV positions.

    Equivalent to ``Aggregator2`` / ``query_feats_aggregator`` in sv_query,
    adapted for the motion setting (multi-view video, explicit frame index).

    Pipeline::

        encoder_features → build_pyramid() → pyramid
        pyramid + query.uv + src_frame_idx → sample_and_fuse → src_fused
        src_fused ‖ Fourier(UV) → query_proj → [B, Q, hidden_dim]

    Parameters
    ----------
    feature_dim : int
        Encoder output channel count (e.g. 2*1024 for MOVIES ViT-L).
    scale_dims : list[int]
        Per-scale projection dims.  Shallow→deep, coarsest dim last.
    upsample_factors : list[int]
        Spatial upsample factor per scale (shallow levels get higher resolution).
    use_learned_upsample : bool
        Use ConvTranspose2d upsamplers (True) or nn.Identity (False).
    uv_fourier_dim : int
        Output dimensionality of the Fourier UV encoder.
    uv_fourier_scale : float
        Frequency scale for Fourier UV encoding.
    hidden_dim : int
        Output dimensionality after ``query_proj``.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        scale_dims: Optional[list] = None,
        upsample_factors: Optional[list] = None,
        use_learned_upsample: bool = True,
        uv_fourier_dim: int = 128,
        uv_fourier_scale: float = 8.0,
        hidden_dim: int = 512,
    ):
        super().__init__()
        scale_dims = list(scale_dims or [256, 512, 1024])
        upsample_factors = list(upsample_factors or [4, 2, 1])

        self.multiscale_sampler = MultiScaleImplicitSampler(
            feature_dim=feature_dim,
            scale_dims=scale_dims,
            upsample_factors=upsample_factors,
            use_learned_upsample=use_learned_upsample,
        )
        self.uv_encoder = FourierFeatureEncoder(2, uv_fourier_dim, uv_fourier_scale)

        input_dim = self.multiscale_sampler.output_dim + uv_fourier_dim
        self.query_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

        self.hidden_dim = hidden_dim
        # expose for decoder: deepest pyramid level channel count
        self.deepest_scale_dim = scale_dims[-1]

    # ── pyramid helpers ──────────────────────────────────────────────────

    @staticmethod
    def _infer_hw(num_tokens: int, aspect_ratio: float = 1.0) -> tuple[int, int]:
        """Infer (H_feat, W_feat) from token count respecting aspect_ratio."""
        best_h, best_w, min_diff = 1, num_tokens, float("inf")
        for w in range(1, int(math.sqrt(num_tokens)) + 2):
            if num_tokens % w != 0:
                continue
            h = num_tokens // w
            for ch, cw in [(h, w), (w, h)]:
                diff = abs(ch / max(cw, 1) - aspect_ratio)
                if diff < min_diff:
                    min_diff, best_h, best_w = diff, ch, cw
        return best_h, best_w

    def build_pyramid(
        self,
        features,
        patch_start_idx: int,
        img_size: Optional[tuple[int, int]],
    ) -> list[torch.Tensor]:
        """Project + reshape + upsample encoder features → list of [BV, C_k, H_k, W_k].

        Args:
            features:        List of multi-hook encoder tensors (or single tensor).
            patch_start_idx: Number of prefix tokens (camera/cls) to strip.
            img_size:        (H, W) of the input image; used to infer H_feat, W_feat.
        """
        feat_last = features[-1] if isinstance(features, (list, tuple)) else features
        if feat_last.ndim == 4:
            feat_last = feat_last.reshape(-1, *feat_last.shape[2:])
        spatial_tokens = feat_last.shape[1] - patch_start_idx
        img_h, img_w = img_size if img_size else (518, 518)
        feat_h, feat_w = self._infer_hw(spatial_tokens, img_h / max(img_w, 1))
        multi_features = (
            list(features) if isinstance(features, (list, tuple)) else [features]
        )
        return self.multiscale_sampler._build_pyramid(
            multi_features, patch_start_idx, feat_h, feat_w,
        )

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        pyramid: list[torch.Tensor],
        uv: torch.Tensor,
        src_frame_idx: int,
        B: int,
        num_views: int,
    ) -> torch.Tensor:
        """Sample source features at query UV and produce query embeddings.

        Args:
            pyramid:       Output of :meth:`build_pyramid`.
            uv:            [B, Q, 2] normalised query coordinates in [0, 1].
            src_frame_idx: Which view/frame to sample source features from.
            B:             Batch size.
            num_views:     Number of views (V) in the sequence.

        Returns:
            [B, Q, hidden_dim] query embeddings.
        """
        src_fused = self.multiscale_sampler.sample_and_fuse(
            pyramid, uv, src_frame_idx, B, num_views,
        )
        uv_feat = self.uv_encoder(uv)
        work_dtype = src_fused.dtype
        return self.query_proj(
            torch.cat([src_fused.to(work_dtype), uv_feat.to(work_dtype)], dim=-1),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 of 3: MotionQueryDecoder  (≈ query_decoder in sv_query)
# ─────────────────────────────────────────────────────────────────────────────


class MotionQueryDecoder(nn.Module):
    """AdaLN cross-attention decoder: query embeddings → 3D displacement + 2D flow.

    Equivalent to ``Head`` / ``query_decoder`` in sv_query, but replaces the
    plain MLP with K AdaLN cross-attention layers that attend to target-frame
    patch tokens.  The attention weights also yield a soft 2D flow prediction.

    Pipeline::

        query_embeds [B, Q, hidden_dim]
        + deepest_pyramid_map [BV, C_d, H_t, W_t]
        + queries (tgt_frame_idx, tgt_time)
        →  K × [AdaLN(time) → CrossAttn(target_tokens) → AdaLN → FFN]
        →  pred_3d  [B, Q, 3]
        →  pred_flow [B, Q, 2]   (if predict_flow_aux=True)

    Parameters
    ----------
    hidden_dim : int
        Query and memory representation dimension.
    num_layers : int
        Number of AdaLN cross-attention decoder layers.
    num_heads : int
        Number of attention heads in each layer.
    deepest_scale_dim : int
        Channel count of the deepest pyramid level (used for memory projection).
    time_fourier_dim, time_fourier_scale : int, float
        Fourier encoding for target-frame time conditioning (AdaLN input).
    memory_fourier_dim, memory_fourier_scale : int, float
        Fourier positional encoding added to target memory tokens.
    decoder_ffn_ratio : int
        FFN hidden size = hidden_dim × decoder_ffn_ratio.
    decoder_dropout : float
        Dropout in attention and FFN.
    predict_flow_aux : bool
        If True, compute soft 2D flow from last-layer attention weights.
    use_continuous_time : bool
        Use ``tgt_time`` (float) instead of ``tgt_frame_idx / max_frames``.
    max_frames : int
        Denominator when ``use_continuous_time=False``.
    output_head_init_std : float
        Std for output head weight initialisation.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        deepest_scale_dim: int = 1024,
        time_fourier_dim: int = 128,
        time_fourier_scale: float = 5.0,
        memory_fourier_dim: int = 128,
        memory_fourier_scale: float = 6.0,
        decoder_ffn_ratio: int = 4,
        decoder_dropout: float = 0.0,
        predict_flow_aux: bool = True,
        use_continuous_time: bool = True,
        max_frames: int = 100,
        output_head_init_std: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.predict_flow_aux = predict_flow_aux
        self.use_continuous_time = use_continuous_time
        self.max_frames = max_frames
        self.output_head_init_std = output_head_init_std

        self.time_encoder = FourierFeatureEncoder(1, time_fourier_dim, time_fourier_scale)
        time_cond_dim = time_fourier_dim

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(deepest_scale_dim),
            nn.Linear(deepest_scale_dim, hidden_dim),
        )
        self.memory_uv_encoder = FourierFeatureEncoder(
            2, memory_fourier_dim, memory_fourier_scale,
        )
        self.memory_pos_proj = nn.Linear(memory_fourier_dim, hidden_dim)

        self.decoder_layers = nn.ModuleList([
            AdaLNCrossAttnDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * decoder_ffn_ratio,
                dropout=decoder_dropout,
                cond_dim=time_cond_dim,
            )
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 3)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "adaln" in name and "time_mlp" in name:
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif name.endswith("output_head"):
                    nn.init.trunc_normal_(m.weight, std=self.output_head_init_std)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                if m.in_proj_weight is not None:
                    nn.init.trunc_normal_(m.in_proj_weight, std=0.02)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)

    # ── utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _make_uv_grid(
        height: int, width: int, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalised UV grid with pixel-center offsets, shape [H*W, 2]."""
        u = torch.linspace(
            0.5 / max(width, 1), 1.0 - 0.5 / max(width, 1), width,
            device=device, dtype=dtype,
        )
        v = torch.linspace(
            0.5 / max(height, 1), 1.0 - 0.5 / max(height, 1), height,
            device=device, dtype=dtype,
        )
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)

    def _encode_target_time(self, queries: dict) -> torch.Tensor:
        """Return [B, Q, time_fourier_dim] time condition."""
        if (
            self.use_continuous_time
            and "tgt_time" in queries
            and queries["tgt_time"] is not None
        ):
            tgt_t = queries["tgt_time"]
        else:
            tgt_t = queries["tgt_frame_idx"].float() / float(max(self.max_frames, 1))
        if tgt_t.ndim == 2:
            tgt_t = tgt_t.unsqueeze(-1)
        return self.time_encoder(tgt_t)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        query_embeds: torch.Tensor,
        queries: dict,
        deepest_pyramid_map: torch.Tensor,
        B: int,
        num_views: int,
        viz_cache: Optional[list] = None,
    ) -> dict:
        """Decode query embeddings against per-target-frame memory.

        Args:
            query_embeds:        [B, Q, hidden_dim] from MotionQueryAggregator.
            queries:             Dict with 'uv', 'tgt_frame_idx', optionally 'tgt_time'.
            deepest_pyramid_map: pyramid[-1], shape [BV, C_d, H_t, W_t].
            B:                   Batch size.
            num_views:           Number of views V.
            viz_cache:           If a list, append attention visualisation entries.

        Returns:
            dict with 'pred_3d' [B, Q, 3] and (optionally) 'pred_flow' [B, Q, 2].
        """
        uv = queries["uv"]                              # [B, Q, 2]
        Q = uv.shape[1]
        work_dtype = query_embeds.dtype

        if Q == 0:
            pred_3d = torch.zeros(B, 0, 3, device=uv.device, dtype=work_dtype)
            out = {"pred_3d": pred_3d}
            if self.predict_flow_aux:
                out["pred_flow"] = torch.zeros(B, 0, 2, device=uv.device, dtype=work_dtype)
            return out

        src_frame = int(queries["src_frame_idx"][0, 0].item())
        time_cond = self._encode_target_time(queries).to(work_dtype)   # [B, Q, cond_dim]
        tgt_frame_idx = queries["tgt_frame_idx"]

        pred_3d = torch.zeros(B, Q, 3, device=uv.device, dtype=work_dtype)
        pred_flow = torch.zeros(B, Q, 2, device=uv.device, dtype=work_dtype)

        deepest_map = deepest_pyramid_map.view(B, num_views, *deepest_pyramid_map.shape[1:])

        # Collect unique target frames across ALL batch items so that frames
        # present in any sample are decoded.  The per-query mask is derived
        # from batch item 0 and assumed to be consistent across the batch
        # (all samples share the same query→frame routing for a given step).
        unique_tgts = tgt_frame_idx.reshape(-1).unique()

        for tf in unique_tgts.tolist():
            mask = tgt_frame_idx[0] == tf
            q_sel = query_embeds[:, mask]               # [B, Q_t, hidden_dim]
            tc_sel = time_cond[:, mask]

            # Build memory from this target frame's deepest-scale tokens
            target_map = deepest_map[:, tf]             # [B, C_d, H_t, W_t]
            _, C_t, H_t, W_t = target_map.shape
            target_tokens = target_map.flatten(2).transpose(1, 2)   # [B, HW, C_d]
            memory = self.memory_proj(target_tokens)    # [B, HW, hidden_dim]

            mem_uv = self._make_uv_grid(H_t, W_t, memory.device, memory.dtype)
            pos = self.memory_pos_proj(self.memory_uv_encoder(mem_uv))
            memory = memory + pos.unsqueeze(0)

            # Decoder layers — extract attention only from the last layer
            h = q_sel
            attn_w = None
            for i, layer in enumerate(self.decoder_layers):
                h, attn_w = layer(
                    h, memory, tc_sel,
                    need_weights=(i == len(self.decoder_layers) - 1),
                )

            pred_3d[:, mask] = self.output_head(self.output_norm(h)).to(pred_3d.dtype)

            soft_uv = None
            if self.predict_flow_aux and attn_w is not None:
                avg_w = attn_w.mean(dim=1)              # [B, Q_t, HW]
                mem_uv_exp = mem_uv.unsqueeze(0).expand(B, -1, -1).to(avg_w.dtype)
                soft_uv = torch.matmul(avg_w, mem_uv_exp)   # [B, Q_t, 2]
                pred_flow[:, mask] = (soft_uv - uv[:, mask]).to(pred_flow.dtype)

            if viz_cache is not None and attn_w is not None:
                viz_cache.append(dict(
                    tgt_frame=tf,
                    src_frame=src_frame,
                    attn_w=attn_w.detach().cpu().float(),
                    mem_uv=mem_uv.detach().cpu().float(),
                    mem_hw=(H_t, W_t),
                    query_uv=uv[:, mask].detach().cpu().float(),
                    pred_flow=(soft_uv - uv[:, mask]).detach().cpu().float()
                        if self.predict_flow_aux and soft_uv is not None else None,
                ))

        if self.training:
            with torch.no_grad():
                pm = pred_3d.detach().norm(dim=-1)
                fm = pred_flow.detach().norm(dim=-1)
                logger.info(
                    "[MotionQueryDecoder] pred=[%.4f,%.4f] flow=[%.4f,%.4f] flow_mean=%.4f",
                    pm.min().item(), pm.max().item(),
                    fm.min().item(), fm.max().item(), fm.mean().item(),
                )

        out = {"pred_3d": pred_3d}
        if self.predict_flow_aux:
            out["pred_flow"] = pred_flow
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Top-level head: chains aggregator → decoder  (slim wrapper, config-driven)
# ─────────────────────────────────────────────────────────────────────────────


class CrossAttnAdaLNFlowHead(nn.Module):
    """v48 — Cross-Attention + AdaLN + 2D Flow Readout sparse motion head.

    Slim wrapper that chains two independently-configurable submodules:

    * ``query_feats_aggregator`` (:class:`MotionQueryAggregator`) —
      samples multi-scale source features and projects to hidden_dim.
    * ``query_decoder`` (:class:`MotionQueryDecoder`) —
      AdaLN cross-attention decoder that attends to target-frame tokens.

    This mirrors the three-module pattern of sv_query
    (``query_banck`` / ``query_feats_aggregator`` / ``query_decoder``),
    making hyper-parameters easy to configure and compare across tasks.

    Parameters
    ----------
    query_feats_aggregator : dict
        Config dict for :class:`MotionQueryAggregator`.
    query_decoder : dict
        Config dict for :class:`MotionQueryDecoder`.
    infer_chunk_size : int
        Max queries per chunk during inference (memory management).
    """

    # ---------------------------------------------------------------------------
    # Legacy weight remapping (pre-refactor → post-refactor key migration)
    # ---------------------------------------------------------------------------
    # Before the query_feats_aggregator / query_decoder split, all sub-modules
    # lived directly under `sparse_motion_head.*`.  This table maps each old
    # flat prefix to its new nested location so that pre-refactor checkpoints
    # load without manual key renaming.
    _LEGACY_KEY_MAP: dict[str, str] = {
        # query_feats_aggregator group
        "multiscale_sampler.": "query_feats_aggregator.multiscale_sampler.",
        "uv_encoder.":         "query_feats_aggregator.uv_encoder.",
        "query_proj.":         "query_feats_aggregator.query_proj.",
        # query_decoder group
        "time_encoder.":       "query_decoder.time_encoder.",
        "memory_proj.":        "query_decoder.memory_proj.",
        "memory_uv_encoder.":  "query_decoder.memory_uv_encoder.",
        "memory_pos_proj.":    "query_decoder.memory_pos_proj.",
        "decoder_layers.":     "query_decoder.decoder_layers.",
        "output_norm.":        "query_decoder.output_norm.",
        "output_head.":        "query_decoder.output_head.",
    }

    @classmethod
    def _remap_legacy_state_dict(cls, state_dict: dict, prefix: str) -> dict:
        """Return a new state-dict with legacy flat keys rewritten to the
        current nested layout.  Keys outside *prefix* are passed through
        unchanged.
        """
        remapped: dict = {}
        for key, val in state_dict.items():
            if not key.startswith(prefix):
                remapped[key] = val
                continue
            suffix = key[len(prefix):]
            new_key = key   # default: keep as-is
            for old_pfx, new_pfx in cls._LEGACY_KEY_MAP.items():
                if suffix.startswith(old_pfx):
                    new_key = prefix + new_pfx + suffix[len(old_pfx):]
                    break
            remapped[new_key] = val
        return remapped

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Intercept legacy flat-layout checkpoints and remap keys before
        delegating to the standard PyTorch loader.

        A ``DeprecationWarning`` is emitted whenever a pre-refactor checkpoint
        is detected so that callers know to re-save the weights.
        """
        # Detect at least one legacy key under this module's prefix.
        has_legacy = any(
            key.startswith(prefix + old_pfx)
            for key in state_dict
            for old_pfx in self._LEGACY_KEY_MAP
        )
        if has_legacy:
            msg = (
                f"[CrossAttnAdaLNFlowHead] Detected a LEGACY checkpoint layout "
                f"(flat keys directly under '{prefix}' without the "
                f"'query_feats_aggregator' / 'query_decoder' nesting introduced "
                f"in the v48 refactor). Keys are being remapped automatically. "
                f"Please re-save the checkpoint with the current model architecture "
                f"to suppress this warning."
            )
            logger.warning(msg)
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            state_dict = self._remap_legacy_state_dict(state_dict, prefix)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def __init__(
        self,
        query_feats_aggregator: dict,
        query_decoder: dict,
        infer_chunk_size: int = 2048,
        **kwargs,
    ):
        super().__init__()
        self.query_feats_aggregator: MotionQueryAggregator = instantiate_from_config(
            query_feats_aggregator
        )
        self.query_decoder: MotionQueryDecoder = instantiate_from_config(query_decoder)

        self.infer_chunk_size = infer_chunk_size
        self.use_feature_sampling = False   # pipeline compatibility flag
        self._viz_cache = None              # set to list via viz_mode() to enable

        logger.info(
            "[CrossAttnAdaLNFlowHead] hidden=%d layers=%d chunk=%d flow=%s",
            self.query_feats_aggregator.hidden_dim,
            len(self.query_decoder.decoder_layers),
            infer_chunk_size,
            self.query_decoder.predict_flow_aux,
        )

    # ── utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _infer_num_views(features, batch_size: int) -> int:
        feat = features[-1] if isinstance(features, (list, tuple)) else features
        if feat.ndim == 4:
            feat = feat.reshape(-1, *feat.shape[2:])
        return max(feat.shape[0] // max(batch_size, 1), 1)

    # ── core forward ─────────────────────────────────────────────────────

    def _forward_with_memory(
        self,
        pyramid: list[torch.Tensor],
        num_views: int,
        queries: dict,
    ) -> dict:
        uv = queries["uv"]                  # [B, Q, 2]
        B = uv.shape[0]
        src_frame = int(queries["src_frame_idx"][0, 0].item())

        # Stage 1: aggregate source features at query UV
        query_embeds = self.query_feats_aggregator(
            pyramid=pyramid,
            uv=uv,
            src_frame_idx=src_frame,
            B=B,
            num_views=num_views,
        )

        # Stage 2: decode against per-target-frame memory
        return self.query_decoder(
            query_embeds=query_embeds,
            queries=queries,
            deepest_pyramid_map=pyramid[-1],
            B=B,
            num_views=num_views,
            viz_cache=self._viz_cache,
        )

    def _forward_chunked_with_memory(
        self,
        pyramid: list[torch.Tensor],
        num_views: int,
        queries: dict,
    ) -> dict:
        B, total_Q = queries["uv"].shape[:2]
        outputs = []
        for start in range(0, total_Q, self.infer_chunk_size):
            end = min(start + self.infer_chunk_size, total_Q)
            chunk = {
                k: (
                    v[:, start:end]
                    if torch.is_tensor(v) and v.ndim >= 2
                    and v.shape[0] == B and v.shape[1] == total_Q
                    else v
                )
                for k, v in queries.items()
            }
            outputs.append(self._forward_with_memory(pyramid, num_views, chunk))

        if not outputs:
            return {}
        return {k: torch.cat([o[k] for o in outputs], dim=1) for k in outputs[0]}

    # ── public interface (called by pipeline) ─────────────────────────────

    def _build_memory(
        self, features, patch_start_idx: int, B: int, img_size=None, **kwargs,
    ) -> dict:
        """Build cached pyramid for per-target-frame inference (pipeline API)."""
        return {
            "pyramid": self.query_feats_aggregator.build_pyramid(
                features, patch_start_idx, img_size,
            ),
            "num_views": self._infer_num_views(features, B),
        }

    def decode_with_memory(self, memory_bundle: dict, rgb_images, queries) -> dict:
        """Decode queries against a pre-built pyramid bundle (pipeline API)."""
        if not isinstance(memory_bundle, dict):
            raise TypeError(
                "CrossAttnAdaLNFlowHead.decode_with_memory expects a dict bundle."
            )
        pyramid = memory_bundle["pyramid"]
        num_views = memory_bundle.get("num_views") or rgb_images.shape[1]
        if not self.training and queries["uv"].shape[1] > self.infer_chunk_size:
            return self._forward_chunked_with_memory(pyramid, num_views, queries)
        return self._forward_with_memory(pyramid, num_views, queries)

    def forward(
        self,
        features,
        rgb_images: torch.Tensor,
        queries: dict,
        img_size: Optional[tuple[int, int]],
        patch_start_idx: int = 0,
        meta_data=None,
    ) -> dict:
        B = queries["uv"].shape[0]
        pyramid = self.query_feats_aggregator.build_pyramid(
            features, patch_start_idx, img_size,
        )
        num_views = self._infer_num_views(features, B)
        if not self.training and queries["uv"].shape[1] > self.infer_chunk_size:
            return self._forward_chunked_with_memory(pyramid, num_views, queries)
        return self._forward_with_memory(pyramid, num_views, queries)

    # ── Attention Visualization ───────────────────────────────────────────

    @contextlib.contextmanager
    def viz_mode(self):
        """Context manager that enables attention caching for visualization."""
        self._viz_cache = []
        try:
            yield
        finally:
            self._viz_cache = None

    @staticmethod
    def _img_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
        img = img_tensor.detach().cpu().float()
        lo, hi = img.min(), img.max()
        if hi - lo > 0:
            img = (img - lo) / (hi - lo)
        return (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)

    @staticmethod
    def _farthest_point_sample(uv: np.ndarray, n: int) -> list[int]:
        N = len(uv)
        if N <= n:
            return list(range(N))
        sel = [np.random.randint(N)]
        dists = np.full(N, np.inf)
        for _ in range(n - 1):
            last = uv[sel[-1]]
            d = np.linalg.norm(uv - last, axis=-1)
            dists = np.minimum(dists, d)
            sel.append(int(np.argmax(dists)))
        return sel

    @torch.no_grad()
    def render_attention_viz(
        self,
        save_dir: str,
        rgb_images: Optional[torch.Tensor] = None,
        gt_flow: Optional[torch.Tensor] = None,
        n_queries: int = 8,
        batch_idx: int = 0,
        dpi: int = 150,
    ):
        """Render attention heatmaps and predicted flow from cached data.

        Call this **inside** (or just after) :meth:`viz_mode` context.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if not self._viz_cache:
            logger.warning("[render_attention_viz] No cached data. "
                           "Wrap forward() with head.viz_mode().")
            return
        os.makedirs(save_dir, exist_ok=True)

        cmap_attn = plt.cm.inferno

        for entry in self._viz_cache:
            tf = entry["tgt_frame"]
            sf = entry["src_frame"]
            attn   = entry["attn_w"][batch_idx]
            mem_uv = entry["mem_uv"]
            H_t, W_t = entry["mem_hw"]
            q_uv   = entry["query_uv"][batch_idx]
            p_flow = entry["pred_flow"][batch_idx] if entry["pred_flow"] is not None else None

            avg_attn = attn.mean(dim=0)
            sel = self._farthest_point_sample(q_uv.numpy(), n_queries)
            n_sel = len(sel)

            src_img = self._img_to_numpy(rgb_images[batch_idx, sf]) if rgb_images is not None else None
            tgt_img = self._img_to_numpy(rgb_images[batch_idx, tf]) if rgb_images is not None else None
            img_h = src_img.shape[0] if src_img is not None else 518
            img_w = src_img.shape[1] if src_img is not None else 518

            colors = plt.cm.tab10(np.linspace(0, 1, max(n_sel, 1)))

            # Figure 1: flow overview
            fig1, (ax_s, ax_t) = plt.subplots(1, 2, figsize=(16, 7))
            if src_img is not None:
                ax_s.imshow(src_img)
            if tgt_img is not None:
                ax_t.imshow(tgt_img)
            ax_s.set_title(f"Source (frame {sf})", fontsize=11)
            ax_t.set_title(f"Target (frame {tf})", fontsize=11)
            for k, idx in enumerate(sel):
                su, sv = q_uv[idx].numpy() * [img_w, img_h]
                ax_s.plot(su, sv, "o", color=colors[k], markersize=8,
                          markeredgecolor="white", markeredgewidth=1.0, zorder=5)
                if p_flow is not None:
                    du, dv = p_flow[idx].numpy() * [img_w, img_h]
                    tu, tv = su + du, sv + dv
                    ax_s.annotate("", xy=(tu, tv), xytext=(su, sv),
                                  arrowprops=dict(arrowstyle="->", color=colors[k], lw=1.5))
                    ax_t.plot(tu, tv, "*", color=colors[k], markersize=12,
                              markeredgecolor="white", markeredgewidth=0.8, zorder=5)
                    ax_t.text(tu + 4, tv - 4, f"Q{k}", fontsize=7,
                              color=colors[k], fontweight="bold")
                ax_s.text(su + 4, sv - 4, f"Q{k}", fontsize=7,
                          color=colors[k], fontweight="bold")
            for ax in (ax_s, ax_t):
                ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0); ax.set_axis_off()
            fig1.suptitle(f"v48 — Predicted Flow  (src={sf} → tgt={tf})",
                          fontsize=13, fontweight="bold")
            fig1.tight_layout()
            fig1.savefig(os.path.join(save_dir, f"flow_overview_tgt{tf:03d}.png"),
                         dpi=dpi, bbox_inches="tight")
            plt.close(fig1)

            # Figure 2: per-query attention heatmaps
            cols = min(n_sel, 4)
            rows = (n_sel + cols - 1) // cols
            fig2, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
            if n_sel == 1:
                axes = np.array([axes])
            axes = np.atleast_2d(axes)
            for k, idx in enumerate(sel):
                r, c = divmod(k, cols)
                ax = axes[r, c]
                attn_map = avg_attn[idx].numpy().reshape(H_t, W_t)
                if tgt_img is not None:
                    ax.imshow(tgt_img, alpha=0.45)
                im = ax.imshow(attn_map, extent=[0, img_w, img_h, 0],
                               cmap=cmap_attn, alpha=0.7, interpolation="bilinear",
                               vmin=0, vmax=attn_map.max() + 1e-8)
                su, sv = q_uv[idx].numpy() * [img_w, img_h]
                ax.plot(su, sv, "o", color="lime", markersize=9,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=6)
                if p_flow is not None:
                    du, dv = p_flow[idx].numpy() * [img_w, img_h]
                    ax.plot(su + du, sv + dv, "*", color="cyan", markersize=13,
                            markeredgecolor="white", markeredgewidth=0.8, zorder=6)
                peak_idx = int(attn_map.argmax())
                peak_px = mem_uv[peak_idx].numpy() * [img_w, img_h]
                ax.plot(peak_px[0], peak_px[1], "D", color="red", markersize=7,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=6)
                ax.set_title(
                    f"Q{k}  src=({su:.0f},{sv:.0f})  "
                    f"peak=({peak_px[0]:.0f},{peak_px[1]:.0f})  "
                    f"max_w={attn_map.max():.3f}", fontsize=8,
                )
                ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0); ax.set_axis_off()
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.05)
                plt.colorbar(im, cax=cax)
            for k in range(n_sel, rows * cols):
                r, c = divmod(k, cols)
                axes[r, c].set_visible(False)
            if n_sel > 0:
                axes[0, 0].legend(fontsize=7, loc="lower left", framealpha=0.7)
            fig2.suptitle(
                f"v48 — Attention Heatmaps  (tgt frame {tf})\n"
                f"○ src UV   ★ pred tgt UV   ◆ attn peak",
                fontsize=11, fontweight="bold",
            )
            fig2.tight_layout()
            fig2.savefig(os.path.join(save_dir, f"attn_heatmap_tgt{tf:03d}.png"),
                         dpi=dpi, bbox_inches="tight")
            plt.close(fig2)

            # Figure 3: per-head attention for first query
            first_idx = sel[0]
            nhead = attn.shape[0]
            fig3_cols = min(nhead, 4)
            fig3_rows = (nhead + fig3_cols - 1) // fig3_cols
            fig3, axes3 = plt.subplots(fig3_rows, fig3_cols,
                                       figsize=(4.5 * fig3_cols, 4 * fig3_rows))
            axes3 = np.atleast_2d(np.asarray(axes3 if nhead > 1 else [axes3]))
            for h_idx in range(nhead):
                r, c = divmod(h_idx, fig3_cols)
                ax = axes3[r, c]
                head_map = attn[h_idx, first_idx].numpy().reshape(H_t, W_t)
                if tgt_img is not None:
                    ax.imshow(tgt_img, alpha=0.4)
                ax.imshow(head_map, extent=[0, img_w, img_h, 0],
                          cmap=cmap_attn, alpha=0.7, interpolation="bilinear",
                          vmin=0, vmax=head_map.max() + 1e-8)
                ax.set_title(f"Head {h_idx}  max={head_map.max():.3f}", fontsize=9)
                ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0); ax.set_axis_off()
            for h_idx in range(nhead, fig3_rows * fig3_cols):
                r, c = divmod(h_idx, fig3_cols)
                axes3[r, c].set_visible(False)
            su, sv = q_uv[first_idx].numpy() * [img_w, img_h]
            fig3.suptitle(
                f"v48 — Per-Head Attention for Q0  "
                f"src=({su:.0f},{sv:.0f})  tgt frame {tf}",
                fontsize=11, fontweight="bold",
            )
            fig3.tight_layout()
            fig3.savefig(os.path.join(save_dir, f"attn_perhead_tgt{tf:03d}.png"),
                         dpi=dpi, bbox_inches="tight")
            plt.close(fig3)

        logger.info("[render_attention_viz] Saved %d target-frame visualizations to %s",
                    len(self._viz_cache), save_dir)
