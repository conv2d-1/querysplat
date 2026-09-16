"""Sparse Motion Head — D4RT-style Query Decoder (arXiv:2512.08924)

Three modes: feature_sampling | single-pass cross-attn | iterative cross-attn
Query format: q = (u, v, t_src, t_tgt, t_cam)
"""
from __future__ import annotations

import logging
import math
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MemoryBundle(NamedTuple):
    """Cross-attention memory: projected features + 2D UV coordinates per token."""
    tokens: torch.Tensor  # [B, V*S, hidden_dim]
    uv: torch.Tensor      # [V*S, 2]  patch-centre UV in [0,1]; same for all batch items
    time: Optional[torch.Tensor] = None  # [V*S] normalised time per token
    view_idx: Optional[torch.Tensor] = None  # [V*S] int view index per token
    feat_hw: Optional[tuple] = None  # (H_feat, W_feat) spatial dims per view


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class FourierFeatureEncoder(nn.Module):
    """Random Fourier feature encoding for low-dimensional inputs."""

    def __init__(self, input_dim: int, embed_dim: int, scale: float = 10.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        self.register_buffer("B", torch.randn(input_dim, embed_dim // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = (2.0 * math.pi * x) @ self.B.to(device=x.device, dtype=x.dtype)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class TokenAdaLN(nn.Module):
    """Per-token adaptive LayerNorm conditioned on an external embedding."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * channels),
        )
        nn.init.zeros_(self.time_mlp[1].weight)
        nn.init.zeros_(self.time_mlp[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        gamma, beta = self.time_mlp(cond).chunk(2, dim=-1)
        return x_norm * (1.0 + gamma) + beta


class LearnedRelativePosBias(nn.Module):
    """Per-head attention bias from spatial-temporal position.

    Uses factored dot-product: separately Fourier-encode + project query and
    memory positions into per-head representations, then compute bias via dot
    product.  By the shift theorem, the dot product of Fourier features of two
    positions naturally captures their *relative* position.
    """

    def __init__(self, num_heads: int, proj_dim: int = 16,
                 fourier_dim: int = 64, fourier_scale: float = 6.0,
                 init_log_scale: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.proj_dim = proj_dim
        self.fourier = FourierFeatureEncoder(3, fourier_dim, scale=fourier_scale)
        self.q_proj = nn.Linear(fourier_dim, num_heads * proj_dim)
        self.m_proj = nn.Linear(fourier_dim, num_heads * proj_dim)
        self.log_scale = nn.Parameter(torch.full((num_heads,), init_log_scale))
        nn.init.trunc_normal_(self.q_proj.weight, std=0.02)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.trunc_normal_(self.m_proj.weight, std=0.02)
        nn.init.zeros_(self.m_proj.bias)

    def forward(self, query_uv: torch.Tensor, query_time: torch.Tensor,
                memory_uv: torch.Tensor, memory_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_uv:    [B, Q, 2]  normalised UV
            query_time:  [B, Q]     normalised time
            memory_uv:   [M, 2]     patch-centre UVs
            memory_time: [M]        normalised time per memory token
        Returns:
            [B, num_heads, Q, M] additive attention logit bias.
        """
        B, Q, _ = query_uv.shape
        H, k = self.num_heads, self.proj_dim

        q_t = query_time.unsqueeze(-1) if query_time.ndim == 2 else query_time
        q_pos = torch.cat([query_uv, q_t], dim=-1)                     # [B, Q, 3]
        m_t = memory_time.unsqueeze(-1) if memory_time.ndim == 1 else memory_time
        m_pos = torch.cat([memory_uv, m_t], dim=-1)                    # [M, 3]

        q_ff = self.fourier(q_pos)                                      # [B, Q, fourier_dim]
        m_ff = self.fourier(m_pos)                                      # [M, fourier_dim]

        q_h = self.q_proj(q_ff).view(B, Q, H, k).permute(0, 2, 1, 3)  # [B, H, Q, k]
        m_h = self.m_proj(m_ff).view(-1, H, k).permute(1, 2, 0).unsqueeze(0)  # [1, H, k, M]

        scale = self.log_scale.exp().view(1, H, 1, 1)
        return torch.matmul(q_h, m_h) * scale                          # [B, H, Q, M]


class CrossAttentionDecoderLayer(nn.Module):
    """(Optional self-attn) + cross-attn + FFN.

    When use_self_attention=True, queries communicate before attending to
    memory.  This mimics the local spatial processing that dense DPT heads
    achieve through convolution, enabling nearby queries on the same rigid
    body to share motion information.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, use_self_attention: bool = False,
                 use_target_time_adaln: bool = False, time_cond_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = nhead
        self.use_self_attention = use_self_attention
        self.use_target_time_adaln = use_target_time_adaln

        if use_self_attention:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True,
            )
            self.norm_sa = nn.LayerNorm(d_model)
            self.dropout_sa = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        if self.use_target_time_adaln:
            if time_cond_dim is None:
                raise ValueError("time_cond_dim must be set when use_target_time_adaln=True")
            self.cross_adaln = TokenAdaLN(d_model, time_cond_dim)
        else:
            self.cross_adaln = None

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                locality_bias: Optional[torch.Tensor] = None,
                time_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt:           [B, Q, d_model]
            memory:        [B, S, d_model]
            locality_bias: [B, Q, S] broadcast to all heads, or
                           [B, H, Q, S] per-head bias.  None → disabled.
            time_cond:     [B, Q, d_cond] target-time conditioning for AdaLN.
        """
        if self.use_self_attention:
            tgt_n = self.norm_sa(tgt)
            sa_out, _ = self.self_attn(
                query=tgt_n, key=tgt_n, value=tgt_n, need_weights=False,
            )
            if not torch.isfinite(sa_out).all():
                finite = torch.isfinite(sa_out)
                logger.warning("[SelfAttn] %d NaN/Inf — zeroing", (~finite).sum().item())
                sa_out = torch.where(finite, sa_out, torch.zeros_like(sa_out.detach()))
            tgt = tgt + self.dropout_sa(sa_out)

        attn_mask = None
        if locality_bias is not None:
            B_loc = tgt.shape[0]
            if locality_bias.ndim == 3:
                attn_mask = (
                    locality_bias
                    .unsqueeze(1)
                    .expand(B_loc, self.num_heads, -1, -1)
                    .contiguous()
                    .reshape(B_loc * self.num_heads,
                             locality_bias.shape[1], locality_bias.shape[2])
                )
            else:
                attn_mask = locality_bias.reshape(
                    B_loc * self.num_heads,
                    locality_bias.shape[2], locality_bias.shape[3],
                )

        query = self.norm1(tgt)
        if self.cross_adaln is not None and time_cond is not None:
            query = self.cross_adaln(tgt, time_cond)

        tgt2, _ = self.cross_attn(
            query=query, key=self.norm_kv(memory), value=self.norm_kv(memory),
            attn_mask=attn_mask, need_weights=False, average_attn_weights=False,
        )

        # Guard: fp16 Q·Kᵀ/√d overflow → NaN propagation
        if not torch.isfinite(tgt2).all():
            mask = torch.isfinite(tgt2)
            logger.warning("[CrossAttn] %d NaN/Inf in attn output — zeroing", (~mask).sum().item())
            tgt2 = torch.where(mask, tgt2, torch.zeros_like(tgt2.detach()))

        tgt = tgt + self.dropout1(tgt2)
        return tgt + self.ffn(self.norm2(tgt))


class CrossAttentionDecoder(nn.Module):
    """Stack of CrossAttentionDecoderLayer."""

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 use_self_attention: bool = False,
                 use_target_time_adaln: bool = False,
                 time_cond_dim: Optional[int] = None):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                       use_self_attention=use_self_attention,
                                       use_target_time_adaln=use_target_time_adaln,
                                       time_cond_dim=time_cond_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                locality_bias: Optional[torch.Tensor] = None,
                time_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory, locality_bias=locality_bias, time_cond=time_cond)
        return self.final_norm(tgt)


# ─────────────────────────────────────────────────────────────────────────────
# Query encoder
# ─────────────────────────────────────────────────────────────────────────────

class D4RTQueryEncoder(nn.Module):
    """Encode query (uv, time, rgb_patch) into a per-query latent vector.

    ``time_mode="none"`` disables query-side time encoding entirely. This is
    useful when target time is injected deeper in the decoder (e.g. via AdaLN)
    and the source frame is fixed, making the original time branches redundant.
    """

    def __init__(self, query_dim: int = 256, max_frames: int = 100,
                 rgb_patch_size: int = 9, fourier_scale: float = 10.0,
                 time_fourier_scale: float = 5.0,
                 use_continuous_time: bool = True, fusion_mode: str = "concat",
                 use_uv_skip: bool = False, time_mode: str = "delta_only"):
        super().__init__()
        if fusion_mode not in ("add", "concat"):
            raise ValueError(f"fusion_mode must be 'add' or 'concat', got {fusion_mode!r}")
        if time_mode not in ("none", "delta_only", "delta_tgt", "delta_tgt_cam"):
            raise ValueError(
                "time_mode must be one of ('none', 'delta_only', 'delta_tgt', 'delta_tgt_cam'), "
                f"got {time_mode!r}"
            )
        self.query_dim = query_dim
        self.max_frames = max_frames
        self.use_continuous_time = use_continuous_time
        self.fusion_mode = fusion_mode
        self.use_uv_skip = use_uv_skip
        self.time_mode = time_mode
        self.num_time_feats = {
            "none": 0,
            "delta_only": 1,
            "delta_tgt": 2,
            "delta_tgt_cam": 3,
        }[time_mode]

        self.pos_encoder = FourierFeatureEncoder(2, query_dim, fourier_scale)

        if use_continuous_time:
            self.src_time_encoder = FourierFeatureEncoder(1, query_dim, scale=time_fourier_scale)
            self.tgt_time_encoder = FourierFeatureEncoder(1, query_dim, scale=time_fourier_scale)
            self.cam_time_encoder = FourierFeatureEncoder(1, query_dim, scale=time_fourier_scale)
        else:
            self.src_time_embed = nn.Embedding(max_frames, query_dim)
            self.tgt_time_embed = nn.Embedding(max_frames, query_dim)
            self.cam_time_embed = nn.Embedding(max_frames, query_dim)

        patch_dim = rgb_patch_size * rgb_patch_size * 3
        self.rgb_patch_encoder = nn.Sequential(
            nn.Linear(patch_dim, query_dim), nn.LayerNorm(query_dim),
            nn.GELU(), nn.Linear(query_dim, query_dim),
        )
        if fusion_mode == "concat":
            self.query_fuse = nn.Sequential(
                nn.Linear(query_dim * (2 + self.num_time_feats), query_dim), nn.LayerNorm(query_dim),
            )
        if use_uv_skip:
            self.uv_skip_gate = nn.Parameter(torch.tensor(0.5))

        self.query_token = nn.Parameter(torch.zeros(1, 1, query_dim))
        nn.init.trunc_normal_(self.query_token, std=0.02)

    def _fuse(self, feats: list) -> torch.Tensor:
        if self.fusion_mode == "add":
            return sum(feats)
        return self.query_fuse(torch.cat(feats, dim=-1))

    def forward(self, uv_norm, src_idx, tgt_idx, cam_idx, rgb_patch,
                src_time=None, tgt_time=None, cam_time=None):
        pos_feat = self.pos_encoder(uv_norm)
        rgb_feat = self.rgb_patch_encoder(rgb_patch)

        if self.use_continuous_time:
            t_max = float(self.max_frames)
            feats = [pos_feat]
            delta_std = tgt_std = cam_std = 0.0
            if self.time_mode != "none":
                src_t = (src_time.unsqueeze(-1) if src_time.ndim == 2 else src_time
                         ) if src_time is not None else (src_idx.float() / t_max).unsqueeze(-1)
                tgt_t = (tgt_time.unsqueeze(-1) if tgt_time.ndim == 2 else tgt_time
                         ) if tgt_time is not None else (tgt_idx.float() / t_max).unsqueeze(-1)
                cam_t = (cam_time.unsqueeze(-1) if cam_time.ndim == 2 else cam_time
                         ) if cam_time is not None else src_t
                delta_t = tgt_t - src_t

                if self.time_mode in ("delta_only", "delta_tgt", "delta_tgt_cam"):
                    delta_feat = self.src_time_encoder(delta_t)
                    feats.append(delta_feat)
                    delta_std = delta_feat.std(1).mean().item()
                if self.time_mode in ("delta_tgt", "delta_tgt_cam"):
                    tgt_feat = self.tgt_time_encoder(tgt_t)
                    feats.append(tgt_feat)
                    tgt_std = tgt_feat.std(1).mean().item()
                if self.time_mode == "delta_tgt_cam":
                    cam_feat = self.cam_time_encoder(cam_t)
                    feats.append(cam_feat)
                    cam_std = cam_feat.std(1).mean().item()
            feats.append(rgb_feat)

            if self.training:
                logger.info(
                    "[DEBUG QueryEncoder] time_mode=%s | pos_std=%.4f | delta_t_std=%.4f | "
                    "tgt_t_std=%.4f | cam_t_std=%.4f | rgb_std=%.4f",
                    self.time_mode,
                    pos_feat.std(1).mean().item(), delta_std, tgt_std, cam_std,
                    rgb_feat.std(1).mean().item(),
                )
        else:
            feats = [pos_feat]
            if self.time_mode in ("delta_only", "delta_tgt", "delta_tgt_cam"):
                delta_idx = (tgt_idx - src_idx).clamp(min=0, max=self.max_frames - 1)
                feats.append(self.src_time_embed(delta_idx))
            if self.time_mode in ("delta_tgt", "delta_tgt_cam"):
                feats.append(self.tgt_time_embed(tgt_idx))
            if self.time_mode == "delta_tgt_cam":
                feats.append(self.cam_time_embed(cam_idx))
            feats.append(rgb_feat)
        fused = self._fuse(feats)
        if self.use_uv_skip:
            fused = fused + self.uv_skip_gate * pos_feat
        fused = fused + self.query_token.expand(fused.shape[0], fused.shape[1], -1)
        return fused


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale Local Implicit Sampler (inspired by InfiniDepth, arXiv:2601.03252)
# ─────────────────────────────────────────────────────────────────────────────

class GatedFusionBlock(nn.Module):
    """Residual gated fusion: FFN(f_deep + gate ⊙ Linear(h_shallow)).

    Implements InfiniDepth Eq. 3: hierarchically fuses features from
    high spatial resolution (shallow ViT layers) to low resolution (deep layers).
    """

    def __init__(self, shallow_dim: int, deep_dim: int):
        super().__init__()
        self.adapt = nn.Linear(shallow_dim, deep_dim)
        self.gate = nn.Parameter(torch.ones(deep_dim) * 0.5)
        self.ffn = nn.Sequential(
            nn.Linear(deep_dim, deep_dim * 2),
            nn.GELU(),
            nn.Linear(deep_dim * 2, deep_dim),
        )
        self.norm = nn.LayerNorm(deep_dim)

    def forward(self, h_shallow: torch.Tensor, f_deep: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate)
        return self.ffn(self.norm(f_deep + g * self.adapt(h_shallow)))


class MultiScaleImplicitSampler(nn.Module):
    """InfiniDepth-style multi-scale local implicit feature sampler.

    Given multi-hook ViT features and continuous query UV coordinates:
      1. Reassemble: project each hook to per-scale hidden dim, reshape to 2D
      2. (Optional) learned upsample of shallow layers for higher spatial resolution
      3. Bilinear-sample at query UV from each scale
      4. Hierarchical gated fusion from shallow (high-res) → deep (low-res)

    Returns a fused feature vector per query point.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        scale_dims: list[int] = (256, 512, 1024),
        upsample_factors: list[int] = (4, 2, 1),
        use_learned_upsample: bool = True,
    ):
        super().__init__()
        n_scales = len(scale_dims)
        assert len(upsample_factors) == n_scales

        self.n_scales = n_scales
        self.scale_dims = list(scale_dims)
        self.upsample_factors = list(upsample_factors)
        self.output_dim = scale_dims[-1]

        self.projects = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, d))
            for d in scale_dims
        ])

        self.upsamplers = nn.ModuleList()
        for k in range(n_scales):
            f = upsample_factors[k]
            d = scale_dims[k]
            if f > 1 and use_learned_upsample:
                self.upsamplers.append(
                    nn.ConvTranspose2d(d, d, kernel_size=f, stride=f, padding=0)
                )
            else:
                self.upsamplers.append(nn.Identity())

        self.fusions = nn.ModuleList([
            GatedFusionBlock(scale_dims[k], scale_dims[k + 1])
            for k in range(n_scales - 1)
        ])

        self._init_weights()
        logger.info(
            "[MultiScaleImplicitSampler] scales=%d, dims=%s, upsample=%s, "
            "learned_upsample=%s, output_dim=%d",
            n_scales, scale_dims, upsample_factors,
            use_learned_upsample, self.output_dim,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_pyramid(
        self,
        multi_features: list[torch.Tensor],
        patch_start_idx: int,
        H_feat: int,
        W_feat: int,
    ) -> list[torch.Tensor]:
        """Project + reshape + upsample → list of [BV, C_k, H_k, W_k]."""
        pyramid = []
        for k in range(self.n_scales):
            idx = min(k, len(multi_features) - 1)
            feat = multi_features[idx]
            if feat.ndim == 4:
                feat = feat.reshape(-1, *feat.shape[2:])
            feat = feat[:, patch_start_idx:]                          # [BV, S, C]
            feat = self.projects[k](feat)                             # [BV, S, d_k]
            BV, S, d_k = feat.shape
            feat_2d = feat.reshape(BV, H_feat, W_feat, d_k).permute(0, 3, 1, 2)
            feat_2d = self.upsamplers[k](feat_2d)                    # [BV, d_k, H_k, W_k]
            pyramid.append(feat_2d)
        return pyramid

    def sample_and_fuse(
        self,
        pyramid: list[torch.Tensor],
        uv: torch.Tensor,
        frame_idx: int,
        B: int,
        V: int,
    ) -> torch.Tensor:
        """Bilinear-sample from each scale and hierarchically fuse.

        Args:
            pyramid: list of [BV, C_k, H_k, W_k] feature maps.
            uv:      [B, Q, 2] normalised query coordinates.
            frame_idx: which view to sample from.
            B, V:    batch size and number of views.
        Returns:
            [B, Q, output_dim] fused features.
        """
        grid = (uv * 2 - 1).unsqueeze(2)                             # [B, Q, 1, 2]

        sampled = []
        for k in range(self.n_scales):
            feat_map = pyramid[k].view(B, V, *pyramid[k].shape[1:])  # [B, V, C_k, H_k, W_k]
            feat_frame = feat_map[:, frame_idx]                       # [B, C_k, H_k, W_k]
            s = F.grid_sample(feat_frame, grid.to(feat_frame.dtype),
                              mode='bilinear', padding_mode='border', align_corners=True)
            sampled.append(s.squeeze(-1).permute(0, 2, 1))           # [B, Q, C_k]

        h = sampled[0]
        for k in range(self.n_scales - 1):
            h = self.fusions[k](h, sampled[k + 1])
        return h                                                      # [B, Q, output_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Residual MLP Block (inspired by PointMLP / RAFT update block)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualMLPBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → GELU → Linear → Add."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


# Sparse Motion Head
# ─────────────────────────────────────────────────────────────────────────────

class SparseMotionHead(nn.Module):
    """D4RT-style query-based sparse 3D motion prediction.

    Modes:
      use_feature_sampling=True  — bilinear-sample encoder features at UV coords.
      num_iters == 1             — single-pass cross-attention.
      num_iters  > 1             — RAFT-style iterative refinement; returns
                                   [B, num_iters, Q, 3] for auxiliary losses.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        query_dim: int = 256,
        hidden_dim: int = 512,
        num_decoder_layers: int = 8,
        num_heads: int = 8,
        max_frames: int = 100,
        num_cameras: int = 1,
        rgb_patch_size: int = 9,
        num_feature_scales: int = 1,
        infer_chunk_size: int = 2048,
        use_continuous_time: bool = True,
        decoder_dropout: float = 0.1,
        query_fusion_mode: str = "concat",
        use_memory_pos_encoding: bool = True,
        memory_fourier_scale: float = 6.0,
        query_time_fourier_scale: float = 5.0,
        query_time_mode: str = "delta_only",
        use_target_time_adaln: bool = False,
        target_time_fourier_scale: float = 5.0,
        predict_confidence: bool = False,
        num_iters: int = 1,
        output_head_init_std: float = 0.01,
        iter_feature_indices: Optional[list[int]] = None,
        use_feature_sampling: bool = False,
        sampling_multi_frame: bool = False,
        sampling_fusion: str = "concat",
        sampling_hidden_layers: int = 2,
        sampling_use_residual: bool = False,
        sampling_residual_expansion: int = 4,
        sampling_dropout: float = 0.0,
        sampling_direct_uv: bool = False,
        sampling_direct_uv_dim: int = 512,
        sampling_direct_uv_scale: float = 10.0,
        sampling_corr_radius: int = 4,
        sampling_corr_num_levels: int = 1,
        use_uv_bias: bool = True,
        uv_bias_init_sigma: float = 0.15,
        use_relative_pos_bias: bool = False,
        relative_pos_bias_dim: int = 16,
        use_memory_pos_gate: bool = False,
        use_query_uv_skip: bool = False,
        use_temporal_mask: bool = False,
        temporal_mask_mode: str = "src_tgt",
        use_local_window: bool = False,
        local_window_radius: int = 8,
        local_window_radii: Optional[list[int]] = None,
        use_source_feature: bool = False,
        share_iter_weights: bool = False,
        detach_iter_disp: bool = True,
        use_self_attention: bool = False,
        use_dynamic_rel_pos_bias: bool = False,
        rel_pos_bias_init_log_scale: float = 0.0,
        sampling_multiscale: bool = False,
        sampling_multiscale_dims: Optional[list[int]] = None,
        sampling_multiscale_upsample: Optional[list[int]] = None,
        sampling_multiscale_learned_upsample: bool = True,
        sampling_predict_flow: bool = False,
        sampling_flow_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        if rgb_patch_size % 2 != 1:
            raise ValueError(f"rgb_patch_size must be odd, got {rgb_patch_size}")

        self.hidden_dim            = hidden_dim
        self.rgb_patch_size        = rgb_patch_size
        self.num_feature_scales    = num_feature_scales
        self.infer_chunk_size      = infer_chunk_size
        self.max_frames            = max_frames
        self.use_memory_pos_encoding = use_memory_pos_encoding
        self.predict_confidence    = predict_confidence
        self.num_iters             = num_iters
        self.output_head_init_std  = output_head_init_std
        self.iter_feature_indices  = tuple(iter_feature_indices) if iter_feature_indices else None
        self.use_feature_sampling  = use_feature_sampling
        self.sampling_multi_frame  = sampling_multi_frame and use_feature_sampling
        self.sampling_fusion       = sampling_fusion
        self.feature_dim           = feature_dim
        self.use_relative_pos_bias = use_relative_pos_bias and (not use_feature_sampling)
        self.use_uv_bias           = use_uv_bias and (not use_feature_sampling) and (not self.use_relative_pos_bias)
        self.num_heads             = num_heads
        self.use_temporal_mask     = use_temporal_mask and (not use_feature_sampling)
        self.use_source_feature    = use_source_feature and (not use_feature_sampling)
        self.share_iter_weights    = share_iter_weights and (num_iters > 1) and (not use_feature_sampling)
        self.detach_iter_disp      = detach_iter_disp
        self.use_self_attention    = use_self_attention and (not use_feature_sampling)
        self.use_target_time_adaln = use_target_time_adaln and (not use_feature_sampling)
        if temporal_mask_mode not in {"src_tgt", "tgt_only", "src_only"}:
            raise ValueError(
                f"temporal_mask_mode must be one of ('src_tgt', 'tgt_only', 'src_only'), "
                f"got {temporal_mask_mode!r}"
            )
        self.temporal_mask_mode    = temporal_mask_mode
        self.use_local_window      = use_local_window and (not use_feature_sampling)
        self.local_window_radius   = local_window_radius
        self.local_window_radii    = tuple(local_window_radii) if local_window_radii else None
        if self.iter_feature_indices is not None and self.use_feature_sampling:
            logger.warning(
                "[SparseMotionHead] iter_feature_indices is ignored in feature-sampling mode."
            )
            self.iter_feature_indices = None
        self.use_dynamic_rel_pos_bias = (
            use_dynamic_rel_pos_bias
            and self.use_relative_pos_bias
            and (num_iters > 1)
            and (not use_feature_sampling)
        )

        self.query_encoder = D4RTQueryEncoder(
            query_dim=query_dim, max_frames=max_frames,
            rgb_patch_size=rgb_patch_size,
            use_continuous_time=use_continuous_time,
            time_fourier_scale=query_time_fourier_scale,
            fusion_mode=query_fusion_mode,
            use_uv_skip=use_query_uv_skip,
            time_mode=query_time_mode,
        )

        self.sampling_multiscale = sampling_multiscale and use_feature_sampling

        if use_feature_sampling:
            self.time_encoder = FourierFeatureEncoder(input_dim=2, embed_dim=hidden_dim, scale=10.0)

            self.sampling_corr_radius = sampling_corr_radius
            self.sampling_corr_num_levels = sampling_corr_num_levels
            use_corr = (sampling_fusion == "corr") and self.sampling_multi_frame

            if use_corr:
                corr_side = 2 * sampling_corr_radius + 1
                corr_dim_per_level = corr_side * corr_side
                corr_total_dim = corr_dim_per_level * sampling_corr_num_levels
                n_feat_streams = 1   # only src_feat, correlation replaces tgt info
                logger.info("[SparseMotionHead] CORRELATION mode: radius=%d, levels=%d, corr_dim=%d",
                            sampling_corr_radius, sampling_corr_num_levels, corr_total_dim)
            else:
                corr_total_dim = 0
                if self.sampling_multi_frame and sampling_fusion == "diff_only":
                    n_feat_streams = 1
                elif self.sampling_multi_frame:
                    n_feat_streams = 2
                else:
                    n_feat_streams = 1

            self.use_corr = use_corr
            self.sampling_direct_uv = sampling_direct_uv
            uv_extra_dim = 0
            if sampling_direct_uv:
                self.direct_uv_encoder = FourierFeatureEncoder(
                    input_dim=2, embed_dim=sampling_direct_uv_dim,
                    scale=sampling_direct_uv_scale,
                )
                uv_extra_dim = sampling_direct_uv_dim
                logger.info("[SparseMotionHead] Direct UV injection: dim=%d, scale=%.1f",
                            sampling_direct_uv_dim, sampling_direct_uv_scale)
            else:
                self.direct_uv_encoder = None

            if self.sampling_multiscale:
                ms_dims = list(sampling_multiscale_dims or [256, 512, 1024])
                ms_up = list(sampling_multiscale_upsample or [4, 2, 1])
                self.multiscale_sampler = MultiScaleImplicitSampler(
                    feature_dim=feature_dim,
                    scale_dims=ms_dims,
                    upsample_factors=ms_up,
                    use_learned_upsample=sampling_multiscale_learned_upsample,
                )
                ms_fused_dim = self.multiscale_sampler.output_dim
                mlp_input_dim = n_feat_streams * ms_fused_dim + corr_total_dim + hidden_dim + query_dim + uv_extra_dim
            else:
                self.multiscale_sampler = None
                ms_fused_dim = feature_dim
                mlp_input_dim = n_feat_streams * feature_dim + corr_total_dim + hidden_dim + query_dim + uv_extra_dim

            self.sampling_predict_flow = sampling_predict_flow and self.sampling_multi_frame
            self.sampling_flow_scale = sampling_flow_scale
            if self.sampling_predict_flow:
                flow_input_dim = ms_fused_dim + hidden_dim + query_dim
                self.flow_head = nn.Sequential(
                    nn.Linear(flow_input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 2),
                )
                nn.init.zeros_(self.flow_head[-1].weight)
                nn.init.zeros_(self.flow_head[-1].bias)
                logger.info("[SparseMotionHead] Flow-warped tgt sampling: flow_input=%d, flow_scale=%.2f",
                            flow_input_dim, sampling_flow_scale)
            else:
                self.flow_head = None

            if sampling_use_residual:
                self.sampling_proj = nn.Sequential(
                    nn.Linear(mlp_input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                self.sampling_blocks = nn.ModuleList([
                    ResidualMLPBlock(hidden_dim, expansion=sampling_residual_expansion,
                                     dropout=sampling_dropout)
                    for _ in range(sampling_hidden_layers)
                ])
                self.sampling_norm = nn.LayerNorm(hidden_dim)
                self.sampling_mlp = None
            else:
                layers = [nn.Linear(mlp_input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
                for _ in range(sampling_hidden_layers - 1):
                    layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
                self.sampling_mlp  = nn.Sequential(*layers)
                self.sampling_proj = self.sampling_blocks = self.sampling_norm = None

            self.sampling_use_residual = sampling_use_residual
            self.output_head   = nn.Linear(hidden_dim, 3)
            self.decoder = self.iter_decoders = self.iter_output_heads = self.disp_encoder = None
            self.shared_decoder = self.shared_output_head = None
            self.query_proj    = None
            logger.info("[SparseMotionHead] FEATURE SAMPLING MODE: mlp_layers=%d, hidden=%d, multi_frame=%s, residual=%s, direct_uv=%s, multiscale=%s",
                        sampling_hidden_layers, hidden_dim, self.sampling_multi_frame, sampling_use_residual, sampling_direct_uv, self.sampling_multiscale)
        elif self.share_iter_weights:
            self.shared_decoder = CrossAttentionDecoder(
                hidden_dim, num_heads, num_decoder_layers, hidden_dim * 4, decoder_dropout,
                use_self_attention=self.use_self_attention,
                use_target_time_adaln=self.use_target_time_adaln,
                time_cond_dim=hidden_dim if self.use_target_time_adaln else None,
            )
            self.shared_output_head = nn.Linear(hidden_dim, 3)
            self.disp_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            )
            self.decoder = self.output_head = None
            self.iter_decoders = self.iter_output_heads = None
        elif num_iters > 1:
            self.iter_decoders = nn.ModuleList([
                CrossAttentionDecoder(hidden_dim, num_heads, num_decoder_layers,
                                      hidden_dim * 4, decoder_dropout,
                                      use_self_attention=self.use_self_attention,
                                      use_target_time_adaln=self.use_target_time_adaln,
                                      time_cond_dim=hidden_dim if self.use_target_time_adaln else None)
                for _ in range(num_iters)
            ])
            self.iter_output_heads = nn.ModuleList([nn.Linear(hidden_dim, 3) for _ in range(num_iters)])
            self.disp_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            )
            self.decoder = self.output_head = None
            self.shared_decoder = self.shared_output_head = None
        else:
            self.decoder = CrossAttentionDecoder(
                hidden_dim, num_heads, num_decoder_layers, hidden_dim * 4, decoder_dropout,
                use_self_attention=self.use_self_attention,
                use_target_time_adaln=self.use_target_time_adaln,
                time_cond_dim=hidden_dim if self.use_target_time_adaln else None,
            )
            self.output_head = nn.Linear(hidden_dim, 3)
            self.iter_decoders = self.iter_output_heads = self.disp_encoder = None
            self.shared_decoder = self.shared_output_head = None

        if not use_feature_sampling:
            self.target_time_encoder = (
                FourierFeatureEncoder(1, hidden_dim, target_time_fourier_scale)
                if self.use_target_time_adaln else None
            )
            self.query_proj = nn.Linear(query_dim, hidden_dim)
            if num_feature_scales > 1:
                self.feature_projs = nn.ModuleList([
                    nn.Linear(feature_dim, hidden_dim) for _ in range(num_feature_scales)
                ])
            else:
                self.feature_proj = nn.Linear(feature_dim, hidden_dim)
            if predict_confidence:
                self.confidence_head = nn.Linear(hidden_dim, 1)
            if use_memory_pos_encoding:
                self.memory_spatial_pos_encoder = FourierFeatureEncoder(2, hidden_dim, memory_fourier_scale)
                self.memory_time_pos_encoder    = FourierFeatureEncoder(1, hidden_dim, memory_fourier_scale)
                if use_memory_pos_gate:
                    self.memory_pos_gate = nn.Parameter(torch.tensor(1.0))
            if self.use_relative_pos_bias:
                self.relative_pos_bias = LearnedRelativePosBias(
                    num_heads=num_heads, proj_dim=relative_pos_bias_dim,
                    fourier_dim=64, fourier_scale=memory_fourier_scale,
                    init_log_scale=rel_pos_bias_init_log_scale,
                )
            if self.use_dynamic_rel_pos_bias:
                self.disp_to_uv_offset = nn.Linear(3, 2)
                nn.init.zeros_(self.disp_to_uv_offset.weight)
                nn.init.zeros_(self.disp_to_uv_offset.bias)
            if self.use_uv_bias:
                self.uv_bias_log_sigma = nn.Parameter(torch.full((), math.log(uv_bias_init_sigma)))
            if self.use_source_feature:
                self.source_feature_proj = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
            bias_tag = " (dyn-rel-pos-bias)" if self.use_dynamic_rel_pos_bias \
                else (" (rel-pos-bias)" if self.use_relative_pos_bias
                       else (" (2D-pos+UV-bias)" if self.use_uv_bias else " (2D-pos)"))
            iter_tag = ""
            if self.share_iter_weights:
                iter_tag = f" (shared-iter×{num_iters}, detach={self.detach_iter_disp})"
            elif num_iters > 1:
                iter_tag = f" (iterative×{num_iters})"
            temporal_tag = f" (temporal-mask:{self.temporal_mask_mode})" if self.use_temporal_mask else ""
            window_tag = ""
            if self.use_local_window:
                if self.local_window_radii is not None:
                    window_tag = f" (local-window:{list(self.local_window_radii)})"
                else:
                    window_tag = f" (local-window:r={self.local_window_radius})"
            stage_tag = f" (iter-feat:{list(self.iter_feature_indices)})" if self.iter_feature_indices else ""
            logger.info(
                "[SparseMotionHead] scales=%d, layers=%d, hidden=%d, num_iters=%d, "
                "output_init_std=%.3f%s%s%s%s%s%s%s%s%s%s",
                num_feature_scales, num_decoder_layers, hidden_dim, num_iters,
                output_head_init_std,
                iter_tag,
                bias_tag,
                " (mem-pos-gate)" if use_memory_pos_gate else "",
                " (uv-skip)" if use_query_uv_skip else "",
                temporal_tag,
                window_tag,
                " (src-feat)" if self.use_source_feature else "",
                " (time-adaln)" if self.use_target_time_adaln else "",
                " (self-attn)" if self.use_self_attention else "",
                stage_tag,
            )

        self._init_weights()
        if self.use_dynamic_rel_pos_bias:
            nn.init.zeros_(self.disp_to_uv_offset.weight)
            nn.init.zeros_(self.disp_to_uv_offset.bias)

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                is_out = 'output_head' in name or 'iter_output_heads' in name or 'shared_output_head' in name
                nn.init.trunc_normal_(m.weight, std=self.output_head_init_std if is_out else 0.02)
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

    # ── RGB patch sampling ────────────────────────────────────────────────────

    def _sample_rgb_patches(self, rgb_images, uv, src_frame_idx):
        B, T, C, H, W = rgb_images.shape
        N, ps = uv.shape[1], self.rgb_patch_size
        half, device = ps // 2, uv.device

        uv_px = uv.clone()
        uv_px[..., 0] *= (W - 1)
        uv_px[..., 1] *= (H - 1)

        gy, gx = torch.meshgrid(
            torch.arange(-half, half + 1, device=device),
            torch.arange(-half, half + 1, device=device), indexing='ij',
        )
        offsets = torch.stack([gx, gy], dim=-1).reshape(-1, 2).float()
        patch_coords = uv_px.unsqueeze(2) + offsets
        grid = torch.empty_like(patch_coords)
        grid[..., 0] = 2.0 * patch_coords[..., 0] / max(W - 1, 1) - 1.0
        grid[..., 1] = 2.0 * patch_coords[..., 1] / max(H - 1, 1) - 1.0

        P = ps * ps
        result = torch.zeros(B, N, C * P, device=device, dtype=uv.dtype)
        fi = src_frame_idx.clamp(0, T - 1)
        if not torch.equal(fi, src_frame_idx):
            logger.warning("[_sample_rgb_patches] src_frame_idx clamped to [0, %d].", T - 1)

        def _sample(imgs, g, n_q):
            s = F.grid_sample(imgs, g, mode='bilinear', padding_mode='border', align_corners=True)
            return s[..., 0].reshape(-1, C, n_q, P).permute(0, 2, 1, 3).reshape(-1, n_q, C * P)

        if torch.all(fi == fi[:, :1]):
            for frame in fi[:, 0].unique().tolist():
                bm = fi[:, 0] == frame
                imgs = rgb_images[bm, frame]
                g = grid[bm].reshape(int(bm.sum()), N * P, 1, 2).to(imgs.dtype)
                result[bm] = _sample(imgs, g, N).to(result.dtype)
        else:
            for b in range(B):
                for frame in fi[b].unique().tolist():
                    mask = fi[b] == frame
                    n_q = int(mask.sum())
                    img = rgb_images[b:b+1, frame]
                    g = grid[b, mask].reshape(1, n_q * P, 1, 2).to(img.dtype)
                    s = F.grid_sample(img, g, mode='bilinear', padding_mode='border', align_corners=True)
                    result[b, mask] = s[0, :, :, 0].reshape(C, n_q, P).permute(1, 0, 2).reshape(n_q, C * P).to(result.dtype)
        return result

    # ── Memory building ───────────────────────────────────────────────────────

    @staticmethod
    def _infer_hw(S: int, aspect_ratio: float = 1.0) -> tuple:
        """Infer (H_feat, W_feat) from S tokens, best matching aspect_ratio.

        Considers both (h, w) and (w, h) orientations so that landscape images
        (aspect_ratio < 1, W > H) are handled correctly.  Previously only
        pairs with h >= w were tested, which always returned a portrait or
        square layout and silently transposed feature maps for landscape inputs.
        """
        best_H, best_W, min_diff = 1, S, float('inf')
        for w in range(1, int(math.sqrt(S)) + 2):
            if w == 0 or S % w != 0:
                continue
            h = S // w
            for ch, cw in [(h, w), (w, h)]:
                diff = abs(ch / max(cw, 1) - aspect_ratio)
                if diff < min_diff:
                    min_diff, best_H, best_W = diff, ch, cw
        return best_H, best_W

    def _add_memory_positional_encoding(self, mem: torch.Tensor, B: int,
                                        H_feat: int, W_feat: int) -> tuple:
        """Add 2D spatial + temporal positional encoding. Returns (mem, uv_grid)."""
        BV, S, _ = mem.shape
        device, dtype = mem.device, mem.dtype

        u_c = torch.linspace(0.5 / max(W_feat, 1), 1 - 0.5 / max(W_feat, 1), W_feat, device=device, dtype=dtype)
        v_c = torch.linspace(0.5 / max(H_feat, 1), 1 - 0.5 / max(H_feat, 1), H_feat, device=device, dtype=dtype)
        vv, uu = torch.meshgrid(v_c, u_c, indexing='ij')
        uv_grid = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)  # [S, 2]

        if not self.use_memory_pos_encoding:
            return mem, uv_grid

        if mem.ndim != 3 or BV % B != 0:
            logger.warning("[_add_memory_positional_encoding] unexpected shape %s for B=%d; skip.",
                           tuple(mem.shape), B)
            return mem, uv_grid

        V = BV // B
        mem_bvs = mem.view(B, V, S, self.hidden_dim)
        gate = getattr(self, 'memory_pos_gate', None)
        gate_val = gate.exp() if gate is not None else 1.0
        mem_bvs = mem_bvs + gate_val * self.memory_spatial_pos_encoder(uv_grid).view(1, 1, S, self.hidden_dim)
        t = torch.linspace(0.0, 1.0, max(V, 1), device=device, dtype=dtype)
        mem_bvs = mem_bvs + gate_val * self.memory_time_pos_encoder(t.view(1, V, 1, 1).expand(B, V, S, 1))
        if self.training and gate is not None:
            logger.info("[DEBUG MemPosGate] gate=%.4f (log=%.4f)", gate_val if isinstance(gate_val, float) else gate_val.item(), gate.item())
        return mem_bvs.view(BV, S, self.hidden_dim), uv_grid

    def _project_features(self, features) -> torch.Tensor:
        """Project encoder features → [BV, S, hidden_dim]."""
        if self.num_feature_scales <= 1:
            feat = features[-1] if isinstance(features, (list, tuple)) else features
            if feat.ndim == 4:
                feat = feat.reshape(-1, *feat.shape[2:])
            return self.feature_proj(feat[:, :])  # register tokens kept; sliced later

        if not isinstance(features, (list, tuple)):
            logger.warning("[_project_features] num_feature_scales=%d but got single tensor.",
                           self.num_feature_scales)
            feat = features.reshape(-1, *features.shape[2:]) if features.ndim == 4 else features
            return self.feature_projs[0](feat)

        parts = []
        for proj, feat in zip(self.feature_projs, features[:self.num_feature_scales]):
            if feat.ndim == 4:
                feat = feat.reshape(-1, *feat.shape[2:])
            parts.append(proj(feat))
        if len(parts) == 0:
            raise RuntimeError("No feature scales projected.")
        if len(set(p.shape[1] for p in parts)) > 1:
            logger.warning("[_project_features] multi-scale seq lengths differ: %s",
                           [p.shape[1] for p in parts])
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    @staticmethod
    def _resolve_feature_index(raw_idx: int, num_features: int) -> int:
        """Resolve a possibly negative feature index against a feature list."""
        idx = raw_idx if raw_idx >= 0 else num_features + raw_idx
        if idx < 0 or idx >= num_features:
            raise IndexError(
                f"Feature index {raw_idx} resolves to {idx}, but only {num_features} features are available."
            )
        return idx

    def _build_iter_memories(self, features, patch_start_idx, B, img_size=None):
        """Build one memory bundle per refinement stage from selected feature hooks."""
        if self.iter_feature_indices is None:
            return self._build_memory(features, patch_start_idx, B, img_size=img_size)
        if not isinstance(features, (list, tuple)):
            logger.warning(
                "[_build_iter_memories] iter_feature_indices=%s requested, but features is not a list/tuple.",
                list(self.iter_feature_indices),
            )
            return self._build_memory(features, patch_start_idx, B, img_size=img_size)

        memories = []
        num_features = len(features)
        for raw_idx in self.iter_feature_indices:
            feat_idx = self._resolve_feature_index(raw_idx, num_features)
            memories.append(
                self._build_memory(features[feat_idx], patch_start_idx, B, img_size=img_size)
            )
        return memories

    def _build_memory(self, features, patch_start_idx, B, img_size=None) -> MemoryBundle:
        """Project encoder features → MemoryBundle(tokens=[B,V*S,D], uv=[V*S,2], time=[V*S])."""
        mem_raw = self._project_features(features)          # [BV, S+reg, hidden_dim]
        mem = mem_raw[:, patch_start_idx:]                   # strip register tokens

        BV, S_per_view, _ = mem.shape
        aspect = (img_size[0] / img_size[1]) if (img_size and img_size[1] > 0) else 1.0
        H_feat, W_feat = self._infer_hw(S_per_view, aspect)
        mem, uv_grid = self._add_memory_positional_encoding(mem, B, H_feat, W_feat)

        V = BV // max(B, 1)
        device, dtype = mem.device, mem.dtype
        t_per_view = torch.linspace(0.0, 1.0, max(V, 1), device=device, dtype=dtype)
        mem_time = t_per_view.repeat_interleave(S_per_view)  # [V*S]
        view_idx = torch.arange(V, device=device).repeat_interleave(S_per_view)  # [V*S]
        return MemoryBundle(
            tokens=mem.view(B, -1, self.hidden_dim),
            uv=uv_grid.repeat(V, 1),
            time=mem_time,
            view_idx=view_idx,
            feat_hw=(H_feat, W_feat),
        )

    # ── Query decoding ────────────────────────────────────────────────────────

    @staticmethod
    def _build_temporal_mask(
        src_frame_idx: torch.Tensor,
        tgt_frame_idx: torch.Tensor,
        mem_view_idx: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        mask_mode: str = "src_tgt",
    ) -> torch.Tensor:
        """Build attention mask that only allows attending to selected view tokens.

        Args:
            src_frame_idx: [B, Q] source frame index per query (== view idx)
            tgt_frame_idx: [B, Q] target frame index per query (== view idx)
            mem_view_idx:  [M]    view index for each memory token
        Returns:
            [B, Q, M] mask: 0 for allowed, -inf for blocked.
        """
        B, Q = src_frame_idx.shape
        M = mem_view_idx.shape[0]

        view_exp = mem_view_idx.view(1, 1, M)                      # [1, 1, M]
        src_match = (view_exp == src_frame_idx.unsqueeze(-1))       # [B, Q, M]
        tgt_match = (view_exp == tgt_frame_idx.unsqueeze(-1))       # [B, Q, M]
        if mask_mode == "src_tgt":
            allow = src_match | tgt_match
        elif mask_mode == "tgt_only":
            allow = tgt_match
        elif mask_mode == "src_only":
            allow = src_match
        else:
            raise ValueError(f"Unsupported mask_mode: {mask_mode!r}")

        mask = torch.where(allow, torch.tensor(0.0, device=device, dtype=dtype),
                           torch.tensor(float('-inf'), device=device, dtype=dtype))
        return mask

    def _get_window_radius(self, iter_idx: int) -> Optional[int]:
        """Return the local attention radius (in feature pixels) for iteration ``iter_idx``."""
        if not self.use_local_window:
            return None
        if self.local_window_radii is not None:
            idx = min(iter_idx, len(self.local_window_radii) - 1)
            return int(self.local_window_radii[idx])
        return int(self.local_window_radius)

    @staticmethod
    def _build_local_window_mask(
        center_uv: torch.Tensor,
        tgt_frame_idx: torch.Tensor,
        mem_uv: torch.Tensor,
        mem_view_idx: torch.Tensor,
        feat_hw: tuple,
        radius: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Restrict attention to a target-view spatial window around ``center_uv``.

        The radius is measured in feature-grid pixels, not image pixels. When the
        requested window contains no target-view token (e.g. near the border), the
        closest target token is kept to avoid an all ``-inf`` mask.
        """
        H_feat, W_feat = feat_hw
        M = mem_uv.shape[0]
        mem_uv = mem_uv.to(device=device, dtype=dtype)
        mem_view_idx = mem_view_idx.to(device=device)

        du = (center_uv[..., 0].unsqueeze(-1) - mem_uv[:, 0].view(1, 1, M)).abs()
        dv = (center_uv[..., 1].unsqueeze(-1) - mem_uv[:, 1].view(1, 1, M)).abs()

        u_thr = (radius + 0.5) / max(W_feat, 1)
        v_thr = (radius + 0.5) / max(H_feat, 1)
        tgt_match = (mem_view_idx.view(1, 1, M) == tgt_frame_idx.unsqueeze(-1))
        allow = tgt_match & (du <= u_thr) & (dv <= v_thr)

        has_any = allow.any(dim=-1, keepdim=True)
        if not bool(has_any.all()):
            dist = (du * max(W_feat, 1)).square() + (dv * max(H_feat, 1)).square()
            dist = dist.masked_fill(~tgt_match, float("inf"))
            nearest = dist.argmin(dim=-1, keepdim=True)
            nearest_mask = torch.zeros_like(allow)
            nearest_mask.scatter_(-1, nearest, True)
            allow = torch.where(has_any, allow, nearest_mask)

        return torch.where(
            allow,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(float("-inf"), device=device, dtype=dtype),
        )

    def _inject_source_feature(
        self,
        q_base: torch.Tensor,
        mem_tokens: torch.Tensor,
        uv: torch.Tensor,
        queries: dict,
        mem: MemoryBundle,
    ) -> torch.Tensor:
        """Sample source-view memory features at query UV and inject into query.

        Gives the decoder an explicit "what to track" signal so cross-attention
        can focus on finding the target match rather than jointly locating +
        predicting motion.
        """
        B, Q, _ = q_base.shape
        H_f, W_f = mem.feat_hw
        S = H_f * W_f
        src_view = queries['src_frame_idx'][0, 0].item()

        src_mask = (mem.view_idx == src_view)  # [V*S]
        src_tokens = mem_tokens[:, src_mask]    # [B, S, hidden_dim]

        if src_tokens.shape[1] != S:
            logger.warning(
                "[_inject_source_feature] src_tokens %d != expected %d; skipping.",
                src_tokens.shape[1], S,
            )
            return q_base

        src_map = (
            src_tokens
            .reshape(B, H_f, W_f, self.hidden_dim)
            .permute(0, 3, 1, 2)
        )  # [B, hidden_dim, H_f, W_f]
        grid = (uv * 2 - 1).unsqueeze(2)  # [B, Q, 1, 2]
        sampled = F.grid_sample(
            src_map, grid.to(dtype=src_map.dtype),
            mode='bilinear', padding_mode='border', align_corners=True,
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B, Q, hidden_dim]

        if self.training:
            logger.info(
                "[DEBUG SrcFeat] sampled_std=%.4f q_base_std=%.4f",
                sampled.std(1).mean().item(), q_base.std(1).mean().item(),
            )

        return q_base + self.source_feature_proj(sampled)

    def _prepare_decoder_context(
        self,
        mem,
        queries: dict,
        uv: torch.Tensor,
        iter_idx: int,
    ) -> dict:
        """Prepare stage-specific memory tensors and attention priors."""
        mem_tokens = mem.tokens if isinstance(mem, MemoryBundle) else mem
        mem_uv = mem.uv if isinstance(mem, MemoryBundle) else None
        mem_time = mem.time if isinstance(mem, MemoryBundle) else None
        mem_view = mem.view_idx if isinstance(mem, MemoryBundle) else None
        mem_feat_hw = mem.feat_hw if isinstance(mem, MemoryBundle) else None

        temporal_mask = None
        if self.use_temporal_mask and mem_view is not None:
            temporal_mask = self._build_temporal_mask(
                queries['src_frame_idx'], queries['tgt_frame_idx'],
                mem_view, device=uv.device, dtype=uv.dtype,
                mask_mode=self.temporal_mask_mode,
            )

        locality_bias = None
        bias_q_time = bias_mem_uv = bias_mem_time = None
        if self.use_relative_pos_bias and mem_uv is not None and mem_time is not None:
            q_time = queries.get('tgt_time')
            if q_time is None:
                q_time = queries['tgt_frame_idx'].float() / max(self.max_frames, 1)
            bias_q_time = q_time
            bias_mem_uv = mem_uv.to(uv.dtype)
            bias_mem_time = mem_time.to(uv.dtype)
            locality_bias = self.relative_pos_bias(
                uv, bias_q_time, bias_mem_uv, bias_mem_time,
            )
            if self.training and iter_idx == 0:
                logger.info(
                    "[DEBUG RelPosBias] scale=[%s] | bias_range=[%.2f, %.2f] | bias_std=%.4f",
                    ",".join(f"{s:.3f}" for s in self.relative_pos_bias.log_scale.exp().tolist()),
                    locality_bias.min().item(), locality_bias.max().item(),
                    locality_bias.std().item(),
                )
        elif self.use_uv_bias and mem_uv is not None:
            sigma = self.uv_bias_log_sigma.exp().clamp(min=1e-3)
            dist_sq = ((uv.unsqueeze(2) - mem_uv.to(uv.dtype).unsqueeze(0).unsqueeze(0)) ** 2).sum(-1)
            locality_bias = -dist_sq / (2.0 * sigma ** 2)
            if self.training and iter_idx == 0:
                logger.info(
                    "[DEBUG UV-Bias] sigma=%.4f | bias_range=[%.2f, %.2f] | bias_mean=%.4f",
                    sigma.item(), locality_bias.min().item(),
                    locality_bias.max().item(), locality_bias.mean().item(),
                )

        if temporal_mask is not None:
            if locality_bias is None:
                locality_bias = temporal_mask
            elif locality_bias.ndim == 4:
                locality_bias = locality_bias + temporal_mask.unsqueeze(1)
            else:
                locality_bias = locality_bias + temporal_mask

        if self.use_local_window and mem_uv is not None and mem_view is not None and mem_feat_hw is not None:
            radius = self._get_window_radius(iter_idx)
            if radius is not None:
                window_mask = self._build_local_window_mask(
                    uv, queries['tgt_frame_idx'], mem_uv, mem_view, mem_feat_hw,
                    radius=radius, device=uv.device, dtype=uv.dtype,
                )
                if locality_bias is None:
                    locality_bias = window_mask
                elif locality_bias.ndim == 4:
                    locality_bias = locality_bias + window_mask.unsqueeze(1)
                else:
                    locality_bias = locality_bias + window_mask
                if self.training and iter_idx == 0:
                    allowed = (window_mask == 0).sum(-1).float().mean().item()
                    logger.info(
                        "[DEBUG LocalWindow iter%d] radius=%d | allowed_tokens=%.0f",
                        iter_idx, radius, allowed,
                    )

        if self.training and temporal_mask is not None and iter_idx == 0:
            n_allowed = (temporal_mask == 0).sum(-1).float()
            m_total = temporal_mask.shape[-1]
            logger.info(
                "[DEBUG TemporalMask] allowed_tokens=%.0f/%d (%.1f%%) | unique_tgt_views=%d",
                n_allowed.mean().item(), m_total,
                n_allowed.mean().item() / m_total * 100,
                queries['tgt_frame_idx'].unique().numel(),
            )

        return {
            "mem": mem,
            "tokens": mem_tokens,
            "uv": mem_uv,
            "time": mem_time,
            "view_idx": mem_view,
            "feat_hw": mem_feat_hw,
            "temporal_mask": temporal_mask,
            "locality_bias": locality_bias,
            "q_time": bias_q_time,
            "bias_mem_uv": bias_mem_uv,
            "bias_mem_time": bias_mem_time,
        }

    def _decode_queries(self, mem, rgb_images, queries):
        """Single-pass → [B,Q,3].  Iterative → [B,num_iters,Q,3]."""
        if isinstance(mem, (list, tuple)) and not isinstance(mem, MemoryBundle):
            mem_seq = list(mem)
        else:
            mem_seq = [mem]

        uv = queries['uv']
        base_ctx = self._prepare_decoder_context(mem_seq[0], queries, uv, iter_idx=0)
        mem_tokens = base_ctx["tokens"]
        rgb_patches = self._sample_rgb_patches(rgb_images, uv, queries['src_frame_idx'])
        query_feat  = self.query_encoder(
            uv, queries['src_frame_idx'], queries['tgt_frame_idx'],
            queries['tgt_camera_idx'], rgb_patches,
            src_time=queries.get('src_time'), tgt_time=queries.get('tgt_time'),
            cam_time=queries.get('cam_time'),
        )
        q_base = self.query_proj(query_feat)   # [B, Q, hidden_dim]
        time_cond = None
        if self.use_target_time_adaln and self.target_time_encoder is not None:
            tgt_time = queries.get('tgt_time')
            if tgt_time is None:
                tgt_time = queries['tgt_frame_idx'].float() / max(self.max_frames, 1)
            time_cond = self.target_time_encoder(
                tgt_time.unsqueeze(-1) if tgt_time.ndim == 2 else tgt_time
            )

        # ── Source feature injection: anchor query with source-view appearance ──
        if self.use_source_feature and isinstance(base_ctx["mem"], MemoryBundle) and base_ctx["feat_hw"] is not None:
            q_base = self._inject_source_feature(q_base, mem_tokens, uv, queries, base_ctx["mem"])

        if self.training:
            self._log_query_debug(uv, rgb_patches, query_feat, q_base, mem_tokens, queries)

        if self.num_iters <= 1:
            decoded  = self.decoder(
                tgt=q_base,
                memory=base_ctx["tokens"],
                locality_bias=base_ctx["locality_bias"],
                time_cond=time_cond,
            )
            pred_3d  = self.output_head(decoded)
            if self.training:
                self._log_pred_debug(decoded, pred_3d, self.output_head)
            if not self.predict_confidence:
                return pred_3d
            return {"pred_3d": pred_3d, "pred_log_variance": self.confidence_head(decoded)}

        B, Q, _ = q_base.shape
        pred_disp, all_preds = q_base.new_zeros(B, Q, 3), []

        if self.share_iter_weights:
            for k in range(self.num_iters):
                stage_ctx = base_ctx if k == 0 else self._prepare_decoder_context(
                    mem_seq[min(k, len(mem_seq) - 1)], queries, uv, iter_idx=k,
                )
                disp_input = pred_disp.detach() if self.detach_iter_disp else pred_disp
                iter_bias = self._get_iter_bias(
                    k, stage_ctx["locality_bias"], pred_disp, uv,
                    stage_ctx["q_time"], stage_ctx["bias_mem_uv"], stage_ctx["bias_mem_time"],
                    stage_ctx["temporal_mask"], stage_ctx["view_idx"],
                    queries['tgt_frame_idx'], stage_ctx["feat_hw"],
                )
                decoded = self.shared_decoder(
                    tgt=q_base + self.disp_encoder(disp_input),
                    memory=stage_ctx["tokens"], locality_bias=iter_bias, time_cond=time_cond,
                )
                if not torch.isfinite(decoded).all():
                    finite = torch.isfinite(decoded)
                    logger.warning("[SharedIter %d] %d NaN/Inf — zeroing", k, (~finite).sum().item())
                    decoded = torch.where(finite, decoded, torch.zeros_like(decoded.detach()))
                pred_disp = pred_disp + self.shared_output_head(decoded)
                all_preds.append(pred_disp)
        else:
            for k, (dec, head) in enumerate(zip(self.iter_decoders, self.iter_output_heads)):
                stage_ctx = base_ctx if k == 0 else self._prepare_decoder_context(
                    mem_seq[min(k, len(mem_seq) - 1)], queries, uv, iter_idx=k,
                )
                iter_bias = self._get_iter_bias(
                    k, stage_ctx["locality_bias"], pred_disp, uv,
                    stage_ctx["q_time"], stage_ctx["bias_mem_uv"], stage_ctx["bias_mem_time"],
                    stage_ctx["temporal_mask"], stage_ctx["view_idx"],
                    queries['tgt_frame_idx'], stage_ctx["feat_hw"],
                )
                decoded = dec(
                    tgt=q_base + self.disp_encoder(pred_disp), memory=stage_ctx["tokens"],
                    locality_bias=iter_bias, time_cond=time_cond,
                )
                if not torch.isfinite(decoded).all():
                    finite = torch.isfinite(decoded)
                    logger.warning("[IterDecode iter%d] %d NaN/Inf — zeroing", k, (~finite).sum().item())
                    decoded = torch.where(finite, decoded, torch.zeros_like(decoded.detach()))
                pred_disp = pred_disp + head(decoded)
                all_preds.append(pred_disp)

        return torch.stack(all_preds, dim=1)   # [B, num_iters, Q, 3]

    @staticmethod
    def _log_query_debug(uv, rgb_patches, query_feat, q_base, mem_tokens, queries):
        tgt_time = queries.get('tgt_time')
        n = min(5, uv.shape[1])
        logger.info(
            "[DEBUG Query Diversity] uv_std=%.4f uv_range=[%.3f,%.3f] | "
            "rgb_patch_std=%.4f | query_feat_std=%.4f | q_base_std=%.4f | "
            "mem_std=%.4f | tgt_time_unique=%d",
            uv.std(1).mean().item(), uv.min().item(), uv.max().item(),
            rgb_patches.std(1).mean().item(), query_feat.std(1).mean().item(),
            q_base.std(1).mean().item(), mem_tokens.std(1).mean().item(),
            tgt_time.unique().numel() if tgt_time is not None else 0,
        )
        logger.info("[DEBUG Sample UVs] %s", ["[%.3f,%.3f]" % (u[0], u[1]) for u in uv[0, :n].tolist()])

    @staticmethod
    def _log_pred_debug(decoded, pred_3d, output_head):
        mag = pred_3d.norm(dim=-1)
        n = min(10, pred_3d.shape[1])
        ps = pred_3d[0, :n]
        avg_pd = (ps.unsqueeze(0) - ps.unsqueeze(1)).norm(dim=-1).sum() / (n * (n - 1) + 1e-8)
        pm = pred_3d.mean(1)
        pd = pm / (pm.norm(dim=-1, keepdim=True) + 1e-8)
        logger.info(
            "[DEBUG Prediction] decoded_std=%.4f | pred_std=%.4f | "
            "pred_mag=[%.4f,%.4f] std=%.4f | avg_pairwise_dist=%.4f | "
            "pred_mean_dir=[%.3f,%.3f,%.3f]",
            decoded.std(1).mean().item(), pred_3d.std(1).mean().item(),
            mag.min().item(), mag.max().item(), mag.std().item(), avg_pd.item(),
            pd[0, 0].item(), pd[0, 1].item(), pd[0, 2].item(),
        )
        logger.info("[DEBUG OutputHead] weight_norm=%.4f | bias_norm=%.4f",
                    output_head.weight.norm().item(),
                    output_head.bias.norm().item() if output_head.bias is not None else 0.0)

    def _get_iter_bias(
        self,
        k: int,
        locality_bias: Optional[torch.Tensor],
        pred_disp: torch.Tensor,
        uv: torch.Tensor,
        q_time: Optional[torch.Tensor],
        mem_uv: Optional[torch.Tensor],
        mem_time: Optional[torch.Tensor],
        temporal_mask: Optional[torch.Tensor],
        mem_view_idx: Optional[torch.Tensor],
        tgt_frame_idx: Optional[torch.Tensor],
        feat_hw: Optional[tuple],
    ) -> Optional[torch.Tensor]:
        """Compute attention bias for iteration *k* of the refinement loop.

        k == 0 always uses the source-position locality_bias.
        k > 0 behaviour depends on ``use_dynamic_rel_pos_bias``:
          - True  → recompute bias from predicted target UV (source UV + learned offset).
          - False → None (free attention, original V23 behaviour).
        """
        if k == 0:
            return locality_bias
        if self.use_dynamic_rel_pos_bias and q_time is not None and mem_uv is not None and mem_time is not None:
            disp_for_bias = pred_disp.detach() if self.detach_iter_disp else pred_disp
            pred_uv_offset = self.disp_to_uv_offset(disp_for_bias)
            dynamic_uv = (uv + pred_uv_offset).clamp(0.0, 1.0)
            dyn_bias = self.relative_pos_bias(dynamic_uv, q_time, mem_uv, mem_time)
            if temporal_mask is not None:
                dyn_bias = dyn_bias + temporal_mask.unsqueeze(1)
            radius = self._get_window_radius(k)
            if radius is not None and mem_view_idx is not None and tgt_frame_idx is not None and feat_hw is not None:
                window_mask = self._build_local_window_mask(
                    dynamic_uv, tgt_frame_idx, mem_uv, mem_view_idx, feat_hw,
                    radius=radius, device=uv.device, dtype=uv.dtype,
                )
                dyn_bias = dyn_bias + window_mask.unsqueeze(1)
            if self.training:
                window_msg = f" | radius={radius}" if radius is not None else ""
                finite_mask = torch.isfinite(dyn_bias)
                finite_bias = dyn_bias[finite_mask]
                if finite_bias.numel() > 0:
                    bias_min = finite_bias.min().item()
                    bias_max = finite_bias.max().item()
                    bias_std = finite_bias.std(unbiased=False).item() if finite_bias.numel() > 1 else 0.0
                else:
                    bias_min = float("nan")
                    bias_max = float("nan")
                    bias_std = float("nan")
                allowed_tokens = (
                    finite_mask[:, 0].sum(-1).float().mean().item()
                    if dyn_bias.ndim == 4 else
                    finite_mask.sum(-1).float().mean().item()
                )
                logger.info(
                    "[DEBUG DynRelPosBias iter%d] uv_offset_mag=%.4f "
                    "| finite_bias_range=[%.2f, %.2f] | finite_bias_std=%.4f "
                    "| allowed_tokens=%.0f%s",
                    k, pred_uv_offset.norm(dim=-1).mean().item(),
                    bias_min, bias_max, bias_std, allowed_tokens, window_msg,
                )
            return dyn_bias
        # Fallback: reuse source-position bias for all refinement iterations.
        # Without ANY spatial prior, cross-attention degenerates to uniform
        # weights (entropy → log(M)), causing all queries to produce identical
        # outputs and prediction collapse.
        return locality_bias

    def decode_with_memory(self, mem, rgb_images, queries):
        """Public entry for per-target-frame inference; auto-chunks if needed."""
        N, B = queries['uv'].shape[1], queries['uv'].shape[0]
        if not self.training and N > self.infer_chunk_size:
            return self._forward_chunked(mem, rgb_images, queries, B, N)
        return self._decode_queries(mem, rgb_images, queries)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, features, rgb_images, queries, img_size, patch_start_idx=0, meta_data=None):
        B = queries['uv'].shape[0]
        if self.use_feature_sampling:
            return self._forward_feature_sampling(features, rgb_images, queries, img_size, patch_start_idx)
        mem = self._build_iter_memories(features, patch_start_idx, B, img_size=img_size)
        return self.decode_with_memory(mem, rgb_images, queries)

    def _forward_feature_sampling(self, features, rgb_images, queries, img_size, patch_start_idx):
        """Bilinear-sample encoder features at query UV locations.

        When sampling_multi_frame=True, samples from both source and target
        frames so the MLP can compare "what was here" vs "what is here now".

        When sampling_multiscale=True, uses InfiniDepth-style multi-scale
        implicit decoder: bilinear-sample from multi-hook feature pyramids
        and hierarchically fuse from shallow (high-res) to deep (semantic).
        """
        uv = queries['uv']   # [B, Q, 2]
        B, Q, _ = uv.shape
        V = rgb_images.shape[1]

        feat = features[-1] if isinstance(features, (list, tuple)) else features
        if feat.ndim == 4:
            feat = feat.reshape(-1, *feat.shape[2:])
        BV, S, C = feat.shape
        S_spatial = S - patch_start_idx

        H_img, W_img  = img_size if img_size else (518, 518)
        aspect = H_img / max(W_img, 1)
        H_feat, W_feat = self._infer_hw(S_spatial, aspect)

        src_frame = queries['src_frame_idx'][0, 0].item()

        # Precompute time & query encodings (needed early for flow prediction)
        src_time = queries.get('src_time', torch.zeros(B, Q, device=uv.device))
        tgt_time = queries.get('tgt_time', torch.zeros(B, Q, device=uv.device))
        time_feat = self.time_encoder(torch.stack([tgt_time - src_time, tgt_time], dim=-1))
        rgb_patches = self._sample_rgb_patches(rgb_images, uv, queries['src_frame_idx'])
        query_feat = self.query_encoder(
            uv, queries['src_frame_idx'], queries['tgt_frame_idx'],
            queries['tgt_camera_idx'], rgb_patches,
            src_time=src_time, tgt_time=tgt_time, cam_time=queries.get('cam_time'),
        )
        pred_flow = None

        # ── Multi-scale implicit sampling (InfiniDepth-style) ──
        if self.sampling_multiscale and self.multiscale_sampler is not None:
            multi_features = features if isinstance(features, (list, tuple)) else [features]
            pyramid = self.multiscale_sampler._build_pyramid(
                multi_features, patch_start_idx, H_feat, W_feat,
            )
            sampled_src = self.multiscale_sampler.sample_and_fuse(
                pyramid, uv, src_frame, B, V,
            )

            if self.sampling_multi_frame:
                tgt_frame_idx = queries['tgt_frame_idx']   # [B, Q]

                if self.sampling_predict_flow and self.flow_head is not None:
                    # Predict 2D flow → sample tgt at corresponding location
                    pred_flow = self.flow_head(
                        torch.cat([sampled_src, time_feat, query_feat], dim=-1)
                    ) * self.sampling_flow_scale                           # [B, Q, 2]
                    tgt_uv = (uv + pred_flow).clamp(0.0, 1.0)             # [B, Q, 2]

                    sampled_tgt = torch.zeros_like(sampled_src)
                    unique_tgts = tgt_frame_idx[0].unique()
                    for tf in unique_tgts:
                        mask = (tgt_frame_idx[0] == tf)
                        tgt_fused = self.multiscale_sampler.sample_and_fuse(
                            pyramid, tgt_uv[:, mask], tf.item(), B, V,
                        )
                        sampled_tgt[:, mask] = tgt_fused

                    if self.training:
                        flow_mag = pred_flow.norm(dim=-1)
                        logger.info(
                            "[DEBUG PredFlow] flow_mag=[%.4f,%.4f] mean=%.4f | "
                            "tgt_uv_range=[%.3f,%.3f]",
                            flow_mag.min().item(), flow_mag.max().item(),
                            flow_mag.mean().item(),
                            tgt_uv.min().item(), tgt_uv.max().item(),
                        )
                else:
                    # Fallback: sample tgt at same UV (original feat_diff)
                    sampled_tgt = torch.zeros_like(sampled_src)
                    unique_tgts = tgt_frame_idx[0].unique()
                    for tf in unique_tgts:
                        mask = (tgt_frame_idx[0] == tf)
                        tgt_fused = self.multiscale_sampler.sample_and_fuse(
                            pyramid, uv[:, mask], tf.item(), B, V,
                        )
                        sampled_tgt[:, mask] = tgt_fused

                if self.sampling_fusion == "concat":
                    sampled = torch.cat([sampled_src, sampled_tgt], dim=-1)
                elif self.sampling_fusion == "diff":
                    sampled = torch.cat([sampled_src, sampled_tgt - sampled_src], dim=-1)
                elif self.sampling_fusion == "diff_only":
                    sampled = sampled_tgt - sampled_src
                else:
                    sampled = torch.cat([sampled_src, sampled_tgt], dim=-1)
            else:
                sampled = sampled_src
            corr_map = None

        # ── Original single-scale sampling ──
        else:
            feat_spatial = (feat[:, patch_start_idx:]
                            .reshape(BV, H_feat, W_feat, C)
                            .permute(0, 3, 1, 2)
                            .view(B, V, C, H_feat, W_feat))

            grid = (uv * 2 - 1).unsqueeze(2)   # [B, Q, 1, 2]

            src_feat  = feat_spatial[:, src_frame]   # [B, C, H, W]
            sampled_src = F.grid_sample(src_feat, grid, mode='bilinear',
                                        padding_mode='border', align_corners=True)
            sampled_src = sampled_src.squeeze(-1).permute(0, 2, 1)   # [B, Q, C]

            if self.sampling_multi_frame and self.use_corr:
                # ── Local Correlation Volume (RAFT-style) ──
                tgt_frame_idx = queries['tgt_frame_idx']   # [B, Q]
                r = self.sampling_corr_radius
                side = 2 * r + 1
                device, dtype = uv.device, uv.dtype

                delta_h = 2.0 / max(H_feat - 1, 1)
                delta_w = 2.0 / max(W_feat - 1, 1)
                oy, ox = torch.meshgrid(
                    torch.arange(-r, r + 1, device=device, dtype=dtype),
                    torch.arange(-r, r + 1, device=device, dtype=dtype),
                    indexing='ij',
                )
                offsets_hw = torch.stack([ox * delta_w, oy * delta_h], dim=-1)
                offsets_flat = offsets_hw.reshape(-1, 2)

                src_feat_norm = F.normalize(sampled_src, dim=-1)

                corr_levels = []
                for lvl in range(self.sampling_corr_num_levels):
                    scale = 2 ** lvl
                    lvl_offsets = offsets_flat * scale

                    corr_lvl = torch.zeros(B, Q, side * side, device=device, dtype=dtype)
                    unique_tgts = tgt_frame_idx[0].unique()
                    for tf in unique_tgts:
                        mask = (tgt_frame_idx[0] == tf)
                        n_q = mask.sum().item()
                        tgt_feat_map = feat_spatial[:, tf.item()]

                        center_grid = (uv[:, mask] * 2 - 1)
                        tgt_grid = center_grid.unsqueeze(2) + lvl_offsets.unsqueeze(0).unsqueeze(0)
                        tgt_grid_flat = tgt_grid.reshape(B, n_q * side * side, 1, 2)
                        tgt_sampled = F.grid_sample(tgt_feat_map, tgt_grid_flat, mode='bilinear',
                                                    padding_mode='border', align_corners=True)
                        tgt_sampled = (tgt_sampled.squeeze(-1)
                                       .reshape(B, C, n_q, side * side)
                                       .permute(0, 2, 3, 1))
                        tgt_sampled_norm = F.normalize(tgt_sampled, dim=-1)

                        src_for_mask = src_feat_norm[:, mask]
                        dot = torch.einsum('bqc,bqpc->bqp', src_for_mask, tgt_sampled_norm)
                        corr_lvl[:, mask] = dot

                    corr_levels.append(corr_lvl)

                corr_map = torch.cat(corr_levels, dim=-1)
                sampled = sampled_src

            elif self.sampling_multi_frame:
                tgt_frame_idx = queries['tgt_frame_idx']
                sampled_tgt = torch.zeros_like(sampled_src)
                unique_tgts = tgt_frame_idx[0].unique()
                for tf in unique_tgts:
                    mask = (tgt_frame_idx[0] == tf)
                    tgt_feat = feat_spatial[:, tf.item()]
                    tgt_grid = grid[:, mask]
                    s = F.grid_sample(tgt_feat, tgt_grid, mode='bilinear',
                                      padding_mode='border', align_corners=True)
                    sampled_tgt[:, mask] = s.squeeze(-1).permute(0, 2, 1)

                feat_diff = sampled_tgt - sampled_src
                if self.sampling_fusion == "concat":
                    sampled = torch.cat([sampled_src, sampled_tgt], dim=-1)
                elif self.sampling_fusion == "diff":
                    sampled = torch.cat([sampled_src, feat_diff], dim=-1)
                elif self.sampling_fusion == "diff_only":
                    sampled = feat_diff
                else:
                    sampled = torch.cat([sampled_src, sampled_tgt], dim=-1)
                corr_map = None
            else:
                sampled = sampled_src
                corr_map = None

        if not (self.sampling_multiscale and self.multiscale_sampler is not None):
            # time_feat & query_feat already computed for multiscale path
            src_time = queries.get('src_time', torch.zeros(B, Q, device=uv.device))
            tgt_time = queries.get('tgt_time', torch.zeros(B, Q, device=uv.device))
            time_feat = self.time_encoder(torch.stack([tgt_time - src_time, tgt_time], dim=-1))
            rgb_patches = self._sample_rgb_patches(rgb_images, uv, queries['src_frame_idx'])
            query_feat = self.query_encoder(
                uv, queries['src_frame_idx'], queries['tgt_frame_idx'],
                queries['tgt_camera_idx'], rgb_patches,
                src_time=src_time, tgt_time=tgt_time, cam_time=queries.get('cam_time'),
            )

        parts = [sampled]
        if corr_map is not None:
            parts.append(corr_map)
        parts.extend([time_feat, query_feat])
        if self.sampling_direct_uv:
            parts.append(self.direct_uv_encoder(uv))
        cat_input = torch.cat(parts, dim=-1)
        if self.sampling_use_residual:
            hidden = self.sampling_proj(cat_input)
            for blk in self.sampling_blocks:
                hidden = blk(hidden)
            hidden = self.sampling_norm(hidden)
        else:
            hidden = self.sampling_mlp(cat_input)
        pred_3d = self.output_head(hidden)

        if self.training:
            mag = pred_3d.norm(dim=-1)
            corr_info = ""
            if corr_map is not None:
                corr_info = " | corr_mean=%.4f | corr_peak=%.4f" % (
                    corr_map.mean().item(), corr_map.max(dim=-1).values.mean().item())
            ms_info = " | multiscale=True" if self.sampling_multiscale else ""
            logger.info(
                "[DEBUG FeatureSampling] sampled_std=%.4f | time_std=%.4f | "
                "query_std=%.4f | hidden_std=%.4f | pred_std=%.4f | pred_mag=[%.4f,%.4f]%s%s",
                sampled.std(1).mean().item(), time_feat.std(1).mean().item(),
                query_feat.std(1).mean().item(), hidden.std(1).mean().item(),
                pred_3d.std(1).mean().item(), mag.min().item(), mag.max().item(),
                corr_info, ms_info,
            )
        if pred_flow is not None:
            return {'pred_3d': pred_3d, 'pred_flow': pred_flow}
        return pred_3d

    def _forward_chunked(self, mem, rgb_images, queries, B, N):
        """Process queries in chunks (D4RT: no inter-query attention → identical result)."""
        C, all_pred = self.infer_chunk_size, []
        for start in range(0, N, C):
            end = min(start + C, N)
            chunk_q = {
                k: (v[:, start:end] if torch.is_tensor(v) and v.ndim >= 2
                    and v.shape[0] == B and v.shape[1] == N else v)
                for k, v in queries.items()
            }
            all_pred.append(self._decode_queries(mem, rgb_images, chunk_q))

        if not all_pred:
            return None
        if isinstance(all_pred[0], dict):
            out = {}
            for key in all_pred[0]:
                out[key] = torch.cat([p[key] for p in all_pred], dim=1)
            return out
        dim = 1 if all_pred[0].ndim == 3 else 2
        return torch.cat(all_pred, dim=dim)
