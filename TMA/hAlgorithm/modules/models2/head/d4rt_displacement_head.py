"""D4RT-Faithful Displacement Head (arXiv:2512.08924)

Pure cross-attention decoder with deterministic Fourier UV embedding,
discrete timestep embeddings, and additive query fusion.

Architecture stack (V5):
  - Memory spatial + temporal positional encoding
  - Source feature injection (encoder feature sampled at query UV)
  - Factored relative position bias with temporal dimension (3D: UV + time)
    using random Fourier features and per-head dot-product — prevents both
    direction collapse and magnitude collapse by creating position-dependent
    attention patterns that differ across space AND time.
  - Decoder dropout for regularization (prevents collapse to trivial zero)
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class D4RTFourierEmbedding(nn.Module):
    """Deterministic Fourier embedding with log-spaced frequencies for 2D coords.

    Uses 2^k frequencies (k=0..num_frequencies-1), NOT random projections.
    """

    def __init__(self, embed_dim: int, num_frequencies: int = 64):
        super().__init__()
        freqs = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freqs", freqs)
        fourier_dim = 2 * 2 * num_frequencies
        self.proj = nn.Linear(fourier_dim, embed_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (B, N, 2) normalised [0, 1] → (B, N, embed_dim)."""
        B, N, _ = coords.shape
        coords_freq = coords.unsqueeze(-1) * self.freqs * 2 * math.pi
        fourier = torch.cat([torch.sin(coords_freq), torch.cos(coords_freq)], dim=-1)
        return self.proj(fourier.reshape(B, N, -1))


class D4RT1DFourierEmbedding(nn.Module):
    """Deterministic Fourier embedding for 1D scalars (e.g. time index)."""

    def __init__(self, embed_dim: int, num_frequencies: int = 64):
        super().__init__()
        freqs = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freqs", freqs)
        fourier_dim = 2 * num_frequencies
        self.proj = nn.Linear(fourier_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 1) or (...,) scalar in [0, 1] → (..., embed_dim)."""
        if x.ndim > 0 and x.shape[-1] != 1:
            x = x.unsqueeze(-1)
        x_freq = x * self.freqs * 2 * math.pi
        fourier = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)
        return self.proj(fourier)


class RelativePosBiasLight(nn.Module):
    """Learned attention bias from 2D spatial offset between query and memory UVs.

    For each (query, memory_token) pair, encodes the 2D offset using Fourier
    features and projects to per-head scalar bias. This directly creates
    position-dependent attention patterns.
    """

    def __init__(self, num_heads: int, fourier_dim: int = 64, num_frequencies: int = 32):
        super().__init__()
        self.num_heads = num_heads
        freqs = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freqs", freqs)
        input_dim = 2 * 2 * num_frequencies
        self.proj = nn.Sequential(
            nn.Linear(input_dim, fourier_dim),
            nn.GELU(),
            nn.Linear(fourier_dim, num_heads),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, q_uv: torch.Tensor, m_uv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_uv: [B, Nq, 2] query UV positions in [0, 1]
            m_uv: [Nm, 2] memory token UV positions (single view) in [0, 1]
        Returns:
            [B, H, Nq, Nm] attention bias
        """
        offset = q_uv.unsqueeze(2) - m_uv.unsqueeze(0).unsqueeze(0)
        offset_freq = offset.unsqueeze(-1) * self.freqs * 2 * math.pi
        fourier = torch.cat([torch.sin(offset_freq), torch.cos(offset_freq)], dim=-1)
        B, Nq, Nm, D2, F2 = fourier.shape
        bias = self.proj(fourier.reshape(B, Nq, Nm, -1))
        return bias.permute(0, 3, 1, 2)


class D4RTTimestepEmbedding(nn.Module):
    """Learnable discrete embeddings for src / tgt / cam timesteps."""

    def __init__(self, max_timesteps: int, embed_dim: int):
        super().__init__()
        self.src_embedding = nn.Embedding(max_timesteps, embed_dim)
        self.tgt_embedding = nn.Embedding(max_timesteps, embed_dim)
        self.cam_embedding = nn.Embedding(max_timesteps, embed_dim)
        nn.init.normal_(self.src_embedding.weight, std=0.02)
        nn.init.normal_(self.tgt_embedding.weight, std=0.02)
        nn.init.normal_(self.cam_embedding.weight, std=0.02)

    def forward(self, t_src, t_tgt, t_cam):
        return (
            self.src_embedding(t_src),
            self.tgt_embedding(t_tgt),
            self.cam_embedding(t_cam),
        )


class D4RTCrossAttention(nn.Module):
    """Cross-attention with optional relative position bias for spatial discrimination."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            attn_bias: optional [B, H, Nq, Nkv] additive bias for attention logits
        """
        B, Nq, C = query.shape
        Nkv = key_value.shape[1]
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(query).reshape(B, Nq, H, d).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, Nkv, H, d).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, Nkv, H, d).transpose(1, 2)

        if attn_bias is not None:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias,
                dropout_p=self.attn_drop if self.training else 0.0,
            )
        else:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop if self.training else 0.0,
            )
        x = x.transpose(1, 2).reshape(B, Nq, C)
        return self.proj_drop(self.proj(x))


class D4RTDecoderBlock(nn.Module):
    """Pre-LN cross-attention + pre-LN MLP (no self-attention)."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = D4RTCrossAttention(dim, num_heads, True, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, query: torch.Tensor, encoder_features: torch.Tensor,
                attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        query = query + self.cross_attn(self.norm1(query), self.norm_kv(encoder_features),
                                        attn_bias=attn_bias)
        query = query + self.mlp(self.norm2(query))
        return query


class D4RTDisplacementHead(nn.Module):
    """Faithful D4RT decoder for displacement prediction.

    Query = FourierEmbed(uv) + TimestepEmbed(t_src) + TimestepEmbed(t_tgt)
          + TimestepEmbed(t_cam) + PatchMLP(rgb_patch) + learnable_token

    Decoder = N × DecoderBlock(cross-attn + MLP) → LayerNorm → Linear(3)
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        max_timesteps: int = 128,
        rgb_patch_size: int = 9,
        num_fourier_freqs: int = 64,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        infer_chunk_size: int = 2048,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rgb_patch_size = rgb_patch_size
        self.infer_chunk_size = infer_chunk_size

        self.feature_proj = nn.Linear(feature_dim, embed_dim)

        self.mem_spatial_fourier = D4RTFourierEmbedding(embed_dim, num_fourier_freqs)
        self.mem_time_fourier = D4RT1DFourierEmbedding(embed_dim, num_fourier_freqs)

        self.fourier_embed = D4RTFourierEmbedding(embed_dim, num_fourier_freqs)
        self.timestep_embed = D4RTTimestepEmbedding(max_timesteps, embed_dim)

        patch_dim = rgb_patch_size * rgb_patch_size * 3
        self.patch_mlp = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.src_feat_proj = nn.Linear(embed_dim, embed_dim)
        self.rel_pos_bias = RelativePosBiasLight(num_heads, fourier_dim=64, num_frequencies=32)
        self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            D4RTDecoderBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, 3)

        self._init_weights()

        logger.info(
            "[D4RTDisplacementHead V5] feature_dim=%d embed_dim=%d depth=%d "
            "heads=%d mlp_ratio=%.1f max_t=%d patch=%d fourier_freqs=%d "
            "dropout=%.2f rel_pos_bias=factored_3D(UV+time)",
            feature_dim, embed_dim, depth, num_heads,
            mlp_ratio, max_timesteps, rgb_patch_size, num_fourier_freqs,
            drop_rate,
        )

    def _init_weights(self):
        nn.init.trunc_normal_(self.query_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── RGB patch extraction (memory-efficient) ──────────────────────────

    def _sample_rgb_patches(
        self,
        rgb_images: torch.Tensor,
        uv: torch.Tensor,
        src_frame_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Extract flattened RGB patches at query UV locations.

        Args:
            rgb_images:    [B, V, C, H, W]
            uv:            [B, Q, 2] normalised UV in [0, 1]
            src_frame_idx: [B, Q] int source frame index

        Returns:
            [B, Q, patch_size^2 * 3]
        """
        B, T, C, H, W = rgb_images.shape
        N, ps = uv.shape[1], self.rgb_patch_size
        half, device = ps // 2, uv.device

        uv_px = uv.clone()
        uv_px[..., 0] *= (W - 1)
        uv_px[..., 1] *= (H - 1)

        gy, gx = torch.meshgrid(
            torch.arange(-half, half + 1, device=device),
            torch.arange(-half, half + 1, device=device),
            indexing="ij",
        )
        offsets = torch.stack([gx, gy], dim=-1).reshape(-1, 2).float()
        patch_coords = uv_px.unsqueeze(2) + offsets
        grid = torch.empty_like(patch_coords)
        grid[..., 0] = 2.0 * patch_coords[..., 0] / max(W - 1, 1) - 1.0
        grid[..., 1] = 2.0 * patch_coords[..., 1] / max(H - 1, 1) - 1.0

        P = ps * ps
        result = torch.zeros(B, N, C * P, device=device, dtype=uv.dtype)
        fi = src_frame_idx.clamp(0, T - 1)

        def _sample(imgs, g, n_q):
            s = F.grid_sample(imgs, g, mode="bilinear", padding_mode="border", align_corners=True)
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
                    img = rgb_images[b : b + 1, frame]
                    g = grid[b, mask].reshape(1, n_q * P, 1, 2).to(img.dtype)
                    s = F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)
                    result[b, mask] = (
                        s[0, :, :, 0]
                        .reshape(C, n_q, P)
                        .permute(1, 0, 2)
                        .reshape(n_q, C * P)
                        .to(result.dtype)
                    )
        return result

    # ── Memory / Query building ──────────────────────────────────────────

    @staticmethod
    def _infer_hw(S: int, aspect_ratio: float = 1.0) -> tuple:
        """Infer (H_feat, W_feat) from S tokens, best matching aspect_ratio.

        Considers both (h, w) and (w, h) orientations so that landscape images
        (aspect_ratio < 1, W > H) are handled correctly.
        """
        best_H, best_W, min_diff = 1, S, float("inf")
        for w in range(1, int(math.sqrt(S)) + 2):
            if w == 0 or S % w != 0:
                continue
            h = S // w
            for ch, cw in [(h, w), (w, h)]:
                diff = abs(ch / max(cw, 1) - aspect_ratio)
                if diff < min_diff:
                    min_diff, best_H, best_W = diff, ch, cw
        return best_H, best_W

    def _build_memory(self, features, patch_start_idx: int, B: int,
                      img_size=None, **kwargs):
        """Project encoder features → memory [B, V*S, D], spatial map, and UV grid.

        Returns:
            memory:   [B, V*S, embed_dim] with spatial+temporal PE
            feat_map: [B, V, embed_dim, H_feat, W_feat] for source feature sampling
            uv_grid:  [S, 2] per-view UV grid (tiled V times for full bias)
            V:        number of views
        """
        feat = features[-1] if isinstance(features, (list, tuple)) else features
        if feat.ndim == 4:
            feat = feat.reshape(-1, *feat.shape[2:])

        mem = feat[:, patch_start_idx:]
        mem = self.feature_proj(mem)

        BV, S, _ = mem.shape
        V = BV // max(B, 1)
        device, dtype = mem.device, mem.dtype

        aspect = (img_size[0] / img_size[1]) if (img_size and img_size[1] > 0) else 1.0
        H_feat, W_feat = self._infer_hw(S, aspect)

        feat_map = mem.view(B, V, H_feat, W_feat, self.embed_dim).permute(0, 1, 4, 2, 3)

        u_c = torch.linspace(0.5 / max(W_feat, 1), 1.0 - 0.5 / max(W_feat, 1), W_feat, device=device, dtype=dtype)
        v_c = torch.linspace(0.5 / max(H_feat, 1), 1.0 - 0.5 / max(H_feat, 1), H_feat, device=device, dtype=dtype)
        vv, uu = torch.meshgrid(v_c, u_c, indexing="ij")
        uv_grid = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)  # [S, 2]

        spatial_pe = self.mem_spatial_fourier(uv_grid.unsqueeze(0))
        mem = mem + spatial_pe

        t_vals = torch.linspace(0.0, 1.0, max(V, 1), device=device, dtype=dtype)
        t_per_token = t_vals.view(V, 1).expand(V, S).reshape(V * S)
        time_pe = self.mem_time_fourier(t_per_token.unsqueeze(-1))
        mem_bvs = mem.view(B, V, S, self.embed_dim)
        mem_bvs = mem_bvs + time_pe.view(1, V, S, self.embed_dim)

        return mem_bvs.reshape(B, V * S, self.embed_dim), feat_map, uv_grid, V

    def _sample_src_features(
        self,
        feat_map: torch.Tensor,
        uv: torch.Tensor,
        src_frame_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinearly sample encoder features at query UV from the source frame.

        Args:
            feat_map:      [B, V, D, H_feat, W_feat]
            uv:            [B, Q, 2] normalised UV in [0, 1]
            src_frame_idx: [B, Q] int source frame index

        Returns:
            [B, Q, embed_dim]
        """
        B, V, D, H, W = feat_map.shape
        Q = uv.shape[1]

        grid = uv.clone()
        grid[..., 0] = 2.0 * grid[..., 0] - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] - 1.0
        grid = grid.unsqueeze(2)  # [B, Q, 1, 2]

        result = torch.zeros(B, Q, D, device=uv.device, dtype=feat_map.dtype)
        fi = src_frame_idx.clamp(0, V - 1)

        if torch.all(fi == fi[:, :1]):
            for frame in fi[:, 0].unique().tolist():
                bm = fi[:, 0] == frame
                imgs = feat_map[bm, frame]  # [B_sub, D, H, W]
                g = grid[bm].to(imgs.dtype)  # [B_sub, Q, 1, 2]
                sampled = F.grid_sample(imgs, g, mode="bilinear", padding_mode="border", align_corners=True)
                result[bm] = sampled[:, :, :, 0].permute(0, 2, 1).to(result.dtype)
        else:
            for b in range(B):
                for frame in fi[b].unique().tolist():
                    mask = fi[b] == frame
                    n_q = int(mask.sum())
                    img = feat_map[b : b + 1, frame]  # [1, D, H, W]
                    g = grid[b, mask].unsqueeze(0).to(img.dtype)  # [1, n_q, 1, 2]
                    sampled = F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)
                    result[b, mask] = sampled[0, :, :, 0].permute(1, 0).to(result.dtype)

        return result

    def _build_query(self, rgb_images: torch.Tensor, queries: dict,
                     feat_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Query fusion with source feature injection for spatial discrimination."""
        uv = queries["uv"]
        t_src = queries["src_frame_idx"]
        t_tgt = queries["tgt_frame_idx"]
        t_cam = queries["tgt_camera_idx"]
        B, Q = uv.shape[:2]

        coord_emb = self.fourier_embed(uv)
        src_emb, tgt_emb, cam_emb = self.timestep_embed(t_src, t_tgt, t_cam)

        rgb_patches = self._sample_rgb_patches(rgb_images, uv, t_src)
        patch_emb = self.patch_mlp(rgb_patches)

        query = coord_emb + src_emb + tgt_emb + cam_emb + patch_emb

        if feat_map is not None:
            src_feat = self._sample_src_features(feat_map, uv, t_src)
            query = query + self.src_feat_proj(src_feat)

        query = query + self.query_token.expand(B, Q, -1)
        return query

    def _decode(self, query: torch.Tensor, memory: torch.Tensor,
                attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            query = block(query, memory, attn_bias=attn_bias)
        return self.norm(query)

    # ── Forward ──────────────────────────────────────────────────────────

    def _compute_rel_pos_bias(self, q_uv: torch.Tensor,
                              per_view_uv: torch.Tensor,
                              num_views: int) -> torch.Tensor:
        """Compute per-head relative position bias [B, H, Nq, V*S].

        Computes bias against one view's UV grid [S, 2], then tiles V times.
        Memory-efficient: V times cheaper than computing against all V*S tokens.
        """
        bias_1v = self.rel_pos_bias(q_uv, per_view_uv)  # [B, H, Nq, S]
        if num_views == 1:
            return bias_1v
        return bias_1v.repeat(1, 1, 1, num_views)  # [B, H, Nq, V*S]

    def forward(self, features, rgb_images, queries, img_size,
                patch_start_idx: int = 0, meta_data=None):
        B, N = queries["uv"].shape[:2]

        memory, feat_map, per_view_uv, num_views = self._build_memory(
            features, patch_start_idx, B, img_size=img_size)

        attn_bias = self._compute_rel_pos_bias(queries["uv"], per_view_uv, num_views)

        if not self.training and N > self.infer_chunk_size:
            return self._forward_chunked(
                memory, feat_map, per_view_uv, num_views, rgb_images, queries, B, N)

        query = self._build_query(rgb_images, queries, feat_map=feat_map)
        decoded = self._decode(query, memory, attn_bias=attn_bias)
        return self.output_head(decoded)

    def decode_with_memory(self, memory_bundle, rgb_images, queries):
        """Public entry for per-target-frame inference; auto-chunks.

        memory_bundle: (memory, feat_map, per_view_uv, num_views) tuple.
        """
        if isinstance(memory_bundle, tuple) and len(memory_bundle) == 4:
            memory, feat_map, per_view_uv, num_views = memory_bundle
        elif isinstance(memory_bundle, tuple) and len(memory_bundle) == 3:
            memory, feat_map, per_view_uv = memory_bundle
            num_views = 1
        elif isinstance(memory_bundle, tuple) and len(memory_bundle) == 2:
            memory, feat_map = memory_bundle
            per_view_uv, num_views = None, 1
        else:
            memory, feat_map, per_view_uv, num_views = memory_bundle, None, None, 1

        B, N = queries["uv"].shape[:2]

        if not self.training and N > self.infer_chunk_size:
            return self._forward_chunked(
                memory, feat_map, per_view_uv, num_views, rgb_images, queries, B, N)

        attn_bias = (self._compute_rel_pos_bias(queries["uv"], per_view_uv, num_views)
                     if per_view_uv is not None else None)
        query = self._build_query(rgb_images, queries, feat_map=feat_map)
        decoded = self._decode(query, memory, attn_bias=attn_bias)
        return self.output_head(decoded)

    def _forward_chunked(self, memory, feat_map, per_view_uv, num_views,
                         rgb_images, queries, B, N):
        C = self.infer_chunk_size
        all_pred = []
        for start in range(0, N, C):
            end = min(start + C, N)
            chunk_q = {
                k: (
                    v[:, start:end]
                    if torch.is_tensor(v) and v.ndim >= 2
                    and v.shape[0] == B and v.shape[1] == N
                    else v
                )
                for k, v in queries.items()
            }
            query = self._build_query(rgb_images, chunk_q, feat_map=feat_map)
            attn_bias = (self._compute_rel_pos_bias(chunk_q["uv"], per_view_uv, num_views)
                         if per_view_uv is not None else None)
            decoded = self._decode(query, memory, attn_bias=attn_bias)
            all_pred.append(self.output_head(decoded))
        return torch.cat(all_pred, dim=1)
