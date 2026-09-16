from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models2.query_aggregator.match_query_agg import PairQueryAggregator


class _CrossAttnBlock(nn.Module):
    """Pre-LN cross-attention + FFN residual block.

    This is intentionally minimal and self-contained so it can be reused as a drop-in
    building block without pulling in any motion-branch specific dependencies.

    Uses ``F.scaled_dot_product_attention`` for Flash Attention / memory-efficient
    attention support, replacing the heavier ``nn.MultiheadAttention`` wrapper.

    QK-norm (``q_norm`` / ``k_norm`` without V-norm) follows the ViT-22B
    convention for stable deep-attention training.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # QK-norm (ViT-22B convention: normalize Q & K, not V)
        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)

        # Explicit Q/K/V/O projections (equivalent to MHA but enables SDPA)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop_p = dropout

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention from q to mem.

        Args:
            q: [B, Q, D] query tokens
            mem: [B, P, D] memory tokens

        Returns:
            Updated query tokens with the same shape as q.
        """
        B, Q_len, D = q.shape
        P_len = mem.shape[1]

        qn = self.q_norm(q)
        kn = self.k_norm(mem)

        Q_ = self.q_proj(qn).view(B, Q_len, self.n_heads, self.head_dim).transpose(1, 2)
        K_ = self.k_proj(kn).view(B, P_len, self.n_heads, self.head_dim).transpose(1, 2)
        V_ = self.v_proj(mem).view(B, P_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            Q_, K_, V_,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, Q_len, D)
        attn_out = self.out_proj(attn_out)

        x = q + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class PairCrossAttnAggregator(PairQueryAggregator):
    """Pair-wise cross-attention aggregator compatible with MVQuery4.pair_forward.

    Design goal: keep the *exact* calling convention and pyramid/UV layout of
    `PairQueryAggregator`, but replace the Roma-style `match_emb` with a learned
    cross-attention readout over the target view's spatial tokens. Query sampling
    uses ``align_corners=True`` (see :meth:`_sample_and_fuse`) to match ``QueryBank5``
    UV normalisation; :meth:`_get_pos_emb_grid` uses the same ``[-1, 1]`` corner grid.
    The parent class keeps ``align_corners=False`` and pixel-centred position grids.

    Output layout (see ``pair_decoder_input``):
      - ``concat`` (default): ``[src_hidden, tgt_hidden, cross_out]`` → width ``3 * embed_dim``.
      - ``cross_only``: ``cross_out`` only → width ``embed_dim`` (cross-attention readout).
    """

    def __init__(
        self,
        cross_attn_dim: int = 256,
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 4,
        cross_ff_ratio: int = 4,
        cross_dropout: float = 0.0,
        pair_decoder_input: str = "concat",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if pair_decoder_input not in ("concat", "cross_only"):
            raise ValueError(
                "pair_decoder_input must be 'concat' or 'cross_only', "
                f"got {pair_decoder_input!r}"
            )
        self.pair_decoder_input = pair_decoder_input

        if cross_attn_dim <= 0:
            raise ValueError(f"cross_attn_dim must be positive, got {cross_attn_dim}")
        if cross_attn_heads <= 0:
            raise ValueError(f"cross_attn_heads must be positive, got {cross_attn_heads}")
        if cross_attn_layers <= 0:
            raise ValueError(f"cross_attn_layers must be positive, got {cross_attn_layers}")
        if cross_attn_dim % cross_attn_heads != 0:
            raise ValueError(
                f"cross_attn_dim ({cross_attn_dim}) must be divisible by "
                f"cross_attn_heads ({cross_attn_heads})"
            )

        self.cross_attn_dim = int(cross_attn_dim)
        self.cross_attn_heads = int(cross_attn_heads)
        self.cross_attn_layers = int(cross_attn_layers)
        self.cross_ff_ratio = int(cross_ff_ratio)
        self.cross_dropout = float(cross_dropout)

        # PairQueryAggregator builds fused 256-D features per pyramid scale by default.
        # We treat the final fused embeddings as both query & memory feature sources.
        in_dim = int(self.embed_dims[-1])
        dim_ff = int(self.cross_attn_dim * self.cross_ff_ratio)

        self.query_in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.cross_attn_dim),
            nn.GELU(),
        )
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.cross_attn_dim),
            nn.GELU(),
        )
        self.cross_blocks = nn.ModuleList(
            [
                _CrossAttnBlock(
                    d_model=self.cross_attn_dim,
                    n_heads=self.cross_attn_heads,
                    dim_ff=dim_ff,
                    dropout=self.cross_dropout,
                )
                for _ in range(self.cross_attn_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(self.cross_attn_dim),
            nn.Linear(self.cross_attn_dim, in_dim),
            nn.GELU(),
        )

    def _sample_and_fuse(self, layer_feats, uv_grid, patch_h, patch_w):
        """Same fusion path as ``PairQueryAggregator`` but ``align_corners=True`` for ``grid_sample``.

        ``QueryBank5`` normalises UV with ``/(W-1),/(H-1)`` into ``[0,1]``; that corner convention
        matches PyTorch ``grid_sample(..., align_corners=True)`` when the grid is ``2*uv-1``.

        Args:
            layer_feats: Per-scale feature maps ``(B_pair, C, H_i, W_i)`` (already projected).
            uv_grid: ``(B_pair, Q, 1, 2)`` in ``[-1, 1]``.
            patch_h, patch_w: Patch grid size before upsample (unused; kept for signature parity).

        Returns:
            ``(B_pair, Q, hidden_dim)`` fused query features.
        """
        sampled_feats_list = []
        B_pair = layer_feats[0].shape[0]

        for i, feats in enumerate(layer_feats):
            input_dtype = feats.dtype
            sampled = F.grid_sample(
                feats.float(),
                uv_grid.float(),
                mode=self.mode,
                align_corners=True,
                padding_mode=self.padding_mode,
            )
            sampled = sampled.to(input_dtype)
            sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()

            sampled_feats_list.append(sampled)

        hidden = sampled_feats_list[0]
        for k in range(1, len(sampled_feats_list)):
            hidden = self.h_projs[k - 1](hidden)

            gate_weights = self.gates[k - 1](torch.cat([hidden, sampled_feats_list[k]], dim=-1))

            hidden = gate_weights * hidden + (1 - gate_weights) * sampled_feats_list[k]
            hidden = self.fuse_norms[k - 1](hidden)

            hidden = self.ffns[k - 1](hidden)

        return hidden

    def _get_pos_emb_grid(self, B, H, W, device):
        """Roma-style random Fourier grid aligned with ``grid_sample(..., align_corners=True)``.

        Parent ``PairQueryAggregator`` uses pixel-centred coordinates ``[-1+1/H, 1-1/H]``, which
        pairs with ``align_corners=False``. This subclass uses corner-aligned ``[-1, 1]``
        ``linspace`` so spatial phase matches :meth:`_sample_and_fuse` and ``QueryBank5`` UV.

        Requires ``with_match_embed=True`` (buffers ``omega`` / ``scale``).

        Args:
            B: Batch size for the expanded grid.
            H, W: Spatial size of the feature map.
            device: Torch device for the grid.

        Returns:
            Tensor of shape ``(B, H, W, match_dim)``.
        """
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

        x_emb = F.linear(grid.reshape(B, H * W, 2), self.scale * self.omega).reshape(B, H, W, -1)
        return torch.cat((x_emb.sin(), x_emb.cos()), dim=-1)

    def forward(self, x, query, meta_data, pair_idx=None, **kwargs):
        if pair_idx is None:
            pair_idx = [(0, 1)]
        num_pair = len(pair_idx)

        patch_w = meta_data["input_width"][0] // self.patch_size
        patch_h = meta_data["input_height"][0] // self.patch_size

        if self.intermediate_layer_idx is not None:
            x = [x[idx] for idx in self.intermediate_layer_idx]
        assert len(x) == len(self.in_chans)

        if x[0].ndim == 3:
            frame_num = meta_data["frames"][0]
            view_num = meta_data["views"][0]
            N = frame_num * view_num
            B = x[0].shape[0] // N
            x = [feat.view(B, N, -1, feat.shape[-1]) for feat in x]
        else:
            B, N = x[0].shape[:2]

        src_pyramids = []
        tgt_pyramids = []
        for i, feats in enumerate(x):
            feats = self.q_projs[i](feats)  # (B, N, L, embed_dim * s^2)
            feats = feats.transpose(-1, -2).view(B * N, -1, patch_h, patch_w)
            feats = F.pixel_shuffle(feats, self.upsample_scales[i])  # (B*N, C, H, W)

            spatial_h = patch_h * self.upsample_scales[i]
            spatial_w = patch_w * self.upsample_scales[i]

            feats = feats.permute(0, 2, 3, 1).contiguous().view(B * N, -1, feats.shape[1])
            feats = self.norms[i](feats)
            feats = feats.permute(0, 2, 1).contiguous().view(B * N, -1, spatial_h, spatial_w)

            if self.interp_refinenet is not None:
                feats = self.interp_refinenet[i](feats)
                feats = feats.permute(0, 2, 3, 1).contiguous().view(B * N, -1, feats.shape[1])
                feats = self.interp_norms[i](feats)
                feats = feats.permute(0, 2, 1).contiguous().view(B * N, -1, spatial_h, spatial_w)

            feats_bnchw = feats.view(B, N, -1, spatial_h, spatial_w)
            src_pyramid = torch.stack([feats_bnchw[:, src] for src, _ in pair_idx], dim=1)
            tgt_pyramid = torch.stack([feats_bnchw[:, tgt] for _, tgt in pair_idx], dim=1)
            src_pyramids.append(src_pyramid.view(B * num_pair, *src_pyramid.shape[2:]))
            tgt_pyramids.append(tgt_pyramid.view(B * num_pair, *tgt_pyramid.shape[2:]))

        uv_grid = query.uv
        uv_grid = uv_grid * 2 - 1
        if uv_grid.ndim == 4:
            uv_grid = (
                torch.stack([uv_grid[:, pair[0]] for pair in pair_idx], dim=1)
                .view(B * num_pair, *uv_grid.shape[-2:])
                .unsqueeze(2)
            )
        elif uv_grid.ndim == 2:
            uv_grid = uv_grid.unsqueeze(0).unsqueeze(2).expand(B * num_pair, -1, -1, -1)
        else:
            raise ValueError(f"Unsupported uv_grid dimension: {uv_grid.ndim}")

        src_hidden = self._sample_and_fuse(src_pyramids, uv_grid, patch_h, patch_w)
        tgt_hidden = self._sample_and_fuse(tgt_pyramids, uv_grid, patch_h, patch_w)

        mem_map = tgt_pyramids[-1]  # [B*P, C, H, W]
        mem = mem_map.flatten(2).transpose(1, 2).contiguous()  # [B*P, H*W, C]

        q = self.query_in_proj(src_hidden)  # [B*P, Q, D]
        mem_proj = self.memory_proj(mem)  # [B*P, H*W, D]
        for blk in self.cross_blocks:
            q = blk(q, mem_proj)
        cross_out = self.readout(q)  # [B*P, Q, C]

        if self.pair_decoder_input == "cross_only":
            hidden = cross_out
        else:
            hidden = torch.cat([src_hidden, tgt_hidden, cross_out], dim=-1)

        if getattr(self, "return_dense_feats", False):
            return hidden, None
        if getattr(self, "return_pyramids", False):
            return hidden, [src_pyramids, tgt_pyramids]
        return hidden

