import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

from hAlgorithm.utils import instantiate_from_config

from hAlgorithm.modules.models2.query_aggregator.match_query_agg import PairQueryAggregator


class Pyramids(torch.nn.Module):
    """Build multi-scale pyramid features from token features.

    This module converts per-layer token features into dense feature maps and
    keeps the explicit view axis for downstream pair/single-view aggregation.
    """
    def __init__(
        self,
        patch_size=14,
        in_chans=[1024, 1024, 1024],
        embed_dims=[256, 512, 1024],
        upsample_scales=[4, 2, 1],
        interp_refinenet_cfg=None,
        intermediate_layer_idx=None,
        pretrain=None,
    ):
        super(Pyramids, self).__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.upsample_scales = upsample_scales
        self.intermediate_layer_idx = intermediate_layer_idx

        self.q_projs = nn.ModuleList()
        for ch0, ch1, s in zip(self.in_chans, self.embed_dims, self.upsample_scales):
            # NOTE: proj + upsample
            self.q_projs.append(nn.Linear(ch0, ch1 * s * s))

        if interp_refinenet_cfg is not None:
            self.interp_refinenet = nn.ModuleList()
            for i in range(len(self.in_chans)):
                interp_refinenet_cfg["input_channel"] = self.embed_dims[i]
                self.interp_refinenet.append(instantiate_from_config(interp_refinenet_cfg))
        else:
            self.interp_refinenet = None

        self.norms = nn.ModuleList()
        for dim_in in self.embed_dims:
            self.norms.append(nn.LayerNorm(dim_in))

        if interp_refinenet_cfg is not None:
            self.interp_norms = nn.ModuleList()
            for dim_in in self.embed_dims:
                self.interp_norms.append(nn.LayerNorm(dim_in))

        if pretrain is not None:
            res = self.load_state_dict(
                torch.load(pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"Pyramids, load pretrain {pretrain}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")

    def forward(self, x, meta_data, **kwargs):
        """Build pyramid maps with explicit batch-view dimensions.

        Args:
            x: list of per-layer token features
               - (B*N, L, C), or
               - (B, N, L, C)
            meta_data: dict containing `input_width`, `input_height`,
                `frames`, `views`

        Returns:
            pyramids: list of (B, N, C_i, H_i, W_i)
        """

        patch_w = meta_data["input_width"][0] // self.patch_size
        patch_h = meta_data["input_height"][0] // self.patch_size

        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        if self.intermediate_layer_idx is not None:
            x = [x[idx] for idx in self.intermediate_layer_idx]

        assert len(x) == len(self.in_chans)
        if x[0].ndim == 3:
            B = x[0].shape[0]
            N = frame_num * view_num
            B = B // N
        elif x[0].ndim == 4:
            B, N = x[0].shape[:2]
            assert N == frame_num * view_num
        else:
            raise NotImplementedError

        # Build pyramid features and keep view axis: (B, N, C, H, W)
        pyramids = []
        for i, feats in enumerate(x):
            spatial_h = patch_h * self.upsample_scales[i]
            spatial_w = patch_w * self.upsample_scales[i]

            feats = self.q_projs[i](feats.view(B * N, -1, feats.shape[-1]))
            feats = feats.transpose(-1, -2).view(B * N, -1, patch_h, patch_w)
            feats = F.pixel_shuffle(feats, self.upsample_scales[i])  # B,C,H,W
            feats = feats.permute(0, 2, 3, 1).contiguous().view(B * N, -1, feats.shape[1])

            feats = self.norms[i](feats)
            feats = feats.permute(0, 2, 1).contiguous().view(B * N, -1, spatial_h, spatial_w)

            if self.interp_refinenet is not None:
                feats = self.interp_refinenet[i](feats)
                feats = feats.permute(0, 2, 3, 1).contiguous().view(B * N, -1, feats.shape[1])
                feats = self.interp_norms[i](feats)
                feats = feats.permute(0, 2, 1).contiguous().view(B * N, -1, spatial_h, spatial_w)

            pyramids.append(feats.view(B, N, feats.shape[1], spatial_h, spatial_w))

        return pyramids


class Aggregator(torch.nn.Module):
    """Sample and fuse multi-scale pyramid features for query points.

    This module assumes pyramids are already built (typically by `Pyramids`)
    and performs query-wise sampling (`grid_sample`) followed by gated
    hierarchical fusion across scales.
    """
    def __init__(
        self,
        embed_dims=[256, 512, 1024],
        mode="bilinear",
        intermediate_layer_idx=None,
        pretrain=None,
        debug=False,
    ):
        super(Aggregator, self).__init__()

        self.mode = mode
        self.embed_dims = embed_dims
        self.intermediate_layer_idx = intermediate_layer_idx
        self.debug = debug

        self.h_projs = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for i in range(len(self.embed_dims) - 1):
            hidden_dim = self.embed_dims[i + 1]

            self.h_projs.append(nn.Linear(self.embed_dims[i], hidden_dim))

            self.gates.append(nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()))

            ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))
            self.ffns.append(ffn)

        self.fuse_norms = nn.ModuleList()
        for i in range(len(self.embed_dims) - 1):
            self.fuse_norms.append(nn.LayerNorm(self.embed_dims[i + 1]))

        if pretrain is not None:
            res = self.load_state_dict(
                torch.load(pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"Aggregator, load pretrain {pretrain}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")

    def forward(self, x, query, meta_data, query_rgb=None, **kwargs):
        """Aggregate sampled query features from pyramid maps.

        Args:
            x: list of pyramid features
               - preferred: (B, N, C, H, W)
               - compatible: (B*N, C, H, W), auto-reshaped by metadata
            query: query object with `uv`
               - (Q, 2), or
               - (B*N, Q, 2), or
               - (B, N, Q, 2)
            meta_data: dict containing `frames`, `views`

        Returns:
            hidden: (B*N, Q, hidden_dim)
        """
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        expected_n = frame_num * view_num

        if self.intermediate_layer_idx is not None:
            x = [x[idx] for idx in self.intermediate_layer_idx]

        if x[0].ndim == 5:
            B, N = x[0].shape[:2]
            assert N == expected_n
        elif x[0].ndim == 4:
            N = expected_n
            BN = x[0].shape[0]
            assert BN % N == 0
            B = BN // N
            x = [feat.view(B, N, *feat.shape[1:]) for feat in x]
        else:
            raise ValueError(f"Unsupported pyramid dimension: {x[0].ndim}")

        uv_grid = query.uv
        Q = uv_grid.shape[-2]
        if uv_grid.ndim == 4:
            uv_grid = uv_grid.view(B * N, Q, 2)

        if uv_grid.ndim == 2:
            uv_grid = uv_grid.unsqueeze(0).unsqueeze(2)  # (1, Q, 1, 2)
            uv_grid = uv_grid.expand(B * N, -1, -1, -1)  # (B*N, Q, 1, 2)
        elif uv_grid.ndim == 3:
            uv_grid = uv_grid.unsqueeze(2)  # (B*N, Q, 1, 2)
        else:
            raise ValueError(f"Unsupported uv_grid dimension: {uv_grid.ndim}")

        # step1: grid_sample
        sampled_feats_list = []
        for i, feats in enumerate(x):
            feats = feats.view(B * N, *feats.shape[2:])

            # (B*N, C, Q, 1)
            input_dtype = feats.dtype
            sampled_feats = F.grid_sample(feats.float(), uv_grid.float(), mode=self.mode, align_corners=False, padding_mode="border")
            sampled_feats = sampled_feats.to(input_dtype)
            # (B*N, Q, C)
            sampled_feats = sampled_feats.squeeze(-1).permute(0, 2, 1).contiguous()

            if self.debug:
                logging.info(f"{i}, sampled_feats: {sampled_feats.max()}, {sampled_feats.min()}")

            sampled_feats_list.append(sampled_feats)

        # Step 2: fuse
        hidden = sampled_feats_list[0]
        for k in range(1, len(sampled_feats_list)):
            hidden = self.h_projs[k - 1](hidden)

            if self.debug:
                logging.info(f"{k}, hidden1: {hidden.max()}, {hidden.min()}")

            gate_weights = self.gates[k - 1](torch.cat([hidden, sampled_feats_list[k]], dim=-1))

            hidden = gate_weights * hidden + (1 - gate_weights) * sampled_feats_list[k]
            hidden = self.fuse_norms[k - 1](hidden)

            if self.debug:
                logging.info(f"{k}, hidden2: {hidden.max()}, {hidden.min()}")

            hidden = self.ffns[k - 1](hidden)

            if self.debug:
                logging.info(f"{k}, hidden3: {hidden.max()}, {hidden.min()}")

        return hidden


def normalize(x: torch.Tensor, dim: int):
    return x / x.norm(dim=dim, keepdim=True).clamp(min=1e-8)


def cosine_similarity(f_A: torch.Tensor, f_B: torch.Tensor) -> torch.Tensor:
    """(B, H_A, W_A, D), (B, H_B, W_B, D) -> (B, H_A*W_A, H_B*W_B)"""
    f_A = normalize(f_A, dim=-1)
    f_B = normalize(f_B, dim=-1)
    return einsum(f_A, f_B, "B H_A W_A D, B H_B W_B D -> B H_A W_A H_B W_B")


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


class PairAggregator(Aggregator):
    """Pair-wise query aggregator on pyramid features.

    Input features are expected to come from `Pyramids`:
      - preferred: list of (B, N, C, H, W)
      - compatible: list of (B*N, C, H, W), reshaped internally by metadata

    For each (src, tgt) pair:
      1. Split src/tgt pyramids from view dimension.
      2. Sample src and tgt pyramids at query UV.
      3. Fuse each pyramid list with gated hierarchical fusion.
      4. Concatenate src/tgt query features.
      5. Optionally append match embedding and/or time embedding.

    Output: (B * num_pair, Q, hidden_dim_total)
    """

    def __init__(
        self,
        # Roma matching params
        match_dim=256,
        match_temp=0.1,
        match_scale=1.0,
        enable_amp=True,
        return_pyramids=False,
        with_match_embed=False,
        with_time_embed=False,
        time_fourier_dim=128,
        time_fourier_scale=5.0,
        padding_mode="border",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.return_pyramids = return_pyramids

        self.enable_amp = enable_amp

        self.with_match_embed = with_match_embed
        self.with_time_embed = with_time_embed

        if self.with_match_embed:
            omega = 2 * torch.pi * torch.randn(match_dim // 2, 2)
            self.register_buffer("omega", omega)
            self.register_buffer("scale", torch.tensor(match_scale))
            self.register_buffer("temp", torch.tensor(match_temp))

        if self.with_time_embed:
            self.time_encoder = _FourierEncoder(out_dim=time_fourier_dim, scale=time_fourier_scale)

        self.padding_mode = padding_mode

    def _get_pos_emb_grid(self, B, H, W, device):
        """Roma-style random Fourier position embedding on a normalized grid."""
        ys = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device)
        xs = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        x_emb = F.linear(grid.reshape(B, H * W, 2), self.scale * self.omega).reshape(B, H, W, -1)
        return torch.cat((x_emb.sin(), x_emb.cos()), dim=-1)  # (B, H, W, match_dim)

    def _compute_match_emb(self, f_src, f_tgt, pos_emb_grid):
        """Compute roma-style match embedding.

        Args:
            f_src: (B, H, W, C) source view features
            f_tgt: (B, H, W, C) target view features
            pos_emb_grid: (B, H, W, match_dim) position embedding on target grid

        Returns:
            match_emb: (B, H, W, match_dim) weighted position embedding
        """
        B, H_A, W_A, _ = f_src.shape
        B, H_B, W_B, _ = f_tgt.shape

        if self.enable_amp:
            with torch.autocast(f_src.device.type, torch.bfloat16, enabled=True):
                attn_logits = (1.0 / self.temp * cosine_similarity(f_src, f_tgt)).reshape(B, H_A * W_A, H_B * W_B)
                attn = torch.softmax(attn_logits, dim=2).reshape(B, H_A, W_A, H_B, W_B)
            attn = attn.float()
        else:
            attn_logits = (1.0 / self.temp * cosine_similarity(f_src, f_tgt)).reshape(B, H_A * W_A, H_B * W_B)
            attn = torch.softmax(attn_logits, dim=2).reshape(B, H_A, W_A, H_B, W_B)

        match_emb = einsum(
            attn,
            pos_emb_grid,
            "B H_A W_A H_B W_B, B H_B W_B D -> B H_A W_A D",
        )
        return match_emb

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

    def _sample_and_fuse(self, layer_feats, uv_grid):
        """Sample query features from pyramid maps, then fuse hierarchically.

        Args:
            layer_feats: list of (B_pair, C, H, W) pyramid features
            uv_grid: (B_pair, Q, 1, 2) query grid in [-1, 1]
        Returns:
            hidden: (B_pair, Q, hidden_dim)
        """
        sampled_feats_list = []
        for feats in layer_feats:
            input_dtype = feats.dtype
            sampled = F.grid_sample(feats.float(), uv_grid.float(), mode=self.mode, align_corners=False, padding_mode=self.padding_mode)
            sampled = sampled.to(input_dtype)
            sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B_pair, Q, embed_dim)

            sampled_feats_list.append(sampled)

        hidden = sampled_feats_list[0]
        for k in range(1, len(sampled_feats_list)):
            hidden = self.h_projs[k - 1](hidden)

            gate_weights = self.gates[k - 1](torch.cat([hidden, sampled_feats_list[k]], dim=-1))

            hidden = gate_weights * hidden + (1 - gate_weights) * sampled_feats_list[k]
            hidden = self.fuse_norms[k - 1](hidden)

            hidden = self.ffns[k - 1](hidden)

        return hidden

    def forward(self, x, query, meta_data, pair_idx=None, **kwargs):
        """
        Args:
            x: list of pyramid features from `Pyramids`
               - (B, N, C, H, W), or
               - (B*N, C, H, W) (auto-reshaped using metadata)
            query: BaseQuery with uv (Q, 2) in [0, 1]
            meta_data: dict with input_width, input_height, frames, views
            pair_idx: list of (src, tgt) tuples, default [(0, 1)]

        Returns:
            feats: (B * num_pair, Q, hidden_dim)
        """
        if pair_idx is None:
            pair_idx = [(0, 1)]
        num_pair = len(pair_idx)

        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        expected_n = frame_num * view_num

        if self.intermediate_layer_idx is not None:
            x = [x[idx] for idx in self.intermediate_layer_idx]

        if x[0].ndim == 5:
            B, N = x[0].shape[:2]
            assert N == expected_n
        elif x[0].ndim == 4:
            N = expected_n
            BN = x[0].shape[0]
            assert BN % N == 0
            B = BN // N
            x = [feat.view(B, N, *feat.shape[1:]) for feat in x]
        else:
            raise ValueError(f"Unsupported pyramid dimension: {x[0].ndim}")

        src_pyramids = []
        tgt_pyramids = []
        for feats in x:
            src_pyramid = torch.stack([feats[:, src] for src, tgt in pair_idx], dim=1)
            tgt_pyramid = torch.stack([feats[:, tgt] for src, tgt in pair_idx], dim=1)

            src_pyramids.append(src_pyramid.view(B * num_pair, *src_pyramid.shape[2:]))
            tgt_pyramids.append(tgt_pyramid.view(B * num_pair, *tgt_pyramid.shape[2:]))

        # Build query UV grid
        uv_grid = query.uv
        uv_grid = uv_grid * 2 - 1

        if uv_grid.ndim == 4:
            # (B*P, Q, 1, 2)
            uv_grid = torch.stack([uv_grid[:, pair[0]] for pair in pair_idx], dim=1).view(B * num_pair, *uv_grid.shape[-2:]).unsqueeze(2)
        elif uv_grid.ndim == 2:
            uv_grid = uv_grid.unsqueeze(0).unsqueeze(2)  # (1, Q, 1, 2)
            uv_grid = uv_grid.expand(B * num_pair, -1, -1, -1)  # (B*P, Q, 1, 2)
        else:
            raise ValueError(f"Unsupported uv_grid dimension: {uv_grid.ndim}")

        src_hidden = self._sample_and_fuse(src_pyramids, uv_grid)
        tgt_hidden = self._sample_and_fuse(tgt_pyramids, uv_grid)
        hidden = [src_hidden, tgt_hidden]

        if self.with_match_embed:
            src_spatial = src_pyramids[-1].permute(0, 2, 3, 1).contiguous()
            tgt_spatial = tgt_pyramids[-1].permute(0, 2, 3, 1).contiguous()
            spatial_h, spatial_w = src_pyramids[-1].shape[2], src_pyramids[-1].shape[3]

            pos_emb_grid = self._get_pos_emb_grid(B * num_pair, spatial_h, spatial_w, device=src_pyramids[-1].device)
            match_emb = self._compute_match_emb(src_spatial, tgt_spatial, pos_emb_grid)

            match_emb = match_emb.permute(0, 3, 1, 2).contiguous()
            match_emb = F.grid_sample(match_emb.float(), uv_grid.float(), mode=self.mode, align_corners=False, padding_mode=self.padding_mode)
            match_emb = match_emb.squeeze(-1).permute(0, 2, 1).contiguous()
            hidden.append(match_emb)

        if self.with_time_embed:
            tgt_time = uv_grid.new_zeros(B, num_pair)
            for i, (_, tgt) in enumerate(pair_idx):
                tgt_time[:, i] = tgt / N

            Q = uv_grid.shape[-3]
            time_emb = self._encode_time(
                tgt_time=tgt_time,
                B=B,
                Q=Q,
                device=uv_grid.device,
                dtype=uv_grid.dtype,
            )
            time_emb = time_emb.view(B * num_pair, 1, time_emb.shape[-1]).expand(-1, Q, -1)
            hidden.append(time_emb)

        hidden = torch.cat(hidden, dim=-1)

        if self.return_pyramids:
            return hidden, [src_pyramids, tgt_pyramids]
        else:
            return hidden  # (B * num_pair, Q, hidden_dim)



class _CrossAttnBlock(nn.Module):
    """Pre-LN cross-attention + FFN residual block.

    This is intentionally minimal and self-contained so it can be reused as a drop-in
    building block without pulling in any motion-branch specific dependencies.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
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
        qn = self.q_norm(q)
        kn = self.k_norm(mem)
        attn_out, _ = self.cross(qn, kn, kn, need_weights=False)
        x = q + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class _CrossAttention(nn.Module):
    """Multi-head cross-attention via fused SDPA.

    Math mirrors `nn.MultiheadAttention(batch_first=True)` for the cross-attn
    case (Q from `q`, K/V from `mem`), but uses
    `F.scaled_dot_product_attention` so Flash / mem-efficient kernels can kick
    in. Parameter layout differs from MHA: separate `q_proj` and a fused
    `kv_proj` instead of MHA's single fused `in_proj_weight`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        B, Q, C = q.shape
        _, P, _ = mem.shape

        q_h = (
            self.q_proj(q)
            .reshape(B, Q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, H, Q, D_h]
        kv = (
            self.kv_proj(mem)
            .reshape(B, P, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # [2, B, H, P, D_h]
        k, v = kv[0], kv[1]

        x = F.scaled_dot_product_attention(
            q_h, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, Q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _CrossAttnBlock2(nn.Module):
    """Pre-LN cross-attention + FFN residual block, SDPA backend.

    Functionally equivalent to `_CrossAttnBlock`: same Pre-LN structure,
    same residual topology, same attention math (scaled dot-product with
    attention-weight dropout, output projection, no proj-drop after MHA),
    same FFN. The only difference is the attention backend
    (`F.scaled_dot_product_attention` instead of `nn.MultiheadAttention`),
    which enables Flash / memory-efficient kernels.

    Caveats vs `_CrossAttnBlock`:
        - Parameter layout differs (separate `q_proj` + fused `kv_proj` vs
          MHA's fused `in_proj_weight`), so a state-dict from
          `_CrossAttnBlock` is NOT directly loadable here without remapping.
        - Default initializations differ slightly (`nn.Linear` kaiming-uniform
          vs MHA's xavier-uniform on `in_proj_weight`); fresh modules will
          therefore not be bit-identical even with the same seed, but
          training behavior is equivalent.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)
        self.cross = _CrossAttention(
            dim=d_model,
            num_heads=n_heads,
            qkv_bias=True,
            proj_bias=True,
            attn_drop=dropout,
            proj_drop=0.0,
        )
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
        qn = self.q_norm(q)
        kn = self.k_norm(mem)
        attn_out = self.cross(qn, kn)
        x = q + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class PairCrossAttnAggregator(torch.nn.Module):
    """Pair-wise cross-attention aggregator compatible with MVQuery4.pair_forward.

    Design goal: keep the *exact* calling convention and pyramid/UV sampling logic
    of `PairQueryAggregator`, but replace the Roma-style `match_emb` with a learned
    cross-attention readout over the target view's spatial tokens.

    Output layout intentionally matches the original (3 * 256 by default):
      [src_hidden, tgt_hidden, cross_out]
    so existing pair decoders configured with `in_chan=256*3` remain valid.

    Cross-attention memory keys default to the last pyramid scale on the target view
    (`tgt_pyramids[-1]`). Set `memory_from_encoder_last=True` to use the last entry in
    the input list `x[-1]` (after `intermediate_layer_idx` filtering if any), reshaped to
    the patch grid `patch_h x patch_w`, as memory instead.
    """

    def __init__(
        self,
        patch_size=14,
        in_chans=[1024, 1024, 1024],
        embed_dims=[256, 512, 1024],
        upsample_scales=[4, 2, 1],
        mode="bilinear",
        padding_mode="border",
        interp_refinenet_cfg=None,
        intermediate_layer_idx=None,

        cross_attn_dim: int = 256,
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 4,
        cross_ff_ratio: int = 4,
        cross_dropout: float = 0.0,

        memory_from_encoder_last: bool = False,
        without_src_hidden=False,
        without_tgt_hidden=False,

        with_cross_sdpa=False,
        align_corners=False,

        return_pyramids=False,
        return_src_hidden=False,

        pretrain=None,
        debug=False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.mode = mode
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.upsample_scales = upsample_scales
        self.intermediate_layer_idx = intermediate_layer_idx
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.debug = debug

        self.without_src_hidden = without_src_hidden
        self.without_tgt_hidden = without_tgt_hidden

        self.return_pyramids = return_pyramids
        self.return_src_hidden = return_src_hidden

        self.q_projs = nn.ModuleList()
        for ch0, ch1, s in zip(self.in_chans, self.embed_dims, self.upsample_scales):
            # NOTE: proj + upsample
            self.q_projs.append(nn.Linear(ch0, ch1 * s * s))

        self.interp_refinenet_cfg = interp_refinenet_cfg
        if self.interp_refinenet_cfg is not None:
            self.interp_refinenet = nn.ModuleList()
            for i in range(len(self.in_chans)):
                self.interp_refinenet_cfg["input_channel"] = self.embed_dims[i]
                self.interp_refinenet.append(instantiate_from_config(self.interp_refinenet_cfg))
        else:
            self.interp_refinenet = None

        self.h_projs = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for i in range(len(self.embed_dims) - 1):
            hidden_dim = self.embed_dims[i + 1]

            self.h_projs.append(nn.Linear(self.embed_dims[i], hidden_dim))

            self.gates.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Sigmoid()
                )
            )

            ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))
            self.ffns.append(ffn)

        self.norms = nn.ModuleList()
        for dim_in in self.embed_dims:
            self.norms.append(nn.LayerNorm(dim_in))

        if self.interp_refinenet_cfg is not None:
            self.interp_norms = nn.ModuleList()
            for dim_in in self.embed_dims:
                self.interp_norms.append(nn.LayerNorm(dim_in))

        self.fuse_norms = nn.ModuleList()
        for i in range(len(self.embed_dims) - 1):
            self.fuse_norms.append(nn.LayerNorm(self.embed_dims[i + 1]))


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

        self.memory_from_encoder_last = bool(memory_from_encoder_last)
        enc_c = int(self.in_chans[-1])
        emb_c = int(self.embed_dims[-1])
        if self.memory_from_encoder_last and enc_c != emb_c:
            self.encoder_mem_adapt = nn.Linear(enc_c, emb_c)
        else:
            self.encoder_mem_adapt = None

        # PairQueryAggregator builds fused 256-D features per pyramid scale by default.
        # We treat the final fused embeddings as both query & memory feature sources.
        in_dim = int(self.embed_dims[-1])
        dim_ff = int(self.cross_attn_dim * self.cross_ff_ratio)

        self.query_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.cross_attn_dim),
            nn.GELU(),
        )
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.cross_attn_dim),
            nn.GELU(),
        )

        self.with_cross_sdpa = with_cross_sdpa
        if self.with_cross_sdpa:
            self.cross_blocks = nn.ModuleList(
                [
                    _CrossAttnBlock2(
                        d_model=self.cross_attn_dim,
                        n_heads=self.cross_attn_heads,
                        dim_ff=dim_ff,
                        dropout=self.cross_dropout,
                    )
                    for _ in range(self.cross_attn_layers)
                ]
            )
        else:
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

        self.pretrain = pretrain
        if pretrain is not None:
            res = self.load_state_dict(
                torch.load(pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"PairCrossAttnAggregator, load pretrain {pretrain}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")

    def _sample_and_fuse(self, layer_feats, uv_grid, patch_h, patch_w):
        """Project, pixel_shuffle, norm, grid_sample, then hierarchical fusion.

        Replicates Aggregator23's per-layer processing exactly:
        q_proj -> pixel_shuffle -> norm -> [interp_refinenet -> interp_norm] -> grid_sample,
        then fuse with fuse_norms.

        Args:
            layer_feats: list of (B_pair, L, C) per-layer features
            uv_grid: (B_pair, Q, 1, 2) query grid in [-1, 1]
            patch_h, patch_w: spatial dimensions of patch features

        Returns:
            hidden: (B_pair, Q, hidden_dim)
        """
        sampled_feats_list = []
        B_pair = layer_feats[0].shape[0]

        for i, feats in enumerate(layer_feats):
            input_dtype = feats.dtype
            sampled = F.grid_sample(feats.float(), uv_grid.float(), mode=self.mode, align_corners=self.align_corners, padding_mode=self.padding_mode)
            sampled = sampled.to(input_dtype)
            sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B_pair, Q, embed_dim)

            sampled_feats_list.append(sampled)

        hidden = sampled_feats_list[0]
        for k in range(1, len(sampled_feats_list)):
            hidden = self.h_projs[k - 1](hidden)

            gate_weights = self.gates[k - 1](torch.cat([hidden, sampled_feats_list[k]], dim=-1))

            hidden = gate_weights * hidden + (1 - gate_weights) * sampled_feats_list[k]
            hidden = self.fuse_norms[k - 1](hidden)

            hidden = self.ffns[k - 1](hidden)

        return hidden

    def _apply_pair_pyramid_fusion(
        self,
        src_pyramids,
        tgt_pyramids,
        pair_idx=None,
        meta_data=None,
        B: int = 1,
        N: int = 1,
        num_pair: int = 1,
        rgb=None,
        **kwargs,
    ):
        """Pair-pyramid fusion hook; default is identity (no-op)."""
        return src_pyramids, tgt_pyramids

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

        src_pyramids, tgt_pyramids = self._apply_pair_pyramid_fusion(
            src_pyramids,
            tgt_pyramids,
            pair_idx=pair_idx,
            meta_data=meta_data,
            B=B,
            N=N,
            num_pair=num_pair,
            rgb=kwargs.get("rgb"),
        )

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

        if not self.without_tgt_hidden:
            tgt_hidden = self._sample_and_fuse(tgt_pyramids, uv_grid, patch_h, patch_w)

        if self.memory_from_encoder_last:
            enc_last = x[-1]  # [B, N, L, C_enc], L == patch_h * patch_w
            L_tok = enc_last.shape[2]
            if L_tok != patch_h * patch_w:
                raise ValueError(
                    f"memory_from_encoder_last: token len {L_tok} != patch_h*patch_w "
                    f"({patch_h}*{patch_w})"
                )
            mem_map = enc_last.view(B, N, patch_h, patch_w, -1).contiguous()
            mem_map = torch.stack([mem_map[:, tgt] for _, tgt in pair_idx], dim=1)
            # [B, P, patch_h, patch_w, C_enc], P == num_pair
            mem_map = mem_map.view(B * num_pair, *mem_map.shape[2:]).contiguous()
            # [B*P, patch_h, patch_w, C_enc]
            if self.encoder_mem_adapt is not None:
                mem_map = self.encoder_mem_adapt(mem_map)  # [B*P, patch_h, patch_w, C_emb]

            mem_map = mem_map.view(B * num_pair, -1, mem_map.shape[-1]).contiguous()  # [B*P, H*W, C_emb]

        else:
            mem_map = tgt_pyramids[-1]  # [B*P, C_emb, H, W]
            mem_map = mem_map.flatten(2).transpose(1, 2).contiguous()  # [B*P, H*W, C_emb]

        query_tokens = self.query_proj(src_hidden)  # [B*P, Q, D]
        memory_tokens = self.memory_proj(mem_map)  # [B*P, H*W, D]
        for blk in self.cross_blocks:
            query_tokens = blk(query_tokens, memory_tokens)
        cross_readout = self.readout(query_tokens)  # [B*P, Q, C]

        if self.without_tgt_hidden:
            if self.without_src_hidden:
                hidden = cross_readout
            else:
                hidden = torch.cat([src_hidden, cross_readout], dim=-1)
        else:
            if self.without_src_hidden:
                hidden = torch.cat([tgt_hidden, cross_readout], dim=-1)
            else:
                hidden = torch.cat([src_hidden, tgt_hidden, cross_readout], dim=-1)

        if getattr(self, "return_pyramids", False):
            return hidden, [src_pyramids, tgt_pyramids]
        if getattr(self, "return_src_hidden", False):
            return hidden, src_hidden
        return hidden
