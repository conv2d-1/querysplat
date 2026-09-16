import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

from hAlgorithm.modules.models2.query_aggregator.infinidepth import Aggregator23, Aggregator24


def normalize(x: torch.Tensor, dim: int):
    return x / x.norm(dim=dim, keepdim=True).clamp(min=1e-8)


def cosine_similarity(f_A: torch.Tensor, f_B: torch.Tensor) -> torch.Tensor:
    """(B, H_A, W_A, D), (B, H_B, W_B, D) -> (B, H_A*W_A, H_B*W_B)"""
    f_A = normalize(f_A, dim=-1)
    f_B = normalize(f_B, dim=-1)
    return einsum(f_A, f_B, "B H_A W_A D, B H_B W_B D -> B H_A W_A H_B W_B")


class MatchQueryAggregator(Aggregator23):
    """Query aggregator with roma-style cross-view matching.

    Inherits Aggregator23's network structure (q_projs, norms, interp_refinenet,
    interp_norms, h_projs, gates, fuse_norms, ffns) and adds cross-view matching.

    For each pair (src, tgt):
      1. Reshape ViT features to spatial (B, H, W, C)
      2. Compute cosine_sim(f_src, f_tgt) -> softmax -> weighted pos_emb -> match_emb
      3. Add match_emb to source view features at match_layer
      4. For all layers: q_proj -> pixel_shuffle -> norm -> [refinenet -> norm] -> grid_sample
      5. Hierarchical fusion with fuse_norms

    Output: (B * num_pair, Q, hidden_dim)
    """

    def __init__(
        self,
        # Roma matching params
        match_layer_idx=-1,
        match_dim=None,
        match_temp=0.1,
        match_scale=1.0,
        enable_amp=True,
        padding_mode="border",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.uv_round = False

        self.match_layer_idx = match_layer_idx
        self.enable_amp = enable_amp

        resolved_idx = match_layer_idx if match_layer_idx >= 0 else len(self.in_chans) + match_layer_idx
        self.resolved_match_layer_idx = resolved_idx

        if match_dim is None:
            match_dim = self.in_chans[resolved_idx]
        self.match_dim = match_dim

        omega = 2 * torch.pi * torch.randn(match_dim // 2, 2)
        self.register_buffer("omega", omega)
        self.register_buffer("scale", torch.tensor(match_scale))
        self.register_buffer("temp", torch.tensor(match_temp))

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
            feats = self.q_projs[i](feats)  # (B_pair, L, embed_dim * s^2)
            feats = feats.transpose(-1, -2).view(B_pair, -1, patch_h, patch_w)
            feats = F.pixel_shuffle(feats, self.upsample_scales[i])  # B,C,H,W

            spatial_h = patch_h * self.upsample_scales[i]
            spatial_w = patch_w * self.upsample_scales[i]

            feats = feats.permute(0, 2, 3, 1).contiguous().view(B_pair, -1, feats.shape[1])
            feats = self.norms[i](feats)
            feats = feats.permute(0, 2, 1).contiguous().view(B_pair, -1, spatial_h, spatial_w)

            if self.interp_refinenet is not None:
                feats = self.interp_refinenet[i](feats)

                feats = feats.permute(0, 2, 3, 1).contiguous().view(B_pair, -1, feats.shape[1])
                feats = self.interp_norms[i](feats)
                feats = feats.permute(0, 2, 1).contiguous().view(B_pair, -1, spatial_h, spatial_w)

            input_dtype = feats.dtype
            sampled = F.grid_sample(feats.float(), uv_grid.float(), mode=self.mode, align_corners=False, padding_mode=self.padding_mode)
            sampled = sampled.to(input_dtype)
            sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B_pair, Q, embed_dim)

            if self.debug:
                logging.info(f"{i}, sampled_feats: {sampled.max()}, {sampled.min()}")

            sampled_feats_list.append(sampled)

        hidden = sampled_feats_list[0]
        for k in range(1, len(sampled_feats_list)):
            hidden = self.h_projs[k - 1](hidden)

            if self.debug:
                logging.info(f"{k}, hidden1: {hidden.max()}, {hidden.min()}")

            gate = torch.sigmoid(self.gates[k - 1]).view(1, 1, -1)
            hidden = sampled_feats_list[k] + gate * hidden
            hidden = self.fuse_norms[k - 1](hidden)

            if self.debug:
                logging.info(f"{k}, hidden2: {hidden.max()}, {hidden.min()}")

            hidden = self.ffns[k - 1](hidden)

            if self.debug:
                logging.info(f"{k}, hidden3: {hidden.max()}, {hidden.min()}")

        return hidden

    def forward(self, x, query, meta_data, pair_idx=None, **kwargs):
        """
        Args:
            x: list of (B, N, L, C) multi-layer ViT patch features
            query: BaseQuery with uv (Q, 2) in [0, 1]
            meta_data: dict with input_width, input_height, frames, views
            pair_idx: list of (src, tgt) tuples, default [(0, 1)]

        Returns:
            feats: (B * num_pair, Q, hidden_dim)
        """
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

        # Build query UV grid
        uv_grid = query.uv
        uv_grid = uv_grid * 2 - 1
        uv_grid = uv_grid.unsqueeze(0).unsqueeze(2)  # (1, Q, 1, 2)
        uv_grid = uv_grid.expand(B * num_pair, -1, -1, -1)  # (B*P, Q, 1, 2)

        # Position embedding on target view grid
        pos_emb_grid = self._get_pos_emb_grid(B, patch_h, patch_w, device=x[0].device)

        # Cross-view matching features (spatial format)
        match_spatial = x[self.match_layer_idx].view(B, N, patch_h, patch_w, -1)  # (B, N, L, C)

        # Per-pair: compute match_emb and build enhanced source features
        pair_layer_feats = []  # list of num_pair, each is list of num_layers x (B, L, C)
        for src, tgt in pair_idx:
            f_src_spatial = match_spatial[:, src]  # (B, H, W, C)
            f_tgt_spatial = match_spatial[:, tgt]  # (B, H, W, C)

            match_emb = self._compute_match_emb(f_src_spatial, f_tgt_spatial, pos_emb_grid)

            combined = (f_src_spatial + match_emb).reshape(B, patch_h * patch_w, -1)  # (B, L, C)
            pair_layer_feats.append(combined)

        # Stack all pairs: (B, num_pair, L, C) -> (B * num_pair, L, C)
        stacked = torch.stack(pair_layer_feats, dim=1).view(B * num_pair, -1, pair_layer_feats[0].shape[-1])

        hidden = self._sample_and_fuse([stacked], uv_grid, patch_h, patch_w)
        return hidden  # (B * num_pair, Q, hidden_dim)


class MatchQueryAggregator3(MatchQueryAggregator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, query, meta_data, pair_idx=None, **kwargs):
        """
        Args:
            x: list of (B, N, L, C) multi-layer ViT patch features
            query: BaseQuery with uv (Q, 2) in [0, 1]
            meta_data: dict with input_width, input_height, frames, views
            pair_idx: list of (src, tgt) tuples, default [(0, 1)]

        Returns:
            feats: (B * num_pair, Q, hidden_dim)
        """
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

        # Build query UV grid
        uv_grid = query.uv
        uv_grid = uv_grid * 2 - 1
        uv_grid = uv_grid.unsqueeze(0).unsqueeze(2)  # (1, Q, 1, 2)
        uv_grid = uv_grid.expand(B * num_pair, -1, -1, -1)  # (B*P, Q, 1, 2)

        # Position embedding on target view grid
        pos_emb_grid = self._get_pos_emb_grid(B, patch_h, patch_w, device=x[0].device)

        stacked = []
        # Cross-view matching features (spatial format)
        for si in range(len(x)):
            match_spatial = x[si].view(B, N, patch_h, patch_w, -1)  # (B, N, L, C)

            # Per-pair: compute match_emb and build enhanced source features
            pair_layer_feats = []  # list of num_pair, each is list of num_layers x (B, L, C)
            for src, tgt in pair_idx:
                f_src_spatial = match_spatial[:, src]  # (B, H, W, C)
                f_tgt_spatial = match_spatial[:, tgt]  # (B, H, W, C)

                match_emb = self._compute_match_emb(f_src_spatial, f_tgt_spatial, pos_emb_grid)

                pair_layer_feats.append(f_src_spatial + match_emb)

            # Stack all pairs: (B, num_pair, L, C) -> (B * num_pair, L, C)
            stacked.append(torch.stack(pair_layer_feats, dim=1).view(B * num_pair, patch_h * patch_w, -1))

        hidden = self._sample_and_fuse(stacked, uv_grid, patch_h, patch_w)
        return hidden  # (B * num_pair, Q, hidden_dim)


class MatchQueryAggregator24(Aggregator24):
    """Query aggregator with roma-style cross-view matching.

    Inherits Aggregator23's network structure (q_projs, norms, interp_refinenet,
    interp_norms, h_projs, gates, fuse_norms, ffns) and adds cross-view matching.

    For each pair (src, tgt):
      1. Reshape ViT features to spatial (B, H, W, C)
      2. Compute cosine_sim(f_src, f_tgt) -> softmax -> weighted pos_emb -> match_emb
      3. Add match_emb to source view features at match_layer
      4. For all layers: q_proj -> pixel_shuffle -> norm -> [refinenet -> norm] -> grid_sample
      5. Hierarchical fusion with fuse_norms

    Output: (B * num_pair, Q, hidden_dim)
    """

    def __init__(
        self,
        # Roma matching params
        match_layer_idx=-1,
        match_dim=None,
        match_temp=0.1,
        match_scale=1.0,
        enable_amp=True,
        return_dense_feats=False,
        padding_mode="border",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.uv_round = False
        self.return_dense_feats = return_dense_feats

        self.match_layer_idx = match_layer_idx
        self.enable_amp = enable_amp

        resolved_idx = match_layer_idx if match_layer_idx >= 0 else len(self.in_chans) + match_layer_idx
        self.resolved_match_layer_idx = resolved_idx

        if match_dim is None:
            match_dim = self.in_chans[resolved_idx]
        self.match_dim = match_dim

        omega = 2 * torch.pi * torch.randn(match_dim // 2, 2)
        self.register_buffer("omega", omega)
        self.register_buffer("scale", torch.tensor(match_scale))
        self.register_buffer("temp", torch.tensor(match_temp))

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

        if self.return_dense_feats:
            feats_list = []
        else:
            feats_list = None

        for i, feats in enumerate(layer_feats):
            feats = self.q_projs[i](feats)  # (B_pair, L, embed_dim * s^2)
            feats = feats.transpose(-1, -2).view(B_pair, -1, patch_h, patch_w)
            feats = F.pixel_shuffle(feats, self.upsample_scales[i])  # B,C,H,W

            spatial_h = patch_h * self.upsample_scales[i]
            spatial_w = patch_w * self.upsample_scales[i]

            feats = feats.permute(0, 2, 3, 1).contiguous().view(B_pair, -1, feats.shape[1])
            feats = self.norms[i](feats)
            feats = feats.permute(0, 2, 1).contiguous().view(B_pair, -1, spatial_h, spatial_w)

            if self.interp_refinenet is not None:
                feats = self.interp_refinenet[i](feats)

                feats = feats.permute(0, 2, 3, 1).contiguous().view(B_pair, -1, feats.shape[1])
                feats = self.interp_norms[i](feats)
                feats = feats.permute(0, 2, 1).contiguous().view(B_pair, -1, spatial_h, spatial_w)

            if self.return_dense_feats:
                feats_list.append(feats)

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

        return hidden, feats_list

    def forward(self, x, query, meta_data, pair_idx=None, **kwargs):
        """
        Args:
            x: list of (B, N, L, C) multi-layer ViT patch features
            query: BaseQuery with uv (Q, 2) in [0, 1]
            meta_data: dict with input_width, input_height, frames, views
            pair_idx: list of (src, tgt) tuples, default [(0, 1)]

        Returns:
            feats: (B * num_pair, Q, hidden_dim)
        """
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

        # Position embedding on target view grid
        pos_emb_grid = self._get_pos_emb_grid(B, patch_h, patch_w, device=x[0].device)

        stacked = []
        # Cross-view matching features (spatial format)
        for si in range(len(x)):
            match_spatial = x[si].view(B, N, patch_h, patch_w, -1)  # (B, N, L, C)

            # Per-pair: compute match_emb and build enhanced source features
            pair_layer_feats = []  # list of num_pair, each is list of num_layers x (B, L, C)
            for src, tgt in pair_idx:
                f_src_spatial = match_spatial[:, src]  # (B, H, W, C)
                f_tgt_spatial = match_spatial[:, tgt]  # (B, H, W, C)

                match_emb = self._compute_match_emb(f_src_spatial, f_tgt_spatial, pos_emb_grid)

                pair_layer_feats.append(f_src_spatial + match_emb)

            # Stack all pairs: (B, num_pair, L, C) -> (B * num_pair, L, C)
            stacked.append(torch.stack(pair_layer_feats, dim=1).view(B * num_pair, patch_h * patch_w, -1))

        hidden, feats_list = self._sample_and_fuse(stacked, uv_grid, patch_h, patch_w)
        if self.return_dense_feats:
            return hidden, feats_list
        else:
            return hidden  # (B * num_pair, Q, hidden_dim)


class MotionQueryAggregator24(Aggregator24):
    """Query aggregator with roma-style cross-view matching.

    Inherits Aggregator23's network structure (q_projs, norms, interp_refinenet,
    interp_norms, h_projs, gates, fuse_norms, ffns) and adds cross-view matching.

    For each pair (src, tgt):
      1. Reshape ViT features to spatial (B, H, W, C)
      2. Compute cosine_sim(f_src, f_tgt) -> softmax -> weighted pos_emb -> match_emb
      3. Add match_emb to source view features at match_layer
      4. For all layers: q_proj -> pixel_shuffle -> norm -> [refinenet -> norm] -> grid_sample
      5. Hierarchical fusion with fuse_norms

    Output: (B * num_pair, Q, hidden_dim)
    """

    def __init__(
        self,
        # Roma matching params
        match_layer_idx=-1,
        match_dim=None,
        match_temp=0.1,
        match_scale=1.0,
        enable_amp=True,
        return_dense_feats=False,
        return_pyramids=False,
        with_delta_feats=False,
        padding_mode="border",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.uv_round = False
        self.return_dense_feats = return_dense_feats
        self.return_pyramids = return_pyramids

        self.match_layer_idx = match_layer_idx
        self.enable_amp = enable_amp

        resolved_idx = match_layer_idx if match_layer_idx >= 0 else len(self.in_chans) + match_layer_idx
        self.resolved_match_layer_idx = resolved_idx

        if match_dim is None:
            match_dim = self.in_chans[resolved_idx]
        self.match_dim = match_dim

        omega = 2 * torch.pi * torch.randn(match_dim // 2, 2)
        self.register_buffer("omega", omega)
        self.register_buffer("scale", torch.tensor(match_scale))
        self.register_buffer("temp", torch.tensor(match_temp))

        self.with_delta_feats = with_delta_feats

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

    def forward(self, x, query, meta_data, pair_idx=None, rgb=None, **kwargs):
        """
        Args:
            x: list of (B, N, L, C) multi-layer ViT patch features
            query: BaseQuery with uv (Q, 2) in [0, 1]
            meta_data: dict with input_width, input_height, frames, views
            pair_idx: list of (src, tgt) tuples, default [(0, 1)]

        Returns:
            feats: (B * num_pair, Q, hidden_dim)
        """
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
            feats = self.q_projs[i](feats)  # (B_pair, L, embed_dim * s^2)
            feats = feats.transpose(-1, -2).view(B * N, -1, patch_h, patch_w)
            feats = F.pixel_shuffle(feats, self.upsample_scales[i])  # B,C,H,W

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

            src_pyramid = torch.stack([feats.view(B, N, -1, spatial_h, spatial_w)[:, src] for src, tgt in pair_idx], dim=1)
            tgt_pyramid = torch.stack([feats.view(B, N, -1, spatial_h, spatial_w)[:, tgt] for src, tgt in pair_idx], dim=1)

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

        src_hidden = self._sample_and_fuse(src_pyramids, uv_grid, patch_h, patch_w)
        tgt_hidden = self._sample_and_fuse(tgt_pyramids, uv_grid, patch_h, patch_w)

        # hidden = torch.cat([src_hidden, tgt_hidden], dim=-1)
        if self.with_delta_feats:
            hidden = torch.cat([src_hidden, tgt_hidden, tgt_hidden - src_hidden], dim=-1)
        else:
            hidden = torch.cat([src_hidden, tgt_hidden], dim=-1)

        if self.return_dense_feats:
            return hidden, None
        elif self.return_pyramids:
            return hidden, [src_pyramids, tgt_pyramids]
        else:
            return hidden  # (B * num_pair, Q, hidden_dim)


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


class PairQueryAggregator(Aggregator24):
    """Query aggregator with roma-style cross-view matching.

    Inherits Aggregator23's network structure (q_projs, norms, interp_refinenet,
    interp_norms, h_projs, gates, fuse_norms, ffns) and adds cross-view matching.

    For each pair (src, tgt):
      1. Reshape ViT features to spatial (B, H, W, C)
      2. Compute cosine_sim(f_src, f_tgt) -> softmax -> weighted pos_emb -> match_emb
      3. Add match_emb to source view features at match_layer
      4. For all layers: q_proj -> pixel_shuffle -> norm -> [refinenet -> norm] -> grid_sample
      5. Hierarchical fusion with fuse_norms

    Output: (B * num_pair, Q, hidden_dim)
    """

    def __init__(
        self,
        # Roma matching params
        match_layer_idx=-1,
        match_dim=None,
        match_temp=0.1,
        match_scale=1.0,
        enable_amp=True,
        return_dense_feats=False,
        return_pyramids=False,
        with_match_embed=False,
        with_time_embed=False,
        time_fourier_dim=128,
        time_fourier_scale=5.0,
        padding_mode="border",
        **kwargs,
    ):
        super().__init__(**kwargs)

        torch.cuda.synchronize

        self.uv_round = False
        self.return_dense_feats = return_dense_feats
        self.return_pyramids = return_pyramids

        self.match_layer_idx = match_layer_idx
        self.enable_amp = enable_amp

        self.with_match_embed = with_match_embed
        self.with_time_embed = with_time_embed

        resolved_idx = match_layer_idx if match_layer_idx >= 0 else len(self.in_chans) + match_layer_idx
        self.resolved_match_layer_idx = resolved_idx

        if match_dim is None:
            match_dim = self.in_chans[resolved_idx]
        self.match_dim = match_dim

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
            x: list of (B, N, L, C) multi-layer ViT patch features
            query: BaseQuery with uv (Q, 2) in [0, 1]
            meta_data: dict with input_width, input_height, frames, views
            pair_idx: list of (src, tgt) tuples, default [(0, 1)]

        Returns:
            feats: (B * num_pair, Q, hidden_dim)
        """
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
            feats = self.q_projs[i](feats)  # (B_pair, L, embed_dim * s^2)
            feats = feats.transpose(-1, -2).view(B * N, -1, patch_h, patch_w)
            feats = F.pixel_shuffle(feats, self.upsample_scales[i])  # B,C,H,W

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

            src_pyramid = torch.stack([feats.view(B, N, -1, spatial_h, spatial_w)[:, src] for src, tgt in pair_idx], dim=1)
            tgt_pyramid = torch.stack([feats.view(B, N, -1, spatial_h, spatial_w)[:, tgt] for src, tgt in pair_idx], dim=1)

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

        src_hidden = self._sample_and_fuse(src_pyramids, uv_grid, patch_h, patch_w)
        tgt_hidden = self._sample_and_fuse(tgt_pyramids, uv_grid, patch_h, patch_w)
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

        if self.return_dense_feats:
            return hidden, None
        elif self.return_pyramids:
            return hidden, [src_pyramids, tgt_pyramids]
        else:
            return hidden  # (B * num_pair, Q, hidden_dim)
