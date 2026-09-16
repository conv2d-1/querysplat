"""Clip-level memory cross-attention aggregator for WFM query motion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models2.query_aggregator.unified_query import PairCrossAttnAggregator


class ClipMemoryCrossAttnAggregator(PairCrossAttnAggregator):
    """Cross-attend src-sampled query features against **all clip frames**.

    Compared to ``PairCrossAttnAggregator`` (tgt-only memory):

    - **Unchanged**: ``QueryBank5`` UV sampling, src/tgt pyramid sampling, decoder
      concat layout ``[src_hidden, tgt_hidden, cross_out]``.
    - **Changed**: cross-attention memory keys/values come from every frame in the
      clip, not only the target frame.

    ``pair_idx`` is still used to pick src/tgt frames for sampling and losses, but
    not to restrict cross-attention memory.
    """

    def __init__(
        self,
        with_frame_pos_embed: bool = True,
        max_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.with_frame_pos_embed = bool(with_frame_pos_embed)
        self.max_frames = int(max_frames)
        embed_dim = int(self.embed_dims[-1])
        if self.with_frame_pos_embed:
            self.frame_embed = nn.Embedding(self.max_frames, embed_dim)
        else:
            self.frame_embed = None

    def _build_clip_memory(self, feats_bnchw: torch.Tensor, B: int, num_pair: int) -> torch.Tensor:
        """Build ``(B*P, N*H*W, C)`` memory from all frames at the last pyramid scale."""
        # feats_bnchw: (B, N, C, H, W)
        n_frames = feats_bnchw.shape[1]
        if n_frames > self.max_frames:
            raise ValueError(
                f"Clip has {n_frames} frames but max_frames={self.max_frames}. "
                "Increase max_frames in the aggregator config."
            )

        mem = feats_bnchw.permute(0, 1, 3, 4, 2).reshape(B, -1, feats_bnchw.shape[2])
        if self.frame_embed is not None:
            _, _, _, height, width = feats_bnchw.shape
            frame_ids = torch.arange(n_frames, device=feats_bnchw.device)
            frame_pe = self.frame_embed(frame_ids)  # (N, C)
            frame_pe = frame_pe.view(1, n_frames, 1, 1, -1).expand(B, -1, height, width, -1)
            frame_pe = frame_pe.reshape(B, -1, feats_bnchw.shape[2])
            mem = mem + frame_pe

        return mem.unsqueeze(1).expand(-1, num_pair, -1, -1).reshape(B * num_pair, -1, feats_bnchw.shape[2])

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
            n = frame_num * view_num
            b = x[0].shape[0] // n
            x = [feat.view(b, n, -1, feat.shape[-1]) for feat in x]
        else:
            b, n = x[0].shape[:2]

        src_pyramids = []
        tgt_pyramids = []
        last_feats_bnchw = None
        for i, feats in enumerate(x):
            feats = self.q_projs[i](feats)
            feats = feats.transpose(-1, -2).view(b * n, -1, patch_h, patch_w)
            feats = F.pixel_shuffle(feats, self.upsample_scales[i])

            spatial_h = patch_h * self.upsample_scales[i]
            spatial_w = patch_w * self.upsample_scales[i]

            feats = feats.permute(0, 2, 3, 1).contiguous().view(b * n, -1, feats.shape[1])
            feats = self.norms[i](feats)
            feats = feats.permute(0, 2, 1).contiguous().view(b * n, -1, spatial_h, spatial_w)

            if self.interp_refinenet is not None:
                feats = self.interp_refinenet[i](feats)
                feats = feats.permute(0, 2, 3, 1).contiguous().view(b * n, -1, feats.shape[1])
                feats = self.interp_norms[i](feats)
                feats = feats.permute(0, 2, 1).contiguous().view(b * n, -1, spatial_h, spatial_w)

            feats_bnchw = feats.view(b, n, -1, spatial_h, spatial_w)
            src_pyramid = torch.stack([feats_bnchw[:, src] for src, _ in pair_idx], dim=1)
            tgt_pyramid = torch.stack([feats_bnchw[:, tgt] for _, tgt in pair_idx], dim=1)

            src_pyramids.append(src_pyramid.view(b * num_pair, *src_pyramid.shape[2:]))
            tgt_pyramids.append(tgt_pyramid.view(b * num_pair, *tgt_pyramid.shape[2:]))
            last_feats_bnchw = feats_bnchw

        uv_grid = query.uv
        uv_grid = uv_grid * 2 - 1
        if uv_grid.ndim == 4:
            uv_grid = (
                torch.stack([uv_grid[:, pair[0]] for pair in pair_idx], dim=1)
                .view(b * num_pair, *uv_grid.shape[-2:])
                .unsqueeze(2)
            )
        elif uv_grid.ndim == 2:
            uv_grid = uv_grid.unsqueeze(0).unsqueeze(2).expand(b * num_pair, -1, -1, -1)
        else:
            raise ValueError(f"Unsupported uv_grid dimension: {uv_grid.ndim}")

        src_hidden = self._sample_and_fuse(src_pyramids, uv_grid, patch_h, patch_w)

        if not self.without_tgt_hidden:
            tgt_hidden = self._sample_and_fuse(tgt_pyramids, uv_grid, patch_h, patch_w)

        if self.memory_from_encoder_last:
            enc_last = x[-1]
            l_tok = enc_last.shape[2]
            if l_tok != patch_h * patch_w:
                raise ValueError(
                    f"memory_from_encoder_last: token len {l_tok} != patch_h*patch_w ({patch_h}*{patch_w})"
                )
            mem_map = enc_last.view(b, n, patch_h, patch_w, -1).contiguous()
            if self.encoder_mem_adapt is not None:
                mem_map = self.encoder_mem_adapt(mem_map)
            mem_map = mem_map.permute(0, 1, 3, 4, 2).reshape(b, -1, mem_map.shape[-1])
            if self.frame_embed is not None:
                frame_ids = torch.arange(n, device=mem_map.device)
                frame_pe = self.frame_embed(frame_ids).view(1, n, 1, 1, -1).expand(b, -1, patch_h, patch_w, -1)
                frame_pe = frame_pe.reshape(b, -1, mem_map.shape[-1])
                mem_map = mem_map + frame_pe
            mem_map = mem_map.unsqueeze(1).expand(-1, num_pair, -1, -1).reshape(b * num_pair, -1, mem_map.shape[-1])
        else:
            if last_feats_bnchw is None:
                raise RuntimeError("Failed to build clip memory: missing pyramid features.")
            mem_map = self._build_clip_memory(last_feats_bnchw, b, num_pair)

        query_tokens = self.query_proj(src_hidden)
        memory_tokens = self.memory_proj(mem_map)
        for blk in self.cross_blocks:
            query_tokens = blk(query_tokens, memory_tokens)
        cross_readout = self.readout(query_tokens)

        if self.without_tgt_hidden:
            hidden = cross_readout if self.without_src_hidden else torch.cat([src_hidden, cross_readout], dim=-1)
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
