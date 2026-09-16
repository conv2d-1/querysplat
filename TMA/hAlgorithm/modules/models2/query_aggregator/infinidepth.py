import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from hAlgorithm.utils import instantiate_from_config
import logging


class Aggregator23(torch.nn.Module):
    """layer norm."""

    def __init__(
        self,
        patch_size=14,
        in_chans=[1024, 1024, 1024],
        embed_dims=[256, 512, 1024],
        upsample_scales=[4, 2, 1],
        mode="bilinear",
        interp_refinenet_cfg=None,
        intermediate_layer_idx=None,
        uv_round=False,
        debug=False,
    ):
        super(Aggregator23, self).__init__()

        self.patch_size = patch_size
        self.mode = mode
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.upsample_scales = upsample_scales
        self.intermediate_layer_idx = intermediate_layer_idx
        self.uv_round = uv_round
        self.debug = debug

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
        self.gates = nn.ParameterList()
        self.ffns = nn.ModuleList()
        for i in range(len(self.embed_dims) - 1):
            hidden_dim = self.embed_dims[i + 1]

            self.h_projs.append(nn.Linear(self.embed_dims[i], hidden_dim))

            gate_param = nn.Parameter(torch.zeros(hidden_dim))
            self.gates.append(gate_param)

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

    def forward(self, x, query, meta_data, query_rgb=None, **kwargs):

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

        uv_grid = query.uv
        Q = uv_grid.shape[-2]

        if not self.uv_round:
            uv_grid = uv_grid * 2 - 1
        uv_grid = uv_grid.unsqueeze(0).unsqueeze(2)  # (1, Q, 1, 2)
        uv_grid = uv_grid.expand(B * N, -1, -1, -1)  # (B*N, Q, 1, 2)

        # step1: grid_sample
        sampled_feats_list = []
        for i, feats in enumerate(x):

            feats = self.q_projs[i](feats.view(B * N, -1, feats.shape[-1]))
            feats = feats.transpose(-1, -2).view(B * N, -1, patch_h, patch_w)
            feats = F.pixel_shuffle(feats, self.upsample_scales[i])  # B,C,H,W
            feats = feats.permute(0, 2, 3, 1).contiguous().view(B * N, -1, feats.shape[1])

            feats = self.norms[i](feats)
            feats = feats.permute(0, 2, 1).contiguous().view(B * N, -1, patch_h * self.upsample_scales[i], patch_w * self.upsample_scales[i])

            if self.interp_refinenet is not None:
                feats = self.interp_refinenet[i](feats)
                
                feats = feats.permute(0, 2, 3, 1).contiguous().view(B * N, -1, feats.shape[1])
                feats = self.interp_norms[i](feats)
                feats = feats.permute(0, 2, 1).contiguous().view(B * N, -1, patch_h * self.upsample_scales[i], patch_w * self.upsample_scales[i])

            if self.uv_round:
                cur_height, cur_width = feats.shape[-2:]
                uv_grid_round = uv_grid.clone().squeeze(2)
                uv_grid_round[..., 0] *= (cur_width -1)
                uv_grid_round[..., 1] *= (cur_height - 1)
                uv_grid_round = torch.round(uv_grid_round).long()
                BN, Q, _ = uv_grid_round.shape
                batch_idx = torch.arange(BN, device=feats.device).view(BN, 1).expand(BN, Q)
                # (B*N, Q, C)
                sampled_feats = feats[batch_idx, :, uv_grid_round[..., 1], uv_grid_round[..., 0]]
            else:
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

            gate = torch.sigmoid(self.gates[k - 1])  # (hidden_dim,)
            gate = gate.view(1, 1, -1)  # (1, 1, hidden_dim)

            hidden = sampled_feats_list[k] + gate * hidden
            hidden = self.fuse_norms[k - 1](hidden)

            if self.debug:
                logging.info(f"{k}, hidden2: {hidden.max()}, {hidden.min()}")

            hidden = self.ffns[k - 1](hidden)

            if self.debug:
                logging.info(f"{k}, hidden3: {hidden.max()}, {hidden.min()}")

        return hidden


class Aggregator24(torch.nn.Module):
    """layer norm. gate"""

    def __init__(
        self,
        patch_size=14,
        in_chans=[1024, 1024, 1024],
        embed_dims=[256, 512, 1024],
        upsample_scales=[4, 2, 1],
        mode="bilinear",
        interp_refinenet_cfg=None,
        intermediate_layer_idx=None,
        uv_round=False,
        align_corners=False,
        pretrain=None,
        debug=False,
    ):
        super(Aggregator24, self).__init__()

        self.patch_size = patch_size
        self.mode = mode
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.upsample_scales = upsample_scales
        self.intermediate_layer_idx = intermediate_layer_idx
        self.uv_round = uv_round
        self.align_corners = align_corners
        self.debug = debug

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

        self.pretrain = pretrain
        if pretrain is not None:
            res = self.load_state_dict(
                torch.load(pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"Aggregator24, load pretrain {pretrain}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")

    def forward(self, x, query, meta_data, query_rgb=None, **kwargs):

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

        uv_grid = query.uv
        Q = uv_grid.shape[-2]
        if uv_grid.ndim == 4:
            uv_grid = uv_grid.view(B * N, Q, 2)

        if not self.uv_round:
            uv_grid = uv_grid * 2 - 1

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

            feats = self.q_projs[i](feats.view(B * N, -1, feats.shape[-1]))
            feats = feats.transpose(-1, -2).view(B * N, -1, patch_h, patch_w)
            feats = F.pixel_shuffle(feats, self.upsample_scales[i])  # B,C,H,W
            feats = feats.permute(0, 2, 3, 1).contiguous().view(B * N, -1, feats.shape[1])

            feats = self.norms[i](feats)
            feats = feats.permute(0, 2, 1).contiguous().view(B * N, -1, patch_h * self.upsample_scales[i], patch_w * self.upsample_scales[i])

            if self.interp_refinenet is not None:
                feats = self.interp_refinenet[i](feats)
                
                feats = feats.permute(0, 2, 3, 1).contiguous().view(B * N, -1, feats.shape[1])
                feats = self.interp_norms[i](feats)
                feats = feats.permute(0, 2, 1).contiguous().view(B * N, -1, patch_h * self.upsample_scales[i], patch_w * self.upsample_scales[i])

            if self.uv_round:
                cur_height, cur_width = feats.shape[-2:]
                uv_grid_round = uv_grid.clone().squeeze(2)
                uv_grid_round[..., 0] *= (cur_width -1)
                uv_grid_round[..., 1] *= (cur_height - 1)
                uv_grid_round = torch.round(uv_grid_round).long()
                BN, Q, _ = uv_grid_round.shape
                batch_idx = torch.arange(BN, device=feats.device).view(BN, 1).expand(BN, Q)
                # (B*N, Q, C)
                sampled_feats = feats[batch_idx, :, uv_grid_round[..., 1], uv_grid_round[..., 0]]
            else:
                # (B*N, C, Q, 1)
                input_dtype = feats.dtype
                sampled_feats = F.grid_sample(feats.float(), uv_grid.float(), mode=self.mode, align_corners=self.align_corners, padding_mode="border")
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

class Aggregator24MemEffi(Aggregator24):
    """Memory-efficient variant of Aggregator24.

    A: optional query chunking (disabled by default).
    B: streaming fusion to avoid storing all sampled levels (enabled).
    C: optional checkpoint on FFN blocks (disabled by default).
    """

    def __init__(
        self,
        enable_q_chunk=False,
        q_chunk_size=8192,
        use_checkpoint=False,
        **kwargs,
    ):
        super(Aggregator24MemEffi, self).__init__(**kwargs)
        self.enable_q_chunk = enable_q_chunk
        self.q_chunk_size = q_chunk_size
        self.use_checkpoint = use_checkpoint

    def _build_scale_feat_map(self, feats, i, B, N, patch_h, patch_w):
        feats = self.q_projs[i](feats.view(B * N, -1, feats.shape[-1]))
        feats = feats.transpose(-1, -2).view(B * N, -1, patch_h, patch_w)
        feats = F.pixel_shuffle(feats, self.upsample_scales[i])  # (B*N, C, H, W)
        feats = feats.permute(0, 2, 3, 1).contiguous().view(B * N, -1, feats.shape[1])

        feats = self.norms[i](feats)
        feats = feats.permute(0, 2, 1).contiguous().view(
            B * N,
            -1,
            patch_h * self.upsample_scales[i],
            patch_w * self.upsample_scales[i],
        )

        if self.interp_refinenet is not None:
            feats = self.interp_refinenet[i](feats)
            feats = feats.permute(0, 2, 3, 1).contiguous().view(B * N, -1, feats.shape[1])
            feats = self.interp_norms[i](feats)
            feats = feats.permute(0, 2, 1).contiguous().view(
                B * N,
                -1,
                patch_h * self.upsample_scales[i],
                patch_w * self.upsample_scales[i],
            )

        return feats

    def _sample_from_feat_map(self, feats, uv_grid_chunk):
        if self.uv_round:
            cur_height, cur_width = feats.shape[-2:]
            uv_grid_round = uv_grid_chunk.clone().squeeze(2)
            uv_grid_round[..., 0] *= (cur_width - 1)
            uv_grid_round[..., 1] *= (cur_height - 1)
            uv_grid_round = torch.round(uv_grid_round).long()
            BN, q_len, _ = uv_grid_round.shape
            batch_idx = torch.arange(BN, device=feats.device).view(BN, 1).expand(BN, q_len)
            sampled_feats = feats[batch_idx, :, uv_grid_round[..., 1], uv_grid_round[..., 0]]
        else:
            input_dtype = feats.dtype
            sampled_feats = F.grid_sample(
                feats.float(),
                uv_grid_chunk.float(),
                mode=self.mode,
                align_corners=False,
                padding_mode="border",
            )
            sampled_feats = sampled_feats.to(input_dtype)
            sampled_feats = sampled_feats.squeeze(-1).permute(0, 2, 1).contiguous()
        return sampled_feats

    def _ffn_forward(self, ffn, hidden):
        can_checkpoint = self.use_checkpoint and self.training and torch.is_grad_enabled() and hidden.requires_grad
        if can_checkpoint:
            return torch_checkpoint(ffn, hidden, use_reentrant=False)
        return ffn(hidden)

    def _fuse_one_chunk(self, feat_maps, uv_grid_chunk):
        hidden = self._sample_from_feat_map(feat_maps[0], uv_grid_chunk)
        if self.debug:
            logging.info(f"0, sampled_feats: {hidden.max()}, {hidden.min()}")

        for k in range(1, len(feat_maps)):
            sampled_k = self._sample_from_feat_map(feat_maps[k], uv_grid_chunk)
            if self.debug:
                logging.info(f"{k}, sampled_feats: {sampled_k.max()}, {sampled_k.min()}")

            hidden = self.h_projs[k - 1](hidden)
            if self.debug:
                logging.info(f"{k}, hidden1: {hidden.max()}, {hidden.min()}")

            gate_weights = self.gates[k - 1](torch.cat([hidden, sampled_k], dim=-1))
            hidden = gate_weights * hidden + (1 - gate_weights) * sampled_k
            hidden = self.fuse_norms[k - 1](hidden)
            if self.debug:
                logging.info(f"{k}, hidden2: {hidden.max()}, {hidden.min()}")

            hidden = self._ffn_forward(self.ffns[k - 1], hidden)
            if self.debug:
                logging.info(f"{k}, hidden3: {hidden.max()}, {hidden.min()}")

        return hidden

    def forward(self, x, query, meta_data, query_rgb=None, **kwargs):
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

        uv_grid = query.uv
        Q = uv_grid.shape[-2]
        if uv_grid.ndim == 4:
            uv_grid = uv_grid.view(B * N, Q, 2)

        if not self.uv_round:
            uv_grid = uv_grid * 2 - 1

        if uv_grid.ndim == 2:
            uv_grid = uv_grid.unsqueeze(0).unsqueeze(2)  # (1, Q, 1, 2)
            uv_grid = uv_grid.expand(B * N, -1, -1, -1)  # (B*N, Q, 1, 2)
        elif uv_grid.ndim == 3:
            uv_grid = uv_grid.unsqueeze(2)  # (B*N, Q, 1, 2)
        else:
            raise ValueError(f"Unsupported uv_grid dimension: {uv_grid.ndim}")

        feat_maps = []
        for i, feats_i in enumerate(x):
            feat_maps.append(self._build_scale_feat_map(feats_i, i, B, N, patch_h, patch_w))

        if not self.enable_q_chunk:
            return self._fuse_one_chunk(feat_maps, uv_grid)

        chunk_size = max(int(self.q_chunk_size), 1)
        hidden_out = None
        for start in range(0, Q, chunk_size):
            end = min(start + chunk_size, Q)
            hidden_chunk = self._fuse_one_chunk(feat_maps, uv_grid[:, start:end])
            if hidden_out is None:
                hidden_out = hidden_chunk.new_empty(
                    hidden_chunk.shape[0],
                    Q,
                    hidden_chunk.shape[-1],
                )
            hidden_out[:, start:end] = hidden_chunk
        return hidden_out

