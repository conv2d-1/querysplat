import torch
import torch.nn as nn
import torch.nn.functional as F


class Aggregator(torch.nn.Module):
    def __init__(
        self,
        patch_size=14,
        in_chans=[1024, 1024, 1024],
        embed_dims=[256, 512, 1024],
        upsample_scales=[4, 2, 1],
        mode="bilinear",
    ):
        super(Aggregator, self).__init__()

        self.patch_size = patch_size
        self.mode = mode
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.upsample_scales = upsample_scales

        self.q_projs = nn.ModuleList()
        for ch0, ch1 in zip(self.in_chans, self.embed_dims):
            self.q_projs.append(nn.Linear(ch0, ch1))

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

    def forward(self, x, query, meta_data, **kwargs):

        patch_w = meta_data["input_width"][0] // self.patch_size
        patch_h = meta_data["input_height"][0] // self.patch_size

        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        assert len(x) == len(self.in_chans)
        B, N = x[0].shape[:2]
        assert N == frame_num * view_num

        uv_grid = query.uv
        Q = uv_grid.shape[0]

        # uv_grid = uv_grid * 2 - 1
        # uv_grid = uv_grid.unsqueeze(0).unsqueeze(2)  # (1, Q, 1, 2)
        # uv_grid = uv_grid.expand(B * N, -1, -1, -1)  # (B*N, Q, 1, 2)

        # step1: grid_sample
        sampled_feats_list = []
        for i, feats in enumerate(x):
            feats = self.q_projs[i](feats.view(B * N, -1, feats.shape[-1]))
            C = feats.shape[-1]

            # (B*N, C, H, W)
            feats = feats.view(B * N, patch_h, patch_w, C).permute(0, 3, 1, 2).contiguous()

            new_h = patch_h * self.upsample_scales[i]
            new_w = patch_w * self.upsample_scales[i]
            feats = F.interpolate(feats, size=(new_h, new_w), mode="bilinear", align_corners=False)

            # (B*N, Q, 2)
            x_coords = (uv_grid[..., 0] * (new_w - 1)).long()
            y_coords = (uv_grid[..., 1] * (new_h - 1)).long()
            # (B*N, C, Q)
            sampled_feats = feats[..., y_coords, x_coords]
            # sampled_feats = F.grid_sample(feats, uv_grid, mode=self.mode, padding_mode="border", align_corners=False)
            # (B*N, Q, C)
            sampled_feats = sampled_feats.permute(0, 2, 1).contiguous()
            sampled_feats_list.append(sampled_feats)

        # Step 2: fuse
        hidden = sampled_feats_list[0]
        for k in range(1, len(sampled_feats_list)):
            hidden = self.h_projs[k - 1](hidden)

            gate = torch.sigmoid(self.gates[k - 1])  # (hidden_dim,)
            gate = gate.view(1, 1, -1)  # (1, 1, hidden_dim)

            hidden = sampled_feats_list[k] + gate * hidden

            hidden = self.ffns[k - 1](hidden)

        return hidden
