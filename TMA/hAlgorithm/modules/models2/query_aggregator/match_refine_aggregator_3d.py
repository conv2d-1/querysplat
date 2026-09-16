import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryWarp3DRefine(nn.Module):
    """Sparse query-space analogue of RoMa's refiner stack.

    It keeps RoMa's multi-scale `FineFeatures` pyramid, but refines sparse query
    matches directly instead of refining a dense warp grid.
    """

    def __init__(
        self,
        embed_dims,
        stage=1,
        hidden_dim=128,
        intermediate_layer_idx=None,
        patch_size=14,
        mode="bilinear",
        warp3d_name="warp3d",
        warp2d_name="warp2d",
        warp2d_detach=True,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode

        self.intermediate_layer_idx = intermediate_layer_idx
        self.patch_size = patch_size

        self.embed_dims = embed_dims
        self.hidden_dim = hidden_dim

        self.warp3d_name = warp3d_name

        self.warp2d_name = warp2d_name
        self.warp2d_detach = warp2d_detach

        self.stage = stage

        # -------------------------------------------
        self.h_projs = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.fuse_norms = nn.ModuleList()
        for i in range(len(self.embed_dims) - 1):
            hidden_dim = self.embed_dims[i + 1]

            self.h_projs.append(nn.Linear(self.embed_dims[i], hidden_dim))

            self.gates.append(nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()))

            ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))
            self.ffns.append(ffn)

            self.fuse_norms.append(nn.LayerNorm(self.embed_dims[i + 1]))

        # --------------------------------------------
        self.output_dim = 3

        layers = []
        in_dim = self.embed_dims[-1] * 2
        for _ in range(2):
            layers.extend(
                [
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.mlp = nn.Sequential(*layers)

    def _sample_and_fuse(self, layer_feats, uv_grid):
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
            sampled = F.grid_sample(feats.float(), uv_grid.float(), mode=self.mode, align_corners=False, padding_mode="border")
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

    def forward(
        self,
        coarse_results,
        query,
        rgb,
        feats,
        pair_idx=None,
        meta_data=None,
        **kwargs,
    ):
        if rgb.ndim != 5:
            raise ValueError("QueryMatchRefine expects rgb with shape (B, N, C, H, W).")

        B, N, _, H, W = rgb.shape
        num_pair = len(pair_idx)
        BP = B * num_pair

        # feats: [src_pyramids, tgt_pyramids]
        # pyramids: B * num_pair, *pyramid.shape[2:]
        if self.intermediate_layer_idx is not None:
            src_pyramids = [feats[0][idx] for idx in self.intermediate_layer_idx]
            tgt_pyramids = [feats[0][idx] for idx in self.intermediate_layer_idx]
        else:
            src_pyramids = feats[0]
            tgt_pyramids = feats[1]

        assert len(src_pyramids) == len(self.embed_dims)

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

        warp3d = coarse_results[self.warp3d_name]

        warp2d = coarse_results[self.warp2d_name]
        if warp2d.ndim == 3:
            warp2d = warp2d.unsqueeze(-2)
        if self.warp2d_detach:
            warp2d = warp2d.detach()

        output = dict()
        src_hidden = self._sample_and_fuse(src_pyramids, uv_grid)
        for i in range(self.stage):
            tgt_hidden = self._sample_and_fuse(tgt_pyramids, warp2d)

            hidden = torch.cat([src_hidden, tgt_hidden], dim=-1)

            warp3d_delta = self.mlp(hidden)

            output[f"refiner_{i + 1}_warp3d"] = warp3d + warp3d_delta

        return output


class QueryWarp3DRefineV2(nn.Module):
    """Sparse query-space analogue of RoMa's refiner stack.

    It keeps RoMa's multi-scale `FineFeatures` pyramid, but refines sparse query
    matches directly instead of refining a dense warp grid.
    """

    def __init__(
        self,
        embed_dims,
        stage=1,
        hidden_dim=128,
        intermediate_layer_idx=None,
        patch_size=14,
        mode="bilinear",
        warp2d_name="warp2d",
        warp2d_detach=True,
        names=None,
        acts=None,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode

        self.intermediate_layer_idx = intermediate_layer_idx
        self.patch_size = patch_size

        self.embed_dims = embed_dims
        self.hidden_dim = hidden_dim

        self.warp2d_name = warp2d_name
        self.warp2d_detach = warp2d_detach

        self.stage = stage

        # -------------------------------------------
        self.h_projs = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.fuse_norms = nn.ModuleList()
        for i in range(len(self.embed_dims) - 1):
            hidden_dim = self.embed_dims[i + 1]

            self.h_projs.append(nn.Linear(self.embed_dims[i], hidden_dim))

            self.gates.append(nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()))

            ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))
            self.ffns.append(ffn)

            self.fuse_norms.append(nn.LayerNorm(self.embed_dims[i + 1]))

        # --------------------------------------------
        self.names = names
        self.acts = acts
        self.output_dim = sum(self.names.values())

        layers = []
        in_dim = self.embed_dims[-1] * 2
        for _ in range(2):
            layers.extend(
                [
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.mlp = nn.Sequential(*layers)

    def _sample_and_fuse(self, layer_feats, uv_grid):
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
            sampled = F.grid_sample(feats.float(), uv_grid.float(), mode=self.mode, align_corners=False, padding_mode="border")
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

    def forward(
        self,
        coarse_results,
        query,
        rgb,
        feats,
        pair_idx=None,
        meta_data=None,
        **kwargs,
    ):
        if rgb.ndim != 5:
            raise ValueError("QueryMatchRefine expects rgb with shape (B, N, C, H, W).")

        B, N, _, H, W = rgb.shape
        num_pair = len(pair_idx)
        BP = B * num_pair

        # feats: [src_pyramids, tgt_pyramids]
        # pyramids: B * num_pair, *pyramid.shape[2:]
        if self.intermediate_layer_idx is not None:
            src_pyramids = [feats[0][idx] for idx in self.intermediate_layer_idx]
            tgt_pyramids = [feats[0][idx] for idx in self.intermediate_layer_idx]
        else:
            src_pyramids = feats[0]
            tgt_pyramids = feats[1]

        assert len(src_pyramids) == len(self.embed_dims)

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

        warp2d = coarse_results[self.warp2d_name]
        if warp2d.ndim == 3:
            warp2d = warp2d.unsqueeze(-2)
        if self.warp2d_detach:
            warp2d = warp2d.detach()

        output = dict()
        src_hidden = self._sample_and_fuse(src_pyramids, uv_grid)
        for i in range(self.stage):
            tgt_hidden = self._sample_and_fuse(tgt_pyramids, warp2d)

            hidden = torch.cat([src_hidden, tgt_hidden], dim=-1)
            x = self.mlp(hidden)

            for name, dim in self.names.items():
                cur_x = x[..., :dim]
                cur_x = self._apply_activation_single(cur_x, self.acts.get(name, "") if self.acts is not None else "")

                output[f"refiner_{i + 1}_{name}"] = coarse_results[name] + cur_x

                x = x[..., dim:]

        return output

    def _apply_activation_single(self, x: torch.Tensor, activation: str = "linear") -> torch.Tensor:
        """
        Apply activation to single channel output, maintaining semantic consistency with value branch in multi-channel case.
        Supports: exp / relu / sigmoid / softplus / tanh / linear / expp1
        """
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "log":
            return torch.log(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "expm1":
            return torch.expm1(x)
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        if act == "inv_log":
            return torch.sign(x) * (torch.expm1(torch.abs(x)))
        # Default linear
        return x
