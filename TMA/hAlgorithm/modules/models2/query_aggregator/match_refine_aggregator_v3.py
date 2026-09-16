import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models2.external.romav2.models.romav2.features import FineFeatures
from hAlgorithm.modules.models2.external.romav2.models.romav2.refiner import Refiners


def _update_cfg(cfg, updates):
    if updates is None:
        return cfg
    for key, value in updates.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _prepare_rgb_for_refiner(rgb: torch.Tensor) -> torch.Tensor:
    # Query models use [-1, 1] normalized images, while FineFeatures expects [0, 1].
    return (rgb + 1.0) * 0.5


def _sample_bhwc(feat: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Sample a BHWC feature map at normalized coordinates.

    Supports grid shapes:
    - (B, Q, 2)      -> returns (B, Q, C)
    - (B, Q, K, 2)   -> returns (B, Q, K, C)
    """
    feat_chw = feat.permute(0, 3, 1, 2).contiguous().float()
    input_dtype = feat.dtype

    if grid.ndim == 3:
        sampled = F.grid_sample(
            feat_chw,
            grid.unsqueeze(2).float(),
            mode=mode,
            align_corners=False,
            padding_mode="zeros",
        )
        return sampled.squeeze(-1).permute(0, 2, 1).contiguous().to(input_dtype)

    if grid.ndim == 4:
        sampled = F.grid_sample(
            feat_chw,
            grid.float(),
            mode=mode,
            align_corners=False,
            padding_mode="zeros",
        )
        return sampled.permute(0, 2, 3, 1).contiguous().to(input_dtype)

    raise ValueError(f"Unsupported grid shape: {grid.shape}")


def _get_refiner_feat_dims(refiner_cfg):
    if refiner_cfg.refiner_type != "roma-4-pow2":
        raise ValueError(f"Unsupported refiner_type: {refiner_cfg.refiner_type}")

    block_cfgs = {
        4: dict(feat_dim=256, local_corr_radius=1),
        2: dict(feat_dim=128, local_corr_radius=1),
        1: dict(feat_dim=64, local_corr_radius=None),
    }
    return {scale: block_cfgs[scale] for scale in refiner_cfg.refine_scales}


class _SparseQueryRefineBlock(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, disp_dim=128, local_corr_radius=None, warp_dim=2, confidence_dim=4, refine_init=4.0, fuse_mlp_nums=1, fuse_mlp_ratio=2):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.local_corr_radius = local_corr_radius
        self.warp_dim = warp_dim
        self.confidence_dim = confidence_dim
        self.refine_init = refine_init
        self.fuse_mlp_nums = fuse_mlp_nums
        self.disp_dim = disp_dim

        self.src_proj = nn.Linear(feat_dim, hidden_dim)
        # self.tgt_proj = nn.Linear(feat_dim, hidden_dim)
        self.disp_proj = nn.Linear(2, disp_dim)

        corr_dim = 0
        if local_corr_radius is not None:
            corr_dim = (2 * local_corr_radius + 1) ** 2
            self.corr_norm = nn.LayerNorm(corr_dim)
        self.corr_dim = corr_dim

        fused_dim = hidden_dim * 2 + disp_dim + corr_dim
        self.fused_norm = nn.LayerNorm(fused_dim)

        fuse = []
        in_dim = fused_dim
        for i in range(fuse_mlp_nums):
            fuse.extend(
                [
                    nn.Linear(in_dim, hidden_dim * fuse_mlp_ratio),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim * fuse_mlp_ratio, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            in_dim = hidden_dim

        self.fuse = nn.Sequential(*fuse)
        self.warp_head = nn.Linear(hidden_dim, warp_dim)
        self.confidence_head = nn.Linear(hidden_dim, confidence_dim)

    def _build_local_window(self, prev_warp, feat_h, feat_w):
        """Build a (2r+1)^2 local sampling grid centred on prev_warp.

        Args:
            prev_warp: (B, Q, 2) current warp in normalised coords [-1, 1].
        Returns:
            (B, Q, K, 2) where K = (2r+1)^2.
        """
        radius = self.local_corr_radius
        if radius is None:
            return None

        ys = torch.linspace(
            -2 * radius / max(feat_h, 1),
            2 * radius / max(feat_h, 1),
            2 * radius + 1,
            device=prev_warp.device,
            dtype=prev_warp.dtype,
        )
        xs = torch.linspace(
            -2 * radius / max(feat_w, 1),
            2 * radius / max(feat_w, 1),
            2 * radius + 1,
            device=prev_warp.device,
            dtype=prev_warp.dtype,
        )
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        offsets = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, -1, 2)  # (1, 1, K, 2)
        return prev_warp.unsqueeze(2) + offsets  # (B, Q, K, 2)

    def forward(
        self,
        src_feat,  # (B, Q, C_feat)  — sampled source features at query_uv
        tgt_feat,  # (B, Q, C_feat)  — sampled target features at prev_warp
        tgt_map,  # (B, H_f, W_f, C_feat) — full target feature map for local window
        prev_warp,  # (B, Q, 2)
        prev_confidence,  # (B, Q, confidence_dim)
        query_uv,  # (B, Q, 2)
        feat_h,
        feat_w,
    ):
        src_hidden = self.src_proj(src_feat.float())  # (B, Q, hidden_dim)
        tgt_hidden = self.src_proj(tgt_feat.float())  # (B, Q, hidden_dim)
        disp_hidden = self.disp_proj((prev_warp - query_uv).float())  # (B, Q, hidden_dim)

        fused_parts = [
            src_hidden,  # (B, Q, hidden_dim)
            tgt_hidden,  # (B, Q, hidden_dim)
            disp_hidden,  # (B, Q, hidden_dim)
        ]

        if self.local_corr_radius is not None:
            local_grid = self._build_local_window(prev_warp, feat_h, feat_w)  # (B, Q, K, 2)
            tgt_window = _sample_bhwc(tgt_map, local_grid)  # (B, Q, K, C_feat)

            local_corr = torch.einsum("bqc,bqkc->bqk", src_feat.float(), tgt_window.float()) / math.sqrt(self.feat_dim)  # (B, Q, K)
            local_corr = self.corr_norm(local_corr)  # (B, Q, K)
            fused_parts.append(local_corr)

        fused = torch.cat(fused_parts, dim=-1)  # (B, Q, hidden_dim*4 + corr_dim)
        fused = self.fused_norm(fused)

        refined_hidden = self.fuse(fused)  # (B, Q, hidden_dim)

        norm = prev_warp.new_tensor((max(feat_w, 1), max(feat_h, 1))).view(1, 1, 2)

        delta_warp = self.warp_head(refined_hidden) / (self.refine_init * norm)  # (B, Q, 2)
        delta_confidence = self.confidence_head(refined_hidden)  # (B, Q, confidence_dim)

        warp = prev_warp + delta_warp  # (B, Q, 2)
        confidence = prev_confidence + delta_confidence  # (B, Q, confidence_dim)

        return warp, confidence


class QueryMatchRefine(nn.Module):
    """Sparse query-space analogue of RoMa's refiner stack.

    It keeps RoMa's multi-scale `FineFeatures` pyramid, but refines sparse query
    matches directly instead of refining a dense warp grid.
    """

    def __init__(
        self,
        refiners=None,
        refiner_features=None,
        hidden_dim=128,
        warp_dim=2,
        confidence_dim=None,
        fuse_mlp_nums=1,
        fuse_mlp_ratio=2,
        disp_dim=128,
        coarse_detach=False,
        refine_detach=False,
        mode="bilinear",
        debug=False,
        **kwargs,
    ):
        super().__init__()
        self.debug = debug
        self.mode = mode

        self.coarse_detach = coarse_detach
        self.refine_detach = refine_detach

        refiner_cfg = _update_cfg(Refiners.Cfg(), refiners)
        feature_cfg = _update_cfg(FineFeatures.Cfg(), refiner_features)
        self.refiner_features = FineFeatures(feature_cfg)

        self.refine_scales = tuple(sorted(refiner_cfg.refine_scales, reverse=True))
        block_cfgs = _get_refiner_feat_dims(refiner_cfg)
        self.confidence_dim = confidence_dim if confidence_dim is not None else refiner_cfg.confidence_dim
        self.blocks = nn.ModuleDict(
            {
                str(scale): _SparseQueryRefineBlock(
                    feat_dim=block_cfgs[scale]["feat_dim"],
                    hidden_dim=hidden_dim,
                    local_corr_radius=block_cfgs[scale]["local_corr_radius"],
                    warp_dim=warp_dim,
                    confidence_dim=self.confidence_dim,
                    fuse_mlp_nums=fuse_mlp_nums,
                    fuse_mlp_ratio=fuse_mlp_ratio,
                    disp_dim=disp_dim if isinstance(disp_dim, int) else disp_dim[scale],
                )
                for scale in self.refine_scales
            }
        )

    def _build_pair_features(self, feat, pair_idx):
        src_feat = torch.stack([feat[:, src] for src, _ in pair_idx], dim=1).view(feat.shape[0] * len(pair_idx), *feat.shape[2:])
        tgt_feat = torch.stack([feat[:, tgt] for _, tgt in pair_idx], dim=1).view(feat.shape[0] * len(pair_idx), *feat.shape[2:])
        return src_feat, tgt_feat

    def forward(
        self,
        coarse_results,
        query,
        rgb,
        pair_idx=None,
        meta_data=None,
        **kwargs,
    ):
        if pair_idx is None:
            pair_idx = [(0, 1)]
        if rgb.ndim != 5:
            raise ValueError("QueryMatchRefine expects rgb with shape (B, N, C, H, W).")

        B, N, _, H, W = rgb.shape
        num_pair = len(pair_idx)
        BP = B * num_pair

        rgb = _prepare_rgb_for_refiner(rgb.view(B * N, *rgb.shape[2:]))  # (B*N, 3, H, W)
        refine_feats = self.refiner_features(rgb)
        refine_feats = {scale: feat.view(B, N, *feat.shape[1:]) for scale, feat in refine_feats.items()}  # (B, N, H_f, W_f, C_feat)

        query_uv = query.uv * 2 - 1
        if query_uv.ndim == 4:
            # (B*P, Q, 2)
            query_grid = torch.stack([query_uv[:, pair[0]] for pair in pair_idx], dim=1).view(B * num_pair, *query_uv.shape[-2:])
        else:
            query_grid = query_uv.unsqueeze(0).expand(BP, -1, -1)  # (BP, Q, 2)

        warp = coarse_results["warp2d"]  # (BP, Q, 2)
        confidence = coarse_results["warp2d_confidence"]  # (BP, Q, confidence_dim)

        if self.coarse_detach:
            warp = warp.detach()
            confidence = confidence.detach()

        output = {}
        for scale in self.refine_scales:
            if scale not in refine_feats:
                continue

            if self.refine_detach:
                warp = warp.detach()
                confidence = confidence.detach()

            src_feat, tgt_feat = self._build_pair_features(refine_feats[scale], pair_idx)
            # src_feat, tgt_feat: (BP, H_f, W_f, C_feat)

            sampled_src = _sample_bhwc(src_feat, query_grid, mode=self.mode)  # (BP, Q, C_feat)

            with torch.no_grad():
                sampled_tgt = _sample_bhwc(tgt_feat, warp, mode=self.mode)  # (BP, Q, C_feat)

            feat_h, feat_w = src_feat.shape[1:3]
            warp, confidence = self.blocks[str(scale)](
                src_feat=sampled_src,  # (BP, Q, C_feat)
                tgt_feat=sampled_tgt,  # (BP, Q, C_feat)
                tgt_map=tgt_feat,  # (BP, H_f, W_f, C_feat)
                prev_warp=warp,  # (BP, Q, 2)
                prev_confidence=confidence,  # (BP, Q, confidence_dim)
                # query_uv=query_uv.unsqueeze(0).expand(BP, -1, -1),  # (BP, Q, 2)
                query_uv=query_grid,
                feat_h=feat_h,
                feat_w=feat_w,
            )

            output[f"refiner_{scale}_warp2d"] = warp
            output[f"refiner_{scale}_warp2d_confidence"] = confidence

        return output
