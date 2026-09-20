"""Sparse pair-query Dynamic Gaussian head for WFM / MVQuery pipelines."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_adapter import (
    SparsePairGaussianAdapter,
)
from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import (
    align_query_feats,
    align_sparse_query_geometry,
    build_frame_displacements,
    canonicalize_scene_flow,
    compute_query_motion_descriptor,
    ensure_bnv_cameras,
    find_pair_index,
    gather_pair_tensor,
    num_query_points,
    rgb_to_sh_dc,
    sample_rgb_at_query_uv,
    select_ref_frame,
    squeeze_query_points,
)


class SparsePairDynamicGaussianHead(nn.Module):
    """Predict Gaussian attributes at sparse query points on the reference frame.

    Geometry and motion come from existing pair decoder outputs.  The reference
    Gaussian means can either use ``warp3d`` directly or be reconstructed by
    unprojecting ``pair_depth`` with the reference camera:

    * ``warp3d @ (ref, ref)``  → reference-frame positions
    * ``pair_depth + K + w2c`` → reference-frame positions
    * ``warp3d_delta @ (ref, t)`` → per-frame displacements in the same frame

    The head learns opacity / scale / rotation / SH, conditioned on
    ``single_feats`` from ``PairCrossAttnAggregator`` and a motion descriptor
    derived from pair displacements.

    When ``use_rgb_feature_residual=True``, RGB features sampled at the same query
    UVs are added to ``single_feats`` before Gaussian attributes are predicted.

    ``direct_prediction=True`` follows AnySplat's Gaussian-attribute path:
    opacity, scale, rotation, and SH are predicted directly from features.  No
    Gaussian initializer, RGB-to-SH anchor, or attribute residual is used.
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 64,
        sh_degree: int = 0,
        motion_feat_dim: int = 32,
        enhanced_motion: bool = True,
        random_src_frame: bool = True,
        predict_attribute_delta: bool = False,
        attribute_delta_hidden_dim: int = 32,
        use_rgb_color_anchor: bool = False,
        use_rgb_feature_residual: bool = False,
        rgb_feature_dim: int = 128,
        direct_prediction: bool = False,
        direct_scale_factor: float = 1e-3,
        direct_scale_max: float = 0.3,
        use_sharp_zero_init: bool = False,
        geometry_source: str = "warp3d",
        min_scale_rate: float = 0.0,
        max_scale_rate: float = 10.0,
        init_opacity: float = 0.5,
        init_scale_px: float = 1.0,
        sharp_zero_activation: str = "sigmoid",
        param_factors: dict | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.sh_degree = sh_degree
        self.d_sh = (sh_degree + 1) ** 2
        self.motion_feat_dim = motion_feat_dim
        self.enhanced_motion = enhanced_motion
        self.random_src_frame = random_src_frame
        self.predict_attribute_delta = predict_attribute_delta
        self.use_rgb_color_anchor = use_rgb_color_anchor
        self.use_rgb_feature_residual = use_rgb_feature_residual
        self.direct_prediction = direct_prediction
        self.direct_scale_factor = direct_scale_factor
        self.direct_scale_max = direct_scale_max
        self.use_sharp_zero_init = use_sharp_zero_init
        self.geometry_source = geometry_source

        if geometry_source not in {"warp3d", "camera_depth"}:
            raise ValueError(
                "geometry_source must be 'warp3d' or 'camera_depth', "
                f"got {geometry_source!r}"
            )
        if direct_prediction and use_sharp_zero_init:
            raise ValueError("direct_prediction and use_sharp_zero_init are mutually exclusive")
        if direct_prediction and use_rgb_color_anchor:
            raise ValueError("direct_prediction does not support an RGB-to-SH color anchor")
        if direct_scale_factor <= 0:
            raise ValueError("direct_scale_factor must be positive")
        if direct_scale_max <= 0:
            raise ValueError("direct_scale_max must be positive")

        if self.use_rgb_feature_residual:
            if direct_prediction:
                # Match AnySplat's input_merger: one 7x7 RGB projection followed
                # by ReLU, added to the decoded Gaussian feature map.
                self.rgb_feature_encoder = nn.Sequential(
                    nn.Conv2d(3, in_dim, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=True),
                )
            else:
                self.rgb_feature_encoder = nn.Sequential(
                    nn.Conv2d(3, rgb_feature_dim, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(rgb_feature_dim, in_dim, kernel_size=1, stride=1, padding=0),
                )
                # Preserve the legacy identity-like initialization outside the
                # AnySplat-style direct-prediction path.
                nn.init.zeros_(self.rgb_feature_encoder[-1].weight)
                nn.init.zeros_(self.rgb_feature_encoder[-1].bias)
        else:
            self.rgb_feature_encoder = None

        motion_in = 6 if enhanced_motion else 1
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_in, motion_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(motion_feat_dim, motion_feat_dim),
        )
        nn.init.normal_(self.motion_proj[-1].weight, std=0.01)
        nn.init.zeros_(self.motion_proj[-1].bias)

        self._attr_non_sh_dim = 1 + 3 + 4
        self._attr_sh_dim = 3 * self.d_sh
        if use_sharp_zero_init:
            self.gaussian_adapter = SparsePairGaussianAdapter(
                sh_degree=sh_degree,
                mode="sharp_zero_init",
                min_scale_rate=min_scale_rate,
                max_scale_rate=max_scale_rate,
                init_opacity=init_opacity,
                init_scale_px=init_scale_px,
                sharp_zero_activation=sharp_zero_activation,
                param_factors=param_factors,
                use_rgb_color_anchor=use_rgb_color_anchor,
            )
            out_dim = self.gaussian_adapter.raw_param_dim
        else:
            self.gaussian_adapter = None
            out_dim = self._attr_non_sh_dim + self._attr_sh_dim

        self.attr_mlp = nn.Sequential(
            nn.Linear(in_dim + motion_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        if use_sharp_zero_init:
            # attr_mlp: Linear, ReLU, Linear, ReLU, Linear — penultimate Linear is [-3].
            nn.init.normal_(self.attr_mlp[-3].weight, std=0.01)
            nn.init.zeros_(self.attr_mlp[-3].bias)
            nn.init.zeros_(self.attr_mlp[-1].weight)
            nn.init.zeros_(self.attr_mlp[-1].bias)
        elif not direct_prediction:
            nn.init.normal_(self.attr_mlp[-1].weight, std=0.01)
            nn.init.constant_(self.attr_mlp[-1].bias, -2.0)
            if use_rgb_color_anchor and self._attr_sh_dim > 0:
                with torch.no_grad():
                    self.attr_mlp[-1].weight[:, self._attr_non_sh_dim:].zero_()
                    self.attr_mlp[-1].bias[self._attr_non_sh_dim:].zero_()

        sh_mask = torch.ones((self.d_sh,), dtype=torch.float32)
        for degree in range(1, sh_degree + 1):
            sh_mask[degree ** 2 : (degree + 1) ** 2] = 0.1 * (0.25 ** degree)
        self.register_buffer("sh_mask", sh_mask, persistent=False)

        if predict_attribute_delta:
            self.delta_mlp = nn.Sequential(
                nn.Linear(motion_in, attribute_delta_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(attribute_delta_hidden_dim, 1 + 3 + 4),
            )
            nn.init.zeros_(self.delta_mlp[-1].weight)
            nn.init.zeros_(self.delta_mlp[-1].bias)
        else:
            self.delta_mlp = None

    def _activate(
        self,
        raw: torch.Tensor,
        sh_anchor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        opacity = raw[..., :1].sigmoid()
        scale = raw[..., 1:4] - 4.0
        rotation = raw[..., 4:8]
        rotation = rotation / (rotation.norm(dim=-1, keepdim=True) + 1e-8)
        sh_raw = raw[..., self._attr_non_sh_dim:]
        sh = sh_raw.reshape(*sh_raw.shape[:-1], 3, self.d_sh).permute(0, 1, 3, 2)
        sh = sh * self.sh_mask.view(1, 1, -1, 1)
        if sh_anchor is not None:
            sh = sh.clone()
            sh[..., 0, :] = rgb_to_sh_dc(sh_anchor) + sh[..., 0, :]
        return opacity, scale, rotation, sh

    def _activate_direct(
        self,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode directly predicted attributes with AnySplat-style mappings."""
        opacity = raw[..., :1].sigmoid()
        scales = self.direct_scale_factor * F.softplus(raw[..., 1:4])
        scales = scales.clamp(min=1e-6, max=self.direct_scale_max)
        log_scale = scales.log()

        rotation = F.normalize(raw[..., 4:8], dim=-1, eps=1e-8)
        sh_raw = raw[..., self._attr_non_sh_dim:]
        sh = sh_raw.reshape(*sh_raw.shape[:-1], 3, self.d_sh).permute(0, 1, 3, 2)
        sh = sh * self.sh_mask.view(1, 1, -1, 1)
        return opacity, log_scale, rotation, sh

    def _sample_rgb_feature_at_query_uv(
        self,
        rgb: torch.Tensor,
        gaussian_query,
        ref_frame: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Encode reference RGB and sample features at Gaussian query UVs."""
        if self.rgb_feature_encoder is None:
            raise RuntimeError("RGB feature encoder is not enabled")

        if rgb.dim() == 5:
            rgb_ref = (rgb[:, ref_frame].float() + 1.0) * 0.5
        elif rgb.dim() == 4:
            rgb_ref = ((rgb[ref_frame].float() + 1.0) * 0.5).unsqueeze(0)
        else:
            raise ValueError(
                f"Expected rgb [B,N,3,H,W] or [N,3,H,W], got {tuple(rgb.shape)}"
            )

        uv = gaussian_query.uv.float()
        if uv.dim() == 2:
            uv = uv.unsqueeze(0)
        elif uv.dim() == 4:
            uv = uv[:, ref_frame]
        if uv.shape[0] == 1 and batch_size > 1:
            uv = uv.expand(batch_size, -1, -1)
        if uv.shape[0] != batch_size:
            raise ValueError(
                f"gaussian query batch mismatch: got {uv.shape[0]}, expected {batch_size}"
            )

        feature_map = self.rgb_feature_encoder(rgb_ref)
        grid = (uv * 2.0 - 1.0).unsqueeze(2)
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).permute(0, 2, 1).contiguous()

    @staticmethod
    def _query_uv_for_ref(
        gaussian_query,
        ref_frame: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if gaussian_query is None or not hasattr(gaussian_query, "uv"):
            raise ValueError("camera_depth geometry requires gaussian_query.uv")
        uv = gaussian_query.uv.to(device=device, dtype=torch.float32)
        if uv.dim() == 2:
            uv = uv.unsqueeze(0)
        elif uv.dim() == 4:
            uv = uv[:, ref_frame]
        elif uv.dim() != 3:
            raise ValueError(f"Unsupported gaussian query UV shape {tuple(uv.shape)}")
        if uv.shape[0] == 1 and batch_size > 1:
            uv = uv.expand(batch_size, -1, -1)
        if uv.shape[0] != batch_size:
            raise ValueError(
                f"gaussian query batch mismatch: got {uv.shape[0]}, expected {batch_size}"
            )
        return uv

    def _unproject_pair_depth(
        self,
        pair_outputs: dict,
        pair_idx: list,
        identity_idx: int,
        batch_size: int,
        num_views: int,
        ref_frame: int,
        gaussian_query,
        intrinsics: torch.Tensor | None,
        w2c: torch.Tensor | None,
        meta_data: dict,
    ) -> torch.Tensor:
        if "pair_depth" not in pair_outputs:
            raise ValueError("camera_depth geometry requires pair_outputs['pair_depth']")
        if intrinsics is None or w2c is None:
            raise ValueError("camera_depth geometry requires intrinsics and w2c")

        depth = gather_pair_tensor(
            pair_outputs["pair_depth"], pair_idx, batch_size, identity_idx,
        )
        if depth.dim() == 2:
            depth = depth.unsqueeze(-1)
        if depth.dim() != 3 or depth.shape[-1] != 1:
            raise ValueError(f"Expected pair depth [B,Q,1], got {tuple(depth.shape)}")

        uv = self._query_uv_for_ref(
            gaussian_query, ref_frame, batch_size, depth.device,
        )
        q = min(depth.shape[1], uv.shape[1])
        depth = depth[:, :q].float()
        uv = uv[:, :q]

        height = int(meta_data["input_height"][0])
        width = int(meta_data["input_width"][0])
        pixels = torch.stack(
            [
                uv[..., 0] * max(width - 1, 1) + 0.5,
                uv[..., 1] * max(height - 1, 1) + 0.5,
                torch.ones_like(uv[..., 0]),
            ],
            dim=-1,
        )

        intrinsics = ensure_bnv_cameras(intrinsics, num_views, (3, 3)).float()
        w2c = ensure_bnv_cameras(w2c, num_views, (4, 4)).float()
        k_ref = intrinsics[:, ref_frame]
        w2c_ref = w2c[:, ref_frame]

        rays = torch.linalg.solve(k_ref, pixels.transpose(1, 2)).transpose(1, 2)
        points_cam = rays * depth
        points_h = torch.cat([points_cam, torch.ones_like(depth)], dim=-1)
        c2w_ref = torch.linalg.inv(w2c_ref)
        return torch.bmm(points_h, c2w_ref.transpose(1, 2))[..., :3]

    def forward(
        self,
        single_feats: torch.Tensor,
        pair_outputs: dict,
        pair_idx: list,
        batch_size: int,
        meta_data: dict,
        num_views: int,
        rgb: torch.Tensor | None = None,
        gaussian_query=None,
        intrinsics: torch.Tensor | None = None,
        w2c: torch.Tensor | None = None,
    ) -> dict:
        if single_feats is None:
            return {}
        if "warp3d" not in pair_outputs:
            return {}

        ref_frame = select_ref_frame(
            num_views=num_views,
            pair_idx=pair_idx,
            random_src_frame=self.random_src_frame and self.training,
            training=self.training,
        )
        if ref_frame is None:
            return {}

        identity_idx = find_pair_index(pair_idx, ref_frame, ref_frame)
        if identity_idx is None:
            return {}

        feats = gather_pair_tensor(single_feats, pair_idx, batch_size, identity_idx)

        if self.geometry_source == "camera_depth":
            sparse_global_points = self._unproject_pair_depth(
                pair_outputs=pair_outputs,
                pair_idx=pair_idx,
                identity_idx=identity_idx,
                batch_size=batch_size,
                num_views=num_views,
                ref_frame=ref_frame,
                gaussian_query=gaussian_query,
                intrinsics=intrinsics,
                w2c=w2c,
                meta_data=meta_data,
            )
        else:
            sparse_global_points = gather_pair_tensor(
                pair_outputs["warp3d"], pair_idx, batch_size, identity_idx,
            )
        warp3d_delta = pair_outputs.get("warp3d_delta")
        displacements = build_frame_displacements(
            warp3d=pair_outputs["warp3d"],
            warp3d_delta=warp3d_delta,
            pair_idx=pair_idx,
            ref_frame=ref_frame,
            batch_size=batch_size,
            num_views=num_views,
        )
        displacements = canonicalize_scene_flow(displacements)
        sparse_global_points, displacements = align_sparse_query_geometry(
            sparse_global_points, displacements,
        )
        num_queries = num_query_points(sparse_global_points)
        feats = align_query_feats(feats, batch_size=batch_size, num_queries=num_queries)
        motion_desc = compute_query_motion_descriptor(
            displacements, ref_frame=ref_frame, enhanced_motion=self.enhanced_motion,
        )
        motion_feat = self.motion_proj(motion_desc)
        if motion_feat.dim() == 2:
            motion_feat = motion_feat.unsqueeze(0)
        if motion_feat.shape[1] == 1 and num_queries > 1:
            motion_feat = motion_feat.expand(-1, num_queries, -1)
        elif motion_feat.shape[1] != num_queries:
            raise ValueError(
                "SparsePairDynamicGaussianHead motion/query mismatch: "
                f"motion={tuple(motion_feat.shape)}, num_queries={num_queries}"
            )

        if feats.shape[0] != motion_feat.shape[0] or feats.shape[1] != motion_feat.shape[1]:
            raise ValueError(
                "SparsePairDynamicGaussianHead feature/query mismatch: "
                f"feats={tuple(feats.shape)}, motion={tuple(motion_feat.shape)}"
            )

        sh_anchor = None
        if self.use_rgb_color_anchor and rgb is not None and gaussian_query is not None:
            if hasattr(gaussian_query, "uv"):
                sh_anchor = sample_rgb_at_query_uv(
                    image=rgb,
                    query_uv=gaussian_query.uv,
                    ref_frame=ref_frame,
                    batch_size=batch_size,
                )
                if sh_anchor.shape[1] > num_queries:
                    sh_anchor = sh_anchor[:, :num_queries]

        if self.use_rgb_feature_residual and rgb is not None and gaussian_query is not None:
            rgb_query_feat = self._sample_rgb_feature_at_query_uv(
                rgb=rgb,
                gaussian_query=gaussian_query,
                ref_frame=ref_frame,
                batch_size=batch_size,
            )
            if rgb_query_feat.shape[1] > num_queries:
                rgb_query_feat = rgb_query_feat[:, :num_queries]
            if rgb_query_feat.shape != feats.shape:
                raise ValueError(
                    "SparsePairDynamicGaussianHead RGB feature/query mismatch: "
                    f"rgb_feat={tuple(rgb_query_feat.shape)}, feats={tuple(feats.shape)}"
                )
            feats = feats + rgb_query_feat

        fused = torch.cat([feats, motion_feat], dim=-1)
        raw = self.attr_mlp(fused)

        if self.use_sharp_zero_init:
            if intrinsics is None:
                return {}
            rgb01 = None
            if sh_anchor is not None:
                rgb01 = sh_anchor.float().clamp(0.0, 1.0)
            decoded = self.gaussian_adapter(
                raw_attr=raw,
                rgb_anchor=rgb01,
                means_ref=squeeze_query_points(sparse_global_points),
                intrinsics=intrinsics,
                ref_frame=ref_frame,
            )
            result = dict(
                **decoded,
                sparse_global_points=sparse_global_points,
                sparse_scene_flow=displacements,
                src_frame_idx=ref_frame,
            )
        elif self.direct_prediction:
            opacity, scale, rotation, sh = self._activate_direct(raw)
            result = dict(
                gs_opacity=opacity,
                gs_scale=scale,
                gs_rotation=rotation,
                gs_sh=sh,
                sparse_global_points=sparse_global_points,
                sparse_scene_flow=displacements,
                src_frame_idx=ref_frame,
            )
        else:
            opacity, scale, rotation, sh = self._activate(raw, sh_anchor=sh_anchor)
            result = dict(
                gs_opacity=opacity,
                gs_scale=scale,
                gs_rotation=rotation,
                gs_sh=sh,
                sparse_global_points=sparse_global_points,
                sparse_scene_flow=displacements,
                src_frame_idx=ref_frame,
            )

        if self.delta_mlp is not None:
            # Shared attribute offset in query space — not per-frame modulation.
            # Values are broadcast to all frames only for API compatibility with
            # ``build_sparse_dynamic_gaussians(..., target_frame_idx=...)``.
            delta_raw = self.delta_mlp(motion_desc)
            delta_raw = delta_raw.unsqueeze(1).expand(-1, num_views, -1, -1)
            result["gs_opacity_delta"] = delta_raw[..., :1]
            result["gs_scale_delta"] = delta_raw[..., 1:4]
            result["gs_rotation_delta"] = delta_raw[..., 4:8]

        return result
