"""Envision4D-inspired sparse pair Gaussian head (raw prediction + fused cues)."""
from __future__ import annotations

import torch
import torch.nn as nn

from hAlgorithm.modules.models2.head.sparse_pair_gaussian_head import (
    SparsePairDynamicGaussianHead,
)
from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import (
    find_pair_index,
    gather_pair_tensor,
    sample_rgb_at_query_uv,
    sample_scalar_at_query_uv,
    select_ref_frame,
)


class SparsePairDynamicGaussianHeadV2(SparsePairDynamicGaussianHead):
    """Predict raw Gaussian attributes with RGB / depth / edge fused features.

    Decoding into final Gaussian parameters is handled by
    :class:`SparsePairGaussianAdapter` in ``MVQueryDual4DGSV2``.
    """

    def __init__(
        self,
        fuse_depth_in_head: bool = True,
        fuse_edge_in_head: bool = True,
        predict_raw_only: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("use_rgb_color_anchor", True)
        kwargs.setdefault("predict_attribute_delta", False)
        super().__init__(**kwargs)
        self.fuse_depth_in_head = fuse_depth_in_head
        self.fuse_edge_in_head = fuse_edge_in_head
        self.predict_raw_only = predict_raw_only

        extra_in = 0
        if self.use_rgb_color_anchor:
            extra_in += 3
        if fuse_depth_in_head:
            extra_in += 1
        if fuse_edge_in_head:
            extra_in += 1
        if extra_in > 0:
            old_in = self.attr_mlp[0].in_features
            old_linear = self.attr_mlp[0]
            self.attr_mlp[0] = nn.Linear(old_in + extra_in, old_linear.out_features)
            with torch.no_grad():
                self.attr_mlp[0].weight[:, :old_in] = old_linear.weight
                self.attr_mlp[0].weight[:, old_in:] = 0.0
                self.attr_mlp[0].bias.copy_(old_linear.bias)
            self._attr_mlp_base_in = old_in
        else:
            self._attr_mlp_base_in = self.attr_mlp[0].in_features

    @staticmethod
    def _zeros_queries(like: torch.Tensor, dim: int) -> torch.Tensor:
        return like.new_zeros(like.shape[0], like.shape[1], dim)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_key = prefix + "attr_mlp.0.weight"
        if weight_key in state_dict:
            ckpt_weight = state_dict[weight_key]
            model_weight = self.attr_mlp[0].weight
            if ckpt_weight.shape != model_weight.shape:
                expanded = model_weight.detach().clone()
                copy_in = min(ckpt_weight.shape[1], model_weight.shape[1])
                copy_out = min(ckpt_weight.shape[0], model_weight.shape[0])
                expanded[:copy_out, :copy_in] = ckpt_weight[:copy_out, :copy_in]
                if copy_in < model_weight.shape[1]:
                    expanded[:, copy_in:] = 0.0
                state_dict[weight_key] = expanded

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        edge_mask: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        return_geometry_only: bool = False,
    ) -> dict:
        if single_feats is None or "warp3d" not in pair_outputs:
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

        from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import (
            align_query_feats,
            align_sparse_query_geometry,
            build_frame_displacements,
            canonicalize_scene_flow,
            compute_query_motion_descriptor,
            num_query_points,
        )

        feats = gather_pair_tensor(single_feats, pair_idx, batch_size, identity_idx)
        sparse_global_points = gather_pair_tensor(
            pair_outputs["warp3d"], pair_idx, batch_size, identity_idx,
        )
        displacements = build_frame_displacements(
            warp3d=pair_outputs["warp3d"],
            warp3d_delta=pair_outputs.get("warp3d_delta"),
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
                "SparsePairDynamicGaussianHeadV2 motion/query mismatch: "
                f"motion={tuple(motion_feat.shape)}, num_queries={num_queries}"
            )

        fuse_parts = [feats, motion_feat]
        has_uv = gaussian_query is not None and hasattr(gaussian_query, "uv")
        rgb_anchor = None
        depth_q = None
        edge_q = None

        if self.use_rgb_color_anchor:
            if has_uv and rgb is not None:
                rgb_anchor = sample_rgb_at_query_uv(
                    image=rgb,
                    query_uv=gaussian_query.uv,
                    ref_frame=ref_frame,
                    batch_size=batch_size,
                )
                if rgb_anchor.shape[1] > num_queries:
                    rgb_anchor = rgb_anchor[:, :num_queries]
            else:
                rgb_anchor = self._zeros_queries(feats, 3)
            fuse_parts.append(rgb_anchor.to(feats.dtype))

        if self.fuse_depth_in_head:
            if has_uv and "pair_depth" in pair_outputs:
                pair_depth = gather_pair_tensor(
                    pair_outputs["pair_depth"], pair_idx, batch_size, identity_idx,
                )
                while pair_depth.dim() < 3:
                    pair_depth = pair_depth.unsqueeze(-1)
                depth_map = pair_depth.transpose(1, 2)
                depth_q = sample_scalar_at_query_uv(
                    depth_map,
                    gaussian_query.uv,
                    ref_frame=ref_frame,
                    batch_size=batch_size,
                )
                if depth_q.shape[1] > num_queries:
                    depth_q = depth_q[:, :num_queries]
            else:
                depth_q = self._zeros_queries(feats, 1)
            fuse_parts.append(depth_q.to(feats.dtype))

        if self.fuse_edge_in_head:
            if has_uv and edge_mask is not None:
                edge_q = sample_scalar_at_query_uv(
                    edge_mask.float(),
                    gaussian_query.uv,
                    ref_frame=ref_frame,
                    batch_size=batch_size,
                )
                if edge_q.shape[1] > num_queries:
                    edge_q = edge_q[:, :num_queries]
            else:
                edge_q = self._zeros_queries(feats, 1)
            fuse_parts.append(edge_q.to(feats.dtype))

        fused = torch.cat(fuse_parts, dim=-1)
        raw_attr = self.attr_mlp(fused)

        result = dict(
            raw_gaussian_attr=raw_attr,
            sparse_global_points=sparse_global_points,
            sparse_scene_flow=displacements,
            src_frame_idx=ref_frame,
            rgb_anchor=rgb_anchor,
            depth_q=depth_q,
            edge_q=edge_q,
        )
        if return_geometry_only:
            return result

        if self.predict_raw_only:
            return result

        opacity, scale, rotation, sh = self._activate(raw_attr, sh_anchor=rgb_anchor)
        result.update(
            gs_opacity=opacity,
            gs_scale=scale,
            gs_rotation=rotation,
            gs_sh=sh,
        )
        return result
