"""Dual-query 4DGS v2: Envision4D-inspired adapter + motion-single-source dense flow."""
from __future__ import annotations

import torch

from hAlgorithm.modules.models2.query_bank.query import BaseQuery
from hAlgorithm.modules.models2.sdk.query_dual_4dgs import MVQueryDual4DGS
from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import (
    build_frame_displacements,
    canonicalize_scene_flow,
    compute_focal_scale_multiplier,
    propagate_motion_flow_to_dense_queries,
)
from hAlgorithm.modules.pipelines2.utils.motion_utils import (
    prepare_decoupled_warp3d_delta_inplace,
)


class MVQueryDual4DGSV2(MVQueryDual4DGS):
    """V2 dual-query model with raw head + Gaussian adapter and optional motion flow source.

    ``gs_flow_source``:
      * ``"motion"`` (default): dense ``sparse_scene_flow`` is propagated from the sparse
        motion branch (single motion source for 4D warp).
      * ``"dense"``: keep dense pair-decoder displacements (ablation).
    """

    def __init__(
        self,
        gaussian_adapter=None,
        gs_flow_source: str = "motion",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gs_flow_source = gs_flow_source
        self.gaussian_adapter = self._instantiate_and_register(
            gaussian_adapter, "gaussian_adapter",
        )

    def _run_sparse_gaussian_head(
        self,
        pair_flat: dict,
        pair_idx,
        batch_size: int,
        num_views: int,
        meta_data: dict,
        rgb=None,
        gaussian_query=None,
        edge_mask=None,
        intrinsics=None,
        motion_query=None,
    ) -> dict:
        if self.sparse_gaussian_head is None or pair_idx is None:
            return {}

        query_feats = pair_flat.pop("_pair_query_feats", None)
        if query_feats is None:
            query_feats = pair_flat.pop("_single_feats", None)
        if query_feats is None:
            return {}

        prepare_decoupled_warp3d_delta_inplace(pair_flat)
        head_out = self.sparse_gaussian_head(
            single_feats=query_feats,
            pair_outputs=pair_flat,
            pair_idx=pair_idx,
            batch_size=batch_size,
            meta_data=meta_data,
            num_views=num_views,
            rgb=rgb,
            gaussian_query=gaussian_query,
            edge_mask=edge_mask,
            intrinsics=intrinsics,
            return_geometry_only=True,
        )
        if not head_out:
            return {}

        raw_attr = head_out.pop("raw_gaussian_attr", None)
        if raw_attr is None:
            return head_out

        focal_mult = None
        if intrinsics is not None and meta_data is not None:
            width = int(meta_data["input_width"][0])
            height = int(meta_data["input_height"][0])
            ref_frame = int(head_out.get("src_frame_idx", 0))
            focal_mult = compute_focal_scale_multiplier(
                intrinsics,
                width=width,
                height=height,
                frame_idx=ref_frame,
                multiplier=getattr(self.gaussian_adapter, "focal_scale_multiplier", 0.1),
            )

        if self.gaussian_adapter is not None:
            ref_frame = int(head_out.get("src_frame_idx", 0))
            decoded = self.gaussian_adapter(
                raw_attr,
                rgb_anchor=head_out.get("rgb_anchor"),
                depth_q=head_out.get("depth_q"),
                focal_mult=focal_mult,
                means_ref=head_out.get("sparse_global_points"),
                intrinsics=intrinsics,
                ref_frame=ref_frame,
            )
            head_out.update(decoded)
        else:
            opacity, scale, rotation, sh = self.sparse_gaussian_head._activate(
                raw_attr, sh_anchor=head_out.get("rgb_anchor"),
            )
            head_out.update(
                gs_opacity=opacity,
                gs_scale=scale,
                gs_rotation=rotation,
                gs_sh=sh,
            )

        return head_out

    def _apply_motion_flow_source(
        self,
        head_out: dict,
        motion_disp: torch.Tensor,
        motion_query,
        gaussian_query,
        batch_size: int,
    ) -> dict:
        if self.gs_flow_source != "motion" or gaussian_query is None:
            return head_out
        ref_frame = int(head_out["src_frame_idx"])
        g_uv = gaussian_query.uv
        if g_uv.dim() == 4:
            g_uv = g_uv[:, ref_frame]
        head_out["sparse_scene_flow"] = propagate_motion_flow_to_dense_queries(
            motion_flow=motion_disp,
            motion_uv=motion_query.uv if motion_query is not None else g_uv,
            dense_uv=g_uv,
            batch_size=batch_size,
            ref_frame=ref_frame,
        )
        return head_out

    def forward(
        self,
        rgb,
        query=None,
        gaussian_query=None,
        query_rgb=None,
        edge_mask=None,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        c2w=None,
        ray_world=None,
        rgb_mask=None,
        meta_data=None,
        pair_idx=None,
        time_idx=None,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        b, n = self.get_batch_views(rgb)

        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb,
            scale=scale,
            prompt_depth=prompt_depth,
            intrinsics=intrinsics,
            ray_directions=ray_directions,
            w2c=w2c,
            c2w=c2w,
            ray_world=ray_world,
            rgb_mask=rgb_mask,
            meta_data=meta_data,
        )

        if isinstance(patch_features, (list, tuple)):
            patch_tokens = [features[:, :, patch_start_idx:] for features in patch_features]
        else:
            patch_tokens = [patch_features[:, :, patch_start_idx:]]

        motion_query = query
        if motion_query is None:
            motion_query = self.query_banck(rgb, edge_mask=edge_mask, meta_data=meta_data)

        gaussian_query = self._resolve_gaussian_query(
            gaussian_query, motion_query, meta_data, rgb,
        )

        query_nums = motion_query.query_nums

        if self.chunk_size is not None and self.chunk_size > 1 and query_nums > self.chunk_size:
            results_list = []
            for q0 in range(0, query_nums, self.chunk_size):
                q1 = min(q0 + self.chunk_size, query_nums)
                if motion_query.uv.ndim == 4:
                    chunk_query = BaseQuery(uv=motion_query.uv[:, :, q0:q1])
                else:
                    chunk_query = BaseQuery(uv=motion_query.uv[q0:q1])
                results_list.append(
                    self.single_forward(rgb, patch_tokens, prompt_depth, chunk_query, query_rgb, meta_data)
                )
            results = {}
            for key in results_list[0].keys():
                results[key] = torch.cat([res[key] for res in results_list], dim=1)
        else:
            results = self.single_forward(
                rgb, patch_tokens, prompt_depth, motion_query, query_rgb, meta_data,
            )

        results = {key: val.view(b, n, *val.shape[-2:]) for key, val in results.items()}

        if self.training and self.decoder_fp32:
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                extra_results = self.decoder(
                    b=b, n=n, patch_features=patch_features, pos=pos,
                    patch_start_idx=patch_start_idx, prompt_depth=prompt_depth,
                    query_points=None, meta_data=meta_data,
                )
        else:
            extra_results = self.decoder(
                b=b, n=n, patch_features=patch_features, pos=pos,
                patch_start_idx=patch_start_idx, prompt_depth=prompt_depth,
                query_points=None, meta_data=meta_data,
            )
        results.update(extra_results)

        if self.ffgs is not None and motion_query.full_uv and results.get("depth") is not None:
            gs_results = self.ffgs(
                patch_features=patch_features,
                patch_start_idx=patch_start_idx,
                rgb=rgb,
                depth=results["depth"],
                depth_width=motion_query.width,
                depth_height=motion_query.height,
                intrinsics=intrinsics,
                w2c=w2c,
                c2w=c2w,
                scale=scale,
            )
            results.update(gs_results)

        if pair_idx is not None:
            num_pair = len(pair_idx)

            if self.time_token_index is not None:
                if isinstance(patch_features, (list, tuple)):
                    time_token = patch_features[-1][:, :, self.time_token_index]
                else:
                    time_token = patch_features[:, :, self.time_token_index]
            else:
                time_token = None

            motion_pair = self._pair_forward_chunked(
                rgb, pair_idx, patch_tokens, time_token, motion_query, query_rgb, meta_data,
            )
            motion_public = self._reshape_pair_results(motion_pair, b, num_pair)
            motion_public["pair_idx"] = pair_idx
            results.update(motion_public)

            if gaussian_query is not None and self.sparse_gaussian_head is not None:
                gs_pair = self._pair_forward_chunked(
                    rgb, pair_idx, patch_tokens, time_token, gaussian_query, query_rgb, meta_data,
                )
                gs_results = self._run_sparse_gaussian_head(
                    pair_flat=gs_pair,
                    pair_idx=pair_idx,
                    batch_size=b,
                    num_views=n,
                    meta_data=meta_data,
                    rgb=rgb,
                    gaussian_query=gaussian_query,
                    edge_mask=edge_mask,
                    intrinsics=intrinsics,
                    motion_query=motion_query,
                )
                if gs_results:
                    ref_frame = int(gs_results["src_frame_idx"])
                    motion_for_disp = dict(motion_pair)
                    prepare_decoupled_warp3d_delta_inplace(motion_for_disp)
                    motion_disp = canonicalize_scene_flow(
                        build_frame_displacements(
                            warp3d=motion_for_disp["warp3d"],
                            warp3d_delta=motion_for_disp.get("warp3d_delta"),
                            pair_idx=pair_idx,
                            ref_frame=ref_frame,
                            batch_size=b,
                            num_views=n,
                        ),
                    )
                    gs_results = self._apply_motion_flow_source(
                        gs_results, motion_disp, motion_query, gaussian_query, b,
                    )
                    results.update(gs_results)
                    results["motion_sparse_scene_flow"] = motion_disp
                results["gaussian_query"] = gaussian_query

        results["query"] = motion_query
        return results
