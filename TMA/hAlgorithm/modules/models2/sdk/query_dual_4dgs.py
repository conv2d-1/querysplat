"""Dual-query 4DGS model: sparse traj queries for motion, dense UV for Gaussian head."""
from __future__ import annotations

import torch

from hAlgorithm.modules.models2.query_bank.query import BaseQuery
from hAlgorithm.modules.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from hAlgorithm.modules.models2.sdk.query import MVQuery6
from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import (
    build_frame_displacements,
    canonicalize_scene_flow,
)
from hAlgorithm.modules.pipelines2.utils.motion_utils import (
    prepare_decoupled_warp3d_delta_inplace,
)
from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import (
    compute_gaussian_query_shape,
)


class MVQueryDual4DGS(MVQuery6):
    """MVQuery6 variant with decoupled query groups for motion vs 4DGS.

    * **Motion query** (``query``): sparse trajectory UVs from the pipeline —
      drives pair decoder outputs used for warp3d / warp2d / warp3d_delta losses.
    * **Gaussian query** (``gaussian_query``): dense per-pixel UV grid —
      drives pair cross-attn + :class:`SparsePairDynamicGaussianHead` only.

    When ``gaussian_query`` is omitted and ``enable_dual_gaussian_query=True``,
    a dense UV grid is built automatically from ``meta_data``.  Use
    ``gaussian_query_scale > 1`` to sample more densely than encoder input
    while keeping UV normalized in ``[0, 1]``.
    """

    def __init__(
        self,
        enable_dual_gaussian_query: bool = True,
        gaussian_query_scale: float = 1.0,
        gaussian_query_patch_size: int = 14,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_dual_gaussian_query = enable_dual_gaussian_query
        self.gaussian_query_scale = float(gaussian_query_scale)
        self.gaussian_query_patch_size = int(gaussian_query_patch_size)

    @staticmethod
    def build_full_uv_query(
        meta_data: dict,
        device: torch.device,
        dtype: torch.dtype,
        gaussian_query_scale: float = 1.0,
        patch_size: int = 14,
    ) -> BaseQuery:
        """Build a dense ``H×W`` query grid in normalized UV space."""
        input_width = int(meta_data["input_width"][0])
        input_height = int(meta_data["input_height"][0])
        width, height = compute_gaussian_query_shape(
            input_width,
            input_height,
            gaussian_query_scale=gaussian_query_scale,
            patch_size=patch_size,
        )
        xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
        ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
        uu, vv = torch.meshgrid(xs, ys, indexing="xy")
        uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)
        uv[..., 0] /= max(width - 1, 1)
        uv[..., 1] /= max(height - 1, 1)
        return BaseQuery(uv=uv, full_uv=True, width=width, height=height)

    def _resolve_gaussian_query(
        self,
        gaussian_query: BaseQuery | None,
        motion_query: BaseQuery | None,
        meta_data: dict | None,
        rgb: torch.Tensor,
    ) -> BaseQuery | None:
        if gaussian_query is not None:
            return gaussian_query
        if not self.enable_dual_gaussian_query or meta_data is None:
            return None
        if motion_query is not None and motion_query.full_uv:
            return motion_query
        return self.build_full_uv_query(
            meta_data,
            rgb.device,
            rgb.dtype,
            gaussian_query_scale=self.gaussian_query_scale,
            patch_size=self.gaussian_query_patch_size,
        )

    def _resolve_gaussian_head_intrinsics(
        self,
        intrinsics: torch.Tensor | None,
        results: dict,
        rgb: torch.Tensor,
    ) -> torch.Tensor | None:
        """Wild infer has no GT K; sharp-zero-init head needs predicted intrinsics."""
        if intrinsics is not None:
            return intrinsics
        pose_enc = results.get("pose_enc")
        if pose_enc is None:
            return None
        if isinstance(pose_enc, (list, tuple)):
            pose_enc = pose_enc[-1]
        _, head_intrinsics = pose_encoding_to_extri_intri(
            pose_encoding=pose_enc.float(),
            image_size_hw=rgb.shape[-2:],
            build_intrinsics=True,
        )
        return head_intrinsics

    def _pair_forward_chunked(
        self,
        rgb: torch.Tensor,
        pair_idx: list,
        patch_tokens,
        time_token,
        query: BaseQuery,
        query_rgb,
        meta_data: dict,
    ) -> dict:
        """Run pair decoder on ``query``, optionally chunked by ``chunk_size``."""
        query_nums = query.query_nums
        if self.chunk_size is not None and self.chunk_size > 1 and query_nums > self.chunk_size:
            pair_results_list = []
            for q0 in range(0, query_nums, self.chunk_size):
                q1 = min(q0 + self.chunk_size, query_nums)
                if query.uv.ndim == 4:
                    chunk_query = BaseQuery(uv=query.uv[:, :, q0:q1])
                else:
                    chunk_query = BaseQuery(uv=query.uv[q0:q1])
                pair_results_list.append(
                    self.pair_forward(
                        rgb,
                        pair_idx=pair_idx,
                        patch_tokens=patch_tokens,
                        time_token=time_token,
                        query=chunk_query,
                        query_rgb=query_rgb,
                        meta_data=meta_data,
                    )
                )
            pair_results: dict = {}
            for key in pair_results_list[0].keys():
                pair_results[key] = torch.cat([res[key] for res in pair_results_list], dim=1)
            return pair_results
        return self.pair_forward(
            rgb,
            pair_idx=pair_idx,
            patch_tokens=patch_tokens,
            time_token=time_token,
            query=query,
            query_rgb=query_rgb,
            meta_data=meta_data,
        )

    def _reshape_pair_results(self, pair_results: dict, batch_size: int, num_pair: int) -> dict:
        public = {
            key: val.view(batch_size, num_pair, *val.shape[-2:])
            for key, val in pair_results.items()
            if not key.startswith("_")
        }
        return public

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
                    intrinsics=self._resolve_gaussian_head_intrinsics(
                        intrinsics, results, rgb,
                    ),
                )
                if gs_results:
                    results.update(gs_results)
                    ref_frame = int(gs_results["src_frame_idx"])
                    motion_for_disp = dict(motion_pair)
                    prepare_decoupled_warp3d_delta_inplace(motion_for_disp)
                    motion_disp = build_frame_displacements(
                        warp3d=motion_for_disp["warp3d"],
                        warp3d_delta=motion_for_disp.get("warp3d_delta"),
                        pair_idx=pair_idx,
                        ref_frame=ref_frame,
                        batch_size=b,
                        num_views=n,
                    )
                    results["motion_sparse_scene_flow"] = canonicalize_scene_flow(
                        motion_disp,
                    )
                results["gaussian_query"] = gaussian_query

        results["query"] = motion_query
        return results
