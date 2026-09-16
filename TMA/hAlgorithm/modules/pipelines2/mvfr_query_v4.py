import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import random

from hAlgorithm.modules.models2.external.depth_anything_3.utils.alignment import least_squares_scale_scalar
from hAlgorithm.modules.models2.gaussians.infinidepth_ply import export_ply
from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from hAlgorithm.modules.pipelines2.utils.outputs import Track3DOutput
from hAlgorithm.modules.pipelines2.utils.render_util import render_video_interpolation
from hAlgorithm.modules.pipelines2.utils.visualize_dynamic import vis_motion_results
from hAlgorithm.modules.pipelines2.utils.visualize_mv import (
    save_video,
    vis_match_results,
    vis_render,
)
from hAlgorithm.utils import instantiate_from_config

# 解耦 warp3d：训练/推理中在算 motion loss 或取分支前调用 ``prepare_decoupled_warp3d_delta_inplace``
# （见 motion_utils），将 decoupled 输出合成为 ``warp3d_delta``，与球坐标式 head 的最终位移约定一致。
from hAlgorithm.modules.pipelines2.utils.motion_utils import (
    prepare_decoupled_warp3d_delta_inplace,
    scalar,
)
from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import (
    resolve_sparse_dgs_render_shape,
    scale_bnv_intrinsics,
)

from .mvfr_query_v1 import MVFRQueryPipeline


class WFMQueryPipeline(MVFRQueryPipeline):
    """基于 Query 的多视角前馈重建 Pipeline (V3 — Local + Global + Camera + FFGS + Match + Motion).

    在 V1 (仅 Local) 的基础上融合 V2 (Global + Camera) 和 V3 (Match) 的能力:
        - **Local depth loss** (Part1): 同 V1, 基于 query 点的局部深度预测。
        - **Global points loss** (Part2): 全局坐标系下的 3D 点预测监督。
        - **Camera loss** (Part3): 基于 pose_encoding 的相机外参/内参监督。
        - **Gaussian rendering loss** (Part4): Gaussian splatting 渲染的 RGB + depth 重建。
        - **Match warp loss** (Part5): 利用 GT 深度 + 相机参数计算视角间 warp, 监督匹配预测。
        - **Pose decoding**: 推理时将 pose_encoding 解码为外参 + 内参, 支持
          可选的 normalize_cameras (以第 0 视角为参考系)。
        - **Subclass hooks**: _infer_extra_model_kwargs / _on_infer_model_results
          供子类注入额外推理逻辑。

    Args:
        query_match_loss: match warp loss 的配置字典, None 时禁用 match loss。
        pair_idx: 视角配对索引列表, 如 [(0,1), (0,2)], 用于计算 GT warp。
        warp3d_delta_direction_loss: 解耦位移上的方向辅助 loss 配置（如 ``SparseDirectionLoss``），
            在 ``prepare_decoupled_warp3d_delta_inplace`` 之后、对 canonical 的
            ``warp3d_delta_direction`` / ``warp3d_delta_magnitude`` 计算。强度由该 loss 配置里的
            ``loss_weight``（可按数据集 ``name`` 取 dict）决定；本 pipeline 仅再乘
            ``task_weight["motion"]``，与 mask / 3D motion 等分支一致。

    解耦 scene flow:
        ``warp3d``（绝对点）与 ``warp3d_delta``（位移）任务独立。若需由
        ``warp3d_delta_direction`` / ``warp3d_delta_magnitude`` 合成 ``warp3d_delta``，调用
        ``prepare_decoupled_warp3d_delta_inplace``。方向/模长的归一化与激活由 head（如
        ``mlp_head`` 的 ``norm`` / ``softplus``）负责；``motion_utils`` 仅对 canonical 的
        ``warp3d_delta_*`` 做乘积合成，避免把 world-point 分支的 key 混入 scene-flow 分支。
    """

    def __init__(
        self,
        static_dataset_names=None,
        dynamic_dataset_names=None,
        sparse_dataset_names=None,
        motion_extrinsics_name=None,
        pair_mode=None,
        global_pair_mode=None,
        query_match_loss=None,
        query_motion2d_loss=None,
        query_motion3d_loss=None,
        query_motion3d_delta_loss=None,
        query_motion_mask_loss=None,
        warp3d_delta_direction_loss=None,
        motion_dynamic_threshold=None,
        motion_dynamic_scale=1.0,
        motion_static_scale=1.0,
        motion_3d_train_novis=False,
        motion_2d_train_novis=False,
        motion_novis_scale=1.0,
        warp3d_delta_to_mask=False,
        align_corners=False,
        sparse_dynamic_gaussian_render_loss=None,
        motion_gs_displacement_consistency_loss=None,
        gs_xyz_offset_loss=None,
        dgs_scale_dropout_prob=0.0,
        dgs_render_normalized_only=False,
        **kwargs,
    ):
        super(WFMQueryPipeline, self).__init__(**kwargs)

        self.static_dataset_names = static_dataset_names
        self.dynamic_dataset_names = dynamic_dataset_names
        self.sparse_dataset_names = sparse_dataset_names

        self.pair_mode = pair_mode
        self.global_pair_mode = global_pair_mode
        self.query_match_loss = instantiate_from_config(query_match_loss) if query_match_loss is not None else None

        self.motion_extrinsics_name = motion_extrinsics_name
        self.query_motion2d_loss = instantiate_from_config(query_motion2d_loss) if query_motion2d_loss is not None else None
        self.query_motion3d_loss = instantiate_from_config(query_motion3d_loss) if query_motion3d_loss is not None else None
        self.query_motion3d_delta_loss = instantiate_from_config(query_motion3d_delta_loss) if query_motion3d_delta_loss is not None else None
        self.query_motion_mask_loss = instantiate_from_config(query_motion_mask_loss) if query_motion_mask_loss is not None else None
        self._warp3d_delta_direction_loss = instantiate_from_config(warp3d_delta_direction_loss) if warp3d_delta_direction_loss is not None else None

        self.save_output_cfg.setdefault("save_match", self.save_output_cfg["save_everything"])
        self.save_output_cfg.setdefault("save_motion", self.save_output_cfg["save_everything"])
        self.save_output_cfg.setdefault("save_motion_rerun", self.save_output_cfg["save_motion"])

        self.motion_dynamic_threshold = motion_dynamic_threshold
        self.motion_dynamic_scale = motion_dynamic_scale
        self.motion_static_scale = motion_static_scale
        self.motion_3d_train_novis = motion_3d_train_novis
        self.motion_2d_train_novis = motion_2d_train_novis
        self.motion_novis_scale = motion_novis_scale
        self.warp3d_delta_to_mask = warp3d_delta_to_mask
        self.align_corners = align_corners
        self.sparse_dynamic_gaussian_render_loss = (
            instantiate_from_config(sparse_dynamic_gaussian_render_loss)
            if sparse_dynamic_gaussian_render_loss is not None else None
        )
        self.motion_gs_displacement_consistency_loss = (
            instantiate_from_config(motion_gs_displacement_consistency_loss)
            if motion_gs_displacement_consistency_loss is not None else None
        )
        self.gs_xyz_offset_loss = (
            instantiate_from_config(gs_xyz_offset_loss)
            if gs_xyz_offset_loss is not None else None
        )
        self.dgs_scale_dropout_prob = float(dgs_scale_dropout_prob or 0.0)
        self.dgs_render_normalized_only = bool(dgs_render_normalized_only)
        if (
            self.dgs_render_normalized_only
            and self.sparse_dynamic_gaussian_render_loss is not None
            and not getattr(
                self.sparse_dynamic_gaussian_render_loss,
                "render_in_normalized_space",
                False,
            )
        ):
            self.sparse_dynamic_gaussian_render_loss.render_in_normalized_space = True
        self.save_output_cfg.setdefault("save_4dgs_results", False)

    def get_camera_type(self, meta_data):
        """从 meta_data 中提取相机模型类型, 默认 PINHOLE."""
        if "camera_type" in meta_data:
            camera_type = meta_data["camera_type"][0][0]
        else:
            camera_type = "PINHOLE"
        return camera_type

    def render_gaussians(self, gaussians, intrinsics, w2c, image_shape):
        """Render pixel-aligned Gaussians via gsplat rasterization.

        Args:
            results: dict with means [BN, P, 3], harmonics [BN, P, 3, K],
                    opacities [BN, P], scales [BN, P, 3], rotations [BN, P, 4].
            intrinsics: [B, N, 3, 3] pixel-space camera intrinsics.
            w2c: [B, N, 4, 4] world-to-camera or None (identity used).
            image_shape: (H, W) render target size.

        Returns:
            render_rgb: [BN, 3, H, W] rendered color in [0, 1].
            render_depth: [BN, H, W] rendered expected depth.
        """
        from gsplat.rendering import rasterization

        means = gaussians["means"]
        harmonics = gaussians["harmonics"]
        opacities = gaussians["opacities"]
        scales = gaussians["scales"]
        rotations = gaussians["rotations"]

        bn = means.shape[0]
        h, w = image_shape
        d_sh = harmonics.shape[-1]
        sh_degree = int(d_sh**0.5) - 1

        colors = harmonics.permute(0, 1, 3, 2).contiguous()

        Ks = intrinsics.reshape(bn, 3, 3).detach()
        if w2c is not None:
            viewmats = w2c.reshape(bn, 4, 4).detach()
        else:
            viewmats = torch.eye(4, device=means.device, dtype=means.dtype)
            viewmats = viewmats.unsqueeze(0).expand(bn, -1, -1).contiguous()

        means = means.float()
        rotations = rotations.float()
        scales = scales.float()
        opacities = opacities.float()
        colors = colors.float()
        viewmats = viewmats.float()
        Ks = Ks.float()

        render_colors_list, render_depths_list = [], []
        try:
            with torch.autocast("cuda", enabled=False):
                for i in range(bn):
                    rc, _, _ = rasterization(
                        means=means[i],
                        quats=rotations[i],
                        scales=scales[i],
                        opacities=opacities[i],
                        colors=colors[i],
                        viewmats=viewmats[i : i + 1],
                        Ks=Ks[i : i + 1],
                        width=w,
                        height=h,
                        render_mode="RGB+ED",
                        sh_degree=sh_degree,
                    )
                    render_colors_list.append(rc[..., :3])
                    render_depths_list.append(rc[..., 3:])
        except Exception as e:
            logging.warning(f"Gaussian rendering failed: {e}")
            return None, None

        render_rgb = torch.cat(render_colors_list, dim=0).permute(0, 3, 1, 2).clamp(0, 1)
        render_depth = torch.cat(render_depths_list, dim=0).squeeze(-1)

        return render_rgb, render_depth

    def get_edge_mask(self, batch):
        """从 batch 中提取边缘 mask, 用于训练时辅助遮挡处理."""
        edge_mask = None
        if self.edge_mask_name is not None and self.edge_mask_name in batch:
            edge_mask = batch[self.edge_mask_name].to(device=self.device)
        return edge_mask

    def get_pair_idx(self, meta_data):
        """根据 pair_mode 和视角数生成视角配对索引列表.

        Args:
            meta_data: 含 'frames' 和 'views' 字段的元数据字典.

        Returns:
            list[tuple[int, int]] | None: 视角配对列表, pair_mode 为 None 时返回 None.
        """
        if self.pair_mode is None:
            return None

        num_views = int(meta_data["frames"][0]) * int(meta_data["views"][0])

        if self.pair_mode == 0:
            return [(0, i) for i in range(1, num_views)]

        if self.pair_mode == 1:
            return [(i, j) for i in range(num_views) for j in range(i + 1, num_views)]

        if self.pair_mode == 2:
            if self.training:
                return [(0, i) for i in range(0, num_views)]
            else:
                return [(0, i) for i in range(1, num_views)]

        if self.pair_mode == 3:
            return [(i, i + 1) for i in range(num_views - 1)]

        if self.pair_mode == 4:
            if self.training:
                return [(0, i) for i in range(0, num_views)] + [(i, 0) for i in range(1, num_views)]
            else:
                return [(0, i) for i in range(1, num_views)]

        if self.pair_mode == 5:
            if self.training:
                return [(i, j) for i in range(num_views) for j in range(num_views)]
            else:
                return [(0, i) for i in range(1, num_views)]

        if self.pair_mode == 6:
            if self.training:
                all_pairs = [(i, j) for i in range(num_views) for j in range(num_views) if j != i]
                selected_pairs = random.sample(all_pairs, num_views)
                return selected_pairs
            else:
                return [(0, i) for i in range(1, num_views)]

        if self.pair_mode == 7:
            if self.training:
                all_pairs = [(i, j) for i in range(num_views) for j in range(i + 1, num_views) if j != i]
                selected_pairs = random.sample(all_pairs, num_views)
                return selected_pairs
            else:
                return [(0, i) for i in range(1, num_views)]

        if self.pair_mode == 8:
            if self.training:
                src = list(range(num_views))
                tgt = list(range(num_views))
                random.shuffle(tgt)
                selected_pairs = [(s, t) for s, t in zip(src, tgt)]
                return selected_pairs
            else:
                return [(0, i) for i in range(1, num_views)]

        if self.pair_mode == 9:
            if self.training:
                src = list(range(num_views))
                selected_src = random.sample(src, 1)[0]
                selected_pairs = [(selected_src, i) for i in src if i != selected_src]
                return selected_pairs
            else:
                return [(0, i) for i in range(1, num_views)]

        if self.pair_mode == 10:
            if self.training:
                all_pairs = [(i, j) for i in range(num_views) for j in range(num_views)]
                selected_pairs = random.sample(all_pairs, num_views)
                return selected_pairs
            else:
                return [(0, i) for i in range(1, num_views)]

        if self.pair_mode == 11:
            if self.training:
                all_pairs = [(i, j) for i in range(num_views) for j in range(num_views) if j != i]
                selected_pairs = random.sample(all_pairs, num_views)
                return [(0, 0)] + selected_pairs
            else:
                return [(0, i) for i in range(1, num_views)]
        
        if self.pair_mode == 12:
            if self.training:
                all_pairs = [(i, j) for i in range(num_views) for j in range(num_views) if j != i]
                selected_pairs = random.sample(all_pairs, num_views)
                return [(i, i) for i in range(num_views)] + selected_pairs
            else:
                return [(0, i) for i in range(1, num_views)]
        
        if self.pair_mode == 13:
            if self.training:
                all_pairs = [(i, j) for i in range(num_views) for j in range(num_views) if j != i]
                selected_pairs = random.sample(all_pairs, num_views)

                base_pairs = [(i, i) for i in range(1, num_views)]
                selected_base_pairs = random.sample(base_pairs, 1)

                return [(0, 0)] + selected_base_pairs + selected_pairs
            else:
                return [(0, i) for i in range(1, num_views)]

        return None

    def get_global_pair_idx(self, meta_data):
        if self.global_pair_mode is None:
            return None

        if self.global_pair_mode == 0:
            return [(0, 0)]

        num_views = int(meta_data["frames"][0]) * int(meta_data["views"][0])

        if self.global_pair_mode == 1:
            return [(i, i) for i in range(num_views)]
        
        return None

    def _ensure_ref_identity_pairs(self, pair_idx, global_pair_idx):
        """Append ``(src, src)`` identity pairs for each source in ``pair_idx``.

        Needed for sparse track export / WorldTrack eval: ``track_3d[0]`` carries
        frame-0 ``warp3d``, and ``src_points`` on ``(0, t)`` is gathered from it.
        """
        if not pair_idx:
            return pair_idx, global_pair_idx
        infer_pair_idx = list(pair_idx)
        extra_global = list(global_pair_idx) if global_pair_idx is not None else []
        for src in sorted({int(p[0]) for p in infer_pair_idx}):
            identity = (src, src)
            if identity not in infer_pair_idx:
                infer_pair_idx.append(identity)
                extra_global.append(identity)
        if extra_global:
            return infer_pair_idx, extra_global
        return infer_pair_idx, global_pair_idx

    def _augment_pair_idx_for_dgs(self, pair_idx, num_views: int):
        """Ensure identity pairs ``(i, i)`` exist for sparse 4DGS geometry."""
        if pair_idx is None or self.sparse_dynamic_gaussian_render_loss is None:
            return pair_idx
        augmented = list(pair_idx)
        existing = {(int(s), int(t)) for s, t in augmented}
        for i in range(num_views):
            identity = (i, i)
            if identity not in existing:
                augmented.append(identity)
                existing.add(identity)
        return augmented

    def get_dgs_render_loss_kwargs(
        self,
        image,
        target_local_depth=None,
        target_depth_mask=None,
        scale=None,
        name=None,
        **kwargs,
    ) -> dict:
        """Pass GT geometry into DGS render loss (FFGS-style ref visibility masks)."""
        out = {}
        if target_local_depth is not None:
            out["target_local_depth"] = target_local_depth
        if target_depth_mask is not None:
            out["target_depth_mask"] = target_depth_mask
        return out

    def _render_sparse_dgs_infer(self, results, intrinsics, extrinsics, scale, height, width, image=None):
        """Render sparse pair 4DGS for all clip frames during inference."""
        if self.sparse_dynamic_gaussian_render_loss is None:
            return None
        if not all(k in results for k in ("gs_opacity", "sparse_global_points", "sparse_scene_flow")):
            return None

        if self.dgs_render_normalized_only:
            scale = None

        return self.sparse_dynamic_gaussian_render_loss.render_frames(
            results=results,
            intrinsics=intrinsics,
            w2c=extrinsics,
            image_shape=(height, width),
            scale=scale,
            denormalize_fn=self.denormalize,
            infer_image=image,
        )

    def get_match_prediction_branches(self, results):
        """从模型输出中提取所有 match 预测分支 (coarse + refiner_N).

        Returns:
            branches: dict, 键为分支名 ("coarse", "refiner_2", ...),
                      值为 {"warp2d": Tensor, "warp2d_confidence": Tensor}。
        """
        branches = {
            "coarse": {
                "warp2d": results["warp2d"],
                "warp2d_confidence": results["warp2d_confidence"],
            }
        }

        refiner_scales = sorted({int(key[len("refiner_") : -len("_warp2d")]) for key in results if key.startswith("refiner_") and key.endswith("_warp2d")})
        for scale in refiner_scales:
            warp_key = f"refiner_{scale}_warp2d"
            conf_key = f"refiner_{scale}_warp2d_confidence"
            if conf_key in results:
                branches[f"refiner_{scale}"] = {
                    "warp2d": results[warp_key],
                    "warp2d_confidence": results[conf_key],
                }
        return branches, refiner_scales

    def get_warp3d_prediction_branches(self, results):
        """从模型输出中提取所有 motion warp3d 预测分支 (coarse + refiner_N).

        Returns:
            branches: dict, 键为分支名 ("coarse", "refiner_2", ...),
                      值为 {"warp3d": Tensor}。
        """
        # 与训练侧一致：若仅有解耦输出则先合成 ``warp3d_delta``，再拆分支供可视化/下游使用。
        prepare_decoupled_warp3d_delta_inplace(results)

        branches = {"coarse": {}}
        if "warp3d" in results:
            branches["coarse"]["warp3d"] = results["warp3d"]
        if "warp3d_confidence" in results:
            branches["coarse"]["warp3d_confidence"] = results["warp3d_confidence"]
        if "warp3d_delta" in results:
            branches["coarse"]["warp3d_delta"] = results["warp3d_delta"]
        if "warp3d_delta_confidence" in results:
            branches["coarse"]["warp3d_delta_confidence"] = results["warp3d_delta_confidence"]
        if "warp3d_delta_direction" in results:
            branches["coarse"]["warp3d_delta_direction"] = results["warp3d_delta_direction"]
        if "warp3d_delta_magnitude" in results:
            branches["coarse"]["warp3d_delta_magnitude"] = results["warp3d_delta_magnitude"]

        refiner_stage = sorted({int(key[len("refiner_") : -len("_warp3d")]) for key in results if key.startswith("refiner_") and key.endswith("_warp3d")})
        for stage in refiner_stage:
            branches[f"refiner_{stage}"] = {}

            warp_key = f"refiner_{stage}_warp3d"
            if warp_key in results:
                branches[f"refiner_{stage}"]["warp3d"] = results[warp_key]

            conf_key = f"refiner_{stage}_warp3d_confidence"
            if conf_key in results:
                branches[f"refiner_{stage}"]["warp3d_confidence"] = results[conf_key]

            delta_key = f"refiner_{stage}_warp3d_delta"
            if delta_key in results:
                branches[f"refiner_{stage}"]["warp3d_delta"] = results[delta_key]

            delta_key = f"refiner_{stage}_warp3d_delta_confidence"
            if delta_key in results:
                branches[f"refiner_{stage}"]["warp3d_delta_confidence"] = results[delta_key]

        return branches, refiner_stage

    def get_match_final_prediction_branch(self, branches):
        """从 branches 中选出最终预测分支和辅助 refiner 分支.

        优先选择最小 scale 的 refiner; 若无 refiner 则退回 coarse。

        Returns:
            (final_name, aux_refiner_names): 最终分支名 和 其余 refiner 名列表。
        """
        refiner_names = sorted(
            [name for name in branches if name.startswith("refiner_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        if refiner_names:
            return refiner_names[0], refiner_names[1:]
        return "coarse", []

    def sample_match_gt_at_query(self, warp_gt, mask_gt, query, B, num_pair, H, W):
        """在 GT warp 图上按 query UV 坐标采样, 得到 query 点对应的 GT warp 和 mask.

        Args:
            warp_gt: (B, P, H, W, 2) GT warp field。
            mask_gt: (B, P, H, W) GT valid mask。
            query: query 对象, 含 .uv (Q, 2) 归一化坐标。
            B: batch size。
            num_pair: 视角配对数 P。
            H, W: GT warp 分辨率。

        Returns:
            (gt_warp_sampled, gt_mask_sampled): (B*P, Q, 2) 和 (B*P, Q)。
        """
        uv = query.uv  # (Q, 2) in [0, 1]
        uv_grid = (uv * 2 - 1).unsqueeze(0).unsqueeze(2)  # (1, Q, 1, 2)
        uv_grid = uv_grid.expand(B * num_pair, -1, -1, -1)  # (B*P, Q, 1, 2)

        warp_gt_chw = warp_gt.view(B * num_pair, H, W, 2).permute(0, 3, 1, 2)
        mask_gt_chw = mask_gt.view(B * num_pair, H, W).unsqueeze(1).float()

        gt_warp_sampled = F.grid_sample(
            warp_gt_chw.float(),
            uv_grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=self.align_corners,
        )
        gt_mask_sampled = F.grid_sample(
            mask_gt_chw.float(),
            uv_grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=self.align_corners,
        )

        gt_warp_sampled = gt_warp_sampled.squeeze(-1).permute(0, 2, 1)
        gt_mask_sampled = gt_mask_sampled.squeeze(-1).squeeze(1)
        gt_mask_sampled = gt_mask_sampled >= 0.999

        return gt_warp_sampled, gt_mask_sampled

    def compute_gt_warp_at_scale1(self, pair_idx, depths, intrinsics, extrinsics, camera_type="PINHOLE", meta_data=None):
        """计算全分辨率 GT warp: 对每对视角, 通过深度 + 外参反投影得到像素对应关系.

        支持 PINHOLE / FISHEYE_BLENDER / FISHEYE_EQUIDISTANT 三种相机模型。

        Args:
            depths: (B, V, C, H, W) 各视角深度图。
            intrinsics: (B, V, 3, 3) 各视角内参。
            extrinsics: (B, V, 4, 4) 各视角外参 (world-to-camera)。
            camera_type: 相机模型类型。
            meta_data: 鱼眼相机所需的额外参数 (distort_k, sensor_size 等)。

        Returns:
            (warp_gt, mask_gt): (B, P, H, W, 2) 和 (B, P, H, W), P = len(pair_idx)。
        """
        B, V, C, H, W = depths.shape
        warp_list, mask_list = [], []
        for pair in pair_idx:
            depth1 = depths[:, pair[0], -1]
            depth2 = depths[:, pair[1], -1]
            K1 = intrinsics[:, pair[0]]
            K2 = intrinsics[:, pair[1]]
            T1 = extrinsics[:, pair[0]]
            T2 = extrinsics[:, pair[1]]
            T_1to2 = T2 @ T1.inverse()

            if camera_type == "FISHEYE_BLENDER":
                from hAlgorithm.datasets_fisheye.utils.fisheye_warp import get_gt_warp_fisheye_blender as blender_depth_to_warp

                distort_k = meta_data["distort_k"]
                sensor_size = meta_data["sensor_size"]
                crop_offset = meta_data.get("crop_offset", None)
                assert distort_k is not None and sensor_size is not None

                distort_k = distort_k.to(depth1.device)
                sensor_size = sensor_size.to(depth1.device)
                if crop_offset is not None:
                    crop_offset = crop_offset.to(depth1.device)

                if distort_k.dim() == 3:
                    k_coeffs1 = distort_k[:, pair[0]]
                    k_coeffs2 = distort_k[:, pair[1]]
                else:
                    k_coeffs1 = k_coeffs2 = distort_k

                if sensor_size.dim() == 3:
                    ss1 = sensor_size[:, pair[0]]
                    ss2 = sensor_size[:, pair[1]]
                else:
                    ss1 = ss2 = sensor_size

                co1 = co2 = 0
                if crop_offset is not None:
                    if crop_offset.dim() == 3:
                        co1 = crop_offset[:, pair[0]]
                        co2 = crop_offset[:, pair[1]]
                    else:
                        co1 = co2 = crop_offset
                warp, warp_mask = blender_depth_to_warp(
                    depth1,
                    depth2,
                    T_1to2,
                    K1,
                    K2,
                    k_coeffs1,
                    k_coeffs2,
                    ss1,
                    ss2,
                    crop_offset1=co1,
                    crop_offset2=co2,
                    depth_interpolation_mode="bilinear",
                    H=H,
                    W=W,
                )
            elif camera_type == "FISHEYE_EQUIDISTANT":
                from hAlgorithm.datasets_fisheye.utils.fisheye_warp import get_gt_warp_fisheye_equidistant as _warp_fn

                warp, warp_mask = _warp_fn(
                    depth1,
                    depth2,
                    T_1to2,
                    K1,
                    K2,
                    depth_interpolation_mode="bilinear",
                    H=H,
                    W=W,
                )
            else:
                from hAlgorithm.modules.models2.external.romav2.utils.utils import get_gt_warp as _warp_fn

                warp, warp_mask = _warp_fn(
                    depth1,
                    depth2,
                    T_1to2,
                    K1,
                    K2,
                    depth_interpolation_mode="bilinear",
                    H=H,
                    W=W,
                )

            warp_list.append(warp.to(dtype=depths.dtype))
            mask_list.append(warp_mask)

        warp_gt = torch.stack(warp_list, dim=1)  # (B, P, H, W, 2)
        mask_gt = torch.stack(mask_list, dim=1)  # (B, P, H, W)
        return warp_gt, mask_gt

    def get_sparse_gt(self, batch, meta_data=None, pair_idx=None, scale=None, extrinsics=None, normalize=True):
        """从 batch 中读取稀疏 SfM 关键点 GT, 同时构造 query / 单帧 3D / pair 监督.

        输入约定:
            batch["sparse"]: 长度为 B 的 list, 每个元素 shape ``(S, N_b, 9)``,
                最后一维布局为 ``[point3D_id, u, v, local_xyz(3), global_xyz(3)]``。

                ⚠ 每帧关键点是独立从 ``annotation/sparse_sfm/{stem}.npy`` 读
                出的 (见 ``hAlgorithm/script/dl3dv_regen/extract_sparse_sfm.py``),
                所以跨视角的相同行索引 ``n`` 通常对应不同的 ``point3D_id``。本函数
                会按 batch element 做一次 pid 并集对齐, 之后输出张量中同一
                ``n`` 才在所有视角对应同一 3D 点; 视角中不存在该 pid 的位置
                pid==0, 对应 ``pair_*_valids`` 为 ``False``。

                ``u, v`` 为原始图像像素坐标 (相对于 ``origin_width × origin_height``);
                ``local_xyz`` 为该视角相机系下的 3D 坐标 (``local_z`` = 度量深度);
                ``global_xyz`` 为 COLMAP 世界系坐标 (各视角恒等)。

        Args:
            batch: 输入数据批次。
            meta_data: 含 ``origin_width``/``origin_height`` 的元数据字典。
            pair_idx: 视角配对列表 ``[(src, tgt), ...]``。
            scale: ``(B, S, 1, 1, 1)`` per-view 尺度 (与 ``get_motion_inputs`` 一致),
                用于把 3D 坐标归一化到无量纲空间。
            extrinsics: 当前未使用 (3D 已经在视角相机系/世界系内, 无需额外变换)。
            normalize: 是否用 ``scale`` 把 3D 坐标做归一化。

        Returns:
            query: ``BaseQuery``, ``query.uv`` ``(B, S, N, 2)`` ∈ ``[0, 1]``。
                视角中不存在该 pid 的位置 uv = 0, 调用方需结合 ``pair_*_valids``
                / ``per_view_valid`` mask 掉这些位置, 否则会采到无效像素特征。
            local_points_3d: ``(B, S, N, 3)`` 各视角相机系下的 3D 坐标 (单帧 3D 监督).
            src_trajs_3d: ``(B, S, N, 3)`` 各视角下的世界系 3D 坐标。
            pair_src_trajs_3d: ``(B, P, N, 3)`` 世界系。
            pair_tgt_trajs_3d: ``(B, P, N, 3)`` 世界系。
            pair_src_trajs_2d: ``(B, P, N, 2)`` ∈ ``[-1, 1]`` (与
                ``romav2.utils.warp_kpts`` / ``QueryMatchLoss`` 约定一致)。
            pair_tgt_trajs_2d: ``(B, P, N, 2)`` ∈ ``[-1, 1]``。
            pair_src_valids: ``(B, P, N)`` bool, src 视角中该关键点真实存在。
            pair_tgt_valids: ``(B, P, N)`` bool, tgt 视角中该关键点真实存在。
                ``pair_src_valids & pair_tgt_valids`` 即 src/tgt 在该 pair 上的
                pid 交集 mask, 用于挑出双方都看到的关键点做匹配/三角化监督。
            pair_src_scale: ``(B, P, 1, 1)`` 或 None。
            pair_tgt_scale: ``(B, P, 1, 1)`` 或 None。
        """
        from hAlgorithm.modules.models2.query_bank.query import BaseQuery

        # 总返回 11 项 (1 query + 12 张量), 早返回时其余 12 项填 None.
        none_returns = (None,) * 12
        query = None

        sparse_raw = batch.get("sparse_gt")
        if sparse_raw is None or pair_idx is None or len(pair_idx) == 0:
            return (query,) + none_returns

        if len(sparse_raw) == 0 or sparse_raw[0] is None:
            return (query,) + none_returns

        # ── 1) 直接按 pid 在 batch 内做 "所有视角并集" 对齐 ──────────────
        # 每帧关键点是独立读出的 (extract_sparse_sfm.py), 跨视角同一行索引
        # 通常对应不同 pid. 这里对每个 batch element 单独取 pid 并集, 把
        # 每个视角的真实点散到统一索引位置, 视角中不存在的 pid 处保持全零
        # (pid==0). 之后 ``per_view_valid[:, src] & per_view_valid[:, tgt]``
        # 就是 src/tgt 在该 pair 上的 pid 交集 mask.
        B = len(sparse_raw)
        S = sparse_raw[0].shape[0]
        C = sparse_raw[0].shape[-1]
        assert C == 9, f"Expected each `sparse` element shape (S, N, 9) with " f"[id, u, v, local_xyz, global_xyz], got {tuple(sparse_raw[0].shape)}"

        aligned_chunks = []
        sizes = []
        for raw_b in sparse_raw:
            raw_b = raw_b.to(self.device, non_blocking=True)  # (S, N_b, 9)
            pids_b = raw_b[..., 0].long()  # (S, N_b)
            valid_b = pids_b > 0
            if not valid_b.any():
                aligned_chunks.append(torch.zeros(S, 0, C, dtype=raw_b.dtype, device=raw_b.device))
                sizes.append(0)
                continue
            unique_pids, inverse = torch.unique(pids_b[valid_b], return_inverse=True)
            Nb = int(unique_pids.numel())
            s_idx, n_idx = valid_b.nonzero(as_tuple=True)
            out_b = torch.zeros(S, Nb, C, dtype=raw_b.dtype, device=raw_b.device)
            # 同一 (b, s, pid) 不会重复出现, 直接 scatter 赋值.
            out_b[s_idx, inverse] = raw_b[s_idx, n_idx]
            aligned_chunks.append(out_b)
            sizes.append(Nb)

        max_N = max(sizes) if sizes else 0
        if max_N == 0:
            return (query,) + none_returns

        ref = aligned_chunks[sizes.index(max_N)]
        sparse_tensor = torch.zeros(B, S, max_N, C, dtype=ref.dtype, device=ref.device)
        pad_mask = torch.zeros(B, max_N, dtype=torch.bool, device=ref.device)
        for b, (t, n) in enumerate(zip(aligned_chunks, sizes)):
            if n > 0:
                sparse_tensor[b, :, :n] = t
                pad_mask[b, :n] = True

        # ── 2) 拆分通道 ────────────────────────────────────────────────
        point_id = sparse_tensor[..., 0]  # (B, S, N)
        trajs_2d_pix = sparse_tensor[..., 1:3].clone()  # (B, S, N, 2) pixel
        local_points_3d = sparse_tensor[..., 3:6].clone()  # (B, S, N, 3) camera-frame xyz
        src_trajs_3d = sparse_tensor[..., 6:9].clone()  # (B, S, N, 3) world-frame xyz

        # ── 3) 2D 归一化 (与 romav2.utils.warp_kpts 一致):
        # u_pix ∈ [0.5, W-0.5] → 2*u/W - 1 ∈ [-1+1/W, 1-1/W], 即
        # align_corners=False 下 grid_sample 的取样坐标。
        width = int(meta_data["origin_width"][0])
        height = int(meta_data["origin_height"][0])
        denom_w = width - 1
        denom_h = height - 1

        # ── 4) query: per-view-per-batch (B, S, N, 2) ∈ [0, 1].
        # 视角中不存在该 pid 的位置 uv 为 0 (clamp 之前是 -1), 由 valids mask 掉.
        query_uv = trajs_2d_pix.clone()
        query_uv[..., 0] = query_uv[..., 0] / denom_w
        query_uv[..., 1] = query_uv[..., 1] / denom_h
        query_uv = torch.clamp(query_uv, 0.0, 1.0)
        query = BaseQuery(uv=query_uv, full_uv=False, width=width, height=height)

        # ── 5) 视角内有效性: pid > 0 且非 batch padding ──────────────────
        per_view_valid = (point_id > 0) & pad_mask.unsqueeze(1)  # (B, S, N)

        # ── 6) 配对 gather; src/tgt valid 取 AND 即得 pid 交集 ──────────
        src_idxs = [pair[0] for pair in pair_idx]
        tgt_idxs = [pair[1] for pair in pair_idx]

        pair_src_trajs_3d = src_trajs_3d[:, src_idxs]  # (B, P, N, 3)
        pair_tgt_trajs_3d = src_trajs_3d[:, tgt_idxs]
        pair_src_trajs_2d = query_uv[:, src_idxs]  # (B, P, N, 2)
        pair_tgt_trajs_2d = query_uv[:, tgt_idxs]
        pair_src_valids = per_view_valid[:, src_idxs]  # (B, P, N)
        pair_tgt_valids = per_view_valid[:, tgt_idxs]

        pair_src_visibs = pair_src_valids.clone()
        pair_tgt_visibs = pair_tgt_valids.clone()

        # ── 7) per-view scale (与 get_motion_inputs 写法一致) ────────────
        if scale is not None:
            pair_src_scale = scale[:, src_idxs].squeeze(-1)  # (B, P, 1, 1)
            pair_tgt_scale = scale[:, tgt_idxs].squeeze(-1)
        else:
            pair_src_scale = pair_tgt_scale = None

        if normalize and pair_src_scale is not None:
            pair_src_trajs_3d = pair_src_trajs_3d / pair_src_scale
            pair_tgt_trajs_3d = pair_tgt_trajs_3d / pair_src_scale
            # 同步把单帧 3D 也归一化 (按各视角 scale 各除一份).
            view_scale = scale.squeeze(-1)  # (B, S, 1, 1)
            local_points_3d = local_points_3d / view_scale
            src_trajs_3d = src_trajs_3d / view_scale

        return (
            query,
            local_points_3d,
            src_trajs_3d,
            pair_src_trajs_3d,
            pair_tgt_trajs_3d,
            pair_src_trajs_2d,
            pair_tgt_trajs_2d,
            pair_src_valids,
            pair_tgt_valids,
            pair_src_visibs,
            pair_tgt_visibs,
            pair_src_scale,
            pair_tgt_scale,
        )

    def get_motion_inputs(self, batch, meta_data=None, pair_idx=None, scale=None, extrinsics=None, normalize=True):
        """Load and optionally scale 3-D trajectories from the batch."""

        from hAlgorithm.modules.models2.query_bank.query import BaseQuery

        trajs_2d = batch.get("trajs_2d")  # List, B, each element is (S, N, 2)
        trajs_3d = batch.get("trajs_3d")  # List, B, each element is (S, N, 3)
        valids = batch.get("valids")  # List, B, each element is (S, N)
        visibs = batch.get("visibs")  # List, B, each element is (S, N)

        if trajs_2d is not None and isinstance(trajs_2d, list):
            B = len(trajs_2d)
            T = trajs_2d[0].shape[0]
            max_N = max(t.shape[1] for t in trajs_2d)

            def _pad(lst, ch=None):
                if lst is None or lst[0] is None:
                    return None
                shape = (B, T, max_N, ch) if ch else (B, T, max_N)
                out = torch.zeros(shape, dtype=lst[0].dtype)
                for i, t in enumerate(lst):
                    out[i, :, : t.shape[1]] = t
                return out.to(self.device, non_blocking=True)

            trajs_2d = _pad(trajs_2d, 2)
            trajs_3d = _pad(trajs_3d, 3) if trajs_3d is not None else None
            valids = _pad(valids)
            visibs = _pad(visibs)
        else:
            # for attr in ("trajs_2d", "trajs_3d", "valids", "visibs"):
            #     v = locals()[attr]
            #     if v is not None:
            #         locals()[attr]  # just reference to keep linter quiet

            if trajs_2d is not None:
                trajs_2d = trajs_2d.to(self.device)
            if trajs_3d is not None:
                trajs_3d = trajs_3d.to(self.device)
            if valids is not None:
                valids = valids.to(self.device)
            if visibs is not None:
                visibs = visibs.to(self.device)

        if self.motion_extrinsics_name in batch and batch[self.motion_extrinsics_name] is not None and trajs_3d is not None:
            motion_extrinsics = batch[self.motion_extrinsics_name].to(self.device)
            base_extrinsics = extrinsics.clone()
            # Train path feeds normalized extrinsics (normalize=True) and needs one denormalize.
            # Infer path already denormalizes upstream (normalize=False), so avoid double scaling.
            if scale is not None and normalize:
                base_extrinsics[..., :3, 3] = self.denormalize(base_extrinsics[..., :3, 3], scale[..., 0, 0])

            # Align GT trajectory coords to first camera.
            src_idx = 0
            raw_to_aligned = torch.linalg.inv(base_extrinsics[:, src_idx]) @ motion_extrinsics[:, src_idx]
            bsz, T_t, Nq, _ = trajs_3d.shape
            pts = trajs_3d.reshape(bsz, -1, 3)
            pts_h = torch.cat(
                [pts, torch.ones(*pts.shape[:2], 1, device=pts.device, dtype=pts.dtype)],
                -1,
            )
            trajs_3d = torch.bmm(raw_to_aligned, pts_h.transpose(1, 2)).transpose(1, 2)[..., :3].reshape(bsz, T_t, Nq, 3)

        if trajs_2d is not None:
            width = int(meta_data["origin_width"][0])
            height = int(meta_data["origin_height"][0])

            trajs_2d[..., 0] /= width - 1
            trajs_2d[..., 1] /= height - 1

            uv_grid = torch.clamp(trajs_2d, min=0.0, max=1.0)

            query = BaseQuery(uv=uv_grid, full_uv=False, width=width, height=height)
        else:
            query = None

        src_idxs = [pair[0] for pair in pair_idx]
        tgt_idxs = [pair[1] for pair in pair_idx]

        pair_src_trajs_3d = trajs_3d[:, src_idxs]
        pair_tgt_trajs_3d = trajs_3d[:, tgt_idxs]

        pair_src_trajs_2d = trajs_2d[:, src_idxs]
        pair_tgt_trajs_2d = trajs_2d[:, tgt_idxs]

        pair_src_valids = valids[:, src_idxs]
        pair_tgt_valids = valids[:, tgt_idxs]

        pair_src_visibs = visibs[:, src_idxs]
        pair_tgt_visibs = visibs[:, tgt_idxs]

        if scale is not None:
            pair_src_scale = scale[:, src_idxs].squeeze(-1)
            pair_tgt_scale = scale[:, tgt_idxs].squeeze(-1)
            assert ((pair_src_scale - pair_tgt_scale) > 0).sum() == 0
        else:
            pair_src_scale = pair_tgt_scale = None

        if normalize and pair_src_scale is not None:
            pair_src_trajs_3d = pair_src_trajs_3d / pair_src_scale
            pair_tgt_trajs_3d = pair_tgt_trajs_3d / pair_src_scale

        return query, pair_src_trajs_3d, pair_tgt_trajs_3d, pair_src_trajs_2d, pair_tgt_trajs_2d, pair_src_valids, pair_tgt_valids, pair_src_visibs, pair_tgt_visibs, pair_src_scale, pair_tgt_scale

    def _sample_pair_map_at_query(self, pair_map, pair_query_uv):
        """Sample pair-wise dense map at per-pair query UV."""
        B, P = pair_query_uv.shape[:2]
        Q = pair_query_uv.shape[2]

        sample_grid = pair_query_uv.view(B * P, Q, 1, 2) * 2 - 1

        if pair_map.ndim == 5:
            # (B, P, H, W, C) -> (B*P, C, H, W)
            pair_map_chw = pair_map.view(B * P, *pair_map.shape[2:]).permute(0, 3, 1, 2).contiguous().float()
            sampled = F.grid_sample(pair_map_chw, sample_grid.float(), mode="bilinear", padding_mode="border", align_corners=self.align_corners)
            sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous().view(B, P, Q, -1)
            return sampled

        if pair_map.ndim == 4:
            # (B, P, H, W) -> (B*P, 1, H, W)
            pair_map_chw = pair_map.view(B * P, *pair_map.shape[-2:]).unsqueeze(1).float()
            sampled = F.grid_sample(pair_map_chw, sample_grid.float(), mode="bilinear", padding_mode="border", align_corners=self.align_corners)
            sampled = sampled.squeeze(-1).squeeze(1).contiguous().view(B, P, Q)
            return sampled

        raise ValueError(f"Unsupported pair_map dimensions: {pair_map.shape}")

    def get_static_motion_inputs(self, image, edge_mask, query, pair_idx, target_local_depth, target_depth_mask, intrinsics, extrinsics, meta_data, scale=None, camera_type="PINHOLE", normalize=True):
        """Build static-scene motion supervision from dense depth + camera poses."""
        if pair_idx is None or len(pair_idx) == 0:
            return (query,) + (None,) * 12

        if query is None:
            model_for_query = self.model
            if hasattr(model_for_query, "module"):
                model_for_query = model_for_query.module
            if hasattr(model_for_query, "_orig_mod"):
                model_for_query = model_for_query._orig_mod
            query_bank = getattr(model_for_query, "query_banck", None)
            if query_bank is None:
                raise ValueError("Static motion supervision needs query sampling, but model.query_banck is unavailable.")

            query = query_bank(image, edge_mask=edge_mask, meta_data=meta_data)

        B, N = target_local_depth.shape[:2]
        num_pair = len(pair_idx)
        src_idxs = [pair[0] for pair in pair_idx]
        tgt_idxs = [pair[1] for pair in pair_idx]

        if query.uv.ndim == 2:
            src_query_uv = query.uv.unsqueeze(0).unsqueeze(0).expand(B, num_pair, -1, -1)
        elif query.uv.ndim == 4:
            src_query_uv = query.uv[:, src_idxs]
        else:
            raise ValueError(f"Unsupported query uv dimension: {query.uv.ndim}")

        target_local_depth = target_local_depth.view(B, N, *target_local_depth.shape[-3:])
        depth_height, depth_width = target_local_depth.shape[-2:]

        # Unify GT to metric scale first:
        # - train path usually feeds normalized depth/extrinsics (normalize=True), so we denormalize once here.
        # - infer path already denormalizes upstream (normalize=False), so we must NOT denormalize again.
        target_match_gt_depth = target_local_depth
        target_match_gt_extrinsics = extrinsics.clone()
        if scale is not None and normalize:
            target_match_gt_depth = self.denormalize(target_match_gt_depth, scale=scale)
            target_match_gt_extrinsics[..., :3, 3] = self.denormalize(target_match_gt_extrinsics[..., :3, 3], scale=scale[..., 0, 0])
        target_match_gt_depth_intrinsics = intrinsics.clone()

        x_scale = depth_width / int(meta_data["input_width"][0])
        y_scale = depth_height / int(meta_data["input_height"][0])
        target_match_gt_depth_intrinsics[..., 0, :] *= x_scale
        target_match_gt_depth_intrinsics[..., 1, :] *= y_scale

        warp_gt, mask_gt = self.compute_gt_warp_at_scale1(
            pair_idx,
            target_match_gt_depth,
            target_match_gt_depth_intrinsics,
            target_match_gt_extrinsics,
            camera_type=camera_type,
            meta_data=meta_data,
        )

        pair_tgt_trajs_2d = self._sample_pair_map_at_query(warp_gt, src_query_uv)
        pair_tgt_trajs_2d = torch.clamp((pair_tgt_trajs_2d + 1) / 2, 0.0, 1.0)
        pair_src_trajs_2d = src_query_uv

        pair_valids = self._sample_pair_map_at_query(mask_gt, src_query_uv) >= 0.999
        pair_src_valids = pair_valids
        pair_tgt_valids = pair_valids
        pair_src_visibs = pair_valids
        pair_tgt_visibs = pair_valids

        if target_match_gt_depth.shape[2] == 1:
            local_points_3d = self.depth_to_points_from_meta(target_match_gt_depth, K=intrinsics, meta_data=meta_data)
        else:
            local_points_3d = target_match_gt_depth[:, :, :3]

        if query.uv.ndim == 2:
            per_view_query_uv = query.uv.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        elif query.uv.ndim == 4:
            per_view_query_uv = query.uv
        else:
            raise ValueError(f"Unsupported query uv dimension for per-view sampling: {query.uv.ndim}")

        local_sparse_points = self._sample_pair_map_at_query(local_points_3d.permute(0, 1, 3, 4, 2).contiguous(), per_view_query_uv)
        view_w2c = target_match_gt_extrinsics
        view_R = view_w2c[..., :3, :3]
        view_t = view_w2c[..., :3, 3]
        local_centered = local_sparse_points - view_t.unsqueeze(2)
        glb_sparse_points = torch.matmul(view_R.transpose(-1, -2).unsqueeze(2), local_centered.unsqueeze(-1)).squeeze(-1)

        pair_src_local_3d = local_points_3d[:, src_idxs].permute(0, 1, 3, 4, 2).contiguous()
        pair_src_local_3d = self._sample_pair_map_at_query(pair_src_local_3d, src_query_uv)

        src_w2c = target_match_gt_extrinsics[:, src_idxs]
        src_R = src_w2c[..., :3, :3]
        src_t = src_w2c[..., :3, 3]

        src_local_centered = pair_src_local_3d - src_t.unsqueeze(2)
        pair_src_trajs_3d = torch.matmul(src_R.transpose(-1, -2).unsqueeze(2), src_local_centered.unsqueeze(-1)).squeeze(-1)
        pair_tgt_trajs_3d = pair_src_trajs_3d.clone()

        if scale is not None:
            pair_src_scale = scale[:, src_idxs].squeeze(-1)
            pair_tgt_scale = scale[:, tgt_idxs].squeeze(-1)
        else:
            pair_src_scale = pair_tgt_scale = None

        if normalize and pair_src_scale is not None:
            pair_src_trajs_3d = pair_src_trajs_3d / pair_src_scale
            pair_tgt_trajs_3d = pair_tgt_trajs_3d / pair_src_scale

        return (
            query,
            local_sparse_points,
            glb_sparse_points,
            pair_src_trajs_3d,
            pair_tgt_trajs_3d,
            pair_src_trajs_2d,
            pair_tgt_trajs_2d,
            pair_src_valids,
            pair_tgt_valids,
            pair_src_visibs,
            pair_tgt_visibs,
            pair_src_scale,
            pair_tgt_scale,
        )

    def train_step(self, batch):
        """训练单步 (V3): 前向 → 计算多任务 loss (Local + Global + Camera + Gaussian + Motion).

        Loss 组成:
            Part1 (lcl)   — 局部深度 loss: query 点在相机坐标系下的深度/点云监督
            Part2 (glb)   — 全局点云 loss: query 点在世界坐标系下的 3D 点监督
            Part3 (cm)    — 相机 loss:   pose_encoding → extrinsics/intrinsics 的监督
            Part4 (rc)    — 渲染 loss:   Gaussian splatting 渲染的 RGB + 深度重建监督
            Part5 (motion) — 2D/3D 运动监督: 动态集直接监督, static 集由深度+位姿投影构造监督

        各输出按是否为 None 动态决定是否计算对应 loss。

        Returns:
            (total_loss, total_loss_dict): 标量总 loss 及各子项 loss 字典。
        """
        self.train()

        (
            name,
            total_iter,
            meta_data,
            image,
            edge_mask,
            query_image,
            intrinsics,
            extrinsics,
            scale,
            prompt_depth,
            target_local_depth,
            target_global_points,
            target_depth_mask,
            target_normal,
            target_normal_mask,
            target_motion_mask,
            target_invalid_mask,
            image_show,
            align_data,
            ray_directions,
            ray_world,
            extrinsics_noise,
            prompt_extrinsics,
        ) = self.get_inputs(batch)

        meta_data["sub_pixel_scale"] = self.training_sub_pixel_scale

        camera_type = self.get_camera_type(meta_data)
        edge_mask = self.get_edge_mask(batch)
        pair_idx = self.get_pair_idx(meta_data)
        num_views = int(meta_data["frames"][0]) * int(meta_data["views"][0])
        pair_idx = self._augment_pair_idx_for_dgs(pair_idx, num_views)
        is_dynamic = self.dynamic_dataset_names is not None and name in self.dynamic_dataset_names
        is_static = self.static_dataset_names is not None and name in self.static_dataset_names
        is_sparse = self.sparse_dataset_names is not None and name in self.sparse_dataset_names
        
        if int(is_dynamic) + int(is_static) + int(is_sparse) > 1:
            raise ValueError(
                f"Dataset '{name}' matched multiple dataset groups: "
                f"dynamic={is_dynamic}, static={is_static}, sparse={is_sparse}. "
                "Only one group can be True."
            )

        if is_dynamic and pair_idx is not None:
            (
                query,
                pair_src_trajs_3d,
                pair_tgt_trajs_3d,
                pair_src_trajs_2d,
                pair_tgt_trajs_2d,
                pair_src_valids,
                pair_tgt_valids,
                pair_src_visibs,
                pair_tgt_visibs,
                pair_src_scale,
                pair_tgt_scale,
            ) = self.get_motion_inputs(batch, meta_data=meta_data, pair_idx=pair_idx, scale=scale, extrinsics=extrinsics, normalize=True)
        else:
            query = pair_src_trajs_3d = pair_tgt_trajs_3d = pair_src_trajs_2d = pair_tgt_trajs_2d = None
            pair_src_valids = pair_tgt_valids = pair_src_visibs = pair_tgt_visibs = None
            pair_src_scale = pair_tgt_scale = None
            local_sparse_points = glb_sparse_points = None

        sparse_gt_flag = False

        if is_sparse and pair_idx is not None:
            (
                query,
                local_sparse_points,
                glb_sparse_points,
                pair_src_trajs_3d,
                pair_tgt_trajs_3d,
                pair_src_trajs_2d,
                pair_tgt_trajs_2d,
                pair_src_valids,
                pair_tgt_valids,
                pair_src_visibs,
                pair_tgt_visibs,
                pair_src_scale,
                pair_tgt_scale,
            ) = self.get_sparse_gt(batch, meta_data=meta_data, pair_idx=pair_idx, scale=scale, extrinsics=extrinsics, normalize=True)
            sparse_gt_flag = query is not None

        if is_static and pair_idx is not None:
            (
                query,
                local_sparse_points,
                glb_sparse_points,
                pair_src_trajs_3d,
                pair_tgt_trajs_3d,
                pair_src_trajs_2d,
                pair_tgt_trajs_2d,
                pair_src_valids,
                pair_tgt_valids,
                pair_src_visibs,
                pair_tgt_visibs,
                pair_src_scale,
                pair_tgt_scale,
            ) = self.get_static_motion_inputs(
                image=image,
                edge_mask=edge_mask,
                query=None,
                pair_idx=pair_idx,
                target_local_depth=target_local_depth,
                target_depth_mask=target_depth_mask,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                meta_data=meta_data,
                scale=scale,
                camera_type=camera_type,
                normalize=True,
            )
            sparse_gt_flag = query is not None

        # w2c 优先级: prompt_extrinsics > noisy extrinsics > GT extrinsics
        if prompt_extrinsics is not None:
            w2c = prompt_extrinsics
        elif extrinsics_noise is not None and extrinsics is not None:
            w2c = torch.matmul(extrinsics_noise, extrinsics)
        else:
            w2c = extrinsics

        results = self.model(
            image,
            edge_mask=edge_mask,
            query=query,
            query_rgb=query_image,
            scale=scale,
            prompt_depth=prompt_depth,
            intrinsics=intrinsics,
            ray_directions=ray_directions,
            w2c=w2c,
            ray_world=ray_world,
            meta_data=meta_data,
            pair_idx=pair_idx,
        )

        B, N = image.shape[:2]

        total_loss, total_loss_dict = 0, dict()

        if self.pose_encoding_type == "absT_quaR_FoV":
            pose_enc = results.get("pose_enc")
        else:
            raise NotImplementedError

        # 提取模型输出: local depth, global points, confidence 及 pose_encoding
        # 各输出可能为 None (取决于模型配置), 后续按需计算对应 loss
        query_depth = results.get("depth", None)
        query_conf = results.get("confidence", None)
        query_global_points = results.get("global_points", None)
        query_global_conf = results.get("global_confidence", None)

        # 构建 grid_sample 采样坐标, 从 [0,1] 归一化到 [-1,1]
        query = results["query"]
        query_uv_grid = query.uv
        query_uv_grid = query_uv_grid * 2 - 1
        if query.uv.ndim == 4:
            query_uv_grid = query_uv_grid.view(B * N, *query_uv_grid.shape[-2:]).unsqueeze(2)  # (1, Q, 1, 2)
        elif query.uv.ndim == 2:
            query_uv_grid = query_uv_grid.unsqueeze(0).unsqueeze(2)  # (1, Q, 1, 2)
            query_uv_grid = query_uv_grid.expand(B * N, -1, -1, -1)  # (B*N, Q, 1, 2)
        else:
            raise ValueError(f"Unsupported uv_grid dimension: {query_uv_grid.ndim}")

        if query_depth is not None or query_global_points is not None:
            # 统一增加通道/空间维度, 方便后续与 target 做 grid_sample 对齐
            if query_depth is not None:
                query_depth = query_depth.unsqueeze(2)
            if query_conf is not None:
                query_conf = query_conf.unsqueeze(2)
            if query_global_points is not None:
                # (B, N, Q, 3) → (B, N, 3, Q) → (B, N, 3, Q, 1)
                query_global_points = query_global_points.permute(0, 1, 3, 2).contiguous().unsqueeze(-1)
            if query_global_conf is not None:
                query_global_conf = query_global_conf.unsqueeze(2)

            if sparse_gt_flag:
                # B N Q C -> B N C Q -> B N C Q 1
                Q, C = local_sparse_points.shape[-2:]
                local_sparse_points = local_sparse_points.permute(0, 1, 3, 2).contiguous().view(B, N, C, Q, 1)
                glb_sparse_points = glb_sparse_points.permute(0, 1, 3, 2).contiguous().view(B, N, C, Q, 1)
                target_query_valid_mask = local_sparse_points.new_ones(B, N, 1, Q, 1).bool()

                # ── Part1: Local depth loss ───────────────────────────────────────
                if query_depth is not None or query_conf is not None:
                    loss, loss_dict = self.get_base_loss(
                        name=name,
                        image=image,
                        coord="local",
                        pred_depth=query_depth,
                        pred_conf=query_conf,
                        target_depth=local_sparse_points,
                        valid_mask=target_query_valid_mask,
                        scale=scale,
                    )
                    total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="lcl")

                # ── Part2: Global points loss ─────────────────────────────────────
                if query_global_points is not None or query_global_conf is not None:
                    loss, loss_dict = self.get_base_loss(
                        name=name,
                        image=image,
                        coord="global",
                        pred_depth=query_global_points,
                        pred_conf=query_global_conf,
                        target_depth=glb_sparse_points,
                        valid_mask=target_query_valid_mask,
                        scale=scale,
                    )
                    total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="glb")
            else:
                if target_depth_mask.ndim == 4:
                    target_depth_mask = target_depth_mask.view(B * N, 1, *target_depth_mask.shape[-2:])
                elif target_depth_mask.ndim == 5:
                    target_depth_mask = target_depth_mask.view(B * N, *target_depth_mask.shape[-3:])
                else:
                    raise NotImplementedError

                # ── Part1: Local depth loss ───────────────────────────────────────
                # 在相机坐标系下监督 query 点的深度 / 3D 点预测
                if query_depth is not None or query_conf is not None:
                    # 合并 B,N → B*N 以进行 grid_sample
                    target_local_depth = target_local_depth.view(B * N, *target_local_depth.shape[-3:])

                    # 在 GT 深度图上采样 query 点对应的深度值和有效 mask
                    target_query_depth = F.grid_sample(target_local_depth, query_uv_grid, mode="bilinear", padding_mode="border", align_corners=self.align_corners)
                    target_query_valid_mask = F.grid_sample(target_depth_mask.float(), query_uv_grid, mode="bilinear", padding_mode="border", align_corners=self.align_corners)

                    if query.full_uv:
                        # (B, N, C, Q, 1) -> (B, N, C, H, W)
                        query_depth = query_depth.squeeze(-1).view(*query_depth.shape[:3], query.height, query.width)
                        query_conf = query_conf.squeeze(-1).view(*query_conf.shape[:3], query.height, query.width)

                        if query_depth.shape[-3] == 1:
                            query_depth = self.depth_to_points_from_meta(
                                query_depth, K=intrinsics, meta_data=meta_data
                            )
                        # (B*N, C, Q, 1) -> (B, N, C, H, W)
                        target_query_depth = target_query_depth.squeeze(-1).view(B, N, *query_depth.shape[-3:])
                        target_query_valid_mask = target_query_valid_mask.squeeze(-1).view(B, N, *query_conf.shape[-3:]) == 1
                    else:
                        # (B*N, C, Q, 1) -> (B, N, C, Q, 1)
                        target_query_depth = target_query_depth.squeeze(-1).view(B, N, *target_query_depth.shape[-3:])
                        target_query_valid_mask = target_query_valid_mask.squeeze(-1).view(B, N, *target_query_valid_mask.shape[-3:]) == 1

                    loss, loss_dict = self.get_base_loss(
                        name=name,
                        image=image,
                        coord="local",
                        pred_depth=query_depth,
                        pred_conf=query_conf,
                        target_depth=target_query_depth,
                        valid_mask=target_query_valid_mask,
                        scale=scale,
                    )
                    total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="lcl")

                # ── Part2: Global points loss ─────────────────────────────────────
                # 在世界坐标系下监督 query 点的 3D 位置 (XYZ) 预测
                if query_global_points is not None or query_global_conf is not None:
                    target_global_points = target_global_points.view(B * N, *target_global_points.shape[-3:])

                    # 在 GT 全局点云图上采样 query 点对应的 3D 坐标
                    target_query_global_points = F.grid_sample(target_global_points, query_uv_grid, mode="bilinear", padding_mode="border", align_corners=self.align_corners)
                    target_query_valid_mask = F.grid_sample(target_depth_mask.float(), query_uv_grid, mode="bilinear", padding_mode="border", align_corners=self.align_corners)

                    if query.full_uv:
                        # (B, N, C, Q, 1) -> (B, N, C, H, W)
                        query_global_points = query_global_points.view(*query_global_points.shape[:3], query.height, query.width)
                        query_global_conf = query_global_conf.squeeze(-1).view(*query_global_conf.shape[:3], query.height, query.width)

                        # (B*N, C, Q, 1) -> (B, N, C, H, W)
                        target_query_global_points = target_query_global_points.view(B, N, *query_global_points.shape[-3:])
                        target_query_valid_mask = target_query_valid_mask.squeeze(-1).view(B, N, *query_global_conf.shape[-3:]) == 1
                    else:
                        # (B*N, C, Q, 1) -> (B, N, C, Q, 1)
                        target_query_global_points = target_query_global_points.view(B, N, *target_query_global_points.shape[-3:])
                        target_query_valid_mask = target_query_valid_mask.squeeze(-1).view(B, N, *target_query_valid_mask.shape[-3:]) == 1

                    loss, loss_dict = self.get_base_loss(
                        name=name,
                        image=image,
                        coord="global",
                        pred_depth=query_global_points,
                        pred_conf=query_global_conf,
                        target_depth=target_query_global_points,
                        valid_mask=target_query_valid_mask,
                        scale=scale,
                    )
                    total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="glb")

        # Pair Decoder 输出: local depth, global points, confidence
        pair_query_depth = results.get("pair_depth", None)
        pair_query_conf = results.get("pair_confidence", None)
        pair_query_global_points = results.get("pair_global_points", None)
        pair_query_global_conf = results.get("pair_global_confidence", None)

        if pair_idx is not None and (pair_query_depth is not None or pair_query_global_points is not None):
            has_pair_local = pair_query_depth is not None
            has_pair_global = pair_query_global_points is not None

            if has_pair_local or has_pair_global:
                num_pair = len(pair_idx)
                pair_idx_src = [pair[0] for pair in pair_idx]

                pair_query = results["query"]
                pair_image = image[:, pair_idx_src]
                pair_scale = scale[:, pair_idx_src] if scale is not None and scale.ndim >= 2 else scale
                pair_intrinsics = intrinsics[:, pair_idx_src] if intrinsics is not None and intrinsics.ndim >= 4 else intrinsics

                # 统一增加通道/空间维度, 方便后续与 target 做 grid_sample 对齐
                if pair_query_depth is not None:
                    pair_query_depth = pair_query_depth.unsqueeze(2)
                if pair_query_conf is not None:
                    pair_query_conf = pair_query_conf.unsqueeze(2)
                if pair_query_global_points is not None and pair_query_global_points.ndim == 4:
                    # (B, P, Q, 3) -> (B, P, 3, Q) -> (B, P, 3, Q, 1)
                    pair_query_global_points = pair_query_global_points.permute(0, 1, 3, 2).contiguous().unsqueeze(-1)
                if pair_query_global_conf is not None:
                    pair_query_global_conf = pair_query_global_conf.unsqueeze(2)

                # 构建 pair 对应的 grid_sample 采样坐标, 从 [0,1] 归一化到 [-1,1]
                pair_query_uv = pair_query.uv
                if pair_query_uv.ndim == 4:
                    pair_query_uv = pair_query_uv[:, pair_idx_src]
                    pair_query_uv_grid = (pair_query_uv * 2 - 1).view(B * num_pair, *pair_query_uv.shape[-2:]).unsqueeze(2)
                elif pair_query_uv.ndim == 2:
                    pair_query_uv_grid = (pair_query_uv * 2 - 1).unsqueeze(0).unsqueeze(2)
                    pair_query_uv_grid = pair_query_uv_grid.expand(B * num_pair, -1, -1, -1)
                else:
                    raise ValueError(f"Unsupported uv_grid dimension for pair supervision: {pair_query_uv.ndim}")

                if sparse_gt_flag:
                    pair_local_sparse_points = local_sparse_points[:, pair_idx_src]
                    pair_glb_sparse_points = glb_sparse_points[:, pair_idx_src]

                    # Unify sparse targets to [B, P, C, Q, 1] for pair supervision.
                    if pair_local_sparse_points.ndim == 4:
                        pair_local_sparse_points = pair_local_sparse_points.permute(0, 1, 3, 2).contiguous().unsqueeze(-1)
                    elif pair_local_sparse_points.ndim != 5:
                        raise ValueError(f"Unsupported pair_local_sparse_points shape: {pair_local_sparse_points.shape}")
                    if pair_glb_sparse_points.ndim == 4:
                        pair_glb_sparse_points = pair_glb_sparse_points.permute(0, 1, 3, 2).contiguous().unsqueeze(-1)
                    elif pair_glb_sparse_points.ndim != 5:
                        raise ValueError(f"Unsupported pair_glb_sparse_points shape: {pair_glb_sparse_points.shape}")

                    target_query_valid_mask = pair_local_sparse_points.new_ones(B, num_pair, 1, pair_local_sparse_points.shape[-2], 1).bool()

                    # ── Pair Part1: Local depth loss ───────────────────────────────
                    if has_pair_local:
                        loss, loss_dict = self.get_base_loss(
                            name=name,
                            image=pair_image,
                            coord="local",
                            pred_depth=pair_query_depth,
                            pred_conf=pair_query_conf,
                            target_depth=pair_local_sparse_points,
                            valid_mask=target_query_valid_mask,
                            scale=pair_scale,
                        )
                        total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="pair_lcl")

                    # ── Pair Part2: Global points loss ─────────────────────────────
                    if has_pair_global:
                        loss, loss_dict = self.get_base_loss(
                            name=name,
                            image=pair_image,
                            coord="global",
                            pred_depth=pair_query_global_points,
                            pred_conf=pair_query_global_conf,
                            target_depth=pair_glb_sparse_points,
                            valid_mask=target_query_valid_mask,
                            scale=pair_scale,
                        )
                        total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="pair_glb")
                else:
                    if target_depth_mask.shape[0] == B * N:
                        target_depth_mask_view = target_depth_mask.view(B, N, *target_depth_mask.shape[-3:])
                    else:
                        target_depth_mask_view = target_depth_mask
                    if target_local_depth.shape[0] == B * N:
                        target_local_depth_view = target_local_depth.view(B, N, *target_local_depth.shape[-3:])
                    else:
                        target_local_depth_view = target_local_depth
                    if target_global_points.shape[0] == B * N:
                        target_global_points_view = target_global_points.view(B, N, *target_global_points.shape[-3:])
                    else:
                        target_global_points_view = target_global_points

                    pair_target_depth_mask = target_depth_mask_view[:, pair_idx_src]
                    pair_target_local_depth = target_local_depth_view[:, pair_idx_src]
                    pair_target_global_points = target_global_points_view[:, pair_idx_src]

                    if pair_target_depth_mask.ndim == 4:
                        pair_target_depth_mask = pair_target_depth_mask.view(B * num_pair, 1, *pair_target_depth_mask.shape[-2:])
                    elif pair_target_depth_mask.ndim == 5:
                        pair_target_depth_mask = pair_target_depth_mask.view(B * num_pair, *pair_target_depth_mask.shape[-3:])
                    else:
                        raise NotImplementedError

                    # ── Pair Part1: Local depth loss ───────────────────────────────
                    if has_pair_local:
                        pair_target_local_depth = pair_target_local_depth.view(B * num_pair, *pair_target_local_depth.shape[-3:])
                        target_query_depth = F.grid_sample(
                            pair_target_local_depth,
                            pair_query_uv_grid,
                            mode="bilinear",
                            padding_mode="border",
                            align_corners=self.align_corners,
                        )
                        target_query_valid_mask = F.grid_sample(
                            pair_target_depth_mask.float(),
                            pair_query_uv_grid,
                            mode="bilinear",
                            padding_mode="border",
                            align_corners=self.align_corners,
                        )

                        if pair_query.full_uv:
                            if pair_query_depth is not None:
                                pair_query_depth = pair_query_depth.squeeze(-1).view(*pair_query_depth.shape[:3], pair_query.height, pair_query.width)
                                if pair_query_depth.shape[-3] == 1:
                                    pair_query_depth = self.depth_to_points_from_meta(
                                        pair_query_depth, K=pair_intrinsics, meta_data=meta_data
                                    )
                                mask_shape = pair_query_depth.shape[-3:]
                            if pair_query_conf is not None:
                                pair_query_conf = pair_query_conf.squeeze(-1).view(*pair_query_conf.shape[:3], pair_query.height, pair_query.width)
                                if pair_query_depth is None:
                                    mask_shape = pair_query_conf.shape[-3:]
                            else:
                                if pair_query_depth is None:
                                    raise RuntimeError("pair_query_depth and pair_query_conf are both None in local pair supervision")

                            target_query_depth = target_query_depth.squeeze(-1).view(B, num_pair, *mask_shape)
                            target_query_valid_mask = target_query_valid_mask.squeeze(-1).view(B, num_pair, *mask_shape) == 1
                        else:
                            target_query_depth = target_query_depth.squeeze(-1).view(B, num_pair, *target_query_depth.shape[-3:])
                            target_query_valid_mask = target_query_valid_mask.squeeze(-1).view(B, num_pair, *target_query_valid_mask.shape[-3:]) == 1

                        loss, loss_dict = self.get_base_loss(
                            name=name,
                            image=pair_image,
                            coord="local",
                            pred_depth=pair_query_depth,
                            pred_conf=pair_query_conf,
                            target_depth=target_query_depth,
                            valid_mask=target_query_valid_mask,
                            scale=pair_scale,
                        )
                        total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="pair_lcl")

                    # ── Pair Part2: Global points loss ─────────────────────────────
                    if has_pair_global:
                        pair_target_global_points = pair_target_global_points.view(B * num_pair, *pair_target_global_points.shape[-3:])
                        target_query_global_points = F.grid_sample(
                            pair_target_global_points,
                            pair_query_uv_grid,
                            mode="bilinear",
                            padding_mode="border",
                            align_corners=self.align_corners,
                        )
                        target_query_valid_mask = F.grid_sample(
                            pair_target_depth_mask.float(),
                            pair_query_uv_grid,
                            mode="bilinear",
                            padding_mode="border",
                            align_corners=self.align_corners,
                        )

                        if pair_query.full_uv:
                            if pair_query_global_points is not None:
                                pair_query_global_points = pair_query_global_points.view(*pair_query_global_points.shape[:3], pair_query.height, pair_query.width)
                                mask_shape = pair_query_global_points.shape[-3:]
                            if pair_query_global_conf is not None:
                                pair_query_global_conf = pair_query_global_conf.squeeze(-1).view(*pair_query_global_conf.shape[:3], pair_query.height, pair_query.width)
                                if pair_query_global_points is None:
                                    mask_shape = pair_query_global_conf.shape[-3:]
                            else:
                                if pair_query_global_points is None:
                                    raise RuntimeError("pair_query_global_points and pair_query_global_conf are both None in global pair supervision")

                            target_query_global_points = target_query_global_points.view(B, num_pair, *mask_shape)
                            target_query_valid_mask = target_query_valid_mask.squeeze(-1).view(B, num_pair, *mask_shape) == 1
                        else:
                            target_query_global_points = target_query_global_points.view(B, num_pair, *target_query_global_points.shape[-3:])
                            target_query_valid_mask = target_query_valid_mask.squeeze(-1).view(B, num_pair, *target_query_valid_mask.shape[-3:]) == 1

                        loss, loss_dict = self.get_base_loss(
                            name=name,
                            image=pair_image,
                            coord="global",
                            pred_depth=pair_query_global_points,
                            pred_conf=pair_query_global_conf,
                            target_depth=target_query_global_points,
                            valid_mask=target_query_valid_mask,
                            scale=pair_scale,
                        )
                        total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="pair_glb")

        # ── Part3: Camera loss ─────────────────────────────────────────────
        # 监督 pose_encoding → extrinsics/intrinsics 的预测精度
        if self.camera_loss is not None and pose_enc is not None and pose_enc[0].requires_grad:
            loss, loss_dict = self.camera_loss(
                name=name,
                pose_enc=pose_enc,
                target_intrinsic=intrinsics,
                target_extrinsics=extrinsics,
                scale=scale,
                image_size_hw=image.shape[-2:],
                valid_mask=target_depth_mask,
            )
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="cm")

        # ── Part4: Gaussian splatting 渲染 loss ───────────────────────────
        # 使用 gsplat 将预测的 Gaussians 渲染为 RGB + depth, 与输入图像 / GT 深度做重建 loss
        if "gaussians" in results and results["gaussians"] is not None:
            gs_render_rgb, gs_render_depth = self.render_gaussians(results["gaussians"], intrinsics=intrinsics, w2c=w2c, image_shape=(query.height, query.width))
            if gs_render_rgb is not None:
                gs_render_rgb = gs_render_rgb.view(B, N, *gs_render_rgb.shape[1:])
                # 渲染深度先除以 scale 再反投影为 3D 点, 使其与 GT 点云在同一空间
                gs_render_depth = gs_render_depth.view(B, N, *gs_render_depth.shape[1:]).unsqueeze(2)

                if scale is not None:
                    gs_render_depth = gs_render_depth / scale
                gs_render_depth = self.depth_to_points_from_meta(
                    gs_render_depth, K=intrinsics, meta_data=meta_data
                )

                gs_target_depth = target_local_depth.view(B, N, *target_local_depth.shape[1:])
                gs_target_mask = target_depth_mask.view(B, N, *target_depth_mask.shape[1:])

                loss, loss_dict = self.get_ffgs_loss(
                    name=name,
                    render_rgb=gs_render_rgb,
                    render_depth=gs_render_depth,
                    target_rgb=image,
                    target_depth=gs_target_depth,
                    target_depth_mask=gs_target_mask,
                )

                total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="rc")

        # ── Part5: Motion, track2d and track3d loss ────────────────────────────────────────
        if (is_dynamic or is_sparse or is_static) and pair_idx is not None:
            # 三维 motion loss 前先合成 ``warp3d_delta``（解耦方向×模长或旧版 finalize 路径）。
            prepare_decoupled_warp3d_delta_inplace(results)

            total_loss_dict["motion"] = 0
            motion_task_weight = self.task_weight.get("motion", 1.0)

            if self.motion_dynamic_threshold is not None:
                pair_3d_delta = (pair_tgt_trajs_3d - pair_src_trajs_3d).abs()
                if pair_src_scale is not None:
                    pair_3d_delta = pair_3d_delta * pair_src_scale
                pair_3d_delta = torch.norm(pair_3d_delta, p=2, dim=-1)
                dynamic_mask = pair_3d_delta >= self.motion_dynamic_threshold

                motion3d_weight = torch.ones_like(dynamic_mask).float()
                motion3d_weight[dynamic_mask] *= self.motion_dynamic_scale
                motion3d_weight[~dynamic_mask] *= self.motion_static_scale
                motion3d_weight = motion3d_weight.unsqueeze(-1)
            else:
                dynamic_mask = motion3d_weight = None

            if self.motion_3d_train_novis:
                pair_valid_mask = (pair_tgt_valids & pair_src_valids & pair_src_visibs).unsqueeze(-1)

                novis_mask = pair_src_visibs & (~pair_tgt_visibs)
                if motion3d_weight is not None:
                    motion3d_weight[novis_mask] *= self.motion_novis_scale
                else:
                    motion3d_weight = torch.ones_like(pair_valid_mask.float())
                    motion3d_weight[novis_mask] *= self.motion_novis_scale
            else:
                pair_valid_mask = (pair_tgt_valids & pair_tgt_visibs & pair_src_valids & pair_src_visibs).unsqueeze(-1)  # (B, num_pair, Q, 1)

            if (
                self.query_motion_mask_loss is not None
                and "motion_mask" in results
                and dynamic_mask is not None
                and pair_tgt_trajs_3d is not None
            ):
                pred_mask = results["motion_mask"]
                if pred_mask.shape[-1] == 1:
                    pred_mask = pred_mask.squeeze(-1)
                B_m, num_pair_m, Q_m = pred_mask.shape[:3]
                mask_valid = pair_valid_mask.squeeze(-1)
                mask_loss, mask_loss_dict = self.query_motion_mask_loss(
                    pred_mask=pred_mask.reshape(B_m * num_pair_m, Q_m),
                    gt_mask=dynamic_mask.float().reshape(B_m * num_pair_m, Q_m),
                    valid_mask=mask_valid.reshape(B_m * num_pair_m, Q_m),
                    name=name,
                )
                mask_loss = torch.nan_to_num(mask_loss, nan=0.0, posinf=0.0, neginf=0.0)
                motion_mask_loss = mask_loss * motion_task_weight
                total_loss = total_loss + motion_mask_loss
                if "motion" in total_loss_dict:
                    total_loss_dict["motion"] += motion_mask_loss
                else:
                    total_loss_dict["motion"] = motion_mask_loss
                total_loss_dict["motion_mask_bce"] = motion_mask_loss
                if isinstance(mask_loss_dict, dict):
                    for stat_k, stat_v in mask_loss_dict.items():
                        total_loss_dict[f"motion_mask_{stat_k}"] = stat_v

            # 可选：对 canonical ``warp3d_delta_*`` 做方向/静态点模长辅助监督（须已执行 prepare）。
            if (
                self._warp3d_delta_direction_loss is not None
                and pair_tgt_trajs_3d is not None
                and results.get("warp3d_delta_direction") is not None
                and results.get("warp3d_delta_magnitude") is not None
            ):
                gt_delta_3d = pair_tgt_trajs_3d - pair_src_trajs_3d
                pred_delta_dir = results["warp3d_delta_direction"]
                pred_delta_mag = results["warp3d_delta_magnitude"]
                valid_mask = pair_valid_mask.squeeze(-1)
                B, num_pair, Q = gt_delta_3d.shape[0], gt_delta_3d.shape[1], gt_delta_3d.shape[2]
                bp = B * num_pair
                scale_flat = None
                if pair_src_scale is not None:
                    scale_src = pair_src_scale
                    while scale_src.dim() > 2:
                        scale_src = scale_src.squeeze(-1)
                    if scale_src.shape[:2] == (B, num_pair):
                        scale_flat = scale_src.reshape(bp)
                dir_loss_out = self._warp3d_delta_direction_loss(
                    pred_direction=pred_delta_dir.reshape(bp, Q, 3),
                    gt_displacement=gt_delta_3d.reshape(bp, Q, 3),
                    valid_mask=valid_mask.reshape(bp, Q).float(),
                    scale=scale_flat,
                    pred_magnitude=pred_delta_mag.reshape(bp, Q, 1),
                    name=name,
                )
                dir_loss = dir_loss_out[0] if isinstance(dir_loss_out, (tuple, list)) else dir_loss_out
                dir_loss = torch.nan_to_num(dir_loss, nan=0.0, posinf=0.0, neginf=0.0)
                motion_dir_loss = dir_loss * motion_task_weight
                total_loss = total_loss + motion_dir_loss
                if "motion" in total_loss_dict:
                    total_loss_dict["motion"] += motion_dir_loss
                else:
                    total_loss_dict["motion"] = motion_dir_loss
                total_loss_dict["motion_warp3d_delta_direction"] = motion_dir_loss
                if isinstance(dir_loss_out, (tuple, list)) and len(dir_loss_out) > 1:
                    dir_stats = dir_loss_out[1]
                    if isinstance(dir_stats, dict):
                        for stat_k, stat_v in dir_stats.items():
                            total_loss_dict[f"motion_dir_stat_{stat_k}"] = stat_v

            if (self.query_motion3d_loss is not None or self.query_motion3d_delta_loss is not None) and ("warp3d" in results or "warp3d_delta" in results) and pair_tgt_trajs_3d is not None:
                if "refiner_1_warp3d" not in results:
                    pair_src_trajs_3d = pair_src_trajs_3d.unsqueeze(-2)
                    pair_tgt_trajs_3d = pair_tgt_trajs_3d.unsqueeze(-2)  # (B, num_pair, Q, 1, 3)

                    warp3d = warp3d_confidence = warp3d_delta = warp3d_delta_confidence = None
                    if "warp3d" in results:
                        warp3d = results["warp3d"].unsqueeze(-2)  # (B, num_pair, Q, 1, 3)
                    if "warp3d_confidence" in results:
                        warp3d_confidence = results["warp3d_confidence"].unsqueeze(-2)  # (B, num_pair, Q, 1, 3)
                    if "warp3d_delta" in results:
                        warp3d_delta = results["warp3d_delta"].unsqueeze(-2)  # (B, num_pair, Q, 1, 3)
                    if "warp3d_delta_confidence" in results:
                        warp3d_delta_confidence = results["warp3d_delta_confidence"].unsqueeze(-2)  # (B, num_pair, Q, 1, 3)

                    sample_loss = dict()
                    for bi in range(pair_src_trajs_3d.shape[0]):
                        for pi in range(pair_src_trajs_3d.shape[1]):
                            if self.query_motion3d_loss is not None and warp3d is not None:
                                loss = self.query_motion3d_loss(
                                    name=name,
                                    pred_depth=warp3d[bi, pi],
                                    pred_conf=warp3d_confidence[bi, pi] if warp3d_confidence is not None else None,
                                    target_depth=pair_tgt_trajs_3d[bi, pi],
                                    valid_mask=pair_valid_mask[bi, pi],
                                    weight=motion3d_weight[bi, pi] if motion3d_weight is not None else None,
                                )
                                for key in loss.keys():
                                    if key not in sample_loss:
                                        sample_loss[key] = 0
                                    sample_loss[key] += loss[key]

                            if self.query_motion3d_delta_loss is not None and (warp3d is not None or warp3d_delta is not None):
                                delta_loss = self.query_motion3d_delta_loss(
                                    name=name,
                                    pred_depth=warp3d[bi, pi] - pair_src_trajs_3d[bi, pi] if warp3d_delta is None else warp3d_delta[bi, pi],
                                    pred_conf=warp3d_delta_confidence[bi, pi] if warp3d_delta_confidence is not None else None,
                                    target_depth=pair_tgt_trajs_3d[bi, pi] - pair_src_trajs_3d[bi, pi],
                                    valid_mask=pair_valid_mask[bi, pi],
                                    weight=motion3d_weight[bi, pi] if motion3d_weight is not None else None,
                                )
                                for key in delta_loss.keys():
                                    fix_key = "delta_" + key
                                    if fix_key not in sample_loss:
                                        sample_loss[fix_key] = 0
                                    sample_loss[fix_key] += delta_loss[key]

                        loss = {key: val / (pair_src_trajs_3d.shape[0] * pair_src_trajs_3d.shape[1]) for key, val in sample_loss.items()}

                    total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, task_name="motion")
                else:
                    pair_src_trajs_3d = pair_src_trajs_3d.unsqueeze(-2)
                    pair_tgt_trajs_3d = pair_tgt_trajs_3d.unsqueeze(-2)  # (B, num_pair, Q, 1, 3)

                    warp3d_branch, warp3d_stage = self.get_warp3d_prediction_branches(results)
                    for branch_name, branch in warp3d_branch.items():
                        warp3d = warp3d_confidence = warp3d_delta = warp3d_delta_confidence = None
                        if "warp3d" in branch:
                            warp3d = branch["warp3d"].unsqueeze(-2)  # (B, num_pair, Q, 1, 3)
                        if "warp3d_confidence" in results:
                            warp3d_confidence = results["warp3d_confidence"].unsqueeze(-2)  # (B, num_pair, Q, 1, 3)
                        if "warp3d_delta" in branch:
                            warp3d_delta = branch["warp3d_delta"].unsqueeze(-2)  # (B, num_pair, Q, 1, 3)
                        if "warp3d_delta_confidence" in results:
                            warp3d_delta_confidence = results["warp3d_delta_confidence"].unsqueeze(-2)  # (B, num_pair, Q, 1, 3)

                        sample_loss = dict()
                        for bi in range(pair_src_trajs_3d.shape[0]):
                            for pi in range(pair_src_trajs_3d.shape[1]):
                                if self.query_motion3d_loss is not None and warp3d is not None:
                                    loss = self.query_motion3d_loss(
                                        name=name,
                                        pred_depth=warp3d[bi, pi],
                                        pred_conf=warp3d_confidence[bi, pi] if warp3d_confidence is not None else None,
                                        target_depth=pair_tgt_trajs_3d[bi, pi],
                                        valid_mask=pair_valid_mask[bi, pi],
                                        weight=motion3d_weight[bi, pi] if motion3d_weight is not None else None,
                                    )
                                    for key in loss.keys():
                                        if key not in sample_loss:
                                            sample_loss[key] = 0
                                        sample_loss[key] += loss[key]

                                if self.query_motion3d_delta_loss is not None and (warp3d is not None or warp3d_delta is not None):
                                    delta_loss = self.query_motion3d_delta_loss(
                                        name=name,
                                        pred_depth=warp3d[bi, pi] - pair_src_trajs_3d[bi, pi] if warp3d_delta is None else warp3d_delta[bi, pi],
                                        pred_conf=warp3d_delta_confidence[bi, pi] if warp3d_delta_confidence is not None else None,
                                        target_depth=pair_tgt_trajs_3d[bi, pi] - pair_src_trajs_3d[bi, pi],
                                        valid_mask=pair_valid_mask[bi, pi],
                                        weight=motion3d_weight[bi, pi] if motion3d_weight is not None else None,
                                    )
                                    for key in delta_loss.keys():
                                        fix_key = "delta_" + key
                                        if fix_key not in sample_loss:
                                            sample_loss[fix_key] = 0
                                        sample_loss[fix_key] += delta_loss[key]

                        loss = {key: val / (pair_src_trajs_3d.shape[0] * pair_src_trajs_3d.shape[1]) for key, val in sample_loss.items()}

                        branch_total = sum(loss.values()) * motion_task_weight
                        total_loss += branch_total

                        if "motion" in total_loss_dict:
                            total_loss_dict["motion"] += branch_total
                        else:
                            total_loss_dict["motion"] = branch_total

                        total_loss_dict[f"motion_3d_{branch_name}"] = branch_total
                        for key, val in loss.items():
                            total_loss_dict[f"motion_3d_{branch_name}_{key}"] = val * motion_task_weight

            if self.query_motion2d_loss is not None and "warp2d" in results and pair_tgt_trajs_2d is not None:
                branches, warp2d_scales = self.get_match_prediction_branches(results)

                num_pair = len(pair_idx)

                for branch_name, branch in branches.items():
                    pred_warp = branch["warp2d"].view(B * num_pair, -1, 2)
                    pred_conf = branch["warp2d_confidence"].view(B * num_pair, -1, branch["warp2d_confidence"].shape[-1])
                    gt_warp = pair_tgt_trajs_2d.view(B * num_pair, -1, 2) * 2 - 1  # NOTE: [-1, 1], match grid sample

                    if self.motion_2d_train_novis:
                        gt_mask = (pair_tgt_valids & pair_src_valids & pair_src_visibs).view(B * num_pair, -1)  # (B, num_pair, Q, 1)

                        novis_mask = (pair_src_visibs & (~pair_tgt_visibs)).view(B * num_pair, -1)
                        motion2d_weight = torch.ones_like(gt_mask.float())
                        motion2d_weight[novis_mask] *= self.motion_novis_scale
                    else:
                        gt_mask = (pair_tgt_valids & pair_tgt_visibs & pair_src_valids & pair_src_visibs).view(B * num_pair, -1)  # (B, num_pair, Q, 1)
                        motion2d_weight = None

                    sample_loss = 0
                    sample_loss_dict = dict()
                    for bpi in range(pred_warp.shape[0]):
                        loss, loss_dict = self.query_motion2d_loss(
                            pred_warp=pred_warp[bpi : bpi + 1],
                            pred_conf=pred_conf[bpi : bpi + 1],
                            gt_warp=gt_warp[bpi : bpi + 1],
                            gt_mask=gt_mask[bpi : bpi + 1],
                            gt_h=query.height,
                            gt_w=query.width,
                            name=name,
                            weight=motion2d_weight[bpi : bpi + 1] if motion2d_weight is not None else None,
                        )
                        sample_loss += loss
                        for key in loss_dict.keys():
                            if key not in sample_loss_dict:
                                sample_loss_dict[key] = 0
                            sample_loss_dict[key] += loss_dict[key]

                    loss = sample_loss / pred_warp.shape[0]
                    loss_dict = {key: val / pred_warp.shape[0] for key, val in sample_loss_dict.items()}

                    branch_total = loss * motion_task_weight
                    total_loss += branch_total
                    if "motion" in total_loss_dict:
                        total_loss_dict["motion"] += branch_total
                    else:
                        total_loss_dict["motion"] = branch_total
                    total_loss_dict[f"motion_{branch_name}"] = branch_total
                    for key, val in loss_dict.items():
                        total_loss_dict[f"motion_{branch_name}_{key}"] = val * motion_task_weight

        # ── Part6: Sparse pair Dynamic 4DGS rendering loss ─────────────────
        if (
            self.sparse_dynamic_gaussian_render_loss is not None
            and "gs_opacity" in results
            and intrinsics is not None
            and extrinsics is not None
        ):
            dgs_scale = scale
            if self.dgs_render_normalized_only:
                dgs_scale = None
            elif (
                self.training
                and self.dgs_scale_dropout_prob > 0.0
                and torch.rand((), device=image.device) < self.dgs_scale_dropout_prob
            ):
                dgs_scale = None
            dgs_kwargs = self.get_dgs_render_loss_kwargs(
                image=image,
                target_local_depth=target_local_depth,
                target_depth_mask=target_depth_mask,
                scale=scale,
                name=name,
            )
            dgs_loss, dgs_loss_dict = self.sparse_dynamic_gaussian_render_loss(
                name=name,
                results=results,
                image=image,
                intrinsics=intrinsics,
                w2c=w2c,
                scale=dgs_scale,
                denormalize_fn=self.denormalize,
                meta_data=meta_data,
                global_step=total_iter,
                pair_idx=pair_idx,
                **dgs_kwargs,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, dgs_loss, dgs_loss_dict, task_name="dgs",
            )

        if (
            self.motion_gs_displacement_consistency_loss is not None
            and "motion_sparse_scene_flow" in results
            and "sparse_scene_flow" in results
        ):
            motion_query = results.get("query")
            gaussian_query = results.get("gaussian_query")
            motion_valid = None
            ref_frame = int(results.get("src_frame_idx", 0))
            if pair_src_valids is not None and pair_idx is not None:
                for pi, (s, t) in enumerate(pair_idx):
                    if int(s) == ref_frame and int(t) == ref_frame:
                        motion_valid = pair_src_valids[:, pi]
                        break
            cons_loss, cons_dict = self.motion_gs_displacement_consistency_loss(
                results=results,
                motion_query=motion_query,
                gaussian_query=gaussian_query,
                meta_data=meta_data,
                valid_mask=motion_valid,
                name=name,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, cons_loss, cons_dict, task_name="dgs",
            )

        if self.gs_xyz_offset_loss is not None and "gs_offset_xyz" in results:
            xyz_loss = self.gs_xyz_offset_loss(
                gs_offset_xyz=results["gs_offset_xyz"],
                name=name,
            )
            if xyz_loss is not None:
                total_loss, total_loss_dict = self.add_loss(
                    total_loss,
                    total_loss_dict,
                    xyz_loss,
                    {"gs_xyz_offset": round(xyz_loss.item(), 6)},
                    task_name="gsreg",
                )

        # Extra INFO
        if scale is not None:
            total_loss_dict["scale"] = round(scale.max().cpu().item(), 2)

        if image is not None:
            aspect_ratio = image.shape[-1] / image.shape[-2]
            total_loss_dict["aspect_ratio"] = round(aspect_ratio, 2)
            total_loss_dict["bs"] = int(image.shape[0])
            total_loss_dict["view"] = int(image.shape[1])

            max_size = max(image.shape[-1], image.shape[-2])
            total_loss_dict["max_size"] = int(max_size)

        if pair_idx is not None:
            total_loss_dict["pair"] = len(pair_idx)

        if self.debug_rgb_path:
            if isinstance(meta_data["data_info"][0], (list, tuple)):
                logging.debug(f"rgb: {name}, {meta_data['data_info'][0][0]['rgb']}")
            else:
                logging.debug(f"rgb: {name}, {meta_data['data_info'][0]['rgb']}")

        return total_loss, total_loss_dict

    # ── Hooks for subclasses ──────────────────────────────────────────────
    def _infer_extra_model_kwargs(self) -> dict:
        """Extra kwargs injected into the model call inside :meth:`infer`.

        Override in subclasses to pass additional flags such as
        ``return_cached=True`` without duplicating the full model-call block.
        """
        return {}

    def _on_infer_model_results(self, results: dict) -> None:
        """Called immediately after the model forward in :meth:`infer`.

        Override in subclasses to capture side-channel data (e.g. cached
        encoder features) from ``results`` before the dict is consumed by
        the rest of the inference logic.
        """

    @torch.no_grad()
    def infer(self, **batch):
        """推理入口 (V3): 在 V1 基础上增加 global points、pose decoding、match 和子类 hook.

        流程:
            1. 模型前向: 提取 local depth / global points / pose_encoding / warp
            2. Pose decoding: pose_encoding → pred_extrinsics / pred_intrinsics
            3. 确定输出分辨率, 反归一化 GT 到真实尺度
            4. (可选) Gaussian splatting 渲染
            5. (可选) 提取 match 预测: warp / overlap / precision, 计算 GT warp
            6. 逐 frame × view: 反投影 + scale-align → postprocess → 附加 match 结果

        Returns:
            mv_outputs: List[ReconstructOutput], 含 local + global + 预测相机 + match 数据。
        """
        self.eval()

        (
            name,
            total_iter,
            meta_data,
            image,
            edge_mask,
            query_image,
            intrinsics,
            extrinsics,
            scale,
            prompt_depth,
            target_local_depth,
            target_global_points,
            target_depth_mask,
            target_normal,
            target_normal_mask,
            target_motion_mask,
            target_invalid_mask,
            image_show,
            align_data,
            ray_directions,
            ray_world,
            extrinsics_noise,
            prompt_extrinsics,
        ) = self.get_inputs(batch)

        camera_type = self.get_camera_type(meta_data)
        pair_idx = self.get_pair_idx(meta_data)
        global_pair_idx = self.get_global_pair_idx(meta_data)

        infer_pair_idx = pair_idx
        if global_pair_idx is not None:
            infer_pair_idx = infer_pair_idx + global_pair_idx
        infer_pair_idx, global_pair_idx = self._ensure_ref_identity_pairs(
            infer_pair_idx, global_pair_idx
        )

        meta_data["sub_pixel_scale"] = self.testing_sub_pixel_scale

        _extra_model_kwargs = self._infer_extra_model_kwargs()

        if prompt_extrinsics is not None:
            w2c = prompt_extrinsics
        elif extrinsics_noise is not None and extrinsics is not None:
            w2c = torch.matmul(extrinsics_noise, extrinsics)
        else:
            w2c = extrinsics

        if "use_amp" in batch and batch["use_amp"]:
            with torch.autocast("cuda", enabled=bool(batch["use_amp"]), dtype=batch["amp_dtype"]):
                results = self.model(
                    image,
                    query_rgb=query_image,
                    scale=scale,
                    prompt_depth=prompt_depth,
                    intrinsics=intrinsics,
                    ray_directions=ray_directions,
                    w2c=w2c,
                    ray_world=ray_world,
                    meta_data=meta_data,
                    pair_idx=infer_pair_idx,
                    **_extra_model_kwargs,
                )
        else:
            results = self.model(
                image,
                query_rgb=query_image,
                scale=scale,
                prompt_depth=prompt_depth,
                intrinsics=intrinsics,
                ray_directions=ray_directions,
                w2c=w2c,
                ray_world=ray_world,
                meta_data=meta_data,
                pair_idx=infer_pair_idx,
                **_extra_model_kwargs,
            )

        self._on_infer_model_results(results)

        # 提取预测: local depth, global points, confidence
        query_depth = results.get("depth", None)
        query_conf = results.get("confidence", None)
        query_global_points = results.get("global_points", None)
        query_global_conf = results.get("global_confidence", None)

        if query_depth is None and query_global_points is None:
            num_views = int(meta_data["frames"][0]) * int(meta_data["views"][0])
            if global_pair_idx is not None and len(global_pair_idx) == num_views:

                pair_depth = results.get("pair_depth", None)
                pair_confidence = results.get("pair_confidence", None)
                warp3d = results.get("warp3d", None)
                warp3d_confidence = results.get("warp3d_confidence", None)

                query_depth = []
                query_conf = []
                query_global_points =[]
                query_global_conf = []

                for src_index in range(num_views):
                    index = infer_pair_idx.index((src_index, src_index))

                    query_depth.append(pair_depth[:, index])
                    query_conf.append(pair_confidence[:, index])

                    query_global_points.append(warp3d[:, index])
                    query_global_conf.append(warp3d_confidence[:, index])
                
                query_depth = torch.stack(query_depth, dim=1)
                query_conf = torch.stack(query_conf, dim=1)
                query_global_points = torch.stack(query_global_points, dim=1)
                query_global_conf = torch.stack(query_global_conf, dim=1)

                if query_global_conf.shape[-1] == 3:
                    query_global_conf = query_global_conf.mean(dim=-1, keepdim=True)

        if query_depth is not None:
            query_depth = query_depth.squeeze(-1).cpu()
        if query_conf is not None:
            query_conf = query_conf.squeeze(-1).cpu()
        if query_global_points is not None:
            # (B, N, Q, 3) → (B, N, 3, Q) 便于后续 view 为 3D 点图
            query_global_points = query_global_points.permute(0, 1, 3, 2).contiguous().cpu()
        if query_global_conf is not None:
            query_global_conf = query_global_conf.squeeze(-1).cpu()

        # 取最后一层 pose_encoding (多层迭代细化时)
        pose_enc = results.get("pose_enc", None)
        if self.pose_encoding_type == "absT_quaR_FoV":
            if pose_enc is not None and isinstance(pose_enc, (list, tuple)):
                pose_enc = pose_enc[-1]
        else:
            raise NotImplementedError

        # 将 pose_encoding 解码为显式的外参矩阵和内参矩阵
        if pose_enc is not None:
            if self.pose_encoding_type == "absT_quaR_FoV":
                pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                    pose_encoding=pose_enc.float(),
                    image_size_hw=image.shape[-2:],
                    build_intrinsics=True,
                )
                if scale is not None:
                    pred_extrinsics[..., :3, 3] = self.denormalize(pred_extrinsics[..., :3, 3], scale=scale[..., 0, 0])

                # 以第 0 视角为参考系, 将所有外参转为相对位姿
                if self.save_output_cfg["output_normalize_cameras"]:
                    w2c_pred = pred_extrinsics
                    base_c2w_pred = w2c_pred[:, 0:1].inverse()
                    pred_extrinsics = w2c_pred @ base_c2w_pred

            else:
                raise NotImplementedError

            pred_extrinsics = pred_extrinsics.cpu()
            pred_intrinsics = pred_intrinsics.cpu()

        else:
            pred_extrinsics, pred_intrinsics = None, None

        # Full-resolution K for 2D motion vis (query_stride / testing_sub_pixel_scale
        # only subsamples the query grid; pose_enc is decoded at input H×W).
        pred_intrinsics_proj = (
            pred_intrinsics.clone() if pred_intrinsics is not None else None
        )

        # 根据 sub_pixel_scale 确定输出分辨率
        if self.testing_sub_pixel_scale == -1:
            width = int(meta_data["origin_width"][0])
            height = int(meta_data["origin_height"][0])
        elif self.testing_sub_pixel_scale == 1:
            width = int(meta_data["input_width"][0])
            height = int(meta_data["input_height"][0])
        else:
            width = int(meta_data["origin_width"][0] / self.testing_sub_pixel_scale)
            height = int(meta_data["origin_height"][0] / self.testing_sub_pixel_scale)

        # Normalized extrinsics for sparse DGS when training in normalized render space.
        w2c_normalized_for_dgs = extrinsics.clone() if extrinsics is not None else None

        # 反归一化: 将 scale-normalized 的 GT 深度 / 点云 / 外参平移量恢复到真实尺度
        if target_local_depth is not None and scale is not None:
            target_local_depth = self.denormalize(target_local_depth, scale=scale)
        if target_global_points is not None and scale is not None:
            target_global_points = self.denormalize(target_global_points, scale=scale)
        # Metric extrinsics for motion UV projection and legacy (non-normalized) DGS.
        if extrinsics is not None and scale is not None:
            extrinsics[..., :3, 3] = self.denormalize(extrinsics[..., :3, 3], scale=scale[..., 0, 0])
        w2c_metric_for_dgs = extrinsics.clone() if extrinsics is not None else None

        if target_depth_mask is not None and target_depth_mask.ndim == 4:
            target_depth_mask = target_depth_mask.unsqueeze(-3)

        input_w = int(meta_data["input_width"][0])
        input_h = int(meta_data["input_height"][0])
        intrinsics_at_input = intrinsics.clone() if intrinsics is not None else None
        pred_intrinsics_at_input = (
            pred_intrinsics.clone() if pred_intrinsics is not None else None
        )

        # 训练时输出分辨率与GT不同时, 按比例缩放内参
        if self.testing_sub_pixel_scale != 1 and intrinsics is not None:
            x_scale = width / int(meta_data["input_width"][0])
            y_scale = height / int(meta_data["input_height"][0])
            intrinsics[..., 0, :] *= x_scale
            intrinsics[..., 1, :] *= y_scale

        if self.testing_sub_pixel_scale != 1 and pred_intrinsics is not None:
            x_scale = width / int(meta_data["input_width"][0])
            y_scale = height / int(meta_data["input_height"][0])
            pred_intrinsics[..., 0, :] *= x_scale
            pred_intrinsics[..., 1, :] *= y_scale

        render_intrinsics = intrinsics
        render_w2c = w2c_metric_for_dgs
        if render_intrinsics is None and pred_intrinsics is not None:
            render_intrinsics = pred_intrinsics.to(device=self.device)
        if render_w2c is None and pred_extrinsics is not None:
            render_w2c = pred_extrinsics.to(device=self.device)
        if intrinsics is None and render_intrinsics is not None:
            logging.info("Gaussian render: using predicted camera (no GT in batch).")

        if self.dgs_render_normalized_only:
            dgs_render_w2c = w2c_normalized_for_dgs
            if dgs_render_w2c is None and pred_extrinsics is not None:
                dgs_render_w2c = pred_extrinsics.to(device=self.device)
        else:
            dgs_render_w2c = render_w2c

        # FFGS rendering
        gaussians = gs_render_rgb = gs_render_depth = None
        if "gaussians" in results and results["gaussians"] is not None:
            gaussians = results["gaussians"]
            if render_intrinsics is not None and render_w2c is not None:
                gs_render_rgb, gs_render_depth = self.render_gaussians(
                    gaussians,
                    intrinsics=render_intrinsics,
                    w2c=render_w2c,
                    image_shape=(height, width),
                )
                if gs_render_rgb is not None:
                    gs_render_rgb = gs_render_rgb.cpu()
                    gs_render_depth = gs_render_depth.cpu()
            else:
                logging.warning("Skip gaussian rendering in infer: missing camera matrices.")

        # Sparse pair Dynamic 4DGS rendering
        dgs_width, dgs_height = resolve_sparse_dgs_render_shape(
            meta_data, results, width, height,
        )
        dgs_base_intrinsics = (
            intrinsics_at_input
            if intrinsics_at_input is not None
            else pred_intrinsics_at_input
        )
        if dgs_base_intrinsics is not None:
            dgs_render_intrinsics = dgs_base_intrinsics
            if dgs_width != input_w or dgs_height != input_h:
                dgs_render_intrinsics = scale_bnv_intrinsics(
                    dgs_base_intrinsics,
                    dgs_width / input_w,
                    dgs_height / input_h,
                )
        else:
            dgs_render_intrinsics = render_intrinsics
        if dgs_render_intrinsics is not None:
            dgs_render_intrinsics = dgs_render_intrinsics.to(device=self.device)
        dgs_renders = self._render_sparse_dgs_infer(
            results=results,
            intrinsics=dgs_render_intrinsics,
            extrinsics=dgs_render_w2c,
            scale=scale,
            height=dgs_height,
            width=dgs_width,
            image=image,
        )
        dgs_render_rgb = dgs_render_depth = None
        if dgs_renders:
            dgs_render_rgb = dgs_renders.get("dgs_render_rgb")
            dgs_render_depth = dgs_renders.get("dgs_render_depth")
            if dgs_render_rgb is not None:
                dgs_render_rgb = dgs_render_rgb.cpu()
            if dgs_render_depth is not None:
                dgs_render_depth = dgs_render_depth.cpu()

        if "sparse_gt" in batch:
            (
                sparse_query,
                local_sparse_points,
                glb_sparse_points,
                pair_src_trajs_3d,
                pair_tgt_trajs_3d,
                pair_src_trajs_2d,
                pair_tgt_trajs_2d,
                pair_src_valids,
                pair_tgt_valids,
                pair_src_visibs,
                pair_tgt_visibs,
                pair_src_scale,
                pair_tgt_scale,
            ) = self.get_sparse_gt(batch, meta_data=meta_data, pair_idx=pair_idx, scale=scale, extrinsics=extrinsics, normalize=False)
            sparse_gt_flag = sparse_query is not None
        else:
            sparse_gt_flag = False

        if pair_idx is not None:
            # 与训练一致：解耦输出先合成 ``warp3d_delta``，后续可见性/轨迹逻辑才对齐。
            prepare_decoupled_warp3d_delta_inplace(results)

        if ("warp3d" in results or "warp3d_delta" in results) and pair_idx is not None:
            query = results["query"]

            warp3d_branches, warp3d_stage = self.get_warp3d_prediction_branches(results)

            if len(warp3d_stage) > 0:
                last_warp3d_stage = warp3d_stage[0]
                warp3d = warp3d_branches[f"refiner_{last_warp3d_stage}"].get("warp3d", None)
                warp3d_delta = warp3d_branches[f"refiner_{last_warp3d_stage}"].get("warp3d_delta", None)
            else:
                warp3d = warp3d_branches["coarse"].get("warp3d", None)
                warp3d_delta = warp3d_branches["coarse"].get("warp3d_delta", None)

            if "warp2d" in results:
                warp2d_branches, warp2d_scales = self.get_match_prediction_branches(results)

                if len(warp2d_scales) > 0:
                    min_warp2d_scale = warp2d_scales[0]
                    warp2d = warp2d_branches[f"refiner_{min_warp2d_scale}"]["warp2d"]
                else:
                    warp2d = warp2d_branches["coarse"]["warp2d"]

                warp2d = warp2d.clone()
                warp2d = (warp2d + 1) * 0.5  # NOTE: [-1, 1] -> [0, 1]
            else:
                warp2d = None

            if "trajs_2d" in batch or "trajs_3d" in batch:
                (
                    motion_query,
                    pair_src_trajs_3d,
                    pair_tgt_trajs_3d,
                    pair_src_trajs_2d,
                    pair_tgt_trajs_2d,
                    pair_src_valids,
                    pair_tgt_valids,
                    pair_src_visibs,
                    pair_tgt_visibs,
                    pair_src_scale,
                    pair_tgt_scale,
                ) = self.get_motion_inputs(batch, meta_data=meta_data, pair_idx=pair_idx, scale=scale, extrinsics=extrinsics, normalize=False)
            elif target_local_depth is not None and intrinsics is not None and extrinsics is not None:
                (
                    _static_query,
                    _local_sparse_points,
                    _glb_sparse_points,
                    pair_src_trajs_3d,
                    pair_tgt_trajs_3d,
                    pair_src_trajs_2d,
                    pair_tgt_trajs_2d,
                    pair_src_valids,
                    pair_tgt_valids,
                    pair_src_visibs,
                    pair_tgt_visibs,
                    pair_src_scale,
                    pair_tgt_scale,
                ) = self.get_static_motion_inputs(
                    image=image,
                    edge_mask=edge_mask,
                    query=query,
                    pair_idx=pair_idx,
                    target_local_depth=target_local_depth,
                    target_depth_mask=target_depth_mask,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    meta_data=meta_data,
                    scale=scale,
                    camera_type=camera_type,
                    normalize=False,
                )
            elif not sparse_gt_flag:
                pair_src_trajs_3d = pair_tgt_trajs_3d = pair_src_trajs_2d = pair_tgt_trajs_2d = None
                pair_src_valids = pair_tgt_valids = pair_src_visibs = pair_tgt_visibs = None
                pair_src_scale = pair_tgt_scale = None

            if "motion_mask" in results:
                motion_mask = results["motion_mask"].squeeze(-1) >= 0
            else:
                motion_mask = None

            if pair_src_scale is not None:
                if warp3d is not None:
                    warp3d[:, :len(pair_idx)] = warp3d[:, :len(pair_idx)] * pair_src_scale
                if warp3d_delta is not None:
                    warp3d_delta[:, :len(pair_idx)] = warp3d_delta[:, :len(pair_idx)] * pair_src_scale
            
            if scale is not None and global_pair_idx is not None:
                global_src_idxs = [pair[0] for pair in global_pair_idx]
                global_pair_scale = scale[:, global_src_idxs].squeeze(-1)  # (B, P_global, 1, 1)

                if warp3d is not None:
                    warp3d[:, len(pair_idx):] = warp3d[:, len(pair_idx):] * global_pair_scale
                if warp3d_delta is not None:
                    warp3d_delta[:, len(pair_idx):] = warp3d_delta[:, len(pair_idx):] * global_pair_scale

            def _cpu_float(t):
                return t.cpu().float() if t is not None else None

            track_3d_data = dict(
                query_uv=query.uv.cpu().float(),
                query_width=query.width,
                query_height=query.height,
                warp3d=_cpu_float(warp3d),
                warp2d=_cpu_float(warp2d),
                warp3d_delta=_cpu_float(warp3d_delta),
                src_trajs_3d_gt=_cpu_float(pair_src_trajs_3d),
                tgt_trajs_3d_gt=_cpu_float(pair_tgt_trajs_3d),
                src_trajs_2d_gt=_cpu_float(pair_src_trajs_2d),
                tgt_trajs_2d_gt=_cpu_float(pair_tgt_trajs_2d),
                src_valids_gt=_cpu_float(pair_src_valids),
                tgt_valids_gt=_cpu_float(pair_tgt_valids),
                src_visibs_gt=_cpu_float(pair_src_visibs),
                tgt_visibs_gt=_cpu_float(pair_tgt_visibs),
                motion_mask=motion_mask.cpu() if motion_mask is not None else None,
            )
        else:
            track_3d_data = None

        scale = scale.cpu() if scale is not None else None
        intrinsics = intrinsics.cpu() if intrinsics is not None else None
        extrinsics = extrinsics.cpu() if extrinsics is not None else None
        target_local_depth = target_local_depth.cpu() if target_local_depth is not None else None
        target_global_points = target_global_points.cpu() if target_global_points is not None else None
        target_depth_mask = target_depth_mask.cpu() if target_depth_mask is not None else None

        def get_single_view_data(data, index):
            return data[:, index] if data is not None else None

        # 逐 frame × view 处理: 反投影 + scale-align / denormalize + postprocess
        mv_outputs = []
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        for fi in range(frame_num):
            for vi in range(view_num):
                index = fi * view_num + vi

                pred_local_depth = pred_local_conf = None
                pred_global_points = pred_global_conf = None

                if query_depth is not None:
                    flat_depth = query_depth[0, index]
                    if int(flat_depth.numel()) == int(height * width):
                        pred_local_depth = flat_depth.view(1, 1, height, width)
                        pred_local_conf = query_conf[0, index].view(1, 1, height, width)
                    else:
                        # Sparse WorldTrack queries (Q << H*W): skip dense depth reshape.
                        pred_local_depth = pred_local_conf = None

                if pred_local_depth is not None:
                    if self.scale_align and target_local_depth is not None and target_depth_mask is not None:
                        if pred_local_depth.shape[-2:] != target_local_depth.shape[-2:]:
                            pred_local_depth_align = F.interpolate(pred_local_depth, target_local_depth.shape[-2:], mode="bilinear", align_corners=self.align_corners, antialias=False)
                            # pred_local_conf = F.interpolate(pred_local_conf, target_local_depth.shape[-2:], mode="bilinear", align_corners=self.align_corners, antialias=False)
                        else:
                            pred_local_depth_align = pred_local_depth

                        if pred_intrinsics is not None:
                            pred_local_depth = self.depth_to_points_from_meta(
                                pred_local_depth,
                                K=pred_intrinsics[:, index],
                                meta_data=meta_data,
                                device=pred_local_depth.device,
                                cache=False,
                                frame_index=index,
                            )
                        elif intrinsics is not None:
                            pred_local_depth = self.depth_to_points_from_meta(
                                pred_local_depth,
                                K=intrinsics[:, index],
                                meta_data=meta_data,
                                device=pred_local_depth.device,
                                cache=False,
                                frame_index=index,
                            )
                        else:
                            pred_local_depth = None

                        scale_factor = least_squares_scale_scalar(target_local_depth[:, index, -1][target_depth_mask[:, index, 0]], pred_local_depth_align[:, -1][target_depth_mask[:, index, 0]])
                        if pred_local_depth is not None:
                            pred_local_depth *= scale_factor
                    elif scale is not None:
                        intrinsics_for_depth = pred_intrinsics if pred_intrinsics is not None else intrinsics
                        if intrinsics_for_depth is not None:
                            pred_local_depth = self.depth_to_points_from_meta(
                                pred_local_depth,
                                K=intrinsics_for_depth[:, index],
                                meta_data=meta_data,
                                device=pred_local_depth.device,
                                cache=False,
                                frame_index=index,
                            )
                            pred_local_depth = self.denormalize(pred_local_depth, scale=scale[:, index])
                        else:
                            pred_local_depth = None

                if query_global_points is not None:
                    flat_global = query_global_points[0, index]
                    if int(flat_global.numel()) == int(3 * height * width):
                        pred_global_points = flat_global.view(1, 3, height, width)
                        pred_global_conf = query_global_conf[0, index].view(1, 1, height, width)
                    else:
                        pred_global_points = pred_global_conf = None

                if pred_global_points is not None:
                    if self.scale_align and target_global_points is not None and target_depth_mask is not None:
                        if pred_global_points.shape[-2:] != target_global_points.shape[-2:]:
                            pred_global_depth_align = F.interpolate(pred_global_points[:, -1].unsqueeze(1), target_global_points.shape[-2:], mode="bilinear", align_corners=self.align_corners, antialias=False)
                            # pred_global_conf = F.interpolate(pred_global_conf, target_global_points.shape[-2:], mode="bilinear", align_corners=self.align_corners, antialias=False)
                        else:
                            pred_global_depth_align = pred_global_points

                        scale_factor = least_squares_scale_scalar(target_global_points[:, index, -1][target_depth_mask[:, index, 0]], pred_global_depth_align[:, -1][target_depth_mask[:, index, 0]])
                        pred_global_points *= scale_factor
                    elif scale is not None:
                        pred_global_points = self.denormalize(pred_global_points, scale=scale[:, index])

                single = self.postprocess(
                    pred_local_points=pred_local_depth,
                    pred_local_conf=pred_local_conf,
                    pred_global_points=pred_global_points,
                    pred_global_conf=pred_global_conf,
                    pred_extrinsics=get_single_view_data(pred_extrinsics, index),
                    pred_intrinsics=get_single_view_data(pred_intrinsics, index),
                    image=get_single_view_data(image, index),
                    image_show=get_single_view_data(image_show, index),
                    scale=get_single_view_data(scale, index),
                    prompt_depth=get_single_view_data(prompt_depth, index),
                    target_local_depth=get_single_view_data(target_local_depth, index),
                    target_global_points=get_single_view_data(target_global_points, index),
                    target_depth_mask=get_single_view_data(target_depth_mask, index),
                    intrinsics=get_single_view_data(intrinsics, index),
                    extrinsics=get_single_view_data(extrinsics, index),
                    align_data=get_single_view_data(align_data, 0),
                )
                # extrinsics are already denormalized to metric above (consistent with
                # SparseMotionPipeline._build_outputs convention).  Clear prompt_scale so
                # that _get_c2w() in visualizer_3d.py does NOT multiply by scale again.
                single.prompt_scale = None

                single.rgb = (image[0, index].permute(1, 2, 0).float().cpu().numpy() + 1) * 0.5

                if gs_render_rgb is not None:
                    single.render_rgb = gs_render_rgb[index].permute(1, 2, 0).numpy()
                if gs_render_depth is not None:
                    single.render_depth = gs_render_depth[index].numpy()

                if dgs_render_rgb is not None and index < dgs_render_rgb.shape[1]:
                    single.dgs_render_rgb = (
                        dgs_render_rgb[0, index].float().permute(1, 2, 0).cpu().numpy()
                    )
                if dgs_render_depth is not None and index < dgs_render_depth.shape[1]:
                    single.dgs_render_depth = dgs_render_depth[0, index].float().cpu().numpy()

                if index == 0 and gaussians is not None:
                    single.gaussians = gaussians

                # 附加当前视角作为 source 的所有 match pair 结果
                if track_3d_data is not None:
                    cur_pairs = [(pi, p) for pi, p in enumerate(infer_pair_idx) if p[0] == index]
                    track_3d_results = dict[Any, Any]()
                    for pair_i, cur_pair in cur_pairs:
                        if track_3d_data is not None:
                            warp3d_out = track_3d_data["warp3d"][:, pair_i] if track_3d_data["warp3d"] is not None else None  # (B, Q, 3), Q = query_h * query_w
                            warp2d_out = track_3d_data["warp2d"][:, pair_i] if track_3d_data["warp2d"] is not None else None
                            motion_mask_out = track_3d_data["motion_mask"][:, pair_i] if track_3d_data["motion_mask"] is not None else None
                            warp3d_delta_out = track_3d_data["warp3d_delta"][:, pair_i] if track_3d_data["warp3d_delta"] is not None else None
                            src_points = None

                            query_h = track_3d_data["query_height"]
                            query_w = track_3d_data["query_width"]

                            src_2d_gt = None
                            if track_3d_data["src_trajs_2d_gt"] is not None:
                                uv_pair_i = pair_i
                                if (
                                    cur_pair[0] == cur_pair[1]
                                    and pair_i >= track_3d_data["src_trajs_2d_gt"].shape[1]
                                ):
                                    uv_pair_i = next(
                                        (
                                            pi
                                            for pi, p in enumerate(infer_pair_idx)
                                            if p[0] == cur_pair[0] and p[1] != p[0]
                                        ),
                                        None,
                                    )
                                if (
                                    uv_pair_i is not None
                                    and uv_pair_i < track_3d_data["src_trajs_2d_gt"].shape[1]
                                ):
                                    src_2d_gt = track_3d_data["src_trajs_2d_gt"][:, uv_pair_i]  # (B, N, 2) normalised

                            if src_2d_gt is not None:
                                col = torch.clamp(torch.round(src_2d_gt[..., 0] * (query_w - 1)).long(), 0, query_w - 1)
                                row = torch.clamp(torch.round(src_2d_gt[..., 1] * (query_h - 1)).long(), 0, query_h - 1)
                                flat_idx = row * query_w + col  # (B, N)

                                if warp3d_out is not None:
                                    warp3d_out = torch.gather(
                                        warp3d_out,
                                        1,
                                        flat_idx.unsqueeze(-1).expand(-1, -1, warp3d_out.shape[-1]),
                                    )  # (B, N, 3)
                                if warp2d_out is not None:
                                    warp2d_out = torch.gather(
                                        warp2d_out,
                                        1,
                                        flat_idx.unsqueeze(-1).expand(-1, -1, warp2d_out.shape[-1]),
                                    )  # (B, N, 2)

                                if motion_mask_out is not None:
                                    motion_mask_out = torch.gather(motion_mask_out, 1, flat_idx)  # (B, N)

                                if warp3d_delta_out is not None:
                                    warp3d_delta_out = torch.gather(
                                        warp3d_delta_out,
                                        1,
                                        flat_idx.unsqueeze(-1).expand(-1, -1, warp3d_delta_out.shape[-1]),
                                    )  # (B, N, 3)

                                    if (cur_pair[0], cur_pair[0]) in infer_pair_idx and track_3d_data["warp3d"] is not None:
                                        src_global_index = infer_pair_idx.index((cur_pair[0], cur_pair[0]))
                                        src_points = torch.gather(
                                            track_3d_data["warp3d"][0, src_global_index].view(-1, 3).unsqueeze(0),
                                            1,
                                            flat_idx.unsqueeze(-1).expand(-1, -1, warp3d_delta_out.shape[-1]),
                                        )  # (B, N, 3)
                                    elif single.local2glb_pointmap is not None:
                                        src_points = torch.gather(
                                            torch.from_numpy(single.local2glb_pointmap).view(-1, 3).unsqueeze(0),
                                            1,
                                            flat_idx.unsqueeze(-1).expand(-1, -1, warp3d_delta_out.shape[-1]),
                                        )  # (B, N, 3)

                                    if warp3d_out is None:
                                        warp3d_out = warp3d_delta_out + track_3d_data["src_trajs_3d_gt"][:, pair_i]

                                query_uv_out = src_2d_gt  # (B, N, 2)
                            else:
                                # 不走 src_trajs_2d_gt 索引，保持与 query 一致的 H,W flat 顺序 (Q = query_h * query_w)
                                # warp3d_out / warp2d_out / motion_mask_out / warp3d_delta_out 已经是 (B, Q, ...) 的 H,W flat 排布
                                ref_for_batch = (
                                    warp3d_out
                                    if warp3d_out is not None
                                    else (warp3d_delta_out if warp3d_delta_out is not None else warp2d_out)
                                )
                                batch_size = ref_for_batch.shape[0] if ref_for_batch is not None else 1
                                raw_query_uv = track_3d_data["query_uv"]
                                if raw_query_uv.ndim == 4:
                                    # (B, T, Q, 2) from sparse WorldTrack query bank
                                    query_uv_out = raw_query_uv[:, fi].expand(batch_size, -1, -1)
                                elif raw_query_uv.ndim == 3:
                                    query_uv_out = raw_query_uv.expand(batch_size, -1, -1)
                                else:
                                    query_uv_out = raw_query_uv.unsqueeze(0).expand(batch_size, -1, -1)

                                if warp3d_delta_out is not None:
                                    if (cur_pair[0], cur_pair[0]) in infer_pair_idx and track_3d_data["warp3d"] is not None:
                                        src_global_index = infer_pair_idx.index((cur_pair[0], cur_pair[0]))
                                        src_points = (
                                            track_3d_data["warp3d"][0, src_global_index]
                                            .view(-1, 3)
                                            .unsqueeze(0)
                                            .expand(batch_size, -1, -1)
                                        )  # (B, Q, 3)
                                    elif single.local2glb_pointmap is not None:
                                        src_points = (
                                            torch.from_numpy(single.local2glb_pointmap)
                                            .view(-1, 3)
                                            .unsqueeze(0)
                                            .expand(batch_size, -1, -1)
                                        )  # (B, Q, 3)

                                    if warp3d_out is None and src_points is not None:
                                        warp3d_out = warp3d_delta_out + src_points

                            if self.warp3d_delta_to_mask:
                                static_mask = warp3d_delta_out.abs() < 0.01
                                if src_points is not None:
                                    warp3d_out[static_mask] = src_points[static_mask]
                                elif track_3d_data["src_trajs_3d_gt"] is not None:
                                    warp3d_out[static_mask] = track_3d_data["src_trajs_3d_gt"][:, pair_i][static_mask]

                            warp3d_uv = None
                            if extrinsics is not None and intrinsics is not None and warp3d_out is not None:
                                tgt_w2c = extrinsics[:, cur_pair[1]]  # (B, 4, 4)
                                tgt_K = intrinsics[:, cur_pair[1]]  # (B, 3, 3)
                                ones = torch.ones(*warp3d_out.shape[:2], 1, dtype=warp3d_out.dtype)
                                warp3d_h = torch.cat([warp3d_out, ones], dim=-1)  # (B, ?, 4)
                                pts_cam = torch.einsum("bij,bqj->bqi", tgt_w2c, warp3d_h)[..., :3]
                                pts_proj = torch.einsum("bij,bqj->bqi", tgt_K, pts_cam)
                                warp3d_uv = pts_proj[..., :2] / (pts_proj[..., 2:3] + 1e-8)
                            elif pred_extrinsics is not None and pred_intrinsics is not None and warp3d_out is not None:
                                tgt_w2c = pred_extrinsics[:, cur_pair[1]]  # (B, 4, 4)
                                k_src = (
                                    pred_intrinsics_proj
                                    if pred_intrinsics_proj is not None
                                    else pred_intrinsics
                                )
                                tgt_K = k_src[:, cur_pair[1]]  # (B, 3, 3)
                                ones = torch.ones(*warp3d_out.shape[:2], 1, dtype=warp3d_out.dtype)
                                warp3d_h = torch.cat([warp3d_out, ones], dim=-1)  # (B, ?, 4)
                                pts_cam = torch.einsum("bij,bqj->bqi", tgt_w2c, warp3d_h)[..., :3]
                                pts_proj = torch.einsum("bij,bqj->bqi", tgt_K, pts_cam)
                                warp3d_uv = pts_proj[..., :2] / (pts_proj[..., 2:3] + 1e-8)

                            def _gt_field(key):
                                v = track_3d_data[key]
                                if v is None or pair_i >= v.shape[1]:
                                    return None
                                return v[0, pair_i]

                            track_3d_results[cur_pair[1]] = Track3DOutput(
                                query_uv=query_uv_out[0],
                                query_width=query_w,
                                query_height=query_h,
                                warp3d=warp3d_out[0],
                                warp3d_uv=warp3d_uv[0] if warp3d_uv is not None else None,
                                warp2d=warp2d_out[0] if warp2d_out is not None else None,
                                src_index=cur_pair[0],
                                tgt_index=cur_pair[1],
                                src_3d_gt=_gt_field("src_trajs_3d_gt"),
                                tgt_3d_gt=_gt_field("tgt_trajs_3d_gt"),
                                src_2d_gt=_gt_field("src_trajs_2d_gt"),
                                tgt_2d_gt=_gt_field("tgt_trajs_2d_gt"),
                                src_valids_gt=_gt_field("src_valids_gt"),
                                tgt_valids_gt=_gt_field("tgt_valids_gt"),
                                src_visibs_gt=_gt_field("src_visibs_gt"),
                                tgt_visibs_gt=_gt_field("tgt_visibs_gt"),
                                motion_mask=motion_mask_out[0] if motion_mask_out is not None else None,
                                warp3d_delta=warp3d_delta_out[0] if warp3d_delta_out is not None else None,
                                src_points=src_points[0] if src_points is not None else None,
                            )

                    single.track_3d = track_3d_results

                # Store frame and view indices
                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                mv_outputs.append(single)

        return mv_outputs

    def get_out_dir_new(self, out_dir, data_idx=None):
        gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir = super().get_out_dir(out_dir, data_idx=data_idx)

        if data_idx is not None:
            motion_out_dir = os.path.join(out_dir, f"motion/{data_idx:06d}")
        else:
            motion_out_dir = os.path.join(out_dir, f"motion")

        return gs_out_dir, motion_out_dir

    def get_gt_out_dir_new(self, out_dir):
        if self.save_output_cfg["gt_out_dir"] is None:
            return None, None, None, None

        abs_gt_out_dir = os.path.join(os.path.dirname(os.path.dirname(out_dir)), self.save_output_cfg["gt_out_dir"], os.path.basename(out_dir))
        _, gt_motion_out_dir = self.get_out_dir_new(abs_gt_out_dir, data_idx=None)

        return gt_motion_out_dir

    def visualize(self, outputs_list, meta_data, out_dir):
        """可视化推理结果 (V3): 在 V1 基础上增加 match 可视化, Gaussian 由本类直接处理.

        流程:
            1. 临时禁用 save_gaussians, 调用父类 (V1) visualize 完成通用可视化
               (跳过 V1 的 Gaussian 处理以避免重复)
            2. 恢复 save_gaussians 标志, 由本方法统一处理 Gaussian PLY 导出与渲染视频
            3. (新增) 保存 match warp 可视化结果
        """
        save_gaussians = self.save_output_cfg["save_gaussians"]

        self.save_output_cfg["save_gaussians"] = False
        super().visualize(outputs_list, meta_data, out_dir)
        self.save_output_cfg["save_gaussians"] = save_gaussians

        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        data_idx = meta_data["data_idx"][0]

        gs_out_dir, motion_out_dir = self.get_out_dir_new(out_dir, data_idx=data_idx)
        gt_motion_out_dir = self.get_gt_out_dir_new(out_dir)

        if self.save_output_cfg.get("save_4dgs_results"):
            from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import (
                vis_sparse_4dgs_results,
            )

            dgs_out_dir = os.path.join(out_dir, f"4dgs/{data_idx:06d}")
            if any(getattr(o, "dgs_render_rgb", None) is not None for o in outputs_list):
                vis_sparse_4dgs_results(
                    outputs_list=outputs_list,
                    dgs_out_dir=dgs_out_dir,
                    data_idx=data_idx,
                )

        if self.save_output_cfg.get("save_motion"):
            os.makedirs(motion_out_dir, exist_ok=True)
            vis_motion_results(
                cfg=self.save_output_cfg,
                mv_outputs=outputs_list,
                motion_out_dir=motion_out_dir,
                data_idx=data_idx,
                frame_num=frame_num,
                view_num=view_num,
                gt_out_dir=gt_motion_out_dir,
            )

        if self.save_output_cfg.get("save_motion_rerun"):
            try:
                from hAlgorithm.modules.pipelines2.utils.visualize_dynamic_rerun import (
                    vis_dynamic_points_rerun,
                )

                rerun_cfg = dict(self.save_output_cfg.get("rerun_vis_cfg") or {})
                nq = self.save_output_cfg.get("rerun_max_queries")
                vis_dynamic_points_rerun(
                    rerun_cfg,
                    outputs_list,
                    out_dir,
                    data_idx,
                    meta_data,
                    max_queries=nq,
                )
            except Exception as e:
                logging.getLogger(__name__).error(
                    "[WFMQueryPipeline.visualize] Rerun motion export failed: %s",
                    e,
                    exc_info=True,
                )

        # Gaussians Visualization
        if self.save_output_cfg["save_gaussians"] and outputs_list[0].gaussians is not None:
            vis_render(cfg=self.save_output_cfg, mv_outputs=outputs_list, gs_out_dir=gs_out_dir, data_idx=data_idx)

            def vis_render_video(cfg, render_fn, mv_outputs, gs_out_dir, data_idx, frame_num, view_num, device="cpu"):
                os.makedirs(gs_out_dir, exist_ok=True)
                gaussians = mv_outputs[0].gaussians
                save_path = os.path.join(gs_out_dir, f"gaussians_{data_idx:06d}.ply")

                means = gaussians["means"].squeeze(0)
                harmonics = gaussians["harmonics"].squeeze(0)
                opacities = gaussians["opacities"].squeeze(0)
                scales = gaussians["scales"].squeeze(0)
                rotations = gaussians["rotations"].squeeze(0)

                export_ply(
                    means=means,
                    harmonics=harmonics,
                    opacities=opacities,
                    path=save_path,
                    scales=scales,
                    rotations=rotations,
                    # focal_length_px=(intrinsics[0, 0, 0], intrinsics[0, 1, 1]),
                    # principal_point_px=(intrinsics[0, 0, 2], intrinsics[0, 1, 2]),
                    # image_shape=(height, width),
                    # extrinsic_matrix=extrinsics[0],
                )

                if cfg["save_render_video"]:
                    # if mv_outputs[0].extrinsics is None or mv_outputs[0].intrinsics is None or cfg["render_video_with_pred_camera"]:
                    #     extrinsics = torch.stack([torch.from_numpy(outputs.extrinsics_pred) for outputs in mv_outputs], dim=0)
                    #     intrinsics = torch.stack([torch.from_numpy(outputs.intrinsics_pred) for outputs in mv_outputs], dim=0)
                    # else:
                    #     extrinsics = torch.stack([torch.from_numpy(outputs.extrinsics) for outputs in mv_outputs], dim=0)
                    #     intrinsics = torch.stack([torch.from_numpy(outputs.intrinsics) for outputs in mv_outputs], dim=0)

                    num_views = len(mv_outputs)
                    extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(num_views, -1, -1).clone()
                    intrinsics = torch.stack([torch.from_numpy(outputs.intrinsics) for outputs in mv_outputs], dim=0)

                    if cfg["save_render_video_with_normalize_c2w"]:
                        scale = torch.tensor([outputs.prompt_scale for outputs in mv_outputs]).unsqueeze(-1)
                        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] / scale

                    extrinsics = extrinsics.reshape(-1, frame_num * view_num, 4, 4).float().to(device)
                    intrinsics = intrinsics.reshape(-1, frame_num * view_num, 3, 3).float().to(device)

                    trajectory = cfg.get("render_video_trajectory", "orbit")
                    render_video = render_video_interpolation(
                        render_fn=render_fn,
                        gaussians=gaussians,
                        w2c=extrinsics,
                        intrinsics=intrinsics,
                        h=mv_outputs[0].pointmap_h,
                        w=mv_outputs[0].pointmap_w,
                        trajectory=trajectory,
                        n_interp=cfg.get("render_video_n_interp", 24),
                        radius=cfg.get("render_video_radius", 0.5),
                        vertical=cfg.get("render_video_vertical", 0.1),
                        forward_amp=cfg.get("render_video_forward_amp", 0.2),
                        device=device,
                    )
                    if render_video is not None:
                        save_video(render_video, gs_out_dir, "video", data_idx, info=True)

            vis_render_video(
                cfg=self.save_output_cfg, render_fn=self.render_gaussians, mv_outputs=outputs_list, gs_out_dir=gs_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num, device=self.device
            )
