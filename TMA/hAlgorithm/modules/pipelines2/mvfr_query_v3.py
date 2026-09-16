import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.modules.models2.external.depth_anything_3.utils.alignment import least_squares_scale_scalar
from hAlgorithm.modules.models2.gaussians.infinidepth_ply import export_ply
from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from hAlgorithm.modules.pipelines2.utils.outputs import DenseMatchingOutput
from hAlgorithm.modules.pipelines2.utils.render_util import render_video_interpolation
from hAlgorithm.modules.pipelines2.utils.visualize_mv import (
    save_video,
    vis_match_results,
    vis_render,
)
from hAlgorithm.utils import instantiate_from_config

from .mvfr_query_v1 import MVFRQueryPipeline as BaseMVFRQueryPipeline
import random

class MVFRQueryPipeline(BaseMVFRQueryPipeline):
    """基于 Query 的多视角前馈重建 Pipeline (V3 — Local + Global + Camera + FFGS + Match).

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
    """

    def __init__(
        self,
        pair_mode=None,
        query_match_loss=None,
        **kwargs,
    ):
        super(MVFRQueryPipeline, self).__init__(**kwargs)

        self.pair_mode = pair_mode
        self.query_match_loss = instantiate_from_config(query_match_loss) if query_match_loss is not None else None

        self.save_output_cfg.setdefault("save_match", self.save_output_cfg["save_everything"])

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

        return None

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
        return branches

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
            align_corners=False,
        )
        gt_mask_sampled = F.grid_sample(
            mask_gt_chw.float(),
            uv_grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
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

    def train_step(self, batch):
        """训练单步 (V3): 前向 → 计算 5 部分 loss (Local + Global + Camera + Gaussian + Match).

        Loss 组成:
            Part1 (lcl)   — 局部深度 loss: query 点在相机坐标系下的深度/点云监督
            Part2 (glb)   — 全局点云 loss: query 点在世界坐标系下的 3D 点监督
            Part3 (cm)    — 相机 loss:   pose_encoding → extrinsics/intrinsics 的监督
            Part4 (rc)    — 渲染 loss:   Gaussian splatting 渲染的 RGB + 深度重建监督
            Part5 (match) — 匹配 loss:   视角间 warp 预测 vs GT warp (由深度 + 外参计算)

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

        # 提取模型输出: local depth, global points, confidence 及 pose_encoding
        # 各输出可能为 None (取决于模型配置), 后续按需计算对应 loss
        query_depth = results.get("depth", None)
        query_conf = results.get("confidence", None)
        query_global_points = results.get("global_points", None)
        query_global_conf = results.get("global_confidence", None)

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

        # 构建 grid_sample 采样坐标, 从 [0,1] 归一化到 [-1,1]
        query = results["query"]
        if query.uv is None:
            query_uv_grid = query.batch_uv
            query_uv_grid = query_uv_grid * 2 - 1
            query_uv_grid = query_uv_grid.unsqueeze(2)  # (1, Q, 1, 2)
        else:
            query_uv_grid = query.uv
            query_uv_grid = query_uv_grid * 2 - 1
            query_uv_grid = query_uv_grid.unsqueeze(0).unsqueeze(2)  # (1, Q, 1, 2)
            query_uv_grid = query_uv_grid.expand(B * N, -1, -1, -1)  # (B*N, Q, 1, 2)

        if self.pose_encoding_type == "absT_quaR_FoV":
            pose_enc = results.get("pose_enc")
        else:
            raise NotImplementedError

        total_loss, total_loss_dict = 0, dict()

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
            target_query_depth = F.grid_sample(target_local_depth, query_uv_grid, mode="bilinear", padding_mode="border", align_corners=False)
            target_query_valid_mask = F.grid_sample(target_depth_mask.float(), query_uv_grid, mode="bilinear", padding_mode="border", align_corners=False)

            if query.full_uv:
                # (B, N, C, Q, 1) -> (B, N, C, H, W)
                query_depth = query_depth.squeeze(-1).view(*query_depth.shape[:3], query.height, query.width)
                query_conf = query_conf.squeeze(-1).view(*query_conf.shape[:3], query.height, query.width)

                if query_depth.shape[-3] == 1:
                    query_depth = self.depth_to_points(query_depth, K=intrinsics)
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
            target_query_global_points = F.grid_sample(target_global_points, query_uv_grid, mode="bilinear", padding_mode="border", align_corners=False)
            target_query_valid_mask = F.grid_sample(target_depth_mask.float(), query_uv_grid, mode="bilinear", padding_mode="border", align_corners=False)

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
                gs_render_depth = self.depth_to_points(gs_render_depth, K=intrinsics)

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

        # ── Part5: Match warp loss ────────────────────────────────────────
        # 基于 dense matching 的 warp 预测监督: 利用 GT 深度 + 相机参数计算 GT warp,
        # 再与模型预测的 warp 做 loss
        if self.query_match_loss is not None and pair_idx is not None and "warp2d" in results:
            num_pair = len(pair_idx)

            branches = self.get_match_prediction_branches(results)
            # reshape target_local_depth to (B, N, C, H, W)
            target_local_depth = target_local_depth.view(B, N, *target_local_depth.shape[-3:])

            # reshape target_local_depth to (B, N, C, H, W)
            target_local_depth = target_local_depth.view(B, N, *target_local_depth.shape[-3:])

            # 反归一化 GT 深度和外参平移量到真实尺度, 用于计算 GT warp
            target_match_gt_depth = self.denormalize(target_local_depth, scale=scale)
            target_match_gt_extrinsics = extrinsics.clone()
            target_match_gt_extrinsics[..., :3, 3] = self.denormalize(extrinsics[..., :3, 3], scale=scale[..., 0, 0])
            target_match_gt_depth_intrinsics = intrinsics.clone()

            depth_height, depth_width = target_match_gt_depth.shape[-2:]

            # 训练时输出分辨率与GT不同时, 按比例缩放内参
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
            gt_warp_sampled, gt_mask_sampled = self.sample_match_gt_at_query(warp_gt, mask_gt, query, B, num_pair, depth_height, depth_width)

            match_weight = self.task_weight.get("match", 1.0)
            total_loss_dict["match"] = 0
            for branch_name, branch in branches.items():
                pred_warp_flat = branch["warp2d"].view(B * num_pair, -1, 2)
                pred_conf_flat = branch["warp2d_confidence"].view(B * num_pair, -1, branch["warp2d_confidence"].shape[-1])
                loss, loss_dict = self.query_match_loss(
                    pred_warp=pred_warp_flat,
                    pred_conf=pred_conf_flat,
                    gt_warp=gt_warp_sampled,
                    gt_mask=gt_mask_sampled,
                    gt_h=depth_height,
                    gt_w=depth_width,
                    name=name,
                )

                branch_total = loss * match_weight
                total_loss += branch_total
                total_loss_dict[f"match_{branch_name}"] = branch_total
                total_loss_dict["match"] += branch_total
                for key, val in loss_dict.items():
                    total_loss_dict[f"match_{branch_name}_{key}"] = val * match_weight

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

        meta_data["sub_pixel_scale"] = self.testing_sub_pixel_scale

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
                    pair_idx=pair_idx,
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
                pair_idx=pair_idx,
            )

        # 提取预测: local depth, global points, confidence
        query_depth = results.get("depth", None)
        query_conf = results.get("confidence", None)
        query_global_points = results.get("global_points", None)
        query_global_conf = results.get("global_confidence", None)

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

        # 反归一化: 将 scale-normalized 的 GT 深度 / 点云 / 外参平移量恢复到真实尺度
        if target_local_depth is not None and scale is not None:
            target_local_depth = self.denormalize(target_local_depth, scale=scale)
        if target_global_points is not None and scale is not None:
            target_global_points = self.denormalize(target_global_points, scale=scale)
        if extrinsics is not None and scale is not None:
            extrinsics[..., :3, 3] = self.denormalize(extrinsics[..., :3, 3], scale=scale[..., 0, 0])

        if target_depth_mask is not None and target_depth_mask.ndim == 4:
            target_depth_mask = target_depth_mask.unsqueeze(-3)

        # Match输出分辨率与GT不同时，按比例缩放 Match GT 内参
        if "warp2d" in results and pair_idx is not None and target_local_depth is not None:
            target_match_gt_depth_intrinsics = intrinsics.clone()
            
            depth_height, depth_width = target_local_depth.shape[-2:]
            x_scale = depth_width / int(meta_data["input_width"][0])
            y_scale = depth_height / int(meta_data["input_height"][0])

            target_match_gt_depth_intrinsics[..., 0, :] *= x_scale
            target_match_gt_depth_intrinsics[..., 1, :] *= y_scale

        # 训练时输出分辨率与GT不同时, 按比例缩放内参
        if self.testing_sub_pixel_scale != 1:
            x_scale = width / int(meta_data["input_width"][0])
            y_scale = height / int(meta_data["input_height"][0])
            intrinsics[..., 0, :] *= x_scale
            intrinsics[..., 1, :] *= y_scale

            if pred_intrinsics is not None:
                pred_intrinsics[..., 0, :] *= x_scale
                pred_intrinsics[..., 1, :] *= y_scale

        # FFGS rendering
        gaussians = gs_render_rgb = gs_render_depth = None
        if "gaussians" in results and results["gaussians"] is not None:
            gaussians = results["gaussians"]
            gs_render_rgb, gs_render_depth = self.render_gaussians(gaussians, intrinsics=intrinsics, w2c=w2c, image_shape=(height, width))
            if gs_render_rgb is not None:
                gs_render_rgb = gs_render_rgb.cpu()
                gs_render_depth = gs_render_depth.cpu()

        # ── 提取 match 预测 (在其他结果移至 CPU 之前) ─────────────────────
        if "warp2d" in results and pair_idx is not None:
            B_m = image.shape[0]
            num_pair = len(pair_idx)
            query = results["query"]

            # 提取各分支 (coarse + refiners) 的 warp 和 confidence
            match_branches = self.get_match_prediction_branches(results)
            match_final_name, match_refiner_names = self.get_match_final_prediction_branch(match_branches)
            match_final_branch = match_branches[match_final_name]
            match_coarse_branch = match_branches["coarse"]

            warp_AB = match_final_branch["warp2d"]
            conf_AB = match_final_branch["warp2d_confidence"]
            coarse_warp_AB = match_coarse_branch["warp2d"]
            coarse_conf_AB = match_coarse_branch["warp2d_confidence"]

            # full_uv 模式下 reshape 为空间网格形式
            if query.full_uv:
                Q_h, Q_w = query.height, query.width
                warp_AB = warp_AB.view(B_m, num_pair, Q_h, Q_w, 2)
                conf_AB = conf_AB.view(B_m, num_pair, Q_h, Q_w, -1)
                coarse_warp_AB = coarse_warp_AB.view(B_m, num_pair, Q_h, Q_w, 2)
                coarse_conf_AB = coarse_conf_AB.view(B_m, num_pair, Q_h, Q_w, -1)

            if len(match_refiner_names) > 0:
                warp_refine = {n.split("_")[1]: match_branches[n]["warp2d"] for n in match_refiner_names}
                if query.full_uv:
                    warp_refine = {s: w.view(B_m, num_pair, Q_h, Q_w, 2) for s, w in warp_refine.items()}
            else:
                warp_refine = None

            # 从 confidence 中拆分 overlap (二值化) 和 precision
            overlap_AB = conf_AB[..., :1]
            precision_AB = conf_AB[..., 1:4]
            coarse_overlap_AB = coarse_conf_AB[..., :1]

            if target_local_depth is not None and extrinsics is not None:
                warp_gt, mask_gt = self.compute_gt_warp_at_scale1(
                    pair_idx,
                    target_local_depth,
                    target_match_gt_depth_intrinsics,
                    extrinsics,
                    camera_type=camera_type,
                    meta_data=meta_data,
                )
                warp_gt, mask_gt = self.sample_match_gt_at_query(warp_gt, mask_gt, query, B_m, num_pair, depth_height, depth_width)

                warp_gt = warp_gt.reshape(B_m, num_pair, query.height, query.width, 2)
                mask_gt = mask_gt.reshape(B_m, num_pair, query.height, query.width)
            else:
                warp_gt = mask_gt = None

            match_data = dict(
                warp_AB=warp_AB,
                coarse_warp_AB=coarse_warp_AB,
                warp_refine=warp_refine,
                overlap_AB=overlap_AB,
                precision_AB=precision_AB,
                coarse_overlap_AB=coarse_overlap_AB,
                warp_gt=warp_gt,
                mask_gt=mask_gt,
            )
        else:
            match_data = None

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
                    pred_local_depth = query_depth[0, index].view(1, 1, height, width)
                    pred_local_conf = query_conf[0, index].view(1, 1, height, width)

                    if self.scale_align:
                        if pred_local_depth.shape[-2:] != target_local_depth.shape[-2:]:
                            pred_local_depth_align = F.interpolate(pred_local_depth, target_local_depth.shape[-2:], mode="bilinear", align_corners=False, antialias=False)
                            # pred_local_conf = F.interpolate(pred_local_conf, target_local_depth.shape[-2:], mode="bilinear", align_corners=False, antialias=False)
                        else:
                            pred_local_depth_align = pred_local_depth

                        if pred_intrinsics is not None:
                            pred_local_depth = self.depth_to_points(pred_local_depth, K=pred_intrinsics[:, index], device=pred_local_depth.device, cache=False)
                        else:
                            pred_local_depth = self.depth_to_points(pred_local_depth, K=intrinsics[:, index], device=pred_local_depth.device, cache=False)

                        scale_factor = least_squares_scale_scalar(target_local_depth[:, index, -1][target_depth_mask[:, index, 0]], pred_local_depth_align[:, -1][target_depth_mask[:, index, 0]])
                        pred_local_depth *= scale_factor
                    elif scale is not None:
                        pred_local_depth = self.depth_to_points(pred_local_depth, K=intrinsics[:, index], device=pred_local_depth.device, cache=False)
                        pred_local_depth = self.denormalize(pred_local_depth, scale=scale[:, index])

                if query_global_points is not None:
                    pred_global_points = query_global_points[0, index].view(1, 3, height, width)
                    pred_global_conf = query_global_conf[0, index].view(1, 1, height, width)

                    if self.scale_align:
                        if pred_global_points.shape[-2:] != target_global_points.shape[-2:]:
                            pred_global_depth_align = F.interpolate(pred_global_points[:, -1].unsqueeze(1), target_global_points.shape[-2:], mode="bilinear", align_corners=False, antialias=False)
                            # pred_global_conf = F.interpolate(pred_global_conf, target_global_points.shape[-2:], mode="bilinear", align_corners=False, antialias=False)
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

                if index == 0 and gaussians is not None:
                    single.gaussians = gaussians

                # 附加当前视角作为 source 的所有 match pair 结果
                if match_data is not None:
                    cur_pairs = [(pi, p) for pi, p in enumerate(pair_idx) if p[0] == index]
                    matching_results = []
                    C_s = image_show.shape[2] if image_show is not None else 3
                    H_s = image_show.shape[3] if image_show is not None else image.shape[-2]
                    W_s = image_show.shape[4] if image_show is not None else image.shape[-1]
                    for pair_i, cur_pair in cur_pairs:
                        img0 = np.ascontiguousarray(image_show[:, cur_pair[0]].reshape(-1, C_s, H_s, W_s).transpose(0, 2, 3, 1).astype(np.uint8)) if image_show is not None else None
                        img1 = np.ascontiguousarray(image_show[:, cur_pair[1]].reshape(-1, C_s, H_s, W_s).transpose(0, 2, 3, 1).astype(np.uint8)) if image_show is not None else None
                        matching_results.append(
                            DenseMatchingOutput(
                                image0=img0,
                                image1=img1,
                                warp=match_data["warp_AB"][:, pair_i],
                                warp_coarse=match_data["coarse_warp_AB"][:, pair_i],
                                warp_refine={s: w[:, pair_i] for s, w in match_data["warp_refine"].items()} if match_data["warp_refine"] is not None else None,
                                overlap=match_data["overlap_AB"][:, pair_i],
                                overlap_coarse=match_data["coarse_overlap_AB"][:, pair_i],
                                warp_gt=match_data["warp_gt"][:, pair_i] if match_data["warp_gt"] is not None else None,
                                overlap_gt=match_data["mask_gt"][:, pair_i][..., None] if match_data["mask_gt"] is not None else None,
                                pred_covariance=match_data["precision_AB"][:, pair_i],
                                extrinsics_image0=extrinsics[:, cur_pair[0]][0].numpy() if extrinsics is not None else None,
                                extrinsics_image1=extrinsics[:, cur_pair[1]][0].numpy() if extrinsics is not None else None,
                                intrinsics_image0=intrinsics[:, cur_pair[0]][0].numpy() if intrinsics is not None else None,
                                intrinsics_image1=intrinsics[:, cur_pair[1]][0].numpy() if intrinsics is not None else None,
                            )
                        )
                    single.dense_matching = matching_results

                # Store frame and view indices
                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                mv_outputs.append(single)

        return mv_outputs

    def get_out_dir_new(self, out_dir, data_idx=None):
        gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir = super().get_out_dir(out_dir, data_idx=data_idx)

        if data_idx is not None:
            match_out_dir = os.path.join(out_dir, f"match/{data_idx:06d}")
        else:
            match_out_dir = os.path.join(out_dir, "match")

        return gs_out_dir, match_out_dir

    def get_gt_out_dir_new(self, out_dir):
        if self.save_output_cfg["gt_out_dir"] is None:
            return None, None, None, None

        abs_gt_out_dir = os.path.join(os.path.dirname(os.path.dirname(out_dir)), self.save_output_cfg["gt_out_dir"], os.path.basename(out_dir))
        _, gt_match_out_dir = self.get_out_dir_new(abs_gt_out_dir, data_idx=None)

        return gt_match_out_dir

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

        gs_out_dir, match_out_dir = self.get_out_dir_new(out_dir, data_idx=data_idx)
        gt_match_out_dir = self.get_gt_out_dir_new(out_dir)

        if self.save_output_cfg.get("save_match"):
            os.makedirs(match_out_dir, exist_ok=True)
            vis_match_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, match_out_dir=match_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num, gt_out_dir=gt_match_out_dir, overlap_act=True)

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
