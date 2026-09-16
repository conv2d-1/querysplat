import logging
import os

import torch
import torch.nn.functional as F

from hAlgorithm.modules.models2.gaussians.infinidepth_ply import export_ply
from hAlgorithm.modules.pipelines2.mvfr_v1 import MVFRPipeline
from hAlgorithm.modules.pipelines2.utils.render_util import render_video_interpolation
from hAlgorithm.modules.pipelines2.utils.visualize_mv import (
    save_video,
    vis_render,
)
from hAlgorithm.modules.utils.ray import get_rays_in_camera_frame, get_rays_in_world_frame


class MVFRQueryPipeline(MVFRPipeline):
    """基于 Query 的多视角前馈重建 Pipeline (V1 — 仅支持 Local 坐标深度和 FFGS).

    在 MVFRPipeline 的基础上引入 query-based 预测范式：模型不再输出密集深度图，
    而是针对稀疏/密集 UV query 点预测对应的深度值与置信度。

    主要流程:
        1. get_inputs   — 从 batch 中提取图像、相机参数、GT 深度等，并做归一化 / 维度对齐
        2. train_step   — 前向 → 采样 GT → 计算 local depth loss + (可选) Gaussian 渲染 loss
        3. infer        — 前向 → 反归一化 / scale-align → 逐视角 postprocess → 输出列表
        4. visualize    — 调用父类可视化 + Gaussian PLY 导出 & 渲染视频

    Args:
        edge_mask_name: batch 中边缘 mask 的字段名，用于辅助训练时遮挡处理。
        training_sub_pixel_scale: 训练时 sub-pixel 缩放因子 (-1 = 原始分辨率)。
        testing_sub_pixel_scale:  推理时 sub-pixel 缩放因子 (-1 = 原始分辨率, 1 = 输入分辨率)。
        scale_align: 推理时是否通过最小二乘对齐预测深度到 GT 尺度。
        scale_shift_align: (保留) 是否做 scale-shift 对齐。
        rgb_patch_scale: 0 = 不使用 query_rgb; 1 = 复用输入图像; -1 = 使用原始未归一化图像。
    """

    def __init__(
        self,
        edge_mask_name=None,
        training_sub_pixel_scale=-1,
        testing_sub_pixel_scale=-1,
        scale_align=False,
        scale_shift_align=False,
        rgb_patch_scale=0,
        **kwargs,
    ):
        super(MVFRQueryPipeline, self).__init__(**kwargs)

        self.edge_mask_name = edge_mask_name

        # -1 → 按原图分辨率输出; 1 → 按网络输入分辨率输出
        self.training_sub_pixel_scale = training_sub_pixel_scale
        self.testing_sub_pixel_scale = testing_sub_pixel_scale
        self.rgb_patch_scale = rgb_patch_scale

        self.scale_align = scale_align
        self.scale_shift_align = scale_shift_align

    def get_inputs(self, batch):
        """从 dataloader batch 中提取并预处理所有输入张量.

        处理逻辑:
            1. 图像 & query_rgb 准备 (含可选的 raw image / backup image)
            2. 相机内参、外参读取 + 平移量归一化 (除以 scale)
            3. (可选) 生成相机坐标系 / 世界坐标系射线方向
            4. GT 深度 & 点云归一化, mask 读取
            5. prompt depth 归一化 & 可选的 mean/max 归一化
            6. invalid mask 应用: 将 invalid 区域置零
            7. 若输入为 SV (4D), 自动添加 view 维度统一为 MV (5D) 格式

        Returns:
            长度为 22 的元组，包含 name, total_iter, meta_data, image, edge_mask,
            query_image, intrinsics, extrinsics, scale, prompt_depth,
            target_local_depth, target_global_points, target_depth_mask,
            target_normal, target_normal_mask, target_motion_mask,
            target_invalid_mask, image_show, align_data, ray_directions,
            ray_world, extrinsics_noise, prompt_extrinsics.
        """
        meta_data = batch["meta_data"]
        name = meta_data["name"][0]
        total_iter = batch.get("total_iter")

        if total_iter is not None:
            meta_data["total_iter"] = total_iter

        image = batch["image"].to(device=self.device)

        query_image = None
        if self.rgb_patch_scale != 0:
            if self.rgb_patch_scale == 1:
                query_image = image.clone()
            if self.rgb_patch_scale == -1:
                query_image = batch["image_raw"].to(device=self.device)
                if query_image.ndim == 4:
                    query_image = query_image.permute(0, 3, 1, 2).contiguous()
                elif query_image.ndim == 5:
                    query_image = query_image.permute(0, 1, 4, 2, 3).contiguous()
                else:
                    raise NotImplementedError
                query_image = query_image / 255.0

        image_backup = None
        if "image_backup" in batch:
            image_backup = batch["image_backup"].to(device=self.device)

        edge_mask = None
        if self.edge_mask_name is not None and self.edge_mask_name in batch:
            edge_mask = batch[self.edge_mask_name].to(device=self.device)

        scale = intrinsics = ray_directions = extrinsics = ray_world = None
        extrinsics_noise = prompt_extrinsics = None

        if self.scale_name is not None and self.scale_name in batch:
            scale = batch[self.scale_name].to(self.device)[..., None, None, None]

        if self.intrinsics_name is not None and self.intrinsics_name in batch:
            intrinsics = batch[self.intrinsics_name].to(device=self.device)

            if self.with_ray_directions:
                w = meta_data["input_width"][0].item()
                h = meta_data["input_height"][0].item()
                ray_directions = get_rays_in_camera_frame(
                    intrinsics=intrinsics.reshape(-1, 3, 3),
                    height=h,
                    width=w,
                    normalize_to_unit_sphere=True,
                )
                if image.ndim == 5:
                    ray_directions = ray_directions.view(*intrinsics.shape[:2], 3, h, w)
                else:
                    ray_directions = ray_directions.view(*intrinsics.shape[0], 3, h, w)

        if self.extrinsics_name is not None and self.extrinsics_name in batch:
            extrinsics = batch[self.extrinsics_name].to(device=self.device)
            if scale is not None:
                extrinsics[..., :3, 3] = self.normalize(extrinsics[..., :3, 3], scale[..., 0, 0])

            if self.with_ray_in_world:
                w = meta_data["input_width"][0].item()
                h = meta_data["input_height"][0].item()
                ray_origins_world, ray_directions_world = get_rays_in_world_frame(
                    intrinsics=intrinsics.reshape(-1, 3, 3),
                    height=h,
                    width=w,
                    normalize_to_unit_sphere=True,
                    camera_pose=extrinsics.reshape(-1, 4, 4).inverse(),  # NOTE: camera_pose is camera2word, extrinsics is word2camera
                )
                if image.ndim == 5:
                    ray_origins_world = ray_origins_world.view(*intrinsics.shape[:2], 3, h, w)
                    ray_directions_world = ray_directions_world.view(*intrinsics.shape[:2], 3, h, w)
                    ray_world = torch.cat([ray_origins_world, ray_directions_world], dim=2)
                else:
                    ray_origins_world = ray_origins_world.view(*intrinsics.shape[0], 3, h, w)
                    ray_directions_world = ray_directions_world.view(*intrinsics.shape[0], 3, h, w)
                    ray_world = torch.cat([ray_origins_world, ray_directions_world], dim=1)

            if self.prompt_extrinsics_name is not None and self.prompt_extrinsics_name in batch:
                prompt_extrinsics = batch[self.prompt_extrinsics_name].to(device=self.device)
                if scale is not None:
                    prompt_extrinsics[..., :3, 3] = self.normalize(prompt_extrinsics[..., :3, 3], scale[..., 0, 0])
            elif self.extrinsics_noise_name is not None and self.extrinsics_noise_name in batch:
                extrinsics_noise = batch[self.extrinsics_noise_name].to(device=self.device)

        target_local_depth = target_global_points = target_depth_mask = None

        if self.target_local_depth_name is not None and self.target_local_depth_name in batch:
            target_local_depth = batch[self.target_local_depth_name].to(device=self.device)
            if scale is not None:
                target_local_depth = self.normalize(target_local_depth, scale)

        if self.target_global_points_name is not None and self.target_global_points_name in batch:
            target_global_points = batch[self.target_global_points_name].to(device=self.device)
            if scale is not None:
                target_global_points = self.normalize(target_global_points, scale)

        if self.target_depth_mask_name is not None and self.target_depth_mask_name in batch:
            target_depth_mask = batch[self.target_depth_mask_name].to(self.device)

        prompt_depth = prompt_depth_mask = None

        if self.prompt_depth_name is not None and self.prompt_depth_name in batch:
            if self.prompt_depth_name == self.target_local_depth_name:
                prompt_depth = target_local_depth.clone() if target_local_depth is not None else None
            else:
                prompt_depth = batch[self.prompt_depth_name].to(self.device)
                if scale is not None:
                    prompt_depth = self.normalize(prompt_depth, scale)

            if self.prompt_depth_mask_name is not None and self.prompt_depth_mask_name in batch:
                if self.prompt_depth_mask_name == self.target_depth_mask_name:
                    prompt_depth_mask = target_depth_mask.clone() if target_depth_mask is not None else None
                else:
                    prompt_depth_mask = batch[self.prompt_depth_mask_name].to(self.device)

        target_normal = target_normal_mask = target_motion_mask = target_invalid_mask = None

        if self.target_normal_name is not None and self.target_normal_name in batch:
            target_normal = batch[self.target_normal_name].to(self.device)

        if self.target_normal_mask_name is not None and self.target_normal_mask_name in batch:
            target_normal_mask = batch[self.target_normal_mask_name].to(self.device)

        if self.target_motion_mask_name is not None and self.target_motion_mask_name in batch:
            target_motion_mask = batch[self.target_motion_mask_name].to(self.device)

        if self.target_invalid_mask_name is not None and self.target_invalid_mask_name in batch:
            target_invalid_mask = batch[self.target_invalid_mask_name].to(self.device)

        image_show = align_data = None

        if not self.training:
            if "image_show" in batch:
                image_show = batch["image_show"]
                image_show = image_show.float().numpy()

            if self.save_output_cfg["output_match_input_res"] and self.align_name in batch:
                align_data = batch[self.align_name]

        if prompt_depth is not None and self.prompt_depth_normalize is not None:
            if self.prompt_depth_normalize == "mean":
                for i in range(image.shape[0]):
                    prompt_depth[i] /= prompt_depth[i][prompt_depth_mask[i].expand(-1, prompt_depth.shape[1], -1, -1)].mean() + 1e-8
            elif self.prompt_depth_normalize == "max":
                for i in range(image.shape[0]):
                    prompt_depth[i] /= (prompt_depth[i] * prompt_depth_mask[i]).max()
            else:
                raise NotImplementedError

        if prompt_depth is not None and prompt_depth_mask is not None:
            prompt_depth = torch.cat([prompt_depth, prompt_depth_mask], dim=-3)

        if target_invalid_mask is not None:

            def with_invalid_mask(x):
                return (x * (~target_invalid_mask)).to(x.dtype) if x is not None else None

            prompt_depth = with_invalid_mask(prompt_depth)
            prompt_depth_mask = with_invalid_mask(prompt_depth_mask)
            target_local_depth = with_invalid_mask(target_local_depth)
            target_global_points = with_invalid_mask(target_global_points)
            target_depth_mask = with_invalid_mask(target_depth_mask)
            target_normal = with_invalid_mask(target_normal)
            target_normal_mask = with_invalid_mask(target_normal_mask)

        # NOTE: sv data -> mv data
        if image.ndim == 4:
            meta_data["frames"] = [1]
            meta_data["views"] = [1]

            def add_dim(x, dim=1):
                return x.unsqueeze(dim) if x is not None else None

            image = add_dim(image)
            intrinsics = add_dim(intrinsics)
            extrinsics = add_dim(extrinsics)
            scale = add_dim(scale)
            prompt_depth = add_dim(prompt_depth)
            target_local_depth = add_dim(target_local_depth)
            target_global_points = add_dim(target_global_points)
            target_depth_mask = add_dim(target_depth_mask)
            target_normal = add_dim(target_normal)
            target_normal_mask = add_dim(target_normal_mask)
            target_motion_mask = add_dim(target_motion_mask)
            target_invalid_mask = add_dim(target_invalid_mask)
            image_show = add_dim(image_show)
            align_data = add_dim(align_data)
            ray_directions = add_dim(ray_directions)
            ray_world = add_dim(ray_world)

        return (
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
        )

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

    def train_step(self, batch):
        """训练单步: 前向推理 → 采样 GT → 计算 local depth loss (+ 可选 Gaussian 渲染 loss).

        流程:
            1. 解析 batch, 确定 world-to-camera 矩阵 (prompt / noisy / GT)
            2. 模型前向得到 query depth, confidence 及 query UV 坐标
            3. 用 grid_sample 在 GT 深度图上采样 query 点对应的深度值
            4. 计算 local depth loss (get_base_loss)
            5. (可选) 若模型输出 Gaussians, 进行 gsplat 渲染并计算 RGB + depth 重建 loss
            6. 记录额外的统计信息 (scale, aspect_ratio 等)

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
        )

        B, N = image.shape[:2]

        # 提取模型输出的 query 深度和置信度
        # depth: (B, N, Q, 1) → unsqueeze(2) → (B, N, 1, Q, 1)
        query_depth = results.get("depth").unsqueeze(2)
        query_conf = results.get("confidence").unsqueeze(2)

        # 构建 grid_sample 所需的采样坐标，从 [0,1] 归一化到 [-1,1]
        query = results["query"]
        if query.uv is None:
            # batch_uv 已按 batch 维度组织
            query_uv_grid = query.batch_uv
            query_uv_grid = query_uv_grid * 2 - 1
            query_uv_grid = query_uv_grid.unsqueeze(2)  # (1, Q, 1, 2)
        else:
            # 共享 uv 需要扩展到 B*N 维度
            query_uv_grid = query.uv
            query_uv_grid = query_uv_grid * 2 - 1
            query_uv_grid = query_uv_grid.unsqueeze(0).unsqueeze(2)  # (1, Q, 1, 2)
            query_uv_grid = query_uv_grid.expand(B * N, -1, -1, -1)  # (B*N, Q, 1, 2)

        # 合并 batch 和 view 维度以便 grid_sample: (B, N, C, H, W) → (B*N, C, H, W)
        target_local_depth = target_local_depth.view(B * N, *target_local_depth.shape[-3:])
        if target_depth_mask.ndim == 4:
            target_depth_mask = target_depth_mask.view(B * N, 1, *target_depth_mask.shape[-2:])
        elif target_depth_mask.ndim == 5:
            target_depth_mask = target_depth_mask.view(B * N, *target_depth_mask.shape[-3:])
        else:
            raise NotImplementedError

        # 在 GT 深度图上双线性采样 query 点对应的深度值和有效 mask
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

        total_loss, total_loss_dict = 0, dict()

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

        # FFGS rendering and loss
        if "gaussians" in results and results["gaussians"] is not None:
            gs_render_rgb, gs_render_depth = self.render_gaussians(results["gaussians"], intrinsics=intrinsics, w2c=w2c, image_shape=(query.height, query.width))
            if gs_render_rgb is not None:
                gs_render_rgb = gs_render_rgb.view(B, N, *gs_render_rgb.shape[1:])
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

    @torch.no_grad()
    def infer(self, **batch):
        """推理入口: 模型前向 → 反归一化 → scale-align → 逐视角 postprocess.

        流程:
            1. 解析输入, 确定 w2c 矩阵, 可选 AMP 推理
            2. 提取 query depth / confidence, 确定输出分辨率
            3. 反归一化 GT 深度 (用于 scale-align 对比)
            4. (可选) Gaussian 渲染
            5. 逐 frame × view 循环:
               - 反投影 depth → 3D points (depth_to_points)
               - scale-align 或 denormalize 到真实尺度
               - postprocess 封装为 ReconstructOutput
               - 附加 rgb / render_rgb / gaussians 等可视化字段

        Returns:
            mv_outputs: List[ReconstructOutput], 每个视角一个输出对象。
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
            )

        # 提取预测: depth (B, N, Q) 和 confidence (B, N, Q)
        query_depth = results.get("depth").squeeze(-1).cpu()
        query_conf = results.get("confidence").squeeze(-1).cpu()

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

        if target_local_depth is not None and scale is not None:
            target_local_depth = self.denormalize(target_local_depth, scale=scale)
        if target_depth_mask is not None and target_depth_mask.ndim == 4:
            target_depth_mask = target_depth_mask.unsqueeze(-3)

        # 输出分辨率与输入不同时，需要按比例缩放内参
        if self.testing_sub_pixel_scale != 1 and intrinsics is not None:
            x_scale = width / int(meta_data["input_width"][0])
            y_scale = height / int(meta_data["input_height"][0])
            intrinsics[..., 0, :] *= x_scale
            intrinsics[..., 1, :] *= y_scale

        # FFGS rendering
        gaussians = gs_render_rgb = gs_render_depth = None
        if "gaussians" in results and results["gaussians"] is not None:
            gaussians = results["gaussians"]
            gs_render_rgb, gs_render_depth = self.render_gaussians(gaussians, intrinsics=intrinsics, w2c=w2c, image_shape=(height, width))
            if gs_render_rgb is not None:
                gs_render_rgb = gs_render_rgb.cpu()
                gs_render_depth = gs_render_depth.cpu()

        scale = scale.cpu() if scale is not None else None
        intrinsics = intrinsics.cpu() if intrinsics is not None else None
        extrinsics = extrinsics.cpu() if extrinsics is not None else None
        target_local_depth = target_local_depth.cpu() if target_local_depth is not None else None
        target_depth_mask = target_depth_mask.cpu() if target_depth_mask is not None else None

        def get_single_view_data(data, index):
            return data[:, index] if data is not None else None

        # 逐 frame × view 处理每个视角的预测结果
        mv_outputs = []
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        for fi in range(frame_num):
            for vi in range(view_num):
                index = fi * view_num + vi

                pred_local_depth = query_depth[0, index].view(1, 1, height, width)
                pred_local_conf = query_conf[0, index].view(1, 1, height, width)

                if self.scale_align:
                    from hAlgorithm.modules.models2.external.depth_anything_3.utils.alignment import least_squares_scale_scalar

                    if pred_local_depth.shape[-2:] != target_local_depth.shape[-2:]:
                        pred_local_depth = F.interpolate(pred_local_depth, target_local_depth.shape[-2:], mode="bilinear", align_corners=False, antialias=False)
                        pred_local_conf = F.interpolate(pred_local_conf, target_local_depth.shape[-2:], mode="bilinear", align_corners=False, antialias=False)

                    pred_local_depth = self.depth_to_points(pred_local_depth, K=intrinsics[:, index], device=pred_local_depth.device, cache=False)
                    scale_factor = least_squares_scale_scalar(target_local_depth[:, index, -1][target_depth_mask[:, index, 0]], pred_local_depth[:, -1][target_depth_mask[:, index, 0]])
                    pred_local_depth *= scale_factor
                elif scale is not None:
                    pred_local_depth = self.depth_to_points(pred_local_depth, K=intrinsics[:, index], device=pred_local_depth.device, cache=False)
                    pred_local_depth = self.denormalize(pred_local_depth, scale=scale[:, index])

                single = self.postprocess(
                    pred_local_points=pred_local_depth,
                    pred_local_conf=pred_local_conf,
                    image=get_single_view_data(image, index),
                    image_show=get_single_view_data(image_show, index),
                    scale=get_single_view_data(scale, index),
                    prompt_depth=get_single_view_data(prompt_depth, index),
                    target_local_depth=get_single_view_data(target_local_depth, index),
                    target_depth_mask=get_single_view_data(target_depth_mask, index),
                    intrinsics=get_single_view_data(intrinsics, index),
                    extrinsics=get_single_view_data(extrinsics, index),
                    align_data=get_single_view_data(align_data, 0),
                )

                single.rgb = (image[0, index].permute(1, 2, 0).float().cpu().numpy() + 1) * 0.5

                if gs_render_rgb is not None:
                    single.render_rgb = gs_render_rgb[index].permute(1, 2, 0).numpy()
                if gs_render_depth is not None:
                    single.render_depth = gs_render_depth[index].numpy()

                if index == 0 and gaussians is not None:
                    single.gaussians = gaussians

                # Store frame and view indices
                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                mv_outputs.append(single)

        return mv_outputs

    def visualize(self, outputs_list, meta_data, out_dir):
        """可视化推理结果: 调用父类通用可视化后，额外处理 Gaussian PLY 导出与渲染视频.

        先临时禁用 save_gaussians 调父类 visualize (避免重复)，随后恢复标志位并
        执行 Gaussian 专属的可视化: PLY 导出 + (可选) 轨迹插值渲染视频。
        """
        # 临时禁用以避免父类重复处理 Gaussians
        save_gaussians = self.save_output_cfg["save_gaussians"]

        self.save_output_cfg["save_gaussians"] = False
        super().visualize(outputs_list, meta_data, out_dir)
        self.save_output_cfg["save_gaussians"] = save_gaussians

        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        data_idx = meta_data["data_idx"][0]

        gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir = self.get_out_dir(out_dir, data_idx=data_idx)

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
