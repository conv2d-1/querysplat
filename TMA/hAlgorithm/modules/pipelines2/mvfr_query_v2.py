import logging
import os

import torch
import torch.nn.functional as F

from hAlgorithm.modules.models2.external.depth_anything_3.utils.alignment import least_squares_scale_scalar
from hAlgorithm.modules.models2.gaussians.infinidepth_ply import export_ply
from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from hAlgorithm.modules.pipelines2.utils.render_util import render_video_interpolation
from hAlgorithm.modules.pipelines2.utils.visualize_mv import (
    save_video,
    vis_render,
)

from .mvfr_query_v1 import MVFRQueryPipeline as BaseMVFRQueryPipeline


class MVFRQueryPipeline(BaseMVFRQueryPipeline):
    """基于 Query 的多视角前馈重建 Pipeline (V2 — 支持 Local + Global + Camera + FFGS).

    相比 V1 的扩展:
        - **Local depth loss**: 同 V1, 基于 query 点的局部深度预测。
        - **Global points loss**: 新增全局坐标系下的 3D 点预测及其损失。
        - **Camera loss**: 新增基于 pose_encoding 的相机外参/内参监督。
        - **Pose decoding**: 推理时将 pose_encoding 解码为外参 + 内参, 支持
          可选的 normalize_cameras (以第 0 视角为参考系)。
        - **Subclass hooks**: _infer_extra_model_kwargs / _on_infer_model_results
    """

    def __init__(self, **kwargs):
        super(MVFRQueryPipeline, self).__init__(**kwargs)

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
        """训练单步: 前向 → 计算 Local + Global + Camera + (可选) Gaussian 渲染 四部分 loss.

        与 V1 的区别:
            - 模型可能同时输出 local depth / global points / pose_encoding / gaussians
            - 各输出按是否为 None 动态决定是否计算对应 loss
            - 新增 Part2 (Global points loss) 和 Part3 (Camera loss)

        Loss 组成:
            Part1 (lcl) — 局部深度 loss: query 点在相机坐标系下的深度/点云监督
            Part2 (glb) — 全局点云 loss: query 点在世界坐标系下的 3D 点监督
            Part3 (cm)  — 相机 loss:   pose_encoding → extrinsics/intrinsics 的监督
            Part4 (rc)  — 渲染 loss:   Gaussian splatting 渲染的 RGB + 深度重建监督

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
        """推理入口 (V2): 在 V1 基础上增加 global points、pose decoding 和子类 hook.

        与 V1 infer 的区别:
            - 额外提取 global_points / global_confidence
            - 解码 pose_encoding → pred_extrinsics / pred_intrinsics
            - 支持 output_normalize_cameras (以第 0 视角为参考坐标系)
            - 外参平移量反归一化到真实尺度
            - 子类 hook: _infer_extra_model_kwargs / _on_infer_model_results

        Returns:
            mv_outputs: List[ReconstructOutput], 含 local + global 点云 + 预测相机参数。
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

        _extra_model_kwargs = self._infer_extra_model_kwargs()
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
                **_extra_model_kwargs,
            )
        self._on_infer_model_results(results)

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

        # 输出分辨率与输入不同时，按比例缩放 GT 和预测内参
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

                # Store frame and view indices
                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                mv_outputs.append(single)

        return mv_outputs

    def visualize(self, outputs_list, meta_data, out_dir):
        """可视化推理结果 (V2): 复用 V1 逻辑，Gaussian 处理方式相同."""
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
