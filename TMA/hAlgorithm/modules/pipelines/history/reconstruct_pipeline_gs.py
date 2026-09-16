import json
import logging
import os

import torch
from einops import pack

from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    extri_intri_to_pose_encoding,
    pose_encoding_to_extri_intri,
)
from hAlgorithm.modules.pipelines.visualize import (
    save_depth_map,
    save_error,
    save_image,
    save_point_cloud,
    save_video,
)
from hAlgorithm.modules.utils.gaussians.camera_trajectory import (
    interpolate_intrinsics,
    interpolate_poses_spline,
)
from hAlgorithm.modules.utils.gaussians.novel_view_mask import calculate_loss_mask
from hAlgorithm.utils import apply_color_map, grid_images, instantiate_from_config

from .outputs import ReconstructOutput
from .prompt_pointmap_pipeline_v2 import PromptPointMapPipeline


class ReconstructPipeline(PromptPointMapPipeline):
    """
    Pipeline for reconstructing 3D scenes from input images and depth information.
    """

    def __init__(
        self,
        p_conf_loss=None,
        mv_target_name=None,
        mv_extrinsics_name=None,
        lpips_loss=None,
        rgb_l1_loss=None,
        ssim_loss=None,
        opacity_loss_weight=0.0,
        pose_loss_weight=None,
        geom_consist_loss=None,
        photo_consist_loss=None,
        save_gaussians=False,
        output_sf_pointmap=False,
        pts_glb2local=False,
        conf_pts_glb2local=False,
        enable_novel_view_loss=True,
        gs_near=0.1,
        gs_far=100.0,
        global_point_loss=True,
        mask_gaussian_ds=[],
        **kwargs,
    ):
        super(ReconstructPipeline, self).__init__(**kwargs)

        self.output_type = ReconstructOutput
        self.extrinsics_c2w = self.model.extrinsics_c2w

        self.mv_target_name = mv_target_name
        self.mv_extrinsics_name = mv_extrinsics_name
        self.rgb_l1_loss = instantiate_from_config(rgb_l1_loss)
        self.ssim_loss = instantiate_from_config(ssim_loss)
        self.lpips_loss = instantiate_from_config(lpips_loss)
        self.opacity_loss_weight = opacity_loss_weight
        self.pose_loss_weight = pose_loss_weight or 1.0
        self.geom_consist_loss = instantiate_from_config(geom_consist_loss)
        self.photo_consist_loss = instantiate_from_config(photo_consist_loss)
        self.global_point_loss = global_point_loss

        if p_conf_loss is not None:
            self.p_conf_loss = instantiate_from_config(p_conf_loss)
        else:
            self.p_conf_loss = self.conf_loss

        self.save_gaussians = save_gaussians
        self.output_sf_pointmap = output_sf_pointmap
        self.pts_glb2local = pts_glb2local
        self.conf_pts_glb2local = conf_pts_glb2local and (not self.pts_glb2local)

        self.gs_near = gs_near
        self.gs_far = gs_far
        self.enable_novel_view_loss = enable_novel_view_loss
        self.mask_gaussian_ds = mask_gaussian_ds

    def get_inputs(self, batch):
        # Extract and move tensors to the appropriate device and dtype if necessary.
        name = batch["meta_data"]["name"][0]
        total_iter = batch.get("total_iter", None)
        image = batch["image"].to(device=self.device, dtype=self.dtype)

        intrinsics = batch.get("intrinsics", None)
        extrinsics = batch.get(self.mv_extrinsics_name, None)  # [n, 4, 4]

        if intrinsics is not None:
            intrinsics = intrinsics.to(device=self.device)
        if extrinsics is not None:
            extrinsics = extrinsics.to(device=self.device)

        image_show = batch.get("image_show", None)
        if image_show is not None:
            if image_show.ndim == 4:
                image_show = image_show.float().squeeze(0).numpy().transpose(1, 2, 0)
            else:
                image_show = image_show.float().squeeze(0).numpy().transpose(0, 2, 3, 1)

        if self.prompt_name is not None:
            prompt_depth = batch[self.prompt_name].to(self.device, self.dtype)
            prompt_mask = (
                batch[self.prompt_mask_name].to(self.device) if self.prompt_mask_name else None
            )

            prompt_scale = batch[self.prompt_scale_name].to(self.device, self.dtype)[
                ..., None, None, None
            ]

            if self.prompt_center_name is not None:
                prompt_center = batch[self.prompt_center_name].to(self.device, self.dtype)[
                    ..., None, None
                ]
            else:
                # Otherwise, create a tensor of zeros with the same shape as prompt_scale.
                prompt_center = None

            # Normalize the prompt depth if it is provided, using the maximum range and center values.
            prompt_depth_norm = self.normalize(prompt_depth, prompt_scale, prompt_center)

            prompt_diffmap = (
                batch[self.prompt_diffmap_name].to(self.device)
                if self.prompt_diffmap_name
                else None
            )

        else:
            prompt_depth = prompt_depth_norm = prompt_scale = prompt_center = prompt_mask = (
                prompt_diffmap
            ) = None

        if self.target_name is not None and self.target_name in batch:
            target = batch[self.target_name].to(device=self.device)
            valid_mask = batch[self.target_mask_name].to(self.device)

            if self.debug:
                logging.info(
                    f'{valid_mask.reshape(valid_mask.shape[0], -1).sum(-1),} {batch["meta_data"]["data_info"]["rgb"]}'
                )

            if self.prompt_interpolate:
                prompt_diffmap = torch.abs(target - prompt_depth)

            target_norm = self.normalize(target, prompt_scale, prompt_center)

            # Optionally clip the target depth to a specified range.
            if self.target_clip is not None:
                target_norm = target_norm.clip(self.target_clip[0], self.target_clip[1])

        else:
            target = target_norm = valid_mask = None

        # Extract edge mask from the batch if it exists.
        edge_mask = batch.get(self.edge_mask_name, None)
        if edge_mask is not None:
            edge_mask = edge_mask.to(device=self.device)

        # After normalize
        if self.prompt_set_none:
            prompt_depth = prompt_depth_norm = prompt_scale = prompt_center = prompt_mask = (
                prompt_diffmap
            ) = None

        return (
            name,
            total_iter,
            image,
            intrinsics,
            extrinsics,
            prompt_depth,
            prompt_depth_norm,
            prompt_scale,
            prompt_center,
            prompt_mask,
            prompt_diffmap,
            target,
            target_norm,
            valid_mask,
            edge_mask,
            image_show,
        )

    def get_reconstruct_loss(
        self,
        name,
        image,
        depth,
        render_rgb,
        render_depth,
        gaussians=None,
        valid_mask=None,
        pose_enc=None,
        pose_enc_fine=None,
        intrinsics_gt=None,
        extrinsics_gt=None,
        prompt_depth=None,
        prompt_scale=None,
    ):
        """
        Computes the reconstruction loss between rendered and target RGB images.

        :param name: Name of the current sample.
        :param image: Target RGB image tensor.
        :param depth: Target depth map tensor.
        :param render_rgb: Rendered RGB image tensor.
        :param render_depth: Rendered depth map tensor.
        :return: Tuple containing total loss and a dictionary of individual losses.
        """
        rgb = (image.float() + 1) * 0.5
        if render_rgb is not None:
            render_rgb = render_rgb.float()
        if valid_mask is not None:
            valid_mask = valid_mask.float()
        if render_depth is not None:
            render_depth = render_depth.float()
        if prompt_depth is not None:
            prompt_depth = prompt_depth.float()
        if prompt_scale is not None:
            prompt_scale = prompt_scale.float()

        loss, loss_dict = 0, dict()

        if self.rgb_l1_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_l1_loss = self.rgb_l1_loss(rgb, render_rgb, mask=valid_mask, name=name)
            loss += rc_l1_loss
            loss_dict["rc_l1_loss"] = rc_l1_loss

        if self.ssim_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_ssim_loss = self.ssim_loss(rgb, render_rgb, mask=valid_mask, name=name)
            loss += rc_ssim_loss
            loss_dict["rc_ssim_loss"] = rc_ssim_loss

        if self.lpips_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_lpips_loss = self.lpips_loss(rgb, render_rgb, mask=valid_mask, name=name)
            loss += rc_lpips_loss
            loss_dict["rc_lpips_loss"] = rc_lpips_loss

        # TODO: render depth loss
        # render_depth = self.normalize(render_depth, prompt_scale.squeeze(2), None)
        # if render_depth is not None:
        #     depth_mask = depth[:, :, 2] > 0
        #     rc_depth_l1_loss = torch.abs(depth[:, :, 2][depth_mask] - render_depth[depth_mask]).mean()
        #     loss += rc_depth_l1_loss
        #     loss_dict["rc_depth_l1_loss"] = rc_depth_l1_loss

        if pose_enc is not None and pose_enc.requires_grad:
            pose_enc_gt = extri_intri_to_pose_encoding(
                extrinsics=extrinsics_gt,
                intrinsics=intrinsics_gt,
                image_size_hw=image.shape[-2:],
                pose_encoding_type="absT_quaR_FoV",
                translation_scale=prompt_scale,
            )

            B, N = pose_enc.shape[:2]
            pose_l1_loss = (pose_enc - pose_enc_gt).abs()
            pose_l1_loss = torch.nan_to_num(pose_l1_loss)
            # NOTE: [T, quat, fov_h, fov_w]
            pose_t_loss = pose_l1_loss[..., 0:3].sum() / (B * N)
            pose_quat_loss = pose_l1_loss[..., 3:7].sum() / (B * N)
            if isinstance(self.pose_loss_weight, dict):
                pose_t_loss = pose_t_loss * self.pose_loss_weight["t"]
                pose_quat_loss = pose_quat_loss * self.pose_loss_weight["quat"]
            else:
                pose_t_loss = pose_t_loss * self.pose_loss_weight
                pose_quat_loss = pose_quat_loss * self.pose_loss_weight

            loss += pose_t_loss + pose_quat_loss
            loss_dict["pose_t_loss"] = pose_t_loss
            loss_dict["pose_quat_loss"] = pose_quat_loss

        if pose_enc_fine is not None:
            B, N = pose_enc_fine.shape[:2]
            pose_l1_loss = (pose_enc_fine - pose_enc_gt).abs()
            pose_l1_loss = torch.nan_to_num(pose_l1_loss)
            # NOTE: [T, quat, fov_h, fov_w]
            pose_t_loss = pose_l1_loss[..., 0:3].sum() / (B * N)
            pose_quat_loss = pose_l1_loss[..., 3:7].sum() / (B * N)
            if isinstance(self.pose_loss_weight, dict):
                pose_t_loss = pose_t_loss * self.pose_loss_weight["t"]
                pose_quat_loss = pose_quat_loss * self.pose_loss_weight["quat"]
            else:
                pose_t_loss = pose_t_loss * self.pose_loss_weight
                pose_quat_loss = pose_quat_loss * self.pose_loss_weight

            loss += pose_t_loss + pose_quat_loss
            loss_dict["pose_fine_t_loss"] = pose_t_loss
            loss_dict["pose_fine_quat_loss"] = pose_quat_loss

        if gaussians is not None and self.opacity_loss_weight > 0:
            opaticity_loss = 0
            for i in range(len(gaussians)):
                opaticity_loss += ((1 - gaussians[i].get_opacity()) ** 2).mean()
            opaticity_loss *= self.opacity_loss_weight
            loss += opaticity_loss
            loss_dict["opaticity_loss"] = opaticity_loss

        return loss, loss_dict

    def get_novel_view_loss(
        self,
        name,
        image,
        depth,
        render_rgb,
        render_depth,
        valid_mask=None,
    ):
        # TODO: use mask
        """
        Computes the reconstruction loss between rendered and target RGB images.

        :param name: Name of the current sample.
        :param image: Target RGB image tensor.
        :param depth: Target depth map tensor.
        :param render_rgb: Rendered RGB image tensor.
        :param render_depth: Rendered depth map tensor.
        :return: Tuple containing total loss and a dictionary of individual losses.
        """
        rgb = (image.float() + 1) * 0.5
        if render_rgb is not None:
            render_rgb = render_rgb.float()
        if render_depth is not None:
            render_depth = render_depth.float()
        if valid_mask is not None:
            valid_mask = valid_mask.float()

        loss, loss_dict = 0, dict()

        if self.rgb_l1_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_l1_loss = self.rgb_l1_loss(rgb, render_rgb, mask=valid_mask, name=name)
            loss += rc_l1_loss
            loss_dict["novel_l1_loss"] = rc_l1_loss

        if self.ssim_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_ssim_loss = self.ssim_loss(rgb, render_rgb, mask=valid_mask, name=name)
            loss += rc_ssim_loss
            loss_dict["novel_ssim_loss"] = rc_ssim_loss

        if self.lpips_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_lpips_loss = self.lpips_loss(rgb, render_rgb, mask=valid_mask, name=name)
            loss += rc_lpips_loss
            loss_dict["novel_lpips_loss"] = rc_lpips_loss

        return loss, loss_dict

    def depth_to_point(self, depth, K, device=None, cache=True):
        if depth.ndim == 4:
            return super().depth_to_point(depth, K, device=device, cache=cache)

        device = device or self.device
        B, N, C, H, W = depth.shape
        cache_key = f"depth_to_point_{H}_{W}"

        if cache_key in self.cache_dict:
            points = self.cache_dict[cache_key]
            if points.device != device:
                points.to(device)
        else:
            grid_x, grid_y = torch.meshgrid(
                torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy"
            )
            points = (
                torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
                .reshape(3, -1)
                .float()
                .to(device)
            )
            if cache:
                self.cache_dict[cache_key] = points
        rays_d = K.inverse() @ points  # (B, 3, HW)
        pts = depth.flatten(3) * rays_d
        depth = pts.reshape(B, N, 3, H, W)
        return depth

    def get_p_extend_loss(
        self,
        name,
        pointmap_pred,
        confidence_pred,
        intrinsics,
        extrinsics,
        gradient_pred,
        invalid_mask_pred,
        prompt_confidence_pred,
        depth_target,
        valid_mask,
        invalid_mask_target,
        prompt_diffmap,
        prompt_scale,
        prompt_center,
    ):
        if self.conf_loss is not None and confidence_pred is not None:
            if not self.pts_glb2local and self.conf_pts_glb2local:
                local_depth_target = self.glb_to_local(
                    depth_target.unsqueeze(1),
                    extrinsics=extrinsics.unsqueeze(1),
                    scale=prompt_scale.unsqueeze(1) if prompt_scale is not None else None,
                    center=prompt_center.unsqueeze(1) if prompt_center is not None else None,
                ).squeeze(1)
                local_pointmap_pred = self.glb_to_local(
                    pointmap_pred.unsqueeze(1),
                    extrinsics=extrinsics.unsqueeze(1),
                    scale=prompt_scale.unsqueeze(1) if prompt_scale is not None else None,
                    center=prompt_center.unsqueeze(1) if prompt_center is not None else None,
                ).squeeze(1)
                local_depth_target = local_depth_target.float().permute(0, 2, 3, 1)
                local_pointmap_pred = local_pointmap_pred.float().permute(0, 2, 3, 1)

        # Convert the prediction and target tensors to float and permute dimensions for loss calculation.
        pointmap_pred = pointmap_pred.float().permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C
        depth_target = depth_target.float().permute(0, 2, 3, 1)
        valid_mask = valid_mask.squeeze(1)

        if confidence_pred is not None:
            confidence_pred = confidence_pred.squeeze(1).unsqueeze(3)  # B,1,H,W -> B,H,W,1

        if prompt_diffmap is not None:
            prompt_diffmap = prompt_diffmap.float().permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C

        if gradient_pred is not None:
            gradient_pred = gradient_pred.float().permute(0, 2, 3, 1)

        loss = 0
        loss_dict = {}

        def update_loss(loss, cur_loss, name):
            if cur_loss is None:
                return loss
            if isinstance(cur_loss, dict):
                loss += sum([val for key, val in cur_loss.items()])
                loss_dict.update(cur_loss)
            else:
                loss += cur_loss
                loss_dict[name] = cur_loss
            return loss

        if self.conf_loss is not None and confidence_pred is not None:
            if not self.pts_glb2local and self.conf_pts_glb2local:
                conf_loss = self.p_conf_loss(
                    confidence_pred,
                    local_pointmap_pred,
                    local_depth_target,
                    prompt_scale,
                    prompt_center,
                    valid_mask,
                    intrinsics=intrinsics,
                    name=name,
                )
            else:
                conf_loss = self.p_conf_loss(
                    confidence_pred,
                    pointmap_pred,
                    depth_target,
                    prompt_scale,
                    prompt_center,
                    valid_mask,
                    intrinsics=intrinsics,
                    name=name,
                )
            loss = update_loss(loss, conf_loss, "conf_loss")

        if self.grad_pred_loss is not None and gradient_pred is not None:
            grad_pred_loss = self.grad_pred_loss(
                gradient_pred, depth_target[..., -1], valid_mask, name=name
            )
            loss = update_loss(loss, grad_pred_loss, "grad_pred_loss")

        if self.mask_loss is not None and invalid_mask_pred is not None:
            mask_loss = self.mask_loss(
                invalid_mask_pred.unsqueeze(1),
                invalid_mask_target.unsqueeze(1),
                valid_mask,
                name=name,
            )
            loss = update_loss(loss, mask_loss, "mask_loss")

        if self.prompt_conf_loss is not None and prompt_confidence_pred is not None:
            prompt_conf_loss = self.prompt_conf_loss(
                prompt_confidence_pred,
                prompt_diffmap=prompt_diffmap,
                valid_mask=valid_mask,
                name=name,
            )
            loss = update_loss(loss, prompt_conf_loss, "prompt_conf_loss")

        return loss, loss_dict

    def get_depth_loss(
        self,
        name,
        total_iter,
        image,
        intrinsics,
        prompt_scale,
        prompt_center,
        prompt_mask,
        prompt_diffmap,
        target_norm,
        valid_mask,
        edge_mask,
        pointmap_pred,
        confidence_pred,
        gradient_pred=None,
        prompt_confidence_pred=None,
    ):
        b, n, c, h, w = pointmap_pred.shape

        total_loss = 0
        total_loss_dict = dict()

        for i in range(n):
            loss, loss_dict = self.get_loss(
                name=name,
                total_iter=total_iter,
                pointmap_pred=pointmap_pred[:, i],
                confidence_pred=confidence_pred[:, i],
                target=target_norm[:, i],
                intrinsics=intrinsics[:, i] if intrinsics is not None else None,
                valid_mask=valid_mask[:, i],
                edge_mask=edge_mask[:, i] if edge_mask is not None else None,
                prompt_mask=prompt_mask[:, i] if prompt_mask is not None else None,
                image=image[:, i],
            )

            if self.use_extend_loss:
                loss_extend, loss_dict_extend = self.get_extend_loss(
                    name=name,
                    pointmap_pred=pointmap_pred[:, i],
                    invalid_mask_pred=None,
                    confidence_pred=confidence_pred[:, i],
                    intrinsics=intrinsics[:, i] if intrinsics is not None else None,
                    gradient_pred=gradient_pred[:, i] if gradient_pred is not None else None,
                    prompt_confidence_pred=(
                        prompt_confidence_pred[:, i] if prompt_confidence_pred is not None else None
                    ),
                    invalid_mask_target=None,
                    valid_mask=valid_mask[:, i],
                    depth_target=target_norm[:, i],
                    prompt_diffmap=prompt_diffmap[:, i] if prompt_diffmap is not None else None,
                    prompt_scale=prompt_scale[:, i] if prompt_scale is not None else None,
                    prompt_center=prompt_center[:, i] if prompt_center is not None else None,
                )
                loss = loss + loss_extend
                loss_dict.update(loss_dict_extend)

            total_loss += loss / n
            for key, val in loss_dict.items():
                if i == 0:
                    total_loss_dict[key] = val / n
                else:
                    total_loss_dict[key] += val / n

        return total_loss, total_loss_dict

    def get_points_loss(
        self,
        name,
        total_iter,
        image,
        intrinsics,
        extrinsics,
        prompt_scale,
        prompt_center,
        prompt_mask,
        prompt_diffmap,
        target_norm,
        valid_mask,
        edge_mask,
        pointmap_pred,
        confidence_pred,
        gradient_pred=None,
        prompt_confidence_pred=None,
    ):
        b, n, c, h, w = pointmap_pred.shape

        total_loss = 0
        total_loss_dict = dict()

        for i in range(n):
            loss, loss_dict = self.get_loss(
                name=name,
                total_iter=total_iter,
                pointmap_pred=pointmap_pred[:, i],
                confidence_pred=confidence_pred[:, i],
                target=target_norm[:, i],
                intrinsics=intrinsics[:, i] if intrinsics is not None else None,
                valid_mask=valid_mask[:, i],
                edge_mask=edge_mask[:, i] if edge_mask is not None else None,
                prompt_mask=prompt_mask[:, i] if prompt_mask is not None else None,
                image=image[:, i],
            )

            if self.use_extend_loss:
                loss_extend, loss_dict_extend = self.get_p_extend_loss(
                    name=name,
                    pointmap_pred=pointmap_pred[:, i],
                    invalid_mask_pred=None,
                    confidence_pred=confidence_pred[:, i],
                    intrinsics=intrinsics[:, i] if intrinsics is not None else None,
                    extrinsics=extrinsics[:, i] if extrinsics is not None else None,
                    gradient_pred=gradient_pred[:, i] if gradient_pred is not None else None,
                    prompt_confidence_pred=(
                        prompt_confidence_pred[:, i] if prompt_confidence_pred is not None else None
                    ),
                    invalid_mask_target=None,
                    valid_mask=valid_mask[:, i],
                    depth_target=target_norm[:, i],
                    prompt_diffmap=prompt_diffmap[:, i] if prompt_diffmap is not None else None,
                    prompt_scale=prompt_scale[:, i] if prompt_scale is not None else None,
                    prompt_center=prompt_center[:, i] if prompt_center is not None else None,
                )
                loss = loss + loss_extend
                loss_dict.update(loss_dict_extend)

            total_loss += loss / n
            for key, val in loss_dict.items():
                if i == 0:
                    total_loss_dict[key] = val / n
                else:
                    total_loss_dict[key] += val / n

        return total_loss, total_loss_dict

    def glb_to_local(self, glb, extrinsics, scale=None, center=None):
        """
        glb: [b, v, 3, h, w], torch.Tensor
        extrinsics: [b, v, 4, 4], torch.Tensor
        scale: [b, v, 1, 1, 1], torch.Tensor
        """
        glb = self.denormalize(glb, scale=scale, center=center)
        b, v, _, h, w = glb.shape
        # Convert glb to homogeneous coordinates: [b, v, 4, h, w]
        extrinsics = extrinsics.clone()
        # if scale is not None:
        #     extrinsics[..., :3, 3] /= scale.reshape(*scale.shape[:3])
        ones = torch.ones((b, v, 1, h, w), device=glb.device, dtype=glb.dtype)
        glb_homo = torch.cat([glb, ones], dim=2)  # [b, v, 4, h, w]
        # Reshape for matrix multiplication
        glb_homo = glb_homo.view(b, v, 4, -1)  # [b, v, 4, h*w]
        # Apply extrinsics transformation
        local_homo = torch.matmul(extrinsics, glb_homo)  # [b, v, 4, h*w]
        # Reshape back to original spatial shape
        local_homo = local_homo.view(b, v, 4, h, w)
        # Normalize by scale
        local = local_homo[:, :, :3, :, :]  # Keep only the first 3 channels
        local = self.normalize(local, scale=scale, center=center)
        return local

    def split_batch(self, batch):
        if "extra_ids" not in batch["meta_data"]:
            return batch, None

        extra_ids = batch["meta_data"]["extra_ids"]

        if extra_ids.shape[1] == 0:
            return batch, None

        batch_novel = {}
        b, num_views = batch["image"].shape[:2]

        batch_indices = torch.arange(b)[:, None]
        remaining_mask = torch.ones((b, num_views), dtype=torch.bool)
        remaining_mask[batch_indices, extra_ids] = False

        for k, v in batch.items():
            if type(v) != torch.Tensor:
                # batch_novel[k] = v
                continue
            if len(v.shape) < 2 or v.shape[1] != num_views:
                # batch_novel[k] = v
                continue
            remaining = v[remaining_mask].reshape(b, -1, *v.shape[2:])
            batch[k] = remaining
            # if k in ["intrinsics", "extrinsics", "mv_extrinsics"]:   # novel views only need these data
            selected = v[batch_indices, extra_ids]
            batch_novel[k] = selected

        if batch["meta_data"]["frames"] > 1:
            batch["meta_data"]["frames"] -= extra_ids.shape[1]

        if batch["meta_data"]["views"] > 1:
            batch["meta_data"]["views"] -= extra_ids.shape[1]

        return batch, batch_novel

    def render_novel_batch(self, gaussians, batch, near, far):
        # render novel views
        b, n, _, h, w = batch["image"].shape
        novel_intrinsics = batch["intrinsics"].to(device=self.device)
        novel_extrinsics = batch[self.mv_extrinsics_name].to(device=self.device)

        novel_intrinsics[..., 0, :] /= h
        novel_intrinsics[..., 1, :] /= w

        novel_results = self.model(
            rendering=True,
            gaussians=gaussians,
            extrinsics=novel_extrinsics,
            intrinsics=novel_intrinsics,
            near=near,
            far=far,
            hw=(h, w),
            depth_mode="depth",
        )
        results = {}
        results["render_novel_rgb"] = novel_results["render_rgb"]
        results["render_novel_depth"] = novel_results["render_depth"]
        return results

    def train_step(self, batch):
        """
        Executes a single training step using a batch of data.

        Parameters:
        - batch (dict): A dictionary containing the batch of data, including images, depth information, and masks.

        Returns:
        - loss (Tensor): The total loss for this training step.
        - loss_dict (dict): A dictionary containing individual loss components.
        """
        # Extract metadata from the batch
        meta_data = batch["meta_data"]

        # If the batch does not contain 'frames' or 'views', defer to the superclass implementation
        if "frames" not in meta_data and "views" not in meta_data:
            return super().train_step(batch)

        batch, batch_novel = self.split_batch(batch)

        self.train()
        # Unpack necessary inputs from the batch
        (
            name,
            total_iter,
            image,
            intrinsics,
            _,
            prompt_depth,
            prompt_depth_norm,
            prompt_scale,
            prompt_center,
            prompt_mask,
            prompt_diffmap,
            target,
            target_norm,
            valid_mask,
            edge_mask,
            image_show,
        ) = self.get_inputs(batch)

        if self.mv_target_name is not None:
            mv_target_norm = self.normalize(
                batch[self.mv_target_name].to(device=self.device),
                scale=prompt_scale,
                center=prompt_center,
            )

        if self.mv_extrinsics_name is not None:
            extrinsics = batch[self.mv_extrinsics_name].to(device=self.device)
        else:
            extrinsics = None

        novel_extrinsics = None
        novel_intrinsics = None
        if batch_novel is not None:
            novel_extrinsics = batch_novel[self.mv_extrinsics_name].to(device=self.device).clone()
            novel_intrinsics = batch_novel["intrinsics"].to(device=self.device).clone()

        # Forward pass through the model
        results = self.model(
            image,
            meta_data=meta_data,
            prompt_depth=prompt_depth_norm,
            prompt_scale=prompt_scale,
            prompt_center=prompt_center,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            with_freeze=True,
            novel_intrinsics=novel_intrinsics,
            novel_extrinsics=novel_extrinsics,
            near=self.gs_near,
            far=self.gs_far,
            gs_mask=valid_mask if name in self.mask_gaussian_ds else None,
        )

        # Extract predictions from the model output
        pointmap_pred = results.get("pointmap", None)
        confidence_pred = results.get("confidence", None)
        mv_pointmap_pred = results.get("mv_pointmap", None)
        mv_confidence_pred = results.get("mv_confidence", None)
        mv_pointmap_stage2_pred = results.get("mv_pointmap_stage2", None)
        mv_confidence_stage2_pred = results.get("mv_confidence_stage2", None)
        mv_depth_pred = results.get("mv_depth", None)
        mv_depth_confidence_pred = results.get("mv_depth_confidence", None)
        render_rgb = results.get("render_rgb", None)
        render_depth = results.get("render_depth", None)
        render_novel_rgb = results.get("render_novel_rgb", None)
        render_novel_depth = results.get("render_novel_depth", None)
        pose_enc = results.get("pose_enc", None)
        pose_enc_fine = results.get("pose_enc_fine", None)
        gaussians = results.get("gaussians", None)

        # assert mv_depth_pred is None or mv_depth_pred.shape[-3] == 1
        assert mv_pointmap_pred is None or mv_pointmap_pred.shape[-3] == 3
        assert mv_pointmap_stage2_pred is None or mv_pointmap_stage2_pred.shape[-3] == 3

        total_loss, total_loss_dict = 0, dict()

        # Compute depth loss if pointmap or confidence predictions are available
        if (pointmap_pred is not None and pointmap_pred.requires_grad) or (
            confidence_pred is not None and confidence_pred.requires_grad
        ):
            # DepthMap trans to PointMap
            if pointmap_pred.shape[-3] == 1:
                pointmap_pred = self.depth_to_point(pointmap_pred, K=intrinsics)

            depth_loss, depth_loss_dict = self.get_depth_loss(
                name=name,
                total_iter=total_iter,
                image=image,
                intrinsics=intrinsics,
                prompt_scale=prompt_scale,
                prompt_center=prompt_center,
                prompt_mask=prompt_mask,
                prompt_diffmap=prompt_diffmap,
                target_norm=target_norm,
                valid_mask=valid_mask,
                edge_mask=edge_mask,
                pointmap_pred=pointmap_pred,
                confidence_pred=confidence_pred,
            )

            total_loss += depth_loss
            total_loss_dict = depth_loss_dict
            total_loss_dict["sfd"] = sum(depth_loss_dict.values())

        # Compute multi-view points loss if multi-view pointmap or confidence predictions are available
        if self.global_point_loss and (
            (mv_pointmap_pred is not None and mv_pointmap_pred.requires_grad)
            or (mv_confidence_pred is not None and mv_confidence_pred.requires_grad)
        ):
            if self.pts_glb2local:
                mv_pointmap_pred = self.glb_to_local(
                    mv_pointmap_pred,
                    extrinsics=extrinsics,
                    scale=prompt_scale,
                    center=prompt_center,
                )
                mvp_loss, mvp_loss_dict = self.get_points_loss(
                    name=name,
                    total_iter=total_iter,
                    image=image,
                    intrinsics=None,
                    extrinsics=extrinsics,
                    prompt_scale=prompt_scale,
                    prompt_center=prompt_center,
                    prompt_mask=prompt_mask,
                    prompt_diffmap=None,
                    target_norm=target_norm,  # NOTE
                    valid_mask=valid_mask,
                    edge_mask=None,
                    pointmap_pred=mv_pointmap_pred,
                    confidence_pred=mv_confidence_pred,
                )
            else:
                mvp_loss, mvp_loss_dict = self.get_points_loss(
                    name=name,
                    total_iter=total_iter,
                    image=image,
                    intrinsics=None,
                    extrinsics=extrinsics,
                    prompt_scale=prompt_scale,
                    prompt_center=prompt_center,
                    prompt_mask=prompt_mask,
                    prompt_diffmap=None,
                    target_norm=mv_target_norm,
                    valid_mask=valid_mask,
                    edge_mask=None,
                    pointmap_pred=mv_pointmap_pred,
                    confidence_pred=mv_confidence_pred,
                )

            total_loss += mvp_loss
            total_loss_dict["mvp"] = 0
            for key, val in mvp_loss_dict.items():
                total_loss_dict[f"mvp_{key}"] = val
                total_loss_dict["mvp"] += val

        # Compute multi-view points loss if multi-view pointmap or confidence predictions are available
        if self.global_point_loss and (
            (mv_pointmap_stage2_pred is not None and mv_pointmap_stage2_pred.requires_grad)
            or (mv_confidence_stage2_pred is not None and mv_confidence_stage2_pred.requires_grad)
        ):
            if self.pts_glb2local:
                mv_pointmap_stage2_pred = self.glb_to_local(
                    mv_pointmap_stage2_pred,
                    extrinsics=extrinsics,
                    scale=prompt_scale,
                    center=prompt_center,
                )
                mvp_loss, mvp_loss_dict = self.get_points_loss(
                    name=name,
                    total_iter=total_iter,
                    image=image,
                    intrinsics=None,
                    extrinsics=extrinsics,
                    prompt_scale=prompt_scale,
                    prompt_center=prompt_center,
                    prompt_mask=prompt_mask,
                    prompt_diffmap=None,
                    target_norm=target_norm,  # NOTE
                    valid_mask=valid_mask,
                    edge_mask=None,
                    pointmap_pred=mv_pointmap_stage2_pred,
                    confidence_pred=mv_confidence_stage2_pred,
                )
            else:
                mvp_loss, mvp_loss_dict = self.get_points_loss(
                    name=name,
                    total_iter=total_iter,
                    image=image,
                    intrinsics=None,
                    extrinsics=extrinsics,
                    prompt_scale=prompt_scale,
                    prompt_center=prompt_center,
                    prompt_mask=prompt_mask,
                    prompt_diffmap=None,
                    target_norm=mv_target_norm,
                    valid_mask=valid_mask,
                    edge_mask=None,
                    pointmap_pred=mv_pointmap_stage2_pred,
                    confidence_pred=mv_confidence_stage2_pred,
                )

            total_loss += mvp_loss
            total_loss_dict["mvp2"] = 0
            for key, val in mvp_loss_dict.items():
                total_loss_dict[f"mvp2_{key}"] = val
                total_loss_dict["mvp2"] += val

        # Compute multi-view depth loss if multi-view depthmap or confidence predictions are available
        if (mv_depth_pred is not None and mv_depth_pred.requires_grad) or (
            mv_depth_confidence_pred is not None and mv_depth_confidence_pred.requires_grad
        ):
            # DepthMap trans to PointMap
            if mv_depth_pred.shape[-3] == 1:
                mv_depth_pred = self.depth_to_point(mv_depth_pred, K=intrinsics)

            mvd_loss, mvd_loss_dict = self.get_depth_loss(
                name=name,
                total_iter=total_iter,
                image=image,
                intrinsics=intrinsics,
                prompt_scale=prompt_scale,
                prompt_center=prompt_center,
                prompt_mask=prompt_mask,
                prompt_diffmap=prompt_diffmap,
                target_norm=target_norm,
                valid_mask=valid_mask,
                edge_mask=edge_mask,
                pointmap_pred=mv_depth_pred,
                confidence_pred=mv_depth_confidence_pred,
            )

            total_loss += mvd_loss
            total_loss_dict["mvd"] = 0
            for key, val in mvd_loss_dict.items():
                total_loss_dict[f"mvd_{key}"] = val
                total_loss_dict["mvd"] += val

        # Compute reconstruction loss
        loss_rc, loss_dict_rc = self.get_reconstruct_loss(
            name=name,
            image=image,
            depth=target,
            render_rgb=render_rgb,
            render_depth=render_depth,
            gaussians=gaussians,
            pose_enc=pose_enc,
            intrinsics_gt=intrinsics,
            extrinsics_gt=extrinsics,
            prompt_depth=prompt_depth_norm,
            prompt_scale=prompt_scale,
        )
        total_loss += loss_rc
        total_loss_dict.update(loss_dict_rc)

        if self.enable_novel_view_loss and render_novel_rgb is not None:
            novel_mask = calculate_loss_mask(batch, batch_novel)
            novel_mask = novel_mask.unsqueeze(2)  # [b,v,1,h,w]
            loss_novel_rc, loss_dict_novel_rc = self.get_novel_view_loss(
                name=name,
                image=batch_novel["image"].to(image.device),
                depth=batch_novel["depth"].to(image.device),
                render_rgb=render_novel_rgb,
                render_depth=render_novel_depth,
                valid_mask=novel_mask,
            )
            total_loss += loss_novel_rc
            total_loss_dict.update(loss_dict_novel_rc)

        return total_loss, total_loss_dict

    @torch.no_grad()
    def infer(self, **batch):
        """
        Executes inference on a given input image and returns the predicted depth map and related outputs.

        Parameters:
        - image (Tensor): The input image tensor.
        - **kwargs: Additional keyword arguments containing depth information and other optional parameters.

        Returns:
        - ReconstructOutput: An object containing the predicted depth map and other related outputs.
        """
        # Extract metadata from the batch input
        meta_data = batch["meta_data"]

        # meta_data['data_info']['view_id']
        # meta_data['data_info']['frame_id']

        # If "frames" or "views" are not in metadata, fallback to the superclass infer method
        if "frames" not in meta_data and "views" not in meta_data:
            return super().infer(**batch)
        batch, batch_novel = self.split_batch(batch)

        # Set the model to evaluation mode.
        self.eval()

        (
            name,
            total_iter,
            image,
            intrinsics,
            extrinsics,
            prompt_depth,
            prompt_depth_norm,
            prompt_scale,
            prompt_center,
            prompt_mask,
            prompt_diffmap,
            target,
            target_norm,
            valid_mask,
            edge_mask,
            image_show,
        ) = self.get_inputs(batch)

        # Perform inference using the shared step method of the model
        novel_intrinsics = None
        novel_extrinsics = None
        if batch_novel is not None:
            novel_intrinsics = batch_novel["intrinsics"].to(device=self.device).clone()
            novel_extrinsics = batch_novel[self.mv_extrinsics_name].to(device=self.device).clone()
            novel_mask = calculate_loss_mask(batch, batch_novel)
            novel_mask = novel_mask.unsqueeze(2).expand_as(batch_novel["image"])

        results = self.model(
            image,
            meta_data=meta_data,
            prompt_depth=prompt_depth_norm,
            prompt_scale=prompt_scale,
            prompt_center=prompt_center,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            novel_extrinsics=novel_extrinsics,
            novel_intrinsics=novel_intrinsics,
            near=meta_data.get("near", self.gs_near),
            far=meta_data.get("far", self.gs_far),
            gs_mask=valid_mask if name in self.mask_gaussian_ds else None,
            # gs_mask=valid_mask,
            target=target,
        )

        # Extract predictions from the model output
        pointmap_pred = results.get("pointmap", None)  # Predicted point cloud map
        confidence_pred = results.get("confidence", None)  # Confidence map for predictions

        mv_depth_pred = results.get("mv_depth", None)  # Multi-view depth prediction
        mv_depth_confidence_pred = results.get(
            "mv_depth_confidence", None
        )  # Confidence for multi-view depth

        if "mv_pointmap_stage2" in results:
            mv_pointmap_pred = results["mv_pointmap_stage2"]  # Multi-view point cloud map
            mv_confidence_pred = results[
                "mv_confidence_stage2"
            ]  # Confidence for multi-view point cloud map
        else:
            mv_pointmap_pred = results.get("mv_pointmap", None)  # Multi-view point cloud map
            mv_confidence_pred = results.get(
                "mv_confidence", None
            )  # Confidence for multi-view point cloud map

        # Pose encoding for camera extrinsics/intrinsics
        if "pose_enc_fine" in results:
            pose_enc = results["pose_enc_fine"]
        else:
            pose_enc = results.get(
                "pose_enc", None
            )  # Pose encoding for camera extrinsics/intrinsics

        gaussians = results.get("gaussians", None)  # Gaussian representation of the scene
        render_rgb = results.get("render_rgb", None)  # Rendered RGB image
        render_depth = results.get("render_depth", None)  # Rendered depth map

        video = None
        if gaussians is not None:
            # render video
            video = self.render_video_interpolation(
                gaussians, extrinsics, intrinsics, image.shape[-2], image.shape[-1]
            )

        novel_render_rgb = results.get("render_novel_rgb", None)
        novel_render_depth = results.get("render_novel_depth", None)

        # Convert depth maps to point clouds if necessary
        if pointmap_pred is not None and pointmap_pred.shape[-3] == 1:
            pointmap_pred = self.depth_to_point(pointmap_pred, K=intrinsics)

        if mv_depth_pred is not None and mv_depth_pred.shape[-3] == 1:
            mv_depth_pred = self.depth_to_point(mv_depth_pred, K=intrinsics)

        # Denormalize point cloud predictions if available
        if not self.output_sf_pointmap and pointmap_pred is not None:
            pointmap_pred = self.denormalize(
                pointmap_pred, scale=prompt_scale, center=prompt_center
            )

        if mv_pointmap_pred is not None:
            assert mv_pointmap_pred.shape[-3] == 3
            mv_pointmap_pred = self.denormalize(
                mv_pointmap_pred, scale=prompt_scale, center=prompt_center
            )

        # Determine the final pointmap and confidence predictions based on configuration
        if self.output_sf_pointmap:
            output_pointmap_pred = pointmap_pred
            output_confidence_pred = confidence_pred
        else:
            output_pointmap_pred = mv_depth_pred
            output_confidence_pred = mv_depth_confidence_pred

        if pose_enc is not None:
            extrinsics_pred, intrinsics_pred = pose_encoding_to_extri_intri(
                pose_encoding=pose_enc,
                image_size_hw=image.shape[-2:],
                translation_scale=prompt_scale,
            )
        else:
            extrinsics_pred, intrinsics_pred = None, None

        # Prepare alignment ground truth and mask if required
        align_gt = align_mask = None
        if self.match_input_res:
            align_gt = batch[self.align_name]
        if self.match_input_res and self.post_align:
            align_mask = batch[self.align_mask_name]

        # Extract frame and view counts from metadata
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        # Process each frame and view combination
        outputs_list = []
        for fi in range(frame_num):
            for vi in range(view_num):
                # Compute the global index for the current frame-view pair
                index = fi * view_num + vi

                # Post-process the predictions for the current frame-view pair
                single = self.postprocess(
                    intrinsics=intrinsics[:, index] if intrinsics is not None else None,
                    extrinsics=extrinsics[:, index] if extrinsics is not None else None,
                    image=image[:, index] if image is not None else None,
                    target=target[:, index] if target is not None else None,
                    prompt_depth=prompt_depth[:, index] if prompt_depth is not None else None,
                    prompt_scale=prompt_scale[:, index] if prompt_scale is not None else None,
                    prompt_center=(prompt_center[:, index] if prompt_center is not None else None),
                    image_show=image_show[index] if image_show is not None else None,
                    pointmap_pred=output_pointmap_pred[:, index],
                    confidence_pred=output_confidence_pred[:, index],
                    gradient_pred=None,
                    prompt_confidence_pred=None,
                    align_gt=align_gt[:, index] if align_gt is not None else None,
                    align_mask=align_mask[:, index] if align_mask is not None else None,
                )

                # Add multi-view point cloud predictions if available
                if mv_pointmap_pred is not None:
                    single.glb_mv_pointmap = (
                        mv_pointmap_pred[0, index].detach().cpu().numpy().transpose(1, 2, 0)
                    )
                    if extrinsics is not None:
                        glb2local_pts = self.glb_to_local(
                            mv_pointmap_pred[0:1, index : index + 1].clone(),
                            extrinsics=extrinsics[0:1, index : index + 1],
                            scale=None,
                            center=None,
                        )[0, 0]
                        single.glb2local_pointmap = (
                            glb2local_pts.detach().cpu().numpy().transpose(1, 2, 0)
                        )

                # Add multi-view confidence predictions if available
                if mv_confidence_pred is not None:
                    single.glb_mv_confidence = (
                        mv_confidence_pred[0, index, 0].detach().cpu().numpy()
                    )

                # Add single-frame point cloud predictions if configured
                if not self.output_sf_pointmap and pointmap_pred is not None:
                    single.sf_pointmap = (
                        pointmap_pred[0, index].detach().cpu().numpy().transpose(1, 2, 0)
                    )

                # Add single-frame confidence predictions if configured
                if not self.output_sf_pointmap and confidence_pred is not None:
                    single.sf_confidence = confidence_pred[0, index, 0].detach().cpu().numpy()

                # Add predicted extrinsics if available
                if extrinsics_pred is not None:
                    single.extrinsics_pred = extrinsics_pred[0, index].detach().cpu().numpy()

                # Add predicted intrinsics if available
                if intrinsics_pred is not None:
                    single.intrinsics_pred = intrinsics_pred[0, index].detach().cpu().numpy()

                # Add Gaussian representation if available
                # Gaussian is shared across views, and maybe across frames in the future
                single.novel_render_depth = None
                single.novel_render_rgb = None
                single.video = None
                if gaussians is not None and vi == 0 and fi == 0:
                    single.gaussians = gaussians[0].export_ply()
                    single.gs_xyz = gaussians[0].get_xyz().detach().cpu().numpy()
                    if novel_render_depth is not None:
                        single.novel_render_depth = (
                            novel_render_depth[0].detach().cpu().numpy()
                        )  # [N, H, W]
                    if novel_render_rgb is not None:
                        single.novel_render_rgb = (
                            novel_render_rgb[0].detach().cpu().numpy().transpose(0, 2, 3, 1)
                        )  # [N, H, W, 3]
                        single.novel_mask = (
                            novel_mask[0].detach().cpu().numpy()
                        )  # save this mask for evaluation
                    if video is not None:
                        single.video = video

                # Add rendered RGB image if available
                if render_rgb is not None:
                    single.render_rgb = (
                        render_rgb[0, index].detach().cpu().numpy().transpose(1, 2, 0)
                    )

                # Add rendered depth map if available
                if render_depth is not None:
                    single.render_depth = render_depth[0, index].detach().cpu().numpy()

                # Store frame and view indices
                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                outputs_list.append(single)
        batch["novel"] = batch_novel
        batch["type"] = "gs"
        outputs_list.append(batch)
        return outputs_list

    @torch.no_grad()
    def render_video_generic(
        self,
        gaussians_prob,
        trajectory_fn,
        h,
        w,
        n_interp: int = 30,
        loop_reverse: bool = True,
    ) -> None:

        extrinsics, intrinsics = trajectory_fn(n_interp)

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01)
            far = result.view(-1)[:16_000_000].quantile(0.99)
            result = 1 - (result - near) / (far - near)
            # colorize_depth_maps
            return apply_color_map(result, "turbo")

        output_prob = self.model(
            rendering=True,
            gaussians=gaussians_prob,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            near=self.gs_near,
            far=self.gs_far,
            hw=(h, w),
            depth_mode="depth",
        )
        depth_color = depth_map(output_prob["render_depth"][0].detach())
        rgb = output_prob["render_rgb"][0]

        video = torch.cat([rgb, depth_color], dim=2).permute(0, 2, 3, 1)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* h w c")[0]

        return video

    def render_video_interpolation(
        self, gaussians, extrinsics, intrinsics, h, w, loop=False, loop_reverse=False
    ):

        intrinsics = intrinsics.clone()
        if not self.extrinsics_c2w:
            extrinsics = extrinsics.clone().inverse()

        if extrinsics.shape[1] == 1:
            return None

        def trajectory_fn(n_interp):
            if loop:
                extrinsics_ = torch.cat([extrinsics, extrinsics[0:, 0:1]], dim=1)
            extrinsics_ = extrinsics
            b, v, _, _ = extrinsics_.shape
            extrinsics_target = interpolate_poses_spline(
                extrinsics_.reshape(b * v, 4, 4)[:, :3].cpu(), n_interp
            )
            extrinsics_target = extrinsics_target.reshape(b, -1, 4, 4).to(self.device).float()

            num_frames = b * extrinsics_target.shape[1]
            t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)

            intrinsics[..., 0, :] /= w
            intrinsics[..., 1, :] /= h
            intrinsics_target = interpolate_intrinsics(
                intrinsics[0, 0],
                intrinsics[0, -1],
                t,
            )
            intrinsics_target = intrinsics_target[None]
            return extrinsics_target, intrinsics_target

        return self.render_video_generic(
            gaussians, trajectory_fn, h, w, n_interp=24, loop_reverse=loop_reverse
        )

    def single_visualize(
        self,
        frame_index,
        view_index,
        index,
        outputs_list,
        meta_data,
        out_dir,
        mv_out_dir,
        glb_out_dir,
        gs_out_dir,
        render_rgb_paths,
    ):
        prefix = f"frame{frame_index:03d}_view{view_index:03d}_"
        outputs = outputs_list[index]
        super().visualize(outputs, meta_data, mv_out_dir, prefix=prefix)

        data_idx = meta_data["data_idx"][0]

        # Global, Point Cloud Visualization
        if outputs.glb_mv_pointmap is not None:
            save_point_cloud(
                outputs.glb_mv_pointmap.reshape(-1, 3).copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                glb_out_dir,
                f"{prefix}glb_point",
                data_idx,
                info=True,
            )
            save_depth_map(
                outputs.glb_mv_pointmap[:, :, -1].copy(),
                glb_out_dir,
                f"{prefix}glb_depth",
                data_idx,
                info=True,
            )

        # Global, Error Map Visualization
        if outputs.pointmap_gt_global is not None:
            save_point_cloud(
                outputs.pointmap_gt_global.reshape(-1, 3).copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                glb_out_dir,
                f"{prefix}glb_point_gt",
                data_idx,
                info=True,
            )

            # pointmap_shape = (outputs.pointmap_h, outputs.pointmap_w)
            # pred = outputs.glb_mv_pointmap.copy()[..., 2].reshape(pointmap_shape)
            # gt = outputs.pointmap_gt_global.copy()[..., 2].reshape(pointmap_shape)
            # save_error(pred, gt, glb_out_dir, f"{prefix}glb_error", data_idx)

        if outputs.glb2local_pointmap is not None:
            save_depth_map(
                outputs.glb2local_pointmap[:, :, -1].copy(),
                glb_out_dir,
                f"{prefix}glb2local_depth",
                data_idx,
                info=True,
            )
            if outputs.pointmap_gt is not None:
                pointmap_shape = (outputs.pointmap_h, outputs.pointmap_w)
                pred = outputs.glb2local_pointmap.copy()[..., 2].reshape(pointmap_shape)
                gt = outputs.pointmap_gt.copy()[..., 2].reshape(pointmap_shape)
                save_error(pred, gt, glb_out_dir, f"{prefix}glb2local_error", data_idx)

        # Global, Confidence Visualization
        if outputs.glb_mv_confidence is not None:
            confidence = outputs.glb_mv_confidence.copy()
            save_depth_map(confidence, glb_out_dir, f"{prefix}glb_conf", data_idx, info=True)
            confidence_mask = confidence > self.output_conf_thresh
            save_depth_map(
                confidence_mask.astype(float), glb_out_dir, f"{prefix}glb_conf_mask", data_idx
            )

            save_point_cloud(
                outputs.glb_mv_pointmap.reshape(-1, 3).copy()[confidence_mask.reshape(-1)],
                (
                    outputs.pointmap_color.copy()[confidence_mask.reshape(-1)]
                    if outputs.pointmap_color is not None
                    else None
                ),
                glb_out_dir,
                f"{prefix}filtered_glb_point",
                data_idx,
                info=True,
            )

        # Single Frame, Point Cloud Visualization
        if outputs.sf_pointmap is not None:
            save_point_cloud(
                outputs.sf_pointmap.reshape(-1, 3).copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                f"{prefix}point",
                data_idx,
                info=True,
            )
            save_depth_map(
                outputs.sf_pointmap[:, :, -1].copy(),
                out_dir,
                f"{prefix}depth",
                data_idx,
                info=True,
            )

        # Single Frame, Confidence Visualization
        if outputs.sf_confidence is not None:
            confidence = outputs.sf_confidence.copy()
            save_depth_map(confidence, out_dir, f"{prefix}sf_conf", data_idx, info=True)
            confidence_mask = (confidence > self.output_conf_thresh).astype(float)
            save_depth_map(confidence_mask, out_dir, f"{prefix}sf_conf_mask", data_idx)

        # Pose Visualization
        pose_data = dict()

        if outputs.intrinsics_pred is not None:
            pose_data["intrinsics_pred"] = outputs.intrinsics_pred.tolist()
            if outputs.intrinsics is not None:
                pose_data["intrinsics_gt"] = outputs.intrinsics.tolist()

        if outputs.extrinsics_pred is not None:
            pose_data["extrinsics_pred"] = outputs.extrinsics_pred.tolist()
            if outputs.extrinsics is not None:
                pose_data["extrinsics_gt"] = outputs.extrinsics.tolist()

        if len(pose_data) > 0:
            save_path = os.path.join(mv_out_dir, f"{prefix}pose_{data_idx:06d}.json")
            with open(save_path, "w") as f:
                json.dump(pose_data, f, indent=2)

        # Gaussians Visualization
        if outputs.render_rgb is not None:
            save_path = save_image(
                outputs.render_rgb.copy(), gs_out_dir, f"{prefix}rgb", data_idx, info=False
            )
            render_rgb_paths.append(save_path)

        if outputs.render_depth is not None:
            save_depth_map(
                outputs.render_depth.copy(), gs_out_dir, f"{prefix}depth", data_idx, info=True
            )

        if outputs.gs_xyz is not None:
            save_point_cloud(
                outputs.gs_xyz.copy(), None, gs_out_dir, f"frame{frame_index:03d}_gs_xyz", data_idx
            )

        # if outputs.gaussians is not None:
        #     gs_path = os.path.join(gs_out_dir, f"frame{frame_index:03d}_gs.ply")
        #     outputs.gaussians.write(gs_path)

    def visualize(self, outputs_list, meta_data, out_dir):
        if isinstance(outputs_list, self.output_type):
            return super().visualize(outputs_list, meta_data, out_dir)

        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        data_idx = meta_data["data_idx"][0]

        gs_out_dir = os.path.join(out_dir, f"gaussians/{data_idx:06d}")
        mv_out_dir = os.path.join(out_dir, f"mvdepth/{data_idx:06d}")
        glb_out_dir = os.path.join(out_dir, f"glb/{data_idx:06d}")
        out_dir = os.path.join(out_dir, f"depth/{data_idx:06d}")
        os.makedirs(gs_out_dir, exist_ok=True)
        os.makedirs(mv_out_dir, exist_ok=True)
        os.makedirs(glb_out_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # TODO: save gaussians
        if self.save_gaussians and outputs_list[0].gaussians is not None:
            save_path = os.path.join(gs_out_dir, "gaussians.ply")
            outputs_list[0].gaussians.write(save_path)

        if outputs_list[0].novel_render_rgb is not None:
            novel_render_rgb = outputs_list[0].novel_render_rgb.copy()  # [N, H, W, 3]
            # convert to [H, N*W, 3]
            n, h, w, _ = novel_render_rgb.shape
            novel_render_rgb = novel_render_rgb.transpose(1, 0, 2, 3).reshape(h, -1, 3)
            save_image(
                novel_render_rgb,
                gs_out_dir,
                f"novel_render_rgb",
                data_idx,
                info=True,
            )

            novel_render_depth = outputs_list[0].novel_render_depth.copy()  # [N, H, W]
            novel_render_depth = novel_render_depth.transpose(1, 0, 2).reshape(h, -1)
            save_depth_map(
                novel_render_depth, gs_out_dir, f"novel_render_depth", data_idx, info=True
            )

        if outputs_list[0].video is not None:
            save_video(outputs_list[0].video.copy(), gs_out_dir, "video", data_idx, info=True)

        if view_num == 1:
            render_rgb_paths = []
            for fi in range(frame_num):

                self.single_visualize(
                    fi,
                    0,
                    fi,
                    outputs_list,
                    meta_data,
                    out_dir,
                    mv_out_dir,
                    glb_out_dir,
                    gs_out_dir,
                    render_rgb_paths,
                )

                if len(render_rgb_paths) > 0:
                    save_path = os.path.join(
                        os.path.dirname(gs_out_dir), f"view{0:03d}_merge_render_rgb.jpg"
                    )
                    grid_images(
                        save_path=save_path,
                        paths=render_rgb_paths,
                        col=min(4, len(render_rgb_paths)),
                    )

        else:
            for fi in range(frame_num):
                render_rgb_paths = []
                for vi in range(view_num):
                    index = fi * view_num + vi

                    self.single_visualize(
                        fi,
                        vi,
                        index,
                        outputs_list,
                        meta_data,
                        out_dir,
                        mv_out_dir,
                        glb_out_dir,
                        gs_out_dir,
                        render_rgb_paths,
                    )

                    if len(render_rgb_paths) > 0:
                        save_path = os.path.join(
                            os.path.dirname(gs_out_dir), f"frame{fi:03d}_merge_render_rgb.jpg"
                        )
                        grid_images(
                            save_path=save_path,
                            paths=render_rgb_paths,
                            col=min(4, len(render_rgb_paths)),
                        )
