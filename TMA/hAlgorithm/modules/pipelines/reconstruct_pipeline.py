import json
import logging
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from einops import pack

from hAlgorithm.datasets_mv.track.vggt_track import visualize_tracks_on_images
from hAlgorithm.modules.models.pi3.utils.pose_enc import pi3_pose_fov_to_extri_intri
from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from hAlgorithm.modules.pipelines.visualize import (
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
        mv_target_name=None,
        mv_extrinsics_name=None,
        sift_mask_name=None,
        with_sift_track=None,
        depth_only_z=False,
        p_l1_loss=None,
        p_conf_loss=None,
        p_normal_loss=None,
        p_depth_loss=None,
        p_grad_loss=None,
        camera_loss=None,
        lpips_loss=None,
        rgb_l1_loss=None,
        ssim_loss=None,
        render_nomal_loss=None,
        render_normal_loss=None,
        render_depth_loss=None,
        novel_lpips_loss=None,
        novel_rgb_l1_loss=None,
        novel_ssim_loss=None,
        track_loss=None,
        d_local_loss=None,
        p_local_loss=None,
        l2g_loss=None,
        vggt_loss=None,
        task_weight=None,
        data_weight=None,
        output_d2p_with_intrinsics_pred=True,
        output_track_vis_thresh=0.5,
        output_track_conf_thresh=0.0,
        novel_view_nums=0,
        render_video_with_pred_camera=False,
        refine_ba=None,
        save_everything=False,
        save_gaussians=True,
        save_render_results=True,
        save_glb_results=True,
        save_glb_sf_results=False,
        save_glb2local_results=False,
        save_local2glb_results=True,
        save_cameras=True,
        save_local_results=False,
        save_track_results=True,
        save_filtered_results=True,
        save_output_ply=False,
        save_output_conf=False,
        save_output_only_local_glb=False,
        gt_out_dir="gt",
        pose_encoding_type="absT_quaR_FoV",
        test_normalize_cameras=True,
        query_points_name=None,
        return_refine_inputs=False,
        **kwargs,
    ):
        super(ReconstructPipeline, self).__init__(**kwargs)

        self.output_type = ReconstructOutput
        # NOTE: PromptPointMapPipeline postprocess
        self.output_global_pointmap = True

        self.mv_target_name = mv_target_name
        self.mv_extrinsics_name = mv_extrinsics_name
        self.sift_mask_name = sift_mask_name
        self.with_sift_track = with_sift_track
        self.novel_view_nums = novel_view_nums
        self.depth_only_z = depth_only_z

        self.p_l1_loss = instantiate_from_config(p_l1_loss) or self.l1_loss
        self.p_normal_loss = instantiate_from_config(p_normal_loss) or self.normal_loss
        self.p_depth_loss = instantiate_from_config(p_depth_loss) or self.depth_loss
        self.p_grad_loss = instantiate_from_config(p_grad_loss) or self.grad_loss

        self.p_conf_loss = instantiate_from_config(p_conf_loss) or self.conf_loss
        self.camera_loss = instantiate_from_config(camera_loss)
        self.track_loss = instantiate_from_config(track_loss)
        self.d_local_loss = instantiate_from_config(d_local_loss)
        self.p_local_loss = instantiate_from_config(p_local_loss)
        self.l2g_loss = instantiate_from_config(l2g_loss)
        self.vggt_loss = instantiate_from_config(vggt_loss)

        self.rgb_l1_loss = instantiate_from_config(rgb_l1_loss)
        self.ssim_loss = instantiate_from_config(ssim_loss)
        self.lpips_loss = instantiate_from_config(lpips_loss)
        # TODO: delete render_nomal_loss
        self.render_normal_loss = instantiate_from_config(render_nomal_loss or render_normal_loss)
        self.render_depth_loss = instantiate_from_config(render_depth_loss)

        if self.novel_view_nums > 0:
            self.novel_rgb_l1_loss = instantiate_from_config(novel_rgb_l1_loss)
            self.novel_ssim_loss = instantiate_from_config(novel_ssim_loss)
            self.novel_lpips_loss = instantiate_from_config(novel_lpips_loss)

        self.task_weight = task_weight or dict(mvd=1.0, mvp=1.0, cm=1.0)
        self.data_weight = data_weight

        self.output_d2p_with_intrinsics_pred = output_d2p_with_intrinsics_pred
        self.output_track_vis_thresh = output_track_vis_thresh
        self.output_track_conf_thresh = output_track_conf_thresh
        self.render_video_with_pred_camera = render_video_with_pred_camera
        self.refine_ba = instantiate_from_config(refine_ba)

        self.pose_encoding_type = pose_encoding_type
        self.test_normalize_cameras = test_normalize_cameras
        self.query_points_name = query_points_name

        self.save_everything = save_everything
        self.save_gaussians = save_gaussians or self.save_everything
        self.save_render_results = save_render_results or self.save_everything
        self.save_glb_results = save_glb_results or self.save_everything
        self.save_glb_sf_results = save_glb_sf_results or self.save_everything
        self.save_glb2local_results = save_glb2local_results or self.save_everything
        self.save_local2glb_results = save_local2glb_results or self.save_everything
        self.save_cameras = save_cameras or self.save_everything
        self.save_local_results = save_local_results or self.save_everything
        self.save_track_results = save_track_results or self.save_everything
        self.save_filtered_results = save_filtered_results or self.save_everything
        self.save_output_ply = save_output_ply
        self.save_output_conf = save_output_conf
        self.save_output_only_local_glb = save_output_only_local_glb
        self.gt_out_dir = gt_out_dir

        self.return_refine_inputs = return_refine_inputs

    def split_novel_batch(self, batch):
        image = batch["image"]
        B, S, C, H, W = image.shape

        novel_batch = dict()
        novel_meta_data = dict()
        novel_data_info = dict()
        for key in batch:
            if key in ["generator", "total_iter"]:
                continue
            elif key == "meta_data":
                meta_data = batch[key]
                data_info = meta_data["data_info"]
                data_info.pop("cam_in", None)
                data_info.pop("extrinsics", None)

                if meta_data["views"][0] > 1:
                    meta_data["views"] = meta_data["views"] - self.novel_view_nums
                    novel_meta_data["views"] = (
                        torch.ones_like(meta_data["views"]) * self.novel_view_nums
                    )
                    novel_meta_data["frames"] = meta_data["frames"]
                elif meta_data["frames"][0] > 1:
                    meta_data["frames"] = meta_data["frames"] - self.novel_view_nums
                    novel_meta_data["frames"] = (
                        torch.ones_like(meta_data["frames"]) * self.novel_view_nums
                    )
                    novel_meta_data["views"] = meta_data["views"]

                for mkey in meta_data:
                    if mkey in ["data_info", "views", "frames"]:
                        continue
                    elif isinstance(meta_data[mkey], (list, tuple)):
                        novel_meta_data[mkey] = meta_data[mkey]
                    else:
                        assert isinstance(meta_data[mkey], torch.Tensor)
                        if meta_data[mkey][0].ndim == 0:
                            novel_meta_data[mkey] = meta_data[mkey]

                for dkey in data_info:
                    if isinstance(data_info[dkey], list):
                        # N, B
                        novel_data_info[dkey] = data_info[dkey][-self.novel_view_nums :]
                        data_info[dkey] = data_info[dkey][: -self.novel_view_nums]
                    else:
                        assert isinstance(data_info[dkey], torch.Tensor)
                        # B, N
                        novel_data_info[dkey] = data_info[dkey][:, -self.novel_view_nums :]
                        data_info[dkey] = data_info[dkey][:, : -self.novel_view_nums]

            else:
                data = batch[key]
                assert data.shape[0] == B and data.shape[1] == S, f"{key}: {data.shape}"
                if data.shape[0] == B and data.shape[1] == S:
                    novel_batch[key] = batch[key][:, -self.novel_view_nums :]
                    batch[key] = batch[key][:, : -self.novel_view_nums]

        novel_meta_data["data_info"] = novel_data_info
        novel_batch["meta_data"] = novel_meta_data

        return batch, novel_batch

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

    def glb_to_local(self, glb, extrinsics, scale=None, center=None):
        """
        glb: [b, v, 3, h, w], torch.Tensor
        extrinsics: [b, v, 4, 4], torch.Tensor
        scale: [b, v, 1, 1, 1], torch.Tensor
        """
        glb = self.denormalize(glb, scale=scale, center=center)
        b, v, _, h, w = glb.shape
        # Convert glb to homogeneous coordinates: [b, v, 4, h, w]
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

    def get_p_loss(
        self,
        name,
        total_iter,
        pointmap_pred,
        confidence_pred,
        target,
        intrinsics,
        valid_mask,
        edge_mask,
        prompt_mask,
        image,
        sift_mask=None,
        **kwargs,
    ):
        # Convert the prediction and target tensors to float and permute dimensions for loss calculation.
        pointmap_pred = pointmap_pred.float().permute(0, 2, 3, 1).contiguous()
        target = target.float().permute(0, 2, 3, 1).contiguous()
        valid_mask = valid_mask.squeeze(1)

        if confidence_pred is not None:
            confidence_pred = confidence_pred.squeeze(1).unsqueeze(-1)

        loss = 0
        loss_dict = dict()

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

        # Base l1 loss
        if self.p_l1_loss is not None:
            l1_Loss = self.p_l1_loss(
                pointmap_pred,
                target,
                valid_mask,
                name=name,
                image=image,
                confidence=confidence_pred,
            )
            loss = update_loss(loss, l1_Loss, "l1_loss")

        # Normal loss
        if self.p_normal_loss is not None and pointmap_pred.shape[-1] == 3:
            normal_loss = self.p_normal_loss(pointmap_pred, target, valid_mask, name=name)
            loss = update_loss(loss, normal_loss, "normal_loss")


        # Depth loss
        if self.p_depth_loss is not None:
            depth_loss = self.p_depth_loss(
                pointmap_pred[..., -1], target[..., -1], valid_mask, name=name
            )
            loss = update_loss(loss, depth_loss, "depth_loss")

        # Gradient loss
        # TODO: for global pointmap, this should be adjusted
        if self.p_grad_loss is not None:
            if getattr(self.p_grad_loss, "require_pointmap", False):
                grad_loss = self.p_grad_loss(pointmap_pred, target, valid_mask, name=name)
            else:
                grad_loss = self.p_grad_loss(
                    pointmap_pred[..., -1], target[..., -1], valid_mask, name=name
                )
            loss = update_loss(loss, grad_loss, "grad_loss")

        return loss, loss_dict

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
        # Convert the prediction and target tensors to float and permute dimensions for loss calculation.
        pointmap_pred = pointmap_pred.float().permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C
        depth_target = depth_target.float().permute(0, 2, 3, 1)
        valid_mask = valid_mask.squeeze(1)

        if confidence_pred is not None:
            confidence_pred = confidence_pred.squeeze(1).unsqueeze(3)  # B,1,H,W -> B,H,W,1

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

        return loss, loss_dict

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
            loss, loss_dict = self.get_p_loss(
                name=name,
                total_iter=total_iter,
                pointmap_pred=pointmap_pred[:, i],
                confidence_pred=confidence_pred[:, i] if confidence_pred is not None else None,
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
                    confidence_pred=confidence_pred[:, i] if confidence_pred is not None else None,
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

    def get_track_loss(
        self,
        name,
        track_pred,
        track_vis_pred,
        track_confidence_pred,
        track_query_points,
        track_vis,
        track_pos_masks,
        image_hw,
    ):
        return self.track_loss(
            name=name,
            track_preds=(
                [d.float() for d in track_pred]
                if isinstance(track_pred, (list, tuple))
                else track_pred.float()
            ),
            vis_preds=track_vis_pred.float(),
            conf_preds=track_confidence_pred.float(),
            track_gt=track_query_points.float(),
            valids=track_pos_masks.float(),
            vis=track_vis.float(),
            image_hw=image_hw,
        )

    def get_reconstruct_loss(
        self,
        name,
        image,
        depth_norm,
        render_rgb,
        render_depth=None,
        render_alpha=None,
        render_normal=None,
        gaussians=None,
        valid_mask=None,
        prompt_scale=None,
        prompt_center=None,
        intrinsics=None,
        **kwargs,
    ):
        """
        Computes the reconstruction loss between rendered and target RGB images.

        :param name: Name of the current sample.
        :param image: Target RGB image tensor.
        :param depth_norm: Target depth map tensor.
        :param render_rgb: Rendered RGB image tensor.
        :param render_depth: Rendered depth map tensor.
        :return: Tuple containing total loss and a dictionary of individual losses.
        """
        rgb = (image.float() + 1) * 0.5
        if depth_norm is not None:
            depth_norm = depth_norm.permute(0, 1, 3, 4, 2).contiguous().float()
            depth_norm = depth_norm.reshape(-1, *depth_norm.shape[-3:])
        if render_rgb is not None:
            render_rgb = render_rgb.float()
        if render_depth is not None:
            render_depth = render_depth.permute(0, 1, 3, 4, 2).contiguous().float()
            render_depth = render_depth.reshape(-1, *render_depth.shape[-3:])
        if render_alpha is not None:
            render_alpha = render_alpha.float()
        if render_normal is not None:
            render_normal = render_normal.reshape(-1, *render_normal.shape[-3:]).float()
        if valid_mask is not None:
            valid_mask = valid_mask.squeeze(2).bool()
            valid_mask = valid_mask.reshape(-1, *valid_mask.shape[-2:])

        loss, loss_dict = 0, dict()

        if self.rgb_l1_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_l1_loss = self.rgb_l1_loss(rgb, render_rgb, mask=None, name=name)
            loss += rc_l1_loss
            loss_dict["rc_l1_loss"] = rc_l1_loss

        if self.ssim_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_ssim_loss = self.ssim_loss(rgb, render_rgb, mask=None, name=name)
            loss += rc_ssim_loss
            loss_dict["rc_ssim_loss"] = rc_ssim_loss

        if self.lpips_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_lpips_loss = self.lpips_loss(rgb, render_rgb, mask=None, name=name)
            loss += rc_lpips_loss
            loss_dict["rc_lpips_loss"] = rc_lpips_loss

        if (
            self.render_normal_loss is not None
            and render_normal is not None
            and render_normal.requires_grad
        ):
            # depth_norm: [BS, H, W, 3]
            # render_normal: [BS, 3, H, W]
            # valid_mask: [BS, H, W]
            rc_norm_loss = self.render_normal_loss(
                target=depth_norm, prediction=render_normal, mask=valid_mask, name=name
            )
            loss += rc_norm_loss
            loss_dict["rc_norm_loss"] = rc_norm_loss

        if (
            self.render_depth_loss is not None
            and render_depth is not None
            and render_depth.requires_grad
        ):
            rc_dpt_loss = self.render_depth_loss(
                target=depth_norm, prediction=render_depth, mask=valid_mask, name=name
            )
            loss += rc_dpt_loss
            loss_dict["rc_dpt_loss"] = rc_dpt_loss

        return loss, loss_dict

    def get_novel_reconstruct_loss(
        self,
        name,
        image,
        depth,
        render_rgb,
        render_depth=None,
        render_alpha=None,
        render_normal=None,
        gaussians=None,
        novel_mask=None,
        valid_mask=None,
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
        # if depth is not None:
        #     depth = depth.permute(0, 1, 3, 4, 2).contiguous().float()
        #     depth = depth.reshape(-1, *depth.shape[-3:])
        if render_rgb is not None:
            render_rgb = render_rgb.float()
        # if render_depth is not None:
        #     render_depth = render_depth.reshape(-1, *render_depth.shape[-2:]).float()
        if render_alpha is not None:
            render_alpha = render_alpha.float()
        # if render_normal is not None:
        #     render_normal = render_normal.reshape(-1, *render_normal.shape[-3:]).float()
        if novel_mask is not None:
            novel_mask = novel_mask.float()
        if valid_mask is not None:
            valid_mask = valid_mask.squeeze(2).bool()
            valid_mask = valid_mask.reshape(-1, *valid_mask.shape[-2:])
            if novel_mask is not None:
                valid_mask = valid_mask * novel_mask

        loss, loss_dict = 0, dict()

        if (
            self.novel_rgb_l1_loss is not None
            and render_rgb is not None
            and render_rgb.requires_grad
        ):
            rc_l1_loss = self.novel_rgb_l1_loss(rgb, render_rgb, mask=novel_mask, name=name)
            loss += rc_l1_loss
            loss_dict["nrc_l1_loss"] = rc_l1_loss

        if self.novel_ssim_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_ssim_loss = self.novel_ssim_loss(rgb, render_rgb, mask=novel_mask, name=name)
            loss += rc_ssim_loss
            loss_dict["nrc_ssim_loss"] = rc_ssim_loss

        if (
            self.novel_lpips_loss is not None
            and render_rgb is not None
            and render_rgb.requires_grad
        ):
            rc_lpips_loss = self.novel_lpips_loss(rgb, render_rgb, mask=novel_mask, name=name)
            loss += rc_lpips_loss
            loss_dict["nrc_lpips_loss"] = rc_lpips_loss

        return loss, loss_dict

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

        if self.novel_view_nums > 0:
            # Split the current batch into two parts: base samples and novel samples
            batch, novel_batch = self.split_novel_batch(batch)
            novel_intrinsics = novel_batch[self.intrinsics_name].to(device=self.device)
            novel_extrinsics = novel_batch[self.mv_extrinsics_name].to(device=self.device)
            novel_valid_mask = novel_batch[self.target_mask_name].to(self.device)

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

        if self.mv_target_name is not None and self.mv_target_name in batch:
            mv_target = batch[self.mv_target_name].to(device=self.device)
            mv_target_norm = self.normalize(
                mv_target,
                scale=prompt_scale,
                center=prompt_center,
            )

        if self.mv_extrinsics_name is not None:
            extrinsics = batch[self.mv_extrinsics_name].to(device=self.device)
        else:
            extrinsics = None

        track_query_points = track_vis = track_pos_masks = None
        if "track_query_points" in batch:
            # if self.track_loss is not None and self.track_loss.get_loss_weight(name) > 0:
            track_query_points = batch["track_query_points"].to(
                device=self.device, dtype=self.dtype
            )
            track_vis = batch["track_vis"].to(device=self.device)
            track_pos_masks = batch["track_pos_masks"].to(device=self.device)

        sift_mask = None
        sift_track_mask = sift_track_points = sift_track_points_cam = sift_track_points_uv = None
        if self.sift_mask_name in batch:
            sift_mask = batch[self.sift_mask_name].to(device=self.device).bool()
            if self.with_sift_track:
                sift_track_points = batch["sift_track_points"].to(device=self.device)
                sift_track_points_cam = batch["sift_track_points_cam"].to(device=self.device)
                sift_track_points_uv = batch["sift_track_points_uv"].to(device=self.device)
                sift_track_mask = batch["sift_track_mask"].to(device=self.device).bool()
                sift_track_vis = batch["sift_track_vis"].to(device=self.device).bool()
        
        query_points = None
        if self.query_points_name is not None and self.query_points_name in batch:
            query_points = batch[self.query_points_name].to(device=self.device)
        elif track_query_points is not None:
            query_points = track_query_points[:, 0]

        # Forward pass through the model
        results = self.model(
            image,
            meta_data=meta_data,
            prompt_depth=prompt_depth_norm,
            prompt_scale=prompt_scale,
            prompt_center=prompt_center,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            novel_intrinsics=novel_intrinsics if self.novel_view_nums > 0 else None,
            novel_extrinsics=novel_extrinsics if self.novel_view_nums > 0 else None,
            query_points=query_points,
            with_freeze=True,
        )

        # Extract predictions from the model output
        mv_depth_pred = results.get("mv_depth", None)
        mv_depth_confidence_pred = results.get("mv_depth_confidence", None)

        mv_pointmap_pred = results.get("mv_pointmap", None)
        mv_confidence_pred = results.get("mv_confidence", None)

        track_pred = results.get("track", None)
        track_vis_pred = results.get("track_vis", None)
        track_confidence_pred = results.get("track_confidence", None)

        pose_enc = results.get("pose_enc", None)
        camera_fov = results.get("camera_fov", None)

        gaussians = results.get("gaussians", None)
        render_rgb = results.get("render_rgb", None)
        render_depth = results.get("render_depth", None)
        render_alpha = results.get("render_alpha", None)
        # render with a larger focal
        render_large_focal_alpha = results.get("large_focal_alpha", None)
        render_normal = results.get("render_normal", None)
        render_novel_rgb = results.get("render_novel_rgb", None)
        render_novel_depth = results.get("render_novel_depth", None)

        # assert mv_depth_pred is None or mv_depth_pred.shape[-3] == 1
        assert mv_pointmap_pred is None or mv_pointmap_pred.shape[-3] == 3

        if self.vggt_loss is not None:
            vggt_pred = dict(
                depth=mv_depth_pred.squeeze(2).float(),
                depth_conf=mv_depth_confidence_pred.squeeze(2).float(),
                world_points=mv_pointmap_pred.permute(0, 1, 3, 4, 2).float(),
                world_points_conf=mv_confidence_pred.squeeze(2).float(),
                pose_enc_list=pose_enc,
            )
            vggt_gt = dict(
                depths=target_norm.permute(0, 1, 3, 4, 2)[..., -1],
                world_points=mv_target_norm.permute(0, 1, 3, 4, 2),
                point_masks=valid_mask.squeeze(2),
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                images=image,
            )
            total_loss_dict = self.vggt_loss(predictions=vggt_pred, batch=vggt_gt)
            total_loss = total_loss_dict.pop("objective")

        else:
            total_loss, total_loss_dict = 0, dict()

            # Compute multi-view points loss if multi-view pointmap or confidence predictions are available
            if (mv_pointmap_pred is not None and mv_pointmap_pred.requires_grad) or (
                mv_confidence_pred is not None and mv_confidence_pred.requires_grad
            ):
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

                mvp_weight = self.task_weight.get("mvp", 1.0)
                total_loss += mvp_loss * mvp_weight
                total_loss_dict["mvp"] = 0
                for key, val in mvp_loss_dict.items():
                    total_loss_dict[f"mvp_{key}"] = val * mvp_weight
                    total_loss_dict["mvp"] += val * mvp_weight

                if self.p_local_loss is not None:
                    local_loss = self.p_local_loss(
                        prediction=mv_pointmap_pred,
                        target=mv_target_norm,
                        mask=valid_mask,
                        sift_point_mask=sift_mask,
                        sift_track_points=sift_track_points,
                        sift_track_points_cam=sift_track_points_cam,
                        sift_track_points_uv=sift_track_points_uv,
                        sift_track_mask=sift_track_mask,
                        sift_track_vis=sift_track_vis,
                        name=name,
                    )
                    total_loss += local_loss
                    total_loss_dict[f"mvp_local_loss"] = local_loss
                    total_loss_dict["mvp"] += local_loss

            # Compute multi-view depth loss if multi-view depthmap or confidence predictions are available
            if (mv_depth_pred is not None and mv_depth_pred.requires_grad) or (
                mv_depth_confidence_pred is not None and mv_depth_confidence_pred.requires_grad
            ):
                # DepthMap trans to PointMap
                if mv_depth_pred.shape[-3] == 1 and not self.depth_only_z:
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

                mvd_weight = self.task_weight.get("mvd", 1.0)
                total_loss += mvd_loss * mvd_weight
                total_loss_dict["mvd"] = 0
                for key, val in mvd_loss_dict.items():
                    total_loss_dict[f"mvd_{key}"] = val * mvd_weight
                    total_loss_dict["mvd"] += val * mvd_weight

                if self.d_local_loss is not None:
                    local_loss = self.d_local_loss(
                        prediction=mv_depth_pred,
                        target=target_norm,
                        mask=valid_mask,
                        sift_point_mask=sift_mask,
                        name=name,
                    )
                    total_loss += local_loss
                    total_loss_dict[f"mvd_local_loss"] = local_loss
                    total_loss_dict["mvd"] += local_loss

            # Compute camera loss
            if pose_enc is not None and pose_enc[0].requires_grad:
                pose_loss, pose_loss_dict = self.camera_loss(
                    pose_enc,
                    extrinsics,
                    intrinsic_gt=intrinsics,
                    scale=prompt_scale,
                    image_size_hw=image.shape[-2:],
                    mask=valid_mask,
                    name=name,
                    fov_pred=camera_fov,
                )

                cm_weight = self.task_weight.get("cm", 1.0)
                total_loss += pose_loss * cm_weight
                total_loss_dict["cm"] = pose_loss * cm_weight
                total_loss_dict.update(
                    {key: val * cm_weight for key, val in pose_loss_dict.items()}
                )

        # Compute local to global loss
        if self.l2g_loss is not None and pose_enc is not None and mv_depth_pred is not None:
            if isinstance(pose_enc, (list, tuple)):
                cur_pose_enc = pose_enc[-1]
            else:
                cur_pose_enc = pose_enc

            local_depth = results["mv_depth"]
            B, S, C, H, W = local_depth.shape
            assert C == 1

            with torch.cuda.amp.autocast(False):
                if self.pose_encoding_type == "absT_quaR_FoV":
                    extrinsics_pred, intrinsics_pred = pose_encoding_to_extri_intri(
                        pose_encoding=cur_pose_enc,
                        image_size_hw=(H, W),
                        translation_scale=None,
                        build_intrinsics=True,
                    )
                    # NOTE: first camera is global
                    # w2c_pred = extrinsics_pred
                    # base_c2w_pred = w2c_pred[:, 0:1].inverse()
                    # extrinsics_pred = w2c_pred @ base_c2w_pred
                else:
                    raise NotImplementedError
                local_points = self.depth_to_point(local_depth, K=intrinsics_pred)
                l2g_points = extrinsics_pred.inverse() @ torch.cat(
                    [local_points, local_points.new_ones([B, S, 1, H, W])], dim=2
                ).reshape(B * S, 4, -1)
                l2g_points = l2g_points.reshape(B, S, 4, H, W)[:, :, :3, :, :]

            l2g_points = l2g_points.float().permute(0, 1, 3, 4, 2)
            mv_target_norm = mv_target_norm.float().permute(0, 1, 3, 4, 2)

            if mv_depth_confidence_pred is not None:
                l2g_conf = mv_depth_confidence_pred.float().squeeze(2).unsqueeze(-1)
            else:
                l2g_conf = None

            l2g_loss_dict = self.l2g_loss(
                name=name,
                prediction=l2g_points,
                target=mv_target_norm,
                mask=valid_mask.squeeze(2),
                confidence=l2g_conf,
            )

            l2g_weight = self.task_weight.get("l2g", 1.0)
            l2g_loss = sum(l2g_loss_dict.values()) * l2g_weight
            total_loss += l2g_loss
            total_loss_dict["l2g"] = l2g_loss
            for key, val in l2g_loss_dict.items():
                total_loss_dict[f"l2g_{key}"] = val * l2g_weight

        # Compute track loss
        if track_pred is not None and track_pred[0].requires_grad:
            track_loss, track_loss_dict = self.get_track_loss(
                name=name,
                track_pred=track_pred,
                track_vis_pred=track_vis_pred,
                track_confidence_pred=track_confidence_pred,
                track_query_points=track_query_points,
                track_vis=track_vis,
                track_pos_masks=track_pos_masks,
                image_hw=image.shape[-2:],
            )

            track_weight = self.task_weight.get("track", 1.0)
            total_loss += track_loss * track_weight
            total_loss_dict["tk"] = track_loss * track_weight
            total_loss_dict.update(
                {key: val * track_weight for key, val in track_loss_dict.items()}
            )

        # Compute reconstruction loss
        if render_rgb is not None or render_depth is not None or render_normal is not None:
            if isinstance(gaussians, (list, tuple)):
                gs_nums = len(gaussians)
                rc_loss = 0
                rc_loss_dict = defaultdict(float)
                for render_idx in range(gs_nums):
                    curr_gaussians = gaussians[render_idx]

                    curr_render_depth = render_depth[render_idx].unsqueeze(2)
                    if curr_gaussians.norm_scales is not None:
                        curr_render_depth = (
                            curr_render_depth
                            * curr_gaussians.norm_scales[:, None, None, None, None]
                        )
                    curr_render_depth_norm = self.normalize(
                        curr_render_depth, scale=prompt_scale, center=prompt_center
                    )
                    curr_render_depth_norm = self.depth_to_point(
                        curr_render_depth_norm, K=intrinsics
                    )

                    curr_render_rgb = render_rgb[render_idx] if render_rgb is not None else None
                    curr_render_alpha = (
                        render_alpha[render_idx] if render_alpha is not None else None
                    )
                    curr_render_normal = (
                        render_normal[render_idx] if render_normal is not None else None
                    )

                    curr_rc_loss, curr_rc_loss_dict = self.get_reconstruct_loss(
                        name=name,
                        image=image,
                        depth_norm=target_norm,  # NOTE
                        render_rgb=curr_render_rgb,
                        render_depth=curr_render_depth_norm,
                        render_alpha=curr_render_alpha,
                        render_normal=curr_render_normal,
                        gaussians=curr_gaussians,
                        valid_mask=valid_mask,
                        prompt_scale=prompt_scale,
                        prompt_center=prompt_center,
                        intrinsics=intrinsics,
                    )
                    rc_loss += curr_rc_loss / gs_nums
                    for key, val in curr_rc_loss_dict.items():
                        rc_loss_dict[key] += val / gs_nums

            else:
                render_depth = render_depth.unsqueeze(2)
                if gaussians.norm_scales is not None:
                    render_depth = render_depth * gaussians.norm_scales[:, None, None, None, None]
                render_depth_norm = self.normalize(
                    render_depth, scale=prompt_scale, center=prompt_center
                )
                render_depth_norm = self.depth_to_point(render_depth_norm, K=intrinsics)

                rc_loss, rc_loss_dict = self.get_reconstruct_loss(
                    name=name,
                    image=image,
                    depth_norm=target_norm,  # NOTE
                    render_rgb=render_rgb,
                    render_depth=render_depth_norm,
                    render_alpha=render_alpha,
                    render_normal=render_normal,
                    gaussians=gaussians,
                    valid_mask=valid_mask,
                    prompt_scale=prompt_scale,
                    prompt_center=prompt_center,
                    intrinsics=intrinsics,
                )

            if render_large_focal_alpha is not None:
                # regularize alpha to encourage gaussian be more compact
                large_focal_alpha_loss = (1 - render_large_focal_alpha).mean() * 0.1
                rc_loss += large_focal_alpha_loss
                rc_loss_dict["rc_lf_alpha_loss"] = large_focal_alpha_loss

            rc_weight = self.task_weight.get("rc", 1.0)
            total_loss += rc_loss * rc_weight
            total_loss_dict["rc"] = rc_loss * rc_weight
            total_loss_dict.update({key: val * rc_weight for key, val in rc_loss_dict.items()})

        if self.novel_view_nums > 0:
            novel_mask = calculate_loss_mask(batch, novel_batch)
            novel_mask = novel_mask.unsqueeze(2)  # [b,v,1,h,w]->[b,v,h,w]
            nrc_loss, nrc_loss_dict = self.get_novel_reconstruct_loss(
                name=name,
                image=novel_batch["image"].to(image.device),
                depth=novel_batch["depth"].to(image.device),
                render_rgb=render_novel_rgb,
                render_depth=render_novel_depth,
                novel_mask=novel_mask,
                valid_mask=novel_valid_mask,
            )
            nrc_weight = self.task_weight.get("nrc", 1.0)
            total_loss += nrc_loss * nrc_weight
            total_loss_dict["nrc"] = nrc_loss * nrc_weight
            total_loss_dict.update({key: val * nrc_weight for key, val in nrc_loss_dict.items()})
        
        if self.data_weight is not None and name in self.data_weight:
            total_loss = total_loss * self.data_weight[name]

            for key in total_loss_dict.keys():
                total_loss_dict[key] = total_loss_dict[key] * self.data_weight[name]
            
            total_loss_dict["data_weight"] = self.data_weight[name]

        if prompt_scale is not None:
            total_loss_dict["scale"] = prompt_scale.max().cpu().item()

        if image is not None:
            aspect_ratio = image.shape[-1] / image.shape[-2]
            total_loss_dict["aspect_ratio"] = aspect_ratio
            total_loss_dict["bs"] = int(image.shape[0])
            total_loss_dict["view"] = int(image.shape[1])

        if self.return_refine_inputs:
            refine_inputs = {**results}
            refine_inputs["name"] = name
            refine_inputs["total_iter"] = total_iter
            refine_inputs["image"] = image
            refine_inputs["intrinsics"] = intrinsics
            refine_inputs["target_norm"] = target_norm
            refine_inputs["valid_mask"] = valid_mask
            refine_inputs["prompt_scale"] = prompt_scale
            refine_inputs["meta_data"] = meta_data

            refine_inputs["extrinsics"] = extrinsics
            refine_inputs["mv_target"] = mv_target

            return total_loss, total_loss_dict, refine_inputs

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

        # If "frames" or "views" are not in metadata, fallback to the superclass infer method
        if "frames" not in meta_data and "views" not in meta_data:
            return super().infer(**batch)

        # Set the model to evaluation mode.
        self.eval()

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

        # Handle extrinsics data if available in the batch
        if self.mv_extrinsics_name is not None and self.mv_extrinsics_name in batch:
            extrinsics = batch[self.mv_extrinsics_name].to(device=self.device)
        else:
            extrinsics = None

        track_query_points = track_vis = None
        if "track_query_points" in batch:
            track_query_points = batch["track_query_points"].to(
                device=self.device, dtype=self.dtype
            )
        if "track_vis" in batch:
            track_vis = batch["track_vis"].to(device=self.device)
        
        query_points = None
        if self.query_points_name is not None and self.query_points_name in batch:
            query_points = batch[self.query_points_name].to(device=self.device)
        elif track_query_points is not None:
            query_points = track_query_points[:, 0]

        # Perform inference using the shared step method of the model
        results = self.model(
            image,
            meta_data=meta_data,
            prompt_depth=prompt_depth_norm,
            prompt_scale=prompt_scale,
            prompt_center=prompt_center,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            query_points=query_points,
            render_video_with_pred_camera=self.render_video_with_pred_camera,
        )

        # Extract predictions from the model output
        # Multi-view depth prediction
        mv_depth_pred = results.get("mv_depth", None)
        # Confidence for multi-view depth
        mv_depth_confidence_pred = results.get("mv_depth_confidence", None)
        # Multi-view point cloud map
        mv_pointmap_pred = results.get("mv_pointmap", None)
        # Confidence for multi-view point cloud map
        mv_confidence_pred = results.get("mv_confidence", None)
        # Track results
        track_pred = results.get("track", None)
        track_vis_pred = results.get("track_vis", None)
        track_confidence_pred = results.get("track_confidence", None)
        # Pose encoding for camera extrinsics/intrinsics
        pose_enc = results.get("pose_enc", None)
        camera_fov = results.get("camera_fov", None)
        # Gaussian representation of the scene
        gaussians = results.get("gaussians", None)
        # Rendered RGB image
        render_rgb = results.get("render_rgb", None)
        # Rendered depth map
        render_depth = results.get("render_depth", None)
        # Rendered normal
        # render_normal = results.get("render_normal", None)

        if track_pred is not None and isinstance(track_pred, (list, tuple)):
            track_pred = track_pred[-1]

        # Pose encoding for camera extrinsics/intrinsics
        if pose_enc is not None:
            if isinstance(pose_enc, (list, tuple)):
                pose_enc = pose_enc[-1]
            if self.pose_encoding_type == "absT_quaR_FoV":
                extrinsics_pred, intrinsics_pred = pose_encoding_to_extri_intri(
                    pose_encoding=pose_enc,
                    image_size_hw=image.shape[-2:],
                    translation_scale=prompt_scale,
                )
                if self.test_normalize_cameras:
                    w2c_pred = extrinsics_pred
                    base_c2w_pred = w2c_pred[:, 0:1].inverse()
                    extrinsics_pred = w2c_pred @ base_c2w_pred
            elif self.pose_encoding_type == "pi3":
                extrinsics_pred, _ = pi3_pose_fov_to_extri_intri(
                    pose=pose_enc,
                    fov=camera_fov,
                    image_size_hw=image.shape[-2:],
                    translation_scale=prompt_scale,
                    pose_encoding_type=self.pose_encoding_type,
                    normalize_cameras=self.test_normalize_cameras,
                )
                intrinsics_pred = intrinsics
            elif self.pose_encoding_type == "pi3_fov":
                extrinsics_pred, intrinsics_pred = pi3_pose_fov_to_extri_intri(
                    pose=pose_enc,
                    fov=camera_fov,
                    image_size_hw=image.shape[-2:],
                    translation_scale=prompt_scale,
                    pose_encoding_type=self.pose_encoding_type,
                    normalize_cameras=self.test_normalize_cameras,
                )
            else:
                raise NotImplementedError
        else:
            extrinsics_pred, intrinsics_pred = None, None

        if self.output_d2p_with_intrinsics_pred and intrinsics_pred is not None:
            K = intrinsics_pred
        else:
            K = intrinsics

        # Convert depth maps to point clouds if necessary
        if mv_depth_pred is not None and mv_depth_pred.shape[-3] == 1:
            mv_depth_pred = self.depth_to_point(mv_depth_pred, K=K)

        if mv_pointmap_pred is not None:
            # assert mv_pointmap_pred.shape[-3] == 3
            if mv_pointmap_pred.shape[-3] == 1:
                mv_pointmap_pred = self.depth_to_point(mv_pointmap_pred, K=K)

            mv_pointmap_pred = self.denormalize(
                mv_pointmap_pred, scale=prompt_scale, center=prompt_center
            )

        output_pointmap_pred = mv_depth_pred
        output_confidence_pred = mv_depth_confidence_pred

        # Prepare alignment ground truth and mask if required
        align_gt = align_mask = None
        if self.match_input_res and self.align_name in batch:
            align_gt = batch[self.align_name]
        if self.match_input_res and self.post_align and self.align_mask_name in batch:
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
                    pointmap_pred=(
                        output_pointmap_pred[:, index] if output_pointmap_pred is not None else None
                    ),
                    confidence_pred=(
                        output_confidence_pred[:, index]
                        if output_confidence_pred is not None
                        else None
                    ),
                    gradient_pred=None,
                    prompt_confidence_pred=None,
                    align_gt=align_gt[:, 0] if align_gt is not None else None,
                    align_mask=align_mask[:, index] if align_mask is not None else None,
                )

                # Add multi-view point cloud predictions if available
                if mv_pointmap_pred is not None:
                    single.glb_mv_pointmap = (
                        mv_pointmap_pred[0, index].detach().cpu().numpy().transpose(1, 2, 0)
                    )
                    if self.save_glb2local_results and extrinsics is not None:
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

                # Add predicted extrinsics if available
                if extrinsics_pred is not None:
                    single.extrinsics_pred = extrinsics_pred[0, index].detach().cpu().numpy()

                # Add predicted intrinsics if available
                if intrinsics_pred is not None:
                    single.intrinsics_pred = intrinsics_pred[0, index].detach().cpu().numpy()

                if (
                    extrinsics_pred is not None
                    and self.output_d2p_with_intrinsics_pred
                    and intrinsics_pred is not None
                ):
                    local_pointmap = single.pointmap
                    local2glb_pointmap = (
                        np.linalg.inv(single.extrinsics_pred)
                        @ np.concatenate(
                            [local_pointmap, np.ones([local_pointmap.shape[0], 1])],
                            axis=-1,
                            dtype=np.float32,
                        ).T
                    )
                    single.local2glb_pointmap = local2glb_pointmap.T[:, :3]
                    single.local2glb_confidence = (
                        mv_depth_confidence_pred[0, index, 0].detach().cpu().numpy()
                    )

                if track_pred is not None:
                    single.track_pred = track_pred[0, index].detach().cpu().numpy()
                    single.track_vis_pred = track_vis_pred[0, index].detach().cpu().numpy()
                    single.track_confidence_pred = (
                        track_confidence_pred[0, index].detach().cpu().numpy()
                    )

                    single.track_gt = track_query_points[0, index].cpu().numpy()
                    if track_vis is not None:
                        single.track_vis = track_vis[0, index].cpu().numpy()

                # Add Gaussian representation if available
                # Gaussian is shared across views, and maybe across frames in the future
                if gaussians is not None and index == 0:
                    if isinstance(gaussians, (list, tuple)):
                        single.gaussians = [gaussians_i[0] for gaussians_i in gaussians]
                    else:
                        single.gaussians = gaussians[0]

                single.rgb = (image[0, index].permute(1, 2, 0).cpu().numpy() + 1) * 0.5

                # Add rendered RGB image if available
                if render_rgb is not None:
                    if isinstance(gaussians, (list, tuple)):
                        single.render_rgb = (
                            render_rgb[-1][0, index].detach().cpu().numpy().transpose(1, 2, 0)
                        )
                    else:
                        single.render_rgb = (
                            render_rgb[0, index].detach().cpu().numpy().transpose(1, 2, 0)
                        )

                # Add rendered depth map if available
                if render_depth is not None:
                    if isinstance(gaussians, (list, tuple)):
                        single.render_depth = render_depth[-1][0, index].detach().cpu().numpy()
                    else:
                        single.render_depth = render_depth[0, index].detach().cpu().numpy()

                # Store frame and view indices
                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                outputs_list.append(single)

        if self.refine_ba is not None:
            reconstruction = self.refine_ba(
                images=batch["image_raw"][0].permute(0, 3, 1, 2).to(self.device),
                outputs_list=outputs_list,
                image_size_hw=image.shape[-2:],
                dtype=self.dtype,
            )
            if reconstruction is not None:
                outputs_list[0].ba_reconstruction = reconstruction

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
            if result[result > 0].numel() == 0:
                near = 0
            else:
                near = result[result > 0][:16_000_000].quantile(0.01)
            far = result.view(-1)[:16_000_000].quantile(0.99)
            result = 1 - (result - near) / (far - near)
            # colorize_depth_maps
            return apply_color_map(result, "turbo")

        output_prob = self.model(
            gaussians=gaussians_prob,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            hw=(h, w),
            depth_mode="depth",
            only_rendering=True,
        )
        depth_color = depth_map(output_prob["render_depth"][0].detach())
        rgb = output_prob["render_rgb"][0]
        normal = (output_prob["render_normal"][0] + 1) * 0.5

        video = torch.cat([rgb, depth_color, normal], dim=3).permute(0, 2, 3, 1)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* h w c")[0]

        return video

    def render_video_interpolation(
        self, gaussians, extrinsics, intrinsics, h, w, loop=False, loop_reverse=False
    ):

        intrinsics = intrinsics.clone()
        extrinsics_inv = extrinsics.clone().inverse()

        if extrinsics_inv.shape[1] == 1:
            return None

        def trajectory_fn(n_interp):
            if loop:
                extrinsics_inv_ = torch.cat([extrinsics_inv, extrinsics_inv[0:, 0:1]], dim=1)
            extrinsics_inv_ = extrinsics_inv
            b, v, _, _ = extrinsics_inv_.shape
            extrinsics_inv_target = interpolate_poses_spline(
                extrinsics_inv_.reshape(b * v, 4, 4)[:, :3].cpu(), n_interp
            )
            extrinsics_inv_target = (
                extrinsics_inv_target.reshape(b, -1, 4, 4).to(self.device).float()
            )

            num_frames = b * extrinsics_inv_target.shape[1]
            t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)

            intrinsics[..., 0, :] /= w
            intrinsics[..., 1, :] /= h
            intrinsics_target = interpolate_intrinsics(
                intrinsics[0, 0],
                intrinsics[0, -1],
                t,
            )
            intrinsics_target = intrinsics_target[None]
            return extrinsics_inv_target, intrinsics_target

        return self.render_video_generic(
            gaussians, trajectory_fn, h, w, n_interp=24, loop_reverse=loop_reverse
        )

    def vis_images(self, outputs_list, gt_out_dir, data_idx, frame_num, view_num):
        if gt_out_dir is not None and outputs_list[0].rgb is not None:
            save_path = os.path.join(gt_out_dir, f"merge_rgb_{data_idx:06d}.jpg")
            if not os.path.exists(save_path):
                os.makedirs(gt_out_dir, exist_ok=True)
                images = [
                    (outputs.rgb[:, :, ::-1] * 255).astype(np.uint8) for outputs in outputs_list
                ]
                grid_images(
                    save_path=save_path,
                    images=images,
                    col=min(4, len(images)) if frame_num == 1 or view_num == 1 else view_num,
                )
                logging.info(f"save images: {save_path}")

                if self.save_everything:
                    for i, image in enumerate(images):
                        save_path = os.path.join(gt_out_dir, f"rgb_{data_idx:06d}_{i:03d}.jpg")
                        cv2.imwrite(save_path, image)

    def vis_render(self, outputs_list, gs_out_dir, data_idx):
        render_out_dir = os.path.join(gs_out_dir, "grid_images")
        if self.save_render_results and outputs_list[0].render_rgb is not None:
            os.makedirs(render_out_dir, exist_ok=True)
            save_path = os.path.join(render_out_dir, f"render_rgb_{data_idx:06d}.jpg")
            render_rgb = [
                (np.clip(outputs.render_rgb[..., ::-1], 0.0, 1.0) * 255).astype(np.uint8)
                for outputs in outputs_list
            ]
            grid_images(save_path, images=render_rgb)

        if self.save_render_results and outputs_list[0].render_depth is not None:
            os.makedirs(render_out_dir, exist_ok=True)
            save_path = os.path.join(render_out_dir, f"render_dpt_{data_idx:06d}.jpg")

            def colorize(depth):
                min_val, max_val = depth.min(), depth.max()
                depth_norm = 1 - (depth - min_val) / (max_val - min_val + 1e-9)
                depth_colored = apply_color_map(depth_norm, "turbo")
                return (depth_colored.clip(0, 1) * 255).astype(np.uint8)

            render_depth = [colorize(outputs.render_depth) for outputs in outputs_list]
            grid_images(save_path, images=render_depth)

    def vis_render_video(self, outputs_list, gs_out_dir, data_idx, frame_num, view_num):
        os.makedirs(gs_out_dir, exist_ok=True)
        gaussians = outputs_list[0].gaussians
        if isinstance(gaussians, (list, tuple)):
            for gi, gaussians_i in enumerate(gaussians):
                save_path = os.path.join(gs_out_dir, f"gaussians_{data_idx:06d}_{gi}.ply")
                gaussians_i.export_ply().write(save_path)
            gaussians = gaussians[-1]
        else:
            save_path = os.path.join(gs_out_dir, f"gaussians_{data_idx:06d}.ply")
            gaussians.export_ply().write(save_path)

        if self.save_render_results:
            if (
                outputs_list[0].extrinsics is None
                or outputs_list[0].intrinsics is None
                or self.render_video_with_pred_camera
            ):
                extrinsics = torch.stack(
                    [torch.from_numpy(outputs.extrinsics_pred) for outputs in outputs_list], dim=0
                )
                intrinsics = torch.stack(
                    [torch.from_numpy(outputs.intrinsics_pred) for outputs in outputs_list], dim=0
                )
            else:
                extrinsics = torch.stack(
                    [torch.from_numpy(outputs.extrinsics) for outputs in outputs_list], dim=0
                )
                intrinsics = torch.stack(
                    [torch.from_numpy(outputs.intrinsics) for outputs in outputs_list], dim=0
                )

            extrinsics = extrinsics.reshape(-1, frame_num * view_num, 4, 4).float().to(self.device)
            intrinsics = intrinsics.reshape(-1, frame_num * view_num, 3, 3).float().to(self.device)

            render_video = self.render_video_interpolation(
                gaussians,
                extrinsics,
                intrinsics,
                outputs_list[0].pointmap_h,
                outputs_list[0].pointmap_w,
            )
            save_video(render_video, gs_out_dir, "video", data_idx, info=True)

    def filtered_points_with_confidence(self, points, confidence, h, w, colors=None):
        if confidence.shape[0] != h or confidence.shape[1] != w:
            confidence = cv2.resize(
                confidence,
                dsize=(w, h),
                interpolation=cv2.INTER_LINEAR,
            )
        confidence = confidence.reshape(-1)

        if self.output_conf_ratio is not None:
            conf_thresh = np.percentile(confidence, self.output_conf_ratio * 100)
            confidence_mask = confidence > conf_thresh
        else:
            confidence_mask = confidence > self.output_conf_thresh

        filtered_points = points[confidence_mask]
        filtered_colors = colors[confidence_mask] if colors is not None else None

        return filtered_points, filtered_colors

    def vis_glb_results(
        self, outputs_list, glb_out_dir, data_idx, frame_num, view_num, gt_out_dir=None
    ):
        if gt_out_dir is not None and outputs_list[0].pointmap_gt_global is not None:
            pointmap_gt_global = [
                outputs.pointmap_gt_global.reshape(-1, 3).copy() for outputs in outputs_list
            ]
            merge_pointmap_gt_global = np.concatenate(pointmap_gt_global, axis=0)
        else:
            pointmap_gt_global = merge_pointmap_gt_global = None

        if outputs_list[0].glb_mv_pointmap is not None:
            glb_mv_pointmap = [
                outputs.glb_mv_pointmap.reshape(-1, 3).copy() for outputs in outputs_list
            ]
            merge_glb_mv_pointmap = np.concatenate(glb_mv_pointmap, axis=0)
        else:
            glb_mv_pointmap = merge_glb_mv_pointmap = None

        if outputs_list[0].local2glb_pointmap is not None:
            local2glb_pointmap = [
                outputs.local2glb_pointmap.reshape(-1, 3).copy() for outputs in outputs_list
            ]
            merge_local2glb_pointmap = np.concatenate(local2glb_pointmap, axis=0)
        else:
            local2glb_pointmap = merge_local2glb_pointmap = None

        if outputs_list[0].pointmap_color is not None:
            pointmap_color = [
                outputs.pointmap_color.reshape(-1, 3).copy() for outputs in outputs_list
            ]
            merge_pointmap_color = np.concatenate(pointmap_color, axis=0)
        else:
            pointmap_color = merge_pointmap_color = None

        if merge_pointmap_gt_global is not None:
            chech_path = os.path.join(gt_out_dir, f"glb_points_gt_{data_idx:06d}.ply")
            if not os.path.exists(chech_path):
                os.makedirs(gt_out_dir, exist_ok=True)
                save_point_cloud(
                    merge_pointmap_gt_global,
                    merge_pointmap_color,
                    gt_out_dir,
                    "glb_points_gt",
                    data_idx,
                    info=True,
                )

        if merge_glb_mv_pointmap is not None:
            os.makedirs(glb_out_dir, exist_ok=True)
            save_point_cloud(
                merge_glb_mv_pointmap,
                merge_pointmap_color,
                glb_out_dir,
                "glb_points",
                data_idx,
                info=False,
            )
            if self.save_filtered_results and outputs_list[0].glb_mv_confidence is not None:
                filtered_points = []
                filtered_points_color = []
                for idx, outputs in enumerate(outputs_list):
                    points = glb_mv_pointmap[idx]
                    colors = pointmap_color[idx] if pointmap_color is not None else None
                    points, colors = self.filtered_points_with_confidence(
                        points=points,
                        confidence=outputs.glb_mv_confidence.copy(),
                        h=outputs.pointmap_h,
                        w=outputs.pointmap_w,
                        colors=colors,
                    )
                    filtered_points.append(points)
                    if colors is not None:
                        filtered_points_color.append(colors)

                filtered_points = np.concatenate(filtered_points, axis=0)
                if len(filtered_points_color) > 0:
                    filtered_points_color = np.concatenate(filtered_points_color, axis=0)
                else:
                    filtered_points_color = None

                save_point_cloud(
                    filtered_points,
                    filtered_points_color,
                    glb_out_dir,
                    f"filtered_glb_points",
                    data_idx,
                    info=False,
                )

        if self.save_local2glb_results and merge_local2glb_pointmap is not None:
            os.makedirs(glb_out_dir, exist_ok=True)
            save_point_cloud(
                merge_local2glb_pointmap,
                merge_pointmap_color,
                glb_out_dir,
                f"lcl2glb_points",
                data_idx,
                info=False,
            )
            if self.save_filtered_results and outputs_list[0].confidence is not None:
                filtered_points = []
                filtered_points_color = []
                for idx, outputs in enumerate(outputs_list):
                    points = local2glb_pointmap[idx]
                    colors = pointmap_color[idx] if pointmap_color is not None else None
                    points, colors = self.filtered_points_with_confidence(
                        points=points,
                        confidence=outputs.local2glb_confidence.copy(),
                        h=outputs.pointmap_h,
                        w=outputs.pointmap_w,
                        colors=colors,
                    )
                    filtered_points.append(points)
                    if colors is not None:
                        filtered_points_color.append(colors)

                filtered_points = np.concatenate(filtered_points, axis=0)
                if len(filtered_points_color) > 0:
                    filtered_points_color = np.concatenate(filtered_points_color, axis=0)
                else:
                    filtered_points_color = None

                save_point_cloud(
                    filtered_points,
                    filtered_points_color,
                    glb_out_dir,
                    f"filtered_lcl2glb_points",
                    data_idx,
                    info=False,
                )

        if self.save_glb_sf_results or self.save_glb2local_results:
            curr_glb_out_dir = os.path.join(glb_out_dir, f"{data_idx:06d}")
            os.makedirs(curr_glb_out_dir, exist_ok=True)
            for frame_index in range(frame_num):
                for view_index in range(view_num):
                    prefix = f"f{frame_index:03d}_v{view_index:03d}_"
                    index = frame_index * view_num + view_index
                    # prefix = f"idx{index:04d}_"
                    colors = pointmap_color[index] if pointmap_color is not None else None

                    if self.save_glb_sf_results and glb_mv_pointmap is not None:
                        save_point_cloud(
                            glb_mv_pointmap[index],
                            colors,
                            curr_glb_out_dir,
                            f"{prefix}glb_points",
                            info=False,
                        )
                    if self.save_glb_sf_results and pointmap_gt_global is not None:
                        chech_path = os.path.join(gt_out_dir, f"{prefix}glb_points_gt.ply")
                        if not os.path.exists(chech_path):
                            os.makedirs(gt_out_dir, exist_ok=True)
                            save_point_cloud(
                                pointmap_gt_global[index],
                                colors,
                                gt_out_dir,
                                f"{prefix}glb_points_gt",
                                info=True,
                            )

                    if (
                        self.save_glb2local_results
                        and outputs_list[index].glb2local_pointmap is not None
                    ):
                        save_point_cloud(
                            outputs_list[index].glb2local_pointmap.reshape(-1, 3).copy(),
                            colors,
                            curr_glb_out_dir,
                            f"{prefix}glb2local_points",
                            info=False,
                        )
                    
                    if self.save_local2glb_results:
                        save_point_cloud(
                            local2glb_pointmap[index],
                            colors,
                            curr_glb_out_dir,
                            f"{prefix}lcl2glb_points",
                            info=False,
                        )


    def vis_camera_results(self, outputs_list, camera_out_dir, frame_num, view_num):
        for frame_index in range(frame_num):
            for view_index in range(view_num):
                prefix = f"f{frame_index:03d}_v{view_index:03d}_"
                index = frame_index * view_num + view_index
                # prefix = f"idx{index:04d}_"
                outputs = outputs_list[index]

                if outputs.intrinsics_pred is not None:
                    os.makedirs(camera_out_dir, exist_ok=True)
                    data = dict(pred=outputs.intrinsics_pred.tolist())
                    if outputs.intrinsics is not None:
                        data["gt"] = outputs.intrinsics.tolist()

                    save_path = os.path.join(camera_out_dir, f"{prefix}intris.json")
                    with open(save_path, "w") as f:
                        json.dump(data, f, indent=2)

                if outputs.extrinsics_pred is not None:
                    os.makedirs(camera_out_dir, exist_ok=True)
                    data = dict(pred=outputs.extrinsics_pred.tolist())
                    if outputs.extrinsics is not None:
                        data["gt"] = outputs.extrinsics.tolist()

                    save_path = os.path.join(camera_out_dir, f"{prefix}extris.json")
                    with open(save_path, "w") as f:
                        json.dump(data, f, indent=2)

    def vis_local_results(self, outputs_list, mv_out_dir, frame_num, view_num, meta_data):
        data_info = meta_data.pop("data_info", None)
        for frame_index in range(frame_num):
            for view_index in range(view_num):
                prefix = f"f{frame_index:03d}_v{view_index:03d}_"
                index = frame_index * view_num + view_index
                meta_data["data_info"] = [data_info[0][index]]

                super().visualize(
                    outputs_list[index], meta_data, mv_out_dir, prefix=prefix
                )

    def vis_track_results(self, outputs_list, track_out_dir, data_idx, gt_out_dir=None):
        if outputs_list[0].track_pred is not None:
            track_pred = torch.stack(
                [torch.from_numpy(outputs.track_pred) for outputs in outputs_list], dim=0
            )
            track_vis_pred = torch.stack(
                [torch.from_numpy(outputs.track_vis_pred) for outputs in outputs_list], dim=0
            )

            track_vis_pred = track_vis_pred >= self.output_track_vis_thresh
            track_confidence_pred = torch.stack(
                [torch.from_numpy(outputs.track_confidence_pred) for outputs in outputs_list], dim=0
            )
            track_vis_pred = track_vis_pred & (
                track_confidence_pred >= self.output_track_conf_thresh
            )

            pointmap_h = outputs_list[0].pointmap_h
            pointmap_w = outputs_list[0].pointmap_w

            image = torch.stack(
                [
                    torch.from_numpy(
                        outputs.pointmap_color[:, [2, 1, 0]].reshape(pointmap_h, pointmap_w, 3)
                    )
                    for outputs in outputs_list
                ],
                dim=0,
            )
            visualize_tracks_on_images(
                image[None],
                track_pred[None],
                track_vis_mask=track_vis_pred[None],
                out_dir=track_out_dir,
                image_format="HWC",  # "CHW" or "HWC"
                normalize_mode="",
                cmap_name="hsv",
            )

        if gt_out_dir is not None and outputs_list[0].track_gt is not None:
            os.makedirs(gt_out_dir, exist_ok=True)
            track_gt = torch.stack(
                [torch.from_numpy(outputs.track_gt) for outputs in outputs_list], dim=0
            )
            track_vis = torch.stack(
                [torch.from_numpy(outputs.track_vis) for outputs in outputs_list], dim=0
            )
            pointmap_h = outputs_list[0].pointmap_h
            pointmap_w = outputs_list[0].pointmap_w

            image = torch.stack(
                [
                    torch.from_numpy(
                        outputs.pointmap_color[:, [2, 1, 0]].reshape(pointmap_h, pointmap_w, 3)
                    )
                    for outputs in outputs_list
                ],
                dim=0,
            )
            visualize_tracks_on_images(
                image[None],
                track_gt[None],
                track_vis_mask=track_vis[None],
                out_dir=gt_out_dir,
                image_format="HWC",  # "CHW" or "HWC"
                normalize_mode="",
                cmap_name="hsv",
                prefix="gt_",
            )

    def get_out_dir(self, out_dir, data_idx=None):
        gs_out_dir = os.path.join(out_dir, "gaussians")
        glb_out_dir = os.path.join(out_dir, "glb")

        if data_idx is not None:
            camera_out_dir = os.path.join(out_dir, f"camera/{data_idx:06d}")
            mv_out_dir = os.path.join(out_dir, f"mvdepth/{data_idx:06d}")
            track_out_dir = os.path.join(out_dir, f"track/{data_idx:06d}")
        else:
            camera_out_dir = os.path.join(out_dir, "camera")
            mv_out_dir = os.path.join(out_dir, "mvdepth")
            track_out_dir = os.path.join(out_dir, "track")

        return gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir

    def get_gt_out_dir(self, out_dir):
        if self.gt_out_dir is None:
            return None, None, None

        abs_gt_out_dir = os.path.join(
            os.path.dirname(os.path.dirname(out_dir)), self.gt_out_dir, os.path.basename(out_dir)
        )
        _, gt_glb_out_dir, _, _, gt_track_out_dir = self.get_out_dir(abs_gt_out_dir, data_idx=None)

        return abs_gt_out_dir, gt_glb_out_dir, gt_track_out_dir

    def visualize(self, outputs_list, meta_data, out_dir):
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        data_idx = meta_data["data_idx"][0]

        gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir = self.get_out_dir(
            out_dir, data_idx=data_idx
        )
        gt_out_dir, gt_glb_out_dir, gt_track_out_dir = self.get_gt_out_dir(out_dir)

        # Images Visualization
        if gt_out_dir is not None:
            self.vis_images(outputs_list, gt_out_dir, data_idx, frame_num, view_num)

        # Gaussians Visualization
        if self.save_gaussians and outputs_list[0].gaussians is not None:
            self.vis_render(outputs_list, gs_out_dir, data_idx)
            self.vis_render_video(outputs_list, gs_out_dir, data_idx, frame_num, view_num)

        # Global results Visualization
        if self.save_glb_results:
            self.vis_glb_results(
                outputs_list, glb_out_dir, data_idx, frame_num, view_num, gt_out_dir=gt_glb_out_dir
            )

        # Camera Visualization
        if self.save_cameras:
            self.vis_camera_results(outputs_list, camera_out_dir, frame_num, view_num)

        # Local results Visualization
        if self.save_local_results:
            os.makedirs(mv_out_dir, exist_ok=True)
            self.vis_local_results(outputs_list, mv_out_dir, frame_num, view_num, meta_data)

        # Track Visualization
        if self.save_track_results and outputs_list[0].track_pred is not None:
            os.makedirs(track_out_dir, exist_ok=True)
            self.vis_track_results(
                outputs_list, track_out_dir, data_idx, gt_out_dir=gt_track_out_dir
            )

        # BA Visualization
        if outputs_list[0].ba_reconstruction is not None:
            from hAlgorithm.modules.models.vggt.ba.np_to_pycolmap import pycolmap_to_batch_np_matrix

            points3D, points3D_rgb, extrinsics, intrinsics, extra_params = (
                pycolmap_to_batch_np_matrix(outputs_list[0].ba_reconstruction)
            )
            os.makedirs(glb_out_dir, exist_ok=True)
            save_point_cloud(points3D, points3D_rgb, glb_out_dir, "ba_points", data_idx, info=False)

    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        if not self.save_output_only_local_glb and outputs[0].ba_reconstruction is not None:
            from hAlgorithm.modules.models.vggt.ba.np_to_pycolmap import pycolmap_to_batch_np_matrix

            points3D, points3D_rgb, extrinsics, intrinsics, extra_params = (
                pycolmap_to_batch_np_matrix(outputs[0].ba_reconstruction)
            )
            os.makedirs(out_dir, exist_ok=True)
            save_point_cloud(points3D, points3D_rgb, out_dir, "ba_points", None, info=False)

        data_info = meta_data["data_info"][0]

        for i, outputs_sf in enumerate(outputs):
            out_dir = os.path.abspath(out_dir)
            scene = data_info[i]["scene"]
            frame_id = int(data_info[i]["frame_id"])
            view_id = int(data_info[i]["view_id"])
            depth_scale = float(data_info[i]["depth_scale"])

            cur_out_dir = cur_out_dir = os.path.join(out_dir, scene)
            os.makedirs(cur_out_dir, exist_ok=True)

            save_path = os.path.join(cur_out_dir, f"depth_{frame_id:06d}_{view_id:06d}.npy")

            # NOTE: local depth * depth_scale
            if not self.save_output_only_local_glb:
                np.save(save_path, outputs_sf.depth_align * depth_scale)
            logging.info(f"output save: {save_path}")

            points_save_dir = cur_out_dir
            points_save_index = view_id

            points_save_path = glb_points_save_path = None
            if self.save_output_ply:
                pointmap_color = (
                    outputs_sf.pointmap_color.copy() if outputs_sf.pointmap_color is not None else None
                )
                if outputs_sf.pointmap is not None:
                    points_save_path = save_point_cloud(
                        outputs_sf.pointmap.copy(),
                        pointmap_color,
                        points_save_dir,
                        f"points_{frame_id:06d}",
                        points_save_index,
                        info=False,
                    )

                filtered_point_save_path = None
                # if not self.save_output_only_local_glb and outputs_sf.filtered_pointmap is not None:
                #     filtered_point_save_path = save_point_cloud(
                #         outputs_sf.filtered_pointmap.copy(),
                #         (
                #             outputs_sf.filtered_pointmap_color.copy()
                #             if outputs_sf.filtered_pointmap_color is not None
                #             else None
                #         ),
                #         points_save_dir,
                #         f"filtered_point_{frame_id:06d}",
                #         points_save_index,
                #     )

            glb_points_save_path = filtered_glb_points_path = None
            # if not self.save_output_only_local_glb and self.save_output_ply and outputs_sf.glb_mv_pointmap is not None:
            #     glb_mv_pointmap = outputs_sf.glb_mv_pointmap.copy().reshape(-1, 3)
            #     glb_points_save_path = save_point_cloud(
            #         glb_mv_pointmap,
            #         pointmap_color,
            #         points_save_dir,
            #         f"glb_points_{frame_id:06d}",
            #         points_save_index,
            #         info=False,
            #     )

            #     if outputs_sf.glb_mv_confidence is not None:
            #         confidence = outputs_sf.glb_mv_confidence.copy()
            #         filtered_points, filtered_colors = self.filtered_points_with_confidence(
            #             points=glb_mv_pointmap,
            #             confidence=confidence,
            #             h=outputs_sf.pointmap_h,
            #             w=outputs_sf.pointmap_w,
            #             colors=pointmap_color,
            #         )
            #         filtered_glb_points_path = save_point_cloud(
            #             filtered_points,
            #             filtered_colors,
            #             points_save_dir,
            #             f"filtered_glb_points_{frame_id:06d}",
            #             points_save_index,
            #             info=False,
            #         )

            intrinsics = outputs_sf.intrinsics_pred
            extrinsics = outputs_sf.extrinsics_pred
            intrinsics_save_path = extrinsics_save_path = local2glb_points_save_path = None
            if intrinsics is not None:
                # infer 时 intrinsics 对应 model input 尺寸
                # save output 时 intrinsics 对应输出的 depth 尺寸
                if self.match_input_res:
                    h, w = outputs_sf.depth_align.shape[:2]
                    ratio_h, ratio_w = h / outputs_sf.pointmap_h, w / outputs_sf.pointmap_w
                    intrinsics_align = intrinsics.copy()
                    intrinsics_align[0, 0] = intrinsics_align[0, 0] * ratio_w
                    intrinsics_align[1, 1] = intrinsics_align[1, 1] * ratio_h
                    intrinsics_align[0, 2] = intrinsics_align[0, 2] * ratio_w
                    intrinsics_align[1, 2] = intrinsics_align[1, 2] * ratio_h
                    intrinsics = intrinsics_align
                intrinsics_save_path = os.path.join(
                    os.path.dirname(save_path), f"intrinsics_{frame_id:06d}_{view_id:06d}.json"
                )
                with open(intrinsics_save_path, "w") as f:
                    json.dump(intrinsics.tolist(), f, indent=2)

            local2glb_points_save_path = filtered_local2glb_points_path = None
            if extrinsics is not None:
                extrinsics_save_path = os.path.join(
                    os.path.dirname(save_path), f"extrinsics_{frame_id:06d}_{view_id:06d}.json"
                )
                with open(extrinsics_save_path, "w") as f:
                    json.dump(extrinsics.tolist(), f, indent=2)

                local2glb_pointmap = outputs_sf.local2glb_pointmap
                if not self.save_output_only_local_glb and self.save_output_ply and local2glb_pointmap is not None:
                    local2glb_points_save_path = save_point_cloud(
                        local2glb_pointmap,
                        (
                            outputs_sf.pointmap_color.copy()
                            if outputs_sf.pointmap_color is not None
                            else None
                        ),
                        points_save_dir,
                        f"lcl2glb_points_{frame_id:06d}",
                        points_save_index,
                        info=False,
                    )

                    if outputs_sf.confidence is not None:
                        confidence = outputs_sf.confidence.copy()
                        filtered_points, filtered_colors = self.filtered_points_with_confidence(
                            points=local2glb_pointmap,
                            confidence=confidence,
                            h=outputs_sf.pointmap_h,
                            w=outputs_sf.pointmap_w,
                            colors=pointmap_color,
                        )
                        filtered_local2glb_points_path = save_point_cloud(
                            filtered_points,
                            filtered_colors,
                            points_save_dir,
                            f"filtered_lcl2glb_points_{frame_id:06d}",
                            points_save_index,
                            info=False,
                        )

            confidence_save_path = None
            if not self.save_output_only_local_glb and self.save_output_conf and outputs_sf.confidence is not None:
                confidence_save_path = os.path.join(
                    os.path.dirname(save_path), f"conf_{frame_id:06d}_{view_id:06d}.npy"
                )
                np.save(confidence_save_path, outputs_sf.confidence)

            glb_confidence_save_path = None
            # if not self.save_output_only_local_glb and self.save_output_conf and outputs_sf.glb_mv_confidence is not None:
            #     glb_confidence_save_path = os.path.join(
            #         os.path.dirname(save_path), f"glb_conf_{frame_id:06d}_{view_id:06d}.npy"
            #     )
            #     np.save(glb_confidence_save_path, outputs_sf.glb_mv_confidence)

            if output_meta_dict is not None:
                info = dict(
                    rgb=data_info[i]["rgb"],
                    depth_scale=depth_scale,
                    pred_depth=save_path,
                )
                if "cam_in" in data_info[i]:
                    info["cam_in"] = data_info[i]["cam_in"]
                if intrinsics_save_path is not None:
                    info["pred_intrinsic"] = intrinsics_save_path
                if extrinsics_save_path is not None:
                    info["pred_extrinsic"] = extrinsics_save_path
                if points_save_path is not None:
                    info["pred_points"] = points_save_path
                if confidence_save_path is not None:
                    info["pred_confidence"] = confidence_save_path
                if glb_points_save_path is not None:
                    info["pred_glb_points"] = glb_points_save_path
                if glb_confidence_save_path is not None:
                    info["pred_glb_confidence"] = glb_confidence_save_path
                if filtered_glb_points_path is not None:
                    info["pred_filtered_glb_points"] = filtered_glb_points_path
                if local2glb_points_save_path is not None:
                    info["pred_local2glb_points"] = local2glb_points_save_path
                if filtered_local2glb_points_path is not None:
                    info["pred_filtered_local2glb_points"] = filtered_local2glb_points_path

                if "depth" in data_info[i]:
                    info["depth"] = data_info[i]["depth"]
                if "lidar_depth" in data_info[i]:
                    info["lidar_depth"] = data_info[i]["lidar_depth"]
                if "confidence" in data_info[i]:
                    info["confidence"] = data_info[i]["confidence"]

                if "mf_files" not in output_meta_dict:
                    output_meta_dict["mf_files"] = dict()
                if scene not in output_meta_dict["mf_files"]:
                    output_meta_dict["mf_files"][scene] = []
                frame_ids = [d["frame_id"] for d in output_meta_dict["mf_files"][scene]]
                if len(frame_ids) == 0:
                    index = None
                else:
                    if frame_id in set(frame_ids):
                        index = frame_ids.index(frame_id)
                    else:
                        index = None

                info["view_id"] = view_id
                if "extrinsics" in data_info[i]:
                    info["extrinsics"] = data_info[i]["extrinsics"]

                if index is None:
                    output_meta_dict["mf_files"][scene].append(
                        dict(
                            frame_id=frame_id,
                            views=[info],
                        )
                    )
                else:
                    output_meta_dict["mf_files"][scene][index]["views"].append(info)
            else:
                raise NotImplementedError