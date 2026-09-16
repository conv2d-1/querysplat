import json
import os

import numpy as np
import torch

from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    extri_intri_to_pose_encoding,
    pose_encoding_to_extri_intri,
)
from hAlgorithm.modules.pipelines.visualize import (
    save_depth_map,
    save_error,
    save_image,
    save_point_cloud,
)
from hAlgorithm.utils import grid_images, instantiate_from_config

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
        p_conf_loss=None,
        camera_loss=None,
        lpips_loss=None,
        rgb_l1_loss=None,
        ssim_loss=None,
        track_loss=None,
        disp_loss=None,
        seq_disp_loss=None,
        mv_loss_weight=0.0,
        task_weight=None,
        loss_with_local2glb=False,
        save_gaussians=False,
        output_sf_pointmap=False,
        output_stage2_pointmap=True,
        output_d2p_with_intrinsics_pred=True,
        output_track_vis_thresh=0.5,
        output_track_conf_thresh=0.0,
        depth_only_main_view=True,
        **kwargs,
    ):
        super(ReconstructPipeline, self).__init__(**kwargs)

        self.output_type = ReconstructOutput

        self.mv_target_name = mv_target_name
        self.mv_extrinsics_name = mv_extrinsics_name
        self.save_gaussians = save_gaussians
        self.output_sf_pointmap = output_sf_pointmap
        self.depth_only_main_view = depth_only_main_view

        self.p_conf_loss = instantiate_from_config(p_conf_loss)
        if self.p_conf_loss is None:
            self.p_conf_loss = self.conf_loss
        self.camera_loss = instantiate_from_config(camera_loss)
        self.rgb_l1_loss = instantiate_from_config(rgb_l1_loss)
        self.ssim_loss = instantiate_from_config(ssim_loss)
        self.lpips_loss = instantiate_from_config(lpips_loss)
        self.track_loss = instantiate_from_config(track_loss)

        # seq disp loss
        self.seq_disp_loss = instantiate_from_config(seq_disp_loss)
        self.disp_loss = instantiate_from_config(disp_loss)

        self.mv_loss_weight = mv_loss_weight
        self.task_weight = task_weight or dict(mvd=1.0, mvp=1.0, mvp2=1.0, cm=1.0)
        self.loss_with_local2glb = loss_with_local2glb

        self.output_stage2_pointmap = output_stage2_pointmap
        self.output_d2p_with_intrinsics_pred = output_d2p_with_intrinsics_pred
        self.output_track_vis_thresh = output_track_vis_thresh
        self.output_track_conf_thresh = output_track_conf_thresh

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
        # extrinsics = extrinsics.clone()
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
        meta_data=None,
    ):
        b, n, c, h, w = pointmap_pred.shape

        total_loss = 0
        total_loss_dict = dict()
        main_views = meta_data["data_info"]["main_view"]  # b, n

        for i in range(n):
            # skip other views
            if self.depth_only_main_view:
                if not main_views[:, i].any():
                    continue
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
        meta_data=None,
    ):
        b, n, c, h, w = pointmap_pred.shape

        total_loss = 0
        total_loss_dict = dict()
        main_views = meta_data["data_info"]["main_view"]  # b, n

        for i in range(n):
            # skip other views
            if self.depth_only_main_view:
                if not main_views[:, i].any():
                    continue
            loss, loss_dict = self.get_loss(
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

    def get_mv_loss(
        self,
        name,
        total_iter,
        prompt_scale,
        prompt_center,
        target,
        valid_mask,
        pointmap_pred,
    ):
        # Temporal Gradient Matching Loss
        de_pointmap_pred = self.denormalize(pointmap_pred, scale=prompt_scale, center=prompt_center)
        target_tg = (target[:, 1:, -1] - target[:, :1, -1]).abs()
        pred_tg = (de_pointmap_pred[:, 1:, -1] - de_pointmap_pred[:, :1, -1]).abs()

        tgm_loss = torch.nn.functional.l1_loss(pred_tg, target_tg, reduction="none")
        loss_mask = valid_mask[:, 1:, -1] * valid_mask[:, :1, -1] * (target_tg < 0.05)
        tgm_loss = (tgm_loss * loss_mask).sum() / loss_mask.sum() * self.mv_loss_weight
        loss_dict = dict(tgm_loss=tgm_loss)

        return tgm_loss, loss_dict

    def get_reconstruct_loss(
        self,
        name,
        image,
        render_rgb,
        render_depth,
        valid_mask=None,
    ):
        """
        Computes the reconstruction loss between rendered and target RGB images.
        Computes the pose loss.

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

        return loss, loss_dict

    def get_track_loss(
        self,
        name,
        track_pred,
        track_vis_pred,
        track_confidence_pred,
        track_query_points,
        track_vis,
        track_pos_masks,
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
        )

    def get_seq_disp_loss(
        self,
        name,
        disp_pred,
        seq_disp_pred,
        disp_gt,
        meta_data=None,
    ):
        main_view = meta_data["data_info"]["main_view"]  # b, n
        disp_pred = disp_pred.squeeze(1)  # b, c, h, w -> b, h, w
        disp_gt = disp_gt.squeeze(2)[main_view]  # b, n, c, h, w -> b, h, w
        seq_disp_pred = [disp.squeeze(1) for disp in seq_disp_pred]  # b, c, h, w -> b, h, w

        loss, loss_dict = 0, dict()
        if self.seq_disp_loss is not None:
            seq_disp_loss = self.seq_disp_loss(seq_disp_pred, disp_gt, disp_gt > 0, name=name)
            loss += seq_disp_loss
            loss_dict["seq_disp_loss"] = seq_disp_loss

        if self.disp_loss is not None:
            disp_loss = self.disp_loss(disp_pred, disp_gt, disp_gt > 0, name=name)
            loss += disp_loss
            loss_dict["disp_loss"] = disp_loss

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

        if "track_query_points" in batch:
            track_query_points = batch["track_query_points"].to(
                device=self.device, dtype=self.dtype
            )
            track_vis = batch["track_vis"].to(device=self.device)
            track_pos_masks = batch["track_pos_masks"].to(device=self.device)
        else:
            track_query_points = track_vis = track_pos_masks = None

        # Forward pass through the model
        results = self.model(
            image,
            meta_data=meta_data,
            prompt_depth=prompt_depth_norm,
            prompt_scale=prompt_scale,
            prompt_center=prompt_center,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            query_points=track_query_points[:, 0] if track_query_points is not None else None,
            with_freeze=True,
        )

        # Extract predictions from the model output
        pointmap_pred = results.get("pointmap", None)
        confidence_pred = results.get("confidence", None)

        mv_depth_pred = results.get("mv_depth", None)
        mv_depth_confidence_pred = results.get("mv_depth_confidence", None)

        mv_pointmap_pred = results.get("mv_pointmap", None)
        mv_confidence_pred = results.get("mv_confidence", None)

        stage2_mv_pointmap_pred = results.get("stage2_mv_pointmap", None)
        stage2_mv_confidence_pred = results.get("stage2_mv_confidence", None)

        track_pred = results.get("track", None)
        track_vis_pred = results.get("track_vis", None)
        track_confidence_pred = results.get("track_confidence", None)

        pose_enc = results.get("pose_enc", None)
        render_rgb = results.get("render_rgb", None)
        render_depth = results.get("render_depth", None)

        seq_disp_pred = results.get("seq_disp", None)
        disp_pred = results.get("disp", None)

        # assert mv_depth_pred is None or mv_depth_pred.shape[-3] == 1
        assert mv_pointmap_pred is None or mv_pointmap_pred.shape[-3] == 3

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
                meta_data=meta_data,
            )

            total_loss += depth_loss
            total_loss_dict = depth_loss_dict
            total_loss_dict["sfd"] = sum(depth_loss_dict.values())

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
                meta_data=meta_data,
            )

            mvp_weight = self.task_weight.get("mvp", 1.0)
            total_loss += mvp_loss * mvp_weight
            total_loss_dict["mvp"] = 0
            for key, val in mvp_loss_dict.items():
                total_loss_dict[f"mvp_{key}"] = val * mvp_weight
                total_loss_dict["mvp"] += val * mvp_weight

        # Compute multi-view points loss if multi-view pointmap or confidence predictions are available
        if (stage2_mv_pointmap_pred is not None and stage2_mv_pointmap_pred.requires_grad) or (
            stage2_mv_confidence_pred is not None and stage2_mv_confidence_pred.requires_grad
        ):
            mvp2_loss, mvp2_loss_dict = self.get_points_loss(
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
                pointmap_pred=stage2_mv_pointmap_pred,
                confidence_pred=stage2_mv_confidence_pred,
            )

            mvp2_weight = self.task_weight.get("mvp2", 1.0)
            total_loss += mvp2_loss * mvp2_weight
            total_loss_dict["mvp2"] = 0
            for key, val in mvp2_loss_dict.items():
                total_loss_dict[f"mvp2_{key}"] = val * mvp2_weight
                total_loss_dict["mvp2"] += val * mvp2_weight

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
                meta_data=meta_data,
            )

            mvd_weight = self.task_weight.get("mvd", 1.0)
            total_loss += mvd_loss * mvd_weight
            total_loss_dict["mvd"] = 0
            for key, val in mvd_loss_dict.items():
                total_loss_dict[f"mvd_{key}"] = val * mvd_weight
                total_loss_dict["mvd"] += val * mvd_weight

        # Compute camera loss
        if pose_enc is not None and pose_enc[0].requires_grad:
            pose_loss, pose_loss_dict = self.camera_loss(
                pose_enc,
                extrinsics,
                intrinsics,
                scale=prompt_scale,
                image_size_hw=image.shape[-2:],
                mask=valid_mask,
                name=name,
            )

            cm_weight = self.task_weight.get("cm", 1.0)
            total_loss += pose_loss * cm_weight
            total_loss_dict["cm"] = pose_loss * cm_weight
            total_loss_dict.update({key: val * cm_weight for key, val in pose_loss_dict.items()})

        # Compute local to global loss
        if (
            self.loss_with_local2glb
            and pose_enc is not None
            and (mv_depth_pred is not None or pointmap_pred is not None)
        ):
            # assert not self.output_d2p_with_intrinsics_pred

            if isinstance(pose_enc, (list, tuple)):
                cur_pose_enc = pose_enc[-1]
            else:
                cur_pose_enc = pose_enc

            extrinsics_pred, _ = pose_encoding_to_extri_intri(
                pose_encoding=cur_pose_enc,
                image_size_hw=None,
                translation_scale=None,
                build_intrinsics=False,
            )

            if mv_depth_pred is not None:
                dtype = mv_depth_pred.dtype
                B, S, C, H, W = mv_depth_pred.shape
                depth_local2glb = extrinsics_pred.float().inverse() @ torch.cat(
                    [mv_depth_pred, mv_depth_pred.new_ones([B, S, 1, H, W])], dim=2
                ).reshape(B * S, 4, -1)
            else:
                dtype = pointmap_pred.dtype
                S = 1
                B, C, H, W = pointmap_pred.shape
                depth_local2glb = extrinsics_pred.float().inverse() @ torch.cat(
                    [pointmap_pred, pointmap_pred.new_ones([B, 1, H, W])], dim=1
                ).reshape(B, 4, -1)

            depth_local2glb = depth_local2glb.reshape(B, S, 4, H, W)[:, :, :3, :, :].to(dtype)

            l2g_loss, l2g_loss_dict = self.get_points_loss(
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
                pointmap_pred=depth_local2glb,
                confidence_pred=None,
            )

            l2g_weight = self.task_weight.get("l2g", 1.0)
            total_loss += l2g_loss * l2g_weight
            total_loss_dict["l2g"] = 0
            for key, val in l2g_loss_dict.items():
                total_loss_dict[f"l2g_{key}"] = val * l2g_weight
                total_loss_dict["l2g"] += val * l2g_weight

        # Compute reconstruction loss
        if track_pred is not None and track_pred[0].requires_grad:
            track_loss, track_loss_dict = self.get_track_loss(
                name=name,
                track_pred=track_pred,
                track_vis_pred=track_vis_pred,
                track_confidence_pred=track_confidence_pred,
                track_query_points=track_query_points,
                track_vis=track_vis,
                track_pos_masks=track_pos_masks,
            )

            track_weight = self.task_weight.get("track", 1.0)
            total_loss += track_loss * track_weight
            total_loss_dict["tk"] = track_loss * track_weight
            total_loss_dict.update(
                {key: val * track_weight for key, val in track_loss_dict.items()}
            )

        if seq_disp_pred is not None and seq_disp_pred[0].requires_grad:

            seq_disp_loss, seq_disp_loss_dict = self.get_seq_disp_loss(
                name=name,
                disp_pred=disp_pred,
                seq_disp_pred=seq_disp_pred,
                disp_gt=batch["disp"].to(device=self.device, dtype=self.dtype),
                meta_data=meta_data,
            )
            disp_weight = self.task_weight.get("disp", 1.0)
            total_loss += seq_disp_loss * disp_weight
            total_loss_dict["disp"] = seq_disp_loss * disp_weight
            total_loss_dict.update(
                {key: val * disp_weight for key, val in seq_disp_loss_dict.items()}
            )

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

        # if self.mv_target_name is not None and self.mv_target_name in batch:
        #     mv_target = batch[self.mv_target_name].to(device=self.device)
        # else:
        #     mv_target = None

        # Handle extrinsics data if available in the batch
        if self.mv_extrinsics_name is not None:
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

        # Perform inference using the shared step method of the model
        results = self.model(
            image,
            meta_data=meta_data,
            prompt_depth=prompt_depth_norm,
            prompt_scale=prompt_scale,
            prompt_center=prompt_center,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            query_points=track_query_points[:, 0] if track_query_points is not None else None,
        )

        # Extract predictions from the model output
        # Predicted point cloud map
        pointmap_pred = results.get("pointmap", None)
        # Confidence map for predictions
        confidence_pred = results.get("confidence", None)
        # Multi-view depth prediction
        mv_depth_pred = results.get("mv_depth", None)
        # Confidence for multi-view depth
        mv_depth_confidence_pred = results.get("mv_depth_confidence", None)
        # Multi-view point cloud map
        mv_pointmap_pred = results.get("mv_pointmap", None)
        # Confidence for multi-view point cloud map
        mv_confidence_pred = results.get("mv_confidence", None)
        # Multi-view point cloud map
        stage2_mv_pointmap_pred = results.get("stage2_mv_pointmap", None)
        # Confidence for multi-view point cloud map
        stage2_mv_confidence_pred = results.get("stage2_mv_confidence", None)
        # Track results
        track_pred = results.get("track", None)
        track_vis_pred = results.get("track_vis", None)
        track_confidence_pred = results.get("track_confidence", None)
        # Pose encoding for camera extrinsics/intrinsics
        pose_enc = results.get("pose_enc", None)
        # Gaussian representation of the scene
        gaussians = results.get("gaussians", None)
        # Rendered RGB image
        render_rgb = results.get("render_rgb", None)
        # Rendered depth map
        render_depth = results.get("render_depth", None)
        # disparity map
        seq_disp_pred = results.get("seq_disp", None)
        disp_pred = results.get("disp", None)

        if track_pred is not None and isinstance(track_pred, (list, tuple)):
            track_pred = track_pred[-1]

        # Pose encoding for camera extrinsics/intrinsics
        if pose_enc is not None:
            if isinstance(pose_enc, (list, tuple)):
                pose_enc = pose_enc[-1]
            extrinsics_pred, intrinsics_pred = pose_encoding_to_extri_intri(
                pose_encoding=pose_enc,
                image_size_hw=image.shape[-2:],
                translation_scale=prompt_scale,
            )
        else:
            extrinsics_pred, intrinsics_pred = None, None

        if self.output_d2p_with_intrinsics_pred and intrinsics_pred is not None:
            K = intrinsics_pred
        else:
            K = intrinsics

        # Convert depth maps to point clouds if necessary
        if pointmap_pred is not None and pointmap_pred.shape[-3] == 1:
            pointmap_pred = self.depth_to_point(pointmap_pred, K=K)

        if mv_depth_pred is not None and mv_depth_pred.shape[-3] == 1:
            mv_depth_pred = self.depth_to_point(mv_depth_pred, K=K)

        if self.output_stage2_pointmap and stage2_mv_pointmap_pred is not None:
            # assert stage2_mv_pointmap_pred.shape[-3] == 3
            if stage2_mv_pointmap_pred.shape[-3] == 1:
                stage2_mv_pointmap_pred = self.depth_to_point(stage2_mv_pointmap_pred, K=K)

            mv_pointmap_pred = self.denormalize(
                stage2_mv_pointmap_pred, scale=prompt_scale, center=prompt_center
            )
            mv_confidence_pred = stage2_mv_confidence_pred
        elif mv_pointmap_pred is not None:
            # assert mv_pointmap_pred.shape[-3] == 3
            if mv_pointmap_pred.shape[-3] == 1:
                mv_pointmap_pred = self.depth_to_point(mv_pointmap_pred, K=K)

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
                    align_gt=align_gt[:, index] if align_gt is not None else None,
                    align_mask=align_mask[:, index] if align_mask is not None else None,
                )
                # Add disparity
                if meta_data["data_info"]["main_view"][0][index]:
                    if disp_pred is not None:
                        h, w = disp_pred.shape[-2:]
                        single.disparity = disp_pred.detach().cpu().numpy() * w
                    if seq_disp_pred is not None:
                        h, w = seq_disp_pred[-1].shape[-2:]
                        single.seq_disparity = [
                            disp_.detach().cpu().numpy() * w for disp_ in seq_disp_pred
                        ]
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
                if gaussians is not None and vi == 0:
                    # single.gaussians = gaussians[0]  # TODO: dist.all_gather_object error
                    single.gs_xyz = gaussians[0].get_xyz().detach().cpu().numpy()

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

        return outputs_list

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

        # Disparity
        if outputs.disparity is not None:
            save_depth_map(
                outputs.disparity.copy(),
                glb_out_dir,
                f"{prefix}disparity_pred_init",
                data_idx,
                info=False,
            )
        # Final Disparity
        if outputs.seq_disparity is not None:
            save_depth_map(
                outputs.seq_disparity[-1].copy(),
                glb_out_dir,
                f"{prefix}disparity_pred_final",
                data_idx,
                info=False,
            )

        # Global, Point Cloud Visualization
        if outputs.glb_mv_pointmap is not None:
            save_point_cloud(
                outputs.glb_mv_pointmap.reshape(-1, 3).copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                glb_out_dir,
                f"{prefix}glb_point",
                data_idx,
                info=False,
            )
            save_depth_map(
                outputs.glb_mv_pointmap[:, :, -1].copy(),
                glb_out_dir,
                f"{prefix}glb_depth",
                data_idx,
                info=False,
            )

        # Global, Error Map Visualization
        if outputs.pointmap_gt_global is not None:
            save_point_cloud(
                outputs.pointmap_gt_global.reshape(-1, 3).copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                glb_out_dir,
                f"{prefix}glb_point_gt",
                data_idx,
                info=False,
            )

        if outputs.glb2local_pointmap is not None:
            save_depth_map(
                outputs.glb2local_pointmap[:, :, -1].copy(),
                glb_out_dir,
                f"{prefix}glb2local_depth",
                data_idx,
                info=False,
            )
            if outputs.pointmap_gt is not None:
                pointmap_shape = (outputs.pointmap_h, outputs.pointmap_w)
                pred = outputs.glb2local_pointmap.copy()[..., 2].reshape(pointmap_shape)
                gt = outputs.pointmap_gt.copy()[..., 2].reshape(pointmap_shape)
                save_error(pred, gt, glb_out_dir, f"{prefix}glb2local_error", data_idx)

        # Global, Confidence Visualization
        if outputs.glb_mv_confidence is not None:
            confidence = outputs.glb_mv_confidence.copy()
            save_depth_map(confidence, glb_out_dir, f"{prefix}glb_conf", data_idx, info=False)
            confidence_mask = confidence > self.output_conf_thresh
            save_depth_map(
                confidence_mask.astype(float), glb_out_dir, f"{prefix}glb_conf_mask", data_idx
            )
            filtered_glb_point = outputs.glb_mv_pointmap.reshape(-1, 3).copy()[
                confidence_mask.reshape(-1)
            ]
            filtered_glb_point_color = (
                outputs.pointmap_color.copy()[confidence_mask.reshape(-1)]
                if outputs.pointmap_color is not None
                else None
            )
            save_point_cloud(
                filtered_glb_point,
                filtered_glb_point_color,
                glb_out_dir,
                f"{prefix}filtered_glb_point",
                data_idx,
                info=False,
            )
        else:
            filtered_glb_point = filtered_glb_point_color = None

        # Single Frame, Point Cloud Visualization
        if outputs.sf_pointmap is not None:
            save_point_cloud(
                outputs.sf_pointmap.reshape(-1, 3).copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                f"{prefix}point",
                data_idx,
                info=False,
            )
            save_depth_map(
                outputs.sf_pointmap[:, :, -1].copy(),
                out_dir,
                f"{prefix}depth",
                data_idx,
                info=False,
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
                outputs.render_depth.copy(), gs_out_dir, f"{prefix}depth", data_idx, info=False
            )

        if outputs.gs_xyz is not None:
            save_point_cloud(
                outputs.gs_xyz.copy(), None, gs_out_dir, f"frame{frame_index:03d}_gs_xyz", data_idx
            )

        return filtered_glb_point, filtered_glb_point_color

    def visualize(self, outputs_list, meta_data, out_dir):
        if isinstance(outputs_list, self.output_type):
            return super().visualize(outputs_list, meta_data, out_dir)

        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        data_idx = meta_data["data_idx"][0]

        gs_out_dir = os.path.join(out_dir, f"gaussians/{data_idx:06d}")
        mv_out_dir = os.path.join(out_dir, f"mvdepth/{data_idx:06d}")
        glb_out_dir = os.path.join(out_dir, f"glb/{data_idx:06d}")
        track_out_dir = os.path.join(out_dir, f"track/{data_idx:06d}")
        out_dir = os.path.join(out_dir, f"depth/{data_idx:06d}")
        os.makedirs(gs_out_dir, exist_ok=True)
        os.makedirs(mv_out_dir, exist_ok=True)
        os.makedirs(glb_out_dir, exist_ok=True)
        os.makedirs(track_out_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # TODO: save gaussians
        if self.save_gaussians and outputs_list[0].gaussians is not None:
            save_path = os.path.join(gs_out_dir, "gaussians.ply")
            outputs_list[0].gaussians.save_ply(path=save_path)

        # Global, Point Cloud Visualization
        if outputs_list[0].pointmap_color is not None:
            glb_pointmap_color = np.concatenate(
                [outputs.pointmap_color.reshape(-1, 3).copy() for outputs in outputs_list], axis=0
            )
        else:
            glb_pointmap_color = None

        if outputs_list[0].glb_mv_pointmap is not None:
            glb_pointmap = np.concatenate(
                [outputs.glb_mv_pointmap.reshape(-1, 3).copy() for outputs in outputs_list], axis=0
            )
            save_point_cloud(
                glb_pointmap,
                glb_pointmap_color,
                os.path.dirname(glb_out_dir),
                f"glb_point",
                data_idx,
                info=False,
            )
        if outputs_list[0].pointmap_gt_global is not None:
            glb_pointmap = np.concatenate(
                [outputs.pointmap_gt_global.reshape(-1, 3).copy() for outputs in outputs_list],
                axis=0,
            )
            save_point_cloud(
                glb_pointmap,
                glb_pointmap_color,
                os.path.dirname(glb_out_dir),
                f"glb_point_gt",
                data_idx,
                info=False,
            )

        # Track Visualization
        if outputs_list[0].track_pred is not None:
            from hAlgorithm.datasets_mv.track.vggt_track import visualize_tracks_on_images

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

        filtered_glb_point_list = []
        filtered_glb_point_color_list = []
        if view_num == 1:
            render_rgb_paths = []
            for fi in range(frame_num):

                filtered_glb_point, filtered_glb_point_color = self.single_visualize(
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
                filtered_glb_point_list.append(filtered_glb_point)
                filtered_glb_point_color_list.append(filtered_glb_point_color)

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

                    filtered_glb_point, filtered_glb_point_color = self.single_visualize(
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
                    filtered_glb_point_list.append(filtered_glb_point)
                    filtered_glb_point_color_list.append(filtered_glb_point_color)

                    if len(render_rgb_paths) > 0:
                        save_path = os.path.join(
                            os.path.dirname(gs_out_dir), f"frame{fi:03d}_merge_render_rgb.jpg"
                        )
                        grid_images(
                            save_path=save_path,
                            paths=render_rgb_paths,
                            col=min(4, len(render_rgb_paths)),
                        )

        if len(filtered_glb_point_list) > 0 and filtered_glb_point_list[0] is not None:
            filtered_glb_point = np.concatenate(filtered_glb_point_list, axis=0)
            if (
                len(filtered_glb_point_color_list) > 0
                and filtered_glb_point_color_list[0] is not None
            ):
                filtered_glb_point_color = np.concatenate(filtered_glb_point_color_list, axis=0)
            else:
                filtered_glb_point_color = None

            save_point_cloud(
                filtered_glb_point,
                filtered_glb_point_color,
                os.path.dirname(glb_out_dir),
                f"filtered_glb_point",
                data_idx,
                info=False,
            )
