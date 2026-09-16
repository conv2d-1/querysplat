import logging
from collections import defaultdict

import numpy as np
import torch

from hAlgorithm.modules.utils.gaussians.novel_view_mask import calculate_loss_mask

from .reconstruct_pipeline import ReconstructPipeline


class Pi3Pipeline(ReconstructPipeline):
    """
    Pipeline for reconstructing 3D scenes from input images and depth information.
    """

    def __init__(self, train_with_pred_camera=False, test_normalize_cameras=False, **kwargs):
        super(Pi3Pipeline, self).__init__(**kwargs)

        self.train_with_pred_camera = train_with_pred_camera
        self.test_normalize_cameras = test_normalize_cameras

    def load_checkpoint(self, ckpt_path=None, state_dict=None):
        if ckpt_path is not None and ckpt_path.endswith("safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(ckpt_path)
            super().load_checkpoint(ckpt_path=None, state_dict=state_dict)
        else:
            super().load_checkpoint(ckpt_path=ckpt_path, state_dict=state_dict)

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
            _,
            prompt_mask,
            _,
            target,
            target_norm,
            valid_mask,
            _,
            _,
        ) = self.get_inputs(batch)

        # if self.mv_target_name is not None:
        #     mv_target = batch[self.mv_target_name].to(device=self.device)
        #     mv_target_norm = self.normalize(mv_target, scale=prompt_scale)

        if self.mv_extrinsics_name is not None:
            extrinsics = batch[self.mv_extrinsics_name].to(device=self.device)
        else:
            extrinsics = None

        # Forward pass through the model
        results = self.model(
            image,
            meta_data=meta_data,
            prompt_depth=prompt_depth_norm,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            novel_intrinsics=novel_intrinsics if self.novel_view_nums > 0 else None,
            novel_extrinsics=novel_extrinsics if self.novel_view_nums > 0 else None,
            render_video_with_pred_camera=self.train_with_pred_camera,
        )

        # Extract predictions from the model output
        local_points_pred = results.get("local_points", None)
        local_conf_pred = results.get("conf", None)
        camera_poses = results.get("camera_poses", None)
        # local -> global
        global_points_pred = results.get("points", None)
        # mv_confidence_pred = local_conf_pred

        gaussians = results.get("gaussians", None)
        render_rgb = results.get("render_rgb", None)
        render_depth = results.get("render_depth", None)
        render_alpha = results.get("render_alpha", None)
        # render with a larger focal
        render_large_focal_alpha = results.get("large_focal_alpha", None)
        render_normal = results.get("render_normal", None)
        render_novel_rgb = results.get("render_novel_rgb", None)
        render_novel_depth = results.get("render_novel_depth", None)

        assert local_points_pred is None or local_points_pred.shape[-3] == 3
        assert global_points_pred is None or global_points_pred.shape[-3] == 3

        total_loss, total_loss_dict = 0, dict()

        # Compute multi-view points loss if multi-view pointmap or confidence predictions are available
        # if global_points_pred is not None and global_points_pred.requires_grad:
        #     mvp_loss, mvp_loss_dict = self.get_points_loss(
        #         name=name,
        #         total_iter=total_iter,
        #         image=image,
        #         intrinsics=None,
        #         extrinsics=extrinsics,
        #         prompt_scale=prompt_scale,
        #         prompt_center=None,
        #         prompt_mask=prompt_mask,
        #         prompt_diffmap=None,
        #         target_norm=mv_target_norm,
        #         valid_mask=valid_mask,
        #         edge_mask=None,
        #         pointmap_pred=global_points_pred,
        #         confidence_pred=None,
        #     )

        #     mvp_weight = self.task_weight.get("mvp", 1.0)
        #     total_loss += mvp_loss * mvp_weight
        #     total_loss_dict["mvp"] = 0
        #     for key, val in mvp_loss_dict.items():
        #         total_loss_dict[f"mvp_{key}"] = val * mvp_weight
        #         total_loss_dict["mvp"] += val * mvp_weight

        # Compute multi-view depth loss if multi-view depthmap or confidence predictions are available
        if (local_points_pred is not None and local_points_pred.requires_grad) or (
            local_conf_pred is not None and local_conf_pred.requires_grad
        ):
            mvd_loss, mvd_loss_dict = self.get_depth_loss(
                name=name,
                total_iter=total_iter,
                image=image,
                intrinsics=intrinsics,
                prompt_scale=None,
                prompt_center=None,
                prompt_mask=prompt_mask,
                prompt_diffmap=None,
                target_norm=target_norm,
                valid_mask=valid_mask,
                edge_mask=None,
                pointmap_pred=local_points_pred,
                confidence_pred=local_conf_pred,
            )

            mvd_weight = self.task_weight.get("mvd", 1.0)
            total_loss += mvd_loss * mvd_weight
            total_loss_dict["mvd"] = 0
            for key, val in mvd_loss_dict.items():
                total_loss_dict[f"mvd_{key}"] = val * mvd_weight
                total_loss_dict["mvd"] += val * mvd_weight

        # Compute camera loss
        if camera_poses is not None and camera_poses.requires_grad:
            pose_loss, pose_loss_dict = self.camera_loss(
                c2w_pred=camera_poses,
                c2w_gt=extrinsics.inverse(),
                mask=valid_mask,
                name=name,
            )

            cm_weight = self.task_weight.get("cm", 1.0)
            total_loss += pose_loss * cm_weight
            total_loss_dict["cm"] = pose_loss * cm_weight
            total_loss_dict.update({key: val * cm_weight for key, val in pose_loss_dict.items()})

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

                    # TODO
                    # curr_render_depth_norm = self.normalize(curr_render_depth, scale=prompt_scale)
                    curr_render_depth_norm = None

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
                        intrinsics=intrinsics,
                    )
                    rc_loss += curr_rc_loss / gs_nums
                    for key, val in curr_rc_loss_dict.items():
                        rc_loss_dict[key] += val / gs_nums

            else:
                render_depth = render_depth.unsqueeze(2)
                if gaussians.norm_scales is not None:
                    render_depth = render_depth * gaussians.norm_scales[:, None, None, None, None]

                # TODO
                # render_depth_norm = self.normalize(render_depth, scale=prompt_scale)
                render_depth_norm = None

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
            _,
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

        # Perform inference using the shared step method of the model
        results = self.model(
            image,
            meta_data=meta_data,
            prompt_depth=prompt_depth_norm,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            render_video_with_pred_camera=self.render_video_with_pred_camera,
        )

        # Extract predictions from the model output
        # Multi-view depth prediction
        local_points_pred = results.get("local_points", None)
        # Confidence for multi-view depth
        local_conf_pred = results.get("conf", None)

        camera_poses = results.get("camera_poses", None)
        # Multi-view point cloud map
        global_points_pred = results.get("points", None)
        mv_confidence_pred = local_conf_pred

        # Gaussian representation of the scene
        gaussians = results.get("gaussians", None)
        # Rendered RGB image
        render_rgb = results.get("render_rgb", None)
        # Rendered depth map
        render_depth = results.get("render_depth", None)
        # Rendered normal
        # render_normal = results.get("render_normal", None)

        # Pose encoding for camera extrinsics/intrinsics
        if camera_poses is not None:
            extrinsics_pred = camera_poses.inverse()
            if prompt_scale is not None:
                extrinsics_pred[:, :, :3, 3] *= prompt_scale[:, :, :, 0, 0]
            # NOTE: model output pointmap, use gt intrinsics
            intrinsics_pred = intrinsics

            if self.test_normalize_cameras:
                # NOTE: trans to first camera
                base_c2w_pred = camera_poses[:, 0:1]
                extrinsics_pred = extrinsics_pred @ base_c2w_pred

        else:
            extrinsics_pred, intrinsics_pred = None, None

        # Convert depth maps to point clouds if necessary
        if global_points_pred is not None:
            global_points_pred = self.denormalize(global_points_pred, scale=prompt_scale)

        output_pointmap_pred = local_points_pred
        output_confidence_pred = local_conf_pred

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
                    prompt_center=None,
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

                # Add multi-view point cloud predictions if available
                if global_points_pred is not None:
                    single.glb_mv_pointmap = (
                        global_points_pred[0, index].detach().cpu().numpy().transpose(1, 2, 0)
                    )
                    if self.save_glb2local_results and extrinsics is not None:
                        glb2local_pts = self.glb_to_local(
                            global_points_pred[0:1, index : index + 1].clone(),
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
                        local_conf_pred[0, index, 0].detach().cpu().numpy()
                    )

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

        return outputs_list
