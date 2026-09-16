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
from hAlgorithm.modules.utils.image_utils import get_img_grad_weight
from hAlgorithm.utils import apply_color_map, grid_images, instantiate_from_config

from .outputs import ReconstructOutput
from .prompt_pointmap_pipeline_v2 import PromptPointMapPipeline


class Gaussian_Finetuning_Pipeline(PromptPointMapPipeline):
    """
    Pipeline for finetuning 3DGS from feedforward GS with input novel images.
    """

    def __init__(
        self,
        p_conf_loss=None,
        mv_extrinsics_name=None,
        lpips_loss=None,
        rgb_l1_loss=None,
        ssim_loss=None,
        save_gaussians=False,
        pts_glb2local=False,
        conf_pts_glb2local=False,
        enable_novel_view_loss=True,
        gs_near=0.01,
        gs_far=100.0,
        **kwargs,
    ):
        super(Gaussian_Finetuning_Pipeline, self).__init__(**kwargs)

        self.output_type = ReconstructOutput
        self.extrinsics_c2w = self.model.extrinsics_c2w

        self.mv_extrinsics_name = mv_extrinsics_name
        self.rgb_l1_loss = instantiate_from_config(rgb_l1_loss)
        self.ssim_loss = instantiate_from_config(ssim_loss)
        self.lpips_loss = instantiate_from_config(lpips_loss)

        self.save_gaussians = save_gaussians

        self.pts_glb2local = pts_glb2local
        self.conf_pts_glb2local = conf_pts_glb2local and (not self.pts_glb2local)

        self.gs_near = gs_near
        self.gs_far = gs_far
        self.enable_novel_view_loss = enable_novel_view_loss

    def get_inputs(self, batch):
        # Extract and move tensors to the appropriate device and dtype if necessary.
        name = batch["meta_data"]["name"][0]
        total_iter = batch.get("total_iter", None)
        image = batch["image"].to(device=self.device, dtype=self.dtype)

        intrinsics = batch.get("intrinsics", None)
        extrinsics = batch.get(self.mv_extrinsics_name, None)  # [n, 4, 4]
        # extrinsics = batch.get("extrinsics", None)

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

        if self.prompt_name is not None:  # None
            prompt_depth = batch[self.prompt_name].to(self.device, self.dtype)  # sparse_pointmap
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
            target = batch[self.target_name].to(device=self.device)  # pointmap
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

    def get_finetune_view_loss(
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

    def get_reconstruct_loss(
        self,
        name,
        image,
        depth,
        render_rgb,
        render_depth,
        valid_mask=None,
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
        # rgb = (image.float() + 1) * 0.5
        rgb = image.float()
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

        # depth_loss = 0
        # if render_depth is not None:
        #     b,n,c,h,w = depth.shape
        #     for i in range(n):
        #         render_depth_i = render_depth[:,i].unsqueeze(1).float().permute(0, 2, 3, 1).contiguous()
        #         depth_i = depth[:,i].float().permute(0, 2, 3, 1).contiguous()
        #         valid_mask_i = valid_mask[:,i].squeeze(1)
        #         depth_loss += self.depth_loss(render_depth_i[...,-1], depth_i[...,-1], valid_mask_i, name=name)
        #     loss += depth_loss
        #     loss_dict["rc_depth_loss"] = depth_loss

        return loss, loss_dict

    def split_batch_novel(self, batch):
        # ['sparse_pointmap_mask', 'sparse_pointmap', 'meta_data', 'pointmap', 'normal', 'pointmap_reff', 'sem_label', 'depth', 'depth_raw_mask', 'depth_mask', 'sparse_pointmap_max_range', 'image', 'intrinsics', 'depth_raw', 'image_show', 'extrinsics_reff', 'extrinsics', 'generator']
        if "extra_ids" not in batch["meta_data"]:
            return batch, None

        # import pdb;pdb.set_trace()
        extra_ids = batch["meta_data"]["extra_ids"]

        if len(extra_ids) == 0:
            return batch, None

        if extra_ids.numel() == 0:
            return batch, None

        batch_novel = {}
        b, num_views = batch["image"].shape[:2]  # [1, 8]

        batch_indices = torch.arange(b)[:, None]
        remaining_mask = torch.ones((b, num_views), dtype=torch.bool)
        # import pdb;pdb.set_trace()
        remaining_mask[batch_indices, extra_ids] = False

        for k, v in batch.items():
            if type(v) != torch.Tensor:
                # batch_novel[k] = v
                continue
            if len(v.shape) < 2 or v.shape[1] != num_views:
                # batch_novel[k] = v
                continue
            remaining = v[remaining_mask].reshape(b, -1, *v.shape[2:])
            batch[k] = remaining  # 重新赋值 推理视角
            # if k in ["intrinsics", "extrinsics", "mv_extrinsics"]:   # novel views only need these data
            selected = v[batch_indices, extra_ids]
            batch_novel[k] = selected

        if batch["meta_data"]["frames"] > 1:
            batch["meta_data"]["frames"] -= extra_ids.shape[1]

        if batch["meta_data"]["views"] > 1:
            batch["meta_data"]["views"] -= extra_ids.shape[1]

        return batch, batch_novel

    def split_batch(self, batch):
        # ['sparse_pointmap_mask', 'sparse_pointmap'(输入的稀疏点云), 'meta_data', 'pointmap'(稠密点云用于监督生成的点云和深度图), 'normal', 'pointmap_reff'(对齐到第一视角坐标系的点云), 'sem_label', 'depth', 'depth_raw_mask', 'depth_mask', 'sparse_pointmap_max_range', 'image', 'intrinsics', 'depth_raw', 'image_show', 'extrinsics_reff', 'extrinsics', 'generator']
        if "extra_ids" not in batch["meta_data"]:
            return batch, None
        # import pdb;pdb.set_trace()
        extra_ids = batch["meta_data"]["extra_ids"]  # [15,14,13,12] # View id的排序是倒序的

        if len(extra_ids) == 0:
            return batch, None

        batch_novel = {}
        batch_infer = {}
        batch_origin = {}
        b, num_views = batch["image"].shape[:2]  # [1, 16]

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
            batch[k] = remaining  # 重新赋值 待微调训练的视角 12个

            selected = v[batch_indices, extra_ids]
            batch_novel[k] = selected

            infer_selected = v[batch_indices, torch.tensor([5, 7, 9, 11])]
            batch_infer[k] = infer_selected

            batch_origin[k] = v

        if batch["meta_data"]["frames"] > 1:
            batch["meta_data"]["frames"] -= extra_ids.shape[1]

        if batch["meta_data"]["views"] > 1:
            batch["meta_data"]["views"] -= extra_ids.shape[1]

        return batch, batch_novel, batch_infer, batch_origin  # batch_novel表示最后测试的视角

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

    def train_step(self, batch, iterations):
        """
        Executes a single training step using a batch of data.

        Parameters:
        - batch (dict): A dictionary containing the batch of data, including images, depth information, and masks.
        - optimizer: for prune and clone gaussains
        Returns:
        - loss (Tensor): The total loss for this training step.
        - loss_dict (dict): A dictionary containing individual loss components.
        """
        # Extract metadata from the batch
        meta_data = batch["meta_data"]

        # If the batch does not contain 'frames' or 'views', defer to the superclass implementation
        if "frames" not in meta_data and "views" not in meta_data:
            return super().train_step(batch)

        # batch, batch_novel, batch_infer, batch_origin = self.split_batch(batch)
        batch, batch_novel = self.split_batch_novel(batch)

        self.train()

        with torch.no_grad():
            # Unpack necessary inputs from the batch
            if self.target_name and self.target_mask_name is not None:
                target_pointmap = batch[self.target_name].to(device=self.device)
                valid_mask = batch[self.target_mask_name].to(self.device)
            else:
                target_pointmap = None
                valid_mask = None

            name = batch["meta_data"]["name"][0]
            image = batch["image"].to(device=self.device, dtype=self.dtype)

            intrinsics = batch.get("intrinsics", None)
            extrinsics = batch.get(self.mv_extrinsics_name, None)  # [n, 4, 4]

            if intrinsics is not None:
                intrinsics = intrinsics.to(device=self.device)
            if extrinsics is not None:
                extrinsics = extrinsics.to(device=self.device)

        if self.mv_extrinsics_name is not None:
            extrinsics = batch[self.mv_extrinsics_name].to(device=self.device)
        else:
            extrinsics = None

        # novel_extrinsics = None
        # novel_intrinsics = None
        # if batch_novel is not None:
        #     novel_extrinsics = batch_novel[self.mv_extrinsics_name].to(device=self.device).clone()
        #     # novel_extrinsics = batch_novel["extrinsics"].to(device=self.device).clone()
        #     novel_intrinsics = batch_novel["intrinsics"].to(device=self.device).clone()

        results = self.model(
            image=image,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            near=self.gs_near,
            far=self.gs_far,
        )

        # Extract predictions from the model output
        render_rgb = results.get("render_rgb", None)
        render_depth = results.get("render_depth", None)

        total_loss, total_loss_dict = 0, dict()
        # Compute reconstruction loss

        loss_rc, loss_dict_rc = self.get_reconstruct_loss(
            name=name,
            image=image,
            depth=target_pointmap,  #  Target
            render_rgb=render_rgb,
            render_depth=render_depth,
            valid_mask=valid_mask,
        )
        total_loss += loss_rc
        total_loss_dict.update(loss_dict_rc)

        # scaling_reg_loss 视角culling后的scale算loss
        # gaussians = results.get("gaussians", None)
        # scaling_reg = 0.005*gaussians.get_scaling().prod(dim=1).mean()

        # total_loss += scaling_reg
        # loss_dict_rc["rc_scaling_loss"] = scaling_reg
        # total_loss_dict.update(loss_dict_rc)

        # 增加normal loss和multi_view loss
        if iterations > 3000:
            normal = results.get("normal", None)
            depth_normal = results.get("depth_normal", None)
            if normal is not None:
                image_weight = 1.0 - get_img_grad_weight(image[0, 0])
                image_weight = (image_weight).clamp(0, 1).detach() ** 2
                normal_loss = (
                    0.015
                    * (image_weight * (((depth_normal[0, 0] - normal[0, 0])).abs().sum(0))).mean()
                )
                total_loss += normal_loss
                loss_dict_rc["rc_normal_loss"] = normal_loss
                total_loss_dict.update(loss_dict_rc)

        # 增加多视图一致性监督 TODO 需要该datasetbatch 额外增加一个camera ref

        return total_loss, total_loss_dict, results

    def postprocess(
        self,
        intrinsics,
        extrinsics,
    ):
        if intrinsics is not None:
            intrinsics = intrinsics.cpu().squeeze(0).numpy()

        if extrinsics is not None:
            extrinsics = extrinsics.cpu().squeeze(0).numpy()

        return self.output_type(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

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

        meta_data = batch["meta_data"]  # 1,1,c,h,w

        # If "frames" or "views" are not in metadata, fallback to the superclass infer method
        if "frames" not in meta_data and "views" not in meta_data:
            return super().infer(**batch)

        # batch, batch_novel, batch_infer, batch_origin = self.split_batch(batch)  # batch_novel [0,1,2,3]
        batch, batch_novel = self.split_batch_novel(batch)

        # Set the model to evaluation mode.
        self.eval()
        image = batch["image"].to(device=self.device, dtype=self.dtype)
        # breakpoint()

        # import numpy as np
        # show = (image[0, 0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        # import cv2
        # cv2.imwrite("debug.jpg", show)

        intrinsics = batch.get("intrinsics", None)
        extrinsics = batch.get(self.mv_extrinsics_name, None)  # [n, 4, 4]
        # extrinsics = batch.get("extrinsics", None)

        if intrinsics is not None:
            intrinsics = intrinsics.to(device=self.device)
        if extrinsics is not None:
            extrinsics = extrinsics.to(device=self.device)

        # Perform inference using the shared step method of the model
        novel_intrinsics = None
        novel_extrinsics = None
        metric_mask = None
        if batch_novel is not None:
            novel_intrinsics = batch_novel["intrinsics"].to(device=self.device).clone()
            novel_extrinsics = batch_novel[self.mv_extrinsics_name].to(device=self.device).clone()

            metric_mask = calculate_loss_mask(batch, batch_novel)
            metric_mask = metric_mask.unsqueeze(2).expand_as(batch_novel["image"])  # 1,16,1,h,w

        # gaussian model to render
        results = self.model(
            image=image,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            novel_extrinsics=novel_extrinsics,
            novel_intrinsics=novel_intrinsics,
            near=meta_data.get("near", self.gs_near),
            far=meta_data.get("far", self.gs_far),
        )

        gaussians = results.get("gaussians", None)  # Gaussian representation of the scene
        render_rgb = results.get("render_rgb", None)  # Rendered RGB image
        render_depth = results.get("render_depth", None)  # Rendered depth map
        render_normal = results.get("normal", None)

        video = None
        if gaussians is not None:
            # render video
            video = self.render_video_interpolation(
                gaussians, extrinsics, intrinsics, image.shape[-2], image.shape[-1]
            )

        novel_render_rgb = results.get("render_novel_rgb", None)
        novel_render_depth = results.get("render_novel_depth", None)

        # Extract frame and view counts from metadata
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        # Process each frame and view combination
        outputs_list = []
        for fi in range(frame_num):
            for vi in range(view_num):
                # Compute the global index for the current frame-view pair
                index = fi * view_num + vi
                # breakpoint()
                single = self.postprocess(
                    intrinsics=intrinsics[:, index] if intrinsics is not None else None,
                    extrinsics=extrinsics[:, index] if extrinsics is not None else None,
                )
                # Add Gaussian representation if available
                # Gaussian is shared across views, and maybe across frames in the future
                single.novel_render_depth = None
                single.novel_render_rgb = None
                single.video = None
                if (
                    gaussians is not None and vi == 0 and fi == 0
                ):  # 当且仅当第一个视角时才保存高斯点云和新视角渲染
                    single.gaussians = gaussians[0].export_ply()  #  只保留一次高斯点云
                    single.gs_xyz = gaussians[0].get_xyz().detach().cpu().numpy()
                    if novel_render_depth is not None:
                        single.novel_render_depth = (
                            novel_render_depth[0].detach().cpu().numpy()
                        )  # [N, H, W]
                    if novel_render_rgb is not None:
                        single.novel_render_rgb = (
                            novel_render_rgb[0].detach().cpu().numpy().transpose(0, 2, 3, 1)
                        )  # [N, H, W, 3]
                    if metric_mask is not None:
                        single.novel_mask = (
                            metric_mask[0].detach().cpu().numpy()
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

                if render_normal is not None:
                    single.render_normal = render_normal[0, index].detach().cpu().numpy()
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
        if output_prob["render_depth"] is not None:
            depth_color = depth_map(output_prob["render_depth"][0].detach())
            rgb = output_prob["render_rgb"][0]

            video = torch.cat([rgb, depth_color], dim=2).permute(0, 2, 3, 1)

            video = rgb.permute(0, 2, 3, 1)
            # video = torch.flip(video, dims=[1])
        else:
            rgb = output_prob["render_rgb"][0]
            video = rgb.permute(0, 2, 3, 1)
            # video = torch.flip(video, dims=[1])
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
            gaussians, trajectory_fn, h, w, n_interp=48, loop_reverse=loop_reverse
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
            # breakpoint()
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
            outputs_list[0].gaussians.write(
                save_path
            )  # 随便取第0个视角的高斯，其实就是所有视角的高斯

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
