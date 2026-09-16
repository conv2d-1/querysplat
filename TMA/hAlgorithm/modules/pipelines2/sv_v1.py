import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.modules.pipelines2.base import Pipeline
from hAlgorithm.modules.pipelines2.utils.outputs import ReconstructOutput
from hAlgorithm.modules.pipelines2.utils.save_outputs import save_sv_outputs
from hAlgorithm.modules.pipelines2.utils.visualize_mv import (
    vis_extra_local_results,
    vis_images,
    vis_local_results,
)
from hAlgorithm.modules.utils.ray import get_rays_in_camera_frame, get_rays_in_world_frame, recover_pinhole_intrinsics_from_ray_directions
from hAlgorithm.utils import instantiate_from_config


class SVPipeline(Pipeline):
    """Pipeline for Single-View Feed-forward Model."""

    def __init__(
        self,
        # inputs name
        intrinsics_name=None,
        extrinsics_name=None,
        scale_name=None,
        prompt_depth_name=None,
        target_local_depth_name=None,
        target_depth_mask_name=None,
        target_normal_name=None,
        target_normal_mask_name=None,
        target_invalid_mask_name=None,
        align_name=None,
        # extra inputs
        with_ray_directions=False,
        with_points_normal=False,
        # opensource model
        moge_cfg=None,
        segformer_cfg=None,
        # local depth loss
        local_depth_l1_loss=None,
        local_depth_grad_loss=None,
        local_depth_grad_extra_loss=None,
        local_depth_normal_loss=None,
        local_depth_conf_loss=None,
        train_pts_with_pred_intrinsics=False,
        # local other loss
        local_normal_loss=None,
        local_ray_directions_loss=None,
        local_invalid_mask_loss=None,
        local_motion_mask_loss=None,
        points_from_ray=False,
        save_output_cfg=None,
        **kwargs,
    ):
        super(SVPipeline, self).__init__(**kwargs)

        # inputs name
        self.intrinsics_name = intrinsics_name
        self.extrinsics_name = extrinsics_name
        self.scale_name = scale_name
        self.prompt_depth_name = prompt_depth_name
        self.target_local_depth_name = target_local_depth_name
        self.target_depth_mask_name = target_depth_mask_name
        self.target_normal_name = target_normal_name
        self.target_normal_mask_name = target_normal_mask_name
        self.target_invalid_mask_name = target_invalid_mask_name
        self.align_name = align_name

        # extra inputs
        self.with_ray_directions = with_ray_directions
        self.with_points_normal = with_points_normal

        # opensource model
        self.moge_cfg = moge_cfg
        self.moge = None

        self.segformer_cfg = segformer_cfg
        self.segformer = None

        # local depth loss
        self.local_depth_l1_loss = instantiate_from_config(local_depth_l1_loss)
        self.local_depth_grad_loss = instantiate_from_config(local_depth_grad_loss)
        self.local_depth_normal_loss = instantiate_from_config(local_depth_normal_loss)
        self.local_depth_conf_loss = instantiate_from_config(local_depth_conf_loss)

        self.local_depth_grad_extra_loss = instantiate_from_config(local_depth_grad_extra_loss)

        self.train_pts_with_pred_intrinsics = train_pts_with_pred_intrinsics

        # local other loss
        self.local_normal_loss = instantiate_from_config(local_normal_loss)
        self.local_ray_directions_loss = instantiate_from_config(local_ray_directions_loss)
        self.local_invalid_mask_loss = instantiate_from_config(local_invalid_mask_loss)
        self.local_motion_mask_loss = instantiate_from_config(local_motion_mask_loss)

        self.points_from_ray = points_from_ray

        self.save_output_cfg = dict(
            save_everything=False,
            save_output_conf=True,
            save_local2glb_results=False,
            save_local_results=True,
            save_extra_local_results=True,
            save_filtered_results=True,
            save_normal=True,
            save_normal_vis=False,
            save_output_only_local_glb=False,
            save_name_match_rgb=True,
            output_normalize_cameras=True,
            output_match_input_res=True,
            output_conf_ratio=0.2,
            output_intrinsics_from_ray=False,
            output_colmap_format=False,
            gt_out_dir="gt",
            save_invalid_mask=False,
        )
        if save_output_cfg is not None:
            self.save_output_cfg.update(save_output_cfg)

        if self.save_output_cfg["save_everything"]:
            for key in self.save_output_cfg:
                if isinstance(self.save_output_cfg[key], bool) and key.startswith("save_"):
                    if key in ["save_output_only_local_glb"]:
                        self.save_output_cfg[key] = False
                    else:
                        self.save_output_cfg[key] = True

    def build_opensource_model(self):
        if self.moge_cfg is not None and self.moge is None:
            from hAlgorithm.modules.models2.external.moge.model import import_model_class_by_version

            pretrained_model_name_or_path = "/mnt/netdata/Team/AI/weights/huggingface/Ruicheng/moge-2-vitl-normal/model.pt"
            self.moge = import_model_class_by_version("v2").from_pretrained(pretrained_model_name_or_path)
            del self.moge.points_head  # NOTE
            del self.moge.scale_head
            del self.moge.mask_head
            self.moge = self.moge.to(self.device).eval()
            self.moge.half()

        if self.segformer_cfg is not None and self.segformer is None:
            from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

            segformer_name = "/mnt/netdata/Team/AI/weights/segformer-b5-finetuned-ade-640-640"
            self.mask_feature_extractor = SegformerFeatureExtractor.from_pretrained(segformer_name)
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(segformer_name)
            self.segformer = self.segformer.to(self.device).eval()

            self.segformer_size = (self.mask_feature_extractor.size["height"], self.mask_feature_extractor.size["width"])
            self.segformer_image_mean = torch.tensor(self.mask_feature_extractor.image_mean).to(self.device, self.dtype)[None, :, None, None]
            self.segformer_image_std = torch.tensor(self.mask_feature_extractor.image_std).to(self.device, self.dtype)[None, :, None, None]

    def get_inputs(self, batch):
        meta_data = batch["meta_data"]
        name = meta_data["name"][0]
        total_iter = batch.get("total_iter")

        if total_iter is not None:
            meta_data["total_iter"] = total_iter

        image_numpy = batch["image"].cpu().permute(0, 2, 3, 1).contiguous().numpy()
        image = batch["image"].to(device=self.device, dtype=self.dtype)

        scale = intrinsics = ray_directions = extrinsics = ray_world = prompt_depth = None

        if self.scale_name is not None and self.scale_name in batch:
            scale = batch[self.scale_name].to(self.device, self.dtype)[..., None, None, None]

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
                ray_directions = ray_directions.view(intrinsics.shape[0], 3, h, w)

        if self.extrinsics_name is not None and self.extrinsics_name in batch:
            extrinsics = batch[self.extrinsics_name].to(device=self.device)
            extrinsics[:, :3, 3] = self.normalize(extrinsics[:, :3, 3], scale[..., 0, 0])

        if self.prompt_depth_name is not None and self.prompt_depth_name in batch:
            prompt_depth = batch[self.prompt_depth_name].to(self.device, self.dtype)
            prompt_depth = self.normalize(prompt_depth, scale)

        target_local_depth = target_global_points = target_depth_mask = None

        if self.target_local_depth_name is not None and self.target_local_depth_name in batch:
            target_local_depth = batch[self.target_local_depth_name].to(device=self.device)
            target_local_depth = self.normalize(target_local_depth, scale)

        if self.target_depth_mask_name is not None and self.target_depth_mask_name in batch:
            target_depth_mask = batch[self.target_depth_mask_name].to(self.device)

        target_normal = target_normal_mask = target_motion_mask = target_invalid_mask = None

        if self.target_normal_name is not None and self.target_normal_name in batch:
            target_normal = batch[self.target_normal_name].to(self.device)

        if self.target_normal_mask_name is not None and self.target_normal_mask_name in batch:
            target_normal_mask = batch[self.target_normal_mask_name].to(self.device)

        if self.target_invalid_mask_name is not None and self.target_invalid_mask_name in batch:
            target_invalid_mask = batch[self.target_invalid_mask_name].to(self.device)

        image_show = align_data = None

        if not self.training:
            if "image_show" in batch:
                image_show = batch["image_show"]
                image_show = image_show.float().numpy()

            if self.save_output_cfg["output_match_input_res"] and self.align_name in batch:
                align_data = batch[self.align_name]

        with torch.no_grad():
            if self.training:
                self.build_opensource_model()
            
                if self.moge is not None and (target_normal is None):
                    output = self.moge.infer((image + 1) * 0.5)
                    if target_normal is None:
                        target_normal = output["normal"].float().permute(0, 3, 1, 2).contiguous()
                    # # if target_local_depth is None:
                    # target_local_depth = output['depth'].float()
                    # # Create a mask of ones
                    # sparse_nums = self.moge_cfg.get("sparse_nums", 0)
                    # if isinstance(sparse_nums, (list, tuple)):
                    #     sparse_nums = int(np.random.uniform(min(sparse_nums), max(sparse_nums)))
                    # if sparse_nums > 0:
                    #     sparsification_mask = torch.ones_like(target_local_depth, device=self.device)
                    #     # Create a mask for valid pixels (depth > 0)
                    #     valid_pixel_mask = target_local_depth > 0
                    #     # Calculate the number of valid pixels
                    #     num_valid_pixels = valid_pixel_mask.sum().item()
                    #     # Calculate the number of valid pixels to set to zero
                    #     num_to_zero = int(num_valid_pixels.sum() - sparse_nums)
                    #     if num_to_zero > 0:
                    #         # Get the indices of valid pixels
                    #         valid_indices = valid_pixel_mask.nonzero(as_tuple=True)
                    #         # Randomly select indices to zero out
                    #         indices_to_zero = torch.randperm(num_valid_pixels)[:num_to_zero]
                    #         # Set selected valid indices to zero in the mask
                    #         sparsification_mask[
                    #             valid_indices[0][indices_to_zero],
                    #             valid_indices[1][indices_to_zero],
                    #             valid_indices[2][indices_to_zero],
                    #             valid_indices[3][indices_to_zero],
                    #         ] = 0
                    #     # Apply the mask on the depth
                    #     prompt_depth = target_local_depth * sparsification_mask

                    # if prompt_depth is not None:
                    #     scale = prompt_depth.max()
                    #     prompt_depth = self.normalize(prompt_depth, scale)
                    # else:
                    #     scale = target_local_depth.max()

                    # target_local_depth = self.normalize(prompt_depth, scale)

                if self.segformer is not None and target_invalid_mask is None:
                    with torch.autocast("cuda", enabled=True, dtype=self.dtype):
                        pixel_values = (image + 1) * 0.5
                        pixel_values = F.interpolate(pixel_values, size=self.segformer_size, mode="bilinear", align_corners=False)
                        pixel_values = (pixel_values - self.segformer_image_mean) / self.segformer_image_std
                        pixel_values = pixel_values.to(self.dtype)

                        logits = self.segformer(pixel_values=pixel_values).logits.float()

                        upsampled_logits = torch.nn.functional.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)  # (H, W)
                        pred_seg = upsampled_logits.argmax(dim=1)  # (H, W)
                        target_invalid_mask = pred_seg == self.segformer_cfg["index"]

                        if "threshold" in self.segformer_cfg:
                            score = upsampled_logits.softmax(dim=1)
                            target_invalid_mask = target_invalid_mask & (score[:, self.segformer_cfg["index"]] >= self.segformer_cfg["threshold"])

                        target_invalid_mask = target_invalid_mask.unsqueeze(1)
                        target_depth_mask = target_depth_mask & (~target_invalid_mask)

                        if target_invalid_mask.sum() > 0 and 0:
                            for bi in range(image.shape[0]):
                                cur_rgb = ((image_numpy[bi] + 1) * 0.5 * 255).astype(np.uint8)
                                cv2.imwrite("rgb.png", cv2.cvtColor(cur_rgb, cv2.COLOR_RGB2BGR))
                                cur_mask = (target_invalid_mask[bi, 0].cpu().long().numpy() * 255).astype(np.uint8)
                                cv2.imwrite("mask.png", cur_mask)
                                breakpoint()

        return (
            name,
            total_iter,
            meta_data,
            image,
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
        )

    def add_loss(self, total_loss, total_loss_dict, loss, loss_dict=None, task_name=None, loss_name=None, prefix=""):
        if isinstance(loss, dict):
            total_loss += sum(loss.values())
            total_loss_dict.update({prefix + k: v for k, v in loss.items()})
        else:
            total_loss += loss
            total_loss_dict[prefix + loss_name] = loss
        return total_loss, total_loss_dict

    def get_base_loss(
        self,
        name,
        pred_depth=None,
        pred_conf=None,
        pred_normal=None,
        pred_ray_directions=None,
        pred_invalid_mask=None,
        target_depth=None,
        target_normal=None,
        target_normal_mask=None,
        target_ray_directions=None,
        target_motion_mask=None,
        target_invalid_mask=None,
        valid_mask=None,
        scale=None,
        **kwargs,
    ):
        if pred_depth is not None:
            if pred_depth.shape[1] == 1:
                pred_depth = pred_depth.float().squeeze(1).unsqueeze(-1)
            else:
                pred_depth = pred_depth.float().permute(0, 2, 3, 1).contiguous()
        if pred_conf is not None:
            pred_conf = pred_conf.float().squeeze(1).unsqueeze(-1)
        if pred_normal is not None:
            pred_normal = pred_normal.float().permute(0, 2, 3, 1).contiguous()
        if pred_ray_directions is not None:
            pred_ray_directions = pred_ray_directions.float().permute(0, 2, 3, 1).contiguous()
        if pred_invalid_mask is not None:
            pred_invalid_mask = pred_invalid_mask.squeeze(1)

        if target_depth is not None:
            if target_depth.shape[1] == 1:
                target_depth = target_depth.float().squeeze(1).unsqueeze(-1)
            else:
                target_depth = target_depth.float().permute(0, 2, 3, 1).contiguous()
        if target_normal is not None:
            target_normal = target_normal.float().permute(0, 2, 3, 1).contiguous()
        if target_normal_mask is not None:
            target_normal_mask = target_normal_mask.squeeze(1)
        if target_ray_directions is not None:
            target_ray_directions = target_ray_directions.float().permute(0, 2, 3, 1).contiguous()
        if target_motion_mask is not None:
            target_motion_mask = target_motion_mask.squeeze(1)
        if target_invalid_mask is not None:
            target_invalid_mask = target_invalid_mask.squeeze(1)
        if valid_mask is not None:
            valid_mask = valid_mask.squeeze(1)

        total_loss = 0
        total_loss_dict = dict()

        # Base depth l1 loss
        if self.local_depth_l1_loss is not None and target_depth is not None and pred_depth is not None and pred_depth.requires_grad:
            loss = self.local_depth_l1_loss(
                name=name,
                pred_depth=pred_depth,
                pred_conf=pred_conf,
                target_depth=target_depth,
                valid_mask=valid_mask,
            )
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="l1_loss")

        if self.local_depth_conf_loss is not None and target_depth is not None and pred_depth is not None and pred_conf is not None and pred_conf.requires_grad:
            loss = self.local_depth_conf_loss(
                name=name,
                pred_depth=pred_depth,
                pred_conf=pred_conf,
                target_depth=target_depth,
                valid_mask=valid_mask,
                scale=scale,
            )
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="conf_loss")

        # Depth gradient loss
        if self.local_depth_grad_loss is not None and target_depth is not None and pred_depth is not None and pred_depth.requires_grad:
            loss = self.local_depth_grad_loss(
                name=name,
                pred_depth=pred_depth[..., -1],
                target_depth=target_depth[..., -1],
                valid_mask=valid_mask,
            )
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="grad_loss")

        # Depth gradient loss
        if self.local_depth_grad_extra_loss is not None and target_depth is not None and pred_depth is not None and pred_depth.requires_grad:
            loss = self.local_depth_grad_extra_loss(
                name=name,
                pred_depth=pred_depth[..., -1],
                target_depth=target_depth[..., -1],
                valid_mask=valid_mask,
            )
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="grad_extra_loss")

        # Depth normal loss
        if self.local_depth_normal_loss is not None and pred_depth is not None and pred_depth.requires_grad:
            loss = self.local_depth_normal_loss(name=name, pred_depth=pred_depth, target_depth=target_depth, target_normal=target_normal, valid_mask=valid_mask)
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="normal_loss")

        # Local pred normal loss
        if self.local_normal_loss is not None and pred_normal is not None and pred_normal.requires_grad:
            if target_normal_mask is not None:
                loss = self.local_normal_loss(name=name, pred_normal=pred_normal, target_depth=target_depth, target_normal=target_normal, valid_mask=target_normal_mask)
            else:
                loss = self.local_normal_loss(name=name, pred_normal=pred_normal, target_depth=target_depth, target_normal=target_normal, valid_mask=valid_mask)
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="pd_normal_loss")

        if self.local_ray_directions_loss is not None and target_ray_directions is not None and pred_ray_directions is not None and pred_ray_directions.requires_grad:
            loss = self.local_ray_directions_loss(name=name, pred_ray_directions=pred_ray_directions, target_ray_directions=target_ray_directions)
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="pd_ray_loss")

        # Local pred invalid_mask loss
        if self.local_invalid_mask_loss is not None and target_invalid_mask is not None and pred_invalid_mask is not None and pred_invalid_mask.requires_grad:
            loss = self.local_invalid_mask_loss(name=name, pred_invalid_mask=pred_invalid_mask, target_invalid_mask=target_invalid_mask)
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="pd_inv_mask_loss")

        return total_loss, total_loss_dict

    def train_step(self, batch):
        self.train()

        (
            name,
            total_iter,
            meta_data,
            image,
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
        ) = self.get_inputs(batch)

        results = self.model(
            image,
            scale=scale,
            prompt_depth=prompt_depth,
            intrinsics=intrinsics,
            ray_directions=ray_directions,
            meta_data=meta_data,
        )

        # Extract predictions from the model outputs
        if "depth" in results:
            pred_local_depth = results["depth"]
        elif "pointmap" in results:
            pred_local_depth = results["pointmap"]
        elif "points" in results:
            pred_local_depth = results["points"]
        else:
            pred_local_depth = None

        pred_local_conf = results.get("confidence")
        pred_local_normal = results.get("normal")
        pred_local_invalid_mask = results.get("invalid_mask")

        pred_ray_directions = results.get("ray")

        total_loss, total_loss_dict = 0, dict()

        # Normalize the ray directions to unit vectors
        if pred_ray_directions is not None:
            pred_ray_directions = pred_ray_directions.pred_ray_directions.norm(dim=2, keepdim=True).clip(min=1e-8)

        # DepthMap trans to PointMap
        if pred_local_depth is not None and pred_local_depth.shape[-3] == 1:
            if self.points_from_ray:
                pred_local_depth = pred_local_depth * pred_ray_directions
            else:
                pred_local_depth = self.depth_to_points(pred_local_depth, K=intrinsics)

        # Normalize the normal maps
        if pred_local_normal is not None:
            tgt_h, tgt_w = pred_local_depth.shape[-2:] if pred_local_depth is not None else image.shape[-2:]
            norm_h, norm_w = pred_local_normal.shape[-2:]
            if tgt_h != norm_h or tgt_w != norm_w:
                pred_local_normal = F.interpolate(pred_local_normal, (tgt_h, tgt_w), mode="bilinear", align_corners=False, antialias=False)
            pred_local_normal = F.normalize(pred_local_normal, dim=-3)

        total_loss, total_loss_dict = self.get_base_loss(
            name=name,
            image=image,
            pred_depth=pred_local_depth,
            pred_conf=pred_local_conf,
            pred_normal=pred_local_normal,
            pred_ray_directions=pred_ray_directions,
            pred_invalid_mask=pred_local_invalid_mask,
            target_depth=target_local_depth,
            target_ray_directions=ray_directions,
            target_normal=target_normal,
            target_normal_mask=target_normal_mask,
            target_invalid_mask=target_invalid_mask,
            valid_mask=target_depth_mask,
            scale=scale,
        )

        if scale is not None:
            total_loss_dict["scale"] = round(scale.max().cpu().item(), 2)

        if image is not None:
            aspect_ratio = image.shape[-1] / image.shape[-2]
            total_loss_dict["aspect_ratio"] = round(aspect_ratio, 2)
            total_loss_dict["bs"] = int(image.shape[0])

        return total_loss, total_loss_dict

    def postprocess(
        self,
        pred_local_points=None,
        pred_local_conf=None,
        pred_local_normal=None,
        pred_local_invalid_mask=None,
        pred_intrinsics=None,
        image=None,
        image_show=None,
        scale=None,
        prompt_depth=None,
        target_local_depth=None,
        target_global_points=None,
        target_depth_mask=None,
        target_normal=None,
        target_normal_mask=None,
        intrinsics=None,
        extrinsics=None,
        align_data=None,
    ):
        # Convert the color pointmap to a NumPy array and normalize it to the [0, 255] range.
        if image_show is not None and pred_local_points is not None:
            points_colors = image_show[0].transpose(1, 2, 0)
        else:
            points_colors = image[0].cpu().float().numpy().transpose(1, 2, 0)
            points_colors = (points_colors + 1) * 0.5 * 255

        if pred_local_points is not None:
            points_h, points_w = pred_local_points.shape[-2:]

            if points_h != points_colors.shape[0] or points_w != points_colors.shape[1]:
                points_colors = cv2.resize(points_colors, dsize=(points_w, points_h), interpolation=cv2.INTER_LINEAR)
            points_colors = points_colors.reshape(-1, 3)

            pred_local_points = pred_local_points[0].cpu().float().numpy()
            pred_local_depth = pred_local_points[-1].clip(1e-3)
            pred_local_points = pred_local_points.reshape(3, -1).transpose(1, 0)

        else:
            pred_local_depth = None
            points_h, points_w = points_colors.shape[:2]
            points_colors = points_colors.reshape(-1, 3)

        if pred_local_points is not None and pred_local_conf is not None:
            pred_local_conf = pred_local_conf[0, 0].cpu().float().numpy()
            conf_thresh = np.percentile(pred_local_conf.reshape(-1), self.save_output_cfg["output_conf_ratio"] * 100)
            filtered_pred_local_points = pred_local_points[pred_local_conf.reshape(-1) > conf_thresh]
            filtered_points_colors = points_colors[pred_local_conf.reshape(-1) > conf_thresh]
        else:
            filtered_pred_local_points = filtered_points_colors = None

        if self.save_output_cfg["output_match_input_res"] and pred_local_depth is not None and align_data is not None:
            align_data = align_data[0].numpy()
            h, w = align_data.shape[:2]
            pred_local_depth = cv2.resize(pred_local_depth, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

            if pred_local_conf is not None:
                pred_local_conf = cv2.resize(pred_local_conf, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

        if pred_local_normal is not None:
            pred_local_normal = pred_local_normal[0].cpu().float().numpy().transpose(1, 2, 0)

        if pred_local_invalid_mask is not None:
            pred_local_invalid_mask = pred_local_invalid_mask[0, 0].cpu().float().numpy()

        # Add predicted intrinsics if available
        if pred_intrinsics is not None:
            pred_intrinsics = pred_intrinsics[0].detach().cpu().float().numpy()

        if pred_local_points is not None and extrinsics is not None:
            local2glb_points = (
                np.linalg.inv(extrinsics.cpu().numpy())
                @ np.concatenate(
                    [pred_local_points, np.ones([pred_local_points.shape[0], 1])],
                    axis=-1,
                    dtype=np.float32,
                ).T
            ).T[:, :3]
        else:
            local2glb_points = None

        if scale is not None:
            scale = float(scale.squeeze().cpu())

        if prompt_depth is not None:
            prompt_h, prompt_w = prompt_depth.shape[-2:]
            prompt_depth = prompt_depth[0].cpu().float().numpy().transpose(1, 2, 0).reshape(-1, 3)
        else:
            prompt_h = prompt_w = None

        if target_local_depth is not None:
            target_local_depth = target_local_depth[0].cpu().float().numpy().reshape(3, -1).transpose(1, 0)

        if target_depth_mask is not None:
            target_depth_mask = target_depth_mask[0].cpu().numpy()

        if target_normal is not None:
            target_normal = target_normal[0].cpu().numpy().transpose(1, 2, 0)

        if target_normal_mask is not None:
            target_normal_mask = target_normal_mask.squeeze().cpu().numpy()

        if intrinsics is not None:
            intrinsics = intrinsics[0].cpu().numpy()

        return ReconstructOutput(
            pointmap_color=points_colors,
            pointmap_h=points_h,
            pointmap_w=points_w,
            pointmap=pred_local_points,
            depth_align=pred_local_depth,
            confidence=pred_local_conf,
            local2glb_pointmap=local2glb_points,
            local2glb_confidence=pred_local_conf,
            filtered_pointmap=filtered_pred_local_points,
            filtered_pointmap_color=filtered_points_colors,
            intrinsics_pred=pred_intrinsics,
            prompt_scale=scale,
            prompt_pointmap=prompt_depth,
            prompt_h=prompt_h,
            prompt_w=prompt_w,
            pointmap_gt=target_local_depth,
            pointmap_gt_global=target_global_points,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            normal=pred_local_normal,
            normal_gt=target_normal,
            normal_mask=target_normal_mask,
            depth_mask=target_depth_mask,
            invalid_mask=pred_local_invalid_mask,
        )

    @torch.no_grad()
    def infer(self, **batch):
        self.eval()

        (
            name,
            total_iter,
            meta_data,
            image,
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
        ) = self.get_inputs(batch)

        if "use_amp" in batch and batch["use_amp"]:
            with torch.autocast("cuda", enabled=bool(batch["use_amp"]), dtype=batch["amp_dtype"]):
                results = self.model(
                    image,
                    scale=scale,
                    prompt_depth=prompt_depth,
                    intrinsics=intrinsics,
                    ray_directions=ray_directions,
                    meta_data=meta_data,
                )
        else:
            results = self.model(
                image,
                scale=scale,
                prompt_depth=prompt_depth,
                intrinsics=intrinsics,
                ray_directions=ray_directions,
                meta_data=meta_data,
            )

        # Extract predictions from the model outputs
        if "depth" in results:
            pred_local_depth = results["depth"]
        elif "pointmap" in results:
            pred_local_depth = results["pointmap"]
        elif "points" in results:
            pred_local_depth = results["points"]
        else:
            pred_local_depth = None

        pred_local_conf = results.get("confidence")
        pred_local_normal = results.get("normal")
        pred_local_invalid_mask = results.get("invalid_mask")

        pred_ray_directions = results.get("ray")
        # Normalize the ray directions to unit vectors
        if pred_ray_directions is not None:
            pred_ray_directions = pred_ray_directions.pred_ray_directions.norm(dim=2, keepdim=True).clip(min=1e-8)

            if self.save_output_cfg["output_intrinsics_from_ray"]:
                assert pred_ray_directions.ndim == 5
                pred_intrinsics = recover_pinhole_intrinsics_from_ray_directions(ray_directions=pred_ray_directions.view(-1, *pred_ray_directions.shape[2:]).permute(0, 2, 3, 1).contiguous())
                pred_intrinsics = pred_intrinsics.view(*pred_ray_directions.shape[:2], *pred_intrinsics.shape[1:])
        else:
            pred_intrinsics = None

        if prompt_depth is not None and scale is not None:
            prompt_depth = self.denormalize(prompt_depth, scale=scale)
        if target_local_depth is not None and scale is not None:
            target_local_depth = self.denormalize(target_local_depth, scale=scale)

        if pred_local_depth is not None:
            if pred_local_depth.shape[-3] == 1:
                if self.points_from_ray:
                    pred_local_depth = pred_ray_directions * pred_local_depth
                elif pred_intrinsics is not None:
                    pred_local_depth = self.depth_to_points(pred_local_depth, K=pred_intrinsics)
                else:
                    pred_local_depth = self.depth_to_points(pred_local_depth, K=intrinsics)

            if scale is not None:
                pred_local_depth = self.denormalize(pred_local_depth, scale=scale)

        # Normalize the normal maps
        if pred_local_normal is not None:
            tgt_h, tgt_w = pred_local_depth.shape[-2:] if pred_local_depth is not None else image.shape[-2:]
            norm_h, norm_w = pred_local_normal.shape[-2:]
            if tgt_h != norm_h or tgt_w != norm_w:
                pred_local_normal = F.interpolate(pred_local_normal, (tgt_h, tgt_w), mode="bilinear", align_corners=False, antialias=False)
            pred_local_normal = F.normalize(pred_local_normal, dim=-3)

        outputs = self.postprocess(
            pred_local_points=pred_local_depth,
            pred_local_conf=pred_local_conf,
            pred_local_normal=pred_local_normal,
            pred_local_invalid_mask=pred_local_invalid_mask,
            pred_intrinsics=pred_intrinsics,
            image=image,
            image_show=image_show,
            scale=scale,
            prompt_depth=prompt_depth,
            target_local_depth=target_local_depth,
            target_global_points=target_global_points,
            target_depth_mask=target_depth_mask,
            target_normal=target_normal,
            target_normal_mask=target_normal_mask,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            align_data=align_data,
        )

        outputs.rgb = (image[0].permute(1, 2, 0).cpu().numpy() + 1) * 0.5

        return outputs

    def get_out_dir(self, out_dir, data_idx=None):
        if data_idx is not None:
            out_dir = os.path.join(out_dir, f"{data_idx:06d}")
        return out_dir

    def get_gt_out_dir(self, out_dir):
        if self.save_output_cfg["gt_out_dir"] is None:
            return None
        abs_gt_out_dir = os.path.join(os.path.dirname(os.path.dirname(out_dir)), self.save_output_cfg["gt_out_dir"], os.path.basename(out_dir))
        return abs_gt_out_dir

    def visualize(self, outputs, meta_data, out_dir):
        data_idx = meta_data["data_idx"][0]

        gt_out_dir = self.get_gt_out_dir(out_dir)
        out_dir = self.get_out_dir(out_dir, data_idx=data_idx)

        # Images Visualization
        if gt_out_dir is not None:
            vis_images(cfg=self.save_output_cfg, mv_outputs=[outputs], gt_out_dir=gt_out_dir, data_idx=data_idx, frame_num=1, view_num=1)

        # Local results Visualization
        if self.save_output_cfg["save_local_results"]:
            os.makedirs(out_dir, exist_ok=True)
            vis_local_results(cfg=self.save_output_cfg, mv_outputs=[outputs], mv_out_dir=out_dir, frame_num=1, view_num=1, meta_data=meta_data)
        if self.save_output_cfg["save_extra_local_results"]:
            os.makedirs(out_dir, exist_ok=True)
            vis_extra_local_results(cfg=self.save_output_cfg, mv_outputs=[outputs], mv_out_dir=out_dir, frame_num=1, view_num=1, meta_data=meta_data)

    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        save_sv_outputs(
            cfg=self.save_output_cfg,
            mv_outputs=[outputs],
            meta_data=meta_data,
            out_dir=out_dir,
            output_meta_dict=output_meta_dict,
        )

    @torch.no_grad()
    def trans_onnx(
        self,
        image,
        prompt_depth=None,
        prompt_scale=None,
    ):
        h, w = image.shape[-2:]
        if not hasattr(self, "patch_h"):
            patch_h, patch_w = (
                h // self.model.rgb_encoder.patch_size,
                w // self.model.rgb_encoder.patch_size,
            )
        else:
            patch_h, patch_w = self.patch_h, self.patch_w

        if prompt_depth is not None and prompt_scale is not None:
            prompt_depth = self.normalize(prompt_depth, prompt_scale)

        if prompt_depth is not None and self.model.depth_encoder is not None:
            prompt_features = self.model.depth_encoder(prompt_depth)
        else:
            prompt_features = None

        self.model.rgb_encoder.normalize = True
        self.model.rgb_encoder.__class__.forward = self.model.rgb_encoder.onnx_forward
        rgb_features = self.model.rgb_encoder(image)
        
        self.model.depth_head.__class__.forward = self.model.depth_head.onnx_forward
        results = self.model.depth_head(rgb_features, prompt_depth=prompt_features, patch_h=patch_h, patch_w=patch_w)

        if "depth" in results:
            pred_local_depth = results["depth"]
        elif "pointmap" in results:
            pred_local_depth = results["pointmap"]
        elif "points" in results:
            pred_local_depth = results["points"]
        else:
            pred_local_depth = None
        pred_local_conf = results.get("confidence")
        pred_local_normal = results.get("normal")
        pred_local_invalid_mask = results.get("invalid_mask")
        pred_local_normal = F.normalize(pred_local_normal, dim=-3)

        if prompt_depth is not None and prompt_scale is not None:
            pred_local_depth = self.denormalize(pred_local_depth, scale=prompt_scale)

        return pred_local_depth, pred_local_conf, pred_local_normal, pred_local_invalid_mask
