import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from hAlgorithm.modules.pipelines2.base import Pipeline
from hAlgorithm.modules.pipelines2.utils.outputs import ReconstructOutput
from hAlgorithm.modules.pipelines2.utils.save_outputs import save_mv_outputs
from hAlgorithm.modules.pipelines2.utils.visualize_mv import (
    vis_camera_results,
    vis_extra_local_results,
    vis_glb_results,
    vis_images,
    vis_local_results,
    vis_render,
    vis_render_video,
    vis_track_results,
)
from hAlgorithm.modules.utils.ray import get_rays_in_camera_frame, get_rays_in_world_frame, recover_pinhole_intrinsics_from_ray_directions
from hAlgorithm.utils import instantiate_from_config


class MVFRPipeline(Pipeline):
    """Pipeline for Multi-View Feed-forward Reconstruction."""

    def __init__(
        self,
        # inputs name
        intrinsics_name=None,
        extrinsics_name=None,
        extrinsics_noise_name=None,
        prompt_extrinsics_name=None,
        scale_name=None,
        prompt_depth_name=None,
        prompt_depth_mask_name=None,
        target_local_depth_name=None,
        target_global_points_name=None,
        target_depth_mask_name=None,
        target_normal_name=None,
        target_normal_mask_name=None,
        target_motion_mask_name=None,
        target_invalid_mask_name=None,
        align_name=None,
        rgb_mask=None,
        # extra inputs
        prompt_depth_normalize=None,
        with_ray_directions=False,
        with_ray_in_world=False,
        with_points_normal=False,
        # local depth loss
        local_depth_l1_loss=None,
        local_depth_grad_loss=None,
        local_depth_normal_loss=None,
        local2global_loss=None,
        local2global_camera_normalize=False,
        local2global_camera_detach=False,
        # local other loss
        local_normal_loss=None,
        local_ray_directions_loss=None,
        local_ray_in_world_loss=None,
        local_invalid_mask_loss=None,
        local_motion_mask_loss=None,
        # global points loss
        global_points_l1_loss=None,
        global_points_grad_loss=None,
        global_points_normal_loss=None,
        # camera loss
        camera_loss=None,
        # reconstruction loss
        rc_rgb_l1_loss=None,
        rc_ssim_loss=None,
        rc_lpips_loss=None,
        rc_depth_loss=None,
        rc_normal_loss=None,
        rc_depth_consistency_loss=None,
        # track loss
        track_loss=None,
        task_weight=None,
        pose_encoding_type="absT_quaR_FoV",
        points_from_ray=False,
        save_output_cfg=None,
        # aux model
        sv_model_cfg=None,
        da3_model_cfg=None,
        segformer_cfg=None,
        # debug
        debug_rgb_path=False,
        **kwargs,
    ):
        super(MVFRPipeline, self).__init__(**kwargs)

        # inputs name
        self.intrinsics_name = intrinsics_name
        self.extrinsics_name = extrinsics_name
        self.extrinsics_noise_name = extrinsics_noise_name
        self.prompt_extrinsics_name = prompt_extrinsics_name
        self.scale_name = scale_name
        self.prompt_depth_name = prompt_depth_name
        self.prompt_depth_mask_name = prompt_depth_mask_name
        self.prompt_depth_normalize = prompt_depth_normalize
        self.target_local_depth_name = target_local_depth_name
        self.target_global_points_name = target_global_points_name
        self.target_depth_mask_name = target_depth_mask_name
        self.target_normal_name = target_normal_name
        self.target_normal_mask_name = target_normal_mask_name
        self.target_motion_mask_name = target_motion_mask_name
        self.target_invalid_mask_name = target_invalid_mask_name
        self.align_name = align_name
        self.rgb_mask = rgb_mask

        # extra inputs
        self.with_ray_directions = with_ray_directions
        self.with_ray_in_world = with_ray_in_world
        self.with_points_normal = with_points_normal

        # local depth loss
        self.local_depth_l1_loss = instantiate_from_config(local_depth_l1_loss)
        self.local_depth_grad_loss = instantiate_from_config(local_depth_grad_loss)
        self.local_depth_normal_loss = instantiate_from_config(local_depth_normal_loss)
        self.local2global_loss = instantiate_from_config(local2global_loss)
        self.local2global_camera_normalize = local2global_camera_normalize
        self.local2global_camera_detach = local2global_camera_detach

        # local other loss
        self.local_normal_loss = instantiate_from_config(local_normal_loss)
        self.local_ray_directions_loss = instantiate_from_config(local_ray_directions_loss)
        self.local_ray_in_world_loss = instantiate_from_config(local_ray_in_world_loss)
        self.local_invalid_mask_loss = instantiate_from_config(local_invalid_mask_loss)
        self.local_motion_mask_loss = instantiate_from_config(local_motion_mask_loss)

        # global points loss
        self.global_points_l1_loss = instantiate_from_config(global_points_l1_loss)
        self.global_points_grad_loss = instantiate_from_config(global_points_grad_loss)
        self.global_points_normal_loss = instantiate_from_config(global_points_normal_loss)

        # camera loss
        self.camera_loss = instantiate_from_config(camera_loss)

        # reconstruction loss
        self.rc_rgb_l1_loss = instantiate_from_config(rc_rgb_l1_loss)
        self.rc_ssim_loss = instantiate_from_config(rc_ssim_loss)
        self.rc_lpips_loss = instantiate_from_config(rc_lpips_loss)
        self.rc_depth_loss = instantiate_from_config(rc_depth_loss)
        self.rc_normal_loss = instantiate_from_config(rc_normal_loss)
        self.rc_depth_consistency_loss = instantiate_from_config(rc_depth_consistency_loss)

        # track loss
        self.track_loss = instantiate_from_config(track_loss)

        # aux model
        self.build_aux_model(
            sv_model_cfg=sv_model_cfg,
            da3_model_cfg=da3_model_cfg,
            segformer_cfg=segformer_cfg,
        )

        self.pose_encoding_type = pose_encoding_type
        self.points_from_ray = points_from_ray

        self.task_weight = task_weight
        if task_weight is None:
            self.task_weight = dict()

        self.save_output_cfg = dict(
            save_everything=False,
            save_output_conf=True,
            save_gaussians=True,
            save_render_results=True,
            save_render_video=False,
            save_render_video_with_normalize_c2w=True,
            render_video_with_pred_camera=True,
            save_glb_results=True,
            save_glb_sf_results=False,
            save_glb2local_results=False,
            save_local2glb_results=True,
            save_cameras=True,
            save_local_results=False,
            save_extra_local_results=False,
            save_track_results=True,
            save_filtered_results=True,
            save_normal=True,
            save_normal_vis=False,
            save_invalid_mask=True,
            save_output_only_local_glb=False,
            save_name_match_rgb=True,
            output_normalize_cameras=True,
            output_match_input_res=True,
            output_conf_ratio=0.2,
            output_intrinsics_from_ray=False,
            output_colmap_format=False,
            gt_out_dir="gt",
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

        self.debug_rgb_path = debug_rgb_path

    def build_aux_model(self, sv_model_cfg, da3_model_cfg, segformer_cfg):
        self.sv_model_cfg = sv_model_cfg
        self.sv_model = None
        if self.sv_model_cfg is not None:
            self.sv_model = instantiate_from_config(self.sv_model_cfg["model"])
            ckpt_path = self.sv_model_cfg["pretrain"]
            res = self.sv_model.load_state_dict(torch.load(ckpt_path, weights_only=False))
            logging.info(f"sv_model parameters are loaded from {ckpt_path}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")

            self.sv_model.eval()
            for params in self.sv_model.parameters():
                params.requires_grad = False

        self.da3_model_cfg = da3_model_cfg
        self.da3_model = None
        if self.da3_model_cfg is not None:
            self.da3_model = instantiate_from_config(self.da3_model_cfg["model"])
            ckpt_path = self.da3_model_cfg["pretrain"]
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(ckpt_path)
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = {key[6:]: val for key, val in state_dict.items()}
            res = self.da3_model.load_state_dict(state_dict, strict=False)
            logging.info(f"da3_model parameters are loaded from {self.da3_model_cfg['pretrain']}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")

            self.da3_model.eval()
            for params in self.da3_model.parameters():
                params.requires_grad = False

        self.segformer_cfg = segformer_cfg
        self.segformer = None
        if self.segformer_cfg is not None:
            from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

            segformer_name = "/mnt/netdata/Team/AI/weights/segformer-b5-finetuned-ade-640-640"
            self.mask_feature_extractor = SegformerFeatureExtractor.from_pretrained(segformer_name)
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(segformer_name)
            self.segformer = self.segformer.eval()
            for params in self.segformer.parameters():
                params.requires_grad = False

            self.segformer_size = (self.mask_feature_extractor.size["height"], self.mask_feature_extractor.size["width"])
            self.segformer_image_mean = torch.tensor(self.mask_feature_extractor.image_mean)[None, :, None, None]
            self.segformer_image_std = torch.tensor(self.mask_feature_extractor.image_std)[None, :, None, None]

    def get_inputs(self, batch):
        meta_data = batch["meta_data"]
        name = meta_data["name"][0]
        total_iter = batch.get("total_iter")

        if total_iter is not None:
            meta_data["total_iter"] = total_iter

        image = batch["image"].to(device=self.device)

        if "image_backup" in batch:
            image_backup = batch["image_backup"].to(device=self.device)
        else:
            image_backup = None

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
                prompt_extrinsics[..., :3, 3] = self.normalize(prompt_extrinsics[..., :3, 3], scale[..., 0, 0])
            elif self.extrinsics_noise_name is not None and self.extrinsics_noise_name in batch:
                extrinsics_noise = batch[self.extrinsics_noise_name].to(device=self.device)

        target_local_depth = target_global_points = target_depth_mask = None

        if self.target_local_depth_name is not None and self.target_local_depth_name in batch:
            target_local_depth = batch[self.target_local_depth_name].to(device=self.device)
            target_local_depth = self.normalize(target_local_depth, scale)

        if self.target_global_points_name is not None and self.target_global_points_name in batch:
            target_global_points = batch[self.target_global_points_name].to(device=self.device)
            target_global_points = self.normalize(target_global_points, scale)

        if self.target_depth_mask_name is not None and self.target_depth_mask_name in batch:
            target_depth_mask = batch[self.target_depth_mask_name].to(self.device)

        prompt_depth = prompt_depth_mask = None

        if self.prompt_depth_name is not None and self.prompt_depth_name in batch:
            if self.prompt_depth_name == self.target_local_depth_name:
                prompt_depth = target_local_depth.clone() if target_local_depth is not None else None
            else:
                prompt_depth = batch[self.prompt_depth_name].to(self.device)
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

        if self.target_invalid_mask_name is not None:
            if isinstance(self.target_invalid_mask_name, (tuple, list)):
                for inv_name in self.target_invalid_mask_name:
                    if inv_name in batch:
                        target_invalid_mask = batch[inv_name].to(self.device)
                        break
            elif self.target_invalid_mask_name in batch:
                target_invalid_mask = batch[self.target_invalid_mask_name].to(self.device)
        
        rgb_mask = None
        if self.rgb_mask is not None and self.rgb_mask in batch:
            rgb_mask = batch[self.rgb_mask].to(self.device)

        image_show = align_data = None

        if not self.training:
            if "image_show" in batch:
                image_show = batch["image_show"]
                if image.ndim > 4:
                    image_show = image_show.float().numpy()

            if self.save_output_cfg["output_match_input_res"] and self.align_name in batch:
                align_data = batch[self.align_name]

        # aux prompt_depth and prompt_depth_mask
        if self.training and self.sv_model is not None:
            prompt_depth, prompt_depth_mask = self.infer_sv_model(image=image, scale=scale, prompt_depth=prompt_depth, prompt_depth_mask=prompt_depth_mask, meta_data=meta_data)
        if self.training and self.da3_model is not None and ("names" not in self.da3_model_cfg or name in self.da3_model_cfg["names"]):
            if "image_backup" in batch:
                image_da3 = batch["image_backup"].clone().to(self.device) / 127.5 - 1
            else:
                image_da3 = image
            target_local_depth, target_depth_mask, prompt_depth, prompt_depth_mask, intrinsics, sky_mask = self.infer_da3_model(
                image=image_da3,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                prompt_depth_mask=prompt_depth_mask,
            )
            target_global_points = None
            if target_invalid_mask is None and sky_mask is not None:
                target_invalid_mask = sky_mask
                if target_depth_mask is not None:
                    target_depth_mask = target_depth_mask & (~target_invalid_mask)

        # extra invalid mask
        if self.training and self.segformer is not None and target_invalid_mask is None:
            target_invalid_mask = self.infer_segformer(image=image, image_backup=image_backup)
            target_depth_mask = target_depth_mask & (~target_invalid_mask)

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
            if target_normal_mask is not None:
                if target_normal_mask.ndim != target_invalid_mask.ndim:
                    target_normal_mask = with_invalid_mask(target_normal_mask.unsqueeze(1)).squeeze(1)
                else:
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
            if not self.training:
                if image_show is not None:
                    image_show = image_show.float().numpy()
            align_data = add_dim(align_data)
            ray_directions = add_dim(ray_directions)
            ray_world = add_dim(ray_world)
            rgb_mask = add_dim(rgb_mask)

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
            extrinsics_noise,
            prompt_extrinsics,
            rgb_mask,
        )

    def infer_sv_model(self, image, scale, prompt_depth, prompt_depth_mask, meta_data):
        breakpoint()
        with torch.inference_mode():
            B, V = image.shape[:2]

            if prompt_depth is not None:
                sv_prompt_depth = self.denormalize(prompt_depth, scale)
                sv_prompt_depth = sv_prompt_depth.view(B * V, *sv_prompt_depth.shape[-3:])
                sv_scale = sv_prompt_depth.view(B * V, -1).max(dim=-1)
                sv_prompt_depth /= sv_scale[:, None, None, None]
            else:
                sv_prompt_depth = None

            with torch.autocast("cuda", enabled=True, dtype=self.dtype):
                results = self.sv_model(
                    image.view(B * V, *image.shape[-3:]),
                    prompt_depth=sv_prompt_depth,
                    meta_data=meta_data,
                )

            local_depth = results["depth"]
            local_conf = results.get("confidence")
            local_invalid_mask = results.get("invalid_mask")

            if local_conf is not None:
                local_depth_mask = local_conf > 0
            if local_invalid_mask is not None:
                local_depth_mask = prompt_depth_mask & (local_invalid_mask > 0)

            local_depth = prompt_depth.view(B, V, *prompt_depth.shape[-3:])
            local_depth_mask = local_depth_mask.view(B, V, *local_depth_mask.shape[-3:])

            local_depth = self.normalize(local_depth, scale)
            local_depth = local_depth * local_depth_mask

        return local_depth, local_depth_mask

    def infer_da3_model(self, image, intrinsics, extrinsics, prompt_depth_mask):
        with torch.inference_mode():
            fake_mv = False
            if image.ndim == 4:
                fake_mv = True
                image = image.view(image.shape[0], 1, *image.shape[-3:])

            if intrinsics is not None and extrinsics is None:
                extrinsics = torch.eye(4, device=image.device)[None, None]
                extrinsics = extrinsics.expand(*intrinsics.shape[:2], 4, 4)

            with torch.autocast("cuda", enabled=True, dtype=self.dtype):
                results = self.da3_model(image, intrinsics=intrinsics, extrinsics=extrinsics)

            depth = confidence = depth_mask = prompt_depth = prompt_depth_mask = None
            
            if "depth" in results:
                depth = results["depth"].unsqueeze(-3)
                intrinsics = results["intrinsics"]
                depth = self.depth_to_points(depth, K=intrinsics, camera_type="PINHOLE")

                confidence = results["depth_conf"].unsqueeze(-3)

                conf_thresh = self.da3_model_cfg.get("conf_thresh", None)
                conf_thresh_percentile = self.da3_model_cfg.get("conf_thresh_percentile", 10.0) / 100.0
                ensure_thresh_percentile = self.da3_model_cfg.get("ensure_thresh_percentile", 90.0) / 100.0
                # sparse_ratio = self.da3_model_cfg.get("sparse_ratio", None)
                # sparse_nums = self.da3_model_cfg.get("sparse_nums", None)

                depth_mask = []
                for bi in range(image.shape[0]):
                    if conf_thresh is not None:
                        lower = torch.quantile(confidence[bi].reshape(-1), conf_thresh_percentile)
                        upper = torch.quantile(confidence[bi].reshape(-1), ensure_thresh_percentile)
                        conf_thresh = min(max(conf_thresh, lower), upper)
                    else:
                        conf_thresh = torch.quantile(confidence[bi].reshape(-1), conf_thresh_percentile)

                    depth_mask.append(confidence[bi] > conf_thresh)

                depth_mask = torch.stack(depth_mask, dim=0)

                if prompt_depth_mask is not None:
                    if prompt_depth_mask.shape[-2:] != depth_mask.shape[-2:]:
                        prompt_depth = torch.nn.functional.interpolate(
                            depth.view(-1, *depth.shape[-3:]),
                            size=prompt_depth_mask.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        prompt_depth = prompt_depth.view(*depth.shape[:2], *prompt_depth.shape[-3:])
                        depth_mask_resize = torch.nn.functional.interpolate(
                            depth_mask.view(-1, *depth_mask.shape[-3:]).float(),
                            size=prompt_depth_mask.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        depth_mask_resize = (depth_mask_resize == 1.0).view(*depth.shape[:2], *depth_mask_resize.shape[-3:])
                        prompt_depth_mask = prompt_depth_mask * depth_mask_resize
                    else:
                        prompt_depth = depth
                        prompt_depth_mask = depth_mask * prompt_depth_mask

                    prompt_depth = prompt_depth * prompt_depth_mask
                else:
                    raise NotImplementedError

            sky_mask = None
            if "sky" in results:
                sky_mask = results["sky"] > 0.3
                if not fake_mv:
                    sky_mask = sky_mask.unsqueeze(-3)

        depth = depth.clone() if depth is not None else None
        depth_mask = depth_mask.bool().clone() if depth_mask is not None else None
        prompt_depth = prompt_depth.clone() if prompt_depth is not None else None
        prompt_depth_mask = prompt_depth_mask.bool().clone() if prompt_depth_mask is not None else None
        intrinsics = intrinsics.clone() if intrinsics is not None else None
        sky_mask = sky_mask.bool().clone() if sky_mask is not None else None

        return depth, depth_mask, prompt_depth, prompt_depth_mask, intrinsics, sky_mask

    def infer_segformer(self, image, image_backup=None):
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=self.dtype):
                if self.segformer_image_mean.device != image.device or self.segformer_image_std.dtype != image.dtype:
                    self.segformer_image_mean = self.segformer_image_mean.to(image.device, image.dtype)
                    self.segformer_image_std = self.segformer_image_std.to(image.device, image.dtype)

                if image_backup is not None:
                    ndim = image_backup.ndim
                    if ndim == 5:
                        b, v = image_backup.shape[:2]
                        image_backup = image_backup.view(-1, *image_backup.shape[-3:])

                    pixel_values = image_backup / 255.0
                    image_numpy = image_backup.cpu().permute(0, 2, 3, 1).numpy()
                else:
                    ndim = image.ndim
                    if ndim == 5:
                        b, v = image.shape[:2]
                        image = image.view(-1, *image.shape[-3:])

                    pixel_values = (image + 1) * 0.5
                    image_numpy = pixel_values.cpu().permute(0, 2, 3, 1).numpy() * 255

                pixel_values = F.interpolate(pixel_values, size=self.segformer_size, mode="bilinear", align_corners=False)
                pixel_values = (pixel_values - self.segformer_image_mean) / self.segformer_image_std

                logits = self.segformer(pixel_values=pixel_values).logits.float()

                upsampled_logits = torch.nn.functional.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)  # (H, W)
                pred_seg = upsampled_logits.argmax(dim=1)  # (H, W)
                target_invalid_mask = pred_seg == self.segformer_cfg["index"]

                if "threshold" in self.segformer_cfg:
                    score = upsampled_logits.softmax(dim=1)
                    target_invalid_mask = target_invalid_mask & (score[:, self.segformer_cfg["index"]] >= self.segformer_cfg["threshold"])

                target_invalid_mask = target_invalid_mask.unsqueeze(1)

                if target_invalid_mask.sum() > 0 and 0:
                    for bi in range(image.shape[0]):
                        cur_rgb = image_numpy[bi].astype(np.uint8)
                        cv2.imwrite("rgb.png", cv2.cvtColor(cur_rgb, cv2.COLOR_RGB2BGR))
                        cur_mask = (target_invalid_mask[bi, 0].cpu().long().numpy() * 255).astype(np.uint8)
                        cv2.imwrite("mask.png", cur_mask)
                        breakpoint()

                if ndim == 5:
                    target_invalid_mask = target_invalid_mask.view(b, v, *target_invalid_mask.shape[-3:])

        return target_invalid_mask

    def get_track_inputs(self, batch):
        track_query_points = track_vis = track_pos_masks = None
        if "track_query_points" in batch:
            track_query_points = batch["track_query_points"].to(device=self.device)
        if "track_vis" in batch:
            track_vis = batch["track_vis"].to(device=self.device)
        if "track_pos_masks" in batch:
            track_pos_masks = batch["track_pos_masks"].to(device=self.device)
        return track_query_points, track_vis, track_pos_masks

    def split_inputs(
        self,
        novel_view_nums,
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
        ray_directions,
        ray_world,
        extrinsics_noise,
        prompt_extrinsics,
        rgb_mask,
        meta_data,
    ):
        def split(data):
            # NOTE: without contiguous will cause error sometimes
            novel_data = data[:, -novel_view_nums:].contiguous()
            data = data[:, :-novel_view_nums].contiguous()
            return data, novel_data

        novel_image = novel_intrinsics = novel_extrinsics = novel_target_local_depth = novel_target_depth_mask = novel_target_normal = novel_target_normal_mask = None
        if image is not None:
            image, novel_image = split(image)
        if intrinsics is not None:
            intrinsics, novel_intrinsics = split(intrinsics)
        if extrinsics is not None:
            extrinsics, novel_extrinsics = split(extrinsics)
        if scale is not None:
            scale, _ = split(scale)
        if prompt_depth is not None:
            prompt_depth, _ = split(prompt_depth)
        if target_local_depth is not None:
            target_local_depth, novel_target_local_depth = split(target_local_depth)
        if target_global_points is not None:
            target_global_points, _ = split(target_global_points)
        if target_depth_mask is not None:
            target_depth_mask, novel_target_depth_mask = split(target_depth_mask)
        if target_normal is not None:
            target_normal, novel_target_normal = split(target_normal)
        if target_normal_mask is not None:
            target_normal_mask, novel_target_normal_mask = split(target_normal_mask)
        if target_motion_mask is not None:
            target_motion_mask, _ = split(target_motion_mask)
        if target_invalid_mask is not None:
            target_invalid_mask, _ = split(target_invalid_mask)
        if ray_directions is not None:
            ray_directions, _ = split(ray_directions)
        if ray_world is not None:
            ray_world, _ = split(ray_world)
        if meta_data is not None:
            if meta_data["views"][0] == 1:
                meta_data["frames"] = meta_data["frames"] - novel_view_nums
            elif meta_data["frames"][0] == 1:
                meta_data["views"] = meta_data["views"] - novel_view_nums
            else:
                raise NotImplementedError
        if extrinsics_noise is not None:
            extrinsics_noise, _ = split(extrinsics_noise)
        if prompt_extrinsics is not None:
            prompt_extrinsics, _ = split(prompt_extrinsics)
        if rgb_mask is not None:
            rgb_mask, _ = split(rgb_mask)

        return (
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
            ray_directions,
            ray_world,
            extrinsics_noise,
            prompt_extrinsics,
            rgb_mask,
            meta_data,
            novel_image,
            novel_intrinsics,
            novel_extrinsics,
            novel_target_local_depth,
            novel_target_depth_mask,
            novel_target_normal,
            novel_target_normal_mask,
        )

    def add_loss(self, total_loss, total_loss_dict, loss, loss_dict=None, task_name=None, loss_name=None, prefix=""):
        if task_name is not None:
            weight = self.task_weight.get(task_name, 1.0)
            if isinstance(loss, dict):
                total_loss += sum(loss.values()) * weight
                if task_name not in total_loss_dict:
                    total_loss_dict[task_name] = 0
                for key, val in loss.items():
                    total_loss_dict[f"{task_name}_{key}"] = val * weight
                    if not str(key).startswith("_"):
                        total_loss_dict[task_name] += val * weight
            else:
                total_loss += loss * weight
                if task_name not in total_loss_dict:
                    total_loss_dict[task_name] = 0
                if loss_dict is not None:
                    for key, val in loss_dict.items():
                        total_loss_dict[f"{task_name}_{key}"] = val * weight
                        if not str(key).startswith("_"):
                            total_loss_dict[task_name] += val * weight
        else:
            if isinstance(loss, dict):
                total_loss += sum(loss.values())
                total_loss_dict.update({prefix + k: v for k, v in loss.items()})
            else:
                total_loss += loss
                total_loss_dict[prefix + loss_name] = loss
        return total_loss, total_loss_dict

    def get_single_base_loss(
        self,
        name,
        coord,
        pred_depth=None,
        pred_conf=None,
        pred_normal=None,
        pred_ray_directions=None,
        pred_ray_in_world=None,
        pred_motion_mask=None,
        pred_invalid_mask=None,
        target_depth=None,
        target_normal=None,
        target_normal_mask=None,
        target_ray_directions=None,
        target_ray_in_world=None,
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
        if pred_ray_in_world is not None:
            pred_ray_in_world = pred_ray_in_world.float().permute(0, 2, 3, 1).contiguous()
        if pred_motion_mask is not None:
            pred_motion_mask = pred_motion_mask.squeeze(1)
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
        if target_ray_in_world is not None:
            target_ray_in_world = target_ray_in_world.float().permute(0, 2, 3, 1).contiguous()
        if target_motion_mask is not None:
            target_motion_mask = target_motion_mask.squeeze(1)
        if target_invalid_mask is not None:
            target_invalid_mask = target_invalid_mask.squeeze(1)
        if valid_mask is not None:
            valid_mask = valid_mask.squeeze(1)

        total_loss = 0
        total_loss_dict = dict()

        # Base depth l1 loss
        depth_l1_loss_func = self.local_depth_l1_loss if coord == "local" else self.global_points_l1_loss
        if depth_l1_loss_func is not None and target_depth is not None and pred_depth is not None and pred_depth.requires_grad:
            loss = depth_l1_loss_func(
                name=name,
                pred_depth=pred_depth,
                pred_conf=pred_conf,
                target_depth=target_depth,
                valid_mask=valid_mask,
            )
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="l1_loss")

        # Depth gradient loss
        depth_grad_loss_func = self.local_depth_grad_loss if coord == "local" else self.global_points_grad_loss
        if depth_grad_loss_func is not None and target_depth is not None and pred_depth is not None and pred_depth.requires_grad:
            if pred_depth.shape[-2] > 1 and pred_depth.shape[-3] > 1:
                loss = depth_grad_loss_func(
                    name=name,
                    pred_depth=pred_depth[..., -1],
                    target_depth=target_depth[..., -1],
                    valid_mask=valid_mask,
                )
                total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="grad_loss")

        # Depth normal loss
        depth_normal_loss_func = self.local_depth_normal_loss if coord == "local" else self.global_points_normal_loss
        if depth_normal_loss_func is not None and pred_depth is not None and pred_depth.requires_grad:
            if pred_depth.shape[-2] > 1 and pred_depth.shape[-3] > 1:
                loss = depth_normal_loss_func(name=name, pred_depth=pred_depth, target_depth=target_depth, target_normal=target_normal, valid_mask=valid_mask)
                total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="normal_loss")

        if coord == "local":
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
                loss = self.local_invalid_mask_loss(name=name, pred_invalid_mask=pred_invalid_mask, target_invalid_mask=target_invalid_mask, valid_mask=None)
                total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="pd_inv_mask_loss")

            # Local pred motion_mask loss
            if self.local_motion_mask_loss is not None and target_motion_mask is not None and pred_motion_mask is not None and pred_motion_mask.requires_grad:
                loss = self.local_motion_mask_loss(name=name, pred_motion_mask=pred_motion_mask, target_motion_mask=target_motion_mask, valid_mask=None)
                total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="pd_mot_mask_loss")

        if coord == "global":
            if self.local_ray_in_world_loss is not None and target_ray_in_world is not None and pred_ray_in_world is not None and pred_ray_in_world.requires_grad:
                loss = self.local_ray_in_world_loss(name=name, pred_ray_directions=pred_ray_in_world, target_ray_directions=target_ray_in_world)
                total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss, loss_name="pd_ray_world_loss")

        return total_loss, total_loss_dict

    def get_base_loss(
        self,
        name,
        image,
        coord="local",
        pred_depth=None,
        pred_conf=None,
        pred_normal=None,
        pred_ray_directions=None,
        pred_ray_in_world=None,
        pred_motion_mask=None,
        pred_invalid_mask=None,
        target_depth=None,
        target_normal=None,
        target_normal_mask=None,
        target_ray_directions=None,
        target_ray_in_world=None,
        target_motion_mask=None,
        target_invalid_mask=None,
        valid_mask=None,
        scale=None,
    ):
        n = image.shape[1]

        total_loss = 0
        total_loss_dict = dict()

        for i in range(n):
            loss, loss_dict = self.get_single_base_loss(
                name=name,
                coord=coord,
                pred_depth=pred_depth[:, i] if pred_depth is not None else None,
                pred_conf=pred_conf[:, i] if pred_conf is not None else None,
                pred_normal=pred_normal[:, i] if pred_normal is not None else None,
                pred_ray_directions=pred_ray_directions[:, i] if pred_ray_directions is not None else None,
                pred_ray_in_world=pred_ray_in_world[:, i] if pred_ray_in_world is not None else None,
                pred_motion_mask=pred_motion_mask[:, i] if pred_motion_mask is not None else None,
                pred_invalid_mask=pred_invalid_mask[:, i] if pred_invalid_mask is not None else None,
                target_depth=target_depth[:, i] if target_depth is not None else None,
                target_normal=target_normal[:, i] if target_normal is not None else None,
                target_normal_mask=target_normal_mask[:, i] if target_normal_mask is not None else None,
                target_ray_directions=target_ray_directions[:, i] if target_ray_directions is not None else None,
                target_ray_in_world=target_ray_in_world[:, i] if target_ray_in_world is not None else None,
                target_motion_mask=target_motion_mask[:, i] if target_motion_mask is not None else None,
                target_invalid_mask=target_invalid_mask[:, i] if target_invalid_mask is not None else None,
                valid_mask=valid_mask[:, i] if valid_mask is not None else None,
                scale=scale[:, i] if scale is not None else None,
            )
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
        pred_track,
        pred_track_vis,
        pred_track_conf,
        track_query_points,
        track_vis,
        track_pos_masks,
        image_hw,
    ):
        return self.track_loss(
            name=name,
            pred_tracks=([d.float() for d in pred_track] if isinstance(pred_track, (list, tuple)) else pred_track.float()),
            vis_preds=pred_track_vis.float(),
            conf_preds=pred_track_conf.float(),
            track_gt=track_query_points.float(),
            valids=track_pos_masks.float(),
            vis=track_vis.float(),
            image_hw=image_hw,
        )

    def get_ffgs_loss(
        self,
        name,
        render_rgb=None,
        render_depth=None,
        render_normal=None,
        target_rgb=None,
        target_depth=None,
        target_depth_mask=None,
        **kwargs,
    ):
        if render_rgb is not None:
            render_rgb = render_rgb.float()
        if render_depth is not None:
            render_depth = render_depth.float().permute(0, 1, 3, 4, 2).contiguous()
            render_depth = render_depth.reshape(-1, *render_depth.shape[-3:])
        if render_normal is not None:
            render_normal = render_normal.float().reshape(-1, *render_normal.shape[-3:])

        if target_rgb is not None:
            target_rgb = (target_rgb.float() + 1) * 0.5
        if target_depth is not None:
            target_depth = target_depth.float().permute(0, 1, 3, 4, 2).contiguous()
            target_depth = target_depth.reshape(-1, *target_depth.shape[-3:])
        if target_depth_mask is not None:
            target_depth_mask = target_depth_mask.squeeze(2).bool()
            target_depth_mask = target_depth_mask.reshape(-1, *target_depth_mask.shape[-2:])

        total_loss, total_loss_dict = 0, dict()

        if self.rc_rgb_l1_loss is not None and target_rgb is not None and render_rgb is not None and render_rgb.requires_grad:
            loss = self.rc_rgb_l1_loss(name=name, rgbs=target_rgb, render_rgbs=render_rgb, mask=None)
            total_loss += loss
            total_loss_dict["l1_loss"] = loss

        if self.rc_ssim_loss is not None and target_rgb is not None and render_rgb is not None and render_rgb.requires_grad:
            loss = self.rc_ssim_loss(name=name, rgbs=target_rgb, render_rgbs=render_rgb, mask=None)
            total_loss += loss
            total_loss_dict["ssim_loss"] = loss

        if self.rc_lpips_loss is not None and target_rgb is not None and render_rgb is not None and render_rgb.requires_grad:
            loss = self.rc_lpips_loss(name=name, rgbs=target_rgb, render_rgbs=render_rgb, mask=None)
            total_loss += loss
            total_loss_dict["lpips_loss"] = loss

        if self.rc_depth_loss is not None and target_depth is not None and render_depth is not None and render_depth.requires_grad:
            loss = self.rc_depth_loss(name=name, pred_depth=render_depth, pred_conf=None, target_depth=target_depth, valid_mask=target_depth_mask)
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss=loss["l1_loss"] if isinstance(loss, dict) else loss, loss_name="dpt_loss")

        if self.rc_normal_loss is not None and target_depth is not None and render_normal is not None and render_normal.requires_grad:
            # depth_norm: [BS, H, W, 3]
            # render_normal: [BS, 3, H, W]
            # valid_mask: [BS, H, W]
            loss = self.rc_normal_loss(name=name, target_depth=target_depth, prediction=render_normal, mask=target_depth_mask)
            total_loss += loss
            total_loss_dict["norm_loss"] = loss

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
            extrinsics_noise,
            prompt_extrinsics,
            rgb_mask,
        ) = self.get_inputs(batch)
        
        camera_type = meta_data.get("camera_type", ["PINHOLE"])[0]

        track_query_points, track_vis, track_pos_masks = self.get_track_inputs(batch)

        novel_view_nums = meta_data.get("novel_view_nums", None)
        novel_view_nums = int(novel_view_nums[0]) if novel_view_nums is not None else 0

        if novel_view_nums > 0:
            (
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
                ray_directions,
                ray_world,
                extrinsics_noise,
                prompt_extrinsics,
                rgb_mask,
                meta_data,
                novel_image,
                novel_intrinsics,
                novel_extrinsics,
                novel_target_local_depth,
                novel_target_depth_mask,
                novel_target_normal,
                novel_target_normal_mask,
            ) = self.split_inputs(
                novel_view_nums=novel_view_nums,
                image=image,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                scale=scale,
                prompt_depth=prompt_depth,
                target_local_depth=target_local_depth,
                target_global_points=target_global_points,
                target_depth_mask=target_depth_mask,
                target_normal=target_normal,
                target_normal_mask=target_normal_mask,
                target_motion_mask=target_motion_mask,
                target_invalid_mask=target_invalid_mask,
                ray_directions=ray_directions,
                ray_world=ray_world,
                extrinsics_noise=extrinsics_noise,
                prompt_extrinsics=prompt_extrinsics,
                rgb_mask=rgb_mask,
                meta_data=meta_data,
            )
            assert track_query_points is None  # TODO
        else:
            novel_intrinsics = novel_extrinsics = None

        if prompt_extrinsics is not None:
            w2c = prompt_extrinsics
        elif extrinsics_noise is not None and extrinsics is not None:
            w2c = torch.matmul(extrinsics_noise, extrinsics)
        else:
            w2c = extrinsics

        results = self.model(
            image,
            scale=scale,
            prompt_depth=prompt_depth,
            intrinsics=intrinsics,
            ray_directions=ray_directions,
            # w2c=extrinsics if extrinsics_noise is None else torch.matmul(extrinsics_noise, extrinsics),
            w2c=w2c,
            ray_world=ray_world,
            query_points=track_query_points[:, 0] if track_query_points is not None else None,
            meta_data=meta_data,
            novel_intrinsics=novel_intrinsics,
            novel_w2c=novel_extrinsics,
            rgb_mask=rgb_mask,
        )

        # Extract predictions from the model outputs
        pred_local_depth = results.get("depth")
        pred_local_conf = results.get("confidence")
        pred_local_normal = results.get("normal")
        pred_local_invalid_mask = results.get("invalid_mask")
        pred_local_motion_mask = results.get("motion_mask")

        pred_ray_directions = results.get("ray")
        pred_ray_in_world = results.get("ray_in_world")

        pred_global_points = results.get("global_points")
        pred_global_conf = results.get("global_confidence")

        pred_track = results.get("track")
        pred_track_vis = results.get("track_vis")
        pred_track_conf = results.get("track_confidence")

        if self.pose_encoding_type == "absT_quaR_FoV":
            pose_enc = results.get("pose_enc")
        else:
            raise NotImplementedError

        gaussians = results.get("ffgs_gaussians")
        render_rgb = results.get("ffgs_render_rgb")
        render_depth = results.get("ffgs_render_depth")
        render_normal = results.get("ffgs_render_normal")
        pred_ffgs_depth = results.get("ffgs_depth")
        pred_ffgs_conf = results.get("ffgs_confidence")

        total_loss, total_loss_dict = 0, dict()

        # Normalize the ray directions to unit vectors
        if pred_ray_directions is not None:
            pred_ray_directions = pred_ray_directions / pred_ray_directions.norm(dim=2, keepdim=True).clip(min=1e-8)

        # DepthMap trans to PointMap
        if pred_local_depth is not None and pred_local_depth.shape[-3] == 1:
            if self.points_from_ray:
                pred_local_depth = pred_local_depth * ray_directions
            else:
                pred_local_depth = self.depth_to_points(pred_local_depth, K=intrinsics, camera_type=camera_type)

        # DepthMap trans to PointMap
        if pred_ffgs_depth is not None and pred_ffgs_depth.shape[-3] == 1:
            if self.points_from_ray:
                pred_ffgs_depth = pred_ffgs_depth * ray_directions
            else:
                pred_ffgs_depth = self.depth_to_points(pred_ffgs_depth, K=intrinsics, camera_type=camera_type)

        # Normalize the normal maps
        if pred_local_normal is not None:
            tgt_h, tgt_w = pred_local_depth.shape[-2:] if pred_local_depth is not None else image.shape[-2:]
            norm_h, norm_w = pred_local_normal.shape[-2:]
            if tgt_h != norm_h or tgt_w != norm_w:
                pred_local_normal = F.interpolate(pred_local_normal, (tgt_h, tgt_w), mode="bilinear", align_corners=False, antialias=False)
            pred_local_normal = F.normalize(pred_local_normal, dim=-3)

        loss, loss_dict = self.get_base_loss(
            name=name,
            image=image,
            coord="local",
            pred_depth=pred_local_depth,
            pred_conf=pred_local_conf,
            pred_normal=pred_local_normal,
            pred_ray_directions=pred_ray_directions,
            pred_invalid_mask=pred_local_invalid_mask,
            pred_motion_mask=pred_local_motion_mask,
            target_depth=target_local_depth,
            target_ray_directions=ray_directions,
            target_normal=target_normal,
            target_normal_mask=target_normal_mask,
            target_invalid_mask=target_invalid_mask,
            valid_mask=target_depth_mask,
            scale=scale,
        )
        total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="lcl")

        loss, loss_dict = self.get_base_loss(
            name=name,
            image=image,
            coord="global",
            pred_depth=pred_global_points,
            pred_conf=pred_global_conf,
            pred_ray_in_world=pred_ray_in_world,
            target_depth=target_global_points,
            target_ray_in_world=ray_world,
            valid_mask=target_depth_mask,
        )
        total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="glb")

        if self.local2global_loss is not None and pose_enc is not None and pred_local_depth is not None:
            tmp_pred_depth = results["depth"]
            B, S, C, H, W = tmp_pred_depth.shape
            assert C == 1

            with torch.cuda.amp.autocast(False):
                if self.pose_encoding_type == "absT_quaR_FoV":
                    pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                        pose_encoding=pose_enc[-1].float() if isinstance(pose_enc, (list, tuple)) else pose_enc.float(),
                        image_size_hw=(H, W),
                        build_intrinsics=True,
                    )
                    if self.local2global_camera_detach:
                        pred_extrinsics = pred_extrinsics.detach()
                        pred_intrinsics = pred_intrinsics.detach()

                    if self.local2global_camera_normalize:
                        w2c_pred = pred_extrinsics
                        base_c2w_pred = w2c_pred[:, 0:1].inverse()
                        pred_extrinsics = w2c_pred @ base_c2w_pred

                else:
                    raise NotImplementedError

                tmp_pred_depth = self.depth_to_points(tmp_pred_depth.float(), K=pred_intrinsics, camera_type=camera_type)
                tmp_pred_depth = pred_extrinsics.inverse() @ torch.cat([tmp_pred_depth, tmp_pred_depth.new_ones([B, S, 1, H, W])], dim=2).reshape(B, S, 4, -1)
                tmp_pred_depth = tmp_pred_depth.reshape(B, S, 4, H, W)[:, :, :3, :, :]

            if pred_local_conf is not None:
                tmp_pred_conf = pred_local_conf.squeeze(2).unsqueeze(-1)
            else:
                tmp_pred_conf = None

            loss = self.local2global_loss(
                name=name,
                pred_depth=tmp_pred_depth.float().permute(0, 1, 3, 4, 2).contiguous(),
                target_depth=target_global_points.float().permute(0, 1, 3, 4, 2).contiguous(),
                valid_mask=target_depth_mask.squeeze(2),
                pred_conf=tmp_pred_conf,
            )
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict=None, task_name="l2g")

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

        if self.track_loss is not None and pred_track is not None and pred_track[0].requires_grad:
            loss, loss_dict = self.get_track_loss(
                name=name,
                pred_track=pred_track,
                pred_track_vis=pred_track_vis,
                pred_track_conf=pred_track_conf,
                track_query_points=track_query_points,
                track_vis=track_vis,
                track_pos_masks=track_pos_masks,
                image_hw=image.shape[-2:],
            )
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="tk")

        if render_depth is not None:
            render_depth = render_depth.unsqueeze(2)
            if hasattr(gaussians, "norm_scales") and gaussians.norm_scales is not None:
                render_depth = render_depth * gaussians.norm_scales[:, None, None, None, None]
            # TODO: close normalize
            # render_depth = self.normalize(render_depth, scale=scale)
            render_depth = self.depth_to_points(render_depth, K=intrinsics, camera_type=camera_type)

        loss, loss_dict = self.get_ffgs_loss(
            name=name,
            render_rgb=render_rgb,
            render_depth=render_depth,
            render_normal=render_normal,
            target_rgb=image,
            target_depth=target_local_depth,
            target_depth_mask=target_depth_mask,
        )
        total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="rc")

        if pred_ffgs_depth is not None:
            loss, loss_dict = self.get_base_loss(
                name=name,
                image=image,
                coord="local",
                pred_depth=pred_ffgs_depth,
                pred_conf=pred_ffgs_conf,
                target_depth=target_local_depth,
                valid_mask=target_depth_mask,
            )
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="rcd")

            if self.local2global_loss is not None and pose_enc is not None and pred_ffgs_depth is not None:
                tmp_pred_depth = results["ffgs_depth"]
                B, S, C, H, W = tmp_pred_depth.shape
                assert C == 1, f"{[B, S, C, H, W]}"

                with torch.cuda.amp.autocast(False):
                    if self.pose_encoding_type == "absT_quaR_FoV":
                        pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                            pose_encoding=pose_enc[-1].float() if isinstance(pose_enc, (list, tuple)) else pose_enc.float(),
                            image_size_hw=(H, W),
                            build_intrinsics=True,
                        )
                    else:
                        raise NotImplementedError

                    tmp_pred_depth = self.depth_to_points(tmp_pred_depth.float(), K=pred_intrinsics, camera_type=camera_type)
                    tmp_pred_depth = pred_extrinsics.inverse() @ torch.cat([tmp_pred_depth, tmp_pred_depth.new_ones([B, S, 1, H, W])], dim=2).reshape(B, S, 4, -1)
                    tmp_pred_depth = tmp_pred_depth.reshape(B, S, 4, H, W)[:, :, :3, :, :]

                if pred_local_conf is not None:
                    tmp_pred_conf = pred_local_conf.squeeze(2).unsqueeze(-1)
                else:
                    tmp_pred_conf = None

                loss = self.local2global_loss(
                    name=name,
                    pred_depth=tmp_pred_depth.float().permute(0, 1, 3, 4, 2).contiguous(),
                    target_depth=target_global_points.float().permute(0, 1, 3, 4, 2).contiguous(),
                    valid_mask=target_depth_mask.squeeze(2),
                    pred_conf=tmp_pred_conf,
                )
                total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict=None, task_name="rcd_l2g")

            if self.rc_depth_consistency_loss is not None and pred_ffgs_depth is not None and render_depth is not None:
                B, S, C, H, W = render_depth.shape
                loss = self.rc_depth_consistency_loss(
                    name=name,
                    pred_depth=render_depth.float().permute(0, 1, 3, 4, 2).contiguous(),
                    target_depth=pred_ffgs_depth.float().permute(0, 1, 3, 4, 2).contiguous(),
                    valid_mask=render_depth.new_ones([B, S, H, W]).bool(),
                )
                total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict=None, task_name="rcd_consis")

        if novel_view_nums > 0 and novel_intrinsics is not None and novel_extrinsics is not None:
            from hAlgorithm.modules.models2.external.worldmirror.models.utils.frustum import calculate_in_frustum_mask

            with torch.no_grad():
                unproject_masks = calculate_in_frustum_mask(
                    depth_1=novel_target_local_depth[:, :, -1],
                    intrinsics_1=novel_intrinsics[..., :3, :3],
                    c2w_1=novel_extrinsics.float().inverse(),
                    depth_2=target_local_depth[:, :, -1],
                    intrinsics_2=intrinsics[..., :3, :3],
                    c2w_2=extrinsics.float().inverse(),
                )  # [B, V, H, W]

                B, V, H, W = unproject_masks.shape
                unproject_masks = unproject_masks.unsqueeze(2)

            novel_render_rgb = results["ffgs_novel_render_rgb"]
            novel_render_depth = results.get("ffgs_novel_render_depth", None)
            novel_render_normal = results.get("ffgs_novel_render_normal", None)

            if 0:
                debug_novel_image = (novel_image.float() + 1) * 0.5
                for i in range(V):
                    cv2.imwrite(f"novel_rgb_{i}.png", (debug_novel_image[0][i].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    cv2.imwrite(f"unproject_masks_{0}.png", (unproject_masks[0][0][0].detach().cpu().float().numpy() * 255).astype(np.uint8))

                debug_image = (image.float() + 1) * 0.5
                for i in range(debug_image.shape[1]):
                    cv2.imwrite(f"rgb_{i}.png", (debug_image[0][i].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

            novel_render_rgb = novel_render_rgb * unproject_masks.float()

            if novel_render_depth is not None:
                novel_render_depth = novel_render_depth.view(B, V, 1, *novel_render_depth.shape[-2:])
                novel_render_depth = novel_render_depth * unproject_masks.float()

                if hasattr(gaussians, "norm_scales") and gaussians.norm_scales is not None:
                    novel_render_depth = novel_render_depth * gaussians.norm_scales[:, None, None, None, None]

                novel_render_depth = self.depth_to_points(novel_render_depth, K=novel_intrinsics, camera_type=camera_type)

            if novel_render_normal is not None:
                novel_render_normal = novel_render_normal * unproject_masks.float()

            loss, loss_dict = self.get_ffgs_loss(
                name=name,
                render_rgb=novel_render_rgb,
                render_depth=novel_render_depth,
                render_normal=novel_render_normal,
                target_rgb=novel_image,
                target_depth=novel_target_local_depth,
                target_depth_mask=novel_target_depth_mask,
            )
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="nrc")

        if scale is not None:
            total_loss_dict["scale"] = round(scale.max().cpu().item(), 2)

        if image is not None:
            aspect_ratio = image.shape[-1] / image.shape[-2]
            max_size = max(image.shape[-1], image.shape[-2])
            total_loss_dict["aspect_ratio"] = round(aspect_ratio, 2)
            total_loss_dict["bs"] = int(image.shape[0])
            total_loss_dict["view"] = int(image.shape[1])
            total_loss_dict["max_size"] = int(max_size)

        if self.debug_rgb_path:
            if isinstance(meta_data["data_info"][0], (list, tuple)):
                logging.debug(f"rgb: {name}, {meta_data['data_info'][0][0]['rgb']}")
            else:
                logging.debug(f"rgb: {name}, {meta_data['data_info'][0]['rgb']}")

        return total_loss, total_loss_dict

    def postprocess(
        self,
        pred_local_points=None,
        pred_local_conf=None,
        pred_local_normal=None,
        pred_local_invalid_mask=None,
        pred_global_points=None,
        pred_global_conf=None,
        pred_extrinsics=None,
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
            if pred_local_points.shape[0] == 3:
                pred_local_points = pred_local_points.reshape(3, -1).transpose(1, 0)
            else:
                pred_local_points = None

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
            if align_data.shape[:2] != pred_local_depth.shape[:2]:
                h, w = align_data.shape[:2]
                pred_local_depth = cv2.resize(pred_local_depth, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

                if pred_local_conf is not None:
                    pred_local_conf = cv2.resize(pred_local_conf, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

        if pred_local_normal is not None:
            pred_local_normal = pred_local_normal[0].cpu().float().numpy().transpose(1, 2, 0)

        if pred_local_invalid_mask is not None:
            pred_local_invalid_mask = pred_local_invalid_mask[0, 0].cpu().float().numpy()

        if pred_global_points is not None:
            pred_global_points = pred_global_points[0].detach().cpu().float().numpy().transpose(1, 2, 0)

        # Add multi-view confidence predictions if available
        if pred_global_conf is not None:
            pred_global_conf = pred_global_conf[0, 0].detach().cpu().float().numpy()

        # Add predicted extrinsics if available
        if pred_extrinsics is not None:
            pred_extrinsics = pred_extrinsics[0].detach().cpu().float().numpy()

        # Add predicted intrinsics if available
        if pred_intrinsics is not None:
            pred_intrinsics = pred_intrinsics[0].detach().cpu().float().numpy()

        if pred_local_points is not None and pred_extrinsics is not None and pred_intrinsics is not None:
            local2glb_points = (
                np.linalg.inv(pred_extrinsics)
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
            if prompt_depth.shape[-3] > 3:
                prompt_depth = prompt_depth[..., :3, :, :]
            prompt_h, prompt_w = prompt_depth.shape[-2:]
            prompt_depth = prompt_depth[0].cpu().float().numpy().transpose(1, 2, 0).reshape(-1, 3)
        else:
            prompt_h = prompt_w = None

        pointmap_gt_h = pointmap_gt_w = None
        if target_local_depth is not None:
            pointmap_gt_h, pointmap_gt_w = target_local_depth[0].shape[-2], target_local_depth[0].shape[-1]

        if target_local_depth is not None:
            target_local_depth = target_local_depth[0].cpu().float().numpy().reshape(3, -1).transpose(1, 0)

        if target_global_points is not None:
            target_global_points = target_global_points[0].cpu().float().numpy().reshape(3, -1).transpose(1, 0)

        if target_depth_mask is not None:
            target_depth_mask = target_depth_mask[0].cpu().numpy()

        if target_normal is not None:
            target_normal = target_normal[0].cpu().numpy().transpose(1, 2, 0)

        if target_normal_mask is not None:
            target_normal_mask = target_normal_mask.squeeze().cpu().numpy()

        if intrinsics is not None:
            intrinsics = intrinsics[0].cpu().numpy()

        if extrinsics is not None:
            extrinsics = extrinsics[0].cpu().numpy()

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
            glb_mv_pointmap=pred_global_points,
            glb_mv_confidence=pred_global_conf,
            extrinsics_pred=pred_extrinsics,
            intrinsics_pred=pred_intrinsics,
            prompt_scale=scale,
            prompt_pointmap=prompt_depth,
            prompt_h=prompt_h,
            prompt_w=prompt_w,
            pointmap_gt=target_local_depth,
            pointmap_gt_h=pointmap_gt_h,
            pointmap_gt_w=pointmap_gt_w,
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
            extrinsics_noise,
            prompt_extrinsics,
            rgb_mask,
        ) = self.get_inputs(batch)

        camera_type = meta_data.get("camera_type", ["PINHOLE"])[0]

        track_query_points, track_vis, _ = self.get_track_inputs(batch)

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
                    scale=scale,
                    prompt_depth=prompt_depth,
                    intrinsics=intrinsics,
                    ray_directions=ray_directions,
                    # w2c=extrinsics if extrinsics_noise is None else torch.matmul(extrinsics_noise, extrinsics),
                    w2c=w2c,
                    ray_world=ray_world,
                    query_points=track_query_points[:, 0] if track_query_points is not None else None,
                    rgb_mask=rgb_mask,
                    meta_data=meta_data,
                )
        else:
            results = self.model(
                image,
                scale=scale,
                prompt_depth=prompt_depth,
                intrinsics=intrinsics,
                ray_directions=ray_directions,
                # w2c=extrinsics if extrinsics_noise is None else torch.matmul(extrinsics_noise, extrinsics),
                w2c=w2c,
                ray_world=ray_world,
                query_points=track_query_points[:, 0] if track_query_points is not None else None,
                rgb_mask=rgb_mask,
                meta_data=meta_data,
            )

        if self.pose_encoding_type == "absT_quaR_FoV":
            pose_enc = results.get("pose_enc", None)
            if pose_enc is not None and isinstance(pose_enc, (list, tuple)):
                pose_enc = pose_enc[-1]
        else:
            raise NotImplementedError

        # Pose encoding for camera extrinsics/intrinsics
        if pose_enc is not None:
            if self.pose_encoding_type == "absT_quaR_FoV":
                pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                    pose_encoding=pose_enc.float(),
                    image_size_hw=image.shape[-2:],
                    build_intrinsics=True,
                )
                if scale is not None:
                    pred_extrinsics[..., :3, 3] = self.denormalize(pred_extrinsics[..., :3, 3], scale=scale[..., 0, 0])

                if self.save_output_cfg["output_normalize_cameras"]:
                    w2c_pred = pred_extrinsics
                    base_c2w_pred = w2c_pred[:, 0:1].inverse()
                    pred_extrinsics = w2c_pred @ base_c2w_pred

            else:
                raise NotImplementedError
        else:
            pred_extrinsics, pred_intrinsics = None, None

        # pred_extrinsics = torch.matmul(extrinsics_noise, extrinsics)
        # if prompt_extrinsics is not None:
        #     pred_extrinsics = prompt_extrinsics
        #     pred_extrinsics[..., :3, 3] = self.denormalize(pred_extrinsics[..., :3, 3], scale=scale[..., 0, 0])

        # Extract predictions from the model outputs
        pred_local_depth = results.get("depth")
        pred_local_conf = results.get("confidence")
        pred_local_normal = results.get("normal")
        pred_local_invalid_mask = results.get("invalid_mask")
        pred_local_motion_mask = results.get("motion_mask")

        pred_ray_directions = results.get("ray")
        # Normalize the ray directions to unit vectors
        if pred_ray_directions is not None:
            pred_ray_directions = pred_ray_directions / pred_ray_directions.norm(dim=2, keepdim=True).clip(min=1e-8)

            if self.save_output_cfg["output_intrinsics_from_ray"]:
                assert pred_ray_directions.ndim == 5
                pred_intrinsics = recover_pinhole_intrinsics_from_ray_directions(ray_directions=pred_ray_directions.view(-1, *pred_ray_directions.shape[2:]).permute(0, 2, 3, 1).contiguous())
                pred_intrinsics = pred_intrinsics.view(*pred_ray_directions.shape[:2], *pred_intrinsics.shape[1:])

        pred_global_points = results.get("global_points")
        pred_global_conf = results.get("global_confidence")

        pred_track = results.get("track")
        if pred_track is not None and isinstance(pred_track, (list, tuple)):
            pred_track = pred_track[-1]
        pred_track_vis = results.get("track_vis")
        pred_track_conf = results.get("track_confidence")

        gaussians = results.get("ffgs_gaussians")
        render_rgb = results.get("ffgs_render_rgb")
        render_depth = results.get("ffgs_render_depth")
        # render_normal = results.get("ffgs_render_normal")
        if pred_local_depth is None:
            pred_local_depth = results.get("ffgs_depth")
            pred_local_conf = results.get("ffgs_confidence")

        if prompt_depth is not None and scale is not None:
            prompt_depth = self.denormalize(prompt_depth, scale=scale)
        if target_local_depth is not None and scale is not None:
            target_local_depth = self.denormalize(target_local_depth, scale=scale)
        if target_global_points is not None and scale is not None:
            target_global_points = self.denormalize(target_global_points, scale=scale)
        if extrinsics is not None and scale is not None:
            extrinsics[..., :3, 3] = self.denormalize(extrinsics[..., :3, 3], scale=scale[..., 0, 0])

        if pred_local_depth is not None:
            if pred_local_depth.shape[-3] == 1:
                if self.points_from_ray:
                    pred_local_depth = pred_ray_directions * pred_local_depth
                elif pred_intrinsics is not None:
                    pred_local_depth = self.depth_to_points(pred_local_depth, K=pred_intrinsics, camera_type=camera_type)
                else:
                    pred_local_depth = self.depth_to_points(pred_local_depth, K=intrinsics, camera_type=camera_type)

            if scale is not None:
                pred_local_depth = self.denormalize(pred_local_depth, scale=scale)

        # Normalize the normal maps
        if pred_local_normal is not None:
            tgt_h, tgt_w = pred_local_depth.shape[-2:] if pred_local_depth is not None else image.shape[-2:]
            norm_h, norm_w = pred_local_normal.shape[-2:]
            if tgt_h != norm_h or tgt_w != norm_w:
                pred_local_normal = F.interpolate(pred_local_normal, (tgt_h, tgt_w), mode="bilinear", align_corners=False, antialias=False)
            pred_local_normal = F.normalize(pred_local_normal, dim=-3)

        if pred_global_points is not None and scale is not None:
            pred_global_points = self.denormalize(pred_global_points, scale=scale)

        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        def get_single_view_data(data, index):
            return data[:, index] if data is not None else None

        mv_outputs = []
        for fi in range(frame_num):
            for vi in range(view_num):
                index = fi * view_num + vi
                single = self.postprocess(
                    pred_local_points=get_single_view_data(pred_local_depth, index),
                    pred_local_conf=get_single_view_data(pred_local_conf, index),
                    pred_local_normal=get_single_view_data(pred_local_normal, index),
                    pred_local_invalid_mask=get_single_view_data(pred_local_invalid_mask, index),
                    pred_global_points=get_single_view_data(pred_global_points, index),
                    pred_global_conf=get_single_view_data(pred_global_conf, index),
                    pred_extrinsics=get_single_view_data(pred_extrinsics, index),
                    pred_intrinsics=get_single_view_data(pred_intrinsics, index),
                    image=get_single_view_data(image, index),
                    image_show=get_single_view_data(image_show, index),
                    scale=get_single_view_data(scale, index),
                    prompt_depth=get_single_view_data(prompt_depth, index),
                    target_local_depth=get_single_view_data(target_local_depth, index),
                    target_global_points=get_single_view_data(target_global_points, index),
                    target_depth_mask=get_single_view_data(target_depth_mask, index),
                    target_normal=get_single_view_data(target_normal, index),
                    target_normal_mask=get_single_view_data(target_normal_mask, index),
                    intrinsics=get_single_view_data(intrinsics, index),
                    extrinsics=get_single_view_data(extrinsics, index),
                    align_data=get_single_view_data(align_data, 0),
                )

                # extra outputs
                if pred_track is not None:
                    single.track_pred = pred_track[0, index].cpu().numpy()
                    single.track_vis_pred = pred_track_vis[0, index].cpu().numpy()
                    single.track_pred_local_conf = pred_track_conf[0, index].cpu().numpy()

                    breakpoint()
                    single.track_gt = track_query_points[0, index].cpu().numpy()
                    if track_vis is not None:
                        single.track_vis = track_vis[0, index].cpu().numpy()

                # Add Gaussian representation if available
                # Gaussian is shared across views, and maybe across frames in the future
                if gaussians is not None and index == 0:
                    single.gaussians = gaussians

                single.rgb = (image[0, index].permute(1, 2, 0).float().cpu().numpy() + 1) * 0.5

                # Add rendered RGB image if available
                if render_rgb is not None:
                    single.render_rgb = render_rgb[0, index].cpu().numpy().transpose(1, 2, 0)

                # Add rendered depth map if available
                if render_depth is not None:
                    single.render_depth = render_depth[0, index].cpu().numpy()

                if pred_ray_directions is not None:
                    single.ray_directions = pred_ray_directions[0, index].cpu().permute(1, 2, 0).contiguous().numpy()
                if ray_directions is not None:
                    single.ray_directions_gt = ray_directions[0, index].cpu().permute(1, 2, 0).contiguous().numpy()

                # Store frame and view indices
                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                mv_outputs.append(single)

        return mv_outputs

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
        if self.save_output_cfg["gt_out_dir"] is None:
            return None, None, None

        abs_gt_out_dir = os.path.join(os.path.dirname(os.path.dirname(out_dir)), self.save_output_cfg["gt_out_dir"], os.path.basename(out_dir))
        _, gt_glb_out_dir, _, _, gt_track_out_dir = self.get_out_dir(abs_gt_out_dir, data_idx=None)

        return abs_gt_out_dir, gt_glb_out_dir, gt_track_out_dir

    def visualize(self, outputs_list, meta_data, out_dir):
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        data_idx = meta_data["data_idx"][0]

        gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir = self.get_out_dir(out_dir, data_idx=data_idx)
        gt_out_dir, gt_glb_out_dir, gt_track_out_dir = self.get_gt_out_dir(out_dir)

        # Images Visualization
        if gt_out_dir is not None:
            vis_images(cfg=self.save_output_cfg, mv_outputs=outputs_list, gt_out_dir=gt_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num)

        # Local results Visualization
        if self.save_output_cfg["save_local_results"]:
            os.makedirs(mv_out_dir, exist_ok=True)
            vis_local_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, mv_out_dir=mv_out_dir, frame_num=frame_num, view_num=view_num, gt_out_dir=gt_out_dir, meta_data=meta_data)
        if self.save_output_cfg["save_extra_local_results"]:
            os.makedirs(mv_out_dir, exist_ok=True)
            vis_extra_local_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, mv_out_dir=mv_out_dir, frame_num=frame_num, view_num=view_num, meta_data=meta_data)

        # Global results Visualization
        if self.save_output_cfg["save_glb_results"]:
            vis_glb_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, glb_out_dir=glb_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num, gt_out_dir=gt_glb_out_dir)

        # Camera Visualization
        if self.save_output_cfg["save_cameras"]:
            vis_camera_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, camera_out_dir=camera_out_dir, frame_num=frame_num, view_num=view_num)

        # Gaussians Visualization
        if self.save_output_cfg["save_gaussians"] and outputs_list[0].gaussians is not None:
            vis_render(cfg=self.save_output_cfg, mv_outputs=outputs_list, gs_out_dir=gs_out_dir, data_idx=data_idx)
            vis_render_video(cfg=self.save_output_cfg, model=self.model, mv_outputs=outputs_list, gs_out_dir=gs_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num, device=self.device)

        # Track Visualization
        if self.save_output_cfg["save_track_results"] and outputs_list[0].track_pred is not None:
            os.makedirs(track_out_dir, exist_ok=True)
            vis_track_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, track_out_dir=track_out_dir, data_idx=data_idx, gt_out_dir=gt_track_out_dir)

    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        save_mv_outputs(
            cfg=self.save_output_cfg,
            mv_outputs=outputs,
            meta_data=meta_data,
            out_dir=out_dir,
            output_meta_dict=output_meta_dict,
        )

    @torch.no_grad()
    def trans_onnx_normal(
        self,
        image,
        pos=None,
        pos_nodiff=None,
    ):
        h, w = image.shape[-2:]

        if not hasattr(self, "patch_h"):
            patch_h, patch_w = (
                h // self.model.rgb_encoder.patch_size,
                w // self.model.rgb_encoder.patch_size,
            )
        else:
            patch_h, patch_w = self.patch_h, self.patch_w

        self.model.fuse_encoder.normalize = True
        self.model.fuse_encoder.pretrained.__class__.get_intermediate_layers = self.model.fuse_encoder.pretrained.get_intermediate_layers_onnx
        mv_features, pos, patch_start_idx = self.model.fuse_encoder(
            image,
            pos=pos,
            pos_nodiff=pos_nodiff,
        )

        self.model.normal_head.__class__.forward = self.model.normal_head.onnx_forward
        results = self.model.normal_head(mv_features, patch_h, patch_w, patch_start_idx)
        pred_local_normal = results["normal"]
        pred_local_normal = F.normalize(pred_local_normal, dim=-3)
        return pred_local_normal

    def trans_onnx_vggt(
        self,
        image,
        camera_token,
        register_token,
        pos,
    ):
        h, w = image.shape[-2:]

        if not hasattr(self, "patch_h"):
            patch_h, patch_w = (
                h // self.model.rgb_encoder.patch_size,
                w // self.model.rgb_encoder.patch_size,
            )
        else:
            patch_h, patch_w = self.patch_h, self.patch_w

        self.model.rgb_encoder.normalize = True
        self.model.rgb_encoder.__class__.forward = self.model.rgb_encoder.onnx_forward
        rgb_features = self.model.rgb_encoder(image)

        self.model.fuse_encoder.__class__.forward = self.model.fuse_encoder.onnx_forward
        mv_features, patch_start_idx = self.model.fuse_encoder(rgb_features, 1, camera_token, register_token, pos)

        self.model.normal_head.__class__.forward = self.model.normal_head.onnx_forward
        results = self.model.normal_head(mv_features, patch_h, patch_w, patch_start_idx)
        pred_local_normal = results["normal"]
        pred_local_normal = F.normalize(pred_local_normal, dim=-3)
        return pred_local_normal

    def simple_inference(
        self,
        frame_ids,
        view_ids,
        img_list,
        lidar_list=None,
        conf_list=None,
        invalid_mask_list=None,
        extrinsics_list=None,
        intrinsics_list=None,
        chunk_size=None,
        overlap=0,
        process_res=504,
        output_dir=None,
        conf_ratio=0.2,
        save_points=False,
        **kwargs,
    ):
        from .utils.kosmo_util import simple_inference

        return simple_inference(
            self=self,
            frame_ids=frame_ids,
            view_ids=view_ids,
            img_list=img_list,
            lidar_list=lidar_list,
            conf_list=conf_list,
            invalid_mask_list=invalid_mask_list,
            extrinsics_list=extrinsics_list,
            intrinsics_list=intrinsics_list,
            chunk_size=chunk_size,
            overlap=overlap,
            process_res=process_res,
            output_dir=output_dir,
            conf_ratio=conf_ratio,
            save_points=save_points,
            **kwargs,
        )

    def simple_inference_align(
        self,
        frame_ids,
        view_ids,
        img_list,
        lidar_list=None,
        conf_list=None,
        invalid_mask_list=None,
        extrinsics_list=None,
        intrinsics_list=None,
        chunk_size=None,
        overlap=0,
        process_res=504,
        output_dir=None,
        conf_ratio=0.2,
        save_points=True,
        **kwargs,
    ):
        from .utils.kosmo_util import simple_inference_align

        return simple_inference_align(
            self=self,
            frame_ids=frame_ids,
            view_ids=view_ids,
            img_list=img_list,
            lidar_list=lidar_list,
            conf_list=conf_list,
            invalid_mask_list=invalid_mask_list,
            extrinsics_list=extrinsics_list,
            intrinsics_list=intrinsics_list,
            chunk_size=chunk_size,
            overlap=overlap,
            process_res=process_res,
            output_dir=output_dir,
            conf_ratio=conf_ratio,
            save_points=save_points,
            **kwargs,
        )

    def simple_inference_fake_sv(
        self,
        frame_ids,
        view_ids,
        img_list,
        lidar_list=None,
        conf_list=None,
        invalid_mask_list=None,
        extrinsics_list=None,
        intrinsics_list=None,
        chunk_size=None,
        overlap=0,
        process_res=504,
        output_dir=None,
        save_points=False,
        save_normal=True,
        conf_ratio=None,
        **kwargs,
    ):
        from .utils.kosmo_util import simple_inference_fake_sv

        return simple_inference_fake_sv(
            self=self,
            frame_ids=frame_ids,
            view_ids=view_ids,
            img_list=img_list,
            lidar_list=lidar_list,
            conf_list=conf_list,
            invalid_mask_list=invalid_mask_list,
            extrinsics_list=extrinsics_list,
            intrinsics_list=intrinsics_list,
            chunk_size=chunk_size,
            overlap=overlap,
            process_res=process_res,
            output_dir=output_dir,
            save_points=save_points,
            save_normal=save_normal,
            conf_ratio=conf_ratio,
            **kwargs,
        )

    def trt_inference_fake_sv(
        self,
        trt_model,
        frame_ids,
        view_ids,
        img_list,
        lidar_list=None,
        conf_list=None,
        invalid_mask_list=None,
        extrinsics_list=None,
        intrinsics_list=None,
        chunk_size=None,
        overlap=0,
        process_res=504,
        output_dir=None,
        save_points=False,
        save_normal=True,
        **kwargs,
    ):
        from .utils.kosmo_util import trt_inference_fake_sv

        return trt_inference_fake_sv(
            self=self,
            trt_model=trt_model,
            frame_ids=frame_ids,
            view_ids=view_ids,
            img_list=img_list,
            lidar_list=lidar_list,
            conf_list=conf_list,
            invalid_mask_list=invalid_mask_list,
            extrinsics_list=extrinsics_list,
            intrinsics_list=intrinsics_list,
            chunk_size=chunk_size,
            overlap=overlap,
            process_res=process_res,
            output_dir=output_dir,
            save_points=save_points,
            save_normal=save_normal,
            **kwargs,
        )

    @torch.no_grad()
    def trans_onnx(
        self,
        image,
        prompt_depth=None,
        prompt_scale=None,
        pos=None,
        pos_nodiff=None,
        pos_stage0=None,
        pos_stage1=None,
        pos_stage2=None,
        pos_stage3=None,
        pos_stage4=None,
    ):
        if image.ndim == 4:
            image = image.unsqueeze(1)
        h, w = image.shape[-2:]
        if not hasattr(self, "patch_h"):
            patch_h, patch_w = (
                h // self.model.fuse_encoder.pretrained.patch_size,
                w // self.model.fuse_encoder.pretrained.patch_size,
            )
        else:
            patch_h, patch_w = self.patch_h, self.patch_w

        if prompt_depth is not None and prompt_scale is not None:
            prompt_depth_mask = (prompt_depth[..., [-1], :, :] > 0).to(prompt_depth.dtype)
            prompt_depth = self.normalize(prompt_depth, prompt_scale)
            prompt_depth = torch.cat([prompt_depth, prompt_depth_mask], dim=-3)

        if prompt_depth is not None and self.model.depth_encoder is not None:
            prompt_features = self.model.depth_encoder(prompt_depth)
        else:
            prompt_features = None

        self.model.fuse_encoder.normalize = True
        self.model.fuse_encoder.pretrained.__class__.get_intermediate_layers = self.model.fuse_encoder.pretrained.get_intermediate_layers_onnx
        rgb_features, pos, patch_start_idx = self.model.fuse_encoder(
            image, prompt_depth=prompt_features,
            pos=pos, pos_nodiff=pos_nodiff,
        )

        self.model.depth_head.__class__.forward = self.model.depth_head.onnx_forward
        results = self.model.depth_head(
            rgb_features,
            prompt_depth=prompt_features,
            patch_h=patch_h, patch_w=patch_w,
            patch_start_idx=patch_start_idx,
            pos_stage0=pos_stage0,
            pos_stage1=pos_stage1,
            pos_stage2=pos_stage2,
            pos_stage3=pos_stage3,
            pos_stage4=pos_stage4,
        )
        if "depth" in results:
            pred_local_depth = results["depth"]
        elif "pointmap" in results:
            pred_local_depth = results["pointmap"]
        elif "points" in results:
            pred_local_depth = results["points"]
        else:
            pred_local_depth = None
        pred_local_depth = pred_local_depth[0]
        pred_local_conf = results.get("confidence")[0]
        pred_local_normal = results.get("normal")[0]
        pred_local_invalid_mask = results.get("invalid_mask")[0]
        pred_local_normal = F.normalize(pred_local_normal, dim=-3)

        if prompt_depth is not None and prompt_scale is not None:
            pred_local_depth = self.denormalize(pred_local_depth, scale=prompt_scale)

        return pred_local_depth, pred_local_conf, pred_local_normal, pred_local_invalid_mask

    @torch.no_grad()
    def trans_onnx_dino(
        self,
        image,
    ):
        h, w = image.shape[-2:]
        
        if not hasattr(self, "patch_h"):
            patch_h, patch_w = (
                h // self.model.rgb_encoder.patch_size,
                w // self.model.rgb_encoder.patch_size,
            )
        else:
            patch_h, patch_w = self.patch_h, self.patch_w

        self.model.rgb_encoder.normalize = True
        self.model.rgb_encoder.__class__.forward = self.model.rgb_encoder.onnx_forward
        rgb_features = self.model.rgb_encoder(image)

        self.model.normal_head.__class__.forward = self.model.normal_head.onnx_forward
        results = self.model.normal_head(rgb_features, patch_h, patch_w, 0)
        pred_local_normal = results["normal"]
        pred_local_normal = F.normalize(pred_local_normal, dim=-3)
        if "invalid_mask" in results:
            pred_local_invalid_mask = results["invalid_mask"]
            return pred_local_normal, pred_local_invalid_mask
        else:
            return pred_local_normal
