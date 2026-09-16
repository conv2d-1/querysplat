import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
import os

from hAlgorithm.modules.utils.ray import get_rays_in_camera_frame, get_rays_in_world_frame, recover_pinhole_intrinsics_from_ray_directions
from hAlgorithm.utils import instantiate_from_config
from hAlgorithm.modules.pipelines2.sv_v1 import SVPipeline
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

class SVFlowLocal(SVPipeline):
    def __init__(
        self,
        pose_encoding_type="absT_quaR",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pose_encoding_type=pose_encoding_type
    
    def get_inputs(self, batch):
        meta_data = batch["meta_data"]
        name = meta_data["name"][0]
        total_iter = batch.get("total_iter")

        if total_iter is not None:
            meta_data["total_iter"] = total_iter
        
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
                ray_directions = ray_directions.view(*intrinsics.shape[:2], 3, h, w)

        if self.extrinsics_name is not None and self.extrinsics_name in batch:
            extrinsics = batch[self.extrinsics_name].to(device=self.device)
            extrinsics[..., :3, 3] = self.normalize(extrinsics[..., :3, 3], scale[..., 0, 0])

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

                b, s, c, h ,w = image.shape
                image = image.reshape(b*s, c, h, w).contiguous()
                if self.moge is not None and (target_normal is None):
                    output = self.moge.infer((image + 1) * 0.5)
                    if target_normal is None:
                        target_normal = output["normal"].float().permute(0, 3, 1, 2).contiguous()
                        target_normal = target_normal.reshape(b, s, *target_normal.shape[1:]).contiguous()

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
                        target_invalid_mask = target_invalid_mask.reshape(b, s, *target_invalid_mask.shape[1:]).unsqueeze(2)
                        target_depth_mask = target_depth_mask & (~target_invalid_mask)

                        if target_invalid_mask.sum() > 0 and 0:
                            image_numpy = batch["image"].cpu().flatten(0, 1).permute(0, 2, 3, 1).contiguous().numpy()
                            for bi in range(image.shape[0]):
                                cur_rgb = ((image_numpy[bi] + 1) * 0.5 * 255).astype(np.uint8)
                                cv2.imwrite("rgb.png", cv2.cvtColor(cur_rgb, cv2.COLOR_RGB2BGR))
                                cur_mask = (target_invalid_mask[bi, 0].cpu().long().numpy() * 255).astype(np.uint8)
                                cv2.imwrite("mask.png", cur_mask)
                                breakpoint()
                image = image.reshape(b, s, c, h, w).contiguous()

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

    def get_base_loss(
        self,
        name,
        image,
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
        n = image.shape[1]

        total_loss = 0
        total_loss_dict = dict()
        
        for i in range(n):
            loss, loss_dict = super().get_base_loss(
                name=name,
                pred_depth=pred_depth[:, i] if pred_depth is not None else None,
                pred_conf=pred_conf[:, i] if pred_conf is not None else None,
                pred_normal=pred_normal[:, i] if pred_normal is not None else None,
                pred_ray_directions=pred_ray_directions[:, i] if pred_ray_directions is not None else None,
                pred_invalid_mask=pred_invalid_mask[:, i] if pred_invalid_mask is not None else None,
                target_depth=target_depth[:, i] if target_depth is not None else None,
                target_normal=target_normal[:, i] if target_normal is not None else None,
                target_normal_mask=target_normal_mask[:, i] if target_normal_mask is not None else None,
                target_ray_directions=target_ray_directions[:, i] if target_ray_directions is not None else None,
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

        pred_global_points = results.get("global_points")
        pred_global_conf = results.get("global_confidence")
        
        if self.pose_encoding_type in ["absT_quaR", "absT_quaR_FoV"]:
            pose_enc = results.get("pose_enc")
        else:
            raise NotImplementedError


        # Normalize the ray directions to unit vectors
        if pred_ray_directions is not None:
            pred_ray_directions = pred_ray_directions / pred_ray_directions.norm(dim=3, keepdim=True).clip(min=1e-8)

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

        total_loss, total_loss_dict = 0, dict()
        
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

        pred_global_points = results.get("global_points")
        pred_global_conf = results.get("global_confidence")
        
        if self.pose_encoding_type in ["absT_quaR", "absT_quaR_FoV"]:
            pose_enc = results.get("pose_enc")
        else:
            raise NotImplementedError

        # Normalize the ray directions to unit vectors
        if pred_ray_directions is not None:
            pred_ray_directions = pred_ray_directions / pred_ray_directions.norm(dim=3, keepdim=True).clip(min=1e-8)

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
        
        # DepthMap trans to PointMap
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
        
        def get_single_view_data(data, index):
            return data[:, index] if data is not None else None
        
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        mv_outputs = []
        for fi in range(frame_num):
            for vi in range(view_num):
                index = fi * view_num + vi
                single = self.postprocess(
                    pred_local_points=get_single_view_data(pred_local_depth, index),
                    pred_local_conf=get_single_view_data(pred_local_conf, index),
                    pred_local_normal=get_single_view_data(pred_local_normal, index),
                    pred_local_invalid_mask=get_single_view_data(pred_local_invalid_mask, index),
                    # pred_global_points=get_single_view_data(pred_global_points, index),
                    # pred_global_conf=get_single_view_data(pred_global_conf, index),
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
                single.rgb = (image[0, index].permute(1, 2, 0).cpu().numpy() + 1) * 0.5
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

    def visualize(self, outputs_list, meta_data, out_dir):
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        data_idx = meta_data["data_idx"][0]

        gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir = self.get_out_dir(out_dir, data_idx=data_idx)
        gt_out_dir = self.get_gt_out_dir(out_dir)
        # Images Visualization
        if gt_out_dir is not None:
            vis_images(cfg=self.save_output_cfg, mv_outputs=outputs_list, gt_out_dir=gt_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num)

        # Local results Visualization
        if self.save_output_cfg["save_local_results"]:
            os.makedirs(mv_out_dir, exist_ok=True)
            vis_local_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, mv_out_dir=mv_out_dir, frame_num=frame_num, view_num=view_num, meta_data=meta_data)
        if self.save_output_cfg["save_extra_local_results"]:
            os.makedirs(mv_out_dir, exist_ok=True)
            vis_extra_local_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, mv_out_dir=mv_out_dir, frame_num=frame_num, view_num=view_num, meta_data=meta_data)

        # Camera Visualization
        if self.save_output_cfg.get("save_cameras", False):
            vis_camera_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, camera_out_dir=camera_out_dir, frame_num=frame_num, view_num=view_num)

    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        save_mv_outputs(
            cfg=self.save_output_cfg,
            mv_outputs=outputs,
            meta_data=meta_data,
            out_dir=out_dir,
            output_meta_dict=output_meta_dict,
        )

    @torch.no_grad()
    def get_init_param(
        self,
        image,
    ):
        state_feat, state_pos = self.model.fuse_encoder._init_state(image.shape[0], None, image.device)
        return state_feat

    @torch.no_grad()
    def trans_onnx(
        self,
        image,
        prompt_depth=None,
        prompt_scale=None,
        flow_state=None,
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

        self.model.fuse_encoder.__class__.forward = self.model.fuse_encoder.forward_flow
        rgb_features, update_state = self.model.fuse_encoder(
            rgb_features, 
            None, 
            flow_state, 
            None,
            h, w
        )
        
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

        return pred_local_depth, pred_local_conf, pred_local_normal, pred_local_invalid_mask, update_state
