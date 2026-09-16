import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.modules.pipelines2.mvfr_v1 import MVFRPipeline as Pipeline
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
from hAlgorithm.utils import (
    instantiate_from_config,
)

import os, sys
sys.path.append(os.getcwd() + '/hAlgorithm/modules/models2/external/cut3r')
from hAlgorithm.modules.models2.external.cut3r.dust3r.utils.camera import (
    camera_to_pose_encoding,
    pose_encoding_to_camera,
)
from hAlgorithm.modules.models2.external.cut3r.dust3r.post_process import estimate_focal_knowing_depth

def estimate_intrinsics_with_depth(pred_local_depth: torch.Tensor):
    '''
    Input Shape: ...,C,H,W or ...,H,W,C
    '''
    if pred_local_depth.ndim == 5:
        B, S, _, _, _ = pred_local_depth.shape
        pts3ds_self = pred_local_depth.view(B*S, *pred_local_depth.shape[2:])
        squeeze_batch = False
    else:
        B = 1
        S, _, _, _ = pred_local_depth.shape
        squeeze_batch = True
        pts3ds_self = pred_local_depth
    if pts3ds_self.shape[-1] != 3:
        pts3ds_self = pts3ds_self.permute(0, 2, 3, 1)
    BS, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(BS, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")    
    intrinsics_est = (
        torch.eye(3).unsqueeze(0).repeat(BS, 1, 1)
    )  # B, 3, 3
    intrinsics_est[:, 0, 0] = focal.detach().cpu()
    intrinsics_est[:, 1, 1] = focal.detach().cpu()
    intrinsics_est[:, 0, 2] = pp[:, 0]
    intrinsics_est[:, 1, 2] = pp[:, 1]
    pred_intrinsics = intrinsics_est.view(B, S, *intrinsics_est.shape[1:])
    if squeeze_batch:
        pred_intrinsics = pred_intrinsics.squeeze(0)
    return pred_intrinsics


class SVFlowPipeline(Pipeline):
    """Pipeline for Multi-View Feed-forward Reconstruction."""

    def __init__(
        self,
        pred_metric_space=True,
        inv_pred_pose=False,
        **kwargs,
    ):
        self.pred_metric_space = pred_metric_space
        self.inv_pred_pose = inv_pred_pose
        super(SVFlowPipeline, self).__init__(**kwargs)

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

        track_query_points, track_vis, track_pos_masks = self.get_track_inputs(batch)
        results = self.model(
            image,
            scale=scale,
            prompt_depth=prompt_depth,
            intrinsics=intrinsics,
            ray_directions=ray_directions,
            w2c=extrinsics,
            ray_world=ray_world,
            query_points=track_query_points[:, 0] if track_query_points is not None else None,
            meta_data=meta_data,
        )

        # Extract predictions from the model outputs
        pred_local_depth = results.get("depth")
        pred_local_conf = results.get("confidence")
        pred_local_normal = results.get("normal")
        pred_local_invalid_mask = results.get("invalid_mask")
        pred_local_motion_mask = results.get("motion_mask")

        pred_ray_directions = results.get("ray")

        pred_global_points = results.get("global_points")
        pred_global_conf = results.get("global_confidence")

        pred_track = results.get("track")
        pred_track_vis = results.get("track_vis")
        pred_track_conf = results.get("track_confidence")

        if self.pose_encoding_type in ["absT_quaR", "absT_quaR_FoV"]:
            pose_enc = results.get("pose_enc")
        else:
            raise NotImplementedError

        gaussians = results.get("gaussians")
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
                pred_local_depth = pred_local_depth * pred_ray_directions
            else:
                pred_local_depth = self.depth_to_points(pred_local_depth, K=intrinsics)

        # DepthMap trans to PointMap
        if pred_ffgs_depth is not None and pred_ffgs_depth.shape[-3] == 1:
            if self.points_from_ray:
                pred_ffgs_depth = pred_ffgs_depth * pred_ray_directions
            else:
                pred_ffgs_depth = self.depth_to_points(pred_ffgs_depth, K=intrinsics)

        # Normalize the normal maps
        if pred_local_normal is not None:
            tgt_h, tgt_w = pred_local_depth.shape[-2:] if pred_local_depth is not None else image.shape[-2:]
            norm_h, norm_w = pred_local_normal.shape[-2:]
            if tgt_h != norm_h or tgt_w != norm_w:
                pred_local_normal = F.interpolate(pred_local_normal, (tgt_h, tgt_w), mode="bilinear", align_corners=False, antialias=False)
            pred_local_normal = F.normalize(pred_local_normal, dim=2)

        # normalize pred if in metric space
        if self.pred_metric_space:
            if pred_local_depth is not None:
                pred_local_depth = self.normalize(pred_local_depth, scale)
            if pred_global_points is not None:
                pred_global_points = self.normalize(pred_global_points, scale)

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
            valid_mask=target_depth_mask,
        )
        total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="lcl")

        loss, loss_dict = self.get_base_loss(
            name=name,
            image=image,
            coord="global",
            pred_depth=pred_global_points,
            pred_conf=pred_global_conf,
            target_depth=target_global_points,
            valid_mask=target_depth_mask,
        )
        total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="glb")

        if self.local2global_loss is not None and pose_enc is not None and pred_local_depth is not None:
            tmp_pred_depth = results["depth"]
            B, S, C, H, W = tmp_pred_depth.shape
            assert C == 1

            with torch.cuda.amp.autocast(False):
                if self.pose_encoding_type == "absT_quaR":
                    pose_matrices = pose_encoding_to_camera(
                        pose_enc.float(),
                        pose_encoding_type=self.pose_encoding_type,
                    )
                    if self.inv_pred_pose:
                        pred_extrinsics = pose_matrices.inverse()
                    else:
                        pred_extrinsics = pose_matrices
                    pred_intrinsics = estimate_intrinsics_with_depth(tmp_pred_depth)
                else:
                    raise NotImplementedError

                tmp_pred_depth = self.depth_to_points(tmp_pred_depth.float(), K=pred_intrinsics)
                tmp_pred_depth = pred_extrinsics.inverse() @ torch.cat([tmp_pred_depth, tmp_pred_depth.new_ones([B, S, 1, H, W])], dim=2).reshape(B, S, 4, -1)
                tmp_pred_depth = tmp_pred_depth.reshape(B, S, 4, H, W)[:, :, :3, :, :]

            if pred_local_conf is not None:
                tmp_pred_conf = pred_local_conf.squeeze(2).unsqueeze(-1)
            else:
                tmp_pred_conf = None

            loss = self.local2global_loss(
                name=name,
                pred_depth=tmp_pred_depth.float().permute(0, 1, 3, 4, 2),
                target_depth=target_global_points.float().permute(0, 1, 3, 4, 2),
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
            render_depth = self.depth_to_points(render_depth, K=intrinsics)

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
                    if self.pose_encoding_type == "absT_quaR":
                        pose_matrices = pose_encoding_to_camera(
                            pose_enc.float(),
                            pose_encoding_type=self.pose_encoding_type,
                        )
                        if self.inv_pred_pose:
                            pred_extrinsics = pose_matrices.inverse()
                        else:
                            pred_extrinsics = pose_matrices
                        pred_intrinsics = estimate_intrinsics_with_depth(tmp_pred_depth)
                    else:
                        raise NotImplementedError

                    tmp_pred_depth = self.depth_to_points(tmp_pred_depth.float(), K=pred_intrinsics)
                    tmp_pred_depth = pred_extrinsics.inverse() @ torch.cat([tmp_pred_depth, tmp_pred_depth.new_ones([B, S, 1, H, W])], dim=2).reshape(B, S, 4, -1)
                    tmp_pred_depth = tmp_pred_depth.reshape(B, S, 4, H, W)[:, :, :3, :, :]

                if pred_local_conf is not None:
                    tmp_pred_conf = pred_local_conf.squeeze(2).unsqueeze(-1)
                else:
                    tmp_pred_conf = None

                loss = self.local2global_loss(
                    name=name,
                    pred_depth=tmp_pred_depth.float().permute(0, 1, 3, 4, 2),
                    target_depth=target_global_points.float().permute(0, 1, 3, 4, 2),
                    valid_mask=target_depth_mask.squeeze(2),
                    pred_conf=tmp_pred_conf,
                )
                total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict=None, task_name="rcd_l2g")

        if scale is not None:
            total_loss_dict["scale"] = round(scale.max().cpu().item(), 2)

        if image is not None:
            aspect_ratio = image.shape[-1] / image.shape[-2]
            total_loss_dict["aspect_ratio"] = round(aspect_ratio, 2)
            total_loss_dict["bs"] = int(image.shape[0])
            total_loss_dict["view"] = int(image.shape[1])

        if ray_directions is not None:
            total_loss_dict["ray"] = ray_directions.max()

        return total_loss, total_loss_dict

    def postprocess(
        self,
        pred_local_points=None,
        pred_local_conf=None,
        pred_local_normal=None,
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
            pred_local_points = pred_local_points.reshape(3, -1).transpose(1, 0)

        else:
            pred_local_depth = None
            points_h, points_w = points_colors.shape[:2]
            points_colors = points_colors.reshape(-1, 3)

        if pred_local_points is not None and pred_local_conf is not None:
            pred_local_conf = pred_local_conf[0].cpu().float().numpy()
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

        if pred_global_points is not None:
            pred_global_points = pred_global_points[0].detach().cpu().float().numpy().transpose(1, 2, 0)

        # Add multi-view confidence predictions if available
        if pred_global_conf is not None:
            pred_global_conf = pred_global_conf[0].detach().cpu().float().numpy()

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
            prompt_h, prompt_w = prompt_depth.shape[-2:]
            prompt_depth = prompt_depth[0].cpu().float().numpy().transpose(1, 2, 0).reshape(-1, 3)
        else:
            prompt_h = prompt_w = None

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
            pointmap_gt_global=target_global_points,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            normal=pred_local_normal,
            normal_gt=target_normal,
            normal_mask=target_normal_mask,
            depth_mask=target_depth_mask,
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
        
        track_query_points, track_vis, _ = self.get_track_inputs(batch)

        if "use_amp" in batch and batch["use_amp"]:
            with torch.autocast("cuda", enabled=bool(batch["use_amp"]), dtype=batch["amp_dtype"]):
                results = self.model(
                    image,
                    scale=scale,
                    prompt_depth=prompt_depth,
                    intrinsics=intrinsics,
                    ray_directions=ray_directions,
                    w2c=extrinsics,
                    ray_world=ray_world,
                    query_points=track_query_points[:, 0] if track_query_points is not None else None,
                    meta_data=meta_data,
                )
        else:
            results = self.model(
                image,
                scale=scale,
                prompt_depth=prompt_depth,
                intrinsics=intrinsics,
                ray_directions=ray_directions,
                w2c=extrinsics,
                ray_world=ray_world,
                query_points=track_query_points[:, 0] if track_query_points is not None else None,
                meta_data=meta_data,
            )

        if self.pose_encoding_type in ["absT_quaR", "absT_quaR_FoV"]:
            pose_enc = results.get("pose_enc", None)
            if pose_enc is not None and isinstance(pose_enc, (list, tuple)):
                pose_enc = pose_enc[-1]
        else:
            raise NotImplementedError


        # Extract predictions from the model outputs
        pred_local_depth = results.get("depth")
        pred_local_conf = results.get("confidence")
        pred_local_normal = results.get("normal")
        pred_local_invalid_mask = results.get("invalid_mask")
        pred_local_motion_mask = results.get("motion_mask")

        pred_ray_directions = results.get("ray")
        # Pose encoding for camera extrinsics/intrinsics
        if pose_enc is not None:
            if self.pose_encoding_type == "absT_quaR":
                pose_matrices = pose_encoding_to_camera(
                    pose_enc.float(),
                    pose_encoding_type=self.pose_encoding_type,
                )
                if self.inv_pred_pose:
                    pred_extrinsics = pose_matrices.inverse() #c2w -> w2c
                else:
                    pred_extrinsics = pose_matrices
                if not self.pred_metric_space:
                    pred_extrinsics[..., :3, 3] = self.denormalize(pred_extrinsics[..., :3, 3], scale=scale[..., 0, 0])
                pred_intrinsics = estimate_intrinsics_with_depth(pred_local_depth)
            else:
                raise NotImplementedError
        else:
            pred_extrinsics, pred_intrinsics = None, None
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
                    pred_local_depth = self.depth_to_points(pred_local_depth, K=pred_intrinsics)
                else:
                    pred_local_depth = self.depth_to_points(pred_local_depth, K=intrinsics)

            if not self.pred_metric_space:
                pred_local_depth = self.denormalize(pred_local_depth, scale=scale)

        # Normalize the normal maps
        if pred_local_normal is not None:
            tgt_h, tgt_w = pred_local_depth.shape[-2:] if pred_local_depth is not None else image.shape[-2:]
            norm_h, norm_w = pred_local_normal.shape[-2:]
            if tgt_h != norm_h or tgt_w != norm_w:
                pred_local_normal = F.interpolate(pred_local_normal, (tgt_h, tgt_w), mode="bilinear", align_corners=False, antialias=False)
            pred_local_normal = F.normalize(pred_local_normal, dim=2)

        if pred_global_points is not None and not self.pred_metric_space:
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

                single.rgb = (image[0, index].permute(1, 2, 0).cpu().numpy() + 1) * 0.5

                # Add rendered RGB image if available
                if render_rgb is not None:
                    single.render_rgb = render_rgb[0, index].cpu().numpy().transpose(1, 2, 0)

                # Add rendered depth map if available
                if render_depth is not None:
                    single.render_depth = render_depth[0, index].cpu().numpy()

                # Store frame and view indices
                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                mv_outputs.append(single)

        return mv_outputs


