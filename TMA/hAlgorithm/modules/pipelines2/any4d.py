"""
Any4D Pipeline for Multi-View Reconstruction with Scene Flow.

This pipeline wraps the Any4D model and handles:
- Input preprocessing (extrinsics to quaternions conversion)
- Output postprocessing (creating ReconstructOutput with scene flow)
- Visualization and saving outputs
"""

import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.modules.models2.external.mapanything.utils.geometry import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
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
from hAlgorithm.modules.pipelines2.utils.visualize_motion import (
    vis_motion_head_results,
    vis_motion_3d_rerun,
)
from hAlgorithm.modules.utils.ray import (
    get_rays_in_camera_frame,
    get_rays_in_world_frame,
    recover_pinhole_intrinsics_from_ray_directions,
)
from hAlgorithm.utils import instantiate_from_config


class Any4DPipeline(Pipeline):
    """Pipeline for Any4D Multi-View Reconstruction with Scene Flow."""

    def __init__(
        self,
        # Input names
        intrinsics_name=None,
        extrinsics_name=None,
        scale_name=None,
        prompt_depth_name=None,
        target_local_depth_name=None,
        target_global_points_name=None,
        target_depth_mask_name=None,
        align_name=None,
        task_weight=None,
        save_output_cfg=None,
        **kwargs,
    ):
        super(Any4DPipeline, self).__init__(**kwargs)

        # Input names
        self.intrinsics_name = intrinsics_name
        self.extrinsics_name = extrinsics_name
        self.scale_name = scale_name
        self.prompt_depth_name = prompt_depth_name
        self.target_local_depth_name = target_local_depth_name
        self.target_global_points_name = target_global_points_name
        self.target_depth_mask_name = target_depth_mask_name
        self.align_name = align_name

        self.task_weight = task_weight
        if task_weight is None:
            self.task_weight = dict()

        # Default save output config
        self.save_output_cfg = dict(
            save_everything=False,
            save_output_conf=True,
            save_gaussians=True,
            save_render_results=True,
            save_glb_results=True,
            save_glb_sf_results=True,  # Enable scene flow saving
            save_glb2local_results=False,
            save_local2glb_results=True,
            save_cameras=True,
            save_local_results=False,
            save_extra_local_results=False,
            save_track_results=True,
            save_filtered_results=True,
            save_normal=True,
            save_output_only_local_glb=False,
            output_normalize_cameras=True,
            output_match_input_res=True,
            output_conf_ratio=0.2,
            output_intrinsics_from_ray=False,
            gt_out_dir="gt",
            # Motion visualization config (Movies-compatible)
            save_motion_results=True,  # 2D rainbow trail visualization
            save_motion_3d=True,  # 3D Rerun visualization
        )
        if save_output_cfg is not None:
            self.save_output_cfg.update(save_output_cfg)

        if self.save_output_cfg["save_everything"]:
            for key in self.save_output_cfg:
                if isinstance(self.save_output_cfg[key], bool) and key.startswith("save_"):
                    self.save_output_cfg[key] = True

    def get_inputs(self, batch):
        """Extract and preprocess inputs from batch."""
        meta_data = batch["meta_data"]
        name = meta_data["name"][0]
        total_iter = batch.get("total_iter")

        image = batch["image"].to(device=self.device, dtype=self.dtype)

        scale = intrinsics = ray_directions = extrinsics = camera_quats = camera_trans = prompt_depth = None

        if self.scale_name is not None and self.scale_name in batch:
            scale = batch[self.scale_name].to(self.device, self.dtype)[..., None, None, None]

        if self.intrinsics_name is not None and self.intrinsics_name in batch:
            intrinsics = batch[self.intrinsics_name].to(device=self.device)

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
            # Convert extrinsics to camera pose (c2w)
            camera_poses = extrinsics.inverse()
            rotation_matrices = camera_poses[:, :, :3, :3]
            camera_quats = rotation_matrix_to_quaternion(rotation_matrices)
            camera_trans = camera_poses[:, :, :3, 3]

        if self.prompt_depth_name is not None and self.prompt_depth_name in batch:
            prompt_depth = batch[self.prompt_depth_name].to(self.device, self.dtype)

        target_local_depth = target_global_points = target_depth_mask = None

        if self.target_local_depth_name is not None and self.target_local_depth_name in batch:
            target_local_depth = batch[self.target_local_depth_name].to(device=self.device)

        if self.target_global_points_name is not None and self.target_global_points_name in batch:
            target_global_points = batch[self.target_global_points_name].to(device=self.device)

        if self.target_depth_mask_name is not None and self.target_depth_mask_name in batch:
            target_depth_mask = batch[self.target_depth_mask_name].to(self.device)

        image_show = align_data = None

        if not self.training:
            if "image_show" in batch:
                image_show = batch["image_show"]
                image_show = image_show.float().numpy()

            if self.save_output_cfg["output_match_input_res"] and self.align_name in batch:
                align_data = batch[self.align_name]

        # Extract trajectory data for scene flow evaluation
        trajs_2d = batch.get('trajs_2d', None)
        trajs_3d = batch.get('trajs_3d', None)
        valids = batch.get('valids', None)
        visibs = batch.get('visibs', None)
        
        # Get raw extrinsics (w2c) for coordinate transformation in evaluation
        # trajs_3d is in WORLD coordinates, needs raw w2c to transform to camera frame
        motion_extrinsics = batch.get('extrinsics', None)
        if motion_extrinsics is not None:
            motion_extrinsics = motion_extrinsics.to(device=self.device)

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
            image_show,
            align_data,
            ray_directions,
            camera_quats,
            camera_trans,
            trajs_3d,
            trajs_2d,
            visibs,
            valids,
            motion_extrinsics,
        )

    def train_step(self, batch):
        """Training step placeholder."""
        total_loss, total_loss_dict = 0, dict()
        return total_loss, total_loss_dict

    def postprocess(
        self,
        pred_local_points=None,
        pred_local_conf=None,
        pred_global_points=None,
        pred_global_conf=None,
        pred_scene_flow=None,
        pred_extrinsics=None,
        pred_intrinsics=None,
        image=None,
        image_show=None,
        prompt_depth=None,
        target_local_depth=None,
        target_global_points=None,
        target_depth_mask=None,
        intrinsics=None,
        extrinsics=None,
        align_data=None,
    ):
        """Convert model outputs to ReconstructOutput format."""
        # Process colors
        if image_show is not None and pred_local_points is not None:
            points_colors = image_show[0].transpose(1, 2, 0)
        else:
            points_colors = image[0].cpu().float().numpy().transpose(1, 2, 0)
            points_colors = (points_colors + 1) * 0.5 * 255

        points_h, points_w = pred_local_points.shape[-3:-1]

        if points_h != points_colors.shape[0] or points_w != points_colors.shape[1]:
            points_colors = cv2.resize(points_colors, dsize=(points_w, points_h), interpolation=cv2.INTER_LINEAR)
        points_colors = points_colors.reshape(-1, 3)

        # Process local points
        pred_local_points = pred_local_points[0].cpu().float().numpy()
        pred_local_depth = pred_local_points[..., -1].clip(1e-3)
        pred_local_points = pred_local_points.reshape(-1, 3)

        # Process confidence
        pred_local_conf = pred_local_conf[0].cpu().float().numpy()
        conf_thresh = np.percentile(pred_local_conf.reshape(-1), self.save_output_cfg["output_conf_ratio"] * 100)
        filtered_pred_local_points = pred_local_points[pred_local_conf.reshape(-1) > conf_thresh]
        filtered_points_colors = points_colors[pred_local_conf.reshape(-1) > conf_thresh]

        # Match input resolution if needed
        if self.save_output_cfg["output_match_input_res"] and pred_local_depth is not None and align_data is not None:
            align_data = align_data[0].numpy()
            h, w = align_data.shape[:2]
            pred_local_depth = cv2.resize(pred_local_depth, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            if pred_local_conf is not None:
                pred_local_conf = cv2.resize(pred_local_conf, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

        # Process global points
        pred_global_points = pred_global_points[0].detach().cpu().float().numpy()
        pred_global_conf = pred_global_conf[0].detach().cpu().float().numpy()

        # Process scene flow
        pred_scene_flow_np = None
        if pred_scene_flow is not None:
            pred_scene_flow_np = pred_scene_flow[0].detach().cpu().float().numpy()

        # Process extrinsics and intrinsics
        pred_extrinsics = pred_extrinsics[0].detach().cpu().float().numpy()
        pred_intrinsics = pred_intrinsics[0].detach().cpu().float().numpy()

        # Process prompt depth
        if prompt_depth is not None:
            prompt_h, prompt_w = prompt_depth.shape[-2:]
            prompt_depth = prompt_depth[0].cpu().float().numpy().transpose(1, 2, 0).reshape(-1, 3)
        else:
            prompt_h = prompt_w = None

        # Process targets
        if target_local_depth is not None:
            target_local_depth = target_local_depth[0].cpu().float().numpy().reshape(3, -1).transpose(1, 0)

        if target_global_points is not None:
            target_global_points = target_global_points[0].cpu().float().numpy().reshape(3, -1).transpose(1, 0)

        if target_depth_mask is not None:
            target_depth_mask = target_depth_mask[0].cpu().numpy()

        if intrinsics is not None:
            intrinsics = intrinsics[0].cpu().numpy()

        if extrinsics is not None:
            extrinsics = extrinsics[0].cpu().numpy()

        output = ReconstructOutput(
            pointmap_color=points_colors,
            pointmap_h=points_h,
            pointmap_w=points_w,
            pointmap=pred_local_points,
            depth_align=pred_local_depth,
            confidence=pred_local_conf,
            filtered_pointmap=filtered_pred_local_points,
            filtered_pointmap_color=filtered_points_colors,
            glb_mv_pointmap=pred_global_points,
            glb_mv_confidence=pred_global_conf,
            extrinsics_pred=pred_extrinsics,
            intrinsics_pred=pred_intrinsics,
            prompt_pointmap=prompt_depth,
            prompt_h=prompt_h,
            prompt_w=prompt_w,
            pointmap_gt=target_local_depth,
            pointmap_gt_global=target_global_points,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth_mask=target_depth_mask,
        )

        # Add scene flow to output
        if pred_scene_flow_np is not None:
            output.scene_flow = pred_scene_flow_np

        return output

    @torch.no_grad()
    def infer(self, **batch):
        """Run inference with Any4D model."""
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
            image_show,
            align_data,
            ray_directions,
            camera_quats,
            camera_trans,
            trajs_3d,
            trajs_2d,
            visibs,
            valids,
            motion_extrinsics,
        ) = self.get_inputs(batch)

        # Run model
        if "use_amp" in batch and batch["use_amp"]:
            with torch.autocast("cuda", enabled=bool(batch["use_amp"]), dtype=batch["amp_dtype"]):
                results = self.model(
                    image,
                    prompt_depth=prompt_depth,
                    ray_directions=ray_directions,
                    camera_quats=camera_quats,
                    camera_trans=camera_trans,
                    meta_data=meta_data,
                )
        else:
            results = self.model(
                image,
                prompt_depth=prompt_depth,
                ray_directions=ray_directions,
                camera_quats=camera_quats,
                camera_trans=camera_trans,
                meta_data=meta_data,
            )

        # Extract predictions
        pred_local_depth = results.get("depth")
        pred_local_conf = results.get("confidence")
        pred_scene_flow = results.get("scene_flow")

        pred_ray_directions = results.get("ray")
        pred_ray_directions = pred_ray_directions / pred_ray_directions.norm(dim=2, keepdim=True).clip(min=1e-8)
        assert pred_ray_directions.ndim == 5
        
        # Debug: Check ray directions before intrinsics recovery
        logging.info(f"[DEBUG] pred_ray_directions shape: {pred_ray_directions.shape}")
        logging.info(f"[DEBUG] pred_ray_directions dtype: {pred_ray_directions.dtype}")
        logging.info(f"[DEBUG] pred_ray_directions min: {pred_ray_directions.min().item():.6f}, max: {pred_ray_directions.max().item():.6f}")
        logging.info(f"[DEBUG] pred_ray_directions has nan: {torch.isnan(pred_ray_directions).any().item()}")
        logging.info(f"[DEBUG] pred_ray_directions has inf: {torch.isinf(pred_ray_directions).any().item()}")
        # Check dz component specifically (index 2)
        dz = pred_ray_directions[..., 2, :, :]
        logging.info(f"[DEBUG] dz (ray z-component) min: {dz.min().item():.6f}, max: {dz.max().item():.6f}")
        logging.info(f"[DEBUG] dz near-zero count (|dz| < 1e-6): {(dz.abs() < 1e-6).sum().item()}")
        
        # Disable autocast for intrinsics recovery (linear solver requires float32)
        with torch.cuda.amp.autocast(enabled=False):
            pred_intrinsics = recover_pinhole_intrinsics_from_ray_directions(
                ray_directions=pred_ray_directions.view(-1, *pred_ray_directions.shape[2:]).permute(0, 2, 3, 1).contiguous().float()
            )
        pred_intrinsics = pred_intrinsics.view(*pred_ray_directions.shape[:2], *pred_intrinsics.shape[1:])

        pred_cam_quats = results.get("cam_quats")
        pred_cam_trans = results.get("cam_trans")
        rotation_matrices = quaternion_to_rotation_matrix(pred_cam_quats.view(-1, 4))
        rotation_matrices = rotation_matrices.view(*pred_cam_quats.shape[:2], 3, 3)

        # Create pose matrices
        pose_matrices = torch.eye(4, device=pred_cam_quats.device).unsqueeze(0).repeat(
            pred_cam_quats.shape[0], pred_cam_quats.shape[1], 1, 1
        )
        pose_matrices[:, :, :3, :3] = rotation_matrices
        pose_matrices[:, :, :3, 3] = pred_cam_trans
        pred_extrinsics = pose_matrices.inverse()

        pred_global_points = results.get("global_points")
        pred_global_conf = results.get("global_confidence")

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
                    pred_global_points=get_single_view_data(pred_global_points, index),
                    pred_global_conf=get_single_view_data(pred_global_conf, index),
                    pred_scene_flow=get_single_view_data(pred_scene_flow, index),
                    pred_extrinsics=get_single_view_data(pred_extrinsics, index),
                    pred_intrinsics=get_single_view_data(pred_intrinsics, index),
                    image=get_single_view_data(image, index),
                    image_show=get_single_view_data(image_show, index),
                    prompt_depth=get_single_view_data(prompt_depth, index),
                    target_local_depth=get_single_view_data(target_local_depth, index),
                    target_global_points=get_single_view_data(target_global_points, index),
                    target_depth_mask=get_single_view_data(target_depth_mask, index),
                    intrinsics=get_single_view_data(intrinsics, index),
                    extrinsics=get_single_view_data(extrinsics, index),
                    align_data=get_single_view_data(align_data, 0),
                )

                single.rgb = (image[0, index].permute(1, 2, 0).cpu().float().numpy() + 1) * 0.5

                # Store frame and view indices
                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                mv_outputs.append(single)

        # Convert Any4D scene flow to Movies-compatible format for visualization
        # Any4D: scene_flow at frame t = displacement from frame 0 to frame t
        # Movies: scene_flow_pred stored in reference frame (frame 0) with shape [num_frames, 3, H, W]
        if pred_scene_flow is not None:
            self._convert_scene_flow_to_movies_format(mv_outputs, pred_scene_flow, frame_num, view_num)

        # Add GT trajectory data for scene flow evaluation
        # These fields are required by Any4DSceneFlowEvalMetrics
        if trajs_3d is not None and trajs_2d is not None:
            mv_outputs[0].trajs_3d = trajs_3d[0].cpu()  # [num_views, num_trajs, 3]
            mv_outputs[0].trajs_2d = trajs_2d[0].cpu()  # [num_views, num_trajs, 2]
            if visibs is not None:
                mv_outputs[0].trajs_visibs = visibs[0].cpu()  # [num_views, num_trajs]
            if valids is not None:
                mv_outputs[0].trajs_valids = valids[0].cpu()  # [num_views, num_trajs]
            # Store origin resolution for coordinate scaling in evaluation
            if "origin_height" in meta_data:
                mv_outputs[0].origin_height = meta_data["origin_height"][0].item()
            if "origin_width" in meta_data:
                mv_outputs[0].origin_width = meta_data["origin_width"][0].item()
        
        # Store raw extrinsics (w2c) for evaluation coordinate transformation
        # CRITICAL: trajs_3d is in WORLD coordinates and needs raw w2c to transform to camera frame
        if motion_extrinsics is not None:
            mv_outputs[0].motion_extrinsics = motion_extrinsics[0].cpu().numpy()  # [num_views, 4, 4]

        return mv_outputs

    def _convert_scene_flow_to_movies_format(self, mv_outputs, pred_scene_flow, frame_num, view_num):
        """
        Convert Any4D scene flow to Movies-compatible format for visualization.
        
        Any4D outputs scene_flow per-frame with shape [H, W, 3] (displacement from frame 0 to frame t).
        Movies expects scene_flow_pred in reference frame (frame 0) with shape [num_frames, 3, H, W].
        
        The semantic meaning is the same:
        - Any4D: scene_flow[t] = P_t - P_0 (flow from frame 0 to frame t)
        - Movies: scene_flow_pred[t] = flow from frame 0 to frame t
        """
        # Aggregate scene flows from all frames
        # pred_scene_flow shape: [B, V, H, W, 3] where V = frame_num * view_num
        # We need to extract flows for view 0 (assuming single view) and convert to [num_frames, 3, H, W]
        
        scene_flows_list = []
        for fi in range(frame_num):
            for vi in range(view_num):
                index = fi * view_num + vi
                output = mv_outputs[index]
                
                if hasattr(output, 'scene_flow') and output.scene_flow is not None:
                    # output.scene_flow shape: [H, W, 3]
                    # Convert to [3, H, W] to match Movies format
                    sf = output.scene_flow
                    if isinstance(sf, np.ndarray):
                        sf = torch.from_numpy(sf)
                    sf = sf.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
                    scene_flows_list.append(sf)
                else:
                    # If no scene flow for this frame, append zeros
                    H, W = output.pointmap_h, output.pointmap_w
                    scene_flows_list.append(torch.zeros(3, H, W))
        
        if scene_flows_list:
            # Stack all scene flows: [num_total_views, 3, H, W]
            scene_flow_pred = torch.stack(scene_flows_list, dim=0)
            
            # Store in reference frame (frame 0, view 0) output for Movies-compatible visualization
            # The visualization functions expect scene_flow_pred in the reference frame's output
            ref_idx = 0  # frame 0, view 0
            mv_outputs[ref_idx].scene_flow_pred = scene_flow_pred
            
            # Also store origin resolution for coordinate scaling in evaluation
            # (if available from meta_data, will be set in visualize method)

    def get_out_dir(self, out_dir, data_idx=None):
        """Get output directory paths."""
        gs_out_dir = os.path.join(out_dir, "gaussians")
        glb_out_dir = os.path.join(out_dir, "glb")

        if data_idx is not None:
            camera_out_dir = os.path.join(out_dir, f"camera/{data_idx:06d}")
            mv_out_dir = os.path.join(out_dir, f"mvdepth/{data_idx:06d}")
            track_out_dir = os.path.join(out_dir, f"track/{data_idx:06d}")
            sf_out_dir = os.path.join(out_dir, f"scene_flow/{data_idx:06d}")
            motion_out_dir = os.path.join(out_dir, f"motion/{data_idx:06d}")
        else:
            camera_out_dir = os.path.join(out_dir, "camera")
            mv_out_dir = os.path.join(out_dir, "mvdepth")
            track_out_dir = os.path.join(out_dir, "track")
            sf_out_dir = os.path.join(out_dir, "scene_flow")
            motion_out_dir = os.path.join(out_dir, "motion")

        return gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir, sf_out_dir, motion_out_dir

    def get_gt_out_dir(self, out_dir):
        """Get ground truth output directory paths."""
        if self.save_output_cfg["gt_out_dir"] is None:
            return None, None, None

        abs_gt_out_dir = os.path.join(
            os.path.dirname(os.path.dirname(out_dir)),
            self.save_output_cfg["gt_out_dir"],
            os.path.basename(out_dir)
        )
        _, gt_glb_out_dir, _, _, gt_track_out_dir, _, _ = self.get_out_dir(abs_gt_out_dir, data_idx=None)

        return abs_gt_out_dir, gt_glb_out_dir, gt_track_out_dir

    def visualize(self, outputs_list, meta_data, out_dir):
        """Visualize outputs."""
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        data_idx = meta_data["data_idx"][0]

        gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir, sf_out_dir, motion_out_dir = self.get_out_dir(
            out_dir, data_idx=data_idx
        )
        gt_out_dir, gt_glb_out_dir, gt_track_out_dir = self.get_gt_out_dir(out_dir)

        # Images Visualization
        if gt_out_dir is not None:
            vis_images(
                cfg=self.save_output_cfg,
                mv_outputs=outputs_list,
                gt_out_dir=gt_out_dir,
                data_idx=data_idx,
                frame_num=frame_num,
                view_num=view_num,
            )

        # Local results Visualization
        if self.save_output_cfg["save_local_results"]:
            os.makedirs(mv_out_dir, exist_ok=True)
            vis_local_results(
                cfg=self.save_output_cfg,
                mv_outputs=outputs_list,
                mv_out_dir=mv_out_dir,
                frame_num=frame_num,
                view_num=view_num,
                meta_data=meta_data,
            )

        if self.save_output_cfg["save_extra_local_results"]:
            os.makedirs(mv_out_dir, exist_ok=True)
            vis_extra_local_results(
                cfg=self.save_output_cfg,
                mv_outputs=outputs_list,
                mv_out_dir=mv_out_dir,
                frame_num=frame_num,
                view_num=view_num,
                meta_data=meta_data,
            )

        # Global results Visualization
        if self.save_output_cfg["save_glb_results"]:
            vis_glb_results(
                cfg=self.save_output_cfg,
                mv_outputs=outputs_list,
                glb_out_dir=glb_out_dir,
                data_idx=data_idx,
                frame_num=frame_num,
                view_num=view_num,
                gt_out_dir=gt_glb_out_dir,
            )

        # Scene Flow Visualization
        if self.save_output_cfg["save_glb_sf_results"]:
            self._visualize_scene_flow(outputs_list, sf_out_dir, data_idx, frame_num, view_num)

        # Camera Visualization
        if self.save_output_cfg["save_cameras"]:
            vis_camera_results(
                cfg=self.save_output_cfg,
                mv_outputs=outputs_list,
                camera_out_dir=camera_out_dir,
                frame_num=frame_num,
                view_num=view_num,
            )

        # Motion Head Visualization (Movies-compatible format)
        if self.save_output_cfg.get("save_motion_results", False) and hasattr(outputs_list[0], 'scene_flow_pred') and outputs_list[0].scene_flow_pred is not None:
            os.makedirs(motion_out_dir, exist_ok=True)
            
            # Set origin resolution for coordinate scaling in visualization
            if "origin_height" in meta_data:
                outputs_list[0].origin_height = meta_data["origin_height"][0].item()
            if "origin_width" in meta_data:
                outputs_list[0].origin_width = meta_data["origin_width"][0].item()
            
            # 2D Rainbow Trails Visualization
            vis_motion_head_results(
                cfg=self.save_output_cfg,
                mv_outputs=outputs_list,
                motion_out_dir=motion_out_dir,
                data_idx=data_idx,
                meta_data=meta_data,
            )
            
            # 3D Rerun Visualization
            if self.save_output_cfg.get("save_motion_3d", False):
                vis_motion_3d_rerun(
                    cfg=self.save_output_cfg,
                    mv_outputs=outputs_list,
                    out_dir=out_dir,
                    data_idx=data_idx,
                    meta_data=meta_data,
                )

    def _visualize_scene_flow(self, outputs_list, sf_out_dir, data_idx, frame_num, view_num):
        """Visualize scene flow outputs."""
        os.makedirs(sf_out_dir, exist_ok=True)

        for fi in range(frame_num):
            for vi in range(view_num):
                index = fi * view_num + vi
                output = outputs_list[index]

                if hasattr(output, 'scene_flow') and output.scene_flow is not None:
                    scene_flow = output.scene_flow
                    # Save scene flow as NPY
                    sf_path = os.path.join(sf_out_dir, f"scene_flow_f{fi:03d}_v{vi:03d}.npy")
                    np.save(sf_path, scene_flow)

                    # Visualize scene flow magnitude
                    sf_magnitude = np.linalg.norm(scene_flow, axis=-1)
                    sf_magnitude_norm = (sf_magnitude - sf_magnitude.min()) / (sf_magnitude.max() - sf_magnitude.min() + 1e-8)
                    sf_magnitude_img = (sf_magnitude_norm * 255).astype(np.uint8)

                    sf_vis_path = os.path.join(sf_out_dir, f"scene_flow_mag_f{fi:03d}_v{vi:03d}.png")
                    cv2.imwrite(sf_vis_path, cv2.applyColorMap(sf_magnitude_img, cv2.COLORMAP_JET))

    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        """Save outputs to disk."""
        save_mv_outputs(
            cfg=self.save_output_cfg,
            mv_outputs=outputs,
            meta_data=meta_data,
            out_dir=out_dir,
            output_meta_dict=output_meta_dict,
        )