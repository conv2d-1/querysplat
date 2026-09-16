import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.modules.models2.external.mapanything.utils.geometry import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
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
from hAlgorithm.utils import (
    instantiate_from_config,
)


class MVFRPipeline(Pipeline):
    """Pipeline for Multi-View Feed-forward Reconstruction."""

    def __init__(
        self,
        # inputs name
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
        super(MVFRPipeline, self).__init__(**kwargs)

        # inputs name
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

        self.save_output_cfg = dict(
            save_everything=False,
            save_output_conf=True,
            save_gaussians=True,
            save_render_results=True,
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
            save_output_only_local_glb=False,
            output_normalize_cameras=True,
            output_match_input_res=True,
            output_conf_ratio=0.2,
            output_intrinsics_from_ray=False,
            # output_use_gt_scale=True,
            gt_out_dir="gt",
        )
        if save_output_cfg is not None:
            self.save_output_cfg.update(save_output_cfg)

        if self.save_output_cfg["save_everything"]:
            for key in self.save_output_cfg:
                if isinstance(self.save_output_cfg[key], bool) and key.startswith("save_"):
                    self.save_output_cfg[key] = True

    def get_inputs(self, batch):
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
            # extrinsics[..., :3, 3] = self.normalize(extrinsics[..., :3, 3], scale[..., 0, 0])

            camera_poses = extrinsics.inverse()
            rotation_matrices = camera_poses[:, :, :3, :3]
            camera_quats = rotation_matrix_to_quaternion(rotation_matrices)
            camera_trans = camera_poses[:, :, :3, 3]

        if self.prompt_depth_name is not None and self.prompt_depth_name in batch:
            prompt_depth = batch[self.prompt_depth_name].to(self.device, self.dtype)
            # prompt_depth = self.normalize(prompt_depth, scale)

        target_local_depth = target_global_points = target_depth_mask = None

        if self.target_local_depth_name is not None and self.target_local_depth_name in batch:
            target_local_depth = batch[self.target_local_depth_name].to(device=self.device)
            # target_local_depth = self.normalize(target_local_depth, scale)

        if self.target_global_points_name is not None and self.target_global_points_name in batch:
            target_global_points = batch[self.target_global_points_name].to(device=self.device)
            # target_global_points = self.normalize(target_global_points, scale)

        if self.target_depth_mask_name is not None and self.target_depth_mask_name in batch:
            target_depth_mask = batch[self.target_depth_mask_name].to(self.device)

        image_show = align_data = None

        if not self.training:
            if "image_show" in batch:
                image_show = batch["image_show"]
                image_show = image_show.float().numpy()

            if self.save_output_cfg["output_match_input_res"] and self.align_name in batch:
                align_data = batch[self.align_name]

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
        )

    def train_step(self, batch):
        total_loss, total_loss_dict = 0, dict()
        return total_loss, total_loss_dict

    def postprocess(
        self,
        pred_local_points=None,
        pred_local_conf=None,
        pred_global_points=None,
        pred_global_conf=None,
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
        # Convert the color pointmap to a NumPy array and normalize it to the [0, 255] range.
        if image_show is not None and pred_local_points is not None:
            points_colors = image_show[0].transpose(1, 2, 0)
        else:
            points_colors = image[0].cpu().float().numpy().transpose(1, 2, 0)
            points_colors = (points_colors + 1) * 0.5 * 255

        points_h, points_w = pred_local_points.shape[-3:-1]

        if points_h != points_colors.shape[0] or points_w != points_colors.shape[1]:
            points_colors = cv2.resize(points_colors, dsize=(points_w, points_h), interpolation=cv2.INTER_LINEAR)
        points_colors = points_colors.reshape(-1, 3)

        pred_local_points = pred_local_points[0].cpu().float().numpy()
        pred_local_depth = pred_local_points[..., -1].clip(1e-3)
        pred_local_points = pred_local_points.reshape(-1, 3)

        pred_local_conf = pred_local_conf[0].cpu().float().numpy()
        conf_thresh = np.percentile(pred_local_conf.reshape(-1), self.save_output_cfg["output_conf_ratio"] * 100)
        filtered_pred_local_points = pred_local_points[pred_local_conf.reshape(-1) > conf_thresh]
        filtered_points_colors = points_colors[pred_local_conf.reshape(-1) > conf_thresh]

        if self.save_output_cfg["output_match_input_res"] and pred_local_depth is not None and align_data is not None:
            align_data = align_data[0].numpy()
            h, w = align_data.shape[:2]
            pred_local_depth = cv2.resize(pred_local_depth, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

            if pred_local_conf is not None:
                pred_local_conf = cv2.resize(pred_local_conf, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

        pred_global_points = pred_global_points[0].detach().cpu().float().numpy()

        # Add multi-view confidence predictions if available
        pred_global_conf = pred_global_conf[0].detach().cpu().float().numpy()

        # Add predicted extrinsics if available
        pred_extrinsics = pred_extrinsics[0].detach().cpu().float().numpy()

        # Add predicted intrinsics if available
        pred_intrinsics = pred_intrinsics[0].detach().cpu().float().numpy()

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
            image_show,
            align_data,
            ray_directions,
            camera_quats,
            camera_trans,
        ) = self.get_inputs(batch)

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

        # Extract predictions from the model outputs
        pred_local_depth = results.get("depth")
        pred_local_conf = results.get("confidence")

        pred_ray_directions = results.get("ray")
        # Normalize the ray directions to unit vectors
        pred_ray_directions = pred_ray_directions / pred_ray_directions.norm(dim=2, keepdim=True).clip(min=1e-8)
        assert pred_ray_directions.ndim == 5
        pred_intrinsics = recover_pinhole_intrinsics_from_ray_directions(ray_directions=pred_ray_directions.view(-1, *pred_ray_directions.shape[2:]).permute(0, 2, 3, 1).contiguous())
        pred_intrinsics = pred_intrinsics.view(*pred_ray_directions.shape[:2], *pred_intrinsics.shape[1:])

        pred_cam_quats = results.get("cam_quats")
        pred_cam_trans = results.get("cam_trans")
        # Convert quaternions to rotation matrices
        rotation_matrices = quaternion_to_rotation_matrix(pred_cam_quats.view(-1, 4))  # (B, 3, 3)
        rotation_matrices = rotation_matrices.view(*pred_cam_quats.shape[:2], 3, 3)

        # Create 4x4 pose matrices
        pose_matrices = torch.eye(4, device=pred_cam_quats.device).unsqueeze(0).repeat(pred_cam_quats.shape[0], pred_cam_quats.shape[1], 1, 1)
        pose_matrices[:, :, :3, :3] = rotation_matrices
        pose_matrices[:, :, :3, 3] = pred_cam_trans

        pred_extrinsics = pose_matrices.inverse()

        pred_global_points = results.get("global_points")
        pred_global_conf = results.get("global_confidence")

        # if self.save_output_cfg["output_use_gt_scale"] and scale is not None:
        #     metric_scaling_factor = results.get("metric_scaling_factor")
        #     pred_local_depth = pred_local_depth / metric_scaling_factor.unsqueeze(-1).unsqueeze(-1) * scale

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
            vis_local_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, mv_out_dir=mv_out_dir, frame_num=frame_num, view_num=view_num, meta_data=meta_data)
        if self.save_output_cfg["save_extra_local_results"]:
            os.makedirs(mv_out_dir, exist_ok=True)
            vis_extra_local_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, mv_out_dir=mv_out_dir, frame_num=frame_num, view_num=view_num, meta_data=meta_data)

        # Global results Visualization
        if self.save_output_cfg["save_glb_results"]:
            vis_glb_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, glb_out_dir=glb_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num, gt_out_dir=gt_glb_out_dir)

        # Camera Visualization
        if self.save_output_cfg["save_cameras"]:
            vis_camera_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, camera_out_dir=camera_out_dir, frame_num=frame_num, view_num=view_num)

    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        save_mv_outputs(
            cfg=self.save_output_cfg,
            mv_outputs=outputs,
            meta_data=meta_data,
            out_dir=out_dir,
            output_meta_dict=output_meta_dict,
        )
