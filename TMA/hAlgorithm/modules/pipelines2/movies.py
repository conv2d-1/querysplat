import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from typing import Optional, Dict, List, Tuple

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
from hAlgorithm.modules.pipelines2.utils.visualize_motion import (
    vis_motion_head_results,
    vis_motion_3d_rerun,
)
from hAlgorithm.modules.pipelines2.utils.dynamic_gaussian_utils import vis_4dgs_results
from hAlgorithm.modules.pipelines2.utils.motion_utils import (
    get_ref_scale,
    get_ref_scale_flat,
    convert_pointmap_to_scene_flow,
    compute_motion_head_loss,
    attach_motion_data_to_output,
)
from hAlgorithm.modules.utils.ray import (
    get_rays_in_camera_frame,
    get_rays_in_world_frame,
    recover_pinhole_intrinsics_from_ray_directions,
)
from hAlgorithm.utils import instantiate_from_config


# =============================================================================
# Default save output configuration
# =============================================================================
_DEFAULT_SAVE_OUTPUT_CFG = dict(
    save_everything=False,
    save_output_conf=True,
    save_gaussians=True,
    save_render_results=True,
    save_render_video=False,
    save_render_video_with_normalize_c2w=True,
    save_glb_results=True,
    save_glb_sf_results=False,
    save_glb2local_results=False,
    save_local2glb_results=True,
    save_cameras=True,
    save_local_results=False,
    save_extra_local_results=False,
    save_track_results=True,
    save_motion_results=False,
    save_motion_3d=True,
    save_4dgs_results=False,
    vis_3d_prefer_gt=True,
    save_raw_motion_flow=False,
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
)


class MoviesPipeline(Pipeline):
    """Pipeline for Multi-View Feed-forward Reconstruction."""

    def __init__(
        self,
        # inputs name
        intrinsics_name=None,
        extrinsics_name=None,
        motion_extrinsics_name="extrinsics",
        scale_name=None,
        prompt_depth_name=None,
        target_local_depth_name=None,
        target_global_points_name=None,
        target_depth_mask_name=None,
        target_normal_name=None,
        target_normal_mask_name=None,
        target_motion_mask_name=None,
        target_invalid_mask_name=None,
        align_name=None,
        # extra inputs
        with_ray_directions=False,
        with_ray_in_world=False,
        with_points_normal=False,
        # local depth loss
        local_depth_l1_loss=None,
        local_depth_grad_loss=None,
        local_depth_normal_loss=None,
        local2global_loss=None,
        # local other loss
        local_normal_loss=None,
        local_ray_directions_loss=None,
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
        # dynamic Gaussian rendering loss (optional; None = off, zero cost)
        dynamic_gaussian_render_loss=None,
        task_weight=None,
        pose_encoding_type="absT_quaR_FoV",
        points_from_ray=False,
        save_output_cfg=None,
        clip_level_time_norm=False,
        **kwargs,
    ):
        super(MoviesPipeline, self).__init__(**kwargs)

        self.clip_level_time_norm = clip_level_time_norm

        # Cache motion prediction type
        self._motion_prediction_type = self._init_motion_prediction_type()

        # Input field names
        self.intrinsics_name = intrinsics_name
        self.extrinsics_name = extrinsics_name
        self.motion_extrinsics_name = motion_extrinsics_name
        self.scale_name = scale_name
        self.prompt_depth_name = prompt_depth_name
        self.target_local_depth_name = target_local_depth_name
        self.target_global_points_name = target_global_points_name
        self.target_depth_mask_name = target_depth_mask_name
        self.target_normal_name = target_normal_name
        self.target_normal_mask_name = target_normal_mask_name
        self.target_motion_mask_name = target_motion_mask_name
        self.target_invalid_mask_name = target_invalid_mask_name
        self.align_name = align_name

        # Extra inputs
        self.with_ray_directions = with_ray_directions
        self.with_ray_in_world = with_ray_in_world
        self.with_points_normal = with_points_normal

        # Initialize all loss functions
        self._init_losses(
            local_depth_l1_loss=local_depth_l1_loss,
            local_depth_grad_loss=local_depth_grad_loss,
            local_depth_normal_loss=local_depth_normal_loss,
            local2global_loss=local2global_loss,
            local_normal_loss=local_normal_loss,
            local_ray_directions_loss=local_ray_directions_loss,
            local_invalid_mask_loss=local_invalid_mask_loss,
            local_motion_mask_loss=local_motion_mask_loss,
            global_points_l1_loss=global_points_l1_loss,
            global_points_grad_loss=global_points_grad_loss,
            global_points_normal_loss=global_points_normal_loss,
            camera_loss=camera_loss,
            rc_rgb_l1_loss=rc_rgb_l1_loss,
            rc_ssim_loss=rc_ssim_loss,
            rc_lpips_loss=rc_lpips_loss,
            rc_depth_loss=rc_depth_loss,
            rc_normal_loss=rc_normal_loss,
            rc_depth_consistency_loss=rc_depth_consistency_loss,
            track_loss=track_loss,
            dynamic_gaussian_render_loss=dynamic_gaussian_render_loss,
            **kwargs,
        )

        self.pose_encoding_type = pose_encoding_type
        self.points_from_ray = points_from_ray
        self.task_weight = task_weight if task_weight is not None else dict()

        # Save output config
        self.save_output_cfg = self._init_save_output_cfg(save_output_cfg)

    # =========================================================================
    # Initialization helpers
    # =========================================================================

    def _init_motion_prediction_type(self) -> str:
        """Detect and cache the motion head prediction type."""
        motion_head = getattr(self.model, "motion_head", None)
        if motion_head is not None:
            pred_type = getattr(motion_head, "prediction_type", "scene_flow")
            logging.info(f"[MoviesPipeline] self.model.motion_head: {motion_head}")
        else:
            pred_type = "scene_flow"
        logging.info(f"[MoviesPipeline] motion_prediction_type cached as: {pred_type}")
        return pred_type

    def _init_losses(self, **loss_configs):
        """Initialize all loss modules from configs."""
        # Standard named losses
        standard_losses = [
            "local_depth_l1_loss", "local_depth_grad_loss", "local_depth_normal_loss",
            "local2global_loss", "local_normal_loss", "local_ray_directions_loss",
            "local_invalid_mask_loss", "local_motion_mask_loss",
            "global_points_l1_loss", "global_points_grad_loss", "global_points_normal_loss",
            "camera_loss",
            "rc_rgb_l1_loss", "rc_ssim_loss", "rc_lpips_loss",
            "rc_depth_loss", "rc_normal_loss", "rc_depth_consistency_loss",
            "track_loss",
            "dynamic_gaussian_render_loss",
        ]
        for name in standard_losses:
            setattr(self, name, instantiate_from_config(loss_configs.get(name)))

        # Optional kwargs-based losses
        self.motion_any4d_loss = (
            instantiate_from_config(loss_configs.get("motion_any4d_loss"))
            if "motion_any4d_loss" in loss_configs else None
        )
        self.motion_pointmap_loss = (
            instantiate_from_config(loss_configs.get("motion_pointmap_loss"))
            if "motion_pointmap_loss" in loss_configs else None
        )

    @staticmethod
    def _init_save_output_cfg(save_output_cfg: Optional[dict]) -> dict:
        """Build save output config with defaults and optional overrides."""
        cfg = dict(_DEFAULT_SAVE_OUTPUT_CFG)
        if save_output_cfg is not None:
            cfg.update(save_output_cfg)

        if cfg["save_everything"]:
            for key in cfg:
                if isinstance(cfg[key], bool) and key.startswith("save_"):
                    if key not in ["save_output_only_local_glb"]:
                        cfg[key] = True
        return cfg

    # =========================================================================
    # Trajectory batch padding
    # =========================================================================

    def _pad_traj_batch(self, trajs_2d_list, trajs_3d_list, valids_list, visibs_list):
        """
        Pad variable-length trajectories from list to batch tensor.

        Data flow:
            Dataset output per sample: [V, N_i, C]  (N_i varies per sample)
            DataLoader collate output: list of length B, each element [V, N_i, C]
            This function output:      [B, V, max_N, C] with zero-padding
        """
        if not trajs_2d_list or trajs_2d_list[0] is None:
            return None, None, None, None

        self._validate_traj_shapes(trajs_2d_list, trajs_3d_list)

        B = len(trajs_2d_list)
        V = trajs_2d_list[0].shape[0]
        max_N = max(t.shape[1] for t in trajs_2d_list)

        def fast_pad(data_list, channels=None):
            if data_list is None or data_list[0] is None:
                return None
            shape = (B, V, max_N, channels) if channels else (B, V, max_N)
            out = torch.zeros(shape, dtype=data_list[0].dtype, device=self.device)
            for i, t in enumerate(data_list):
                curr_N = t.shape[1]
                if channels:
                    out[i, :, :curr_N, :] = t.to(self.device, non_blocking=True)
                else:
                    out[i, :, :curr_N] = t.to(self.device, non_blocking=True)
            return out

        return (
            fast_pad(trajs_2d_list, 2),
            fast_pad(trajs_3d_list, 3),
            fast_pad(valids_list),
            fast_pad(visibs_list),
        )

    @staticmethod
    def _validate_traj_shapes(trajs_2d_list, trajs_3d_list):
        """Validate trajectory tensor dimensions."""
        for i, t in enumerate(trajs_2d_list):
            if t.dim() != 3:
                raise ValueError(
                    f"[Dataset Bug] trajs_2d[{i}] expected 3D [V, N, 2], got {t.dim()}D {t.shape}.\n"
                    f"Each sample should output [num_views, track_point_num, 2]."
                )
        if trajs_3d_list and trajs_3d_list[0] is not None:
            for i, t in enumerate(trajs_3d_list):
                if t.dim() != 3:
                    raise ValueError(
                        f"[Dataset Bug] trajs_3d[{i}] expected 3D [V, N, 3], got {t.dim()}D {t.shape}.\n"
                        f"Each sample should output [num_views, track_point_num, 3]."
                    )

    # =========================================================================
    # Input loading
    # =========================================================================

    def _load_batch_field(self, batch, field_name, dtype=None):
        """Load a named field from batch to device, or return None."""
        if field_name is None or field_name not in batch:
            return None
        tensor = batch[field_name].to(device=self.device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _load_scale(self, batch):
        """Load and reshape scale from batch."""
        if self.scale_name is None or self.scale_name not in batch:
            return None
        return batch[self.scale_name].to(self.device, self.dtype)[..., None, None, None]

    def _load_intrinsics_and_rays(self, batch, meta_data, intrinsics, extrinsics, scale):
        """Load intrinsics and compute ray directions if needed."""
        ray_directions = None
        ray_world = None

        if intrinsics is None:
            return intrinsics, ray_directions, ray_world

        if self.with_ray_directions:
            w = meta_data["input_width"][0].item()
            h = meta_data["input_height"][0].item()
            ray_directions = get_rays_in_camera_frame(
                intrinsics=intrinsics.reshape(-1, 3, 3),
                height=h, width=w,
                normalize_to_unit_sphere=True,
            )
            ray_directions = ray_directions.view(*intrinsics.shape[:2], 3, h, w)

        return intrinsics, ray_directions, ray_world

    def _load_extrinsics_and_world_rays(self, batch, meta_data, intrinsics, extrinsics, scale, ray_directions):
        """Load extrinsics, normalize by scale, and compute world rays if needed."""
        ray_world = None

        if extrinsics is None:
            return extrinsics, ray_world

        # Per-view normalization: extrinsics[v].T /= scale[v]
        extrinsics[..., :3, 3] = self.normalize(extrinsics[..., :3, 3], scale[..., 0, 0])

        if self.with_ray_in_world:
            w = meta_data["input_width"][0].item()
            h = meta_data["input_height"][0].item()
            ray_origins_world, ray_directions_world = get_rays_in_world_frame(
                intrinsics=intrinsics.reshape(-1, 3, 3),
                height=h, width=w,
                normalize_to_unit_sphere=True,
                camera_pose=extrinsics.reshape(-1, 4, 4).inverse(),
            )
            ray_origins_world = ray_origins_world.view(*intrinsics.shape[:2], 3, h, w)
            ray_directions_world = ray_directions_world.view(*intrinsics.shape[:2], 3, h, w)
            ray_world = torch.cat([ray_origins_world, ray_directions_world], dim=2)

        return extrinsics, ray_world

    def _load_motion_extrinsics(self, batch, scale):
        """
        Load motion extrinsics with reference-view scale normalization.

        Reference-view normalization ensures consistency with trajs_3d
        which is also normalized by scale[0].
        """
        motion_ext_name = self.motion_extrinsics_name or "extrinsics"
        if motion_ext_name not in batch:
            return None

        motion_extrinsics = batch[motion_ext_name].to(device=self.device)
        if scale is not None:
            ref_scale_flat = get_ref_scale_flat(scale)
            motion_extrinsics[..., :3, 3] = self.normalize(
                motion_extrinsics[..., :3, 3], ref_scale_flat
            )
        return motion_extrinsics

    def _load_trajectories(self, batch, scale):
        """
        Load and pad trajectory data, normalize 3D coords by reference scale.

        Returns:
            trajs_2d: [B, V, N, 2], trajs_3d: [B, V, N, 3],
            valids: [B, V, N], visibs: [B, V, N]
        """
        trajs_2d = batch.get("trajs_2d", None)
        trajs_3d = batch.get("trajs_3d", None)
        valids = batch.get("valids", None)
        visibs = batch.get("visibs", None)

        if trajs_2d is not None:
            if not isinstance(trajs_2d, list):
                raise ValueError(
                    f"[Data Format Error] trajs_2d should be a list of tensors, got {type(trajs_2d)}.\n"
                    f"Expected: list of [V, N, 2] tensors (one per batch sample).\n"
                    f"Got shape: {trajs_2d.shape if hasattr(trajs_2d, 'shape') else 'N/A'}"
                )
            trajs_2d, trajs_3d, valids, visibs = self._pad_traj_batch(
                trajs_2d, trajs_3d, valids, visibs
            )

        # Normalize trajs_3d by reference view's scale (view 0)
        if trajs_3d is not None and scale is not None:
            ref_scale = get_ref_scale(scale)
            trajs_3d = trajs_3d / ref_scale  # [B, V, N, 3] / [B, 1, 1, 1]

        return trajs_2d, trajs_3d, valids, visibs

    def get_inputs(self, batch):
        """
        Extract and normalize inputs from batch.

        ============================================================================
        SCALE NORMALIZATION STRATEGY
        ============================================================================
        scale shape: [B, V, 1, 1, 1] where each view may have its own scale.

        Per-view normalization (each view divided by its own scale):
            - extrinsics[v].T, prompt_depth[v], target_local_depth[v],
              target_global_points[v]

        Reference-view normalization (all views divided by view 0's scale):
            - trajs_3d[all views], motion_extrinsics[v].T
        ============================================================================
        """
        meta_data = batch["meta_data"]
        name = meta_data["name"][0]
        total_iter = batch.get("total_iter")

        if total_iter is not None:
            meta_data["total_iter"] = total_iter

        image = batch["image"].to(device=self.device, dtype=self.dtype)
        scale = self._load_scale(batch)

        # Intrinsics and extrinsics
        intrinsics = self._load_batch_field(batch, self.intrinsics_name)
        intrinsics, ray_directions, ray_world = self._load_intrinsics_and_rays(
            batch, meta_data, intrinsics, None, scale
        )

        extrinsics = self._load_batch_field(batch, self.extrinsics_name)
        if extrinsics is not None:
            extrinsics, ray_world = self._load_extrinsics_and_world_rays(
                batch, meta_data, intrinsics, extrinsics, scale, ray_directions
            )

        motion_extrinsics = self._load_motion_extrinsics(batch, scale)

        # Prompt depth (per-view normalization)
        prompt_depth = self._load_batch_field(batch, self.prompt_depth_name, dtype=self.dtype)
        if prompt_depth is not None:
            prompt_depth = self.normalize(prompt_depth, scale)

        # Target depths and masks (per-view normalization)
        target_local_depth = self._load_batch_field(batch, self.target_local_depth_name)
        if target_local_depth is not None:
            target_local_depth = self.normalize(target_local_depth, scale)

        target_global_points = self._load_batch_field(batch, self.target_global_points_name)
        if target_global_points is not None:
            target_global_points = self.normalize(target_global_points, scale)

        target_depth_mask = self._load_batch_field(batch, self.target_depth_mask_name)
        target_normal = self._load_batch_field(batch, self.target_normal_name)
        target_normal_mask = self._load_batch_field(batch, self.target_normal_mask_name)
        target_motion_mask = self._load_batch_field(batch, self.target_motion_mask_name)
        target_invalid_mask = self._load_batch_field(batch, self.target_invalid_mask_name)

        # Time info
        time_idx, query_times = self._parse_time_info(meta_data)

        # Trajectories (normalized by ref scale)
        trajs_2d, trajs_3d, valids, visibs = self._load_trajectories(batch, scale)

        # Inference-only data
        image_show = align_data = None
        if not self.training:
            if "image_show" in batch:
                image_show = batch["image_show"].float().cpu().numpy()
            if self.save_output_cfg["output_match_input_res"] and self.align_name in batch:
                align_data = batch[self.align_name]

        return (
            name, total_iter, meta_data, image, intrinsics, extrinsics, scale, prompt_depth,
            target_local_depth, target_global_points, target_depth_mask, target_normal,
            target_normal_mask, target_motion_mask, target_invalid_mask, image_show, align_data,
            ray_directions, ray_world, time_idx, query_times, trajs_2d, trajs_3d, valids, visibs,
            motion_extrinsics,
        )

    def get_track_inputs(self, batch):
        """Load tracking inputs from batch."""
        track_query_points = batch.get("track_query_points")
        track_vis = batch.get("track_vis")
        track_pos_masks = batch.get("track_pos_masks")

        if track_query_points is not None:
            track_query_points = track_query_points.to(device=self.device, dtype=self.dtype)
        if track_vis is not None:
            track_vis = track_vis.to(device=self.device)
        if track_pos_masks is not None:
            track_pos_masks = track_pos_masks.to(device=self.device)

        return track_query_points, track_vis, track_pos_masks

    def _parse_time_info(self, meta_data, max_frames=1000):
        """Parse time information from metadata and normalize to [0, 1] range.

        Two normalization modes controlled by ``self.clip_level_time_norm``:
          - False (default): video-level — ``frame_id / total_frames_in_video``
          - True:  clip-level  — normalize frame_id by the full clip window
            (``clip_start_frame_id`` .. ``clip_end_frame_id``), so the time
            value reflects position within the *clip_maxlen*-frame clip, not
            just the randomly selected *view_num* frames.
        """
        if "data_info" not in meta_data:
            return None, None

        data_info = meta_data["data_info"]
        batch_size = len(data_info)
        num_frames = len(data_info[0]) if batch_size > 0 else 0

        if num_frames == 0:
            return None, None

        if self.clip_level_time_norm:
            clip_starts = meta_data.get("clip_start_frame_id")
            clip_ends = meta_data.get("clip_end_frame_id")
            assert clip_starts is not None and clip_ends is not None, (
                "clip_level_time_norm=True requires clip_start_frame_id and "
                "clip_end_frame_id in meta_data (provided by BaseTrackDataset)"
            )

            time_idx_list = []
            for batch_idx in range(batch_size):
                raw_ids = [float(fd.get("frame_id", 0)) for fd in data_info[batch_idx]]
                cs = float(clip_starts[batch_idx]) if hasattr(clip_starts, '__getitem__') else float(clip_starts)
                ce = float(clip_ends[batch_idx]) if hasattr(clip_ends, '__getitem__') else float(clip_ends)
                span = ce - cs
                if span < 1e-6:
                    normalized = [0.5] * len(raw_ids)
                else:
                    normalized = [(fid - cs) / span for fid in raw_ids]
                time_idx_list.append(normalized)
        else:
            actual_max_frames = meta_data.get("max_frames", max_frames)
            if isinstance(actual_max_frames, torch.Tensor):
                actual_max_frames = actual_max_frames.flatten()[0].item()
            elif not isinstance(actual_max_frames, (int, float)):
                try:
                    actual_max_frames = int(actual_max_frames[0])
                except (TypeError, IndexError):
                    actual_max_frames = max_frames

            time_idx_list = []
            for batch_idx in range(batch_size):
                frame_ids = [
                    float(frame_data.get("frame_id", 0)) / actual_max_frames
                    for frame_data in data_info[batch_idx]
                ]
                time_idx_list.append(frame_ids)

        time_idx = torch.tensor(time_idx_list, dtype=torch.float32, device=self.device)
        meta_data["time_idx"] = time_idx

        query_times = time_idx[:, :1].clone()
        return time_idx, query_times

    # =========================================================================
    # Input splitting for novel views
    # =========================================================================

    def split_inputs(self, novel_view_nums, image, intrinsics, extrinsics, scale, prompt_depth,
                     target_local_depth, target_global_points, target_depth_mask, target_normal,
                     target_normal_mask, target_motion_mask, target_invalid_mask, ray_directions,
                     ray_world, meta_data):
        """Split inputs into main and novel views."""
        def split(data):
            if data is None:
                return None, None
            return data[:, :-novel_view_nums].contiguous(), data[:, -novel_view_nums:].contiguous()

        image, novel_image = split(image)
        intrinsics, novel_intrinsics = split(intrinsics)
        extrinsics, novel_extrinsics = split(extrinsics)
        scale, _ = split(scale)
        prompt_depth, _ = split(prompt_depth)
        target_local_depth, novel_target_local_depth = split(target_local_depth)
        target_global_points, _ = split(target_global_points)
        target_depth_mask, novel_target_depth_mask = split(target_depth_mask)
        target_normal, novel_target_normal = split(target_normal)
        target_normal_mask, novel_target_normal_mask = split(target_normal_mask)
        target_motion_mask, _ = split(target_motion_mask)
        target_invalid_mask, _ = split(target_invalid_mask)
        ray_directions, _ = split(ray_directions)
        ray_world, _ = split(ray_world)

        if meta_data is not None:
            if meta_data["views"][0] == 1:
                meta_data["frames"] = meta_data["frames"] - novel_view_nums
            elif meta_data["frames"][0] == 1:
                meta_data["views"] = meta_data["views"] - novel_view_nums
            else:
                raise NotImplementedError

        return (
            image, intrinsics, extrinsics, scale, prompt_depth, target_local_depth,
            target_global_points, target_depth_mask, target_normal, target_normal_mask,
            target_motion_mask, target_invalid_mask, ray_directions, ray_world, meta_data,
            novel_image, novel_intrinsics, novel_extrinsics, novel_target_local_depth,
            novel_target_depth_mask, novel_target_normal, novel_target_normal_mask,
        )

    # =========================================================================
    # Loss computation
    # =========================================================================

    def add_loss(self, total_loss, total_loss_dict, loss, loss_dict=None,
                 task_name=None, loss_name=None, prefix=""):
        """Add loss to total with optional task weighting."""
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
                if loss_dict is not None:
                    if task_name not in total_loss_dict:
                        total_loss_dict[task_name] = 0
                    for key, val in loss_dict.items():
                        total_loss_dict[f"{task_name}_{key}"] = val * weight
                        if not str(key).startswith("_"):
                            total_loss_dict[task_name] += val * weight
                else:
                    total_loss_dict[task_name] = (
                        round(loss.item() * weight, 6) if hasattr(loss, "item") else loss * weight
                    )
        else:
            if isinstance(loss, dict):
                total_loss += sum(loss.values())
                total_loss_dict.update({prefix + k: v for k, v in loss.items()})
            else:
                total_loss += loss
                total_loss_dict[prefix + loss_name] = loss
        return total_loss, total_loss_dict

    @staticmethod
    def _reshape_pred(tensor, squeeze_channel=False, permute_channels=False):
        """Reshape prediction tensor: squeeze channel dim or permute BCHW -> BHWC."""
        if tensor is None:
            return None
        tensor = tensor.float()
        if squeeze_channel and tensor.shape[1] == 1:
            return tensor.squeeze(1).unsqueeze(-1)
        if permute_channels:
            return tensor.permute(0, 2, 3, 1).contiguous()
        return tensor

    @staticmethod
    def _reshape_depth(tensor):
        """Reshape depth/points tensor to BHWC format."""
        if tensor is None:
            return None
        tensor = tensor.float()
        if tensor.shape[1] == 1:
            return tensor.squeeze(1).unsqueeze(-1)
        return tensor.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _squeeze_mask(tensor):
        """Squeeze channel dim from mask tensor."""
        if tensor is None:
            return None
        return tensor.squeeze(1)

    def get_single_base_loss(self, name, coord, pred_depth=None, pred_conf=None, pred_normal=None,
                             pred_ray_directions=None, pred_motion_mask=None, pred_invalid_mask=None,
                             target_depth=None, target_normal=None, target_normal_mask=None,
                             target_ray_directions=None, target_motion_mask=None, target_invalid_mask=None,
                             valid_mask=None, scale=None, **kwargs):
        """Compute base loss for a single view."""
        # Reshape predictions
        pred_depth = self._reshape_depth(pred_depth)
        pred_conf = self._reshape_pred(pred_conf, squeeze_channel=True)
        pred_normal = self._reshape_pred(pred_normal, permute_channels=True)
        pred_ray_directions = self._reshape_pred(pred_ray_directions, permute_channels=True)
        pred_motion_mask = self._squeeze_mask(pred_motion_mask)
        pred_invalid_mask = self._squeeze_mask(pred_invalid_mask)

        # Reshape targets
        target_depth = self._reshape_depth(target_depth)
        target_normal = self._reshape_pred(target_normal, permute_channels=True)
        target_ray_directions = self._reshape_pred(target_ray_directions, permute_channels=True)
        target_motion_mask = self._squeeze_mask(target_motion_mask)
        target_invalid_mask = self._squeeze_mask(target_invalid_mask)
        valid_mask = self._squeeze_mask(valid_mask)

        total_loss = 0
        total_loss_dict = dict()

        # Select loss functions based on coordinate system
        if coord == "local":
            depth_l1_loss_func = self.local_depth_l1_loss
            depth_grad_loss_func = self.local_depth_grad_loss
            depth_normal_loss_func = self.local_depth_normal_loss
        else:
            depth_l1_loss_func = self.global_points_l1_loss
            depth_grad_loss_func = self.global_points_grad_loss
            depth_normal_loss_func = self.global_points_normal_loss

        # Depth L1 loss
        if (depth_l1_loss_func is not None
                and target_depth is not None
                and pred_depth is not None and pred_depth.requires_grad):
            loss = depth_l1_loss_func(
                name=name, pred_depth=pred_depth, pred_conf=pred_conf,
                target_depth=target_depth, valid_mask=valid_mask,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, loss=loss, loss_name="l1_loss"
            )

        # Depth gradient loss
        if (depth_grad_loss_func is not None
                and target_depth is not None
                and pred_depth is not None and pred_depth.requires_grad):
            loss = depth_grad_loss_func(
                name=name, pred_depth=pred_depth[..., -1],
                target_depth=target_depth[..., -1], valid_mask=valid_mask,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, loss=loss, loss_name="grad_loss"
            )

        # Depth normal loss
        if (depth_normal_loss_func is not None
                and pred_depth is not None and pred_depth.requires_grad):
            loss = depth_normal_loss_func(
                name=name, pred_depth=pred_depth, target_depth=target_depth,
                target_normal=target_normal, valid_mask=valid_mask,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, loss=loss, loss_name="normal_loss"
            )

        # Local-specific losses
        if coord == "local":
            total_loss, total_loss_dict = self._compute_local_specific_losses(
                name, total_loss, total_loss_dict,
                pred_normal=pred_normal, pred_ray_directions=pred_ray_directions,
                pred_motion_mask=pred_motion_mask, pred_invalid_mask=pred_invalid_mask,
                target_depth=target_depth, target_normal=target_normal,
                target_normal_mask=target_normal_mask, target_ray_directions=target_ray_directions,
                target_motion_mask=target_motion_mask, target_invalid_mask=target_invalid_mask,
                valid_mask=valid_mask,
            )

        return total_loss, total_loss_dict

    def _compute_local_specific_losses(self, name, total_loss, total_loss_dict, *,
                                        pred_normal, pred_ray_directions,
                                        pred_motion_mask, pred_invalid_mask,
                                        target_depth, target_normal, target_normal_mask,
                                        target_ray_directions, target_motion_mask,
                                        target_invalid_mask, valid_mask):
        """Compute local coordinate-specific losses (normal, ray, masks)."""
        if (self.local_normal_loss is not None
                and pred_normal is not None and pred_normal.requires_grad):
            mask = target_normal_mask if target_normal_mask is not None else valid_mask
            loss = self.local_normal_loss(
                name=name, pred_normal=pred_normal, target_depth=target_depth,
                target_normal=target_normal, valid_mask=mask,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, loss=loss, loss_name="pd_normal_loss"
            )

        if (self.local_ray_directions_loss is not None
                and target_ray_directions is not None
                and pred_ray_directions is not None and pred_ray_directions.requires_grad):
            loss = self.local_ray_directions_loss(
                name=name, pred_ray_directions=pred_ray_directions,
                target_ray_directions=target_ray_directions,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, loss=loss, loss_name="pd_ray_loss"
            )

        if (self.local_invalid_mask_loss is not None
                and target_invalid_mask is not None
                and pred_invalid_mask is not None and pred_invalid_mask.requires_grad):
            loss = self.local_invalid_mask_loss(
                name=name, pred_invalid_mask=pred_invalid_mask,
                target_invalid_mask=target_invalid_mask, valid_mask=None,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, loss=loss, loss_name="pd_inv_mask_loss"
            )

        if (self.local_motion_mask_loss is not None
                and target_motion_mask is not None
                and pred_motion_mask is not None and pred_motion_mask.requires_grad):
            loss = self.local_motion_mask_loss(
                name=name, pred_motion_mask=pred_motion_mask,
                target_motion_mask=target_motion_mask, valid_mask=None,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, loss=loss, loss_name="pd_mot_mask_loss"
            )

        return total_loss, total_loss_dict

    def get_base_loss(self, name, image, coord="local", pred_depth=None, pred_conf=None,
                      pred_normal=None, pred_ray_directions=None, pred_motion_mask=None,
                      pred_invalid_mask=None, target_depth=None, target_normal=None,
                      target_normal_mask=None, target_ray_directions=None, target_motion_mask=None,
                      target_invalid_mask=None, valid_mask=None, scale=None):
        """Compute base loss across all views."""
        n = image.shape[1]
        total_loss = 0
        total_loss_dict = dict()

        def _get_view(data, i):
            return data[:, i] if data is not None else None

        for i in range(n):
            loss, loss_dict = self.get_single_base_loss(
                name=name, coord=coord,
                pred_depth=_get_view(pred_depth, i),
                pred_conf=_get_view(pred_conf, i),
                pred_normal=_get_view(pred_normal, i),
                pred_ray_directions=_get_view(pred_ray_directions, i),
                pred_motion_mask=_get_view(pred_motion_mask, i),
                pred_invalid_mask=_get_view(pred_invalid_mask, i),
                target_depth=_get_view(target_depth, i),
                target_normal=_get_view(target_normal, i),
                target_normal_mask=_get_view(target_normal_mask, i),
                target_ray_directions=_get_view(target_ray_directions, i),
                target_motion_mask=_get_view(target_motion_mask, i),
                target_invalid_mask=_get_view(target_invalid_mask, i),
                valid_mask=_get_view(valid_mask, i),
                scale=_get_view(scale, i),
            )
            total_loss += loss / n
            for key, val in loss_dict.items():
                total_loss_dict[key] = total_loss_dict.get(key, 0) + val / n

        return total_loss, total_loss_dict

    def get_track_loss(self, name, pred_track, pred_track_vis, pred_track_conf,
                       track_query_points, track_vis, track_pos_masks, image_hw):
        """Compute tracking loss."""
        return self.track_loss(
            name=name,
            pred_tracks=(
                [d.float() for d in pred_track]
                if isinstance(pred_track, (list, tuple))
                else pred_track.float()
            ),
            vis_preds=pred_track_vis.float(),
            conf_preds=pred_track_conf.float(),
            track_gt=track_query_points.float(),
            valids=track_pos_masks.float(),
            vis=track_vis.float(),
            image_hw=image_hw,
        )

    def get_ffgs_loss(self, name, render_rgb=None, render_depth=None, render_normal=None,
                      target_rgb=None, target_depth=None, target_depth_mask=None, **kwargs):
        """Compute feed-forward Gaussian splatting loss."""
        # Prepare renders
        if render_rgb is not None:
            render_rgb = render_rgb.float()
        if render_depth is not None:
            render_depth = (render_depth.float().permute(0, 1, 3, 4, 2).contiguous()
                            .reshape(-1, *render_depth.shape[-3:]))
        if render_normal is not None:
            render_normal = render_normal.float().reshape(-1, *render_normal.shape[-3:])

        # Prepare targets
        if target_rgb is not None:
            target_rgb = (target_rgb.float() + 1) * 0.5
        if target_depth is not None:
            target_depth = (target_depth.float().permute(0, 1, 3, 4, 2).contiguous()
                            .reshape(-1, *target_depth.shape[-3:]))
        if target_depth_mask is not None:
            target_depth_mask = (target_depth_mask.squeeze(2).bool()
                                 .reshape(-1, *target_depth_mask.shape[-2:]))

        total_loss, total_loss_dict = 0, dict()

        # RGB losses
        for loss_func, loss_key in [
            (self.rc_rgb_l1_loss, "l1_loss"),
            (self.rc_ssim_loss, "ssim_loss"),
            (self.rc_lpips_loss, "lpips_loss"),
        ]:
            if (loss_func is not None
                    and target_rgb is not None
                    and render_rgb is not None and render_rgb.requires_grad):
                loss = loss_func(name=name, rgbs=target_rgb, render_rgbs=render_rgb, mask=None)
                total_loss += loss
                total_loss_dict[loss_key] = loss

        # Depth loss
        if (self.rc_depth_loss is not None
                and target_depth is not None
                and render_depth is not None and render_depth.requires_grad):
            loss = self.rc_depth_loss(
                name=name, pred_depth=render_depth, pred_conf=None,
                target_depth=target_depth, valid_mask=target_depth_mask,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict,
                loss=loss["l1_loss"] if isinstance(loss, dict) else loss,
                loss_name="dpt_loss",
            )

        # Normal loss
        if (self.rc_normal_loss is not None
                and target_depth is not None
                and render_normal is not None and render_normal.requires_grad):
            loss = self.rc_normal_loss(
                name=name, target_depth=target_depth, prediction=render_normal,
                mask=target_depth_mask,
            )
            total_loss += loss
            total_loss_dict["norm_loss"] = loss

        return total_loss, total_loss_dict

    # =========================================================================
    # Local-to-global consistency loss (shared between local and ffgs depth)
    # =========================================================================

    def _compute_local2global_loss(self, name, raw_depth, pose_enc, pred_local_conf,
                                    target_global_points, target_depth_mask):
        """
        Compute local-to-global consistency loss.

        Transforms predicted local depth to global coordinates using predicted
        camera parameters and compares with target global points.
        """
        B, S, C, H, W = raw_depth.shape

        with torch.amp.autocast("cuda", enabled=False):
            pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                pose_encoding=(
                    pose_enc[-1].float()
                    if isinstance(pose_enc, (list, tuple))
                    else pose_enc.float()
                ),
                image_size_hw=(H, W),
                build_intrinsics=True,
            )
            tmp_pred_depth = self.depth_to_points(raw_depth.float(), K=pred_intrinsics)
            tmp_pred_depth = pred_extrinsics.inverse() @ torch.cat(
                [tmp_pred_depth, tmp_pred_depth.new_ones([B, S, 1, H, W])], dim=2
            ).reshape(B, S, 4, -1)
            tmp_pred_depth = tmp_pred_depth.reshape(B, S, 4, H, W)[:, :, :3, :, :]

        tmp_pred_conf = (
            pred_local_conf.squeeze(2).unsqueeze(-1)
            if pred_local_conf is not None
            else None
        )
        return self.local2global_loss(
            name,
            tmp_pred_depth.float().permute(0, 1, 3, 4, 2).contiguous(),
            target_global_points.float().permute(0, 1, 3, 4, 2).contiguous(),
            target_depth_mask.squeeze(2),
            tmp_pred_conf,
        )

    # =========================================================================
    # Prediction normalization helpers (shared between train and infer)
    # =========================================================================

    def _normalize_ray_directions(self, pred_ray_directions):
        """Normalize predicted ray directions to unit vectors."""
        if pred_ray_directions is None:
            return None
        return pred_ray_directions / pred_ray_directions.norm(dim=2, keepdim=True).clip(min=1e-8)

    def _convert_depth_to_pointmap(self, depth, pred_ray_directions, intrinsics):
        """Convert single-channel depth to 3-channel pointmap."""
        if depth is None or depth.shape[-3] != 1:
            return depth
        if self.points_from_ray:
            return depth * pred_ray_directions
        return self.depth_to_points(depth, K=intrinsics)

    def _normalize_pred_normal(self, pred_normal, target_shape_hw):
        """Resize and normalize predicted normals to match target shape."""
        if pred_normal is None:
            return None
        tgt_h, tgt_w = target_shape_hw
        if pred_normal.shape[-2:] != (tgt_h, tgt_w):
            pred_normal = F.interpolate(
                pred_normal, (tgt_h, tgt_w), mode="bilinear", align_corners=False
            )
        return F.normalize(pred_normal, dim=-3)

    # =========================================================================
    # Forward pass (shared between train_step and infer)
    # =========================================================================

    def _run_model_forward(self, image, scale, prompt_depth, intrinsics, ray_directions,
                           extrinsics, ray_world, track_query_points, meta_data,
                           query_times, time_idx, novel_intrinsics=None, novel_extrinsics=None,
                           use_amp=False, amp_dtype=None):
        """Run model forward pass with optional AMP."""
        query_points = track_query_points[:, 0] if track_query_points is not None else None

        model_kwargs = dict(
            scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics,
            ray_directions=ray_directions, w2c=extrinsics, ray_world=ray_world,
            query_points=query_points, meta_data=meta_data,
            time_idx=time_idx, query_times=query_times,
        )

        if novel_intrinsics is not None:
            model_kwargs["novel_intrinsics"] = novel_intrinsics
            model_kwargs["novel_w2c"] = novel_extrinsics

        if use_amp:
            with torch.autocast("cuda", enabled=True, dtype=amp_dtype):
                return self.model(image, **model_kwargs)
        return self.model(image, **model_kwargs)

    # =========================================================================
    # Training step
    # =========================================================================

    def train_step(self, batch):
        """Training step - compute all losses."""
        self.train()

        # Get inputs
        (name, total_iter, meta_data, image, intrinsics, extrinsics, scale, prompt_depth,
         target_local_depth, target_global_points, target_depth_mask, target_normal,
         target_normal_mask, target_motion_mask, target_invalid_mask, image_show, align_data,
         ray_directions, ray_world, time_idx, query_times, trajs_2d, trajs_3d, valids, visibs,
         motion_extrinsics) = self.get_inputs(batch)

        track_query_points, track_vis, track_pos_masks = self.get_track_inputs(batch)

        # Handle novel views
        novel_view_nums = int(meta_data.get("novel_view_nums", [0])[0])
        novel_intrinsics = novel_extrinsics = None
        if novel_view_nums > 0:
            (image, intrinsics, extrinsics, scale, prompt_depth, target_local_depth,
             target_global_points, target_depth_mask, target_normal, target_normal_mask,
             target_motion_mask, target_invalid_mask, ray_directions, ray_world, meta_data,
             novel_image, novel_intrinsics, novel_extrinsics, novel_target_local_depth,
             novel_target_depth_mask, novel_target_normal, novel_target_normal_mask) = self.split_inputs(
                novel_view_nums, image, intrinsics, extrinsics, scale, prompt_depth,
                target_local_depth, target_global_points, target_depth_mask, target_normal,
                target_normal_mask, target_motion_mask, target_invalid_mask, ray_directions,
                ray_world, meta_data,
            )
            assert track_query_points is None

        # Forward pass
        results = self._run_model_forward(
            image, scale, prompt_depth, intrinsics, ray_directions, extrinsics,
            ray_world, track_query_points, meta_data, query_times, time_idx,
            novel_intrinsics=novel_intrinsics, novel_extrinsics=novel_extrinsics,
        )

        # Extract predictions
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
        pose_enc = results.get("pose_enc") if self.pose_encoding_type == "absT_quaR_FoV" else None
        gaussians = results.get("ffgs_gaussians")
        render_rgb = results.get("ffgs_render_rgb")
        render_depth = results.get("ffgs_render_depth")
        render_normal = results.get("ffgs_render_normal")
        pred_ffgs_depth = results.get("ffgs_depth")
        pred_ffgs_conf = results.get("ffgs_confidence")
        pred_scene_flow = results.get("scene_flow", None)
        pred_dynamic_pointmap = results.get("dynamic_pointmap", None)
        motion_prediction_type = getattr(self, "_motion_prediction_type", "scene_flow")

        total_loss, total_loss_dict = 0, dict()

        # Normalize ray directions
        pred_ray_directions = self._normalize_ray_directions(pred_ray_directions)

        # Convert depth to pointmap
        pred_local_depth = self._convert_depth_to_pointmap(
            pred_local_depth, pred_ray_directions, intrinsics
        )
        pred_ffgs_depth = self._convert_depth_to_pointmap(
            pred_ffgs_depth, pred_ray_directions, intrinsics
        )

        # Normalize normal maps
        if pred_local_normal is not None:
            tgt_hw = pred_local_depth.shape[-2:] if pred_local_depth is not None else image.shape[-2:]
            pred_local_normal = self._normalize_pred_normal(pred_local_normal, tgt_hw)

        # --- Local depth loss ---
        loss, loss_dict = self.get_base_loss(
            name, image, "local", pred_local_depth, pred_local_conf, pred_local_normal,
            pred_ray_directions, pred_local_motion_mask, pred_local_invalid_mask,
            target_local_depth, target_normal, target_normal_mask, ray_directions,
            target_motion_mask, target_invalid_mask, target_depth_mask, scale,
        )
        total_loss, total_loss_dict = self.add_loss(
            total_loss, total_loss_dict, loss, loss_dict, task_name="lcl"
        )

        # --- Global points loss ---
        loss, loss_dict = self.get_base_loss(
            name, image, "global", pred_global_points, pred_global_conf,
            target_depth=target_global_points, valid_mask=target_depth_mask,
        )
        total_loss, total_loss_dict = self.add_loss(
            total_loss, total_loss_dict, loss, loss_dict, task_name="glb"
        )

        # --- Local-to-global consistency loss ---
        if self.local2global_loss is not None and pose_enc is not None and pred_local_depth is not None:
            loss = self._compute_local2global_loss(
                name, results["depth"], pose_enc, pred_local_conf,
                target_global_points, target_depth_mask,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, loss, task_name="l2g"
            )

        # --- Camera loss ---
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
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, loss, loss_dict, task_name="cm"
            )

        # --- Track loss ---
        if self.track_loss is not None and pred_track is not None and pred_track[0].requires_grad:
            loss, loss_dict = self.get_track_loss(
                name, pred_track, pred_track_vis, pred_track_conf,
                track_query_points, track_vis, track_pos_masks, image.shape[-2:],
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, loss, loss_dict, task_name="tk"
            )

        # --- Render depth preprocessing ---
        if render_depth is not None:
            render_depth = render_depth.unsqueeze(2)
            if hasattr(gaussians, "norm_scales") and gaussians.norm_scales is not None:
                render_depth = render_depth * gaussians.norm_scales[:, None, None, None, None]
            render_depth = self.depth_to_points(render_depth, K=intrinsics)

        # --- Rendering loss ---
        loss, loss_dict = self.get_ffgs_loss(
            name, render_rgb, render_depth, render_normal,
            image, target_local_depth, target_depth_mask,
        )
        total_loss, total_loss_dict = self.add_loss(
            total_loss, total_loss_dict, loss, loss_dict, task_name="rc"
        )

        # --- FFGS depth loss ---
        if pred_ffgs_depth is not None:
            loss, loss_dict = self.get_base_loss(
                name, image, "local", pred_ffgs_depth, pred_ffgs_conf,
                target_depth=target_local_depth, valid_mask=target_depth_mask,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, loss, loss_dict, task_name="rcd"
            )

            # FFGS local-to-global
            if self.local2global_loss is not None and pose_enc is not None:
                loss = self._compute_local2global_loss(
                    name, results["ffgs_depth"], pose_enc, pred_local_conf,
                    target_global_points, target_depth_mask,
                )
                total_loss, total_loss_dict = self.add_loss(
                    total_loss, total_loss_dict, loss, task_name="rcd_l2g"
                )

            # Depth consistency
            if self.rc_depth_consistency_loss is not None and render_depth is not None:
                B, S, C, H, W = render_depth.shape
                loss = self.rc_depth_consistency_loss(
                    name,
                    render_depth.float().permute(0, 1, 3, 4, 2).contiguous(),
                    pred_ffgs_depth.float().permute(0, 1, 3, 4, 2).contiguous(),
                    render_depth.new_ones([B, S, H, W]).bool(),
                )
                total_loss, total_loss_dict = self.add_loss(
                    total_loss, total_loss_dict, loss, task_name="rcd_consis"
                )

        # --- Motion loss ---
        if pred_scene_flow is not None or pred_dynamic_pointmap is not None:
            original_h = meta_data["origin_height"][0].item()
            original_w = meta_data["origin_width"][0].item()
            motion_ref_frame_idx = results.get("src_frame_idx", 0)
            loss, loss_dict = compute_motion_head_loss(
                name=name,
                pred_scene_flow=pred_scene_flow,
                pred_dynamic_pointmap=pred_dynamic_pointmap,
                motion_prediction_type=motion_prediction_type,
                trajs_3d=trajs_3d, trajs_2d=trajs_2d,
                visibs=visibs, valids=valids,
                original_h=original_h, original_w=original_w,
                scale=scale, extrinsics=motion_extrinsics,
                motion_any4d_loss=self.motion_any4d_loss,
                motion_pointmap_loss=self.motion_pointmap_loss,
                device=self.device,
                intrinsics=intrinsics,
                target_depth=target_local_depth,
                target_motion_mask=target_motion_mask,
                ref_frame_idx=motion_ref_frame_idx,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, loss, loss_dict, task_name="motion"
            )

        # --- Dynamic Gaussian rendering loss (optional) ---
        # Zero cost when dynamic_gaussian_render_loss is None or when gs_* keys
        # are absent (e.g. no gaussian_head configured in the model).
        if (self.dynamic_gaussian_render_loss is not None
                and "gs_opacity" in results
                and intrinsics is not None
                and extrinsics is not None):
            dgs_loss, dgs_loss_dict = self.dynamic_gaussian_render_loss(
                name=name,
                results=results,
                image=image,
                intrinsics=intrinsics,
                w2c=extrinsics,
                scale=scale,
                denormalize_fn=self.denormalize,
                meta_data=meta_data,
                global_step=total_iter,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, dgs_loss, dgs_loss_dict, task_name="dgs"
            )

        # --- Novel view rendering loss ---
        if novel_view_nums > 0 and novel_intrinsics is not None and novel_extrinsics is not None:
            total_loss, total_loss_dict = self._compute_novel_view_loss(
                name, results, gaussians, intrinsics, extrinsics, target_local_depth,
                novel_intrinsics, novel_extrinsics, novel_image,
                novel_target_local_depth, novel_target_depth_mask,
                novel_target_normal, novel_target_normal_mask,
                total_loss, total_loss_dict,
            )

        # Add metrics
        if scale is not None:
            total_loss_dict["scale"] = round(scale.max().cpu().item(), 2)
        if image is not None:
            total_loss_dict["aspect_ratio"] = round(image.shape[-1] / image.shape[-2], 2)
            total_loss_dict["bs"] = int(image.shape[0])
            total_loss_dict["view"] = int(image.shape[1])

        return total_loss, total_loss_dict

    def _compute_novel_view_loss(self, name, results, gaussians, intrinsics, extrinsics,
                                  target_local_depth, novel_intrinsics, novel_extrinsics,
                                  novel_image, novel_target_local_depth, novel_target_depth_mask,
                                  novel_target_normal, novel_target_normal_mask,
                                  total_loss, total_loss_dict):
        """Compute novel view rendering loss."""
        from hAlgorithm.modules.models2.external.worldmirror.models.utils.frustum import calculate_in_frustum_mask

        with torch.no_grad():
            unproject_masks = calculate_in_frustum_mask(
                novel_target_local_depth[:, :, -1], novel_intrinsics[..., :3, :3],
                novel_extrinsics.float().inverse(), target_local_depth[:, :, -1],
                intrinsics[..., :3, :3], extrinsics.float().inverse(),
            ).unsqueeze(2)

        novel_render_rgb = results["ffgs_novel_render_rgb"] * unproject_masks.float()
        novel_render_depth = results.get("ffgs_novel_render_depth")
        novel_render_normal = results.get("ffgs_novel_render_normal")

        if novel_render_depth is not None:
            B, V, H, W = unproject_masks.squeeze(2).shape
            novel_render_depth = novel_render_depth.view(B, V, 1, H, W) * unproject_masks.float()
            if hasattr(gaussians, "norm_scales") and gaussians.norm_scales is not None:
                novel_render_depth = novel_render_depth * gaussians.norm_scales[:, None, None, None, None]
            novel_render_depth = self.depth_to_points(novel_render_depth, K=novel_intrinsics)

        if novel_render_normal is not None:
            novel_render_normal = novel_render_normal * unproject_masks.float()

        loss, loss_dict = self.get_ffgs_loss(
            name, novel_render_rgb, novel_render_depth, novel_render_normal,
            novel_image, novel_target_local_depth, novel_target_depth_mask,
        )
        return self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="nrc")

    # =========================================================================
    # Postprocessing
    # =========================================================================

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
        """Convert model outputs to ReconstructOutput for visualization/saving."""
        # Build color map
        if image_show is not None and pred_local_points is not None:
            points_colors = image_show[0].transpose(1, 2, 0)
        else:
            points_colors = image[0].cpu().float().numpy().transpose(1, 2, 0)
            points_colors = (points_colors + 1) * 0.5 * 255

        if pred_local_points is not None:
            points_h, points_w = pred_local_points.shape[-2:]

            if points_h != points_colors.shape[0] or points_w != points_colors.shape[1]:
                points_colors = cv2.resize(
                    points_colors, dsize=(points_w, points_h), interpolation=cv2.INTER_LINEAR
                )
            points_colors = points_colors.reshape(-1, 3)

            pred_local_points = pred_local_points[0].cpu().float().numpy()
            pred_local_depth = pred_local_points[-1].clip(1e-3)
            pred_local_points = pred_local_points.reshape(3, -1).transpose(1, 0)
        else:
            pred_local_depth = None
            points_h, points_w = points_colors.shape[:2]
            points_colors = points_colors.reshape(-1, 3)

        # Confidence-based filtering
        if pred_local_points is not None and pred_local_conf is not None:
            pred_local_conf = pred_local_conf[0, 0].cpu().float().numpy()
            conf_thresh = np.percentile(
                pred_local_conf.reshape(-1),
                self.save_output_cfg["output_conf_ratio"] * 100,
            )
            conf_mask = pred_local_conf.reshape(-1) > conf_thresh
            filtered_pred_local_points = pred_local_points[conf_mask]
            filtered_points_colors = points_colors[conf_mask]
        else:
            filtered_pred_local_points = filtered_points_colors = None

        # Align to input resolution
        if (self.save_output_cfg["output_match_input_res"]
                and pred_local_depth is not None and align_data is not None):
            align_data = align_data[0].cpu().numpy()
            h, w = align_data.shape[:2]
            pred_local_depth = cv2.resize(
                pred_local_depth, dsize=(w, h), interpolation=cv2.INTER_LINEAR
            )
            if pred_local_conf is not None:
                pred_local_conf = cv2.resize(
                    pred_local_conf, dsize=(w, h), interpolation=cv2.INTER_LINEAR
                )

        # Convert tensors to numpy
        def _to_numpy_transpose(tensor, batch_idx=0, permute_hw=True):
            """Convert [B, C, H, W] tensor to [H, W, C] numpy."""
            if tensor is None:
                return None
            arr = tensor[batch_idx].cpu().float().numpy()
            return arr.transpose(1, 2, 0) if permute_hw else arr

        def _to_numpy_detach(tensor, batch_idx=0, permute_hw=True):
            """Convert [B, C, H, W] tensor to [H, W, C] numpy with detach."""
            if tensor is None:
                return None
            arr = tensor[batch_idx].detach().cpu().float().numpy()
            return arr.transpose(1, 2, 0) if permute_hw else arr

        pred_local_normal = _to_numpy_transpose(pred_local_normal)
        pred_global_points = _to_numpy_detach(pred_global_points)
        pred_global_conf = (
            pred_global_conf[0, 0].detach().cpu().float().numpy()
            if pred_global_conf is not None else None
        )
        pred_extrinsics = (
            pred_extrinsics[0].detach().cpu().float().numpy()
            if pred_extrinsics is not None else None
        )
        pred_intrinsics = (
            pred_intrinsics[0].detach().cpu().float().numpy()
            if pred_intrinsics is not None else None
        )

        # Local-to-global transform
        if (pred_local_points is not None
                and pred_extrinsics is not None
                and pred_intrinsics is not None):
            local2glb_points = (
                np.linalg.inv(pred_extrinsics)
                @ np.concatenate(
                    [pred_local_points, np.ones([pred_local_points.shape[0], 1])],
                    axis=-1, dtype=np.float32,
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

    # =========================================================================
    # Inference
    # =========================================================================

    @torch.no_grad()
    def infer(self, **batch):
        self.eval()

        (name, total_iter, meta_data, image, intrinsics, extrinsics, scale, prompt_depth,
         target_local_depth, target_global_points, target_depth_mask, target_normal,
         target_normal_mask, target_motion_mask, target_invalid_mask, image_show, align_data,
         ray_directions, ray_world, time_idx, query_times, trajs_2d, trajs_3d, valids, visibs,
         motion_extrinsics) = self.get_inputs(batch)

        track_query_points, track_vis, _ = self.get_track_inputs(batch)

        # Forward pass
        use_amp = batch.get("use_amp", False)
        amp_dtype = batch.get("amp_dtype", None)
        results = self._run_model_forward(
            image, scale, prompt_depth, intrinsics, ray_directions, extrinsics,
            ray_world, track_query_points, meta_data, query_times, time_idx,
            use_amp=bool(use_amp) if use_amp else False,
            amp_dtype=amp_dtype,
        )

        # Decode pose encoding
        pred_extrinsics, pred_intrinsics = self._decode_pose_encoding(
            results, image, scale
        )

        # Extract and normalize predictions
        pred_local_depth = results.get("depth")
        pred_local_conf = results.get("confidence")
        pred_local_normal = results.get("normal")

        pred_ray_directions = self._normalize_ray_directions(results.get("ray"))

        if pred_ray_directions is not None and self.save_output_cfg["output_intrinsics_from_ray"]:
            assert pred_ray_directions.ndim == 5
            pred_intrinsics = recover_pinhole_intrinsics_from_ray_directions(
                ray_directions=pred_ray_directions.view(
                    -1, *pred_ray_directions.shape[2:]
                ).permute(0, 2, 3, 1).contiguous()
            )
            pred_intrinsics = pred_intrinsics.view(
                *pred_ray_directions.shape[:2], *pred_intrinsics.shape[1:]
            )

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

        # Motion predictions
        pred_scene_flow = results.get("scene_flow", None)
        pred_dynamic_pointmap = results.get("dynamic_pointmap", None)

        if pred_dynamic_pointmap is not None and pred_scene_flow is None:
            pred_scene_flow = convert_pointmap_to_scene_flow(pred_dynamic_pointmap)

        # Denormalize scene_flow
        if pred_scene_flow is not None and scale is not None:
            scale_for_flow = scale[:, 0:1, :, :, :]
            pred_scene_flow = self.denormalize(pred_scene_flow, scale=scale_for_flow)

        if pred_local_depth is None:
            pred_local_depth = results.get("ffgs_depth")
            pred_local_conf = results.get("ffgs_confidence")

        # Denormalize targets and extrinsics
        if prompt_depth is not None and scale is not None:
            prompt_depth = self.denormalize(prompt_depth, scale=scale)
        if target_local_depth is not None and scale is not None:
            target_local_depth = self.denormalize(target_local_depth, scale=scale)
        if target_global_points is not None and scale is not None:
            target_global_points = self.denormalize(target_global_points, scale=scale)
        if extrinsics is not None and scale is not None:
            extrinsics[..., :3, 3] = self.denormalize(extrinsics[..., :3, 3], scale=scale[..., 0, 0])

        # Convert depth to pointmap and denormalize
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

        # Normalize normals
        if pred_local_normal is not None:
            tgt_hw = pred_local_depth.shape[-2:] if pred_local_depth is not None else image.shape[-2:]
            norm_h, norm_w = pred_local_normal.shape[-2:]
            if tgt_hw[0] != norm_h or tgt_hw[1] != norm_w:
                pred_local_normal = F.interpolate(
                    pred_local_normal, tgt_hw, mode="bilinear",
                    align_corners=False, antialias=False,
                )
            pred_local_normal = F.normalize(pred_local_normal, dim=-3)

        if pred_global_points is not None and scale is not None:
            pred_global_points = self.denormalize(pred_global_points, scale=scale)

        # --- 4DGS rendering pass (vis only, no grad) ---
        # At this point scene_flow and global_points are already denormalized.
        # The call is gated so existing configs without gaussian_head are unaffected.
        dgs_render_rgb = None
        dgs_render_depth = None
        if (self.dynamic_gaussian_render_loss is not None
                and "gs_opacity" in results
                and pred_global_points is not None
                and pred_scene_flow is not None):
            _dgs_renders = self.dynamic_gaussian_render_loss.render_frames(
                gs_opacity=results["gs_opacity"],
                gs_scale=results["gs_scale"],
                gs_rotation=results["gs_rotation"],
                gs_sh=results["gs_sh"],
                src_frame_idx=int(results.get("src_frame_idx", 0)),
                global_points=pred_global_points,
                scene_flow=pred_scene_flow,
                intrinsics=intrinsics,
                w2c=extrinsics,
                frame_num=int(meta_data["frames"][0]),
                view_num=int(meta_data["views"][0]),
            )
            dgs_render_rgb   = _dgs_renders.get("dgs_render_rgb")    # [B, N, 3, H, W]
            dgs_render_depth = _dgs_renders.get("dgs_render_depth")  # [B, N, 1, H, W]

        # Build per-view outputs
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        mv_outputs = self._build_mv_outputs(
            frame_num=frame_num, view_num=view_num,
            pred_local_depth=pred_local_depth, pred_local_conf=pred_local_conf,
            pred_local_normal=pred_local_normal, pred_global_points=pred_global_points,
            pred_global_conf=pred_global_conf, pred_extrinsics=pred_extrinsics,
            pred_intrinsics=pred_intrinsics, image=image, image_show=image_show,
            scale=scale, prompt_depth=prompt_depth,
            target_local_depth=target_local_depth, target_global_points=target_global_points,
            target_depth_mask=target_depth_mask, target_normal=target_normal,
            target_normal_mask=target_normal_mask, intrinsics=intrinsics,
            extrinsics=extrinsics, align_data=align_data,
            pred_track=pred_track, pred_track_vis=pred_track_vis,
            pred_track_conf=pred_track_conf, track_query_points=track_query_points,
            track_vis=track_vis, gaussians=gaussians, render_rgb=render_rgb,
            render_depth=render_depth, pred_scene_flow=pred_scene_flow,
            trajs_3d=trajs_3d, trajs_2d=trajs_2d, visibs=visibs, valids=valids,
            motion_extrinsics=motion_extrinsics, meta_data=meta_data,
            dgs_render_rgb=dgs_render_rgb, dgs_render_depth=dgs_render_depth,
        )
        return mv_outputs

    def _decode_pose_encoding(self, results, image, scale):
        """Decode pose encoding to extrinsics and intrinsics."""
        if self.pose_encoding_type != "absT_quaR_FoV":
            raise NotImplementedError

        pose_enc = results.get("pose_enc", None)
        if pose_enc is None:
            return None, None

        if isinstance(pose_enc, (list, tuple)):
            pose_enc = pose_enc[-1]

        pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
            pose_encoding=pose_enc.float(),
            image_size_hw=image.shape[-2:],
            build_intrinsics=True,
        )
        if scale is not None:
            pred_extrinsics[..., :3, 3] = self.denormalize(
                pred_extrinsics[..., :3, 3], scale=scale[..., 0, 0]
            )

        if self.save_output_cfg["output_normalize_cameras"]:
            w2c_pred = pred_extrinsics
            base_c2w_pred = w2c_pred[:, 0:1].inverse()
            pred_extrinsics = w2c_pred @ base_c2w_pred

        return pred_extrinsics, pred_intrinsics

    def _build_mv_outputs(self, frame_num, view_num, *, pred_local_depth, pred_local_conf,
                           pred_local_normal, pred_global_points, pred_global_conf,
                           pred_extrinsics, pred_intrinsics, image, image_show, scale,
                           prompt_depth, target_local_depth, target_global_points,
                           target_depth_mask, target_normal, target_normal_mask,
                           intrinsics, extrinsics, align_data, pred_track, pred_track_vis,
                           pred_track_conf, track_query_points, track_vis, gaussians,
                           render_rgb, render_depth, pred_scene_flow, trajs_3d, trajs_2d,
                           visibs, valids, motion_extrinsics, meta_data,
                           dgs_render_rgb=None, dgs_render_depth=None):
        """Build per-view output list."""

        def get_view(data, index):
            return data[:, index] if data is not None else None

        mv_outputs = []
        for fi in range(frame_num):
            for vi in range(view_num):
                index = fi * view_num + vi
                single = self.postprocess(
                    pred_local_points=get_view(pred_local_depth, index),
                    pred_local_conf=get_view(pred_local_conf, index),
                    pred_local_normal=get_view(pred_local_normal, index),
                    pred_global_points=get_view(pred_global_points, index),
                    pred_global_conf=get_view(pred_global_conf, index),
                    pred_extrinsics=get_view(pred_extrinsics, index),
                    pred_intrinsics=get_view(pred_intrinsics, index),
                    image=get_view(image, index),
                    image_show=get_view(image_show, index),
                    scale=get_view(scale, index),
                    prompt_depth=get_view(prompt_depth, index),
                    target_local_depth=get_view(target_local_depth, index),
                    target_global_points=get_view(target_global_points, index),
                    target_depth_mask=get_view(target_depth_mask, index),
                    target_normal=get_view(target_normal, index),
                    target_normal_mask=get_view(target_normal_mask, index),
                    intrinsics=get_view(intrinsics, index),
                    extrinsics=get_view(extrinsics, index),
                    align_data=get_view(align_data, 0),
                )

                # Attach extra outputs
                self._attach_track_data(
                    single, index, pred_track, pred_track_vis,
                    pred_track_conf, track_query_points, track_vis,
                )
                self._attach_gaussian_data(single, index, gaussians)
                self._attach_rgb_data(single, index, image, render_rgb, render_depth)
                self._attach_motion_data(
                    single, index, pred_scene_flow, trajs_3d, trajs_2d,
                    visibs, valids, motion_extrinsics, scale, meta_data,
                )
                self._attach_dgs_render_data(single, index, dgs_render_rgb, dgs_render_depth)

                single.frame_index = fi
                single.view_index = vi
                single.total_index = index
                mv_outputs.append(single)

        return mv_outputs

    @staticmethod
    def _attach_track_data(single, index, pred_track, pred_track_vis,
                            pred_track_conf, track_query_points, track_vis):
        """Attach tracking data to single output."""
        if pred_track is None:
            return
        single.track_pred = pred_track[0, index].cpu().numpy()
        single.track_vis_pred = pred_track_vis[0, index].cpu().numpy()
        single.track_pred_local_conf = pred_track_conf[0, index].cpu().numpy()
        single.track_gt = track_query_points[0, index].cpu().numpy()
        if track_vis is not None:
            single.track_vis = track_vis[0, index].cpu().numpy()

    @staticmethod
    def _attach_gaussian_data(single, index, gaussians):
        """Attach Gaussian representation to first view."""
        if gaussians is not None and index == 0:
            single.gaussians = gaussians

    @staticmethod
    def _attach_rgb_data(single, index, image, render_rgb, render_depth):
        """Attach RGB and render data to single output."""
        single.rgb = (image[0, index].permute(1, 2, 0).float().cpu().numpy() + 1) * 0.5
        if render_rgb is not None:
            single.render_rgb = render_rgb[0, index].float().cpu().numpy().transpose(1, 2, 0)
        if render_depth is not None:
            single.render_depth = render_depth[0, index].float().cpu().numpy()

    @staticmethod
    def _attach_dgs_render_data(single, index, dgs_render_rgb, dgs_render_depth):
        """Attach 4DGS rendered frames to single output.

        ``dgs_render_rgb``  is ``[B, N, 3, H, W]`` in ``[0, 1]``; indexed by
        frame/view ``index`` to produce a per-view numpy array ``[H, W, 3]``.
        """
        if dgs_render_rgb is not None and index < dgs_render_rgb.shape[1]:
            single.dgs_render_rgb = (
                dgs_render_rgb[0, index].float().cpu().numpy().transpose(1, 2, 0)
            )
        if dgs_render_depth is not None and index < dgs_render_depth.shape[1]:
            single.dgs_render_depth = dgs_render_depth[0, index].float().cpu().numpy()

    def _attach_motion_data(self, single, index, pred_scene_flow, trajs_3d, trajs_2d,
                             visibs, valids, motion_extrinsics, scale, meta_data):
        """Attach motion-related data to single output (delegates to shared util)."""
        attach_motion_data_to_output(
            single=single,
            index=index,
            pred_scene_flow=pred_scene_flow,
            trajs_3d=trajs_3d,
            trajs_2d=trajs_2d,
            visibs=visibs,
            valids=valids,
            motion_extrinsics=motion_extrinsics,
            scale=scale,
            meta_data=meta_data,
            is_training=self.training,
            denormalize_fn=self.denormalize,
        )

    # =========================================================================
    # Output directories
    # =========================================================================

    def get_out_dir(self, out_dir, data_idx=None):
        gs_out_dir = os.path.join(out_dir, "gaussians")
        glb_out_dir = os.path.join(out_dir, "glb")
        dgs_out_dir = os.path.join(out_dir, "4dgs")

        subdirs = ["camera", "mvdepth", "track", "motion"]
        if data_idx is not None:
            paths = [os.path.join(out_dir, f"{s}/{data_idx:06d}") for s in subdirs]
        else:
            paths = [os.path.join(out_dir, s) for s in subdirs]

        return gs_out_dir, glb_out_dir, *paths, dgs_out_dir

    def get_gt_out_dir(self, out_dir):
        if self.save_output_cfg["gt_out_dir"] is None:
            return None, None, None

        abs_gt_out_dir = os.path.join(
            os.path.dirname(os.path.dirname(out_dir)),
            self.save_output_cfg["gt_out_dir"],
            os.path.basename(out_dir),
        )
        _, gt_glb_out_dir, _, _, gt_track_out_dir, _, _ = self.get_out_dir(abs_gt_out_dir, data_idx=None)
        return abs_gt_out_dir, gt_glb_out_dir, gt_track_out_dir

    # =========================================================================
    # Visualization
    # =========================================================================

    def visualize(self, outputs_list, meta_data, out_dir):
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        data_idx = meta_data["data_idx"][0]

        gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir, motion_out_dir, dgs_out_dir = (
            self.get_out_dir(out_dir, data_idx=data_idx)
        )
        gt_out_dir, gt_glb_out_dir, gt_track_out_dir = self.get_gt_out_dir(out_dir)

        cfg = self.save_output_cfg

        # Images
        if gt_out_dir is not None:
            vis_images(
                cfg=cfg, mv_outputs=outputs_list, gt_out_dir=gt_out_dir,
                data_idx=data_idx, frame_num=frame_num, view_num=view_num,
            )

        # Local results
        if cfg["save_local_results"]:
            os.makedirs(mv_out_dir, exist_ok=True)
            vis_local_results(
                cfg=cfg, mv_outputs=outputs_list, mv_out_dir=mv_out_dir,
                frame_num=frame_num, view_num=view_num, meta_data=meta_data,
            )
        if cfg["save_extra_local_results"]:
            os.makedirs(mv_out_dir, exist_ok=True)
            vis_extra_local_results(
                cfg=cfg, mv_outputs=outputs_list, mv_out_dir=mv_out_dir,
                frame_num=frame_num, view_num=view_num, meta_data=meta_data,
            )

        # Global results
        if cfg["save_glb_results"]:
            vis_glb_results(
                cfg=cfg, mv_outputs=outputs_list, glb_out_dir=glb_out_dir,
                data_idx=data_idx, frame_num=frame_num, view_num=view_num,
                gt_out_dir=gt_glb_out_dir,
            )

        # Camera
        if cfg["save_cameras"]:
            vis_camera_results(
                cfg=cfg, mv_outputs=outputs_list, camera_out_dir=camera_out_dir,
                frame_num=frame_num, view_num=view_num,
            )

        # Gaussians
        if cfg["save_gaussians"] and outputs_list[0].gaussians is not None:
            vis_render(
                cfg=cfg, mv_outputs=outputs_list, gs_out_dir=gs_out_dir, data_idx=data_idx,
            )
            vis_render_video(
                cfg=cfg, model=self.model, mv_outputs=outputs_list,
                gs_out_dir=gs_out_dir, data_idx=data_idx,
                frame_num=frame_num, view_num=view_num, device=self.device,
            )

        # Tracking
        if cfg["save_track_results"] and outputs_list[0].track_pred is not None:
            os.makedirs(track_out_dir, exist_ok=True)
            vis_track_results(
                cfg=cfg, mv_outputs=outputs_list, track_out_dir=track_out_dir,
                data_idx=data_idx, gt_out_dir=gt_track_out_dir,
            )

        # Motion
        if (cfg["save_motion_results"]
                and hasattr(outputs_list[0], 'scene_flow_pred')
                and outputs_list[0].scene_flow_pred is not None):
            os.makedirs(motion_out_dir, exist_ok=True)
            vis_motion_head_results(
                cfg=cfg, mv_outputs=outputs_list, motion_out_dir=motion_out_dir,
                data_idx=data_idx, meta_data=meta_data,
            )
            if cfg.get("save_motion_3d", False):
                vis_motion_3d_rerun(
                    cfg=cfg, mv_outputs=outputs_list, out_dir=out_dir,
                    data_idx=data_idx, meta_data=meta_data,
                )

        # 4DGS render comparison
        if (cfg.get("save_4dgs_results", False)
                and any(getattr(o, "dgs_render_rgb", None) is not None for o in outputs_list)):
            vis_4dgs_results(
                outputs_list=outputs_list,
                dgs_out_dir=dgs_out_dir,
                data_idx=data_idx,
                frame_num=frame_num,
                view_num=view_num,
            )

    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        save_mv_outputs(
            cfg=self.save_output_cfg,
            mv_outputs=outputs,
            meta_data=meta_data,
            out_dir=out_dir,
            output_meta_dict=output_meta_dict,
        )