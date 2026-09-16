import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from typing import Optional, Dict, List, Tuple

from hAlgorithm.modules.pipelines2.mvfr_v1 import MVFRPipeline
from hAlgorithm.modules.pipelines2.utils.motion_utils import (
    compute_motion_head_loss,
    attach_motion_data_to_output,
    convert_pointmap_to_scene_flow,
    get_ref_scale,
    get_ref_scale_flat,
)
from hAlgorithm.modules.pipelines2.utils.visualize_motion import (
    vis_motion_head_results,
    vis_motion_3d_rerun,
)
from hAlgorithm.utils import instantiate_from_config


class MVFRMotionPipeline(MVFRPipeline):
    """Pipeline for Multi-View Feed-forward Reconstruction with Motion/Scene Flow support.
    
    Extends MVFRPipeline with:
    - Scene flow / dynamic pointmap prediction and loss (aligned with movies.py)
    - Trajectory data handling (trajs_2d, trajs_3d, valids, visibs)
    - Motion visualization
    
    Uses single-source pairing strategy: frame 0 → all frames [0, 1, ..., V-1]
    """

    def __init__(
        self,
        # Motion specific inputs
        motion_extrinsics_name="extrinsics",  # Raw w2c extrinsics for motion loss
        # Motion losses (aligned with movies.py naming)
        motion_any4d_loss=None,      # Scene flow loss
        motion_pointmap_loss=None,   # Dynamic pointmap loss
        # Save output config updates
        save_output_cfg=None,
        clip_level_time_norm=False,
        **kwargs,
    ):
        # Initialize parent class
        super(MVFRMotionPipeline, self).__init__(save_output_cfg=save_output_cfg, **kwargs)

        # Motion specific parameters
        self.motion_extrinsics_name = motion_extrinsics_name
        self.clip_level_time_norm = clip_level_time_norm

        # Cache motion prediction type from model (aligned with movies.py)
        self._motion_prediction_type = self._init_motion_prediction_type()

        # Motion losses (aligned with movies.py naming)
        self.motion_any4d_loss = instantiate_from_config(motion_any4d_loss)
        self.motion_pointmap_loss = instantiate_from_config(motion_pointmap_loss)

        # Update save_output_cfg with motion-specific options
        motion_save_cfg = dict(
            save_motion_results=False,
            save_motion_3d=True,
            save_raw_motion_flow=False,
        )
        self.save_output_cfg.update(motion_save_cfg)
        if save_output_cfg is not None:
            self.save_output_cfg.update(save_output_cfg)

    def _init_motion_prediction_type(self) -> str:
        """Detect and cache the motion head prediction type (aligned with movies.py)."""
        motion_head = getattr(self.model, "motion_head", None)
        if motion_head is not None:
            pred_type = getattr(motion_head, "prediction_type", "scene_flow")
            logging.info(f"[MVFRMotionPipeline] self.model.motion_head: {motion_head}")
        else:
            pred_type = "scene_flow"
        logging.info(f"[MVFRMotionPipeline] motion_prediction_type cached as: {pred_type}")
        return pred_type

    def _parse_time_info(self, meta_data):
        """Parse time information from metadata using clip-level normalization.

        Normalizes frame_ids by the full clip window
        (``clip_start_frame_id`` .. ``clip_end_frame_id``) so the time value
        reflects position within the *clip_maxlen*-frame clip.

        Sets ``meta_data["time_idx"]`` so both the backbone
        (``_modify_cam_token``) and the motion head read consistent values.
        """
        if "data_info" not in meta_data:
            return
        data_info = meta_data["data_info"]
        batch_size = len(data_info)
        if batch_size == 0 or len(data_info[0]) == 0:
            return

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

        time_idx = torch.tensor(time_idx_list, dtype=torch.float32, device=self.device)
        meta_data["time_idx"] = time_idx

    def _pad_traj_batch(self, trajs_2d_list, trajs_3d_list, valids_list, visibs_list):
        B = len(trajs_2d_list)
        T, _, C = trajs_2d_list[0].shape
        max_N = max(t.shape[1] for t in trajs_2d_list)

        def fast_pad(data_list, channels=None):
            if data_list is None or data_list[0] is None:
                return None
            
            # 1. 预先分配固定内存，加速 HtoD 传输
            shape = (B, T, max_N, channels) if channels else (B, T, max_N)
            out = torch.zeros(shape, dtype=data_list[0].dtype, pin_memory=True)
            
            # 2. 填充数据
            for i, t in enumerate(data_list):
                # t 的形状为 [T, N_i, C] 或 [T, N_i]
                out[i, :, :t.shape[1]] = t
                
            # 3. 异步推送到 GPU
            return out.to(self.device, non_blocking=True)

        # 统一处理
        trajs_2d = fast_pad(trajs_2d_list, channels=2)
        trajs_3d = fast_pad(trajs_3d_list, channels=3)
        valids   = fast_pad(valids_list)
        visibs   = fast_pad(visibs_list)

        return trajs_2d, trajs_3d, valids, visibs

    def get_inputs(self, batch):
        """Override get_inputs to add motion-specific inputs."""
        # Get base inputs from parent
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
        ) = super().get_inputs(batch)

        # Inject clip-level time normalization into meta_data so both
        # backbone (_modify_cam_token) and motion head see consistent time_idx.
        if self.clip_level_time_norm:
            self._parse_time_info(meta_data)

        # Load motion extrinsics (raw w2c) for motion loss
        # Use reference-view scale (view 0) to match movies.py convention
        motion_extrinsics = None
        motion_ext_name = self.motion_extrinsics_name or 'extrinsics'
        if motion_ext_name in batch:
            motion_extrinsics = batch[motion_ext_name].to(device=self.device)
            if scale is not None:
                ref_scale_flat = get_ref_scale_flat(scale)  # [B, 1]
                motion_extrinsics[..., :3, 3] = self.normalize(motion_extrinsics[..., :3, 3], ref_scale_flat)

        # Load trajectory data (may be a list due to variable-length skip collation)
        trajs_2d = batch.get('trajs_2d', None)
        trajs_3d = batch.get('trajs_3d', None)
        valids = batch.get('valids', None)
        visibs = batch.get('visibs', None)
        
        # Handle list format (variable-length trajectories) with dynamic padding
        if trajs_2d is not None and isinstance(trajs_2d, list):
            # Each element has shape [T, N_i, C] where N_i varies per sample
            # Pad N dimension to batch max
            trajs_2d, trajs_3d, valids, visibs = self._pad_traj_batch(
                trajs_2d, trajs_3d, valids, visibs
            )
        elif trajs_2d is not None:
            # Already tensor format, just move to device
            trajs_2d = trajs_2d.to(self.device)
            if trajs_3d is not None:
                trajs_3d = trajs_3d.to(self.device)
            if valids is not None:
                valids = valids.to(self.device)
            if visibs is not None:
                visibs = visibs.to(self.device)

        # Normalize trajs_3d by reference view's scale (view 0) to match movies.py
        if trajs_3d is not None and scale is not None:
            ref_scale = get_ref_scale(scale)  # [B, 1, 1, 1]
            trajs_3d = trajs_3d / ref_scale

        # Debug: log if tracking data is missing
        if not self.training and trajs_3d is None:
            logging.debug(f"[get_inputs] trajs_3d not found in batch. Available keys: {list(batch.keys())}")

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
            motion_extrinsics,
            trajs_2d,
            trajs_3d,
            valids,
            visibs,
        )


    def train_step(self, batch):
        """Override train_step to add motion loss computation."""
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
            motion_extrinsics,
            trajs_2d,
            trajs_3d,
            valids,
            visibs,
        ) = self.get_inputs(batch)

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
                meta_data=meta_data,
            )
            assert track_query_points is None  # TODO
        else:
            novel_intrinsics = novel_extrinsics = None

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
            novel_intrinsics=novel_intrinsics,
            novel_w2c=novel_extrinsics,
            extrinsics_noise=extrinsics_noise,
            prompt_extrinsics=prompt_extrinsics,
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

        # Motion/Scene flow output
        pred_scene_flow = results.get('scene_flow', None)  # [B, query_view, num_view, 3, H, W]

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
            pred_local_normal = F.normalize(pred_local_normal, dim=-3)

        loss, loss_dict = self.get_base_loss(
            name=name,
            image=image,
            coord="local",
            pred_depth=pred_local_depth,
            pred_conf=pred_local_conf,
            pred_normal=pred_local_normal,
            pred_ray_directions=pred_ray_directions,
            pred_ray_in_world=pred_ray_in_world,
            pred_invalid_mask=pred_local_invalid_mask,
            pred_motion_mask=pred_local_motion_mask,
            target_depth=target_local_depth,
            target_ray_directions=ray_directions,
            target_ray_in_world=ray_world,
            target_normal=target_normal,
            target_normal_mask=target_normal_mask,
            target_motion_mask=target_motion_mask,
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
            target_depth=target_global_points,
            valid_mask=target_depth_mask,
        )
        total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="glb")

        if self.local2global_loss is not None and pose_enc is not None and pred_local_depth is not None:
            from hAlgorithm.modules.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri

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
                from hAlgorithm.modules.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri

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

                    tmp_pred_depth = self.depth_to_points(tmp_pred_depth.float(), K=pred_intrinsics)
                    tmp_pred_depth = pred_extrinsics.inverse() @ torch.cat([tmp_pred_depth, tmp_pred_depth.new_ones([B, S, 1, H, W])], dim=2).reshape(B, S, 4, -1)
                    tmp_pred_depth = tmp_pred_depth.reshape(B, S, 4, H, W)[:, :, :3, :, :]

                if pred_ffgs_conf is not None:
                    tmp_pred_conf = pred_ffgs_conf.squeeze(2).unsqueeze(-1)
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

        # Motion loss (aligned with movies.py - using compute_motion_head_loss)
        pred_dynamic_pointmap = results.get('dynamic_pointmap', None)
        motion_prediction_type = getattr(self, "_motion_prediction_type", "scene_flow")
        
        if pred_scene_flow is not None or pred_dynamic_pointmap is not None:
            original_h = meta_data["origin_height"][0].item() if "origin_height" in meta_data else image.shape[-2]
            original_w = meta_data["origin_width"][0].item() if "origin_width" in meta_data else image.shape[-1]
            if isinstance(original_h, torch.Tensor):
                original_h = original_h.item()
            if isinstance(original_w, torch.Tensor):
                original_w = original_w.item()

            motion_ref_frame_idx = results.get("src_frame_idx", 0)
            loss, loss_dict = compute_motion_head_loss(
                name=name,
                pred_scene_flow=pred_scene_flow,
                pred_dynamic_pointmap=pred_dynamic_pointmap,
                motion_prediction_type=motion_prediction_type,
                trajs_3d=trajs_3d,
                trajs_2d=trajs_2d,
                visibs=visibs,
                valids=valids,
                original_h=original_h,
                original_w=original_w,
                scale=scale,
                extrinsics=motion_extrinsics,
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

        # Novel view synthesis loss
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
                )

                B, V, H, W = unproject_masks.shape
                unproject_masks = unproject_masks.unsqueeze(2)

            novel_render_rgb = results["ffgs_novel_render_rgb"]
            novel_render_depth = results.get("ffgs_novel_render_depth", None)
            novel_render_normal = results.get("ffgs_novel_render_normal", None)

            novel_render_rgb = novel_render_rgb * unproject_masks.float()

            if novel_render_depth is not None:
                novel_render_depth = novel_render_depth.view(B, V, 1, *novel_render_depth.shape[-2:])
                novel_render_depth = novel_render_depth * unproject_masks.float()

                if hasattr(gaussians, "norm_scales") and gaussians.norm_scales is not None:
                    novel_render_depth = novel_render_depth * gaussians.norm_scales[:, None, None, None, None]

                novel_render_depth = self.depth_to_points(novel_render_depth, K=novel_intrinsics)

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
            total_loss_dict["aspect_ratio"] = round(aspect_ratio, 2)
            total_loss_dict["bs"] = int(image.shape[0])
            total_loss_dict["view"] = int(image.shape[1])

        return total_loss, total_loss_dict

    def visualize(self, outputs_list, meta_data, out_dir):
        """Override visualize to add motion visualization."""
        # Call parent's visualize first
        super().visualize(outputs_list, meta_data, out_dir)
        
        # Motion visualization - check if scene_flow_pred is available
        has_scene_flow = (
            hasattr(outputs_list[0], 'scene_flow_pred') and 
            outputs_list[0].scene_flow_pred is not None
        )
        
        if not has_scene_flow:
            logging.warning(f"[MVFRMotionPipeline.visualize] scene_flow_pred not found on outputs[0]")
            return
        
        data_idx = meta_data["data_idx"][0]
        motion_out_dir = os.path.join(out_dir, f"motion/{data_idx:06d}")
        
        logging.info(f"[MVFRMotionPipeline.visualize] Motion visualization enabled for data_idx={data_idx}")
        logging.info(f"[MVFRMotionPipeline.visualize] save_motion_results={self.save_output_cfg.get('save_motion_results', False)}")
        logging.info(f"[MVFRMotionPipeline.visualize] save_motion_3d={self.save_output_cfg.get('save_motion_3d', False)}")
        
        # 2D motion visualization (rainbow trails)
        if self.save_output_cfg.get("save_motion_results", False):
            os.makedirs(motion_out_dir, exist_ok=True)
            logging.info(f"[MVFRMotionPipeline.visualize] Calling vis_motion_head_results, output to {motion_out_dir}")
            vis_motion_head_results(
                cfg=self.save_output_cfg,
                mv_outputs=outputs_list,
                motion_out_dir=motion_out_dir,
                data_idx=data_idx,
                meta_data=meta_data
            )
        
        # 3D motion visualization (rerun)
        if self.save_output_cfg.get("save_motion_3d", False):
            logging.info(f"[MVFRMotionPipeline.visualize] Calling vis_motion_3d_rerun, parent out_dir={out_dir}")
            vis_motion_3d_rerun(
                cfg=self.save_output_cfg,
                mv_outputs=outputs_list,
                out_dir=out_dir,  # Pass parent out_dir, function creates rerun_vis subdirectory
                data_idx=data_idx,
                meta_data=meta_data
            )

    @torch.no_grad()
    def infer(self, **batch):
        """Override infer to handle extended get_inputs return values."""
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
            motion_extrinsics,
            trajs_2d,
            trajs_3d,
            valids,
            visibs,
        ) = self.get_inputs(batch)

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
                    w2c=w2c,
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
                w2c=w2c,
                ray_world=ray_world,
                query_points=track_query_points[:, 0] if track_query_points is not None else None,
                meta_data=meta_data,
            )

        # Import pose encoding utility
        from hAlgorithm.modules.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from hAlgorithm.modules.utils.ray import recover_pinhole_intrinsics_from_ray_directions

        if self.pose_encoding_type == "absT_quaR_FoV":
            pose_enc = results.get("pose_enc", None)
            if pose_enc is not None and isinstance(pose_enc, (list, tuple)):
                pose_enc = pose_enc[-1]
        else:
            raise NotImplementedError

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

        pred_local_depth = results.get("depth")
        pred_local_conf = results.get("confidence")
        pred_local_normal = results.get("normal")
        pred_local_invalid_mask = results.get("invalid_mask")
        pred_local_motion_mask = results.get("motion_mask")

        pred_ray_directions = results.get("ray")
        if pred_ray_directions is not None:
            pred_ray_directions = pred_ray_directions / pred_ray_directions.norm(dim=2, keepdim=True).clip(min=1e-8)
            if self.save_output_cfg.get("output_intrinsics_from_ray", False):
                pred_intrinsics = recover_pinhole_intrinsics_from_ray_directions(
                    ray_directions=pred_ray_directions.view(-1, *pred_ray_directions.shape[2:]).permute(0, 2, 3, 1).contiguous()
                )
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
            if scale is not None:
                pred_local_depth = self.denormalize(pred_local_depth, scale=scale)

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

                if gaussians is not None and index == 0:
                    single.gaussians = gaussians

                single.rgb = (image[0, index].permute(1, 2, 0).cpu().numpy() + 1) * 0.5

                if render_rgb is not None:
                    single.render_rgb = render_rgb[0, index].cpu().numpy().transpose(1, 2, 0)

                if render_depth is not None:
                    single.render_depth = render_depth[0, index].cpu().numpy()

                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                mv_outputs.append(single)

        # Motion predictions (aligned with movies.py)
        pred_scene_flow = results.get('scene_flow', None)
        pred_dynamic_pointmap = results.get('dynamic_pointmap', None)

        # Convert dynamic_pointmap to scene_flow if needed (aligned with movies.py)
        if pred_dynamic_pointmap is not None and pred_scene_flow is None:
            pred_scene_flow = convert_pointmap_to_scene_flow(pred_dynamic_pointmap)

        # Denormalize scene_flow (aligned with movies.py: using scale[:, 0:1])
        if pred_scene_flow is not None and scale is not None:
            scale_for_flow = scale[:, 0:1, :, :, :]
            pred_scene_flow = self.denormalize(pred_scene_flow, scale=scale_for_flow)

        # Attach motion data to all views (aligned with movies.py)
        for index, single in enumerate(mv_outputs):
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

        return mv_outputs
