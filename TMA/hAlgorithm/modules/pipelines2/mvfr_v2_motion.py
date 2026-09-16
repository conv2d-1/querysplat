import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.modules.pipelines2.mvfr_v2 import MVFRPipeline
from hAlgorithm.modules.pipelines2.utils.motion_utils import (
    attach_motion_data_to_output,
    compute_motion_head_loss,
    convert_pointmap_to_scene_flow,
    get_ref_scale,
    get_ref_scale_flat,
)
from hAlgorithm.modules.pipelines2.utils.visualize_motion import (
    vis_motion_3d_rerun,
    vis_motion_head_results,
)
from hAlgorithm.utils import instantiate_from_config


class MVFRMotionV2Pipeline(MVFRPipeline):
    """MVFR v2 pipeline with additional motion supervision/output."""

    def __init__(
        self,
        motion_extrinsics_name="extrinsics",
        motion_any4d_loss=None,
        motion_pointmap_loss=None,
        clip_level_time_norm=False,
        save_output_cfg=None,
        **kwargs,
    ):
        super().__init__(save_output_cfg=save_output_cfg, **kwargs)
        self.motion_extrinsics_name = motion_extrinsics_name
        self.clip_level_time_norm = clip_level_time_norm
        self.motion_any4d_loss = instantiate_from_config(motion_any4d_loss)
        self.motion_pointmap_loss = instantiate_from_config(motion_pointmap_loss)
        self._motion_prediction_type = self._init_motion_prediction_type()

        extra_save_cfg = dict(
            save_motion_results=False,
            save_motion_3d=True,
            save_raw_motion_flow=False,
        )
        self.save_output_cfg.update(extra_save_cfg)
        if save_output_cfg is not None:
            self.save_output_cfg.update(save_output_cfg)

    def _init_motion_prediction_type(self) -> str:
        motion_head = getattr(self.model, "motion_head", None)
        if motion_head is None:
            return "scene_flow"
        pred_type = getattr(motion_head, "prediction_type", "scene_flow")
        logging.info("[MVFRMotionV2Pipeline] motion prediction type: %s", pred_type)
        return pred_type

    def _parse_time_info(self, meta_data):
        if "data_info" not in meta_data:
            return
        data_info = meta_data["data_info"]
        batch_size = len(data_info)
        if batch_size == 0 or len(data_info[0]) == 0:
            return

        clip_starts = meta_data.get("clip_start_frame_id")
        clip_ends = meta_data.get("clip_end_frame_id")
        assert clip_starts is not None and clip_ends is not None, (
            "clip_level_time_norm=True requires clip_start_frame_id and clip_end_frame_id in meta_data"
        )

        time_idx_list = []
        for batch_idx in range(batch_size):
            raw_ids = [float(fd.get("frame_id", 0)) for fd in data_info[batch_idx]]
            cs = float(clip_starts[batch_idx]) if hasattr(clip_starts, "__getitem__") else float(clip_starts)
            ce = float(clip_ends[batch_idx]) if hasattr(clip_ends, "__getitem__") else float(clip_ends)
            span = ce - cs
            if span < 1e-6:
                normalized = [0.5] * len(raw_ids)
            else:
                normalized = [(fid - cs) / span for fid in raw_ids]
            time_idx_list.append(normalized)

        meta_data["time_idx"] = torch.tensor(time_idx_list, dtype=torch.float32, device=self.device)

    def _pad_traj_batch(self, trajs_2d_list, trajs_3d_list, valids_list, visibs_list):
        bsz = len(trajs_2d_list)
        tnum = trajs_2d_list[0].shape[0]
        max_n = max(t.shape[1] for t in trajs_2d_list)

        def fast_pad(data_list, channels=None):
            if data_list is None or data_list[0] is None:
                return None
            shape = (bsz, tnum, max_n, channels) if channels else (bsz, tnum, max_n)
            out = torch.zeros(shape, dtype=data_list[0].dtype, pin_memory=True)
            for i, t in enumerate(data_list):
                out[i, :, : t.shape[1]] = t
            return out.to(self.device, non_blocking=True)

        trajs_2d = fast_pad(trajs_2d_list, channels=2)
        trajs_3d = fast_pad(trajs_3d_list, channels=3)
        valids = fast_pad(valids_list)
        visibs = fast_pad(visibs_list)
        return trajs_2d, trajs_3d, valids, visibs

    def get_motion_inputs(self, batch, meta_data, scale):
        if self.clip_level_time_norm:
            self._parse_time_info(meta_data)

        motion_extrinsics = None
        motion_ext_name = self.motion_extrinsics_name or "extrinsics"
        if motion_ext_name in batch:
            motion_extrinsics = batch[motion_ext_name].to(device=self.device).clone()
            if scale is not None:
                ref_scale_flat = get_ref_scale_flat(scale)
                motion_extrinsics[..., :3, 3] = self.normalize(motion_extrinsics[..., :3, 3], ref_scale_flat)

        trajs_2d = batch.get("trajs_2d", None)
        trajs_3d = batch.get("trajs_3d", None)
        valids = batch.get("valids", None)
        visibs = batch.get("visibs", None)

        if trajs_2d is not None and isinstance(trajs_2d, list):
            trajs_2d, trajs_3d, valids, visibs = self._pad_traj_batch(trajs_2d, trajs_3d, valids, visibs)
        elif trajs_2d is not None:
            trajs_2d = trajs_2d.to(self.device)
            if trajs_3d is not None:
                trajs_3d = trajs_3d.to(self.device)
            if valids is not None:
                valids = valids.to(self.device)
            if visibs is not None:
                visibs = visibs.to(self.device)

        if trajs_3d is not None and scale is not None:
            trajs_3d = trajs_3d / get_ref_scale(scale)

        return motion_extrinsics, trajs_2d, trajs_3d, valids, visibs

    def _forward_motion_results(self, batch):
        (
            _name,
            _total_iter,
            meta_data,
            image,
            intrinsics,
            extrinsics,
            scale,
            prompt_depth,
            target_local_depth,
            _target_global_points,
            target_depth_mask,
            _target_normal,
            _target_normal_mask,
            target_motion_mask,
            _target_invalid_mask,
            _image_show,
            _align_data,
            ray_directions,
            ray_world,
            extrinsics_noise,
            _target_match_gt_depth,
            _target_match_gt_extrinsics,
            _target_match_gt_intrinsics,
            _precomputed_warp,
            _precomputed_warp_mask,
        ) = super().get_inputs(batch)

        motion_extrinsics, trajs_2d, trajs_3d, valids, visibs = self.get_motion_inputs(batch, meta_data, scale)
        track_query_points, _track_vis, _track_pos_masks = self.get_track_inputs(batch)
        rgb_mask = batch.get("rgb_mask", None)
        if rgb_mask is not None:
            rgb_mask = rgb_mask.to(self.device)

        if "prompt_extrinsics" in batch:
            w2c = batch["prompt_extrinsics"].to(self.device)
            if scale is not None:
                w2c[..., :3, 3] = self.normalize(w2c[..., :3, 3], scale[..., 0, 0])
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
            w2c=w2c,
            ray_world=ray_world,
            query_points=track_query_points[:, 0] if track_query_points is not None else None,
            meta_data=meta_data,
            rgb_mask=rgb_mask,
        )
        return (
            results,
            meta_data,
            image,
            intrinsics,
            scale,
            target_local_depth,
            target_depth_mask,
            target_motion_mask,
            motion_extrinsics,
            trajs_2d,
            trajs_3d,
            valids,
            visibs,
        )

    def train_step(self, batch):
        self.train()
        (
            name,
            _total_iter,
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
            _image_show,
            _align_data,
            ray_directions,
            ray_world,
            extrinsics_noise,
            target_match_gt_depth,
            target_match_gt_extrinsics,
            target_match_gt_intrinsics,
            precomputed_warp,
            precomputed_warp_mask,
        ) = super().get_inputs(batch)
        motion_extrinsics, trajs_2d, trajs_3d, valids, visibs = self.get_motion_inputs(batch, meta_data, scale)
        track_query_points, track_vis, track_pos_masks = self.get_track_inputs(batch)

        rgb_mask = batch.get("rgb_mask", None)
        if rgb_mask is not None:
            rgb_mask = rgb_mask.to(self.device)
        prompt_extrinsics = batch.get("prompt_extrinsics", None)
        if prompt_extrinsics is not None:
            prompt_extrinsics = prompt_extrinsics.to(self.device)
            if scale is not None:
                prompt_extrinsics[..., :3, 3] = self.normalize(prompt_extrinsics[..., :3, 3], scale[..., 0, 0])

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
                meta_data=meta_data,
            )

            def _split_optional(data):
                if data is None:
                    return None
                return data[:, :-novel_view_nums].contiguous()

            motion_extrinsics = _split_optional(motion_extrinsics)
            rgb_mask = _split_optional(rgb_mask)
            prompt_extrinsics = _split_optional(prompt_extrinsics)
            assert track_query_points is None
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
            w2c=w2c,
            ray_world=ray_world,
            query_points=track_query_points[:, 0] if track_query_points is not None else None,
            meta_data=meta_data,
            novel_intrinsics=novel_intrinsics,
            novel_w2c=novel_extrinsics,
            rgb_mask=rgb_mask,
        )

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
        pose_enc = results.get("pose_enc")
        gaussians = results.get("ffgs_gaussians")
        render_rgb = results.get("ffgs_render_rgb")
        render_depth = results.get("ffgs_render_depth")
        render_normal = results.get("ffgs_render_normal")
        pred_ffgs_depth = results.get("ffgs_depth")
        pred_ffgs_conf = results.get("ffgs_confidence")
        pred_scene_flow = results.get("scene_flow", None)
        pred_dynamic_pointmap = results.get("dynamic_pointmap", None)
        match_results = results.get("match")

        total_loss, total_loss_dict = 0, dict()
        if pred_ray_directions is not None:
            pred_ray_directions = pred_ray_directions / pred_ray_directions.norm(dim=2, keepdim=True).clip(min=1e-8)
        if pred_local_depth is not None and pred_local_depth.shape[-3] == 1:
            if self.points_from_ray:
                pred_local_depth = pred_local_depth * pred_ray_directions
            else:
                pred_local_depth = self.depth_to_points(pred_local_depth, K=intrinsics)
        if pred_ffgs_depth is not None and pred_ffgs_depth.shape[-3] == 1:
            if self.points_from_ray:
                pred_ffgs_depth = pred_ffgs_depth * pred_ray_directions
            else:
                pred_ffgs_depth = self.depth_to_points(pred_ffgs_depth, K=intrinsics)
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
            bsz, seq, ch, h, w = tmp_pred_depth.shape
            assert ch == 1
            with torch.cuda.amp.autocast(False):
                pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                    pose_encoding=pose_enc[-1].float() if isinstance(pose_enc, (list, tuple)) else pose_enc.float(),
                    image_size_hw=(h, w),
                    build_intrinsics=True,
                )
                tmp_pred_depth = self.depth_to_points(tmp_pred_depth.float(), K=pred_intrinsics)
                tmp_pred_depth = pred_extrinsics.inverse() @ torch.cat([tmp_pred_depth, tmp_pred_depth.new_ones([bsz, seq, 1, h, w])], dim=2).reshape(bsz, seq, 4, -1)
                tmp_pred_depth = tmp_pred_depth.reshape(bsz, seq, 4, h, w)[:, :, :3, :, :]
            tmp_pred_conf = pred_local_conf.squeeze(2).unsqueeze(-1) if pred_local_conf is not None else None
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

        if pred_scene_flow is not None or pred_dynamic_pointmap is not None:
            original_h = meta_data["origin_height"][0].item() if "origin_height" in meta_data else image.shape[-2]
            original_w = meta_data["origin_width"][0].item() if "origin_width" in meta_data else image.shape[-1]
            motion_ref_frame_idx = results.get("src_frame_idx", 0)
            loss, loss_dict = compute_motion_head_loss(
                name=name,
                pred_scene_flow=pred_scene_flow,
                pred_dynamic_pointmap=pred_dynamic_pointmap,
                motion_prediction_type=self._motion_prediction_type,
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
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="motion")

        if match_results is not None and self.dense_match_loss is not None:
            pair_idx = match_results.pop("pair_idx")
            if precomputed_warp is not None:
                precomputed_pair_idx = meta_data.get("pair_idx", None)
                if precomputed_pair_idx is not None:
                    assert len(pair_idx) == len(precomputed_pair_idx)
                    assert pair_idx == precomputed_pair_idx
            if target_match_gt_depth is not None:
                assert target_match_gt_intrinsics is not None
                match_gt_depth = target_match_gt_depth
                match_gt_intrinsics = target_match_gt_intrinsics
            else:
                match_gt_depth = target_local_depth
                match_gt_intrinsics = intrinsics
            loss, loss_dict = self.get_dense_match_loss(
                name=name,
                pred_match=match_results,
                H=image.shape[-2],
                W=image.shape[-1],
                warp_gt=precomputed_warp,
                warp_mask_gt=precomputed_warp_mask,
                meta_data=meta_data,
                depths=match_gt_depth,
                intrinsics=match_gt_intrinsics,
                extrinsics=target_match_gt_extrinsics,
                pair_idx=pair_idx,
            )
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="match")

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
                bsz, view, h, w = unproject_masks.shape
                unproject_masks = unproject_masks.unsqueeze(2)
            novel_render_rgb = results["ffgs_novel_render_rgb"] * unproject_masks.float()
            novel_render_depth = results.get("ffgs_novel_render_depth", None)
            novel_render_normal = results.get("ffgs_novel_render_normal", None)
            if novel_render_depth is not None:
                novel_render_depth = novel_render_depth.view(bsz, view, 1, *novel_render_depth.shape[-2:])
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
        total_loss_dict["aspect_ratio"] = round(image.shape[-1] / image.shape[-2], 2)
        total_loss_dict["bs"] = int(image.shape[0])
        total_loss_dict["view"] = int(image.shape[1])
        return total_loss, total_loss_dict

    @torch.no_grad()
    def infer(self, **batch):
        self.eval()
        (
            _name,
            _total_iter,
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
            target_match_gt_depth,
            target_match_gt_extrinsics,
            target_match_gt_intrinsics,
            precomputed_warp,
            precomputed_warp_mask,
        ) = super().get_inputs(batch)
        motion_extrinsics, trajs_2d, trajs_3d, valids, visibs = self.get_motion_inputs(batch, meta_data, scale)
        track_query_points, _track_vis, _ = self.get_track_inputs(batch)
        rgb_mask = batch.get("rgb_mask", None)
        if rgb_mask is not None:
            rgb_mask = rgb_mask.to(self.device)
        prompt_extrinsics = batch.get("prompt_extrinsics", None)
        if prompt_extrinsics is not None:
            prompt_extrinsics = prompt_extrinsics.to(self.device)
            if scale is not None:
                prompt_extrinsics[..., :3, 3] = self.normalize(prompt_extrinsics[..., :3, 3], scale[..., 0, 0])
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
                    rgb_mask=rgb_mask,
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
                rgb_mask=rgb_mask,
            )

        from hAlgorithm.modules.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from hAlgorithm.modules.pipelines2.utils.outputs import DenseMatchingOutput
        from hAlgorithm.modules.models2.external.romav2.models.romav2.geometry import bhwc_interpolate
        from hAlgorithm.modules.utils.ray import recover_pinhole_intrinsics_from_ray_directions

        pose_enc = results.get("pose_enc", None)
        if pose_enc is not None and isinstance(pose_enc, (list, tuple)):
            pose_enc = pose_enc[-1]
        if pose_enc is not None:
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
            pred_extrinsics, pred_intrinsics = None, None

        pred_local_depth = results.get("depth")
        pred_local_conf = results.get("confidence")
        pred_local_normal = results.get("normal")
        pred_local_invalid_mask = results.get("invalid_mask")
        pred_global_points = results.get("global_points")
        pred_global_conf = results.get("global_confidence")
        pred_ray_directions = results.get("ray")
        pred_track = results.get("track")
        if pred_track is not None and isinstance(pred_track, (list, tuple)):
            pred_track = pred_track[-1]
        gaussians = results.get("ffgs_gaussians")
        render_rgb = results.get("ffgs_render_rgb")
        render_depth = results.get("ffgs_render_depth")
        if pred_local_depth is None:
            pred_local_depth = results.get("ffgs_depth")
            pred_local_conf = results.get("ffgs_confidence")

        pred_matching = results.get("match", None)
        if pred_matching is not None:
            pair_idx = pred_matching.pop("pair_idx")
            num_pair = len(pair_idx)
            warp_AB = confidence_AB = coarse_warp_AB = coarse_confidence_AB = None
            overlap_AB = precision_AB = coarse_overlap_AB = None
            if "final" in pred_matching:
                warp_AB, confidence_AB = pred_matching["final"]["warp_AB"], pred_matching["final"]["confidence_AB"]
            if "coarse" in pred_matching:
                coarse_warp_AB, coarse_confidence_AB = pred_matching["coarse"]["warp_AB"], pred_matching["coarse"]["confidence_AB"]
                coarse_overlap_AB = coarse_confidence_AB
            if confidence_AB is not None:
                overlap_AB = confidence_AB[..., :1].sigmoid()
                precision_AB = confidence_AB[..., 1:4]
                overlap_AB[overlap_AB > 0.5] = 1
                overlap_AB[overlap_AB < 0.5] = 0
            if precomputed_warp is not None:
                match_warp = precomputed_warp.get(1, None)
                match_warp_mask = precomputed_warp_mask.get(1, None) if precomputed_warp_mask is not None else None
            else:
                if target_match_gt_depth is not None:
                    match_gt_depth = target_match_gt_depth
                    match_gt_intrinsics = target_match_gt_intrinsics
                elif target_local_depth is not None:
                    match_gt_depth = target_local_depth
                    match_gt_intrinsics = intrinsics
                else:
                    match_gt_depth = None
                if match_gt_depth is not None:
                    camera_type = meta_data.get("camera_type", ["PINHOLE"])[0]
                    if isinstance(camera_type, (list, tuple)):
                        camera_type = camera_type[0]
                    warp_scales, warp_mask_scales = self.get_warp_gt(
                        depths=match_gt_depth,
                        intrinsics=match_gt_intrinsics,
                        extrinsics=target_match_gt_extrinsics,
                        pair_idx=pair_idx,
                        H=image.shape[-2],
                        W=image.shape[-1],
                        camera_type=camera_type,
                        meta_data=meta_data,
                    )
                    match_warp, match_warp_mask = warp_scales[1], warp_mask_scales[1]
                else:
                    match_warp = match_warp_mask = None
        else:
            pair_idx = []
            num_pair = 0
            warp_AB = overlap_AB = precision_AB = None
            coarse_warp_AB = coarse_overlap_AB = None
            match_warp = match_warp_mask = None

        if prompt_depth is not None and scale is not None:
            prompt_depth = self.denormalize(prompt_depth, scale=scale)
        if target_local_depth is not None and scale is not None:
            target_local_depth = self.denormalize(target_local_depth, scale=scale)
        if target_global_points is not None and scale is not None:
            target_global_points = self.denormalize(target_global_points, scale=scale)
        if extrinsics is not None and scale is not None:
            extrinsics[..., :3, 3] = self.denormalize(extrinsics[..., :3, 3], scale=scale[..., 0, 0])
        if pred_ray_directions is not None:
            pred_ray_directions = pred_ray_directions / pred_ray_directions.norm(dim=2, keepdim=True).clip(min=1e-8)
            if self.save_output_cfg.get("output_intrinsics_from_ray", False):
                pred_intrinsics = recover_pinhole_intrinsics_from_ray_directions(
                    ray_directions=pred_ray_directions.view(-1, *pred_ray_directions.shape[2:]).permute(0, 2, 3, 1).contiguous()
                )
                pred_intrinsics = pred_intrinsics.view(*pred_ray_directions.shape[:2], *pred_intrinsics.shape[1:])
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

        if image_show is not None and hasattr(image_show, "shape") and image_show.ndim == 5:
            _, _, c_show, h_show, w_show = image_show.shape
        else:
            c_show, h_show, w_show = 3, image.shape[-2], image.shape[-1]
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
                if num_pair > 0 and pred_matching is not None:
                    cur_pairs = [(pair_i, pair) for pair_i, pair in enumerate(pair_idx) if pair[0] == index]
                    matching_results = []
                    for pair_i, cur_pair in cur_pairs:
                        if image_show is not None and hasattr(image_show, "shape") and image_show.ndim == 5:
                            image0 = np.ascontiguousarray(
                                image_show[:, cur_pair[0]].reshape(-1, c_show, h_show, w_show).transpose(0, 2, 3, 1).astype(np.uint8)
                            )
                            image1 = np.ascontiguousarray(
                                image_show[:, cur_pair[1]].reshape(-1, c_show, h_show, w_show).transpose(0, 2, 3, 1).astype(np.uint8)
                            )
                        else:
                            image0 = np.ascontiguousarray((((image[0, cur_pair[0]].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8))[None])
                            image1 = np.ascontiguousarray((((image[0, cur_pair[1]].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8))[None])
                        matching_results.append(
                            DenseMatchingOutput(
                                image0=image0,
                                image1=image1,
                                warp=warp_AB[:, pair_i] if warp_AB is not None else None,
                                overlap=overlap_AB[:, pair_i] if overlap_AB is not None else None,
                                warp_coarse=bhwc_interpolate(coarse_warp_AB[:, pair_i], (h_show, w_show)) if coarse_warp_AB is not None else None,
                                overlap_coarse=bhwc_interpolate(coarse_overlap_AB[:, pair_i], (h_show, w_show)) if coarse_overlap_AB is not None else None,
                                warp_gt=match_warp[:, pair_i] if match_warp is not None else None,
                                overlap_gt=match_warp_mask[:, pair_i][..., None] if match_warp_mask is not None else None,
                                pred_covariance=precision_AB[:, pair_i] if precision_AB is not None else None,
                                extrinsics_image0=target_match_gt_extrinsics[:, cur_pair[0]].cpu()[0].numpy() if target_match_gt_extrinsics is not None else None,
                                extrinsics_image1=target_match_gt_extrinsics[:, cur_pair[1]].cpu()[0].numpy() if target_match_gt_extrinsics is not None else None,
                                intrinsics_image0=intrinsics[:, cur_pair[0]].cpu()[0].numpy() if intrinsics is not None else None,
                                intrinsics_image1=intrinsics[:, cur_pair[1]].cpu()[0].numpy() if intrinsics is not None else None,
                            )
                        )
                    single.dense_matching = matching_results
                single.frame_index = fi
                single.view_index = vi
                single.total_index = index
                mv_outputs.append(single)

        pred_scene_flow = results.get("scene_flow", None)
        pred_dynamic_pointmap = results.get("dynamic_pointmap", None)
        if pred_dynamic_pointmap is not None and pred_scene_flow is None:
            pred_scene_flow = convert_pointmap_to_scene_flow(pred_dynamic_pointmap)
        if pred_scene_flow is not None and scale is not None:
            pred_scene_flow = self.denormalize(pred_scene_flow, scale=scale[:, 0:1, :, :, :])
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

    def visualize(self, outputs_list, meta_data, out_dir):
        super().visualize(outputs_list, meta_data, out_dir)
        if len(outputs_list) == 0:
            return
        has_scene_flow = hasattr(outputs_list[0], "scene_flow_pred") and outputs_list[0].scene_flow_pred is not None
        if not has_scene_flow:
            return
        data_idx = meta_data["data_idx"][0]
        motion_out_dir = os.path.join(out_dir, f"motion/{data_idx:06d}")
        if self.save_output_cfg.get("save_motion_results", False):
            os.makedirs(motion_out_dir, exist_ok=True)
            vis_motion_head_results(
                cfg=self.save_output_cfg,
                mv_outputs=outputs_list,
                motion_out_dir=motion_out_dir,
                data_idx=data_idx,
                meta_data=meta_data,
            )
        if self.save_output_cfg.get("save_motion_3d", False):
            vis_motion_3d_rerun(
                cfg=self.save_output_cfg,
                mv_outputs=outputs_list,
                out_dir=out_dir,
                data_idx=data_idx,
                meta_data=meta_data,
            )
