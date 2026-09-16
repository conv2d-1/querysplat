import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from hAlgorithm.modules.pipelines2.mvfr_v1 import MVFRPipeline as MVFRPipelineV1
from hAlgorithm.modules.pipelines2.utils.visualize_mv import vis_match_results
from hAlgorithm.modules.utils.ray import recover_pinhole_intrinsics_from_ray_directions
from hAlgorithm.utils import instantiate_from_config


class MVFRPipeline(MVFRPipelineV1):
    """Pipeline for Multi-View Feed-forward Reconstruction with Dense Matching.
    Extends MVFRPipelineV1 with dense matching training and inference capabilities.
    """

    def __init__(
        self,
        target_match_gt_depth_name=None,
        target_match_gt_depth_intrinsics_name=None,
        target_match_gt_extrinsics_name=None,
        match_scales=None,
        dense_match_loss=None,
        match_precision_type="pixel_conf",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_match_gt_depth_name = target_match_gt_depth_name
        self.target_match_gt_depth_intrinsics_name = target_match_gt_depth_intrinsics_name
        self.target_match_gt_extrinsics_name = target_match_gt_extrinsics_name
        self.match_scales = match_scales
        self.dense_match_loss = instantiate_from_config(dense_match_loss)
        self.match_precision_type = match_precision_type
        self.save_output_cfg.setdefault("save_match", self.save_output_cfg["save_everything"])
    
    def get_camera_type(self, meta_data):
        if "camera_type" in meta_data:
            camera_type = meta_data["camera_type"][0][0]
        else:
            camera_type = "PINHOLE"
        return camera_type

    def get_match_inputs(self, batch):
        """Extract match GT inputs from batch."""
        target_match_gt_depth = target_match_gt_depth_intrinsics = target_match_gt_extrinsics = None
        if self.target_match_gt_depth_name is not None and self.target_match_gt_depth_name in batch:
            target_match_gt_depth = batch[self.target_match_gt_depth_name].to(self.device)
        if self.target_match_gt_extrinsics_name is not None and self.target_match_gt_extrinsics_name in batch:
            target_match_gt_extrinsics = batch[self.target_match_gt_extrinsics_name].to(self.device)
        if self.target_match_gt_depth_intrinsics_name is not None and self.target_match_gt_depth_intrinsics_name in batch:
            target_match_gt_depth_intrinsics = batch[self.target_match_gt_depth_intrinsics_name].to(self.device)

        if batch["image"].ndim == 4:
            def add_dim(x, dim=1):
                return x.unsqueeze(dim) if x is not None else None
            target_match_gt_depth = add_dim(target_match_gt_depth)
            target_match_gt_extrinsics = add_dim(target_match_gt_extrinsics)
            target_match_gt_depth_intrinsics = add_dim(target_match_gt_depth_intrinsics)

        return target_match_gt_depth, target_match_gt_depth_intrinsics, target_match_gt_extrinsics

    def get_warp_gt(self, depths, intrinsics, extrinsics, pair_idx, camera_type="PINHOLE", meta_data=None):
        warp_scales = {scale: [] for scale in self.match_scales}
        warp_mask_scales = {scale: [] for scale in self.match_scales}
        B, V, C, H, W = depths.shape
        for pair in pair_idx:
            depth1 = depths[:, pair[0], -1]
            depth2 = depths[:, pair[1], -1]

            K1 = intrinsics[:, pair[0]]
            K2 = intrinsics[:, pair[1]]

            T1 = extrinsics[:, pair[0]]
            T2 = extrinsics[:, pair[1]]

            T_1to2 = T2 @ T1.inverse()
            for scale in self.match_scales:
                h1, w1 = int(H / scale), int(W / scale)
                if camera_type == "FISHEYE_BLENDER":
                    from hAlgorithm.datasets_fisheye.utils.fisheye_warp import get_gt_warp_fisheye_blender as blender_depth_to_warp
                    distort_k = meta_data["distort_k"]
                    sensor_size = meta_data["sensor_size"]
                    crop_offset = meta_data.get("crop_offset", None)
                    assert distort_k is not None and sensor_size is not None

                    distort_k = distort_k.to(depth1.device)
                    sensor_size = sensor_size.to(depth1.device)
                    if crop_offset is not None:
                        crop_offset = crop_offset.to(depth1.device)

                    if distort_k.dim() == 3:
                        k_coeffs1 = distort_k[:, pair[0]]
                        k_coeffs2 = distort_k[:, pair[1]]
                    else:
                        k_coeffs1 = k_coeffs2 = distort_k

                    if sensor_size.dim() == 3:
                        ss1 = sensor_size[:, pair[0]]
                        ss2 = sensor_size[:, pair[1]]
                    else:
                        ss1 = ss2 = sensor_size

                    co1 = co2 = 0
                    if crop_offset is not None:
                        if crop_offset.dim() == 3:
                            co1 = crop_offset[:, pair[0]]
                            co2 = crop_offset[:, pair[1]]
                        else:
                            co1 = co2 = crop_offset
                    warp, warp_mask = blender_depth_to_warp(
                        depth1, depth2, T_1to2, K1, K2,
                        k_coeffs1, k_coeffs2, ss1, ss2,
                        crop_offset1=co1, crop_offset2=co2,
                        depth_interpolation_mode='bilinear',
                        H=h1, W=w1,
                    )
                elif camera_type == "FISHEYE_EQUIDISTANT":
                    from hAlgorithm.datasets_fisheye.utils.fisheye_warp import get_gt_warp_fisheye_equidistant as fisheye_depth_to_warp
                    warp, warp_mask = fisheye_depth_to_warp(
                        depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode='bilinear',
                        H=h1, W=w1,
                    )
                else:
                    from hAlgorithm.modules.models2.external.romav2.utils.utils import get_gt_warp as romav1_depth_to_warp
                    warp, warp_mask = romav1_depth_to_warp(
                        depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode='bilinear',
                        H=h1, W=w1,
                    )
                warp_scales[scale].append(warp.to(dtype=depths.dtype))
                warp_mask_scales[scale].append(warp_mask)
        for k, v in warp_scales.items():
            warp_scales[k] = torch.stack(v, dim=1)
        for k, v in warp_mask_scales.items():
            warp_mask_scales[k] = torch.stack(v, dim=1)
        return warp_scales, warp_mask_scales

    def get_dense_match_loss(
        self,
        name,
        pred_match,
        warp_gt=None,
        warp_mask_gt=None,
        depths=None,
        intrinsics=None,
        extrinsics=None,
        pair_idx=None,
        camera_type="PINHOLE",
        meta_data=None,
    ):
        if isinstance(camera_type, (list, tuple)):
            camera_type = camera_type[0]

        if warp_gt is None:
            assert depths is not None
            warp_gt, warp_mask_gt = self.get_warp_gt(
                depths, intrinsics, extrinsics, pair_idx,
                camera_type=camera_type, meta_data=meta_data,
            )

        return self.dense_match_loss(
            pred_match,
            warp_gt,
            warp_mask_gt,
            pair_idx,
            name=name,
        )

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

        camera_type = self.get_camera_type(meta_data)

        target_match_gt_depth, target_match_gt_depth_intrinsics, target_match_gt_extrinsics = self.get_match_inputs(batch)
        if target_match_gt_depth_intrinsics is not None:
            target_match_gt_depth_intrinsics = intrinsics

        track_query_points, track_vis, track_pos_masks = self.get_track_inputs(batch)

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
        match_results = results.get("match")

        total_loss, total_loss_dict = 0, dict()

        if pred_ray_directions is not None:
            pred_ray_directions = pred_ray_directions / pred_ray_directions.norm(dim=2, keepdim=True).clip(min=1e-8)

        if pred_local_depth is not None and pred_local_depth.shape[-3] == 1:
            if self.points_from_ray:
                pred_local_depth = pred_local_depth * ray_directions
            else:
                pred_local_depth = self.depth_to_points(pred_local_depth, K=intrinsics, camera_type=camera_type)

        if pred_ffgs_depth is not None and pred_ffgs_depth.shape[-3] == 1:
            if self.points_from_ray:
                pred_ffgs_depth = pred_ffgs_depth * ray_directions
            else:
                pred_ffgs_depth = self.depth_to_points(pred_ffgs_depth, K=intrinsics, camera_type=camera_type)

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

        # Dense match loss
        if match_results is not None and self.dense_match_loss is not None:
            pair_idx = match_results.pop("pair_idx")
            precomputed_warp = batch.get("warp_scales", None)
            precomputed_warp_mask = batch.get("warp_mask_scales", None)
            if precomputed_warp is not None:
                precomputed_pair_idx = meta_data.get("pair_idx", None)
                if precomputed_pair_idx is not None:
                    assert len(pair_idx) == len(precomputed_pair_idx)
                    precomputed_pair_idx = [(pa[0][0].item(), pa[1][0].item()) for pa in precomputed_pair_idx]
                    assert pair_idx == precomputed_pair_idx
                precomputed_warp = {
                    k: v.to(device=self.device, dtype=image.dtype) for k, v in precomputed_warp.items()
                }
                precomputed_warp_mask = {
                    k: v.to(device=self.device) for k, v in precomputed_warp_mask.items()
                }
            loss, loss_dict = self.get_dense_match_loss(
                name=name, pred_match=match_results,
                warp_gt=precomputed_warp, warp_mask_gt=precomputed_warp_mask, meta_data=meta_data,
                depths=target_match_gt_depth, intrinsics=target_match_gt_depth_intrinsics,
                extrinsics=target_match_gt_extrinsics, pair_idx=pair_idx,
            )
            total_loss, total_loss_dict = self.add_loss(total_loss, total_loss_dict, loss, loss_dict, task_name="match")

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

        camera_type = self.get_camera_type(meta_data)

        target_match_gt_depth, target_match_gt_depth_intrinsics, target_match_gt_extrinsics = self.get_match_inputs(batch)
        if target_match_gt_depth_intrinsics is not None:
            target_match_gt_depth_intrinsics = intrinsics

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
        if pred_local_depth is None:
            pred_local_depth = results.get("ffgs_depth")
            pred_local_conf = results.get("ffgs_confidence")

        # Dense matching post-processing
        pred_matching = results.get("match", None)
        if pred_matching is not None:
            from hAlgorithm.modules.models2.external.romav2.models.romav2.romav2 import _map_confidence
            from hAlgorithm.modules.models2.external.romav2.models.romav2.geometry import bhwc_interpolate

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
                if self.match_precision_type == "roma":
                    overlap_AB, precision_AB = _map_confidence(
                        confidence=confidence_AB, threshold=None
                    )
                elif self.match_precision_type == "pixel_conf":
                    overlap_AB = confidence_AB[..., :1]
                    precision_AB = confidence_AB[..., 1:4]

            if target_match_gt_depth is not None:
                warp_scales, warp_mask_scales = self.get_warp_gt(
                    depths=target_match_gt_depth, 
                    intrinsics=target_match_gt_depth_intrinsics,
                    extrinsics=target_match_gt_extrinsics, 
                    pair_idx=pair_idx,
                    camera_type=camera_type, 
                    meta_data=meta_data,
                )
                warp, warp_mask = warp_scales[1], warp_mask_scales[1]
            else:
                warp = warp_mask = None
        else:
            pair_idx = []
            num_pair = 0
            warp_AB = confidence_AB = coarse_warp_AB = coarse_confidence_AB = None
            overlap_AB = precision_AB = coarse_overlap_AB = None
            warp = warp_mask = None

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

                if pred_track is not None:
                    single.track_pred = pred_track[0, index].cpu().numpy()
                    single.track_vis_pred = pred_track_vis[0, index].cpu().numpy()
                    single.track_pred_local_conf = pred_track_conf[0, index].cpu().numpy()

                    breakpoint()
                    single.track_gt = track_query_points[0, index].cpu().numpy()
                    if track_vis is not None:
                        single.track_vis = track_vis[0, index].cpu().numpy()

                if gaussians is not None and index == 0:
                    single.gaussians = gaussians

                single.rgb = (image[0, index].permute(1, 2, 0).float().cpu().numpy() + 1) * 0.5

                if render_rgb is not None:
                    single.render_rgb = render_rgb[0, index].cpu().numpy().transpose(1, 2, 0)

                if render_depth is not None:
                    single.render_depth = render_depth[0, index].cpu().numpy()

                # Dense match outputs
                if num_pair > 0 and pred_matching is not None and image_show is not None:
                    from hAlgorithm.modules.pipelines2.utils.outputs import DenseMatchingOutput
                    from hAlgorithm.modules.models2.external.romav2.models.romav2.geometry import bhwc_interpolate

                    B_show, V_show, C_show, H_show, W_show = image_show.shape
                    cur_pairs = [(pair_i, pair) for pair_i, pair in enumerate(pair_idx) if pair[0] == index]
                    matching_results = []
                    for pair_i, cur_pair in cur_pairs:
                        image0 = np.ascontiguousarray(image_show[:, cur_pair[0]].reshape(-1, C_show, H_show, W_show).transpose(0, 2, 3, 1).astype(np.uint8))
                        image1 = np.ascontiguousarray(image_show[:, cur_pair[1]].reshape(-1, C_show, H_show, W_show).transpose(0, 2, 3, 1).astype(np.uint8))

                        matching_results.append(DenseMatchingOutput(
                            image0=image0,
                            image1=image1,
                            warp=warp_AB[:, pair_i] if warp_AB is not None else None,
                            overlap=overlap_AB[:, pair_i] if overlap_AB is not None else None,
                            warp_coarse=bhwc_interpolate(coarse_warp_AB[:, pair_i], (H_show, W_show)) if coarse_warp_AB is not None else None,
                            overlap_coarse=bhwc_interpolate(coarse_overlap_AB[:, pair_i], (H_show, W_show)) if coarse_overlap_AB is not None else None,
                            warp_gt=warp[:, pair_i] if warp is not None else None,
                            overlap_gt=warp_mask[:, pair_i][..., None] if warp_mask is not None else None,
                            pred_covariance=precision_AB[:, pair_i] if precision_AB is not None else None,
                            extrinsics_image0=target_match_gt_extrinsics[:, cur_pair[0]].cpu()[0].numpy() if target_match_gt_extrinsics is not None else None,
                            extrinsics_image1=target_match_gt_extrinsics[:, cur_pair[1]].cpu()[0].numpy() if target_match_gt_extrinsics is not None else None,
                            intrinsics_image0=intrinsics[:, cur_pair[0]].cpu()[0].numpy() if intrinsics is not None else None,
                            intrinsics_image1=intrinsics[:, cur_pair[1]].cpu()[0].numpy() if intrinsics is not None else None,
                        ))
                    single.dense_matching = matching_results

                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                mv_outputs.append(single)

        return mv_outputs

    def get_out_dir(self, out_dir, data_idx=None):
        gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir = super().get_out_dir(out_dir, data_idx=data_idx)

        if data_idx is not None:
            match_out_dir = os.path.join(out_dir, f"match/{data_idx:06d}")
        else:
            match_out_dir = os.path.join(out_dir, "match")

        return gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir, match_out_dir

    def get_gt_out_dir(self, out_dir):
        if self.save_output_cfg["gt_out_dir"] is None:
            return None, None, None, None

        abs_gt_out_dir = os.path.join(os.path.dirname(os.path.dirname(out_dir)), self.save_output_cfg["gt_out_dir"], os.path.basename(out_dir))
        _, gt_glb_out_dir, _, _, gt_track_out_dir, gt_match_out_dir = self.get_out_dir(abs_gt_out_dir, data_idx=None)

        return abs_gt_out_dir, gt_glb_out_dir, gt_track_out_dir, gt_match_out_dir

    def visualize(self, outputs_list, meta_data, out_dir):
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        data_idx = meta_data["data_idx"][0]

        gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir, match_out_dir = self.get_out_dir(out_dir, data_idx=data_idx)
        gt_out_dir, gt_glb_out_dir, gt_track_out_dir, gt_match_out_dir = self.get_gt_out_dir(out_dir)

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

        if gt_out_dir is not None:
            vis_images(cfg=self.save_output_cfg, mv_outputs=outputs_list, gt_out_dir=gt_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num)

        if self.save_output_cfg["save_local_results"]:
            os.makedirs(mv_out_dir, exist_ok=True)
            vis_local_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, mv_out_dir=mv_out_dir, frame_num=frame_num, view_num=view_num, gt_out_dir=gt_out_dir, meta_data=meta_data)
        if self.save_output_cfg["save_extra_local_results"]:
            os.makedirs(mv_out_dir, exist_ok=True)
            vis_extra_local_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, mv_out_dir=mv_out_dir, frame_num=frame_num, view_num=view_num, meta_data=meta_data)

        if self.save_output_cfg["save_glb_results"]:
            vis_glb_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, glb_out_dir=glb_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num, gt_out_dir=gt_glb_out_dir)

        if self.save_output_cfg["save_cameras"]:
            vis_camera_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, camera_out_dir=camera_out_dir, frame_num=frame_num, view_num=view_num)

        if self.save_output_cfg["save_gaussians"] and outputs_list[0].gaussians is not None:
            vis_render(cfg=self.save_output_cfg, mv_outputs=outputs_list, gs_out_dir=gs_out_dir, data_idx=data_idx)
            vis_render_video(cfg=self.save_output_cfg, model=self.model, mv_outputs=outputs_list, gs_out_dir=gs_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num, device=self.device)

        if self.save_output_cfg["save_track_results"] and outputs_list[0].track_pred is not None:
            os.makedirs(track_out_dir, exist_ok=True)
            vis_track_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, track_out_dir=track_out_dir, data_idx=data_idx, gt_out_dir=gt_track_out_dir)

        if self.save_output_cfg.get("save_match"):
            os.makedirs(match_out_dir, exist_ok=True)
            vis_match_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, match_out_dir=match_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num, gt_out_dir=gt_match_out_dir, overlap_act=True)