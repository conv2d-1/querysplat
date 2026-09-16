import logging
import traceback
from collections import defaultdict

import torch
import torch.nn as nn

from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from hAlgorithm.modules.utils.ray import compute_rays
from hAlgorithm.utils import instantiate_from_config

from .base import MVBase


class MVGS(MVBase):
    def __init__(
        self,
        gaussian_head=None,
        gaussian_render=None,
        gaussian_cfg=None,
        **kwargs,
    ):
        super(MVGS, self).__init__(**kwargs)

        # Gaussian Decoder
        self.gaussian_head = self._instantiate_and_register(gaussian_head, "gaussian_head")
        if self.gaussian_head is not None: 
            self.gaussian_render = self._instantiate_and_register(gaussian_render, "gaussian_render")

            self.register_buffer("gs_near", torch.tensor([[float(gs_near)]]))
            self.register_buffer("gs_far", torch.tensor([[float(gs_far)]]))

            self.gs_features = self.gaussian_cfg["features"]
            self.gs_near = self.gaussian_cfg("near", 0.005)
            self.gs_far = self.gaussian_cfg("far", 100.0)
            self.gs_rgb_with_ray = self.gaussian_cfg.get("rgb_with_ray", False)
            self.normalize_cameras = self.gaussian_cfg.get("normalize_cameras", True)

    def depth_denormalize(self, depth, scale=None, center=None):
        # Adjust depth values based on scale and center
        if scale is not None: depth = depth * scale
        if center is not None: depth = depth + center
        return depth
            
    def get_cameras(self, results, hw, prompt_scale):
        # Pose encoding for camera extrinsics/intrinsics
        with torch.autocast("cuda", dtype=torch.float32):
            pose_enc = results["pose_enc"]

            if isinstance(pose_enc, (list, tuple)):
                pose_enc = pose_enc[-1]

            w2c, intrinsics = pose_encoding_to_extri_intri(
                pose_encoding=pose_enc,
                image_size_hw=hw,
                translation_scale=prompt_scale,
            )

            if self.normalize_cameras:
                # NOTE: trans to first camera
                w2c_pred = w2c.float()
                base_c2w_pred = w2c_pred[:, 0:1].inverse()
                extrinsics = w2c_pred @ base_c2w_pred

            w2c = w2c.float()
            intrinsics = intrinsics.float()
    
        return w2c, intrinsics

    def forward(
        self,
        rgb=None,
        prompt_depth=None,
        prompt_scale=None,
        prompt_center=None,
        w2c=None,
        intrinsics=None,
        novel_extrinsics=None,
        novel_intrinsics=None,
        query_points=None,
        with_freeze=False,
        meta_data=None,
        only_rendering=False,
        render_video_with_pred_camera=False,
        **kwargs,
    ):
        if only_rendering:
            return rendering(
                gaussian_render=self.gaussian_render,
                gs_near=self.gs_near,
                gs_far=self.gs_far,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                **kwargs,
            )

        b, n, c, h, w = rgb.shape

        patch_features, prompt_depth, prompt_ray, prompt_extra, camera_tokens = self.froward_aggregator(rgb, prompt_depth, prompt_scale, intrinsics, w2c, meta_data)

        if self.local_depth_head is not None:
            local_depth_results = self.local_depth_head(
                patch_features,
                prompt_features=prompt_features,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
            )
            depth = local_depth_results["pointmap"]
            results["depth"] = depth.view(b, n, *depth.shape[-3:])

            confidence = local_depth_results["confidence"]
            results["confidence"] = confidence.view(b, n, *confidence.shape[-3:])

            local_feature_maps = local_depth_results.get("features", None)

        if self.glb_depth_head is not None:
            glb_depth_results = self.glb_depth_head(
                patch_features,
                prompt_features=prompt_features,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
            )
            glb_pointmap = glb_depth_results["pointmap"]
            results["glb_pointmap"] = glb_pointmap.view(b, n, *glb_pointmap.shape[-3:])

            glb_confidence = glb_depth_results["confidence"]
            results["glb_confidence"] = glb_confidence.view(b, n, *glb_confidence.shape[-3:])

            glb_feature_maps = glb_depth_results.get("features", None)

        if self.camera_head is not None:
            pose_enc = self.camera_head(
                camera_tokens,
                meta_data=meta_data,
            )
            results["pose_enc"] = pose_enc

        if self.track_head is not None and query_points is not None:
            track, track_vis, track_confidence = self.track_head(
                patch_features,
                query_points=query_points,
                prompt_features=prompt_features,
                meta_data=meta_data,
                local_feature_maps=local_feature_maps,
                glb_feature_maps=glb_feature_maps,
            )
            results["track"] = track
            results["track_vis"] = track_vis
            results["track_confidence"] = track_confidence

        if self.gaussian_head is None:
            return results

        gs_features = []
        for feature_name in self.gs_features:
            feature = features_dict[feature_name]
            gs_features.append(feature)
        if len(gs_features) > 0:
            if gs_features[0].ndim == 4:
                gs_features = torch.cat(gs_features, dim=1)  # [B, N*C, H, W]
            elif gs_features[0].ndim == 5:
                gs_features = torch.cat(gs_features, dim=2)  # [B, F*V, N*C, H, W]

        b, n, c, h, w = rgb.shape

        gaussians_input_pointmap = results["depth"]  # [B, F*V, 1, H, W]
        gaussians_input_pointmap = self.depth_denormalize(gaussians_input_pointmap, scale=prompt_scale, center=prompt_center)

        if (
            extrinsics is None
            or intrinsics is None
            or render_video_with_pred_camera
            or self.gs_rgb_with_ray
        ):
            extrinsics_pred, intrinsics_pred = self.get_cameras(
                results, hw=(h, w), prompt_scale=prompt_scale
            )

            if self.gs_rgb_with_ray:
                ray_o, ray_d = compute_rays(
                    c2w=extrinsics.inverse(),
                    fxfycxcy=intrinsics[:, :, [0, 1, 0, 1], [0, 1, 2, 2]],
                    h=h,
                    w=w,
                    device=rgb.device,
                )
                o_cross_d = torch.cross(ray_o, ray_d, dim=2)
                rays = torch.cat([o_cross_d, ray_d], dim=2).to(rgb.dtype)
                rgb = torch.cat([rgb, rays], dim=2)  # NOTE
            else:
                extrinsics = extrinsics_pred
                intrinsics = intrinsics_pred

        extrinsics = extrinsics.clone()
        if not self.extrinsics_c2w:
            extrinsics = extrinsics.inverse()

        intrinsics = intrinsics.clone()
        intrinsics[..., 0, :] /= w
        intrinsics[..., 1, :] /= h

        gs_mask = kwargs.get("gs_mask", None)
        # if gs_mask is None:
        #     gs_mask = results["mv_depth_confidence"] > 0.0

        gaussians = self.gaussian_head(
            images=rgb,
            features=gs_features,
            depth=pointmap,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            prompt_scale=prompt_scale,
        )
        if gs_mask is not None:
            # TODO: support downsample
            if self.gaussian_head.downsample:
                gs_mask = gs_mask[..., ::2, ::2]
            gaussians.apply_mask(gs_mask)
        results["gaussians"] = gaussians

        # rendering novel views
        if novel_extrinsics is not None and novel_intrinsics is not None:
            novel_view_nums = novel_extrinsics.shape[1]
            novel_extrinsics = novel_extrinsics.clone()
            if not self.extrinsics_c2w:
                novel_extrinsics = novel_extrinsics.inverse()

            novel_intrinsics = novel_intrinsics.clone()
            novel_intrinsics[..., 0, :] /= w
            novel_intrinsics[..., 1, :] /= h

            cur_extrinsics = torch.cat([extrinsics, novel_extrinsics], dim=1)
            cur_intrinsics = torch.cat([intrinsics, novel_intrinsics], dim=1)

            novel_render = self.rendering(
                gaussians=gaussians,
                gaussian_render=self.gaussian_render,
                gs_near=self.gs_near,
                gs_far=self.gs_far,
                extrinsics=cur_extrinsics,
                intrinsics=cur_intrinsics,
                hw=(h, w),
            )

            for key in novel_render:
                results[key] = novel_render[key][:, :-novel_view_nums]
                results[key.replace("render_", "render_novel_")] = novel_render[key][
                    :, -novel_view_nums:
                ]
        else:
            # rendering reference views
            render = self.rendering(
                gaussians=gaussians,
                gaussian_render=self.gaussian_render,
                gs_near=self.gs_near,
                gs_far=self.gs_far,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                hw=(h, w),
            )
            results.update(render)


        else:
            return results


def rendering(
    gaussians,
    gaussian_render,
    gs_near,
    gs_far,
    extrinsics,
    intrinsics,
    hw,
    depth_mode="depth",
    **kwargs,
):
    """
    extrinsics: [B, F*V, 4, 4], c2w
    intrinsics: [B, F*V, 3, 3], normalized intrinsics
    """
    with torch.autocast("cuda", dtype=torch.float32):
        b, n = extrinsics.shape[:2]
        near = gs_near.repeat(b, n)
        far = gs_far.repeat(b, n)
        try:
            renders = gaussian_render(
                gaussians,
                extrinsics.clone(),
                intrinsics.clone(),
                near,
                far,
                hw,
                depth_mode=depth_mode,
            )

            results = {}
            results["render_rgb"] = renders.color
            results["render_depth"] = renders.depth

            if hasattr(renders, "alpha"):
                results["render_alpha"] = renders.alpha
            if hasattr(renders, "normal"):
                results["render_normal"] = renders.normal
        except Exception as e:
            results = dict()
            torch.cuda.empty_cache()
            traceback.print_exc()
            logging.error(e)
            logging.warning("gaussian_render error and skip!!")

    return results
