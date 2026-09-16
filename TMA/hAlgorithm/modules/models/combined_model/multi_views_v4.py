import logging
import traceback
from collections import defaultdict

import torch

from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from hAlgorithm.modules.utils.ray import compute_rays
from hAlgorithm.utils import instantiate_from_config

from .base import CombinedModel


class MVCombinedModel(CombinedModel):
    def __init__(
        self,
        rgb_encoder=None,
        prompt_encoder=None,
        decoder=None,
        mv_decoder=None,
        head=None,
        gaussian_head=None,
        gaussian_render=None,
        mv_depth_head=None,
        mv_point_head=None,
        camera_head=None,
        track_head=None,
        freeze_modules=[],
        extrinsics_c2w=False,
    ):
        # Initialize the parent class with necessary components
        super(MVCombinedModel, self).__init__(
            rgb_encoder=rgb_encoder,
            prompt_encoder=prompt_encoder,
            decoder=decoder,
            head=head,
            freeze_modules=freeze_modules,
        )

        # Store configurations and parameters
        self.mv_decoder_cfg = mv_decoder
        self.gaussian_head_cfg = gaussian_head
        self.gaussian_render_cfg = gaussian_render
        self.mv_depth_head_cfg = mv_depth_head
        self.mv_point_head_cfg = mv_point_head
        self.camera_head_cfg = camera_head
        self.track_head_cfg = track_head
        self.extrinsics_c2w = extrinsics_c2w

        self.build_mv_decoder()
        self.build_mv_head()

    def build_mv_decoder(self):
        # Instantiate the decoder module using the updated configuration.
        # If the instantiation is successful, add its name to the module_names list..
        self.mv_decoder = instantiate_from_config(self.mv_decoder_cfg)
        if self.mv_decoder is not None:
            self.module_names.append("mv_decoder")

    def build_mv_head(self):
        self.gaussian_head = instantiate_from_config(self.gaussian_head_cfg)
        if self.gaussian_head is not None:
            self.module_names.append("gaussian_head")
            self.gaussian_render = instantiate_from_config(self.gaussian_render_cfg)

        self.mv_depth_head = instantiate_from_config(self.mv_depth_head_cfg)
        if self.mv_depth_head is not None:
            self.module_names.append("mv_depth_head")

        self.mv_point_head = instantiate_from_config(self.mv_point_head_cfg)
        if self.mv_point_head is not None:
            self.module_names.append("mv_point_head")

        self.camera_head = instantiate_from_config(self.camera_head_cfg)
        if self.camera_head is not None:
            self.module_names.append("camera_head")

        self.track_head = instantiate_from_config(self.track_head_cfg)
        if self.track_head is not None:
            self.module_names.append("track_head")

    def point_denormalize(self, depth, scale=None, center=None):
        # Adjust depth values based on scale and center if provided
        if scale is not None:
            depth = depth * scale
        if center is not None:
            depth = depth + center
        return depth


class MVCombinedModelV2(MVCombinedModel):
    """
    Version 2 of the Multi-View Combined Model.

    Adds support for the `forward_mv` and `forward_gs` method, enabling Gaussian Splatting inference.
    """

    def __init__(
        self,
        gs_features=None,
        gs_depth=None,
        gs_near=0.005,
        gs_far=100.0,
        gs_large_focal=False,
        gs_rgb_with_ray=False,
        return_features=False,
        normalize_cameras=True,
        **kwargs,
    ):
        self.return_features = return_features
        self.normalize_cameras = normalize_cameras

        # Initialize the parent class with necessary components
        super(MVCombinedModelV2, self).__init__(**kwargs)

        self.gs_features = gs_features
        self.gs_depth = gs_depth
        self.gs_large_focal = gs_large_focal
        self.gs_rgb_with_ray = gs_rgb_with_ray

        if self.gaussian_head is not None:
            self.register_buffer("gs_near", torch.tensor([[float(gs_near)]]))
            self.register_buffer("gs_far", torch.tensor([[float(gs_far)]]))

    def get_cameras(self, results, hw, prompt_scale):
        # Pose encoding for camera extrinsics/intrinsics
        pose_enc = results["pose_enc"]
        if isinstance(pose_enc, (list, tuple)):
            pose_enc = pose_enc[-1]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding=pose_enc,
            image_size_hw=hw,
            translation_scale=prompt_scale,
        )
        if self.normalize_cameras:
            # NOTE: trans to first camera
            w2c_pred = extrinsics.float()
            base_c2w_pred = w2c_pred[:, 0:1].inverse()
            extrinsics = w2c_pred @ base_c2w_pred

        extrinsics = extrinsics.float()
        intrinsics = intrinsics.float()
        return extrinsics, intrinsics

    def forward_mv(
        self,
        rgb,
        prompt_depth=None,
        prompt_scale=None,
        meta_data=None,
        intrinsics=None,
        query_points=None,
    ):
        # Extract frame number and view number from metadata
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        # Unpack input dimensions for batch size (b), number of frames (n), channels (c), height (h), width (w)
        b, n, c, h, w = rgb.shape
        assert n == frame_num * view_num, "Number of frames must match frame_num * view_num"

        # Reshape input tensor for processing
        rgb = rgb.view(b * n, c, h, w)

        # If a depth prompt is provided, reshape it as well
        if prompt_depth is not None:
            prompt_depth = prompt_depth.view(b * n, *prompt_depth.shape[-3:])

        if prompt_scale is not None:
            prompt_scale = prompt_scale.view(b * n, *prompt_scale.shape[-3:])

        prompt_features = rgb_features = camera_tokens = None
        local_feature_maps = glb_feature_maps = None
        local_refine_features = glb_refine_features = None
        results = dict()
        features_dict = dict()

        # If a prompt encoder exists and a depth prompt is provided, encode the depth prompt
        if self.prompt_encoder is not None and prompt_depth is not None:
            prompt_features = self.prompt_encoder(prompt_depth, meta_data=meta_data)

        # Encode RGB images using the RGB encoder
        if self.rgb_encoder is not None:
            rgb_features = self.rgb_encoder(rgb, condition=prompt_features, meta_data=meta_data)

        # Process multi-view depth results
        if self.head is not None:
            sf_results = self.head(
                rgb_features,
                prompt_features=prompt_features,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
            )
            results["pointmap"] = sf_results.pop("pointmap")
            results["confidence"] = sf_results.pop("confidence")
            results["pointmap"] = results["pointmap"].view(b, n, *results["pointmap"].shape[-3:])
            results["confidence"] = results["confidence"].view(
                b, n, *results["confidence"].shape[-3:]
            )

        # Further process features using the multi-view decoder
        if self.mv_decoder is not None:
            patch_features, camera_tokens = self.mv_decoder(
                rgb_features,
                prompt_features=prompt_features,
                prompt_scale=prompt_scale,
                meta_data=meta_data,
                intrinsics=intrinsics,
            )
        else:
            patch_features = rgb_features

        # Process multi-view depth results
        if self.mv_depth_head is not None:
            mv_depth_results = self.mv_depth_head(
                patch_features,
                prompt_features=prompt_features,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
            )
            results["mv_depth"] = mv_depth_results.pop("pointmap")
            results["mv_depth_confidence"] = mv_depth_results.pop("confidence")
            results["mv_depth"] = results["mv_depth"].view(b, n, *results["mv_depth"].shape[-3:])
            results["mv_depth_confidence"] = results["mv_depth_confidence"].view(
                b, n, *results["mv_depth_confidence"].shape[-3:]
            )
            local_feature_maps = mv_depth_results.get("features", None)
            local_refine_features = mv_depth_results.get("refine_features", None)

        # Process multi-view point cloud results
        if self.mv_point_head is not None:
            mv_point_results = self.mv_point_head(
                patch_features,
                prompt_features=prompt_features,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
            )
            results["mv_pointmap"] = mv_point_results.pop("pointmap")
            results["mv_confidence"] = mv_point_results.pop("confidence")
            results["mv_pointmap"] = results["mv_pointmap"].view(
                b, n, *results["mv_pointmap"].shape[-3:]
            )
            results["mv_confidence"] = results["mv_confidence"].view(
                b, n, *results["mv_confidence"].shape[-3:]
            )
            glb_feature_maps = mv_point_results.get("features", None)
            glb_refine_features = mv_point_results.get("refine_features", None)

        # Process camera pose encoding
        if self.camera_head is not None:
            pose_enc = self.camera_head(
                camera_tokens,
                meta_data=meta_data,
            )
            results["pose_enc"] = pose_enc

        # Process tracking results
        if self.track_head is not None and query_points is not None:
            track, track_vis, track_confidence = self.track_head(
                patch_features,
                query_points=query_points,
                prompt_features=prompt_features,
                meta_data=meta_data,
                local_feature_maps=local_feature_maps,
                glb_feature_maps=glb_feature_maps,
                pose_enc=results.get("pose_enc", None),
                local_depth=results.get("mv_depth", None),
                intrinsics=intrinsics,
            )
            results["track"] = track
            results["track_vis"] = track_vis
            results["track_confidence"] = track_confidence

        if prompt_features is not None:
            features_dict["prompt_features"] = prompt_features
        if rgb_features is not None:
            features_dict["rgb_features"] = rgb_features
        if patch_features is not None:
            features_dict["patch_features"] = patch_features

        if local_feature_maps is not None:
            features_dict["local_feature_maps"] = local_feature_maps
        if glb_feature_maps is not None:
            features_dict["glb_feature_maps"] = glb_feature_maps

        if local_refine_features is not None:
            features_dict["local_refine_features"] = local_refine_features
        if glb_refine_features is not None:
            features_dict["glb_refine_features"] = glb_refine_features

        return features_dict, results

    def forward_gs(
        self,
        rgb,
        prompt_depth=None,
        prompt_scale=None,
        prompt_center=None,
        extrinsics=None,
        intrinsics=None,
        novel_extrinsics=None,
        novel_intrinsics=None,
        query_points=None,
        render_video_with_pred_camera=False,
        meta_data=None,
        **kwargs,
    ):
        features_dict, results = self.forward_mv(
            rgb,
            prompt_depth,
            prompt_scale=prompt_scale,
            meta_data=meta_data,
            intrinsics=intrinsics,
            query_points=query_points,
        )

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

        pointmap = results[self.gs_depth]  # [B, F*V, 1, H, W]
        pointmap = self.point_denormalize(pointmap, scale=prompt_scale, center=prompt_center)

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

            novel_render = self.rendering(gaussians, cur_extrinsics, cur_intrinsics, (h, w))

            for key in novel_render:
                results[key] = novel_render[key][:, :-novel_view_nums]
                results[key.replace("render_", "render_novel_")] = novel_render[key][
                    :, -novel_view_nums:
                ]
        else:
            # rendering reference views
            render = self.rendering(gaussians, extrinsics, intrinsics, (h, w))
            results.update(render)

        if self.gs_large_focal:
            intrinsics[..., 0, 0] *= 2.0
            intrinsics[..., 1, 1] *= 2.0
            render = self.rendering(gaussians, extrinsics, intrinsics, (h, w))
            if render.get("render_alpha", None) is not None:
                results["large_focal_alpha"] = render["render_alpha"]

        return features_dict, results

    def rendering(self, gaussians, extrinsics, intrinsics, hw, depth_mode="depth", **kwargs):
        """
        extrinsics: [B, F*V, 4, 4], c2w
        intrinsics: [B, F*V, 3, 3], normalized intrinsics
        """
        with torch.autocast("cuda", dtype=torch.float32):
            b, n = extrinsics.shape[:2]
            near = self.gs_near.repeat(b, n)
            far = self.gs_far.repeat(b, n)
            try:
                renders = self.gaussian_render(
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

    def forward(
        self,
        rgb=None,
        prompt_depth=None,
        prompt_scale=None,
        prompt_center=None,
        extrinsics=None,
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
        # Optionally freeze modules if specified
        if with_freeze:
            self.freeze()

        assert (meta_data is not None) or only_rendering

        if only_rendering:
            return self.rendering(extrinsics=extrinsics, intrinsics=intrinsics, **kwargs)
        elif "frames" not in meta_data and "views" not in meta_data:
            return super(MVCombinedModelV2, self).forward(
                rgb, prompt_depth=prompt_depth, meta_data=meta_data
            )
        elif self.gaussian_head is None:
            features_dict, results = self.forward_mv(
                rgb,
                prompt_depth,
                prompt_scale=prompt_scale,
                meta_data=meta_data,
                intrinsics=intrinsics,
                query_points=query_points,
            )
        else:
            features_dict, results = self.forward_gs(
                rgb,
                prompt_depth=prompt_depth,
                prompt_scale=prompt_scale,
                prompt_center=prompt_center,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                novel_extrinsics=novel_extrinsics,
                novel_intrinsics=novel_intrinsics,
                query_points=query_points,
                render_video_with_pred_camera=render_video_with_pred_camera,
                meta_data=meta_data,
                **kwargs,
            )

        if self.return_features:
            return features_dict, results
        else:
            return results


class MVCombinedModelV3(MVCombinedModelV2):
    """
    Version 3 of the Multi-View Combined Model.

    This version introduces the following changes:
    1. mv_decoder return pos and patch_start_idx.
    """

    def __init__(self, **kwargs):
        # Initialize the parent class with necessary components
        super(MVCombinedModelV3, self).__init__(**kwargs)

    def forward_mv(
        self,
        rgb,
        prompt_depth=None,
        prompt_scale=None,
        meta_data=None,
        intrinsics=None,
        query_points=None,
    ):
        # Extract frame number and view number from metadata
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        # Unpack input dimensions for batch size (b), number of frames (n), channels (c), height (h), width (w)
        b, n, c, h, w = rgb.shape
        assert n == frame_num * view_num, "Number of frames must match frame_num * view_num"

        # Reshape input tensor for processing
        rgb = rgb.view(b * n, c, h, w)

        # If a depth prompt is provided, reshape it as well
        if prompt_depth is not None:
            prompt_depth = prompt_depth.view(b * n, *prompt_depth.shape[-3:])

        if prompt_scale is not None:
            prompt_scale = prompt_scale.view(b * n, *prompt_scale.shape[-3:])

        prompt_features = rgb_features = camera_tokens = None
        local_feature_maps = glb_feature_maps = None
        results = dict()
        features_dict = dict()

        # If a prompt encoder exists and a depth prompt is provided, encode the depth prompt
        if self.prompt_encoder is not None and prompt_depth is not None:
            prompt_features = self.prompt_encoder(prompt_depth, meta_data=meta_data)

        # Encode RGB images using the RGB encoder
        if self.rgb_encoder is not None:
            rgb_features = self.rgb_encoder(rgb, condition=prompt_features, meta_data=meta_data)

        # Process multi-view depth results
        if self.head is not None:
            sf_results = self.head(
                rgb_features,
                prompt_features=prompt_features,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
            )
            results["pointmap"] = sf_results.pop("pointmap")
            results["confidence"] = sf_results.pop("confidence")
            results["pointmap"] = results["pointmap"].view(b, n, *results["pointmap"].shape[-3:])
            results["confidence"] = results["confidence"].view(
                b, n, *results["confidence"].shape[-3:]
            )

        # Further process features using the multi-view decoder
        if self.mv_decoder is not None:
            patch_features, pos, patch_start_idx = self.mv_decoder(
                rgb_features,
                prompt_features=prompt_features,
                prompt_scale=prompt_scale,
                meta_data=meta_data,
                intrinsics=intrinsics,
            )
        else:
            raise NotImplementedError

        # Process multi-view depth results
        if self.mv_depth_head is not None:
            mv_depth_results = self.mv_depth_head(
                patch_features,
                prompt_features=prompt_features,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
                pos=pos,
                patch_start_idx=patch_start_idx,
            )
            results["mv_depth"] = mv_depth_results.pop("pointmap")
            results["mv_depth_confidence"] = mv_depth_results.pop("confidence")
            results["mv_depth"] = results["mv_depth"].view(b, n, *results["mv_depth"].shape[-3:])
            results["mv_depth_confidence"] = results["mv_depth_confidence"].view(
                b, n, *results["mv_depth_confidence"].shape[-3:]
            )
            local_feature_maps = mv_depth_results.get("features", None)

        # Process multi-view point cloud results
        if self.mv_point_head is not None:
            mv_point_results = self.mv_point_head(
                patch_features,
                prompt_features=prompt_features,
                return_dict=True,
                meta_data=meta_data,
                intrinsics=intrinsics,
                pos=pos,
                patch_start_idx=patch_start_idx,
            )
            results["mv_pointmap"] = mv_point_results.pop("pointmap")
            results["mv_confidence"] = mv_point_results.pop("confidence")
            results["mv_pointmap"] = results["mv_pointmap"].view(
                b, n, *results["mv_pointmap"].shape[-3:]
            )
            results["mv_confidence"] = results["mv_confidence"].view(
                b, n, *results["mv_confidence"].shape[-3:]
            )
            glb_feature_maps = mv_point_results.get("features", None)

        # Process camera pose encoding
        if self.camera_head is not None:
            if isinstance(patch_features, (list, tuple)):
                camera_tokens = [patch_features[-1][:, :, :patch_start_idx]]
            else:
                camera_tokens = [patch_features[:, :, :patch_start_idx]]

            pose_enc = self.camera_head(
                camera_tokens,
                meta_data=meta_data,
            )
            results["pose_enc"] = pose_enc

        # Process tracking results
        if self.track_head is not None and query_points is not None:
            if isinstance(patch_features, (list, tuple)):
                track_features = [features[:, :, patch_start_idx:] for features in patch_features]
            else:
                track_features = patch_features[:, :, patch_start_idx:]

            track, track_vis, track_confidence = self.track_head(
                track_features,
                query_points=query_points,
                prompt_features=prompt_features,
                meta_data=meta_data,
                local_feature_maps=local_feature_maps,
                glb_feature_maps=glb_feature_maps,
            )
            results["track"] = track
            results["track_vis"] = track_vis
            results["track_confidence"] = track_confidence

        features_dict["rgb_features"] = rgb_features
        features_dict["patch_features"] = patch_features
        features_dict["patch_pos"] = pos
        features_dict["patch_start_idx"] = patch_start_idx

        return features_dict, results

    def forward_gs(
        self,
        rgb,
        prompt_depth=None,
        prompt_scale=None,
        prompt_center=None,
        extrinsics=None,
        intrinsics=None,
        novel_extrinsics=None,
        novel_intrinsics=None,
        query_points=None,
        render_video_with_pred_camera=False,
        meta_data=None,
        **kwargs,
    ):
        features_dict, results = self.forward_mv(
            rgb,
            prompt_depth,
            prompt_scale=prompt_scale,
            meta_data=meta_data,
            intrinsics=intrinsics,
            query_points=query_points,
        )

        rgb_features = features_dict["rgb_features"]
        patch_features = features_dict["patch_features"]
        pos = features_dict["patch_pos"]
        patch_start_idx = features_dict["patch_start_idx"]

        b, n, c, h, w = rgb.shape

        pointmap = results[self.gs_depth]  # [B, F*V, 1, H, W]
        pointmap = self.point_denormalize(pointmap, scale=prompt_scale, center=prompt_center)

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

        gaussians = self.gaussian_head(
            images=rgb,
            rgb_features=rgb_features,
            features=patch_features,
            pos=pos,
            patch_start_idx=patch_start_idx,
            depth=pointmap,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            meta_data=meta_data,
        )
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

            novel_render = self.rendering(gaussians, cur_extrinsics, cur_intrinsics, (h, w))

            for key in novel_render:
                results[key] = novel_render[key][:, :-novel_view_nums]
                results[key.replace("render_", "render_novel_")] = novel_render[key][
                    :, -novel_view_nums:
                ]
        else:
            # rendering reference views
            render = self.rendering(gaussians, extrinsics, intrinsics, (h, w))
            results.update(render)

        return features_dict, results
