import logging

import torch

from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
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
        feat_upsampler=None,
        gaussian_head=None,
        gaussian_render=None,
        mv_depth_head=None,
        mv_point_head=None,
        camera_head=None,
        track_head=None,
        freeze_modules=[],
        extrinsics_c2w=False,
        gs_features=[],  # ['rgb', 'mv_point', 'mv_depth', 'mv_decoder']
        use_predict_pose=False,
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
        self.mono_upsampler = feat_upsampler
        self.mv_upsampler = feat_upsampler
        self.gaussian_head_cfg = gaussian_head
        self.gaussian_render_cfg = gaussian_render
        self.mv_depth_head_cfg = mv_depth_head
        self.mv_point_head_cfg = mv_point_head
        self.camera_head_cfg = camera_head
        self.track_head_cfg = track_head
        self.extrinsics_c2w = extrinsics_c2w
        self.gs_features = gs_features
        self.use_predict_pose = use_predict_pose

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

        self.mono_upsampler = instantiate_from_config(self.mono_upsampler)
        if self.mono_upsampler is not None:
            self.module_names.append("mono_upsampler")

        self.mv_upsampler = instantiate_from_config(self.mv_upsampler)
        if self.mv_upsampler is not None:
            self.module_names.append("mv_upsampler")

    def point_denormalize(self, depth, scale=None, center=None):
        # Adjust depth values based on scale and center if provided
        if scale is not None:
            depth = depth * scale
        if center is not None:
            depth = depth + center
        return depth

    def forward_mv(self, rgb, prompt_depth=None, meta_data=None):
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

        camera_tokens, features_dict, results = None, {}, dict()

        # Encode RGB images using the RGB encoder
        if self.rgb_encoder is not None:
            rgb_features = self.rgb_encoder(rgb, meta_data=meta_data)

        # If a prompt encoder exists and a depth prompt is provided, encode the depth prompt
        if self.prompt_encoder is not None and prompt_depth is not None:
            if self.prompt_encoder.input_channel == 6:
                prompt_depth = torch.cat([prompt_depth, rgb], dim=1)
            prompt_features = self.prompt_encoder(prompt_depth, meta_data=meta_data)

        # Decode features and process through the main head model
        if self.decoder is not None:
            features = self.decoder(
                rgb_features,
                prompt_features=prompt_features,
                meta_data=meta_data,
            )

        if self.head is not None:
            results = self.head(
                features, patch_h=None, patch_w=None, return_dict=True, meta_data=meta_data
            )
            for key in results.keys():
                results[key] = results[key].view(b, n, *results[key].shape[-3:])

        # Further process features using the multi-view decoder
        if self.mv_decoder is not None:
            patch_features, camera_tokens = self.mv_decoder(
                rgb_features,
                meta_data=meta_data,
            )

        # Process multi-view depth results
        if self.mv_depth_head is not None:
            mv_depth_results = self.mv_depth_head(
                patch_features,
                prompt_features=prompt_features,
                return_dict=True,
                meta_data=meta_data,
            )
            results["mv_depth"] = mv_depth_results.pop("pointmap")
            results["mv_depth_confidence"] = mv_depth_results.pop("confidence")
            results["mv_depth"] = results["mv_depth"].view(b, n, *results["mv_depth"].shape[-3:])
            results["mv_depth_confidence"] = results["mv_depth_confidence"].view(
                b, n, *results["mv_depth_confidence"].shape[-3:]
            )

        # Process multi-view point cloud results
        if self.mv_point_head is not None:
            mv_point_results = self.mv_point_head(
                patch_features,
                prompt_features=prompt_features,
                return_dict=True,
                meta_data=meta_data,
            )
            results["mv_pointmap"] = mv_point_results.pop("pointmap")
            results["mv_confidence"] = mv_point_results.pop("confidence")
            results["mv_pointmap"] = results["mv_pointmap"].view(
                b, n, *results["mv_pointmap"].shape[-3:]
            )
            results["mv_confidence"] = results["mv_confidence"].view(
                b, n, *results["mv_confidence"].shape[-3:]
            )

        # Process camera pose encoding
        if self.camera_head is not None:
            pose_enc = self.camera_head(
                camera_tokens,
                meta_data=meta_data,
            )
            results["pose_enc"] = pose_enc

        # Process tracking results
        if self.track_head is not None:
            track_results = self.track_head(
                patch_features,
                return_dict=True,
                meta_data=meta_data,
            )
            results["track"] = track_results.pop("track_list")[-1]  # track of the last iteration
            results["track_vis"] = track_results.pop("vis")
            results["track_confidence"] = track_results.pop("conf")

        for feature_name in self.gs_features:
            if feature_name == "rgb":
                assert self.rgb_encoder is not None
                assert self.mono_upsampler is not None
                _, phw, c = rgb_features[0].shape
                features_dict[feature_name] = [
                    feat_.view(b, n, phw, c) for feat_ in rgb_features
                ]  # [B, F*V, ph*pw, C] * 4
            elif feature_name == "mv_point":
                assert self.mv_point_head is not None
                _, fc, fh, fw = mv_point_results["features"].shape
                features_dict[feature_name] = mv_point_results["features"].view(
                    b, n, fc, fh, fw
                )  # [B, F*V, 64, H, W]
            elif feature_name == "mv_depth":
                assert self.mv_depth_head is not None
                _, fc, fh, fw = mv_depth_results["features"].shape
                features_dict[feature_name] = mv_depth_results["features"].view(
                    b, n, fc, fh, fw
                )  # [B, F*V, 64, H, W]
            elif feature_name == "mv_decoder":
                assert self.mv_decoder is not None
                assert self.mv_upsampler is not None
                features_dict[feature_name] = patch_features  # [B, F*V, ph*pw, C] * 4

        return features_dict, results

    def forward_gs(
        self,
        rgb,
        prompt_depth=None,
        prompt_scale=None,
        prompt_center=None,
        extrinsics=None,
        intrinsics=None,
        meta_data=None,
        **kwargs,
    ):
        features_dict, results = self.forward_mv(rgb, prompt_depth, meta_data=meta_data)
        if self.use_predict_pose:
            pose_enc = results["pose_enc"]
            extrinsics_pred, _ = pose_encoding_to_extri_intri(
                pose_enc, translation_scale=prompt_scale, build_intrinsics=False
            )
            extrinsics_pred = extrinsics_pred.float().inverse()

        fine_features = []
        for feature_name in self.gs_features:
            if feature_name == "rgb":
                feature = self.mono_upsampler(
                    features_dict[feature_name]
                )  # [B, F*V, ph*pw, C] * 4 -> [B, F*V, C, H, W]
            elif feature_name == "mv_decoder":
                feature = self.mv_upsampler(
                    features_dict[feature_name]
                )  # [B, F*V, ph*pw, C] * 4 -> [B, F*V, C, H, W]
            else:
                feature = features_dict[feature_name]
            fine_features.append(feature)
        fine_features = torch.cat(fine_features, dim=2)  # [B, F*V, N*C, H, W]

        b, n, c, h, w = rgb.shape

        pointmap = results["mv_depth"]  # [B, F*V, 1, H, W]
        pointmap = self.point_denormalize(pointmap, scale=prompt_scale, center=prompt_center)
        extrinsics = extrinsics.clone()
        if not self.extrinsics_c2w:
            extrinsics = extrinsics.inverse()

        intrinsics = intrinsics.clone()
        intrinsics[..., 0, :] /= w
        intrinsics[..., 1, :] /= h

        target = kwargs.get("target", None)

        gs_mask = kwargs.get("gs_mask", None)
        # if gs_mask is None:
        #     gs_mask = results["mv_depth_confidence"] > 0.0
        gaussians = self.gaussian_head(
            images=rgb,
            features=fine_features,
            depth=pointmap[:, :, -1:],
            # depth=target[:, :, -1:],
            extrinsics=extrinsics_pred if self.use_predict_pose else extrinsics,
            intrinsics=intrinsics,
        )
        if gs_mask is not None:
            # TODO: support downsample
            if self.gaussian_head.downsample:
                gs_mask = gs_mask[..., ::2, ::2]
            gaussians.apply_mask(gs_mask)
        results["gaussians"] = gaussians

        near = kwargs.get("near", 0.01)
        far = kwargs.get("far", 100)

        # rendering reference views
        try:
            render = self.rendering(gaussians, extrinsics, intrinsics, near, far, (h, w))
            results.update(render)
        except:
            torch.cuda.empty_cache()
            pass

        # rendering novel views
        if kwargs.get("novel_extrinsics", None) is not None:
            novel_extrinsics = kwargs["novel_extrinsics"]
            novel_intrinsics = kwargs["novel_intrinsics"]
            if not self.extrinsics_c2w:
                novel_extrinsics = novel_extrinsics.inverse()
            novel_intrinsics[..., 0, :] /= w
            novel_intrinsics[..., 1, :] /= h
            try:
                novel_render = self.rendering(
                    gaussians, novel_extrinsics, novel_intrinsics, near, far, (h, w)
                )

                results["render_novel_rgb"] = novel_render["render_rgb"]
                results["render_novel_depth"] = novel_render["render_depth"]
            except:
                torch.cuda.empty_cache()
                pass

        return results

    def rendering(
        self, gaussians, extrinsics, intrinsics, near, far, hw, depth_mode="depth", **kwargs
    ):
        """
        extrinsics: [B, F*V, 4, 4], c2w
        intrinsics: [B, F*V, 3, 3], normalized intrinsics
        """
        with torch.autocast("cuda", dtype=torch.float32):
            b, n = extrinsics.shape[:2]
            near = torch.ones((b, n), device=extrinsics[0].device) * near
            far = torch.ones((b, n), device=extrinsics[0].device) * far
            renders = self.gaussian_render(
                gaussians, extrinsics, intrinsics, near, far, hw, depth_mode=depth_mode
            )
        results = {}
        results["render_rgb"] = renders.color
        results["render_depth"] = renders.depth

        return results

    def forward(
        self,
        rgb=None,
        prompt_depth=None,
        prompt_scale=None,
        prompt_center=None,
        extrinsics=None,
        intrinsics=None,
        with_freeze=False,
        meta_data=None,
        **kwargs,
    ):
        # Optionally freeze modules if specified
        if with_freeze:
            self.freeze()

        if kwargs.get("rendering", False):
            return self.rendering(extrinsics=extrinsics, intrinsics=intrinsics, **kwargs)

        assert meta_data is not None

        if "frames" not in meta_data and "views" not in meta_data:
            return super(MVCombinedModel, self).forward(
                rgb, prompt_depth=prompt_depth, meta_data=meta_data
            )
        elif self.gaussian_head is None:
            return self.forward_mv(rgb, prompt_depth, meta_data=meta_data)[-1]
        else:
            return self.forward_gs(
                rgb,
                prompt_depth=prompt_depth,
                prompt_scale=prompt_scale,
                prompt_center=prompt_center,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                meta_data=meta_data,
                **kwargs,
            )
