import torch

from hAlgorithm.modules.models.pi3.utils.pose_enc import (
    pi3_pose_fov_to_extri_intri,
)
from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)

from .multi_views_v4 import MVCombinedModelV2


class Pi3CombinedModel(MVCombinedModelV2):
    """
    Pi3CombinedModel,
    1. mv_decoder return pos.
    2. 为了共用 pipeline, camera head 输出的是 w2c!!!"""

    def __init__(self, pose_encoding_type="absT_quaR_FoV", normalize_cameras=True, **kwargs):
        super(Pi3CombinedModel, self).__init__(normalize_cameras=normalize_cameras, **kwargs)

        self.pose_encoding_type = pose_encoding_type

    def get_cameras(self, results, hw, prompt_scale, intrinsics=None):
        # Pose encoding for camera extrinsics/intrinsics
        pose_enc = results["pose_enc"]

        if isinstance(pose_enc, (list, tuple)):
            pose_enc = pose_enc[-1]
        if self.pose_encoding_type == "absT_quaR_FoV":
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                pose_encoding=pose_enc,
                image_size_hw=hw,
                translation_scale=prompt_scale,
            )
            if self.normalize_cameras:
                w2c_pred = extrinsics.float()
                base_c2w_pred = w2c_pred[:, 0:1].inverse()
                extrinsics = w2c_pred @ base_c2w_pred
            intrinsics = intrinsics.float()
        elif self.pose_encoding_type == "pi3":
            extrinsics, _ = pi3_pose_fov_to_extri_intri(
                pose=pose_enc,
                fov=None,
                image_size_hw=hw,
                translation_scale=prompt_scale,
                pose_encoding_type=self.pose_encoding_type,
                normalize_cameras=self.normalize_cameras,
            )
        elif self.pose_encoding_type == "pi3_fov":
            extrinsics, intrinsics = pi3_pose_fov_to_extri_intri(
                pose=pose_enc,
                fov=results["camera_fov"],
                image_size_hw=hw,
                translation_scale=prompt_scale,
                pose_encoding_type=self.pose_encoding_type,
                normalize_cameras=self.normalize_cameras,
            )
            intrinsics = intrinsics.float()
        else:
            raise NotImplementedError

        extrinsics = extrinsics.float()

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

        # Further process features using the multi-view decoder
        if self.mv_decoder is not None:
            patch_features, pos, patch_start_idx = self.mv_decoder(
                rgb_features,
                prompt_features=prompt_features,
                prompt_scale=prompt_scale,
                meta_data=meta_data,
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
            local_refine_features = mv_depth_results.get("refine_features", None)

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
            glb_refine_features = mv_point_results.get("refine_features", None)

        # Process camera pose encoding
        if self.camera_head is not None:
            if self.pose_encoding_type == "absT_quaR_FoV":
                if isinstance(patch_features, (list, tuple)):
                    camera_tokens = [patch_features[-1][:, :, :patch_start_idx]]
                else:
                    camera_tokens = [patch_features[:, :, :patch_start_idx]]

                pose_enc = self.camera_head(
                    camera_tokens,
                    meta_data=meta_data,
                )
            elif self.pose_encoding_type == "pi3":
                pose_enc = self.camera_head(
                    patch_features,
                    pos=pos,
                    patch_start_idx=patch_start_idx,
                    meta_data=meta_data,
                )
                pose_enc = pose_enc[0]
            elif self.pose_encoding_type == "pi3_fov":
                pose_enc, fov = self.camera_head(
                    patch_features,
                    pos=pos,
                    patch_start_idx=patch_start_idx,
                    meta_data=meta_data,
                )
                results["camera_fov"] = fov
            else:
                raise NotImplementedError
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
