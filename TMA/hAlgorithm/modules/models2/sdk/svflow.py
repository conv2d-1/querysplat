import os, sys
sys.path.append(os.getcwd() + '/hAlgorithm/modules/models2/external/cut3r')
from .base import MVBase

class SVFlowCUT3R(MVBase):
    def __init__(
        self,
        **kwargs,
    ):
        super(SVFlowCUT3R, self).__init__(**kwargs)

    def aggregator(self, rgb, scale, prompt_depth, intrinsics, ray_directions, w2c, ray_world, meta_data):
        if rgb.ndim == 5:
            frame_num, view_num = meta_data["frames"][0], meta_data["views"][0]

            b, n, c, h, w = rgb.shape
            assert n == frame_num * view_num, "Number of frames must match frame_num * view_num"

            rgb = rgb.view(b * n, c, h, w)

            if prompt_depth is not None:
                prompt_depth = prompt_depth.view(b * n, *prompt_depth.shape[-3:])

            if scale is not None:
                scale = scale.view(b * n, *scale.shape[-3:])

            if intrinsics is not None:
                intrinsics = intrinsics.view(b * n, 3, 3)

            if w2c is not None:
                w2c = w2c.view(b * n, 4, 4)

            if ray_directions is not None:
                ray_directions = ray_directions.view(b * n, *ray_directions.shape[-3:])
            
            if ray_world is not None:
                ray_world = ray_world.view(b * n, *ray_world.shape[-3:])

        prompt_ray = prompt_ray_in_world = prompt_extra = None

        if self.extra_encoder is not None:
            prompt_extra = self.extra_encoder(intrinsics=intrinsics, w2c=w2c, scale=scale, meta_data=meta_data)

        if self.ray_encoder is not None and intrinsics is not None:
            prompt_ray = self.ray_encoder(ray_directions=ray_directions, intrinsics=intrinsics, w2c=w2c, meta_data=meta_data)

        if self.ray_in_world_encoder is not None:
            prompt_ray_in_world = self.ray_in_world_encoder(ray_directions=ray_world, intrinsics=intrinsics, w2c=w2c, meta_data=meta_data)

        if self.depth_encoder is not None and prompt_depth is not None:
            prompt_depth = self.depth_encoder(prompt_depth, prompt_ray=prompt_ray, prompt_ray_in_world=prompt_ray_in_world, prompt_extra=prompt_extra, meta_data=meta_data)

        patch_features, patch_pos = self.rgb_encoder(
            rgb,
            prompt_depth=prompt_depth,
            prompt_ray=prompt_ray,
            prompt_ray_in_world=prompt_ray_in_world,
            prompt_extra=prompt_extra,
            meta_data=meta_data,
        )

        if self.fuse_encoder is not None:
            patch_features, pos, patch_start_idx = self.fuse_encoder(
                patch_features,
                patch_pos,
                prompt_depth=prompt_depth,
                prompt_ray=prompt_ray,
                prompt_ray_in_world=prompt_ray_in_world,
                prompt_extra=prompt_extra,
                scale=scale,
                intrinsics=intrinsics,
                w2c=w2c,
                meta_data=meta_data,
            )
        else:
            pos = patch_start_idx = None

        return patch_features, pos, patch_start_idx, prompt_depth

    def forward(
        self,
        rgb,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        ray_world=None,
        meta_data=None,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        b, n, c, h, w = rgb.shape

        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics, ray_directions=ray_directions, w2c=w2c, ray_world=ray_world, meta_data=meta_data
        )

        results = dict()

        if self.depth_head is not None:
            depth_results = self.depth_head(
                patch_features,
                prompt_depth=prompt_depth,
                img_shape=rgb.shape,
                pos=pos,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in depth_results.items():
                results[key] = val.view(b, n, *val.shape[1:])

        if self.glb_points_head is not None:
            glb_depth_results = self.glb_points_head(
                patch_features,
                prompt_depth=prompt_depth,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in glb_depth_results.items():
                results["global_" + key] = val.view(b, n, *val.shape[1:])

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

        return results