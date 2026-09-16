import torch
from .base import MVBase

class MatchBase(MVBase):
    def __init__(
        self,
        match_head=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.match_head = self._instantiate_and_register(match_head, "match_head")

    def aggregator(self, rgb, scale, prompt_depth, intrinsics, ray_directions, w2c, c2w, ray_world, meta_data):
        if rgb.ndim == 5:
            frame_num, view_num = meta_data["frames"][0], meta_data["views"][0]

            b, n, c, h, w = rgb.shape
            assert n == frame_num * view_num or b % int(frame_num * view_num) == 0, "Number of frames must match frame_num * view_num"

            rgb = rgb.view(b * n, c, h, w)

            if prompt_depth is not None:
                prompt_depth = prompt_depth.view(b * n, *prompt_depth.shape[-3:])

            if scale is not None:
                scale = scale.view(b * n, *scale.shape[-3:])

            if intrinsics is not None:
                intrinsics = intrinsics.view(b * n, 3, 3)

            if w2c is not None:
                w2c = w2c.view(b * n, 4, 4)
            
            if c2w is not None:
                c2w = c2w.view(b * n, 4, 4)

            if ray_directions is not None:
                ray_directions = ray_directions.view(b * n, *ray_directions.shape[-3:])
            
            if ray_world is not None:
                ray_world = ray_world.view(b * n, *ray_world.shape[-3:])
        else:
            n = 1
            b, c, h, w = rgb.shape

        cam_token = prompt_ray = prompt_ray_in_world = prompt_extra = None

        if self.training and self.geometric_prob is not None and self.geometric_prob < 1.0:
            geometric_input_mask = torch.rand(b, device=rgb.device) <= self.geometric_prob
            geometric_input_mask = geometric_input_mask[:, None].repeat(1, n)
        else:
            geometric_input_mask = None

        if self.extra_encoder is not None:
            prompt_extra = self.extra_encoder(intrinsics=intrinsics, w2c=w2c, c2w=c2w, scale=scale, meta_data=meta_data)

        if self.camera_encoder is not None:
            cam_token = self.camera_encoder(intrinsics=intrinsics, w2c=w2c, c2w=c2w, scale=scale, meta_data=meta_data)
            if self.training and self.cam_prob is not None and self.cam_prob < 1.0:
                cam_input_mask = torch.rand(b, device=rgb.device) <= self.cam_prob
                cam_input_mask = cam_input_mask[:, None].repeat(1, n)
                if geometric_input_mask is not None:
                    cam_input_mask *= geometric_input_mask
                cam_token *= cam_input_mask.unsqueeze(-1).float()
            elif self.training and geometric_input_mask is not None:
                cam_token *= geometric_input_mask.unsqueeze(-1).float()

        if self.ray_encoder is not None and intrinsics is not None:
            prompt_ray = self.ray_encoder(ray_directions=ray_directions, intrinsics=intrinsics, w2c=w2c, c2w=c2w, meta_data=meta_data)

        if self.ray_in_world_encoder is not None:
            prompt_ray_in_world = self.ray_in_world_encoder(ray_directions=ray_world, intrinsics=intrinsics, w2c=w2c, c2w=c2w, meta_data=meta_data)

        if self.depth_encoder is not None and prompt_depth is not None:
            prompt_depth = self.depth_encoder(prompt_depth, prompt_ray=prompt_ray, prompt_ray_in_world=prompt_ray_in_world, prompt_extra=prompt_extra, meta_data=meta_data)
            if self.training and self.depth_prob is not None and self.depth_prob < 1.0:
                depth_input_mask = torch.rand(b, device=rgb.device) <= self.depth_prob
                depth_input_mask = depth_input_mask[:, None].repeat(1, n)
                if geometric_input_mask is not None:
                    depth_input_mask *= geometric_input_mask
                if prompt_depth.ndim == 3:
                    prompt_depth *= depth_input_mask.view(-1, 1, 1).float()
                else:
                    prompt_depth *= depth_input_mask.view(-1, 1, 1, 1).float()
            elif self.training and geometric_input_mask is not None:
                if prompt_depth.ndim == 3:
                    prompt_depth *= depth_input_mask.view(-1, 1, 1).float()
                else:
                    prompt_depth *= geometric_input_mask.view(-1, 1, 1, 1).float()

        if self.rgb_encoder is not None:
            rgb_features = self.rgb_encoder(
                rgb,
                prompt_depth=prompt_depth,
                prompt_ray=prompt_ray,
                prompt_ray_in_world=prompt_ray_in_world,
                prompt_extra=prompt_extra,
                meta_data=meta_data,
            )
            # reshape back to b, n
            rgb_features = [feat.reshape(b, n, *feat.shape[1:]) for feat in rgb_features]
        else:
            rgb_features = None

        if self.fuse_encoder is not None:
            patch_features, pos, patch_start_idx = self.fuse_encoder(
                rgb.view(b, n, c, h, w),
                prompt_depth=prompt_depth,
                prompt_ray=prompt_ray,
                prompt_ray_in_world=prompt_ray_in_world,
                prompt_extra=prompt_extra,
                scale=scale,
                intrinsics=intrinsics,
                w2c=w2c,
                c2w=c2w,
                cam_token=cam_token,
                meta_data=meta_data,
            )
        else:
            pos = patch_start_idx = None

        return rgb_features, patch_features, pos, patch_start_idx, prompt_depth

    def forward(
        self,
        rgb,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        c2w=None,
        ray_world=None,
        query_points=None,
        meta_data=None,
        pair_idx=None,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        assert rgb.ndim == 5
        b, n, c, h, w = rgb.shape

        rgb_features, patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics, ray_directions=ray_directions, w2c=w2c, c2w=c2w, ray_world=ray_world, meta_data=meta_data
        )

        results = dict()

        if self.depth_head is not None:
            depth_results = self.depth_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                prompt_depth=prompt_depth,
                meta_data=meta_data,
            )
            for key, val in depth_results.items():
                results[key] = val.view(b, n, *val.shape[-3:])

        if self.glb_points_head is not None:
            glb_depth_results = self.glb_points_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in glb_depth_results.items():
                results["global_" + key] = val.view(b, n, *val.shape[-3:])

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

        if self.track_head is not None and query_points is not None:
            track, track_vis, track_confidence = self.track_head(
                patch_features,
                query_points=query_points,
                meta_data=meta_data,
            )
            results["track"] = track
            results["track_vis"] = track_vis
            results["track_confidence"] = track_confidence

        if self.normal_head is not None:
            normal_results = self.normal_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in normal_results.items():
                results[key] = val.view(b, n, *val.shape[1:])

        if self.match_head is not None:
            results["match"] = self.match_head(
                images=rgb,
                rgb_features=rgb_features,
                patch_feats=patch_features,
                bidirectional=False,
                pair_idx=pair_idx,
                patch_start_idx=patch_start_idx,
                patch_w=meta_data.get("patch_w", None),
                patch_h=meta_data.get("patch_h", None),
                meta_data=meta_data,
            )

        return results