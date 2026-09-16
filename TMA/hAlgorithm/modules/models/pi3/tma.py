import logging
import traceback

import torch

from hAlgorithm.utils import instantiate_from_config

from .models.pi3 import Pi3, homogenize_points


class TMAPi3(Pi3):
    """采用 Pi3Pipeline 调用, camera head 延续 Pi3 的格式，输出 c2w."""
    def __init__(
        self,
        pos_type="rope100",
        decoder_size="large",
        use_checkpoint=False,
        gaussian_head=None,
        gaussian_render=None,
        gs_features=None,
        gs_depth=None,
        gs_near=0.005,
        gs_far=100.0,
        normalize=True,
        extrinsics_c2w=False,
        freeze_pi3=False,
        normalize_cameras=False,
    ):
        super(TMAPi3, self).__init__(pos_type=pos_type, decoder_size=decoder_size, use_checkpoint=use_checkpoint)

        self.normalize = normalize
        self.extrinsics_c2w = extrinsics_c2w
        self.freeze_pi3 = freeze_pi3
        self.normalize_cameras = normalize_cameras

        self.gaussian_head = instantiate_from_config(gaussian_head)
        self.gaussian_render = instantiate_from_config(gaussian_render)

        self.gs_features = gs_features
        self.gs_depth = gs_depth

        if self.gaussian_head is not None:
            self.register_buffer("gs_near", torch.tensor([[float(gs_near)]]))
            self.register_buffer("gs_far", torch.tensor([[float(gs_far)]]))

            self.gaussian_proj = torch.nn.Linear(
                1024, (self.gaussian_head.feat_dim) * self.patch_size**2
            )

    def forward_mv(self, rgb, **kwargs):

        if self.normalize:
            imgs = (rgb + 1) * 0.5
        else:
            imgs = rgb

        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # encode by dinov2
        imgs = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, pos = self.decode(hidden, N, H, W)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        # with torch.amp.autocast(device_type='cuda', enabled=False):
        # local points
        point_hidden = point_hidden.float()
        ret = self.point_head([point_hidden[:, self.patch_start_idx :]], (H, W)).reshape(
            B, N, H, W, -1
        )
        xy, z = ret.split([2, 1], dim=-1)
        z = torch.exp(z)
        local_points = torch.cat([xy * z, z], dim=-1)

        # confidence
        conf_hidden = conf_hidden.float()
        conf = self.conf_head([conf_hidden[:, self.patch_start_idx :]], (H, W)).reshape(
            B, N, H, W, -1
        )

        # camera
        camera_hidden = camera_hidden.float()
        camera_poses = self.camera_head(
            camera_hidden[:, self.patch_start_idx :], patch_h, patch_w
        ).reshape(B, N, 4, 4)

        # unproject local points using camera poses
        points = torch.einsum(
            "bnij, bnhwj -> bnhwi", camera_poses, homogenize_points(local_points)
        )[..., :3]

        results = dict(
            points=points.permute(0, 1, 4, 2, 3),
            local_points=local_points.permute(0, 1, 4, 2, 3),
            conf=conf.squeeze(-1).unsqueeze(2),
            camera_poses=camera_poses,
        )

        features_dict = dict(
            point_hidden=point_hidden,
        )

        return features_dict, results

    def forward_gs(
        self,
        rgb,
        prompt_depth=None,
        extrinsics=None,
        intrinsics=None,
        novel_extrinsics=None,
        novel_intrinsics=None,
        render_video_with_pred_camera=False,
        meta_data=None,
        **kwargs,
    ):
        if self.freeze_pi3:
            with torch.no_grad():
                features_dict, results = self.forward_mv(rgb)
        else:
            features_dict, results = self.forward_mv(rgb)

        b, n, c, h, w = rgb.shape

        tokens = features_dict["point_hidden"][:, self.patch_start_idx :]
        feat = self.gaussian_proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(b * n, -1, h // self.patch_size, w // self.patch_size)
        feat = torch.nn.functional.pixel_shuffle(feat, self.patch_size)  # B,3,H,W
        feat = feat.view(b, n, -1, h, w)

        pointmap = results[self.gs_depth]
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            if render_video_with_pred_camera or extrinsics is None:
                c2w = results["camera_poses"].clone()
            else:
                if self.extrinsics_c2w:
                    c2w = extrinsics.clone()
                else:
                    c2w = extrinsics.clone().inverse()
            
            if self.normalize_cameras:
                base_w2c = c2w[:, 0:1].inverse()
                c2w = base_w2c @ c2w

            intrinsics = intrinsics.clone()
            intrinsics[..., 0, :] /= w
            intrinsics[..., 1, :] /= h

        gaussians = self.gaussian_head(
            images=rgb,
            features=feat,
            depth=pointmap,
            extrinsics=c2w,
            intrinsics=intrinsics,
            prompt_scale=None,
        )
        results["gaussians"] = gaussians

        # rendering novel views
        if novel_extrinsics is not None and novel_intrinsics is not None:
            novel_view_nums = novel_extrinsics.shape[1]
            novel_extrinsics = novel_extrinsics.clone()
            if self.extrinsics_c2w:
                novel_c2w = novel_extrinsics
            else:
                novel_c2w = novel_extrinsics.inverse()

            novel_intrinsics = novel_intrinsics.clone()
            novel_intrinsics[..., 0, :] /= w
            novel_intrinsics[..., 1, :] /= h

            cur_c2w = torch.cat([c2w, novel_c2w], dim=1)
            cur_intrinsics = torch.cat([intrinsics, novel_intrinsics], dim=1)

            novel_render = self.rendering(gaussians, cur_c2w, cur_intrinsics, (h, w))

            for key in novel_render:
                results[key] = novel_render[key][:, :-novel_view_nums]
                results[key.replace("render_", "render_novel_")] = novel_render[key][
                    :, -novel_view_nums:
                ]
        else:
            # rendering reference views
            render = self.rendering(gaussians, c2w, intrinsics, (h, w))
            results.update(render)

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
        meta_data=None,
        only_rendering=False,
        render_video_with_pred_camera=False,
        **kwargs,
    ):
        assert (meta_data is not None) or only_rendering

        if only_rendering:
            return self.rendering(extrinsics=extrinsics, intrinsics=intrinsics, **kwargs)
        elif self.gaussian_head is None:
            if self.freeze_pi3:
                with torch.no_grad():
                    features_dict, results = self.forward_mv(rgb)
            else:
                features_dict, results = self.forward_mv(rgb)
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

        return results
