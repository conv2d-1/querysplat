import traceback

import torch

from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)

from .multi_views_v4 import MVCombinedModelV2


class VGGTCombinedModel(MVCombinedModelV2):
    "VGGT and FFGS"

    def __init__(self, **kwargs):
        # Initialize the parent class with necessary components
        super(VGGTCombinedModel, self).__init__(**kwargs)

    def forward_gs(
        self,
        rgb,
        query_points=None,
        meta_data=None,
        novel_view_nums=0,
        **kwargs,
    ):

        with torch.no_grad():
            features_dict, results = self.forward_mv(
                rgb,
                meta_data=meta_data,
                query_points=query_points,
            )

        gs_features = []
        for feature_name in self.gs_features:
            feature = features_dict[feature_name]
            gs_features.append(feature)
        gs_features = torch.cat(gs_features, dim=2)  # [B, F*V, N*C, H, W]

        b, n, c, h, w = rgb.shape

        pointmap = results[self.gs_depth]  # [B, F*V, 1, H, W]

        pose_enc = results["pose_enc"]
        if isinstance(pose_enc, (list, tuple)):
            pose_enc = pose_enc[-1]

        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding=pose_enc,
            image_size_hw=(h, w),
            translation_scale=None,
        )
        extrinsics = extrinsics.float()
        intrinsics = intrinsics.float()

        if not self.extrinsics_c2w:
            extrinsics = extrinsics.float().inverse().to(intrinsics.dtype)

        intrinsics[..., 0, :] /= w
        intrinsics[..., 1, :] /= h

        if novel_view_nums > 0:
            gaussians = self.gaussian_head(
                images=rgb[:, :-novel_view_nums],
                features=gs_features[:, :-novel_view_nums],
                depth=pointmap[:, :-novel_view_nums, -1:],
                extrinsics=extrinsics[:, :-novel_view_nums],
                intrinsics=intrinsics[:, :-novel_view_nums],
                prompt_scale=None,
            )
        else:
            gaussians = self.gaussian_head(
                images=rgb,
                features=gs_features,
                depth=pointmap[:, :, -1:],
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                prompt_scale=None,
            )
        gs_mask = kwargs.get("gs_mask", None)
        if gs_mask is not None:
            # TODO: support downsample
            if self.gaussian_head.downsample:
                gs_mask = gs_mask[..., ::2, ::2]
            gaussians.apply_mask(gs_mask)
        results["gaussians"] = gaussians

        # rendering reference views
        render = self.rendering(gaussians, extrinsics, intrinsics, (h, w))
        if novel_view_nums > 0:
            for key in render:
                results[key] = render[key][:, :-novel_view_nums]
                results[key.replace("render_", "render_novel_")] = render[key][:, -novel_view_nums:]
        else:
            results.update(render)

        if self.gs_large_focal:
            intrinsics[..., 0, 0] *= 2.0
            intrinsics[..., 1, 1] *= 2.0
            render = self.rendering(gaussians, extrinsics, intrinsics, (h, w))
            if render.get("render_alpha", None) is not None:
                results["large_focal_alpha"] = render["render_alpha"]

        return results
