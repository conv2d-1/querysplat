import logging

import numpy as np
import torch
import torch.nn as nn

from hAlgorithm.modules.utils.gaussians.graphics_utils import normal_from_depth_image
from hAlgorithm.utils import instantiate_from_config


def render_normal(intrinsic_matrix, extrinsic_matrix, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    st = max(int(scale / 2) - 1, 0)
    if offset is not None:
        offset = offset[st::scale, st::scale]
    normal_ref = normal_from_depth_image(
        depth[st::scale, st::scale], intrinsic_matrix[0][0], extrinsic_matrix[0][0], offset
    )

    normal_ref = normal_ref.permute(2, 0, 1)
    return normal_ref  # 3,h,w


class FinetuneGSModel(nn.Module):  #
    def __init__(
        self,
        gaussian_parameters=None,
        gaussian_render=None,
        use_predict_pose=False,
        extrinsics_c2w=False,
        gs_near=0.01,
        gs_far=100.0,
    ):
        # Initialize the parent class with necessary components
        super(FinetuneGSModel, self).__init__()

        self.use_predict_pose = use_predict_pose
        self.extrinsics_c2w = extrinsics_c2w

        self.gaussian_parameters = instantiate_from_config(
            gaussian_parameters
        )  # 传入高斯ply预训练的模型
        self.gaussian_render = instantiate_from_config(gaussian_render)

        self.register_buffer("gs_near", torch.tensor([[float(gs_near)]]))
        self.register_buffer("gs_far", torch.tensor([[float(gs_far)]]))

    def rendering(self, gaussians, camera_batch, depth_mode="depth", mw_score=False, **kwargs):
        """
        extrinsics: [B, F*V, 4, 4], c2w
        intrinsics: [B, F*V, 3, 3], normalized intrinsics
        """
        with torch.autocast("cuda", dtype=torch.float32):
            renders = self.gaussian_render(gaussians, camera_batch)

        results = {}
        results["render_rgb"] = renders.color
        results["render_depth"] = renders.depth
        results["render_normal"] = renders.normal
        results["radii"] = renders.radii
        results["visibility_filter"] = renders.visibility_filter
        results["screenspace_points"] = renders.screenspace_points

        if mw_score:
            results["important_score"] = renders.important_score

        # compute normal ref
        if self.training:
            intrinsics, extrinsics = camera_batch[0][0].get_calib_matrix_nerf()
            depth_normal = (
                render_normal([[intrinsics.cuda()]], [[extrinsics.cuda()]], renders.depth[0, 0])[
                    None, None
                ]
                * (renders.alpha).detach()
            )
            results["depth_normal"] = depth_normal

        return results

    def forward(self, cameras_batch, **kwargs):
        """
        数据读取的外参都是w2c, 但是gaussian渲染需要c2w
        """
        gaussians = self.gaussian_parameters
        render = self.rendering(gaussians, cameras_batch)

        results = dict(gaussians=gaussians)
        results.update(render)

        return results
