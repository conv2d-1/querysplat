#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math

import torch
from diff_plane_rasterization2 import (
    GaussianRasterizationSettings as PlaneGaussianRasterizationSettings,
)
from diff_plane_rasterization2 import GaussianRasterizer as PlaneGaussianRasterizer
from jaxtyping import Float
from torch import Tensor

from hAlgorithm.modules.utils.gaussians.types_finetune_gs import Gaussians


def render_cuda(
    gaussians: Gaussians,
    cameras,
    background_color: Float[Tensor, "view 3"],
    fake_color_fn: callable = None,
    use_sh: bool = True,
) -> Float[Tensor, "batch 3 height width"]:

    degree = gaussians.active_sh_degree

    assert use_sh or fake_color_fn != None

    shs = gaussians.get_shs() if use_sh else fake_color_fn

    screenspace_points = torch.zeros_like(
        gaussians.get_xyz(), dtype=gaussians.get_xyz().dtype, requires_grad=True, device="cuda"
    )
    screenspace_points_abs = torch.zeros_like(
        gaussians.get_xyz(), dtype=gaussians.get_xyz().dtype, requires_grad=True, device="cuda"
    )
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    means2D = screenspace_points
    means2D_abs = screenspace_points_abs

    opacity = gaussians.get_opacity()
    means = gaussians.get_xyz()
    scales = gaussians.get_scale()
    rotations = gaussians.get_rotation()

    views = len(cameras)

    return_plane = True
    debug = False

    all_images = []
    all_depths = []
    all_normals = []
    all_alphas = []

    all_radii = []
    all_visibility_filter = []

    for v in range(views):
        camera = cameras[v]
        tanfovx = math.tan(camera.fovX * 0.5)
        tanfovy = math.tan(camera.fovY * 0.5)

        settings = PlaneGaussianRasterizationSettings(
            image_height=camera.img_h,
            image_width=camera.img_w,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background_color[v],
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=degree,
            campos=camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            opt_cam=True,
            debug=debug,
        )

        rasterizer = PlaneGaussianRasterizer(raster_settings=settings)

        # TODO: check this
        global_normal = gaussians.get_normal(camera.camera_center)
        local_normal = global_normal @ camera.world_view_transform[:3, :3]
        pts_in_cam = (
            means @ camera.world_view_transform[:3, :3] + camera.world_view_transform[3, :3]
        )

        local_distance = (local_normal * pts_in_cam).sum(-1).abs()
        input_all_map = torch.zeros((means.shape[0], 5)).to(means.device)
        input_all_map[:, :3] = local_normal
        input_all_map[:, 3] = 1.0
        input_all_map[:, 4] = local_distance

        rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
            means3D=means,
            means2D=means2D,
            means2D_abs=means2D_abs,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            all_map=input_all_map,
            cov3D_precomp=None,
            cam_q_w2c=camera.w2c_quat,
            cam_t_w2c=camera.w2c_trans,
        )

        rendered_normal = out_all_map[0:3]
        rendered_alpha = out_all_map[3:4,]
        rendered_distance = out_all_map[4:5,]

        all_images.append(rendered_image)
        all_depths.append(plane_depth[0])
        all_normals.append(rendered_normal)
        all_alphas.append(rendered_alpha[0])

        all_radii.append(radii)
        all_visibility_filter.append(radii > 0)

    return (
        torch.stack(all_images),
        torch.stack(all_depths),
        torch.stack(all_normals),
        torch.stack(all_alphas),
        torch.stack(all_radii),
        torch.stack(all_visibility_filter),
        screenspace_points,
    )
