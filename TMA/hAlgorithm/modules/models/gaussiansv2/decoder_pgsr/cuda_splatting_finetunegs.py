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
from math import isqrt

import numpy as np
import torch
from diff_plane_rasterization import (
    GaussianRasterizationSettings as PlaneGaussianRasterizationSettings,
)
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from hAlgorithm.modules.utils.gaussians.projection import get_fov, homogenize_points
from hAlgorithm.modules.utils.gaussians.types_finetune_gs import Gaussians


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def get_projection_matrix_with_k(w, h, k, near=0.01, far=100):
    """
    k: [batch, 3, 3]
    w: int
    h: int
    """
    b = k.shape[0]

    fx, fy, cx, cy = k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2]
    fx *= w
    fy *= h
    cx *= w
    cy *= h

    opengl_proj = torch.zeros((b, 4, 4), dtype=torch.float32, device=k.device)
    opengl_proj[:, 0, 0] = 2 * fx / w
    opengl_proj[:, 1, 1] = 2 * fy / h
    opengl_proj[:, 0, 2] = -(w - 2 * cx) / w
    opengl_proj[:, 1, 2] = -(h - 2 * cy) / h
    opengl_proj[:, 3, 2] = 1
    opengl_proj[:, 2, 2] = far / (far - near)
    opengl_proj[:, 2, 3] = -(far * near) / (far - near)

    return opengl_proj


def render_cuda(
    extrinsics: Float[Tensor, "view 4 4"],
    intrinsics: Float[Tensor, "view 3 3"],
    near: Float[Tensor, "view"],
    far: Float[Tensor, "view"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "view 3"],
    gaussians: Gaussians,
    fake_color_fn: callable = None,
    scale_invariant: bool = False,
    use_sh: bool = True,
) -> Float[Tensor, "batch 3 height width"]:

    # _, d_sh, _ = gaussians.get_shs().shape
    # degree = isqrt(d_sh) - 1
    degree = gaussians.active_sh_degree

    # if gaussians.get_norm_scale() is not None:
    #     extrinsics[:, :3, 3] /= gaussians.get_norm_scale()
    #     near /= gaussians.get_norm_scale()
    #     far /= gaussians.get_norm_scale()

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

    # rescale_factor = torch.ones_like(near)

    v, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    # fov_x, fov_y = get_fov(intrinsics,w,h).unbind(dim=-1)

    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    # projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = get_projection_matrix_with_k(w, h, intrinsics, near, far)
    projection_matrix = rearrange(projection_matrix, "v i j -> v j i")
    view_matrix = rearrange(extrinsics.inverse(), "v i j -> v j i")
    full_projection = view_matrix @ projection_matrix  # G @ full_projection

    return_plane = True
    debug = False

    all_images = []
    all_depths = []
    all_normals = []
    all_alphas = []

    all_radii = []
    all_visibility_filter = []

    for i in range(v):

        settings = PlaneGaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),  #
            tanfovy=tan_fov_y[i].item(),  #
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],  #
            projmatrix=full_projection[i],  #
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,
            render_geo=return_plane,
            debug=debug,
        )

        rasterizer = PlaneGaussianRasterizer(raster_settings=settings)

        # TODO: check this
        global_normal = gaussians.get_normal(extrinsics[i, :3, 3])
        local_normal = global_normal @ settings.viewmatrix[:3, :3]
        pts_in_cam = means @ settings.viewmatrix[:3, :3] + settings.viewmatrix[3, :3]

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
        )
        # np.savetxt(str(i)+"_world_view_transform_pgsr.txt", view_matrix[i].cpu().numpy())
        rendered_normal = out_all_map[0:3]
        rendered_alpha = out_all_map[3:4,]
        rendered_distance = out_all_map[4:5,]

        all_images.append(rendered_image)
        all_depths.append(plane_depth[0])
        all_normals.append(rendered_normal)
        all_alphas.append(rendered_alpha[0])

        # all_screenspace_points.append(screenspace_points)
        all_radii.append(radii)
        all_visibility_filter.append(radii > 0)

    # del gaussians.grad_screenspace_points
    # gaussians.grad_screenspace_points = all_screenspace_points
    # import pdb;pdb.set_trace()
    # screenspace_points = torch.stack(all_screenspace_points)
    return (
        torch.stack(all_images),
        torch.stack(all_depths),
        torch.stack(all_normals),
        torch.stack(all_alphas),
        torch.stack(all_radii),
        torch.stack(all_visibility_filter),
        screenspace_points,
    )
