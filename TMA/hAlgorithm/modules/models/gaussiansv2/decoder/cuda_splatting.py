from math import isqrt
from typing import Literal

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from hAlgorithm.modules.utils.gaussians.projection import get_fov, homogenize_points
from hAlgorithm.modules.utils.gaussians.types import GaussiansV2 as Gaussians

DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


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


import time


def render_cuda(
    extrinsics: Float[Tensor, "view 4 4"],
    intrinsics: Float[Tensor, "view 3 3"],
    near: Float[Tensor, "view"],
    far: Float[Tensor, "view"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "view 3"],
    gaussians: Gaussians,
    fake_color_fn: callable = None,
    scale_invariant: bool = True,
    use_sh: bool = True,
) -> Float[Tensor, "batch 3 height width"]:

    _, d_sh, _ = gaussians.get_shs().shape
    degree = isqrt(d_sh) - 1

    assert use_sh or fake_color_fn != None

    shs = gaussians.get_shs() if use_sh else fake_color_fn
    opacity = gaussians.get_opacity()
    means = gaussians.get_xyz()
    # covariances = gaussians.get_covariance()
    scales = gaussians.get_scale()
    rotations = gaussians.get_rotation()

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        rescale_factor = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * rescale_factor[:, None]
        near = near * rescale_factor
        far = far * rescale_factor
    else:
        rescale_factor = torch.ones_like(near)

    v, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "v i j -> v j i")
    view_matrix = rearrange(extrinsics.inverse(), "v i j -> v j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(v):
        means_i = means * rescale_factor[i, None, None]
        # covariances_i = covariances * (scale[i, None, None, None] ** 2)
        scales_i = scales * rescale_factor[i, None]
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(means_i, requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        image, radii = rasterizer(
            means3D=means_i,
            means2D=mean_gradients,
            shs=shs if use_sh else None,
            colors_precomp=None if use_sh else shs(i),
            opacities=opacity,
            scales=scales_i,
            rotations=rotations,
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


def render_depth_cuda(
    extrinsics: Float[Tensor, "view 4 4"],
    intrinsics: Float[Tensor, "view 3 3"],
    near: Float[Tensor, "view"],
    far: Float[Tensor, " view"],
    image_shape: tuple[int, int],
    gaussians: Gaussians,
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
) -> Float[Tensor, "batch height width"]:

    # Specify colors according to Gaussian depths.
    def fake_color_fn(idx):
        """
        extrinsics_i: [4 4]
        xyz: [g 3]
        return: [g 3]
        """
        # Specify colors according to Gaussian depths.
        camera_space_gaussians = einsum(
            extrinsics[idx].inverse(), homogenize_points(gaussians.get_xyz()), "i j, g j -> g i"
        )
        fake_color = camera_space_gaussians[..., 2]
        if mode == "disparity":
            fake_color = 1 / fake_color
        elif mode == "log":
            fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()
        fake_color = repeat(fake_color, "g -> g c", c=3)
        return fake_color

    # Render using depth as color.
    b, _, _ = extrinsics.shape
    result = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=extrinsics.dtype, device=extrinsics.device),
        gaussians=gaussians,
        fake_color_fn=fake_color_fn,
        scale_invariant=scale_invariant,
        use_sh=False,
    )
    return result.mean(dim=1)
