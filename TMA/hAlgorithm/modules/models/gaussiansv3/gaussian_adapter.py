from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from pytorch3d import transforms
from torch import Tensor, nn

from hAlgorithm.modules.utils.gaussians.projection import get_world_rays
from hAlgorithm.modules.utils.gaussians.sh_rotation import rotate_sh
from hAlgorithm.modules.utils.gaussians.types import GaussiansV2 as Gaussians


@torch.no_grad()
def compute_rotation_quat(normal: Float[Tensor, "n 3"]) -> Float[Tensor, "n 4"]:
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=normal.dtype, device=normal.device).unsqueeze(0)
    axis = torch.linalg.cross(z_axis, normal)
    axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-8)
    angle = torch.acos(torch.clamp((z_axis * normal).sum(dim=1, keepdim=True), -1.0, 1.0))
    rotvec = axis * angle
    quat = transforms.axis_angle_to_quaternion(rotvec)
    return quat.view(-1, 4)


class GaussianAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sh_degree = cfg["sh_degree"]
        self.gaussian_scale_min = cfg["gaussian_scale_min"]
        self.gaussian_scale_max = cfg["gaussian_scale_max"]
        self.normalize = cfg.get("normalize", False)
        self.offset = cfg.get("offset", False)

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "b v 1 1 1 4 4"],
        means: Float[Tensor, "b n xyz"] | None,
        rgb: Float[Tensor, "b n 1 rgb"] | None,
        raw_gaussians: Float[Tensor, "b n 1 c"],
        normal: Float[Tensor, "b n () 3"] | None = None,
        eps: float = 1e-8,
    ) -> Gaussians:
        b = means.shape[0]
        # TODO: compute scales and rotations from pointmap, predict their residuals
        opacities, offset, mask, scales, rotations, sh = raw_gaussians.split(
            (1, 3, 1, 3, 4, 3 * self.d_sh), dim=-1
        )
        opacities = opacities[..., 0]
        # scales will be activated with exp, so subtract 4 to make it smaller
        # scales = scales - 4
        scales = torch.sigmoid(scales - 1) * 4 - 5
        # scales[..., -1] = scales[..., -1] * 0.0  - 5   # make the last dimension small
        mask = mask.sigmoid()[..., 0] > 0.5

        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask  # [b n 3 dsh]

        if rgb is not None:
            sh[..., 0] = sh[..., 0] + RGB2SH(rgb)

        # Normalize the quaternion features to yield a valid quaternion.
        if normal is None:
            rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        else:
            rotations_delta = rotations
            rotations = compute_rotation_quat(
                rearrange(normal, "b n () xyz -> (b n) xyz")
            )  # [b*n, 4]
            rotations = rearrange(rotations, "(b n) qxyz -> b n () qxyz", b=b)
            rotations = rotations + rotations_delta
            rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        norm_scales = None
        if self.normalize:
            ba = means.shape[0]
            norm_scales = (
                torch.quantile(torch.linalg.norm(means.reshape(ba, -1, 3), dim=-1), 0.9, dim=1)
                * 0.1
            )

        if self.offset:
            offset = rearrange(offset, "b n () xyz -> b n xyz")
            offset = offset.sigmoid() * 0.5 - 0.25  # [-0.25, 0.25]
            if norm_scales is not None:
                means = means + offset * norm_scales
            else:
                means = means + offset

        gaussians = Gaussians(
            means=means,
            harmonics=rearrange(sh, "b n 1 c d_sh -> b n d_sh c"),
            opacities=rearrange(opacities, "b n 1 -> b n"),
            scales=rearrange(scales, "b n () xyz -> b n xyz"),
            rotations=rearrange(rotations, "b n 1 qxyz -> b n qxyz"),
            mask=rearrange(mask, "b n 1 -> b n"),
            norm_scales=norm_scales,
        )

        return gaussians

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0
