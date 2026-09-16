from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from pytorch3d import transforms
from torch import Tensor, nn

from hAlgorithm.modules.utils.gaussians.projection import get_world_rays
from hAlgorithm.modules.utils.gaussians.sh_rotation import rotate_sh
from hAlgorithm.modules.utils.gaussians.types import GaussiansV2 as Gaussians


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
        intrinsics: Float[Tensor, "b v 1 1 1 3 3"] | None,
        coordinates: Float[Tensor, "b v n 1 1 2"],
        depths: Float[Tensor, "b v n 1 1"] | None,
        opacities: Float[Tensor, "b v n 1 1"],
        raw_gaussians: Float[Tensor, "b v n 1 1 c"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
    ) -> Gaussians:
        # TODO: compute scales and rotations from pointmap, predict their residuals
        if self.offset:
            offset, mask, scales, rotations, sh = raw_gaussians.split(
                (3, 1, 3, 4, 3 * self.d_sh), dim=-1
            )
        else:
            mask, scales, rotations, sh = raw_gaussians.split((1, 3, 4, 3 * self.d_sh), dim=-1)
        # scales will be activated with exp, so subtract 5 to make it smaller
        scales = scales - 4
        # 看一下这个scale的分布情况。load模型
        # scales[..., -1] = scales[..., -1] * 0 - 6  # make the last dimension small
        mask = mask.sigmoid()[..., 0] > 0.5  # [b, v, n, 1, 1]

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)
        # https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2127778441

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # transform rotations to world-space
        # TODO: double check if this is correct
        c2w_rotations = extrinsics[..., :3, :3]
        if not torch.allclose(torch.det(c2w_rotations), c2w_rotations.new_tensor(1.0)):
            cw2_quat = transforms.matrix_to_quaternion(c2w_rotations)
            c2w_rotations = transforms.quaternion_to_matrix(cw2_quat)

        rotations = c2w_rotations @ transforms.quaternion_to_matrix(rotations)
        rotations = transforms.matrix_to_quaternion(rotations)

        # Compute Gaussian means.
        origins, directions = get_world_rays(
            coordinates, extrinsics, intrinsics, with_normalize=False
        )
        means = origins + directions * depths[..., None]

        norm_scales = None
        if self.normalize:
            ba = means.shape[0]
            norm_scales = torch.quantile(
                torch.linalg.norm(means.reshape(ba, -1, 3), dim=-1), 0.5, dim=1
            )

        if self.offset:
            if norm_scales is not None:
                means = means + offset * norm_scales
            else:
                means = means + offset

        gaussians = Gaussians(
            means=rearrange(
                means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            harmonics=rearrange(
                rotate_sh(sh, c2w_rotations[..., None, :, :]),
                "b v r srf spp c d_sh -> b (v r srf spp) d_sh c",
            ),
            opacities=rearrange(
                opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
            scales=rearrange(
                scales,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rotations=rearrange(
                rotations.broadcast_to((*scales.shape[:-1], 4)),
                "b v r srf spp xyzq -> b (v r srf spp) xyzq",
            ),
            mask=rearrange(
                mask,
                "b v r srf spp -> b (v r srf spp)",
            ),
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
