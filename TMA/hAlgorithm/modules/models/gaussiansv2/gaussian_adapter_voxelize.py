import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from pytorch3d import transforms
from torch import Tensor, nn
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.voxel_grid import voxel_grid
from torch_geometric.utils import scatter, softmax

from hAlgorithm.modules.utils.gaussians.projection import get_world_rays
from hAlgorithm.modules.utils.gaussians.sh_rotation import rotate_sh
from hAlgorithm.modules.utils.gaussians.types import GaussiansV2 as Gaussians


def scatter_weighted_mean(
    features: Tensor,
    weights: Tensor,
    cluster: Tensor,
    weights_cluster: Tensor,
    dim: int,
) -> Tensor:
    """_summary_

    Args:
        features (Tensor): [N, D] features at each point
        weights (Optional[Tensor], optional): [N,] weights of each point. Defaults to None.
        cluster (LongTensor): [N] IDs of each point (clusters.max() should be <= N, or you'll OOM)
        weights_cluster (Tensor): [N,] aggregated weights of each cluster, used to normalize
        dim (int): Dimension along which to do the reduction -- should be 0

    Returns:
        Tensor: Agggregated features, weighted by weights and normalized by weights_cluster
    """
    assert dim == 0, "Dim != 0 not yet implemented"
    feature_cluster = scatter(features * weights[:, None], cluster, dim=dim, reduce="sum")
    feature_cluster = feature_cluster / weights_cluster[:, None]
    return feature_cluster


def voxelize(
    pos: Tensor,
    voxel_size: float,
    batch: Optional[Tensor] = None,
    start: Optional[Union[float, Tensor]] = None,
    end: Optional[Union[float, Tensor]] = None,
) -> Tuple[Tensor]:
    """Returns voxel indices and packed (consecutive) indices for points

    Args:
        pos (Tensor): [N, 3] locations
        voxel_size (float): Size (resolution) of each voxel in the grid
        batch (Optional[Tensor], optional): Batch index of each point in pos. Defaults to None.
        start (Optional[Union[float, Tensor]], optional): Mins along each coordinate for the voxel grid.
            Defaults to None, in which case the starts are inferred from min values in pos.
        end (Optional[Union[float, Tensor]], optional):  Maxes along each coordinate for the voxel grid.
            Defaults to None, in which case the starts are inferred from max values in pos.
    Returns:
        voxel_idx (LongTensor): Idx of each point's voxel coordinate. E.g. [0, 0, 4, 3, 3, 4]
        cluster_consecutive_idx (LongTensor): Packed idx -- contiguous in cluster ID. E.g. [0, 0, 2, 1, 1, 2]
        batch_sample: See https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/max_pool.html
    """
    voxel_cluster = voxel_grid(pos=pos, batch=batch, size=voxel_size, start=start, end=end)
    cluster_consecutive_idx, perm = consecutive_cluster(voxel_cluster)
    batch_sample = batch[perm] if batch is not None else None
    cluster_idx = voxel_cluster
    return cluster_idx, cluster_consecutive_idx, batch_sample


def reduce_pointcloud(
    voxel_cluster: Tensor,
    pos: Tensor,
    features: Tensor,
    weights: Optional[Tensor] = None,
    feature_reduce: str = "mean",
) -> Tuple[Tensor]:
    """Pools values within each voxel

    Args:
        voxel_cluster (LongTensor): [N] IDs of each point
        pos (Tensor): [N, 3] position of each point
        features (Tensor): [N, D] features at each point
        weights (Optional[Tensor], optional): [N,] weights of each point. Defaults to None.
        rgbs (Optional[Tensor], optional): [N, 3] colors of each point. Defaults to None.
        feature_reduce (str, optional): Feature reduction method. Defaults to 'mean'.

    Raises:
        NotImplementedError: if unknown reduction method

    Returns:
        pos_cluster (Tensor): weighted average position within each voxel
        feature_cluster (Tensor): aggregated feature of each voxel
        weights_cluster (Tensor): aggregated weights of each voxel
        rgb_cluster (Tensor): colors of each voxel
    """
    if weights is None:
        weights = torch.ones_like(pos[..., 0])
    weights_cluster = scatter(weights, voxel_cluster, dim=0, reduce="sum")

    pos_cluster = scatter_weighted_mean(pos, weights, voxel_cluster, weights_cluster, dim=0)

    if feature_reduce == "mean":
        feature_cluster = scatter_weighted_mean(
            features, weights, voxel_cluster, weights_cluster, dim=0
        )
    elif feature_reduce == "max":
        feature_cluster = scatter(features, voxel_cluster, dim=0, reduce="max")
    elif feature_reduce == "sum":
        feature_cluster = scatter(features * weights[:, None], voxel_cluster, dim=0, reduce="sum")
    else:
        raise NotImplementedError(f"Unknown feature reduction method {feature_reduce}")

    return pos_cluster, feature_cluster


class GaussianAdapterVoxelize(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sh_degree = cfg["sh_degree"]
        self.gaussian_scale_min = cfg["gaussian_scale_min"]
        self.gaussian_scale_max = cfg["gaussian_scale_max"]
        self.normalize = cfg.get("normalize", False)
        self.offset = cfg.get("offset", False)
        self.voxel_size = cfg.get("voxel_size", 0.025)

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
        depths: Float[Tensor, "b v n 1 c"] | None,
        opacities: Float[Tensor, "b v n 1 1"],
        raw_gaussians: Float[Tensor, "b v n 1 1 c"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
    ) -> Gaussians:
        if self.offset:
            offset, mask, scales, rotations, sh = raw_gaussians.split(
                (3, 1, 3, 4, 3 * self.d_sh), dim=-1
            )
        else:
            offset = None
            mask, scales, rotations, sh = raw_gaussians.split((1, 3, 4, 3 * self.d_sh), dim=-1)
        # scales will be activated with exp, so subtract 3 to make it smaller
        scales = scales - 3
        mask = mask[..., 0]  # [b, v, n, 1, 1]  use as weights

        # Compute Gaussian means.
        if depths.shape[-1] == 1:
            origins, directions = get_world_rays(
                coordinates, extrinsics, intrinsics, with_normalize=False
            )
            means = origins + directions * depths[..., None]
        else:
            # TODO: depths is pointmap, merge them to a single point cloud with extrinsics
            # this may produce more directly gradient flow for extrinsics and depths
            pass

        ba = means.shape[0]
        norm_scales = (
            torch.quantile(torch.linalg.norm(means.reshape(ba, -1, 3), dim=-1), 0.9, dim=1) * 0.1
        )

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # transform rotations to world-space
        c2w_rotations = extrinsics[..., :3, :3]
        if not torch.allclose(torch.det(c2w_rotations), c2w_rotations.new_tensor(1.0)):
            # logging.warning("c2w_rotations is not orthogonal")
            cw2_quat = transforms.matrix_to_quaternion(c2w_rotations)
            c2w_rotations = transforms.quaternion_to_matrix(cw2_quat)

        rotations = c2w_rotations @ transforms.quaternion_to_matrix(rotations)
        rotations = transforms.matrix_to_quaternion(rotations)

        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh * self.sh_mask
        sh = rotate_sh(sh, c2w_rotations[..., None, :, :])

        sh = sh.reshape(-1, 3, self.d_sh).reshape(-1, 3 * self.d_sh)
        scales = scales.reshape(-1, 3)
        rotations = rotations.reshape(-1, 4)
        opacities = opacities.reshape(-1)

        if offset is not None:
            offset = offset.reshape(-1, 3)
            feat = torch.cat([scales, rotations, sh, opacities.unsqueeze(-1), offset], dim=-1)
        else:
            feat = torch.cat([scales, rotations, sh, opacities.unsqueeze(-1)], dim=-1)

        weights = mask.reshape(-1)
        with torch.no_grad():
            means = means / norm_scales
            means = means.reshape(-1, means.shape[-1])
            batch = (
                torch.kron(torch.arange(start=0, end=ba), torch.ones(means.size(0)))
                .long()
                .to(device=means.device)
            )
            cluster_voxel_idx, cluster_consecutive_idx, _ = voxelize(
                means, voxel_size=self.voxel_size, batch=batch
            )
            weights = softmax(weights, index=cluster_consecutive_idx)
            means = means * norm_scales

        means, feat = reduce_pointcloud(
            cluster_consecutive_idx, pos=means, features=feat, weights=weights
        )

        if offset is not None:
            scales, rotations, sh, opacities, offset = feat.split((3, 4, 3 * self.d_sh, 1, 3), dim=-1)
            means = means + offset * norm_scales
        else:
            scales, rotations, sh, opacities = feat.split((3, 4, 3 * self.d_sh, 1), dim=-1)

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        sh = sh.reshape(-1, 3, self.d_sh).transpose(-1, -2)

        gaussians = Gaussians(
            means=means.unsqueeze(0),
            harmonics=sh.unsqueeze(0),
            opacities=opacities.unsqueeze(0).squeeze(-1),
            scales=scales.unsqueeze(0),
            rotations=rotations.unsqueeze(0),
            mask=None,
            norm_scales=norm_scales if self.normalize else None,
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
