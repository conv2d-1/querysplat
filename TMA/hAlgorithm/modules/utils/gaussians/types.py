import logging
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import quaternion_to_matrix
from torch import Tensor


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


@dataclass
class Gaussians:
    # TODO: Feature aggregation using PointCloud pooling
    # https://github.com/ok-robot/ok-robot/blob/174c742b6a1866a2da8f75a89e2c544bb77ab5f9/ok-robot-navigation/voxel_map/voxel.py#L171
    # https://github.com/MIT-SPARK/ensemble_pose/blob/95b0ec9002ed0406e6283bfedc01769e18ac034a/src/casper3d/point_transformer.py#L581
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]


class GaussiansV2:
    def __init__(
        self,
        means: Float[Tensor, "batch g 3"] | None = None,
        offset: Float[Tensor, "batch g 3"] | None = None,
        scales: Float[Tensor, "batch g 3"] | None = None,  # before exp activation
        rotations: Float[Tensor, "batch g 4"] | None = None,  # normalized, world space
        harmonics: Float[Tensor, "batch g d_sh 3"] | None = None,
        opacities: Float[Tensor, " batch"] | None = None,  # before sigmoid
        mask: Float[Tensor, "batch"] | None = None,
        norm_scales: Float[Tensor, "batch"] | None = None,
        dtype=torch.float32,
    ):

        self.batch_idx = 0
        bs = means.shape[0]
        self.means = [mean.squeeze(0) for mean in means.to(dtype=dtype).split(bs, dim=0)]
        self.scales = [scales.squeeze(0) for scales in scales.to(dtype=dtype).split(bs, dim=0)]
        self.rotations = [
            rotations.squeeze(0) for rotations in rotations.to(dtype=dtype).split(bs, dim=0)
        ]
        self.harmonics = [
            harmonics.squeeze(0) for harmonics in harmonics.to(dtype=dtype).split(bs, dim=0)
        ]
        self.opacities = [
            opacities.squeeze(0) for opacities in opacities.to(dtype=dtype).split(bs, dim=0)
        ]

        self.scaling_activation = lambda x: torch.clamp(torch.exp(x), max=0.5)
        self.opacity_activation = torch.sigmoid

        self.norm_scales = norm_scales

        if self.norm_scales is not None:
            # self.means = [means / s for (means, s) in zip(self.means, self.norm_scales)]
            self.means = [self.means[i] / self.norm_scales[i] for i in range(bs)]

        if offset is not None:
            self.means = [self.means[i] + offset[i] for i in range(bs)]

    @property
    def batch_size(self):
        return len(self.means) if self.means is not None else 0

    def construct_list_of_attributes(self, save_rest=False):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self.harmonics[0].shape[-1] * 1):
            l.append("f_dc_{}".format(i))
        if save_rest:
            for i in range(self.harmonics[0].shape[-1] * (self.harmonics[0].shape[-2] - 1)):
                l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scales[0].shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.rotations[0].shape[1]):
            l.append("rot_{}".format(i))
        return l

    def export_ply(self, path=None, save_rest=False):
        xyz = self.means[0].detach().cpu().numpy()
        if self.norm_scales is not None:
            xyz = xyz * self.norm_scales[0].item()
        normals = np.zeros_like(xyz)
        # f_dc: [B, n, d_sh**2, 3] -> [n, 1, 3]
        f_dc = (
            self.harmonics[0][..., 0:1, :].detach().flatten(start_dim=1).contiguous().cpu().numpy()
        )  # 3
        # f_rest：[B, n, d_sh**2, 3] -> [n, 3, 15]
        if save_rest:
            f_rest = (
                self.harmonics[0][..., 1:, :]
                .detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )  # 45

        opacities = self.opacities[0].detach().cpu().numpy()[:, None]
        scale = self.scales[0].detach().cpu().numpy()
        if self.norm_scales is not None:
            scale = np.log(np.exp(scale) * self.norm_scales[0].item())
        rotation = self.rotations[0].detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes(save_rest)
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if save_rest:
            attributes = np.concatenate(
                (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
            )
        else:
            attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")

        if path is None:
            return PlyData([el])

        PlyData([el]).write(path)
        logging.info(f"Saved ply to {path}")

    def get_norm_scale(self):
        if self.norm_scales is not None:
            return self.norm_scales[self.batch_idx]
        else:
            return None

    def get_xyz(self):
        return self.means[self.batch_idx]

    def get_opacity(self):
        opacity = torch.sigmoid(self.opacities[self.batch_idx])[..., None]
        opacity = (opacity > 0.01).float() * opacity
        return opacity

    def get_scale(self):
        return self.scaling_activation(self.scales[self.batch_idx])

    def get_rotation(self):
        return self.rotations[self.batch_idx]

    def get_shs(self):
        return self.harmonics[self.batch_idx]

    def get_mask(self):
        return torch.ones_like(self.mask[self.batch_idx])

    def __len__(self):
        return self.batch_size

    def __getitem__(self, batch_idx):
        self.batch_idx = batch_idx
        return self

    def apply_mask(self, mask):
        # return # TODO: handle mask
        bs = mask.shape[0]
        mask = mask.reshape(bs, -1)
        for i in range(bs):
            self.means[i] = self.means[i][mask[i]]
            self.scales[i] = self.scales[i][mask[i]]
            self.rotations[i] = self.rotations[i][mask[i]]
            self.harmonics[i] = self.harmonics[i][mask[i]]
            self.opacities[i] = self.opacities[i][mask[i]]
            # if self.mask is not None:
            #     self.mask[i] = self.mask[i][mask[i]]

    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation())

    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scale().min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)

    def get_normal(self, camera_center):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = camera_center - self.get_xyz()
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global

    def get_valid_ratio(self):
        """
        return the ratio of valid gaussians, 0-1
        """
        return torch.mean(self.mask.float())

    def __repr__(self):
        repr_str = self.__class__.__name__ + "(\n"
        attributes = ["means", "covariances", "scales", "rotations", "harmonics", "opacities"]
        for attribute in attributes:
            data = getattr(self, attribute, None)
            if data is not None:
                repr_str += f"\t{attribute}=[{data.min().item():.5f}, {data.max().item():.5f}],\n"
        repr_str += ")"
        return repr_str
