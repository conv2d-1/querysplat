import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation

from hAlgorithm.utils import instantiate_from_config


class LidarPattern:
    def __init__(
        self,
        csv_path,
        blur_t=None,
        patch_crop_t=None,
        pts_jitter_t=None,
        uv_jitter_t=None,
        random_project_t=None,
    ):

        self.pattern_xyz = self._get_pattern_xyz(csv_path)
        self.vitrual_height = 1200
        self.vitrual_width = 1800
        self.virtual_K = torch.tensor([[500.0, 0.0, 899.5], [0.0, 500, 599.5], [0.0, 0.0, 1.0]])
        self.virtual_sparse_mask = self._update_mask(
            self.virtual_K, self.vitrual_height, self.vitrual_width
        )

        T_lidar_cam = (
            torch.tensor(
                [
                    -6.7119238721546903e-03,
                    -9.9997379279737819e-01,
                    2.7136315816647105e-03,
                    9.1279374175436162e-02,
                    -1.0944713540763151e-02,
                    -2.6400686954256569e-03,
                    -9.9993661963286173e-01,
                    5.5663805977618592e-03,
                    9.9991757826505157e-01,
                    -6.7411983882722515e-03,
                    -1.0926706770336470e-02,
                    2.3159830041157750e-02,
                    0,
                    0,
                    0,
                    1,
                ]
            )
            .reshape(4, 4)
            .inverse()
        )
        T_vcam_lidar = torch.eye(4)
        T_vcam_lidar[:3, :3] = torch.from_numpy(
            Rotation.from_euler("yzx", [-90, 90, 0], degrees=True).as_matrix()
        )
        self.T_vcam_cam = T_vcam_lidar @ T_lidar_cam
        self.T_cam_vcam = self.T_vcam_cam.inverse()

        # Transform parameters
        self.blur_t = instantiate_from_config(blur_t) if blur_t else lambda x: x
        self.pts_jitter_t = instantiate_from_config(pts_jitter_t) if pts_jitter_t else lambda x: x
        self.uv_jitter_t = instantiate_from_config(uv_jitter_t) if uv_jitter_t else lambda x: x
        self.patch_crop_t = instantiate_from_config(patch_crop_t) if patch_crop_t else lambda x: x
        self.random_project_t = instantiate_from_config(random_project_t)

    def _get_pattern_xyz(self, csv_path):
        csv_data = np.loadtxt(csv_path, delimiter=",", skiprows=1).astype(np.float32)
        h_angles = csv_data[:, 1] / 180.0 * np.pi
        v_angles = csv_data[:, 2] / 180.0 * np.pi
        rhos = np.ones_like(h_angles)

        ys = -np.sin(v_angles) * rhos
        xzs = np.cos(v_angles) * rhos
        zs = np.cos(h_angles) * xzs
        xs = np.sin(h_angles) * xzs

        xyz = np.vstack([xs, ys, zs]).T
        return torch.from_numpy(xyz)

    def _update_mask(self, Kcam, height, width):
        uvd = self.pattern_xyz @ Kcam.T
        valid_depth_mask = uvd[:, 2] > 0
        uv1 = uvd / uvd[:, 2:]
        us = uv1[:, 0].long()
        vs = uv1[:, 1].long()
        valid_uv_mask = (us >= 0) & (us < width) & (vs >= 0) & (vs < height)
        valid_mask = valid_depth_mask & valid_uv_mask
        self.us = us[valid_mask]
        self.vs = vs[valid_mask]

        mask = torch.zeros((1, height, width), dtype=torch.bool)
        mask[:, self.vs, self.us] = 1.0
        return mask

    def get_sparse_depth(self, pointmap, mask, K, tgth=None, tgtw=None, **kwargs):
        """
        pointmap: 3, h, w
        mask: h, w
        K: 3, 3
        """
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        if isinstance(pointmap, np.ndarray):
            pointmap = torch.from_numpy(pointmap)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).bool()
        if mask.ndim == 3:
            mask = mask.squeeze()

        if self.random_project_t is not None:
            pointmap = self.random_project_t(pointmap)

        pointmap = self.blur_t(pointmap)
        points3D = pointmap[:, mask]

        # ===== project to virtual camera =====
        points3Dv = (self.T_vcam_cam[:3, :3] @ points3D) + self.T_vcam_cam[:3, 3:]
        points2Dv = self.virtual_K @ points3Dv
        depthv = points2Dv[2]
        points2Dv = points2Dv[0:2] / (points2Dv[2:3] + 1e-8)
        uv = torch.round(points2Dv).long()
        valid_mask = (
            (uv[0] >= 0)
            & (uv[0] < self.vitrual_width)
            & (uv[1] >= 0)
            & (uv[1] < self.vitrual_height)
        )
        valid_depthv = depthv[valid_mask]
        valid_uv = uv[:, valid_mask]
        depthmapv = np.zeros_like(self.virtual_sparse_mask, dtype=np.float32)
        depthmapv += np.inf
        np.minimum.at(depthmapv[0], (valid_uv[1], valid_uv[0]), valid_depthv)
        depthmapv[depthmapv == np.inf] = 0.0
        mask = self.patch_crop_t(self.virtual_sparse_mask.clone())
        depthmapv[~mask] = 0
        depthmapv = torch.from_numpy(depthmapv)
        # ===== project back to real camera =====
        depthv = depthmapv[depthmapv > 0]
        vu = torch.nonzero(depthmapv)
        v, u = vu[:, 1], vu[:, 2]
        points3Dv = self.virtual_K.inverse() @ (torch.vstack([u, v, torch.ones_like(u)]) * depthv)
        points3D = (self.T_cam_vcam[:3, :3] @ points3Dv) + self.T_cam_vcam[:3, 3:]
        # depth = points3D[2]
        points2D = K @ points3D
        uv = points2D[0:2] / (points2D[2:3] + 1e-8)
        uv = torch.round(uv).long()
        if tgth is None and tgtw is None:
            _, h, w = pointmap.shape
        else:
            h, w = tgth, tgtw

        uv = self.uv_jitter_t(uv)
        valid_mask = (uv[0] >= 0) & (uv[0] < w) & (uv[1] >= 0) & (uv[1] < h)
        uv = uv[:, valid_mask]

        sparse_pointmap = torch.zeros(3, h, w)
        sparse_pointmap_mask = torch.zeros(1, h, w, dtype=torch.bool)

        points3D = self.pts_jitter_t(points3D.float())
        sparse_pointmap[:, uv[1], uv[0]] = points3D[:, valid_mask]
        sparse_pointmap_mask[0, uv[1], uv[0]] = True

        return sparse_pointmap, sparse_pointmap_mask


if __name__ == "__main__":
    csv_path = "/mnt/personal/zm/VLidar/Config/ft2_pattern.csv"
    pattern = LidarPattern(csv_path)

    pointmap = np.load(
        "/mnt/personal/lzh/moge_test/hypersim_original/pointmap/Hypersim/evermotion_dataset/scenes/ai_003_010/images/scene_cam_00_final_preview/frame.0047.tonemap.npy"
    )
    pointmap = torch.from_numpy(pointmap).permute(2, 0, 1)
    mask = pointmap[2] > 0
    K = [886.81, 0, 512.0, 0, 886.81, 384.0, 0, 0, 1]
    mask_sparse, pointmap_sparse = pattern.get_sparse_depth(pointmap, mask, K)
