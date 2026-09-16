import numpy as np
import torch
import torch.nn as nn


def get_surface_normalv2(xyz, patch_size=5, mask_valid=None):
    """
    xyz: xyz coordinates, in [b, h, w, c]
    patch: [p1, p2, p3,
            p4, p5, p6,
            p7, p8, p9]
    surface_normal = [(p9-p1) x (p3-p7)] + [(p6-p4) - (p8-p2)]
    return: normal [h, w, 3, b]
    """
    b, h, w, c = xyz.shape
    half_patch = patch_size // 2

    if mask_valid == None:
        mask_valid = xyz[:, :, :, 2] > 0  # [b, h, w]
    mask_pad = torch.zeros(
        (b, h + patch_size - 1, w + patch_size - 1), device=mask_valid.device
    ).bool()
    mask_pad[:, half_patch:-half_patch, half_patch:-half_patch] = mask_valid

    xyz_pad = torch.zeros(
        (b, h + patch_size - 1, w + patch_size - 1, c), dtype=xyz.dtype, device=xyz.device
    )
    xyz_pad[:, half_patch:-half_patch, half_patch:-half_patch, :] = xyz

    xyz_left = xyz_pad[:, half_patch : half_patch + h, :w, :]  # p4
    xyz_right = xyz_pad[:, half_patch : half_patch + h, -w:, :]  # p6
    xyz_top = xyz_pad[:, :h, half_patch : half_patch + w, :]  # p2
    xyz_bottom = xyz_pad[:, -h:, half_patch : half_patch + w, :]  # p8
    xyz_horizon = xyz_left - xyz_right  # p4p6
    xyz_vertical = xyz_top - xyz_bottom  # p2p8

    xyz_left_in = xyz_pad[:, half_patch : half_patch + h, 1 : w + 1, :]  # p4
    xyz_right_in = xyz_pad[
        :, half_patch : half_patch + h, patch_size - 1 : patch_size - 1 + w, :
    ]  # p6
    xyz_top_in = xyz_pad[:, 1 : h + 1, half_patch : half_patch + w, :]  # p2
    xyz_bottom_in = xyz_pad[
        :, patch_size - 1 : patch_size - 1 + h, half_patch : half_patch + w, :
    ]  # p8
    xyz_horizon_in = xyz_left_in - xyz_right_in  # p4p6
    xyz_vertical_in = xyz_top_in - xyz_bottom_in  # p2p8

    n_img_1 = torch.cross(xyz_horizon_in, xyz_vertical_in, dim=3)
    n_img_2 = torch.cross(xyz_horizon, xyz_vertical, dim=3)

    # re-orient normals consistently
    orient_mask = torch.sum(n_img_1 * xyz, dim=3) > 0
    n_img_1[orient_mask] *= -1
    orient_mask = torch.sum(n_img_2 * xyz, dim=3) > 0
    n_img_2[orient_mask] *= -1

    n_img1_L2 = torch.sqrt(torch.sum(n_img_1**2, dim=3, keepdim=True) + 1e-4)
    n_img1_norm = n_img_1 / (n_img1_L2 + 1e-8)

    n_img2_L2 = torch.sqrt(torch.sum(n_img_2**2, dim=3, keepdim=True) + 1e-4)
    n_img2_norm = n_img_2 / (n_img2_L2 + 1e-8)

    # average 2 norms
    n_img_aver = n_img1_norm + n_img2_norm
    n_img_aver_L2 = torch.sqrt(torch.sum(n_img_aver**2, dim=3, keepdim=True) + 1e-4)
    n_img_aver_norm = n_img_aver / (n_img_aver_L2 + 1e-8)
    # re-orient normals consistently
    orient_mask = torch.sum(n_img_aver_norm * xyz, dim=3) > 0
    n_img_aver_norm[orient_mask] *= -1
    # n_img_aver_norm_out = n_img_aver_norm.permute((1, 2, 3, 0))  # [h, w, c, b]

    # get mask for normals
    mask_p4p6 = (
        mask_pad[:, half_patch : half_patch + h, :w] & mask_pad[:, half_patch : half_patch + h, -w:]
    )
    mask_p2p8 = (
        mask_pad[:, :h, half_patch : half_patch + w] & mask_pad[:, -h:, half_patch : half_patch + w]
    )
    mask_normal = mask_p2p8 & mask_p4p6
    n_img_aver_norm[~mask_normal] = 0

    # a = torch.sum(n_img1_norm_out*n_img2_norm_out, dim=2).cpu().numpy().squeeze()
    # plt.imshow(np.abs(a), cmap='rainbow')
    # plt.show()
    n_img_aver_norm[mask_normal] = n_img_aver_norm[mask_normal] / torch.norm(
        n_img_aver_norm[mask_normal], dim=-1, keepdim=True
    )
    return n_img_aver_norm.permute(0, 3, 1, 2).contiguous(), mask_normal  # [b, h, w, 3], [b, h, w]


class PointMap2Normal(nn.Module):
    """Layer to compute surface normal from point map"""

    def __init__(
        self,
    ):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(PointMap2Normal, self).__init__()

    def forward(self, pointmap, masks):
        """
        Args:
            pointmap (B,H,W,3): point map
        Returns:
            normal (B,H,W,3): normalized surface normal
            normal_masks (B,H,W,1): valid mask for surface normal
        """
        normals, normal_masks = get_surface_normalv2(pointmap, mask_valid=masks)
        normal_masks = normal_masks & masks
        return normals, normal_masks


if __name__ == "__main__":
    import open3d as o3d

    d2n = PointMap2Normal()
    pointmap = np.load("/mnt/personal/lzh/nyu_test/pointmap/rgb_0010.npy")[None, ...]

    pointmap = torch.from_numpy(pointmap).cuda().float()
    mask = torch.ones_like(pointmap[..., 0]).bool().cuda()
    breakpoint()
    normal_pred, normal_mask_pred = d2n(pointmap, mask)
    normal_pred = normal_pred.permute(0, 2, 3, 1).contiguous()

    pcd = o3d.geometry.PointCloud()
    pointmap = pointmap[normal_mask_pred].squeeze()
    normal = normal_pred[normal_mask_pred].squeeze()
    pcd.points = o3d.utility.Vector3dVector(pointmap.reshape(-1, 3).cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(normal_pred.reshape(-1, 3).cpu().numpy())
    o3d.io.write_point_cloud("pointmap.ply", pcd)
