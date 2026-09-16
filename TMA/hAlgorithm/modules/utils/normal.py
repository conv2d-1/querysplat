import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        (b, h + patch_size - 1, w + patch_size - 1, c),
        dtype=xyz.dtype,
        device=xyz.device,
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
    n_img_aver_norm[mask_normal] = n_img_aver_norm[mask_normal] / (
        torch.norm(n_img_aver_norm[mask_normal], dim=-1, keepdim=True) + 1e-8
    )
    return (
        n_img_aver_norm.permute(0, 3, 1, 2).contiguous(),
        mask_normal,
    )  # [b, h, w, 3], [b, h, w]


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
        normals, normal_masks = get_surface_normalv2(pointmap, mask_valid=masks.squeeze())
        normal_masks = normal_masks & masks
        return normals, normal_masks


class Depth2Normal(nn.Module):
    """Layer to compute surface normal from depth map"""

    def __init__(
        self,
    ):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(Depth2Normal, self).__init__()

    def init_img_coor(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        y, x = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32, device="cuda"),
                torch.arange(0, width, dtype=torch.float32, device="cuda"),
            ],
            indexing="ij",
        )
        meshgrid = torch.stack((x, y))

        # generate homogeneous pixel coordinates
        ones = torch.ones((1, 1, height * width), device="cuda")

        xy = meshgrid.reshape(2, -1).unsqueeze(0)
        xy = torch.cat([xy, ones], 1)

        self.register_buffer("xy", xy, persistent=False)

    def back_projection(self, depth, inv_K, img_like_out=False, scale=1.0):
        """
        Args:
            depth (Nx1xHxW): depth map
            inv_K (Nx4x4): inverse camera intrinsics
            img_like_out (bool): if True, the output shape is Nx4xHxW; else Nx4x(HxW)
        Returns:
            points (Nx4x(HxW)): 3D points in homogeneous coordinates
        """
        B, C, H, W = depth.shape
        depth = depth.contiguous()
        # xy = self.init_img_coor(height=H, width=W)
        xy = self.xy  # xy.repeat(depth.shape[0], 1, 1)
        # ones = self.ones.repeat(depth.shape[0],1,1)

        points = torch.matmul(inv_K[:, :3, :3], xy)
        points = depth.view(depth.shape[0], 1, -1) * points
        depth_descale = points[:, 2:3, :] / scale
        points = torch.cat((points[:, 0:2, :], depth_descale), dim=1)
        # points = torch.cat([points, ones], 1)

        if img_like_out:
            points = points.reshape(depth.shape[0], 3, H, W)
        return points

    def forward(self, depth, intrinsics, masks, scale):
        """
        Args:
            depth (Nx1xHxW): depth map
            #inv_K (Nx4x4): inverse camera intrinsics
            intrinsics (Nx4): camera intrinsics
        Returns:
            normal (Nx3xHxW): normalized surface normal
            mask (Nx1xHxW): valid mask for surface normal
        """
        B, C, H, W = depth.shape
        if "xy" not in self._buffers or self.xy.shape[-1] != H * W:
            self.init_img_coor(height=H, width=W)
        # Compute 3D point cloud
        inv_K = intrinsics.inverse()

        xyz = self.back_projection(depth, inv_K, scale=scale)  # [N, 4, HxW]

        xyz = xyz.view(depth.shape[0], 3, H, W)
        xyz = xyz[:, :3].permute(0, 2, 3, 1).contiguous()  # [b, h, w, c]

        normals, normal_masks = get_surface_normalv2(xyz, mask_valid=masks.squeeze())
        normal_masks = normal_masks & masks
        return normals, normal_masks


def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))


def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def pointmap_to_normal_svd(pointmap: torch.Tensor, depth_mask: torch.Tensor, patch_size: int = 3):
    """
    Compute surface normals via local plane fitting using SVD.

    Args:
        pointmap: [B, H, W, 3] or [H, W, 3] — 3D points in camera coordinates
        depth_mask: [B, H, W] or [H, W] — boolean mask (True = valid)
        patch_size: int, must be odd (e.g., 3)

    Returns:
        normals: [B, 3, H, W] — normalized surface normals
        valid_mask: [B, H, W] — True where normal is valid
    """
    assert patch_size % 2 == 1, "patch_size must be odd"
    squeeze_batch = False
    # Standardize pointmap to [B, H, W, 3]
    if pointmap.dim() == 3:
        pointmap = pointmap.unsqueeze(0)  # [H, W, 3] -> [1, H, W, 3]
        squeeze_batch = True
    assert pointmap.dim() == 4 and pointmap.shape[-1] == 3

    # Standardize depth_mask to [B, H, W]
    if depth_mask.dim() == 2:
        depth_mask = depth_mask.unsqueeze(0)  # [H, W] -> [1, H, W]
    assert depth_mask.dim() == 3, f"depth_mask must be 2D or 3D, got {depth_mask.dim()}D"

    B, H, W, _ = pointmap.shape
    assert depth_mask.shape == (
        B,
        H,
        W,
    ), f"Shape mismatch: pointmap {pointmap.shape}, mask {depth_mask.shape}"

    device = pointmap.device

    half = patch_size // 2

    # Pad pointmap and mask
    # Use constant padding with zeros; we'll mask out invalid points later
    pointmap_pad = F.pad(
        pointmap, (0, 0, half, half, half, half), mode="constant", value=0.0
    )  # [B, H+p, W+p, 3]
    mask_pad = F.pad(
        depth_mask.float(), (half, half, half, half), mode="constant", value=0.0
    ).bool()  # [B, H+p, W+p]

    # Unfold to get patches: [B, H, W, patch_size, patch_size, 3]
    patches_xyz = pointmap_pad.unfold(1, patch_size, 1).unfold(
        2, patch_size, 1
    )  # [B, H, W, 3, ps, ps]
    patches_xyz = patches_xyz.permute(0, 1, 2, 4, 5, 3)  # [B, H, W, ps, ps, 3]

    patches_mask = mask_pad.unfold(1, patch_size, 1).unfold(2, patch_size, 1)  # [B, H, W, ps, ps]

    # Reshape to [B*H*W, ps*ps, 3] for batched SVD
    patches_xyz = patches_xyz.reshape(B * H * W, patch_size * patch_size, 3)
    patches_mask = patches_mask.reshape(B * H * W, patch_size * patch_size)

    # Count valid points per patch
    valid_counts = patches_mask.sum(dim=1)  # [B*H*W]

    # Minimum number of points to fit a plane (at least 3 non-collinear)
    min_valid = 3
    enough_points = valid_counts >= min_valid  # [B*H*W]

    # Initialize normals
    normals = torch.zeros(B * H * W, 3, device=device)

    if enough_points.any():
        # Select only patches with enough valid points
        valid_patches_xyz = patches_xyz[enough_points]  # [N, K, 3]
        valid_patches_mask = patches_mask[enough_points]  # [N, K]

        # Zero out invalid points (set to 0, but they won't affect centroid if masked)
        # Better: compute centroid only from valid points
        valid_counts_sel = valid_counts[enough_points].float()  # [N]

        # Compute centroid (mean of valid points)
        weighted_sum = (valid_patches_xyz * valid_patches_mask.unsqueeze(-1)).sum(dim=1)  # [N, 3]
        centroid = weighted_sum / valid_counts_sel.unsqueeze(-1)  # [N, 3]

        # Center the points
        centered = valid_patches_xyz - centroid.unsqueeze(1)  # [N, K, 3]

        # Zero out invalid points in centered (to avoid NaN in SVD)
        centered = centered * valid_patches_mask.unsqueeze(-1)

        # Perform SVD: [N, K, 3] -> compute covariance-like matrix
        # We can use torch.svd on each patch, but it's slow. Instead, compute SVD directly.
        # Note: SVD of centered (N, K, 3) gives U, S, V where V is [N, 3, 3]
        try:
            _, _, V = torch.svd(centered)
        except RuntimeError as e:
            # Fallback: use eig of covariance matrix
            cov = torch.bmm(centered.transpose(-1, -2), centered)  # [N, 3, 3]
            _, V = torch.linalg.eigh(cov)
            V = V.transpose(-1, -2)

        # Normal is the last column of V (smallest singular value)
        normal_est = V[:, :, -1]  # [N, 3]

        # Re-orient: normal should point towards camera (i.e., opposite to view direction)
        # View direction ≈ centroid (from origin to surface)
        dot = (normal_est * centroid).sum(dim=1, keepdim=True)  # [N, 1]
        normal_est = torch.where(dot > 0, -normal_est, normal_est)

        # Store back
        normals[enough_points] = normal_est

    # Reshape back to [B, H, W, 3]
    normals = normals.reshape(B, H, W, 3)
    valid_mask = enough_points.reshape(B, H, W)

    # Normalize (in case of numerical issues)
    norm = torch.norm(normals, dim=-1, keepdim=True)
    normals = normals / (norm + 1e-8)

    # Zero out invalid normals
    normals = normals * valid_mask.unsqueeze(-1)

    # Output format: [B, 3, H, W]
    normals = normals.permute(0, 3, 1, 2).contiguous()

    if squeeze_batch:
        normals = normals.squeeze(0)
        valid_mask = valid_mask.squeeze(0)
    return normals, valid_mask

class PointMap2NormalSVD(nn.Module):
    """Layer to compute surface normal from point map"""

    def __init__(
        self,
    ):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(PointMap2NormalSVD, self).__init__()

    def forward(self, pointmap, masks):
        """
        Args:
            pointmap (B,H,W,3): point map
        Returns:
            normal (B,H,W,3): normalized surface normal
            normal_masks (B,H,W,1): valid mask for surface normal
        """
        normals, normal_masks = pointmap_to_normal_svd(pointmap, masks.squeeze())
        normal_masks = normal_masks & masks
        return normals, normal_masks
