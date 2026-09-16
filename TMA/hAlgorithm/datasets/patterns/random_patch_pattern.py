import random

import cv2
import numpy as np
import torch

from hAlgorithm.utils import instantiate_from_config


class Pattern:
    def __init__(
        self,
        sparse_ratio=0.05,
        sparse_nums=None,
        seed=0,
        blur_t=None,
        patch_crop_t=None,
        pts_jitter_t=None,
        uv_jitter_t=None,
        random_noise=None,
        project_noise_t=None,
        resize_compression=None,
        min_valid_num=500,
        patch_size=14,
        patch_dropout_ratio=None,
    ):
        self.sparse_ratio = sparse_ratio
        self.sparse_nums = sparse_nums
        self.seed = seed
        self.min_valid_num = min_valid_num

        self.patch_size = patch_size
        self.patch_dropout_ratio = patch_dropout_ratio

        # Transform parameters
        self.blur_t = instantiate_from_config(blur_t) if blur_t else lambda x: x
        self.pts_jitter_t = instantiate_from_config(pts_jitter_t) if pts_jitter_t else lambda x: x
        self.uv_jitter_t = instantiate_from_config(uv_jitter_t) if uv_jitter_t else lambda x: x
        self.random_noise = instantiate_from_config(random_noise)
        self.resize_compression = instantiate_from_config(resize_compression)
        self.project_noise_t = instantiate_from_config(project_noise_t)

    def load_sparse(self, depth, valid_mask):
        """
        Randomly selects a certain ratio of valid points from the given depth map and valid mask to generate a sparse depth map.

        Args:
            depth (torch.Tensor): The original depth map.
            valid_mask (torch.Tensor): A boolean mask indicating which points are valid.

        Returns:
            tuple: A tuple containing the sparsified depth map and the newly generated selection mask.
        """
        # Clone the original depth map to create a sparse depth map
        sparse_depth = depth.clone()

        # Get the indices of all valid points using the valid mask
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)

        # Count the number of valid points
        num_valid_points = len(valid_indices[0])

        # Determine the number of points to select based on a threshold
        if num_valid_points < self.min_valid_num:
            num_to_select = (
                num_valid_points  # Select all points if there are fewer than min_valid_num
            )
        elif self.sparse_nums is not None:
            if isinstance(self.sparse_nums, (list, tuple)):
                min_num, max_num = min(self.sparse_nums), max(self.sparse_nums)
                num_to_select = int(np.random.uniform(min_num, max_num))
            else:
                num_to_select = int(self.sparse_nums)
        elif isinstance(self.sparse_ratio, float):
            num_to_select = int(
                num_valid_points * self.sparse_ratio
            )  # Otherwise, select a ratio of points
        else:
            min_ratio, max_ratio = min(self.sparse_ratio), max(self.sparse_ratio)
            sparse_ratio = np.random.uniform(min_ratio, max_ratio)
            num_to_select = int(num_valid_points * sparse_ratio)

        # Set a manual seed for reproducibility of random operations
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Randomly permute the indices of valid points and select the desired number of points
        random_indices = torch.randperm(num_valid_points)[:num_to_select]

        # Use the random indices to select the corresponding valid indices
        selected_valid_indices = tuple(idx[random_indices] for idx in valid_indices)

        # Create a mask that is True only at the locations of the selected valid points
        selected_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        selected_mask[selected_valid_indices] = True

        # Zero out the values in the sparse depth map where the selected mask is False
        # Note: There seems to be a typo in the original code. It should be 'sparse_depth' instead of 'sparse_depth_norm'
        if selected_mask.shape[0] != sparse_depth.shape[0]:
            selected_mask = selected_mask.repeat(sparse_depth.shape[0], 1, 1)
        sparse_depth[~selected_mask] = 0
        selected_mask = selected_mask[0:1, ...]

        # Return the sparse depth map and the selection mask
        return sparse_depth, selected_mask

    def patch_crop(self, sparse_pointmap, sparse_pointmap_mask):
        """
        sparse_pointmap 3,H,W
        sparse_pointmap_mask 1,H,W
        """
        if self.patch_dropout_ratio is None:
            return sparse_pointmap, sparse_pointmap_mask

        patch_size = self.patch_size
        dropout_ratio = np.random.uniform(*self.patch_dropout_ratio)

        C, H, W = sparse_pointmap.shape
        
        # 检查是否可整除（简单处理：裁剪到最后完整 patch）
        H_patch = (H // patch_size) * patch_size
        W_patch = (W // patch_size) * patch_size

        # 裁剪到可整除尺寸
        spm_crop = sparse_pointmap[:, :H_patch, :W_patch]
        spm_mask_crop = sparse_pointmap_mask[:, :H_patch, :W_patch]

        # reshape to (C, H//P, P, W//P, P)
        spm_reshaped = spm_crop.view(C, H_patch // patch_size, patch_size, W_patch // patch_size, patch_size)
        mask_reshaped = spm_mask_crop.view(1, H_patch // patch_size, patch_size, W_patch // patch_size, patch_size)

        # 合并空间 patch 维度: (C, N_patches, P*P)
        spm_patches = spm_reshaped.permute(0, 1, 3, 2, 4).contiguous().view(C, -1, patch_size * patch_size)
        mask_patches = mask_reshaped.permute(0, 1, 3, 2, 4).contiguous().view(1, -1, patch_size * patch_size)

        N_patches = spm_patches.shape[1]

        # 随机选择要丢弃的 patch 索引
        num_drop = int(N_patches * dropout_ratio)
        if num_drop > 0:
            # 生成随机排列
            rand_idx = torch.randperm(N_patches)[:num_drop]
            # 将选中的 patch 置零
            spm_patches[:, rand_idx, :] = 0
            mask_patches[:, rand_idx, :] = False

        # 重建图像
        spm_recon = spm_patches.view(C, H_patch // patch_size, W_patch // patch_size, patch_size, patch_size)
        mask_recon = mask_patches.view(1, H_patch // patch_size, W_patch // patch_size, patch_size, patch_size)

        spm_recon = spm_recon.permute(0, 1, 3, 2, 4).contiguous().view(C, H_patch, W_patch)
        mask_recon = mask_recon.permute(0, 1, 3, 2, 4).contiguous().view(1, H_patch, W_patch)

        # 填回原图（保留边缘未处理部分为原值）
        sparse_pointmap[:, :H_patch, :W_patch] = spm_recon
        sparse_pointmap_mask[:, :H_patch, :W_patch] = mask_recon

        return sparse_pointmap, sparse_pointmap_mask


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

        if self.project_noise_t is not None:
            pointmap = self.project_noise_t(pointmap)

        pointmap = self.blur_t(pointmap)
        if self.random_noise is not None:
            dep = torch.clone(pointmap[2:3, :, :])
            pointmap = self.random_noise(pointmap, mask, edge_inputs=dep)
        if self.resize_compression is not None:
            pointmap = self.resize_compression(pointmap, mask)

        points3D = pointmap[:, mask]

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

        points3D = self.pts_jitter_t(points3D.float())
        points3D = points3D[:, valid_mask]

        sparse_pointmap = np.zeros((3, h, w))
        sparse_pointmap_mask = torch.zeros(1, h, w, dtype=torch.bool)
        sparse_pointmap[:, uv[1].numpy(), uv[0].numpy()] = points3D.numpy()
        sparse_pointmap = torch.from_numpy(sparse_pointmap).float()
        sparse_pointmap_mask[0, uv[1], uv[0]] = True

        sparse_pointmap, sparse_pointmap_mask = self.load_sparse(
            sparse_pointmap, sparse_pointmap_mask
        )

        sparse_pointmap, sparse_pointmap_mask = self.patch_crop(sparse_pointmap, sparse_pointmap_mask)

        return sparse_pointmap, sparse_pointmap_mask

