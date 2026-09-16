from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.utils import instantiate_from_config


class SparseAnchorInterpolation:
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
        patch_drop_prompt=None,
    ):
        self.sparse_ratio = sparse_ratio
        self.sparse_nums = sparse_nums
        self.seed = seed

        # Transform parameters
        self.blur_t = instantiate_from_config(blur_t) if blur_t else lambda x: x
        self.pts_jitter_t = instantiate_from_config(pts_jitter_t) if pts_jitter_t else lambda x: x
        self.uv_jitter_t = instantiate_from_config(uv_jitter_t) if uv_jitter_t else lambda x: x
        self.patch_crop_t = instantiate_from_config(patch_crop_t) if patch_crop_t else lambda x: x
        self.random_noise = instantiate_from_config(random_noise)
        self.patch_drop_prompt = (
            instantiate_from_config(patch_drop_prompt) if patch_drop_prompt else lambda x: x
        )

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
        if num_valid_points < 500:
            num_to_select = num_valid_points  # Select all points if there are fewer than 500
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

    def sparse_anchor_interpolation(
        self,
        gt_depth: torch.Tensor,  # (3, H, W)  XYZ (metres)
        rgb: np.ndarray,  # (H, W, 3)  uint8|float
        stride: int = 7,
        knn_k: int = 4,
        chunk: int = 8192,
        sigma_s: Optional[float] = None,  # spatial σ  (pixels)
        sigma_c: Optional[float] = None,  # colour  σ  (0-1 RGB)
        depth_thresh: float = 0.20,  # max |Z_p-Z_a| (metres)
    ) -> torch.Tensor:
        """
        Sparse-LiDAR simulation with joint bilateral interpolation +
        depth-consistency filtering (anchors > depth_thresh are ignored).
        """
        # ---------- sanity ----------
        assert gt_depth.ndim == 3 and gt_depth.size(0) == 3
        H, W = map(int, gt_depth.shape[1:])
        assert stride > 0 and knn_k > 0
        device, dtype = gt_depth.device, gt_depth.dtype
        N = H * W

        # ---------- RGB tensor ----------
        rgb_f32 = rgb.astype(np.float32)
        if rgb.dtype == np.uint8 or rgb_f32.max() > 1.01:
            rgb_f32 /= 255.0
        rgb_t = torch.from_numpy(rgb_f32).permute(2, 0, 1).unsqueeze(0)  # 1,3,H',W'
        if rgb_t.shape[-2:] != (H, W):
            rgb_t = F.interpolate(rgb_t, size=(H, W), mode="bilinear", align_corners=False)
        rgb_t = rgb_t.squeeze(0).permute(1, 2, 0).to(device, torch.float32)  # H,W,3
        rgb_flat = rgb_t.reshape(N, 3)  # N,3

        # ---------- anchor / non-anchor masks ----------
        anchor_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        anchor_mask[stride // 2 :: stride, stride // 2 :: stride] = True
        non_mask = ~anchor_mask
        anchor_idx = anchor_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
        non_idx = non_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
        N_a, N_n = anchor_idx.numel(), non_idx.numel()
        assert knn_k <= N_a, "knn_k > #anchors"

        # ---------- coords ----------
        ys = torch.div(torch.arange(N, device=device), W, rounding_mode="floor")
        xs = torch.arange(N, device=device) % W
        coords = torch.stack((ys, xs), dim=1).float()  # N,2
        anchor_coords = coords[anchor_idx]  # N_a,2

        # ---------- depth arrays ----------
        anchor_depth = gt_depth.view(3, N)[:, anchor_idx]  # 3,N_a
        anchor_z = anchor_depth[2]  # N_a
        z_flat = gt_depth[2].reshape(-1)  # N

        # ---------- σ defaults ----------
        sigma_s = float(stride) if sigma_s is None else float(sigma_s)
        sigma_c = 0.10 if sigma_c is None else float(sigma_c)
        inv2s2 = 0.5 / (sigma_s * sigma_s)
        inv2c2 = 0.5 / (sigma_c * sigma_c)

        # ---------- output buffer ----------
        out = torch.empty(3, N_n, device=device, dtype=dtype)

        # ---------- batched loop ----------
        for s in range(0, N_n, chunk):
            e = min(s + chunk, N_n)
            cur_idx = non_idx[s:e]  # c
            cur_coord = coords[cur_idx]  # c,2

            # spatial KNN
            dist_spatial = torch.cdist(cur_coord, anchor_coords)  # c,N_a
            dist_spatial, nn_idx = dist_spatial.topk(knn_k, largest=False)  # c,k

            # colour distances
            rgb_non = rgb_flat[cur_idx]  # c,3
            rgb_anchor = rgb_flat[anchor_idx][nn_idx]  # c,k,3
            dist_rgb = torch.norm(rgb_anchor - rgb_non.unsqueeze(1), dim=-1)  # c,k

            # joint bilateral weights
            w = torch.exp(-(dist_spatial**2) * inv2s2 - (dist_rgb**2) * inv2c2)  # c,k

            # ---------- depth-consistency filter ----------
            z_non = z_flat[cur_idx]  # c
            anchor_z_knn = anchor_z[nn_idx]  # c,k
            mask_depth = torch.abs(anchor_z_knn - z_non.unsqueeze(1)) < depth_thresh  # c,k
            w_filtered = w * mask_depth

            # 若全部权重被过滤，回退到未过滤版本
            w_sum = w_filtered.sum(dim=-1, keepdim=True)
            no_valid = w_sum.squeeze(-1) < 1e-12
            w_filtered = torch.where(no_valid.unsqueeze(-1), w, w_filtered)
            w_sum = w_filtered.sum(dim=-1, keepdim=True) + 1e-12
            w_filtered = w_filtered / w_sum  # 重新归一化

            # gather depth & blend
            depth_knn = anchor_depth.t()[nn_idx]  # c,k,3
            interp_xyz = (depth_knn * w_filtered.unsqueeze(-1)).sum(dim=1)  # c,3
            out[:, s:e] = interp_xyz.t().to(dtype)

        # ---------- assemble ----------
        dense = gt_depth.clone()
        dense.view(3, N)[:, non_idx] = out
        return dense

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

        mask = self.patch_crop_t(mask)
        pointmap = self.blur_t(pointmap)
        if self.random_noise is not None:
            dep = torch.clone(pointmap[2:3, :, :])
            pointmap = self.random_noise(pointmap, mask, edge_inputs=dep)

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

        # 使用anchor interpolation
        rgb = kwargs.get("rgb", None)
        if rgb is not None:
            sparse_pointmap = self.sparse_anchor_interpolation(sparse_pointmap, rgb)
        else:
            raise ValueError("rgb is not provided")

        sparse_pointmap = self.patch_drop_prompt(sparse_pointmap)
        sparse_pointmap_mask = self.patch_drop_prompt(sparse_pointmap_mask)

        return sparse_pointmap, sparse_pointmap_mask
