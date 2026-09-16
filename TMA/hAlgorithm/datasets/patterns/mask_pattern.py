import random
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from hAlgorithm.utils import instantiate_from_config
import json


class RandomMaskPattern:
    def __init__(
        self,
        mask_json,
        mask_ratio=None,
        sparse_ratio=None,
        sparse_nums=None,
        min_valid_num=500,
        resolution=None,
        seed=0,
        blur_t=None,
        patch_crop_t=None,
        pts_jitter_t=None,
        uv_jitter_t=None,
        random_noise=None,
        project_noise_t=None,
        resize_compression=None,
        patch_crop_sparse=False,
    ):
        self.mask_json = mask_json
        self.mask_ratio = mask_ratio
        

        with open(self.mask_json, "r") as f:
            self.mask_paths = json.load(f)

        self.resolution = resolution
        self.seed = seed
        self.patch_crop_sparse = patch_crop_sparse

        self.sparse_ratio = sparse_ratio
        self.sparse_nums = sparse_nums
        self.min_valid_num = min_valid_num

        # Transform parameters
        self.blur_t = instantiate_from_config(blur_t) if blur_t else lambda x: x
        self.pts_jitter_t = instantiate_from_config(pts_jitter_t) if pts_jitter_t else lambda x: x
        self.uv_jitter_t = instantiate_from_config(uv_jitter_t) if uv_jitter_t else lambda x: x
        self.patch_crop_t = instantiate_from_config(patch_crop_t) if patch_crop_t else lambda x: x
        self.random_noise = instantiate_from_config(random_noise)
        self.resize_compression = instantiate_from_config(resize_compression)
        self.project_noise_t = instantiate_from_config(project_noise_t)

    def load_sparse_with_mask(self, depth, valid_mask, mask_path):
        """
        Generates a sparse depth map by loading a binary mask from a .npy file.
        The mask is automatically resized to match the spatial size of depth using nearest interpolation if needed.

        Args:
            depth (torch.Tensor): Original depth map, shape (C, H, W).
            valid_mask (torch.Tensor): Boolean mask of valid points, shape (H, W) or (1, H, W).
            mask_path (str): Path to a .npy file containing a binary mask.

        Returns:
            tuple: (sparse_depth, selected_mask) where selected_mask is (1, H, W).
        """
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # Load mask from .npy
        mask_array = np.load(mask_path)

        # Handle common shapes: squeeze to (H, W)
        if mask_array.ndim == 3:
            if mask_array.shape[0] == 1:
                mask_array = mask_array.squeeze(0)  # (1, H, W) -> (H, W)
            elif mask_array.shape[2] == 1:
                mask_array = mask_array.squeeze(2)  # (H, W, 1) -> (H, W)
            else:
                raise ValueError(f"Unsupported 3D mask shape: {mask_array.shape}. Expected (1, H, W) or (H, W, 1).")
        elif mask_array.ndim == 2:
            pass  # OK
        else:
            raise ValueError(f"Mask must be 2D or 3D with singleton channel. Got shape: {mask_array.shape}")

        # Target spatial size from depth
        _, H, W = depth.shape
        mask_h, mask_w = mask_array.shape

        # Convert to tensor for resizing
        mask_tensor = torch.from_numpy(mask_array).float()  # (Hm, Wm)

        # Resize if needed using nearest interpolation
        if (mask_h, mask_w) != (H, W):
            # Add batch and channel dims: (1, 1, Hm, Wm)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            # Resize to (1, 1, H, W)
            mask_tensor = F.interpolate(mask_tensor, size=(H, W), mode="nearest")
            # Remove extra dims: (H, W)
            mask_tensor = mask_tensor.squeeze(0).squeeze(0)

        # Convert to boolean on the same device as valid_mask
        loaded_mask = (mask_tensor > 0).to(valid_mask.device).bool()

        # Normalize valid_mask to (H, W)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.squeeze(0)
        valid_mask = valid_mask.bool()

        # Final selection: valid AND in mask
        selected_mask_2d = valid_mask & loaded_mask  # (H, W)

        # Apply to depth
        sparse_depth = depth.clone()
        selected_mask_full = selected_mask_2d.unsqueeze(0)  # (1, H, W)
        if selected_mask_full.shape[0] != depth.shape[0]:
            selected_mask_full = selected_mask_full.repeat(depth.shape[0], 1, 1)  # (C, H, W)
        sparse_depth[~selected_mask_full] = 0
        return sparse_depth, selected_mask_2d.unsqueeze(0)

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

    def get_pattern_mask_path(self, tgt_resolution, data_seed):
        seed = data_seed
        if self.seed is not None:
            seed = data_seed + self.seed
        torch.manual_seed(seed)
        random.seed(seed)
        if self.resolution is not None:
            pattern_path = random.choice(self.mask_paths[str(self.resolution)])
        else:
            resolutions = list(self.mask_paths.keys())
            pattern_res = resolutions[0]
            for res in resolutions:
                if abs(int(res) - tgt_resolution) < abs(int(pattern_res) - tgt_resolution):
                    pattern_res = res
            pattern_path = random.choice(self.mask_paths[pattern_res])
        return pattern_path

    def get_sparse_depth(self, pointmap, mask, K, tgth=None, tgtw=None, data_idx=None, **kwargs):
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

        if not self.patch_crop_sparse:
            mask = self.patch_crop_t(mask)

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

        if self.mask_ratio is None or random.random() < self.mask_ratio:
            target_resolution = max(h, w)
            mask_path = self.get_pattern_mask_path(target_resolution, data_seed=data_idx)
            sparse_pointmap, sparse_pointmap_mask = self.load_sparse_with_mask(sparse_pointmap, sparse_pointmap_mask, mask_path)
        else:
            if self.patch_crop_sparse:
                sparse_pointmap_mask = self.patch_crop_t(sparse_pointmap_mask)
            sparse_pointmap, sparse_pointmap_mask = self.load_sparse(sparse_pointmap, sparse_pointmap_mask)

        return sparse_pointmap, sparse_pointmap_mask
