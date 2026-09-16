import random

import cv2
import numpy as np
import torch

from hAlgorithm.utils import instantiate_from_config


class RandomPattern:
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
        patch_crop_sparse=False,
    ):
        self.sparse_ratio = sparse_ratio
        self.sparse_nums = sparse_nums
        self.seed = seed
        self.min_valid_num = min_valid_num
        self.patch_crop_sparse = patch_crop_sparse

        # Transform parameters
        self.blur_t = instantiate_from_config(blur_t) if blur_t else lambda x: x
        self.pts_jitter_t = instantiate_from_config(pts_jitter_t) if pts_jitter_t else lambda x: x
        self.uv_jitter_t = instantiate_from_config(uv_jitter_t) if uv_jitter_t else lambda x: x
        self.patch_crop_t = instantiate_from_config(patch_crop_t) if patch_crop_t else lambda x: x
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

        if self.patch_crop_sparse:
            sparse_pointmap_mask = self.patch_crop_t(sparse_pointmap_mask)

        sparse_pointmap, sparse_pointmap_mask = self.load_sparse(
            sparse_pointmap, sparse_pointmap_mask
        )

        return sparse_pointmap, sparse_pointmap_mask


class SFMRandomPattern(RandomPattern):
    def __init__(
        self,
        is_lidar=False,
        sfm_prob=0.5,
        sfm_min_nums=0,
        sfm_pattern="sift",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.is_lidar = is_lidar
        self.sfm_prob = sfm_prob
        self.sfm_min_nums = sfm_min_nums
        self.sfm_pattern = sfm_pattern

    def load_sfm_points(self, rgb, valid_mask):
        # Use feature detection methods like SIFT or ORB
        assert rgb is not None
        height, width = rgb.shape[:2]
        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        if self.sfm_pattern == "sift":
            detector = cv2.SIFT.create()
        elif self.sfm_pattern == "orb":
            detector = cv2.ORB.create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
        else:
            raise NotImplementedError(f"Unsupported pattern: {self.sfm_pattern}")

        keypoints = detector.detect(gray)
        mask = torch.zeros([height, width], dtype=torch.bool)

        for keypoint in keypoints:
            x = round(keypoint.pt[1])
            y = round(keypoint.pt[0])
            mask[x, y] = 1
        mask = mask & valid_mask

        if mask.sum() < self.sfm_min_nums:
            return valid_mask
        else:
            return mask

    def get_sparse_depth(self, pointmap, mask, K, tgth=None, tgtw=None, rgb=None, **kwargs):
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

        if not self.is_lidar and random.random() <= self.sfm_prob:
            mask = self.load_sfm_points(rgb, mask)

        if not self.is_lidar:
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

        if not self.is_lidar:
            uv = self.uv_jitter_t(uv)

        valid_mask = (uv[0] >= 0) & (uv[0] < w) & (uv[1] >= 0) & (uv[1] < h)
        uv = uv[:, valid_mask]

        if not self.is_lidar:
            points3D = self.pts_jitter_t(points3D.float())

        points3D = points3D[:, valid_mask]

        sparse_pointmap = np.zeros((3, h, w))
        sparse_pointmap_mask = torch.zeros(1, h, w, dtype=torch.bool)
        sparse_pointmap[:, uv[1].numpy(), uv[0].numpy()] = points3D.numpy()
        sparse_pointmap = torch.from_numpy(sparse_pointmap).float()
        sparse_pointmap_mask[0, uv[1], uv[0]] = True

        if not self.is_lidar:
            sparse_pointmap, sparse_pointmap_mask = self.load_sparse(
                sparse_pointmap, sparse_pointmap_mask
            )

        return sparse_pointmap, sparse_pointmap_mask
