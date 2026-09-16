import cv2
import numpy as np
import torch

from hAlgorithm.datasets.transforms.edge_filter import edge_filter
from hAlgorithm.utils import instantiate_from_config


class MultiPatterns:
    depth2points = True

    def __init__(
        self,
        sparse_nums,
        pattern,
        rand_noise=0,
        rand_noise_mode="min_max",
        rand_noise_scale=0,
        edge_noise=0,
        sfm_max_dropout_rate=0.0,
        crop_patch_ratio=0,
        crop_patch_range=None,
        uv_jitter_ratio=0,
        uv_jitter_noise=0,
        uv_jitter_max=0,
        z_jitter_ratio=0,
        z_jitter_noise=0,
        z_jitter_scale=0,
        z_jitter_mode="scale",
        points_aug_mask="selected_mask",
        seed=0,
    ):
        """
        Initializes the MultiPatterns instance with various parameters controlling the generation of sparse depth maps.

        Parameters:
            sparse_nums (list): The target number of sparse samples.
            pattern (str): Specification string for the sampling pattern.
            rand_noise (float or list): Specifies the level of random noise to add.
            edge_noise (float or list): Specifies the level of noise to add around edges.
            sfm_max_dropout_rate (float): Maximum dropout rate for SfM points.
            crop_patch_ratio (float): Proportion of images to apply patch cropping on.
            crop_patch_range (tuple): Range for the size of patches to crop.
            seed (int): Seed for reproducible random number generation.
        """

        self.sparse_nums = sparse_nums
        self.pattern = pattern

        self.rand_noise = rand_noise
        self.rand_noise_mode = rand_noise_mode
        self.rand_noise_scale = rand_noise_scale

        self.edge_noise = edge_noise
        self.sfm_max_dropout_rate = sfm_max_dropout_rate

        self.crop_patch_ratio = crop_patch_ratio
        self.crop_patch_range = crop_patch_range

        self.uv_jitter_ratio = uv_jitter_ratio
        self.uv_jitter_noise = uv_jitter_noise
        self.uv_jitter_max = uv_jitter_max

        self.z_jitter_ratio = z_jitter_ratio
        self.z_jitter_noise = z_jitter_noise
        self.z_jitter_scale = z_jitter_scale
        self.z_jitter_mode = z_jitter_mode

        self.points_aug_mask = points_aug_mask
        self.seed = seed

        self.all_weights, self.all_patterns = self.parse_pattern(pattern=pattern)

    def parse_pattern(self, pattern):
        """
        Parse the pattern string into weights and patterns.

        Args:
            pattern (str): Pattern specification.

        Returns:
            tuple: Lists of weights and patterns.
        """
        all_weights = []
        all_patterns = []

        for pattern_item in pattern.split("+"):
            if "*" in pattern_item:
                weight, pattern = pattern_item.split("*")
                weight = float(weight)
            else:
                weight, pattern = 1.0, pattern_item

            all_weights.append(weight)
            all_patterns.append(pattern)

        # Normalize weights to sum to 1
        total_weight = sum(all_weights)
        all_weights = [w / total_weight for w in all_weights]

        return all_weights, all_patterns

    def get_sparse_mask_and_index(self, valid_idx, height, width):
        """
        Generates a sparse mask and corresponding indices based on the specified sampling pattern.

        This method determines the number of samples to take within a specified range,
        randomly selects indices from the provided valid indices, and creates a boolean mask
        indicating which positions in the image are sampled.

        Args:
            valid_idx (torch.Tensor): A tensor containing valid indices where sampling can occur.
            height (int): The height of the image.
            width (int): The width of the image.

        Returns:
            tuple: A tuple containing:
                - selected_mask (torch.Tensor): A boolean mask of shape (height * width) where True indicates sampled positions.
                - selected_idx (torch.Tensor): Indices of the selected samples from the valid_idx tensor.
        """
        if isinstance(self.sparse_nums, (list, tuple)):
            num_start, num_end = min(self.sparse_nums), max(self.sparse_nums)
            num_sample = int(np.random.randint(int(num_start), int(num_end)))
        else:
            num_sample = self.sparse_nums

        # Set a manual seed for reproducibility of random operations
        if self.seed is not None:
            torch.manual_seed(self.seed)

        idx_sample = torch.randperm(len(valid_idx))[:num_sample]
        selected_idx = valid_idx[idx_sample]

        selected_mask = torch.zeros((height * width)).bool()
        selected_mask[selected_idx] = 1

        return selected_mask, selected_idx

    def add_noise(self, depth, selected_idx, noise, noise_scale=None, noise_mode="min_max"):
        """
        Add noise to the depth map while sampling on valid_mask.

        Args:
            depth (torch.Tensor): Depth map tensor.
            selected_idx (torch.Tensor): indicating valid pixels.
            noise (str): Noise specification in the format 0.1 or [0.0, 0.1].

        Returns:
            torch.Tensor: Depth map with added noise.
        """

        if noise is not None:
            if isinstance(noise, (list, tuple)):
                # Generate a uniform random probability for noise
                noise_prob = np.random.uniform(min(noise), max(noise))
            else:
                noise_prob = float(noise)

            # Sample noise_mask only on valid_mask
            noise_mask = torch.zeros(depth.shape).bool()  # Initialize noise mask

            # Randomly sample from valid indices based on noise probability
            num_valid = len(selected_idx)  # Number of valid pixels
            sampled_idx = torch.from_numpy(
                np.random.choice(num_valid, size=int(num_valid * noise_prob), replace=False)
            )

            # Create noise_mask by setting sampled indices to 1
            noise_mask[selected_idx[sampled_idx]] = 1

            # Define the range of depth values for noise
            if noise_mode == "min_max":
                selected_depth = depth[selected_idx].numpy()
                depth_min, depth_max = np.percentile(selected_depth, 10), np.percentile(
                    selected_depth, 90
                )
                noise_values = torch.tensor(
                    np.random.uniform(depth_min, depth_max, size=noise_mask.shape)
                ).float()
            elif noise_mode == "scale":
                noise_scale = torch.tensor(
                    np.random.uniform(1 - noise_scale, 1 + noise_scale, size=noise_mask.shape)
                ).float()
                noise_values = depth * noise_scale
            else:
                raise ValueError("noise_mode is not exists!")

            # Apply noise to the depth map where noise_mask is True
            depth[noise_mask == 1] = noise_values[noise_mask == 1]

        return depth

    def uv_jitter(self, uv, selected_idx):
        jitter_ratio = np.random.uniform(min(self.uv_jitter_noise), max(self.uv_jitter_noise))
        jitter_mask = np.random.rand(selected_idx.shape[0]) < jitter_ratio
        jitter_u = np.random.uniform(
            -self.uv_jitter_max, self.uv_jitter_max, selected_idx.shape[0]
        ).astype(np.int32)
        jitter_v = np.random.uniform(
            -self.uv_jitter_max, self.uv_jitter_max, selected_idx.shape[0]
        ).astype(np.int32)
        uv[0, selected_idx[jitter_mask]] += jitter_u[jitter_mask]
        uv[1, selected_idx[jitter_mask]] += jitter_v[jitter_mask]
        return uv

    def get_pointmap(self, depth, intrinsics, valid_mask):
        """
        Generates a point map from a depth image using camera intrinsics.

        This method computes 3D points in space corresponding to each pixel in the depth image,
        applying optional UV and XY jitter to simulate noise in the sampling process.

        Args:
            depth (torch.Tensor): A tensor of shape (C, H, W) representing the depth image.
            intrinsics (torch.Tensor): A tensor of shape (3, 3) representing the camera intrinsics matrix.

        Returns:
            torch.Tensor: A tensor of shape (3, H, W) representing the 3D points in space.
        """

        C, H, W = depth.shape
        assert C == 1, "Depth image should have only one channel."

        # Generate meshgrid for pixel coordinates
        u, v = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy")
        uvz = torch.stack([u, v, torch.ones_like(u)], dim=0).reshape(3, -1).float()
        rays_d = intrinsics.inverse() @ uvz  # (3, HW)
        pointmap = depth.flatten(1) * rays_d
        pointmap_mask = None
        selected_idx = torch.nonzero(valid_mask.reshape(-1))[:, 0]

        if self.uv_jitter_ratio > 0 and np.random.rand() < self.uv_jitter_ratio:
            uv = self.uv_jitter(uvz[:2].long(), selected_idx)

            valid_mask = (uv[0] >= 0) & (uv[0] < W) & (uv[1] >= 0) & (uv[1] < H)
            uv = uv[:, valid_mask]
            pointmap = pointmap[:, valid_mask]

            new_pointmap = np.zeros([3, H, W])
            # NOTE: torch get error new_pointmap
            new_pointmap[:, uv[1].numpy(), uv[0].numpy()] = pointmap.numpy()
            new_pointmap = torch.from_numpy(new_pointmap).float()

            pointmap_mask = torch.zeros(1, H, W, dtype=torch.bool)
            pointmap_mask[0, uv[1], uv[0]] = True

            pointmap = new_pointmap.reshape([3, -1])

        if self.z_jitter_ratio > 0 and np.random.rand() < self.z_jitter_ratio:
            # Add noise to the depth map
            depth = self.add_noise(
                pointmap[2],
                selected_idx,
                self.z_jitter_noise,
                noise_scale=self.z_jitter_scale,
                noise_mode=self.z_jitter_mode,
            )
            pointmap[2, ...] = depth

        pointmap = pointmap.reshape([3, H, W])
        return pointmap, pointmap_mask

    def get_sparse_depth(self, depth, intrinsics, valid_mask, rgb=None, pattern=None, **kwargs):
        """
        Generate a sparse depth map based on the specified pattern.

        Args:
            depth (torch.Tensor): Depth map tensor of shape (1, h, w).
            intrinsics (torch.Tensor): Camera intrinsics matrix of shape (3, 3).
            valid_mask (torch.Tensor): Valid mask tensor of shape (1, h, w).
            rgb (torch.Tensor, optional): RGB image tensor of shape (3, h, w). Defaults to None.
            pattern (str, optional): Custom pattern specification. Defaults to None.

        Returns:
            tuple: Sparse point map and its mask.
        """

        # Parse the pattern if provided, otherwise use default patterns
        if pattern is not None:
            all_weights, all_patterns = self.parse_pattern(pattern=pattern)
        else:
            all_weights, all_patterns = self.all_weights, self.all_patterns

        # Randomly select a pattern based on weights
        selected_pattern = np.random.choice(all_patterns, p=all_weights)

        channel, height, width = depth.shape
        assert channel == 1, "Depth image should have only one channel."
        depth = depth.reshape(-1)
        valid_mask = valid_mask.reshape(-1).bool()
        assert depth.size() == valid_mask.size(), "Depth and valid mask sizes must match."

        # Get the indices of all valid points using the valid mask
        valid_idx = torch.nonzero(valid_mask)

        # Handle different sampling patterns
        if selected_pattern is None or selected_pattern.lower() in [
            "",
            "none",
            "nan",
        ]:
            # Default random sampling pattern
            selected_mask, selected_idx = self.get_sparse_mask_and_index(valid_idx, height, width)

            # Add noise to the depth map
            depth = self.add_noise(
                depth,
                selected_idx,
                self.rand_noise,
                noise_scale=self.rand_noise_scale,
                noise_mode=self.rand_noise_mode,
            )

        elif selected_pattern in ["sift", "orb"]:
            # Use feature detection methods like SIFT or ORB
            assert rgb is not None
            gray = cv2.cvtColor(
                ((rgb + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_RGB2GRAY
            )

            if selected_pattern == "sift":
                detector = cv2.SIFT.create()
            elif selected_pattern == "orb":
                detector = cv2.ORB.create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
            else:
                raise NotImplementedError(f"Unsupported pattern: {selected_pattern}")

            keypoints = detector.detect(gray)
            if len(keypoints) < 20:
                return self.get_sparse_depth(
                    pointmap=pointmap, valid_mask=valid_mask, rgb=rgb, pattern="1000"
                )

            selected_mask = torch.zeros([height, width]).bool()
            for keypoint in keypoints:
                x = round(keypoint.pt[1])
                y = round(keypoint.pt[0])
                selected_mask[x, y] = valid_mask.reshape(height, width)[x, y]
            selected_mask = selected_mask.reshape(-1)

            if selected_mask.sum() < 20:
                return self.get_sparse_depth(
                    pointmap=pointmap, valid_mask=valid_mask, rgb=rgb, pattern="1000"
                )

            # Optionally apply dropout to keypoints
            if self.sfm_max_dropout_rate > 0.0:
                keep_prob = 1.0 - np.random.uniform(0.0, self.sfm_max_dropout_rate)
                mask_keep = keep_prob * torch.ones_like(selected_mask)
                mask_keep = torch.bernoulli(mask_keep)
                selected_mask = selected_mask & mask_keep.bool()

            # Add noise to the depth map
            selected_idx = torch.nonzero(selected_mask)
            depth = self.add_noise(depth, selected_idx, self.rand_noise)

        elif "depth_edge" in selected_pattern:
            # Edge-based sampling pattern
            edge_mask = edge_filter(
                depth.reshape(height, width),
                valid_mask=valid_mask.reshape(height, width),
                times=0.05,
            ).reshape(-1)

            selected_mask, _ = self.get_sparse_mask_and_index(valid_idx, height, width)

            # Add noise to the depth map
            selected_edge_idx = torch.nonzero(selected_mask & edge_mask)
            depth = self.add_noise(depth, selected_edge_idx, self.edge_noise)

        else:
            raise NotImplementedError(f"Unsupported pattern: {selected_pattern}")

        # Replace NaN values with zero
        depth = torch.nan_to_num(depth).reshape([1, height, width])
        selected_mask = selected_mask.bool().reshape([1, height, width])

        if self.points_aug_mask != "valid_mask":
            pointmap, pointmap_mask = self.get_pointmap(depth, intrinsics, selected_mask)
        else:
            pointmap, pointmap_mask = self.get_pointmap(depth, intrinsics, valid_mask)

        if pointmap_mask is not None:
            selected_mask = selected_mask & pointmap_mask

        # Optionally apply cropping to the depth map
        if self.crop_patch_range is not None and np.random.rand() < self.crop_patch_ratio:
            patch_w = int(np.random.uniform(*self.crop_patch_range) * width)
            patch_h = int(np.random.uniform(*self.crop_patch_range) * height)

            x = np.random.randint(0, width - patch_w)
            y = np.random.randint(0, height - patch_h)

            selected_mask[0, y : y + patch_h, x : x + patch_w] = False

        sparse_pointmap = pointmap * selected_mask.type_as(pointmap)

        return sparse_pointmap, selected_mask
