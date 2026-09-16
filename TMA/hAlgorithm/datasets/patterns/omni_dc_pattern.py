import cv2
import numpy as np
import torch

from hAlgorithm.datasets.transforms.edge_filter import edge_filter


def add_noise(dep, valid_mask, noise):
    """
    Add noise to the depth map.

    Args:
        dep (torch.Tensor): Depth map tensor.
        valid_mask (torch.Tensor): Binary mask indicating valid pixels.
        noise (str): Noise specification in the format "0.1" or "0.0~0.1".

    Returns:
        torch.Tensor: Depth map with added noise.
    """
    if noise not in ["0", "0.0", "0.00"]:
        if "~" in noise:
            noise_prob_low, noise_prob_high = noise.split("~")
            noise_prob_low, noise_prob_high = float(noise_prob_low), float(noise_prob_high)
        else:
            noise_prob_low, noise_prob_high = float(noise), float(noise)

        # Generate a uniform random probability for noise
        noise_prob = np.random.uniform(noise_prob_low, noise_prob_high)
        noise_mask = torch.tensor(np.random.binomial(n=1, p=noise_prob, size=dep.shape))
        noise_mask = noise_mask * valid_mask

        # Define the range of depth values for noise
        depth_min, depth_max = np.percentile(dep.numpy(), 10), np.percentile(dep.numpy(), 90)
        noise_values = torch.tensor(np.random.uniform(depth_min, depth_max, size=dep.shape)).float()

        # Apply noise to the depth map where mask is True
        dep[noise_mask == 1] = noise_values[noise_mask == 1]

    return dep


def add_noise_v2(dep, valid_mask, noise):
    """
    Add noise to the depth map while sampling on valid_mask.

    Args:
        dep (torch.Tensor): Depth map tensor.
        valid_mask (torch.Tensor): Binary mask indicating valid pixels.
        noise (str): Noise specification in the format "0.1" or "0.0~0.1".

    Returns:
        torch.Tensor: Depth map with added noise.
    """
    if noise not in ["0", "0.0", "0.00"]:
        if "~" in noise:
            noise_prob_low, noise_prob_high = noise.split("~")
            noise_prob_low, noise_prob_high = float(noise_prob_low), float(noise_prob_high)
        else:
            noise_prob_low, noise_prob_high = float(noise), float(noise)

        # Generate a uniform random probability for noise
        noise_prob = np.random.uniform(noise_prob_low, noise_prob_high)

        # Sample noise_mask only on valid_mask
        noise_mask = torch.zeros_like(valid_mask)  # Initialize noise mask
        valid_indices = torch.nonzero(
            valid_mask, as_tuple=True
        )  # Get indices where valid_mask is True

        # Randomly sample from valid indices based on noise probability
        num_valid = len(valid_indices[0])  # Number of valid pixels
        sampled_indices = np.random.choice(
            num_valid, size=int(num_valid * noise_prob), replace=False
        )

        # Create noise_mask by setting sampled indices to 1
        noise_mask[valid_indices[0][sampled_indices], valid_indices[1][sampled_indices]] = 1

        # Define the range of depth values for noise
        depth_min, depth_max = np.percentile(dep[valid_mask].numpy(), 10), np.percentile(
            dep[valid_mask].numpy(), 90
        )
        noise_values = torch.tensor(
            np.random.uniform(depth_min, depth_max, size=noise_mask.shape)
        ).float()

        # Apply noise to the depth map where noise_mask is True
        dep[noise_mask == 1] = noise_values[noise_mask == 1]

    return dep


class OMNIDCPattern:
    """
    A class for generating sparse depth maps using different sampling patterns.

    Args:
        pattern (str): Pattern specification in the format "0.8*100~2000+0.2*sift".
        rand_noise (str): Noise specification in the format "0.1" or "0.0~0.1".
        seed (int): Random seed for reproducibility.
    """

    depth2points = False

    def __init__(
        self, pattern, rand_noise="0.0", edge_noise="0.0", train_sfm_max_dropout_rate=0.0, seed=0
    ):
        self.pattern = pattern
        self.rand_noise = rand_noise
        self.edge_noise = edge_noise
        self.seed = seed
        self.train_sfm_max_dropout_rate = train_sfm_max_dropout_rate

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

    def get_sparse_depth(self, pointmap, valid_mask, rgb=None, pattern=None, **kwargs):
        """
        Generate a sparse depth map based on the specified pattern.

        Args:
            pointmap (torch.Tensor): Point map tensor of shape (3, h, w).
            valid_mask (torch.Tensor): Valid mask tensor of shape (h, w).
            rgb (torch.Tensor, optional): RGB image tensor of shape (3, h, w). Defaults to None.
            pattern (str, optional): Custom pattern specification. Defaults to None.

        Returns:
            tuple: Sparse point map and its mask.
        """
        if pattern is not None:
            all_weights, all_patterns = self.parse_pattern(pattern=pattern)
        else:
            all_weights, all_patterns = self.all_weights, self.all_patterns

        # Randomly select a pattern based on weights
        selected_pattern = np.random.choice(all_patterns, p=all_weights)

        dep = torch.clone(pointmap[2:3, :, :])  # Extract depth channel
        channel, height, width = dep.shape
        assert channel == 1

        # Get the indices of all valid points using the valid mask
        idx_nnz = torch.nonzero(valid_mask.view(-1), as_tuple=True)[0]

        # Count the number of valid points
        num_idx = len(idx_nnz)

        if "~" in selected_pattern:
            # If the pattern specifies a range, randomly sample within the range
            num_start, num_end = selected_pattern.split("~")[:2]
            if num_start.isdigit():
                num_start = int(num_start)
                num_end = int(num_end)
                selected_pattern = str(np.random.randint(num_start, num_end))

        if selected_pattern.isdigit():
            # Sample a fixed number of points
            num_sample = int(selected_pattern)

            # Set a manual seed for reproducibility of random operations
            if self.seed is not None:
                torch.manual_seed(self.seed)
            idx_sample = torch.randperm(num_idx)[:num_sample]

            idx_nnz = idx_nnz[idx_sample[:]]

            # Create a mask for sampled points
            mask = torch.zeros((channel * height * width)).bool()
            mask[idx_nnz] = 1
            mask = mask.view((channel, height, width))

            # Add noise to the depth map
            dep = add_noise(dep, valid_mask, self.rand_noise)
            dep_sp = dep * mask.type_as(dep)

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
            mask = torch.zeros([1, height, width])

            if len(keypoints) < 20:
                # Fallback to a default pattern if too few keypoints are detected
                return self.get_sparse_depth(
                    pointmap=pointmap, valid_mask=valid_mask, rgb=rgb, pattern="1000"
                )

            for keypoint in keypoints:
                x = round(keypoint.pt[1])
                y = round(keypoint.pt[0])
                mask[0, x, y] = 1.0 * valid_mask[0, x, y]

            if mask.sum() < 20:
                # Fallback to a default pattern if too few keypoints are detected
                return self.get_sparse_depth(
                    pointmap=pointmap, valid_mask=valid_mask, rgb=rgb, pattern="1000"
                )

            # Optionally apply dropout to keypoints
            if self.train_sfm_max_dropout_rate > 0.0:
                keep_prob = 1.0 - np.random.uniform(0.0, self.train_sfm_max_dropout_rate)
                mask_keep = keep_prob * torch.ones_like(mask)
                mask_keep = torch.bernoulli(mask_keep)
                mask = mask * mask_keep

            # Add noise to the depth map
            dep = add_noise(dep, valid_mask, self.rand_noise)
            dep_sp = dep * mask.type_as(dep)

        elif "rgb_edge" in selected_pattern or "depth_edge" in selected_pattern:
            # Use feature detection methods like SIFT or ORB
            if "rgb_edge" in selected_pattern:
                assert rgb is not None
                edge_mask = edge_filter(rgb, valid_mask=valid_mask.bool(), times=0.05)
                num_start, num_end = selected_pattern.split("rgb_edge_")[1].split("~")
            elif "rgb_depth_edge" in selected_pattern:
                assert rgb is not None
                rgb_edge_mask = edge_filter(rgb, valid_mask=valid_mask.bool(), times=0.05)
                dep_edge_mask = edge_filter(dep, valid_mask=valid_mask.bool(), times=0.05)
                edge_mask = rgb_edge_mask * dep_edge_mask
                num_start, num_end = selected_pattern.split("rgb_depth_edge_")[1].split("~")
            elif "depth_edge" in selected_pattern:
                edge_mask = edge_filter(dep, valid_mask=valid_mask.bool(), times=0.05)
                num_start, num_end = selected_pattern.split("depth_edge_")[1].split("~")
            else:
                raise NotImplementedError(f"Unsupported pattern: {selected_pattern}")

            # Sample a fixed number of points
            num_start = int(num_start)
            num_end = int(num_end)
            num_sample = int(np.random.randint(num_start, num_end))

            # Set a manual seed for reproducibility of random operations
            idx_sample = torch.randperm(num_idx)[:num_sample]

            idx_nnz = idx_nnz[idx_sample[:]]

            # Create a mask for sampled points
            mask = torch.zeros((channel * height * width)).bool()
            mask[idx_nnz] = 1
            mask = mask.view((channel, height, width))

            # Add noise to the depth map
            dep = add_noise_v2(dep, edge_mask & mask, self.rand_noise)
            dep_sp = dep * mask.type_as(dep)

        else:
            raise NotImplementedError(f"Unsupported pattern: {selected_pattern}")

        # Replace NaN values with zero
        dep_sp = torch.nan_to_num(dep_sp)  # 1,h,w
        mask = mask.bool()  # 1,h,w

        # Combine sparse depth with other channels
        sparse_pointmap = torch.cat([pointmap[0:2], dep_sp], dim=0)
        sparse_pointmap[:, ~mask[0]] = 0

        return sparse_pointmap, mask
