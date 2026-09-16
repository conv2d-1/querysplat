import numpy as np
import torch

from hAlgorithm.datasets.transforms.edge_filter import edge_filter


class PatchCropMask:
    def __init__(
        self,
        patch_range=(0.05, 0.25),
        p=0.5,
        grid_p=0.0,
        grid_crop_config=None,
        reverse_p=0.0,
    ):
        self.patch_range = patch_range
        self.p = p
        self.grid_p = grid_p
        self.grid_crop = (
            GridCropMask(p=grid_p, **grid_crop_config)
            if self.grid_p > 0 and grid_crop_config is not None
            else None
        )
        self.reverse_p = reverse_p

    def __call__(self, mask):
        """
        Randomly remove a patch from the mask
        mask: 1, h, w
        """

        if self.grid_crop is not None:
            mask = self.grid_crop(mask)

        if np.random.rand() > self.p:
            return mask

        if mask.ndim == 3:
            _, h, w = mask.shape
        else:
            h, w = mask.shape

        patch_w = int(np.random.uniform(*self.patch_range) * w)
        patch_h = int(np.random.uniform(*self.patch_range) * h)

        x = np.random.randint(0, w - patch_w)
        y = np.random.randint(0, h - patch_h)

        if mask.ndim == 3:
            mask[:, y : y + patch_h, x : x + patch_w] = False
        else:
            mask[y : y + patch_h, x : x + patch_w] = False
        
        if np.random.rand() < self.reverse_p:
            mask = ~mask

        return mask


class GridCropMask:
    def __init__(self, grid_range=(14, 34), grid_wid=(2, 5), grid_ratio=(0.75, 1.0), p=0.5):
        self.grid_range = list(grid_range)
        self.p = p
        self.grid_wid = list(grid_wid)
        self.grid_ratio = list(grid_ratio)

        self.grid_range[1] += 1
        self.grid_wid[1] += 1

    def __call__(self, mask):
        """
        Randomly remove a grid from the mask
        mask: 1, h, w
        """

        if mask.ndim == 3:
            _, h, w = mask.shape
        else:
            h, w = mask.shape

        if np.random.rand() < self.p:
            grid_wid = np.random.randint(*self.grid_wid)
            grid_range = np.random.randint(*self.grid_range)
            grid_num_h = int(h * np.random.uniform(*self.grid_ratio) / (grid_wid + grid_range))
            start_h = int(np.random.uniform(0, h - grid_num_h * (grid_wid + grid_range)))

            for i in range(grid_num_h):
                mask[
                    ...,
                    start_h
                    + i * (grid_wid + grid_range) : start_h
                    + i * (grid_wid + grid_range)
                    + grid_wid,
                    :,
                ] = False

        if np.random.rand() < self.p:
            grid_wid = np.random.randint(*self.grid_wid)
            grid_range = np.random.randint(*self.grid_range)
            grid_num_w = int(w * np.random.uniform(*self.grid_ratio) / (grid_wid + grid_range))
            start_w = int(np.random.uniform(0, w - grid_num_w * (grid_wid + grid_range)))

            for i in range(grid_num_w):
                mask[
                    ...,
                    :,
                    start_w
                    + i * (grid_wid + grid_range) : start_w
                    + i * (grid_wid + grid_range)
                    + grid_wid,
                ] = False

        return mask


class RandomUVJitter:
    def __init__(self, max_jitter=3, jitter_ratio=(0.05, 0.5), p=0.5):
        self.max_jitter = max_jitter
        self.jitter_ratio = jitter_ratio
        self.p = p

    def __call__(self, uv):
        """
        uv: 2, n
        """
        if np.random.rand() > self.p:
            return uv

        jitter_ratio = np.random.uniform(*self.jitter_ratio)
        jitter_mask = np.random.rand(uv.shape[1]) < jitter_ratio
        jitter_u = np.random.uniform(-self.max_jitter, self.max_jitter, uv.shape[1]).astype(
            np.int32
        )
        jitter_v = np.random.uniform(-self.max_jitter, self.max_jitter, uv.shape[1]).astype(
            np.int32
        )
        uv[0, jitter_mask] += jitter_u[jitter_mask]
        uv[1, jitter_mask] += jitter_v[jitter_mask]
        return uv

class RandomUVShift:
    def __init__(self, max_jitter=3, p=0.5):
        self.max_jitter = max_jitter
        self.p = p

    def __call__(self, uv):
        """
        uv: 2, n
        """
        if np.random.rand() > self.p:
            return uv
        
        jitter_u = np.random.randint(-self.max_jitter, self.max_jitter+1, size=1).astype(
            np.int32
        )
        
        jitter_v = np.random.randint(-self.max_jitter, self.max_jitter+1, size=1).astype(
            np.int32
        )
        uv[0, :] += jitter_u
        uv[1, :] += jitter_v
        return uv


class Random3DJitter:
    def __init__(self, noise_std=0.05, p=0.5, seed=None):
        self.noise_std = noise_std
        self.p = p
        self.seed = seed

    def __call__(self, pts):
        if self.seed is not None:
            np.random.seed(self.seed)

        if np.random.rand() > self.p:
            return pts
        z_weight = pts[:, 2]
        noise = np.random.normal(0, self.noise_std, pts.shape)
        pts = pts + torch.from_numpy(noise).float() * z_weight[:, None]
        return pts


class FTRandom3DJitter:
    def __init__(self, normal_noise_std=0.01, proximity_noise_std=0.02, p=0.5, range_thresh=2.0):

        self.normal_noise_std = normal_noise_std
        self.proximity_noise_std = proximity_noise_std
        self.range_thresh = range_thresh
        self.p = p

    def __call__(self, pts):
        if np.random.rand() > self.p:
            return pts

        z_weight = pts[:, 2:3]
        range_mask = (z_weight <= self.range_thresh).float()

        normal_noise = torch.from_numpy(
            np.random.normal(0, self.normal_noise_std, pts.shape)
        ).float()
        proximity_noise = torch.from_numpy(
            np.random.normal(0, self.proximity_noise_std, pts.shape)
        ).float()
        pts = pts + normal_noise * z_weight * (1 - range_mask) + proximity_noise * range_mask

        return pts


class RangeRandom3DJitter:
    def __init__(
        self,
        range_list,
        noise_mean_list,
        noise_std_list,
        p=0.5,
        seed=None,
    ):
        self.range_list = range_list
        self.noise_mean_list = (
            noise_mean_list if isinstance(noise_mean_list, list) else [noise_mean_list]
        )
        self.noise_std_list = (
            noise_std_list if isinstance(noise_std_list, list) else [noise_std_list]
        )
        self.min_range_list = [0] + self.range_list[:-1]

        while len(self.noise_std_list) < len(self.range_list):
            self.noise_std_list.append(self.noise_std_list[-1])

        while len(self.noise_mean_list) < len(self.range_list):
            self.noise_mean_list.append(self.noise_mean_list[-1])

        assert len(self.range_list) == len(self.noise_mean_list) == len(self.noise_std_list)

        self.p = p
        self.seed = seed

    def __call__(self, pts):
        if self.seed is not None:
            np.random.seed(self.seed)

        if np.random.rand() > self.p:
            return pts

        z_weight = pts[:, 2:3]
        for min_range, max_range, noise_mean, noise_std in zip(
            self.min_range_list, self.range_list, self.noise_mean_list, self.noise_std_list
        ):

            range_mask = ((z_weight <= max_range) & (z_weight >= min_range)).float()
            normal_noise = torch.from_numpy(
                np.random.normal(noise_mean, noise_std, pts.shape)
            ).float()
            pts = pts + normal_noise * range_mask

        return pts

class MixRangeRandom3DJitter:
    def __init__(
        self,
        configs,
        p=0.5,
        seed=None,
    ):
        self.jitter_list = []
        for cfg in configs:
            self.jitter_list.append(RangeRandom3DJitter(p=1.0, seed=seed, **cfg))
        self.p = p
        self.seed = seed
    
    def __call__(self, pts):
        if self.seed is not None:
            np.random.seed(self.seed)

        if np.random.rand() > self.p:
            return pts
    
        jitter = np.random.choice(self.jitter_list)
        pts = jitter(pts)
        
        return pts


class BlurPointmap:
    def __init__(self, scale_range=(0.3, 0.8), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, pointmap):
        """
        pointmap: 3, h, w
        """
        if np.random.rand() > self.p:
            return pointmap

        scale = np.random.uniform(*self.scale_range)
        _, h, w = pointmap.shape
        pointmap_l = torch.nn.functional.interpolate(
            pointmap.unsqueeze(0), scale_factor=scale, mode="bilinear"
        )
        pointmap = torch.nn.functional.interpolate(pointmap_l, (h, w), mode="bilinear")

        return pointmap[0]


class RandomNoise:
    def __init__(
        self,
        rand_global_noise,
        rand_edge_noise,
        global_noise_type="minmax",
        scale_range=[0.9 - 1.1],
        range_noise_config=None,
    ):
        """
        rand_noise (str): Noise specification in the format "0.1" or "0.0~0.1".
        """
        self.rand_global_noise = rand_global_noise
        self.rand_edge_noise = rand_edge_noise
        assert global_noise_type in ["scale", "minmax"]
        self.global_noise_type = global_noise_type
        self.scale_range = scale_range
        self.range_noise = (
            RangeRandomNoise(**range_noise_config) if range_noise_config is not None else None
        )

    def __call__(self, pointmap, mask: torch.Tensor, edge_inputs=None):
        if edge_inputs is not None:
            if isinstance(edge_inputs, (list, tuple)):
                edge_mask = [
                    edge_filter(edge_in.squeeze(), valid_mask=mask.bool(), times=0.05)
                    for edge_in in edge_inputs
                ]
                edge_mask = torch.all(torch.stack(edge_mask), dim=0)
            else:
                edge_mask = edge_filter(edge_inputs.squeeze(), valid_mask=mask.bool(), times=0.05)
            mask_input = edge_mask & mask
        else:
            mask_input = mask

        dep = torch.clone(pointmap[2:3, :, :])
        if self.range_noise is not None:
            dep = self.range_noise(dep, mask)

        if mask_input.sum() > 0:
            dep = self.add_noise(dep, mask_input, self.rand_edge_noise, noise_type="minmax")
        if mask.sum() > 0:
            dep = self.add_noise(
                dep, mask, self.rand_global_noise, noise_type=self.global_noise_type
            )
        dep_sp = dep * mask.type_as(dep)
        dep_sp = torch.nan_to_num(dep_sp)  # 1,h,w

        # Combine sparse depth with other channels
        sparse_pointmap = torch.cat([pointmap[0:2], dep_sp], dim=0)
        sparse_pointmap[:, ~mask] = 0

        return sparse_pointmap

    def add_noise(self, dep, valid_mask, noise, noise_type):
        """
        Add noise to the depth map while sampling on valid_mask.

        Args:
            dep (torch.Tensor): Depth map tensor.  [1, H, W]
            valid_mask (torch.Tensor): Binary mask indicating valid pixels.  [1, H, W] or [H, W]
            noise (str): Noise specification in the format "0.1" or "0.0~0.1".

        Returns:
            torch.Tensor: Depth map with added noise.
        """
        if len(valid_mask.shape) == 2:
            valid_mask = valid_mask.unsqueeze(0)

        # Parse noise probability
        if "~" in noise:
            noise_prob_low, noise_prob_high = map(float, noise.split("~"))
        else:
            noise_prob = float(noise)
            noise_prob_low = noise_prob_high = noise_prob

        # Early exit if no noise needed
        if noise_prob_high <= 0:
            return dep

        # Generate a uniform random probability for noise
        noise_prob = np.random.uniform(noise_prob_low, noise_prob_high)

        # Sample noise_mask only on valid_mask
        noise_mask = torch.zeros_like(valid_mask)  # Initialize noise mask
        valid_indices = torch.nonzero(
            valid_mask.squeeze(), as_tuple=True
        )  # Get indices where valid_mask is True

        # Randomly sample from valid indices based on noise probability
        num_valid = len(valid_indices[0])  # Number of valid pixels
        sampled_indices = np.random.choice(
            num_valid, size=int(num_valid * noise_prob), replace=False
        )

        # Create noise_mask by setting sampled indices to 1
        noise_mask[..., valid_indices[0][sampled_indices], valid_indices[1][sampled_indices]] = 1

        if noise_type in ["minmax"]:
            noise_values = self.get_noise_minmax(dep, valid_mask, noise_mask)
        elif noise_type in ["scale"]:
            noise_values = self.get_noise_scale(dep, valid_mask, noise_mask)
        else:
            raise NotImplementedError

        # Apply noise to the depth map where noise_mask is True
        dep[noise_mask == 1] = noise_values[noise_mask == 1]

        return dep

    def get_noise_minmax(self, dep, valid_mask, noise_mask):
        # Define the range of depth values for noise
        depth_min, depth_max = np.percentile(dep[valid_mask].numpy(), 10), np.percentile(
            dep[valid_mask].numpy(), 90
        )
        noise_values = torch.tensor(
            np.random.uniform(depth_min, depth_max, size=noise_mask.shape)
        ).float()
        return noise_values

    def get_noise_scale(self, dep, valid_mask, noise_mask):
        min_scale = self.scale_range[0]
        max_scale = self.scale_range[1]
        noise_scales = torch.tensor(
            np.random.uniform(min_scale, max_scale, size=noise_mask.shape)
        ).float()
        noise_values = dep * noise_scales
        return noise_values


class RangeRandomNoise:
    def __init__(self, range_list, noise_mean_list, noise_std_list):
        self.range_list = range_list
        self.noise_mean_list = (
            noise_mean_list if isinstance(noise_mean_list, list) else [noise_mean_list]
        )
        self.noise_std_list = (
            noise_std_list if isinstance(noise_std_list, list) else [noise_std_list]
        )
        self.min_range_list = [0] + self.range_list[:-1]

        assert len(self.range_list) == len(self.noise_mean_list) == len(self.noise_std_list)

    def __call__(self, depth, mask: torch.Tensor):
        """
        depthmap : Input depthmap, 1, h, w
        """
        depthmap = torch.clone(depth)
        for min_range, max_range, noise_mean, noise_std in zip(
            self.min_range_list, self.range_list, self.noise_mean_list, self.noise_std_list
        ):
            range_mask = (depthmap >= min_range) & (depthmap < max_range) & mask
            noise = torch.randn_like(depthmap) * noise_std + noise_mean
            depthmap[range_mask] = depthmap[range_mask] + noise[range_mask]

        return depthmap


class PatchDropPrompt:
    def __init__(self, patch_range=(0.05, 0.25), p=0.5):
        self.patch_range = patch_range
        self.p = p

    def __call__(self, prompt):
        """
        Randomly remove a patch from the mask
        prompt: 3, h, w
        """

        if np.random.rand() > self.p:
            return prompt

        if prompt.ndim == 3:
            _, h, w = prompt.shape
        else:
            h, w = prompt.shape

        patch_w = int(np.random.uniform(*self.patch_range) * w)
        patch_h = int(np.random.uniform(*self.patch_range) * h)

        x = np.random.randint(0, w - patch_w)
        y = np.random.randint(0, h - patch_h)

        if prompt.ndim == 3:
            prompt[:, y : y + patch_h, x : x + patch_w] = 0
        else:
            prompt[y : y + patch_h, x : x + patch_w] = 0

        return prompt

class RandomProjectNoise:
    def __init__(self, noise_mean, noise_std, rot_noise_std, prob=0.2):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.rot_noise_std = rot_noise_std
        self.prob = prob
        
    
    def __call__(self, pointmap):
        
        if np.random.rand() > self.prob:
            return pointmap
        
        transform = self.generate_random_transform(self.noise_mean, self.noise_std, self.rot_noise_std)
        pointmap = self.apply_transform_to_point_cloud(pointmap, transform)
        return pointmap

    def apply_transform_to_point_cloud(self, point_cloud, transform):
        """
        Apply a 4x4 homogeneous transformation to a point cloud of shape (3, H, W).
        
        Args:
            point_cloud (Tensor): Shape (3, H, W)
            transform (Tensor): Shape (4, 4)

        Returns:
            transformed_pc (Tensor): Transformed point cloud of shape (3, H, W)
        """
        H, W = point_cloud.shape[1], point_cloud.shape[2]
        B = 1  # batch size is 1

        # Reshape point cloud to (B, 3, H*W)
        pc = point_cloud.view(B, 3, -1)

        # Add homogeneous coordinate (w=1)
        ones = torch.ones((B, 1, H * W), device=pc.device)
        homogeneous_pc = torch.cat([pc, ones], dim=1)  # shape (B, 4, H*W)

        # Apply transformation
        transformed = torch.bmm(transform.unsqueeze(0), homogeneous_pc)  # shape (B, 4, H*W)

        # Remove homogeneous coordinate and reshape back
        transformed_xyz = transformed[:, :3, :]  # shape (B, 3, H*W)
        transformed_xyz = transformed_xyz.view(B, 3, H, W).squeeze(0)

        return transformed_xyz
    
    def generate_random_transform(self, noise_mean, noise_std, rot_noise_std):
        """
        Generate a random transformation matrix with:
        - Translation in X, Y, Z with Gaussian noise.
        - Rotation around X, Y, Z axes with Gaussian noise.
        
        Args:
            noise_mean (float): Mean of translation noise.
            noise_std (float): Std of translation noise.
            rot_noise_std (float): Std of rotation angles in radians.

        Returns:
            transform_matrix: 4x4 homogeneous transformation matrix.
        """
        # Generate translation noise using torch.randn for better compatibility
        tx = torch.randn(1) * noise_std + noise_mean
        ty = torch.randn(1) * noise_std + noise_mean
        tz = torch.randn(1) * noise_std + noise_mean

        # Generate rotation angles with Gaussian noise
        rx = torch.randn(1) * rot_noise_std
        ry = torch.randn(1) * rot_noise_std
        rz = torch.randn(1) * rot_noise_std

        # Build rotation matrices for each axis
        def rotate(axis, angle):
            c = torch.cos(angle)
            s = torch.sin(angle)
            if axis == 'x':
                return torch.tensor([
                    [1, 0, 0],
                    [0, c, -s],
                    [0, s, c]
                ])
            elif axis == 'y':
                return torch.tensor([
                    [c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c]
                ])
            elif axis == 'z':
                return torch.tensor([
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]
                ])

        R_x = rotate('x', rx)
        R_y = rotate('y', ry)
        R_z = rotate('z', rz)

        # Combine rotations
        R = R_z @ R_y @ R_x

        # Build homogeneous transformation matrix
        transform = torch.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = torch.tensor([tx, ty, tz])

        return transform

class ResizeCompression:
    def __init__(self, scale_factor, p):
        self.p = p
        self.scale_factor = scale_factor
    def __call__(self, pointmap: torch.Tensor, mask: torch.Tensor):
        if np.random.rand() > self.p:
            return pointmap

        _, h, w = pointmap.shape
        dep = torch.clone(pointmap[[-1], :, :])[None, ...]

        dep = torch.nn.functional.interpolate(
            dep, scale_factor=self.scale_factor, mode='bilinear'
        )
        dep = torch.nn.functional.interpolate(
            dep, size=(h, w), mode='bilinear'
        )

        pointmap = torch.cat([pointmap[0:2], dep[0]], dim=0)
        return pointmap
        
class RandomPatchMinMaxJitter:
    def __init__(self, patch_range=[0.25, 0.5], p=0.2, max_jitter_ratio=0.4) -> None:
        self.patch_range = patch_range
        self.p = p
        self.max_jitter_ratio = max_jitter_ratio
    def __call__(self, pointmap: torch.Tensor, mask: torch.Tensor):
        
        if np.random.rand() > self.p:
            return pointmap

        _, h, w = pointmap.shape

        patch_w = int(np.random.uniform(*self.patch_range) * w)
        patch_h = int(np.random.uniform(*self.patch_range) * h)

        x = np.random.randint(0, w - patch_w)
        y = np.random.randint(0, h - patch_h)
        
        patch_mask = torch.zeros_like(mask, dtype=torch.bool)
        patch_mask[y : y + patch_h, x : x + patch_w] = True
        patch_mask = patch_mask * mask
        patch_valid = patch_mask.sum()
        max_jitter_num = patch_valid * self.max_jitter_ratio
        if patch_valid > max_jitter_num * 2:
            jitter_num = max_jitter_num
        else:
            jitter_num = int(patch_valid / 2)
        jitter_num = np.random.randint(int(jitter_num/2), jitter_num)
        dep = torch.clone(pointmap[-1, :, :])
        patch_dep = dep[patch_mask]
        max_d, _ = torch.topk(patch_dep, jitter_num)
        max_d = max_d.min()
        min_d, _ = torch.topk(patch_dep, jitter_num, largest=False)
        min_d = min_d.max()
        if np.random.rand() > 0.5:
            dep[patch_mask & (dep<min_d)] = max_d
        else:
            dep[patch_mask & (dep>max_d)] = min_d
        pointmap = torch.cat([pointmap[0:2], dep[None, ...]], dim=0)
        return pointmap