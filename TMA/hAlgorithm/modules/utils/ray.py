import cv2
import numpy as np
import torch

import einops as ein
from einops import rearrange


def compute_rays(c2w, fxfycxcy, h, w, device="cuda"):
    """
    Args:
        c2w (torch.tensor): [b, v, 4, 4]
        fxfycxcy (torch.tensor): [b, v, 4]
        h (int): height of the image
        w (int): width of the image
    Returns:
        ray_o (torch.tensor): [b, v, 3, h, w]
        ray_d (torch.tensor): [b, v, 3, h, w]
    """

    b, v = fxfycxcy.size()[:2]
    

    fxfycxcy = fxfycxcy.reshape(b * v, 4)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(device), x.to(device)
    x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
    
    if c2w is not None:
        c2w = c2w.reshape(b * v, 4, 4)
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
    else:
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_o = None

    ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

    return ray_o, ray_d

def get_rays_in_camera_frame(intrinsics, height, width, normalize_to_unit_sphere):
    """
    Convert camera intrinsics to a raymap (ray origins + directions) in camera frame.
    Note: Currently only supports pinhole camera model.

    Args:
        - intrinsics: 3x3 or Bx3x3 torch tensor
        - height: int
        - width: int
        - normalize_to_unit_sphere: bool

    Returns:
        - ray_origins: (HxWx3 or BxHxWx3) tensor
        - ray_directions: (HxWx3 or BxHxWx3) tensor
    """
    # Add batch dimension if not present
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size = intrinsics.shape[0]
    device = intrinsics.device

    # Compute rays in camera frame associated with each pixel
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width , device=device).float() + 0.5,
        torch.arange(height, device=device).float() + 0.5,
        indexing="xy",
    )
    x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)

    fx = intrinsics[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics[:, 1, 2].view(-1, 1, 1)

    xx = (x_grid - cx) / fx
    yy = (y_grid - cy) / fy
    ray_directions = torch.stack((xx, yy, torch.ones_like(xx)), dim=1)

    # Normalize ray directions to unit sphere if required (else rays will lie on unit plane)
    if normalize_to_unit_sphere:
        ray_directions = ray_directions / torch.norm(
            ray_directions, dim=1, keepdim=True
        )

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        ray_directions = ray_directions.squeeze(0)

    return ray_directions


def get_rays_in_camera_frame_fisheye(
    intrinsics,
    height,
    width,
    distortion_coeffs=None,
    fisheye_model="equidistant",
    k_coeffs=None,
    sensor_size=None,
    crop_offset=None,
):
    """
    Convert fisheye camera intrinsics to a raymap (ray directions) in camera frame.

    Supported fisheye models:
        - "equidistant": r_d = theta  (f-theta model, no distortion coefficients needed)
        - "opencv_fisheye": OpenCV fisheye (Kannala-Brandt) model
            r_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
            Unprojection uses Newton's method to invert.
        - "blender": Blender polynomial fisheye model
            theta = |k0 + k1*r + k2*r^2 + k3*r^3 + k4*r^4|, where r is physical
            radius on sensor plane (mm).

    Args:
        - intrinsics: 3x3 or Bx3x3 torch tensor (fx, fy, cx, cy)
        - height: int
        - width: int
        - distortion_coeffs: None, (4,) or (B, 4) torch tensor [k1, k2, k3, k4]
            Required for "opencv_fisheye", ignored for "equidistant".
        - fisheye_model: str, one of "equidistant", "opencv_fisheye", "blender"
        - k_coeffs: None, (5,) or (B, 5) tensor, required for "blender"
        - sensor_size: None, (2,) or (B, 2) tensor [sensor_w, sensor_h] in mm,
            required for "blender"
        - crop_offset: None, (2,) or (B, 2) tensor [offset_x, offset_y] in pixels,
            optional for "blender"

    Returns:
        - ray_directions: (3xHxW or Bx3xHxW) tensor, unit-normalized
    """
    eps = 1e-8
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size = intrinsics.shape[0]
    device = intrinsics.device
    dtype = intrinsics.dtype

    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        indexing="xy",
    )
    x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)

    fx = intrinsics[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics[:, 1, 2].view(-1, 1, 1)

    mx = (x_grid - cx) / fx
    my = (y_grid - cy) / fy
    r_d = torch.sqrt(mx ** 2 + my ** 2).clamp(min=eps)

    if fisheye_model == "equidistant":
        theta = r_d

        phi = torch.atan2(my, mx)
        sin_theta = torch.sin(theta)
        ray_directions = torch.stack([
            sin_theta * torch.cos(phi),
            sin_theta * torch.sin(phi),
            torch.cos(theta),
        ], dim=1)  # [B, 3, H, W]

    elif fisheye_model == "opencv_fisheye":
        if distortion_coeffs is None:
            raise ValueError("distortion_coeffs is required for opencv_fisheye model")
        if distortion_coeffs.dim() == 1:
            distortion_coeffs = distortion_coeffs.unsqueeze(0)

        ray_list = []
        for b in range(batch_size):
            K_np = intrinsics[b].detach().cpu().numpy().astype(np.float64)
            D_np = distortion_coeffs[b].detach().cpu().numpy().astype(np.float64).reshape(4, 1)

            uv_np = torch.stack([x_grid[b], y_grid[b]], dim=-1)
            uv_np = uv_np.detach().cpu().numpy().astype(np.float64).reshape(-1, 1, 2)

            undistorted = cv2.fisheye.undistortPoints(uv_np, K_np, D_np)  # (HW, 1, 2)
            xy = undistorted[:, 0, :]  # (HW, 2)

            dirs = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float64)], axis=1)
            dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
            dirs = dirs.reshape(height, width, 3).transpose(2, 0, 1)  # (3, H, W)
            ray_list.append(torch.from_numpy(dirs).to(device=device, dtype=dtype))

        ray_directions = torch.stack(ray_list, dim=0)  # [B, 3, H, W]

    elif fisheye_model == "blender":
        if k_coeffs is None:
            raise ValueError("k_coeffs is required for blender model")
        if sensor_size is None:
            raise ValueError("sensor_size is required for blender model")

        k_coeffs = torch.as_tensor(k_coeffs, device=device, dtype=dtype)
        sensor_size = torch.as_tensor(sensor_size, device=device, dtype=dtype)
        if crop_offset is None:
            crop_offset = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        else:
            crop_offset = torch.as_tensor(crop_offset, device=device, dtype=dtype)

        if k_coeffs.dim() == 1:
            k_coeffs = k_coeffs.unsqueeze(0)
        if sensor_size.dim() == 1:
            sensor_size = sensor_size.unsqueeze(0)
        if crop_offset.dim() == 1:
            crop_offset = crop_offset.unsqueeze(0)

        if k_coeffs.shape[0] == 1 and batch_size > 1:
            k_coeffs = k_coeffs.expand(batch_size, -1)
        if sensor_size.shape[0] == 1 and batch_size > 1:
            sensor_size = sensor_size.expand(batch_size, -1)
        if crop_offset.shape[0] == 1 and batch_size > 1:
            crop_offset = crop_offset.expand(batch_size, -1)

        if k_coeffs.shape != (batch_size, 5):
            raise ValueError(f"k_coeffs must have shape ({batch_size}, 5), got {tuple(k_coeffs.shape)}")
        if sensor_size.shape != (batch_size, 2):
            raise ValueError(
                f"sensor_size must have shape ({batch_size}, 2), got {tuple(sensor_size.shape)}"
            )
        if crop_offset.shape != (batch_size, 2):
            raise ValueError(
                f"crop_offset must have shape ({batch_size}, 2), got {tuple(crop_offset.shape)}"
            )

        pixel_size_w = sensor_size[:, 0].view(-1, 1, 1) / (width + 2 * crop_offset[:, 0].view(-1, 1, 1))
        pixel_size_h = sensor_size[:, 1].view(-1, 1, 1) / (height + 2 * crop_offset[:, 1].view(-1, 1, 1))
        x_phys = (x_grid - cx) * pixel_size_w
        y_phys = (y_grid - cy) * pixel_size_h
        r_phys = torch.sqrt(x_phys ** 2 + y_phys ** 2)

        k0 = k_coeffs[:, 0].view(-1, 1, 1)
        k1 = k_coeffs[:, 1].view(-1, 1, 1)
        k2 = k_coeffs[:, 2].view(-1, 1, 1)
        k3 = k_coeffs[:, 3].view(-1, 1, 1)
        k4 = k_coeffs[:, 4].view(-1, 1, 1)

        theta = torch.abs(k0 + k1 * r_phys + k2 * r_phys ** 2 + k3 * r_phys ** 3 + k4 * r_phys ** 4)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        factor = torch.zeros_like(r_phys)
        mask_valid_r = r_phys > eps
        factor[mask_valid_r] = sin_theta[mask_valid_r] / r_phys[mask_valid_r]

        ray_x = x_phys * factor
        ray_y = y_phys * factor
        ray_z = cos_theta

        ray_x[~mask_valid_r] = 0.0
        ray_y[~mask_valid_r] = 0.0
        ray_z[~mask_valid_r] = 1.0
        ray_directions = torch.stack([ray_x, ray_y, ray_z], dim=1)  # [B, 3, H, W]

    else:
        raise ValueError(f"Unknown fisheye model: {fisheye_model}. "
                         f"Supported: 'equidistant', 'opencv_fisheye', 'blender'")

    ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True).clamp(min=eps)

    if squeeze_batch_dim:
        ray_directions = ray_directions.squeeze(0)

    return ray_directions


def get_rays_in_world_frame(
    intrinsics, height, width, normalize_to_unit_sphere, camera_pose=None
):
    """
    Convert camera intrinsics & camera_pose (if provided) to a raymap (ray origins + directions) in camera or world frame (if camera_pose is provided).
    Note: Currently only supports pinhole camera model.

    Args:
        - intrinsics: 3x3 or Bx3x3 torch tensor
        - height: int
        - width: int
        - normalize_to_unit_sphere: bool
        - camera_pose: 4x4 or Bx4x4 torch tensor

    Returns:
        # - ray_origins: (HxWx3 or BxHxWx3) tensor
        # - ray_directions: (HxWx3 or BxHxWx3) tensor
        - ray_origins: (3xHxW or Bx3xHxW) tensor
        - ray_directions: (3xHxW or Bx3xHxW) tensor
    """
    # Get rays in camera frame
    ray_directions = get_rays_in_camera_frame(
        intrinsics, height, width, normalize_to_unit_sphere
    )

    if ray_directions.ndim == 4:
        ray_directions = ray_directions.permute(0, 2, 3, 1).contiguous()
    else:
        ray_directions = ray_directions.permute(1, 2, 0).contiguous()

    if intrinsics.dim() == 2:
        ray_origins = torch.zeros((height, width, 3), device=intrinsics.device)
    else:
        ray_origins = torch.zeros((intrinsics.shape[0], height, width, 3), device=intrinsics.device)

    if camera_pose is not None:
        # Add batch dimension if not present
        if camera_pose.dim() == 2:
            camera_pose = camera_pose.unsqueeze(0)
            ray_origins = ray_origins.unsqueeze(0)
            ray_directions = ray_directions.unsqueeze(0)
            squeeze_batch_dim = True
        else:
            squeeze_batch_dim = False

        # Convert rays from camera frame to world frame
        ray_origins_homo = torch.cat(
            [ray_origins, torch.ones_like(ray_origins[..., :1])], dim=-1
        )
        ray_directions_homo = torch.cat(
            [ray_directions, torch.zeros_like(ray_directions[..., :1])], dim=-1
        )
        ray_origins_world = ein.einsum(
            camera_pose, ray_origins_homo, "b i k, b h w k -> b h w i"
        )
        ray_directions_world = ein.einsum(
            camera_pose, ray_directions_homo, "b i k, b h w k -> b h w i"
        )
        ray_origins_world = ray_origins_world[..., :3]
        ray_directions_world = ray_directions_world[..., :3]

        # Remove batch dimension if it was added
        if squeeze_batch_dim:
            ray_origins_world = ray_origins_world.squeeze(0)
            ray_directions_world = ray_directions_world.squeeze(0)
    else:
        ray_origins_world = ray_origins
        ray_directions_world = ray_directions
    
    if ray_origins_world.ndim == 4:
        ray_origins_world = ray_origins_world.permute(0, 3, 1, 2).contiguous()
        ray_directions_world = ray_directions_world.permute(0, 3, 1, 2).contiguous()
    else:
        ray_origins_world = ray_origins_world.permute(2, 0, 1).contiguous()
        ray_directions_world = ray_directions_world.permute(2, 0, 1).contiguous()

    return ray_origins_world, ray_directions_world


def recover_pinhole_intrinsics_from_ray_directions(
    ray_directions, use_geometric_calculation=False
):
    """
    Recover pinhole camera intrinsics from ray directions, supporting both batched and non-batched inputs.

    Args:
        ray_directions: Tensor of shape [H, W, 3] or [B, H, W, 3] containing unit normalized ray directions

    Returns:
        Dictionary containing camera intrinsics (fx, fy, cx, cy) as tensors
    """
    # Add batch dimension if not present
    if ray_directions.dim() == 3:  # [H, W, 3]
        ray_directions = ray_directions.unsqueeze(0)  # [1, H, W, 3]
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size, height, width, _ = ray_directions.shape
    device = ray_directions.device

    # Create pixel coordinate grid
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device).float(),
        torch.arange(height, device=device).float(),
        indexing="xy",
    )

    # Expand grid for all batches
    x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]
    y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]

    # Determine if high resolution or not
    is_high_res = height * width > 1000000

    if is_high_res or use_geometric_calculation:
        # For high-resolution cases, use direct geometric calculation
        # Define key points
        center_h, center_w = height // 2, width // 2
        quarter_w, three_quarter_w = width // 4, 3 * width // 4
        quarter_h, three_quarter_h = height // 4, 3 * height // 4

        # Get rays at key points
        center_rays = ray_directions[:, center_h, center_w, :].clone()  # [B, 3]
        left_rays = ray_directions[:, center_h, quarter_w, :].clone()  # [B, 3]
        right_rays = ray_directions[:, center_h, three_quarter_w, :].clone()  # [B, 3]
        top_rays = ray_directions[:, quarter_h, center_w, :].clone()  # [B, 3]
        bottom_rays = ray_directions[:, three_quarter_h, center_w, :].clone()  # [B, 3]

        # Normalize rays to have dz = 1
        center_rays = center_rays / center_rays[:, 2].unsqueeze(1)  # [B, 3]
        left_rays = left_rays / left_rays[:, 2].unsqueeze(1)  # [B, 3]
        right_rays = right_rays / right_rays[:, 2].unsqueeze(1)  # [B, 3]
        top_rays = top_rays / top_rays[:, 2].unsqueeze(1)  # [B, 3]
        bottom_rays = bottom_rays / bottom_rays[:, 2].unsqueeze(1)  # [B, 3]

        # Calculate fx directly (vectorized across batch)
        fx_left = (quarter_w - center_w) / (left_rays[:, 0] - center_rays[:, 0])
        fx_right = (three_quarter_w - center_w) / (right_rays[:, 0] - center_rays[:, 0])
        fx = (fx_left + fx_right) / 2  # Average for robustness

        # Calculate cx
        cx = center_w - fx * center_rays[:, 0]

        # Calculate fy and cy
        fy_top = (quarter_h - center_h) / (top_rays[:, 1] - center_rays[:, 1])
        fy_bottom = (three_quarter_h - center_h) / (
            bottom_rays[:, 1] - center_rays[:, 1]
        )
        fy = (fy_top + fy_bottom) / 2

        cy = center_h - fy * center_rays[:, 1]
    else:
        # For standard resolution, use regression with sampling for efficiency
        # Sample a grid of points (but more dense than for high-res)
        step_h = max(1, height // 50)
        step_w = max(1, width // 50)

        h_indices = torch.arange(0, height, step_h, device=device)
        w_indices = torch.arange(0, width, step_w, device=device)

        # Extract subset of coordinates
        x_sampled = x_grid[:, h_indices[:, None], w_indices[None, :]]  # [B, H', W']
        y_sampled = y_grid[:, h_indices[:, None], w_indices[None, :]]  # [B, H', W']
        rays_sampled = ray_directions[
            :, h_indices[:, None], w_indices[None, :], :
        ]  # [B, H', W', 3]

        # Reshape for linear regression
        x_flat = x_sampled.reshape(batch_size, -1)  # [B, N]
        y_flat = y_sampled.reshape(batch_size, -1)  # [B, N]

        # Extract ray direction components
        dx = rays_sampled[..., 0].reshape(batch_size, -1)  # [B, N]
        dy = rays_sampled[..., 1].reshape(batch_size, -1)  # [B, N]
        dz = rays_sampled[..., 2].reshape(batch_size, -1)  # [B, N]

        # Compute ratios for linear regression
        ratio_x = dx / dz  # [B, N]
        ratio_y = dy / dz  # [B, N]

        # Since torch.linalg.lstsq doesn't support batched input, we'll use a different approach
        # For x-direction: x = cx + fx * (dx/dz)
        # We can solve this using normal equations: A^T A x = A^T b
        # Create design matrices
        ones = torch.ones_like(x_flat)  # [B, N]
        A_x = torch.stack([ones, ratio_x], dim=2)  # [B, N, 2]
        b_x = x_flat.unsqueeze(2)  # [B, N, 1]

        # Compute A^T A and A^T b for each batch
        ATA_x = torch.bmm(A_x.transpose(1, 2), A_x)  # [B, 2, 2]
        ATb_x = torch.bmm(A_x.transpose(1, 2), b_x)  # [B, 2, 1]

        # Solve the system for each batch
        solution_x = torch.linalg.solve(ATA_x, ATb_x).squeeze(2)  # [B, 2]
        cx, fx = solution_x[:, 0], solution_x[:, 1]

        # Repeat for y-direction
        A_y = torch.stack([ones, ratio_y], dim=2)  # [B, N, 2]
        b_y = y_flat.unsqueeze(2)  # [B, N, 1]

        ATA_y = torch.bmm(A_y.transpose(1, 2), A_y)  # [B, 2, 2]
        ATb_y = torch.bmm(A_y.transpose(1, 2), b_y)  # [B, 2, 1]

        solution_y = torch.linalg.solve(ATA_y, ATb_y).squeeze(2)  # [B, 2]
        cy, fy = solution_y[:, 0], solution_y[:, 1]

    # Create intrinsics matrices
    batch_size = fx.shape[0]
    intrinsics = torch.zeros(batch_size, 3, 3, device=ray_directions.device)

    # Fill in the intrinsics matrices
    intrinsics[:, 0, 0] = fx  # focal length x
    intrinsics[:, 1, 1] = fy  # focal length y
    intrinsics[:, 0, 2] = cx  # principal point x
    intrinsics[:, 1, 2] = cy  # principal point y
    intrinsics[:, 2, 2] = 1.0  # bottom-right element is always 1

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        intrinsics = intrinsics.squeeze(0)

    return intrinsics


def debug_visualize_rays(ray_directions, save_path, rgb_mask=None, title="Ray Directions", batch_idx=0,
                         theta_max_deg=200.0):
    """
    Visualize ray direction maps and save as image. Produces a 1x3 figure:
        theta heatmap | 3D rays (front view) | 3D rays (side view)

    Args:
        ray_directions: (3, H, W) or (B, 3, H, W) tensor
        save_path: str, output image path (e.g. "debug_rays.png")
        title: str, figure suptitle
        batch_idx: int, which batch element to visualize
        theta_max_deg: float, upper clamp for theta colorbar (degrees)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ray_directions.dim() == 4:
        ray_directions = ray_directions[batch_idx]
    ray_directions = ray_directions.detach().cpu().float()

    dx = ray_directions[0]  # [H, W]
    dy = ray_directions[1]
    dz = ray_directions[2]

    theta_deg = torch.rad2deg(torch.acos(dz.clamp(-1, 1)))
    if rgb_mask is not None:
        theta_deg[~rgb_mask[batch_idx, 0].cpu()] = 0.0

    fig = plt.figure(figsize=(18, 5.5))
    fig.suptitle(title, fontsize=14, y=1.02)

    ax_theta = fig.add_subplot(1, 3, 1)
    im = ax_theta.imshow(theta_deg.numpy(), cmap="inferno", vmin=0, vmax=theta_max_deg)
    ax_theta.set_title(f"theta (deg), clamp [0, {theta_max_deg}]")
    ax_theta.set_xlabel("u")
    ax_theta.set_ylabel("v")
    plt.colorbar(im, ax=ax_theta, fraction=0.046, pad=0.04)

    H, W = dx.shape
    step_h = max(1, H // 15)
    step_w = max(1, W // 15)
    sample_dx = dx[::step_h, ::step_w].numpy()
    sample_dy = dy[::step_h, ::step_w].numpy()
    sample_dz = dz[::step_h, ::step_w].numpy()
    sh, sw = sample_dx.shape
    origins = torch.zeros(sh, sw).numpy()

    ax3d = fig.add_subplot(1, 3, 2, projection="3d")
    ax3d.quiver(
        origins, origins, origins,
        sample_dx, sample_dy, sample_dz,
        length=0.3, normalize=True, arrow_length_ratio=0.15, alpha=0.6, linewidth=0.6,
    )
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("3D rays (front)")
    ax3d.set_xlim(-1, 1)
    ax3d.set_ylim(-1, 1)
    ax3d.set_zlim(-0.5, 1.5)
    ax3d.view_init(elev=-60, azim=-90)

    ax3d_side = fig.add_subplot(1, 3, 3, projection="3d")
    ax3d_side.quiver(
        origins, origins, origins,
        sample_dx, sample_dy, sample_dz,
        length=0.3, normalize=True, arrow_length_ratio=0.15, alpha=0.6, linewidth=0.6,
    )
    ax3d_side.set_xlabel("X")
    ax3d_side.set_ylabel("Y")
    ax3d_side.set_zlabel("Z")
    ax3d_side.set_title("3D rays (side)")
    ax3d_side.set_xlim(-1, 1)
    ax3d_side.set_ylim(-1, 1)
    ax3d_side.set_zlim(-0.5, 1.5)
    ax3d_side.view_init(elev=0, azim=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[debug_visualize_rays] saved to {save_path}")


def debug_visualize_rays_numpy(ray_directions, save_path, rgb_mask=None, title="Ray Directions", batch_idx=0,
                               theta_max_deg=200.0):
    """
    Visualize ray direction maps and save as image (pure numpy version).
    Produces a 1x3 figure:
        theta heatmap | 3D rays (front view) | 3D rays (side view)

    Args:
        ray_directions: (3, H, W) or (B, 3, H, W) numpy array
        save_path: str, output image path (e.g. "debug_rays.png")
        rgb_mask: optional (B, 1, H, W) numpy bool array
        title: str, figure suptitle
        batch_idx: int, which batch element to visualize
        theta_max_deg: float, upper clamp for theta colorbar (degrees)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ray_directions.ndim == 4:
        ray_directions = ray_directions[batch_idx]
    ray_directions = ray_directions.astype(np.float32)

    dx = ray_directions[..., 0]  # [H, W]
    dy = ray_directions[..., 1]
    dz = ray_directions[..., 2]

    theta_deg = np.rad2deg(np.arccos(np.clip(dz, -1, 1)))
    if rgb_mask is not None:
        theta_deg[~rgb_mask[batch_idx, 0]] = 0.0

    fig = plt.figure(figsize=(18, 5.5))
    fig.suptitle(title, fontsize=14, y=1.02)

    ax_theta = fig.add_subplot(1, 3, 1)
    im = ax_theta.imshow(theta_deg, cmap="inferno", vmin=0, vmax=theta_max_deg)
    ax_theta.set_title(f"theta (deg), clamp [0, {theta_max_deg}]")
    ax_theta.set_xlabel("u")
    ax_theta.set_ylabel("v")
    plt.colorbar(im, ax=ax_theta, fraction=0.046, pad=0.04)

    H, W = dx.shape
    step_h = max(1, H // 15)
    step_w = max(1, W // 15)
    sample_dx = dx[::step_h, ::step_w]
    sample_dy = dy[::step_h, ::step_w]
    sample_dz = dz[::step_h, ::step_w]
    sh, sw = sample_dx.shape
    origins = np.zeros((sh, sw))

    ax3d = fig.add_subplot(1, 3, 2, projection="3d")
    ax3d.quiver(
        origins, origins, origins,
        sample_dx, sample_dy, sample_dz,
        length=0.3, normalize=True, arrow_length_ratio=0.15, alpha=0.6, linewidth=0.6,
    )
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("3D rays (front)")
    ax3d.set_xlim(-1, 1)
    ax3d.set_ylim(-1, 1)
    ax3d.set_zlim(-0.5, 1.5)
    ax3d.view_init(elev=-60, azim=-90)

    ax3d_side = fig.add_subplot(1, 3, 3, projection="3d")
    ax3d_side.quiver(
        origins, origins, origins,
        sample_dx, sample_dy, sample_dz,
        length=0.3, normalize=True, arrow_length_ratio=0.15, alpha=0.6, linewidth=0.6,
    )
    ax3d_side.set_xlabel("X")
    ax3d_side.set_ylabel("Y")
    ax3d_side.set_zlabel("Z")
    ax3d_side.set_title("3D rays (side)")
    ax3d_side.set_xlim(-1, 1)
    ax3d_side.set_ylim(-1, 1)
    ax3d_side.set_zlim(-0.5, 1.5)
    ax3d_side.view_init(elev=0, azim=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[debug_visualize_rays_numpy] saved to {save_path}")


def debug_visualize_rays_comparison_numpy(
    ray_directions_pred, ray_directions_gt, save_path,
    rgb_mask=None, batch_idx=0, theta_max_deg=200.0, error_max_deg=10.0,
):
    """
    Compare predicted vs GT ray direction maps with angular error visualization.

    Layout (2 rows x 3 cols):
        Row 1: GT theta | Pred theta | Angular error heatmap
        Row 2: GT-Pred overlay (front) | GT-Pred overlay (side) | Error histogram

    Args:
        ray_directions_pred: (H, W, 3) or (B, H, W, 3) numpy array
        ray_directions_gt:   (H, W, 3) or (B, H, W, 3) numpy array
        save_path: output image path
        rgb_mask: optional (B, 1, H, W) numpy bool array
        batch_idx: which batch element to visualize
        theta_max_deg: upper clamp for theta colorbar
        error_max_deg: upper clamp for angular error colorbar
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ray_directions_pred.ndim == 4:
        ray_directions_pred = ray_directions_pred[batch_idx]
    if ray_directions_gt.ndim == 4:
        ray_directions_gt = ray_directions_gt[batch_idx]

    pred = ray_directions_pred.astype(np.float64)
    gt = ray_directions_gt.astype(np.float64)

    pred_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    gt_norm = gt / (np.linalg.norm(gt, axis=-1, keepdims=True) + 1e-8)

    cos_sim = np.clip(np.sum(pred_norm * gt_norm, axis=-1), -1.0, 1.0)
    angular_error_deg = np.rad2deg(np.arccos(cos_sim))

    if rgb_mask is not None:
        valid = rgb_mask[batch_idx, 0].astype(bool)
    else:
        valid = np.ones(angular_error_deg.shape, dtype=bool)

    valid_errors = angular_error_deg[valid]
    mean_err = np.mean(valid_errors) if valid_errors.size > 0 else 0.0
    median_err = np.median(valid_errors) if valid_errors.size > 0 else 0.0
    max_err = np.max(valid_errors) if valid_errors.size > 0 else 0.0
    pct_below_1 = np.mean(valid_errors < 1.0) * 100 if valid_errors.size > 0 else 0.0
    pct_below_5 = np.mean(valid_errors < 5.0) * 100 if valid_errors.size > 0 else 0.0

    theta_gt = np.rad2deg(np.arccos(np.clip(gt[..., 2], -1, 1)))
    theta_pred = np.rad2deg(np.arccos(np.clip(pred[..., 2], -1, 1)))
    if rgb_mask is not None:
        theta_gt[~valid] = 0.0
        theta_pred[~valid] = 0.0
        angular_error_deg[~valid] = 0.0

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Ray Comparison  |  mean={mean_err:.2f}°  median={median_err:.2f}°  "
        f"max={max_err:.2f}°  <1°={pct_below_1:.1f}%  <5°={pct_below_5:.1f}%",
        fontsize=13, y=0.98,
    )

    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(theta_gt, cmap="inferno", vmin=0, vmax=theta_max_deg)
    ax1.set_title("GT theta (deg)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(theta_pred, cmap="inferno", vmin=0, vmax=theta_max_deg)
    ax2.set_title("Pred theta (deg)")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(angular_error_deg, cmap="turbo", vmin=0, vmax=error_max_deg)
    ax3.set_title("Angular error (deg)")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    H, W = pred.shape[:2]
    step_h = max(1, H // 15)
    step_w = max(1, W // 15)

    gt_dx = gt[::step_h, ::step_w, 0]
    gt_dy = gt[::step_h, ::step_w, 1]
    gt_dz = gt[::step_h, ::step_w, 2]
    pred_dx = pred[::step_h, ::step_w, 0]
    pred_dy = pred[::step_h, ::step_w, 1]
    pred_dz = pred[::step_h, ::step_w, 2]
    sh, sw = gt_dx.shape
    origins = np.zeros((sh, sw))

    for subplot_idx, (elev, azim, view_label) in enumerate([
        (-60, -90, "front"), (0, 0, "side"),
    ]):
        ax3d = fig.add_subplot(2, 3, 4 + subplot_idx, projection="3d")
        ax3d.quiver(
            origins, origins, origins,
            gt_dx, gt_dy, gt_dz,
            length=0.3, normalize=True, arrow_length_ratio=0.15,
            alpha=0.5, linewidth=0.6, color="dodgerblue", label="GT",
        )
        ax3d.quiver(
            origins, origins, origins,
            pred_dx, pred_dy, pred_dz,
            length=0.3, normalize=True, arrow_length_ratio=0.15,
            alpha=0.5, linewidth=0.6, color="orangered", label="Pred",
        )
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
        ax3d.set_title(f"Overlay ({view_label})")
        ax3d.set_xlim(-1, 1); ax3d.set_ylim(-1, 1); ax3d.set_zlim(-0.5, 1.5)
        ax3d.view_init(elev=elev, azim=azim)
        ax3d.legend(fontsize=8, loc="upper right")

    ax_hist = fig.add_subplot(2, 3, 6)
    if valid_errors.size > 0:
        bins = np.linspace(0, min(error_max_deg * 2, max_err + 1), 80)
        ax_hist.hist(valid_errors, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
        ax_hist.axvline(mean_err, color="red", linestyle="--", linewidth=1.2, label=f"mean={mean_err:.2f}°")
        ax_hist.axvline(median_err, color="orange", linestyle="--", linewidth=1.2, label=f"median={median_err:.2f}°")
        ax_hist.legend(fontsize=9)
    ax_hist.set_title("Angular error distribution")
    ax_hist.set_xlabel("Error (deg)")
    ax_hist.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[debug_visualize_rays_comparison] saved to {save_path}")


if __name__ == "__main__":
    import os

    save_dir = os.path.join(os.path.dirname(__file__), "debug_ray_vis")
    os.makedirs(save_dir, exist_ok=True)

    H, W = 512, 512

    # --- Pinhole ---
    intrinsics_pinhole = torch.zeros(1, 3, 3).float()
    intrinsics_pinhole[0, 0, 0] = 367.0
    intrinsics_pinhole[0, 1, 1] = 367.0
    intrinsics_pinhole[0, 0, 2] = W / 2.0
    intrinsics_pinhole[0, 1, 2] = H / 2.0

    rays_pinhole = get_rays_in_camera_frame(
        intrinsics=intrinsics_pinhole, height=H, width=W, normalize_to_unit_sphere=True,
    )
    print(f"Pinhole ray_directions shape: {rays_pinhole.shape}")
    debug_visualize_rays(rays_pinhole, os.path.join(save_dir, "pinhole_rays.png"),
                         title=f"Pinhole (fx=fy=367, {W}x{H})")

    recovered = recover_pinhole_intrinsics_from_ray_directions(rays_pinhole.permute(0, 2, 3, 1))
    print(f"Recovered pinhole intrinsics:\n{recovered}")

    # --- Fisheye equidistant ---
    intrinsics_fe = torch.zeros(1, 3, 3).float()
    intrinsics_fe[0, 0, 0] = 200.0
    intrinsics_fe[0, 1, 1] = 200.0
    intrinsics_fe[0, 0, 2] = W / 2.0
    intrinsics_fe[0, 1, 2] = H / 2.0

    rays_equidist = get_rays_in_camera_frame_fisheye(
        intrinsics=intrinsics_fe, height=H, width=W, fisheye_model="equidistant",
    )
    print(f"Fisheye equidistant ray_directions shape: {rays_equidist.shape}")
    debug_visualize_rays(rays_equidist, os.path.join(save_dir, "fisheye_equidistant_rays.png"),
                         title=f"Fisheye Equidistant (fx=fy=200, {W}x{H})")

    # --- Fisheye opencv KB4 ---
    dist_coeffs = torch.tensor([0.1, -0.05, 0.02, -0.005])
    rays_kb4 = get_rays_in_camera_frame_fisheye(
        intrinsics=intrinsics_fe, height=H, width=W,
        distortion_coeffs=dist_coeffs, fisheye_model="opencv_fisheye",
    )
    print(f"Fisheye opencv_fisheye ray_directions shape: {rays_kb4.shape}")
    debug_visualize_rays(rays_kb4, os.path.join(save_dir, "fisheye_opencv_kb4_rays.png"),
                         title=f"Fisheye OpenCV KB4 (fx=fy=200, k=[0.1,-0.05,0.02,-0.005], {W}x{H})")

    print(f"\nAll visualizations saved to: {save_dir}")
