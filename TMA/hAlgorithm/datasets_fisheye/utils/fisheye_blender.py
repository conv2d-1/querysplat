import torch
import numpy as np
import warnings

def load_pointmap_blender_polynomial_fisheye(
    depth,
    intrinsics,
    k_coeffs, 
    sensor_size, # [k0, k1, k2, k3, k4]
    dist_coeffs=None,
    crop_offset=None,
):
    """
    Converts a depth map from a polynomial fisheye camera (e.g., calibrated with OpenCV)
    into a 3D point map (ray-based), using the inverse polynomial model:
        theta = k0 + k1*r + k2*r^2 + k3*r^3 + k4*r^4

    Args:
        depth (torch.Tensor or np.ndarray): Depth map of shape (H, W) or (1, H, W).
            Must represent **ray distance** (Euclidean distance from camera center).
        intrinsics (torch.Tensor or np.ndarray): 3x3 camera intrinsic matrix.
        k_coeffs (list or np.ndarray): Polynomial coefficients [k0, k1, k2, k3, k4].
        dist_coeffs: Ignored (for compatibility). Polynomial model subsumes distortion.
        crop_offset (tuple or list or None): (offset_x, offset_y) in pixels, the top-left
            corner of the crop window in the original (pre-crop) image. Used to restore
            correct u,v coordinates after cropping. None means no crop was applied.

    Returns:
        torch.Tensor: Point map of shape (3, H, W), in camera coordinates.
    """
    if dist_coeffs is not None:
        warnings.warn("dist_coeffs is ignored for polynomial fisheye model.")

    # Convert to numpy if needed
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze(0).cpu().numpy()  # Remove batch dim if present
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()
    if isinstance(k_coeffs, torch.Tensor):
        k_coeffs = k_coeffs.cpu().numpy()

    k_coeffs = np.asarray(k_coeffs, dtype=np.float64)
    assert k_coeffs.shape == (5,), f"Expected 5 coefficients, got {k_coeffs.shape}"

    height, width = depth.shape
    k0, k1, k2, k3, k4 = k_coeffs
    sensor_w , sensor_h = sensor_size

    # Create pixel coordinate grids (center of pixel)
    # If crop was applied, shift u,v to original image coordinates
    offset_x = crop_offset[0] if crop_offset is not None else 0.0
    offset_y = crop_offset[1] if crop_offset is not None else 0.0
    assert offset_x >= 0 and offset_y >= 0, f"Crop offset must be non-negative, got {offset_x}, {offset_y}"
    u, v = np.meshgrid(np.arange(width, dtype=np.float64) + 0.5,
                       np.arange(height, dtype=np.float64) + 0.5,
                       indexing="xy")

    # Unproject to normalized camera coordinates
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]

    # 计算每个像素的物理尺寸 (mm/pixel)
    # center crop 时原始分辨率 = crop后分辨率 + 两侧各裁掉的像素
    pixel_size_w = sensor_w / (width + 2 * offset_x)
    pixel_size_h = sensor_h / (height + 2 * offset_y)

    # 计算相对于光心的物理偏移 (mm)
    x_phys = (u - cx) * pixel_size_w
    y_phys = (v - cy) * pixel_size_h

    # 计算物理半径 r (mm)
    r_phys = np.sqrt(x_phys**2 + y_phys**2)

    # Compute theta using polynomial model: theta = k0 + k1*r + k2*r^2 + ...
    poly_val = (k0
                    + k1 * r_phys
                    + k2 * r_phys**2
                    + k3 * r_phys**3
                    + k4 * r_phys**4)
    theta = np.abs(poly_val)

    # Clamp theta to valid range [0, pi] for numerical safety
    # theta = np.clip(theta, 0.0, np.pi)

    # Precompute sin and cos
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Initialize direction vectors
    x_3d = np.zeros_like(x_phys)
    y_3d = np.zeros_like(y_phys)
    z_3d = np.ones_like(x_phys)

    # Avoid division by zero at center
    eps = 1e-8
    mask_valid = r_phys > eps

    # For valid pixels (r > eps), compute direction
    # Direction = [ (x_cam / r) * sin(theta), (y_cam / r) * sin(theta), cos(theta) ]
    # Equivalent to: x_cam * (sin(theta) / r), etc.
    factor = np.where(mask_valid, sin_theta / r_phys, 0.0)
    x_3d[mask_valid] = x_phys[mask_valid] * factor[mask_valid]
    y_3d[mask_valid] = y_phys[mask_valid] * factor[mask_valid]
    z_3d[mask_valid] = cos_theta[mask_valid]

    # At center (r <= eps), we assume theta ≈ 0 => direction = (0, 0, 1)
    # But if k0 != 0, this may be inconsistent. However, physically center must be (0,0,1).
    # So we enforce it.
    center_mask = ~mask_valid
    if np.any(center_mask):
        # Optional: warn if k0 is large
        if abs(k0) > 1e-3:
            warnings.warn(f"k0 = {k0:.4f} is non-zero; center direction forced to (0,0,1).")
        x_3d[center_mask] = 0.0
        y_3d[center_mask] = 0.0
        z_3d[center_mask] = 1.0

    # 归一化 (消除数值误差，确保单位向量)
    norm = np.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
   # 防止除零
    norm = np.where(norm < eps, 1.0, norm)
    x_3d /= norm
    y_3d /= norm
    z_3d /= norm

    # Scale by depth (ray distance)
    points = np.stack([
        x_3d * depth,
        y_3d * depth,
        z_3d * depth
    ], axis=0)  # Shape: (3, H, W)

    return torch.from_numpy(points).float()