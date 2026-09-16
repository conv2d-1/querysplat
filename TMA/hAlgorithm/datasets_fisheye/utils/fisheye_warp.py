import torch
import torch.nn.functional as F
import math


def _unproject_pinhole(kpts0_pixel, kpts0_depth, K0):
    """Pinhole unprojection: pixel + z-depth -> 3D point in camera coords.
    Args:
        kpts0_pixel: [N, L, 2] pixel coordinates
        kpts0_depth: [N, L] z-depth values
        K0: [N, 3, 3] intrinsics
    Returns:
        kpts0_cam: [N, 3, L] 3D points in camera coordinates
    """
    kpts0_h = (
        torch.cat([kpts0_pixel, torch.ones_like(kpts0_pixel[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
    return kpts0_cam


def _unproject_fisheye_equidistant(kpts0_pixel, kpts0_depth, K0):
    """Fisheye equidistant unprojection: pixel + radial depth -> 3D point.
    Args:
        kpts0_pixel: [N, L, 2] pixel coordinates
        kpts0_depth: [N, L] radial depth (Euclidean distance)
        K0: [N, 3, 3] intrinsics
    Returns:
        kpts0_cam: [N, 3, L] 3D points in camera coordinates
    """
    eps = 1e-8
    fx0 = K0[:, 0, 0].unsqueeze(1)
    fy0 = K0[:, 1, 1].unsqueeze(1)
    cx0 = K0[:, 0, 2].unsqueeze(1)
    cy0 = K0[:, 1, 2].unsqueeze(1)

    x_cam = (kpts0_pixel[..., 0] - cx0) / fx0
    y_cam = (kpts0_pixel[..., 1] - cy0) / fy0

    r = torch.sqrt(x_cam**2 + y_cam**2)
    sin_r = torch.sin(r)
    cos_r = torch.cos(r)

    mask_valid = r > eps
    factor = torch.ones_like(r)
    factor[mask_valid] = sin_r[mask_valid] / r[mask_valid]

    dir_x = x_cam * factor
    dir_y = y_cam * factor
    dir_z = cos_r

    norm = torch.sqrt(dir_x**2 + dir_y**2 + dir_z**2).clamp(min=eps)
    dir_x = dir_x / norm
    dir_y = dir_y / norm
    dir_z = dir_z / norm

    kpts0_cam = torch.stack([
        dir_x * kpts0_depth,
        dir_y * kpts0_depth,
        dir_z * kpts0_depth
    ], dim=1)  # [N, 3, L]
    return kpts0_cam


def _project_pinhole(w_kpts0_cam, K1):
    """Pinhole projection: 3D point -> pixel coordinates.
    Args:
        w_kpts0_cam: [N, 3, L] 3D points in target camera coordinates
        K1: [N, 3, 3] target intrinsics
    Returns:
        w_kpts0_pixel: [N, L, 2] pixel coordinates
        w_kpts0_depth_computed: [N, L] z-depth in target camera
    """
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0_pixel = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)
    return w_kpts0_pixel, w_kpts0_depth_computed


def _project_fisheye_equidistant(w_kpts0_cam, K1):
    """Fisheye equidistant projection: 3D point -> pixel coordinates.
    Args:
        w_kpts0_cam: [N, 3, L] 3D points in target camera coordinates
        K1: [N, 3, 3] target intrinsics
    Returns:
        w_kpts0_pixel: [N, L, 2] pixel coordinates
        w_kpts0_depth_computed: [N, L] radial depth in target camera
    """
    eps = 1e-8
    w_kpts0_radial_depth = torch.norm(w_kpts0_cam, dim=1)

    dir_w = w_kpts0_cam / (w_kpts0_radial_depth.unsqueeze(1) + eps)
    dir_x_w = dir_w[:, 0, :]
    dir_y_w = dir_w[:, 1, :]
    dir_z_w = dir_w[:, 2, :]

    theta = torch.acos(dir_z_w.clamp(-1.0 + eps, 1.0 - eps))
    sin_theta = torch.sin(theta)
    factor_proj = torch.ones_like(theta)
    mask_theta = theta > eps
    factor_proj[mask_theta] = theta[mask_theta] / sin_theta[mask_theta]

    x_prime = dir_x_w * factor_proj
    y_prime = dir_y_w * factor_proj

    fx1 = K1[:, 0, 0].unsqueeze(1)
    fy1 = K1[:, 1, 1].unsqueeze(1)
    cx1 = K1[:, 0, 2].unsqueeze(1)
    cy1 = K1[:, 1, 2].unsqueeze(1)

    w_kpts0_pixel = torch.stack([
        cx1 + fx1 * x_prime,
        cy1 + fy1 * y_prime
    ], dim=-1)

    return w_kpts0_pixel, w_kpts0_radial_depth


@torch.no_grad()
def warp_kpts_general(
    kpts0, depth0, depth1, T_0to1, K0, K1,
    src_camera_type="PINHOLE",
    tgt_camera_type="PINHOLE",
    depth_interpolation_mode="bilinear",
    relative_depth_error_threshold=0.05,
):
    """Warp kpts0 from I0 to I1 with mixed camera models.

    Supports all combinations: pinhole<->pinhole, fisheye<->fisheye,
    pinhole->fisheye, fisheye->pinhole.

    Args:
        kpts0: [N, L, 2] normalized coordinates in (-1, 1)
        depth0: [N, H, W] depth map (z-depth for pinhole, radial for fisheye)
        depth1: [N, H, W] depth map (z-depth for pinhole, radial for fisheye)
        T_0to1: [N, 3, 4] or [N, 4, 4] rigid transform from cam0 to cam1
        K0, K1: [N, 3, 3] intrinsics
        src_camera_type: "PINHOLE" or "FISHEYE_EQUIDISTANT"
        tgt_camera_type: "PINHOLE" or "FISHEYE_EQUIDISTANT"
    Returns:
        valid_mask: [N, L] boolean mask
        warped_kpts0: [N, L, 2] normalized coordinates in (-1, 1)
    """
    n, h, w = depth0.shape
    eps = 1e-8

    if depth_interpolation_mode == "combined":
        valid_bilinear, warp_bilinear = warp_kpts_general(
            kpts0, depth0, depth1, T_0to1, K0, K1,
            src_camera_type=src_camera_type, tgt_camera_type=tgt_camera_type,
            depth_interpolation_mode="bilinear",
            relative_depth_error_threshold=relative_depth_error_threshold,
        )
        valid_nearest, warp_nearest = warp_kpts_general(
            kpts0, depth0, depth1, T_0to1, K0, K1,
            src_camera_type=src_camera_type, tgt_camera_type=tgt_camera_type,
            depth_interpolation_mode="nearest-exact",
            relative_depth_error_threshold=relative_depth_error_threshold,
        )
        nearest_valid_bilinear_invalid = (~valid_bilinear) & valid_nearest
        warp = warp_bilinear.clone()
        warp[nearest_valid_bilinear_invalid] = warp_nearest[nearest_valid_bilinear_invalid]
        valid = valid_bilinear | valid_nearest
        return valid, warp

    kpts0_depth = F.grid_sample(
        depth0[:, None], kpts0[:, :, None],
        mode=depth_interpolation_mode, align_corners=False
    )[:, 0, :, 0]

    kpts0_pixel = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2),
        dim=-1
    )

    nonzero_mask = kpts0_depth > eps

    if src_camera_type == "FISHEYE_EQUIDISTANT":
        kpts0_cam = _unproject_fisheye_equidistant(kpts0_pixel, kpts0_depth, K0)
    else:
        kpts0_cam = _unproject_pinhole(kpts0_pixel, kpts0_depth, K0)

    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]

    if tgt_camera_type == "FISHEYE_EQUIDISTANT":
        w_kpts0_pixel, w_kpts0_depth_computed = _project_fisheye_equidistant(w_kpts0_cam, K1)
    else:
        w_kpts0_pixel, w_kpts0_depth_computed = _project_pinhole(w_kpts0_cam, K1)

    covisible_mask = (
        (w_kpts0_pixel[..., 0] >= 0) &
        (w_kpts0_pixel[..., 0] <= w - 1) &
        (w_kpts0_pixel[..., 1] >= 0) &
        (w_kpts0_pixel[..., 1] <= h - 1)
    )

    w_kpts0_norm = torch.stack(
        (2 * w_kpts0_pixel[..., 0] / w - 1,
         2 * w_kpts0_pixel[..., 1] / h - 1),
        dim=-1
    )

    w_kpts0_depth_sampled = F.grid_sample(
        depth1[:, None], w_kpts0_norm[:, :, None],
        mode=depth_interpolation_mode, align_corners=False
    )[:, 0, :, 0]

    safe_depth = torch.where(
        w_kpts0_depth_sampled > eps,
        w_kpts0_depth_sampled,
        torch.ones_like(w_kpts0_depth_sampled) * eps
    )
    relative_depth_error = torch.abs(
        (w_kpts0_depth_sampled - w_kpts0_depth_computed) / safe_depth
    )
    consistent_mask = relative_depth_error < relative_depth_error_threshold

    valid_mask = nonzero_mask & covisible_mask & consistent_mask
    return valid_mask, w_kpts0_norm


def get_gt_warp_general(
    depth1, depth2, T_1to2, K1, K2,
    src_camera_type="PINHOLE", tgt_camera_type="PINHOLE",
    depth_interpolation_mode='bilinear',
    relative_depth_error_threshold=0.05,
    H=None, W=None,
):
    """Compute GT warp between two views with potentially different camera models.

    Args:
        depth1, depth2: [B, H_orig, W_orig] depth maps
        T_1to2: [B, 4, 4] transform from cam1 to cam2
        K1, K2: [B, 3, 3] intrinsics
        src_camera_type: camera model of view 1
        tgt_camera_type: camera model of view 2
    Returns:
        x2: [B, H, W, 2] normalized warp coordinates
        prob: [B, H, W] validity mask
    """
    if H is None:
        B, H, W = depth1.shape
    else:
        B = depth1.shape[0]
    with torch.no_grad():
        x1_n = torch.meshgrid(
            *[
                torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=depth1.device)
                for n in (B, H, W)
            ],
            indexing='ij'
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        mask, x2 = warp_kpts_general(
            x1_n.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
            src_camera_type=src_camera_type,
            tgt_camera_type=tgt_camera_type,
            depth_interpolation_mode=depth_interpolation_mode,
            relative_depth_error_threshold=relative_depth_error_threshold,
        )
        prob = mask.float().reshape(B, H, W)
        x2 = x2.reshape(B, H, W, 2)
        return x2, prob

def get_gt_warp_fisheye_equidistant(depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode = 'bilinear', relative_depth_error_threshold = 0.05, H = None, W = None):
    
    if H is None:
        B,H,W = depth1.shape
    else:
        B = depth1.shape[0]
    with torch.no_grad():
        x1_n = torch.meshgrid(
            *[
                torch.linspace(
                    -1 + 1 / n, 1 - 1 / n, n, device=depth1.device
                )
                for n in (B, H, W)
            ],
            indexing = 'ij'
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        mask, x2 = warp_kpts_fisheye_equidistant(
            x1_n.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
            depth_interpolation_mode = depth_interpolation_mode,
            relative_depth_error_threshold = relative_depth_error_threshold,
        )
        prob = mask.float().reshape(B, H, W)
        x2 = x2.reshape(B, H, W, 2)
        return x2, prob

@torch.no_grad()
def warp_kpts_fisheye_equidistant(
    kpts0, 
    depth0, 
    depth1, 
    T_0to1, 
    K0, 
    K1, 
    smooth_mask=False, 
    return_relative_depth_error=False, 
    depth_interpolation_mode="bilinear", 
    relative_depth_error_threshold=0.05
):
    """
    Warp kpts0 from I0 to I1 using equidistant fisheye model.
    Depth consistency uses radial depth (Euclidean distance from camera center).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - normalized coordinates in (-1, 1)
        depth0 (torch.Tensor): [N, H, W] - radial depth map (Euclidean distance)
        depth1 (torch.Tensor): [N, H, W]
        T_0to1 (torch.Tensor): [N, 3, 4] - rigid transform from cam0 to cam1
        K0, K1 (torch.Tensor): [N, 3, 3] - intrinsics (only fx, fy, cx, cy used)
        smooth_mask (bool): return soft mask via exponential weighting
        depth_interpolation_mode (str): "bilinear", "nearest-exact", or "combined"
        relative_depth_error_threshold (float): threshold for depth consistency
    
    Returns:
        valid_mask (torch.Tensor): [N, L] - boolean mask of valid warped points
        warped_kpts0 (torch.Tensor): [N, L, 2] - normalized warped coordinates in (-1,1)
    """
    n, h, w = depth0.shape

    # Handle combined interpolation mode recursively (same as original)
    if depth_interpolation_mode == "combined":
        if smooth_mask:
            raise NotImplementedError("Combined mode not supported with smooth_mask")
        valid_bilinear, warp_bilinear = warp_kpts_fisheye_equidistant(
            kpts0, depth0, depth1, T_0to1, K0, K1,
            smooth_mask=False,
            return_relative_depth_error=False,
            depth_interpolation_mode="bilinear",
            relative_depth_error_threshold=relative_depth_error_threshold
        )
        valid_nearest, warp_nearest = warp_kpts_fisheye_equidistant(
            kpts0, depth0, depth1, T_0to1, K0, K1,
            smooth_mask=False,
            return_relative_depth_error=False,
            depth_interpolation_mode="nearest-exact",
            relative_depth_error_threshold=relative_depth_error_threshold
        )
        nearest_valid_bilinear_invalid = (~valid_bilinear) & valid_nearest
        warp = warp_bilinear.clone()
        warp[nearest_valid_bilinear_invalid] = warp_nearest[nearest_valid_bilinear_invalid]
        valid = valid_bilinear | valid_nearest
        return valid, warp

    # Step 1: Sample depth at kpts0 (using normalized coords for grid_sample)
    kpts0_depth = F.grid_sample(
        depth0[:, None], 
        kpts0[:, :, None], 
        mode=depth_interpolation_mode, 
        align_corners=False
    )[:, 0, :, 0]  # [N, L]
    
    # Convert normalized coords (-1,1) -> pixel coords (0.5, w-0.5)
    kpts0_pixel = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), 
        dim=-1
    )  # [N, L, 2]
    
    # Non-zero depth mask
    nonzero_mask = kpts0_depth > 1e-8  # Avoid zero depth

    # Step 2: Fisheye unprojection (pixel + depth -> 3D point in cam0)
    # Extract intrinsics
    fx0 = K0[:, 0, 0].unsqueeze(1)  # [N, 1]
    fy0 = K0[:, 1, 1].unsqueeze(1)
    cx0 = K0[:, 0, 2].unsqueeze(1)
    cy0 = K0[:, 1, 2].unsqueeze(1)
    
    # Normalize pixel coords to unit plane
    x_cam = (kpts0_pixel[..., 0] - cx0) / fx0  # [N, L]
    y_cam = (kpts0_pixel[..., 1] - cy0) / fy0
    
    # Radial distance = theta (equidistant model)
    r = torch.sqrt(x_cam**2 + y_cam**2)  # [N, L]
    
    # Compute direction vector components
    sin_r = torch.sin(r)
    cos_r = torch.cos(r)
    
    # Avoid division by zero at center
    eps = 1e-8
    mask_valid = r > eps
    factor = torch.ones_like(r)
    factor[mask_valid] = sin_r[mask_valid] / r[mask_valid]
    
    # Direction vector (unit length)
    dir_x = x_cam * factor
    dir_y = y_cam * factor
    dir_z = cos_r
    
    # Normalize for numerical stability (optional but recommended)
    norm = torch.sqrt(dir_x**2 + dir_y**2 + dir_z**2).clamp(min=eps)
    dir_x = dir_x / norm
    dir_y = dir_y / norm
    dir_z = dir_z / norm
    
    # 3D point in cam0 coordinates (radial depth * direction)
    kpts0_cam = torch.stack([
        dir_x * kpts0_depth,
        dir_y * kpts0_depth,
        dir_z * kpts0_depth
    ], dim=1)  # [N, 3, L]

    # Step 3: Rigid transform to cam1 coordinates
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # [N, 3, L]
    
    # Compute radial depth in cam1 (Euclidean norm)
    w_kpts0_radial_depth = torch.norm(w_kpts0_cam, dim=1)  # [N, L]

    # Step 4: Fisheye projection (3D point in cam1 -> pixel coords)
    # Normalize to unit direction vector
    dir_w = w_kpts0_cam / (w_kpts0_radial_depth.unsqueeze(1) + eps)  # [N, 3, L]
    dir_x_w = dir_w[:, 0, :]  # [N, L]
    dir_y_w = dir_w[:, 1, :]
    dir_z_w = dir_w[:, 2, :]
    
    # Compute theta = angle from optical axis
    theta = torch.acos(dir_z_w.clamp(-1.0 + eps, 1.0 - eps))  # [N, L]
    
    # Compute projection factor: theta / sin(theta)
    sin_theta = torch.sin(theta)
    factor_proj = torch.ones_like(theta)
    mask_theta = theta > eps
    factor_proj[mask_theta] = theta[mask_theta] / sin_theta[mask_theta]
    
    # Normalized image coordinates (before scaling by focal length)
    x_prime = dir_x_w * factor_proj  # [N, L]
    y_prime = dir_y_w * factor_proj
    
    # Convert to pixel coordinates using K1
    fx1 = K1[:, 0, 0].unsqueeze(1)
    fy1 = K1[:, 1, 1].unsqueeze(1)
    cx1 = K1[:, 0, 2].unsqueeze(1)
    cy1 = K1[:, 1, 2].unsqueeze(1)
    
    w_kpts0_pixel = torch.stack([
        cx1 + fx1 * x_prime,
        cy1 + fy1 * y_prime
    ], dim=-1)  # [N, L, 2]

    # Step 5: Covisibility check (within image boundaries)
    covisible_mask = (
        (w_kpts0_pixel[..., 0] >= 0) &
        (w_kpts0_pixel[..., 0] <= w - 1) &
        (w_kpts0_pixel[..., 1] >= 0) &
        (w_kpts0_pixel[..., 1] <= h - 1)
    )  # [N, L]

    # Convert warped pixels back to normalized coordinates (-1, 1)
    w_kpts0_norm = torch.stack(
        (2 * w_kpts0_pixel[..., 0] / w - 1, 
         2 * w_kpts0_pixel[..., 1] / h - 1),
        dim=-1
    )

    # Step 6: Sample depth from depth1 at warped locations
    w_kpts0_depth_sampled = F.grid_sample(
        depth1[:, None], 
        w_kpts0_norm[:, :, None], 
        mode=depth_interpolation_mode, 
        align_corners=False
    )[:, 0, :, 0]  # [N, L]

    # Step 7: Depth consistency check using radial depth
    # Avoid division by zero in relative error
    safe_depth = torch.where(
        w_kpts0_depth_sampled > eps, 
        w_kpts0_depth_sampled, 
        torch.ones_like(w_kpts0_depth_sampled) * eps
    )
    relative_depth_error = torch.abs(
        (w_kpts0_depth_sampled - w_kpts0_radial_depth) / safe_depth
    )
    
    if not smooth_mask:
        consistent_mask = relative_depth_error < relative_depth_error_threshold
    else:
        consistent_mask = torch.exp(-relative_depth_error / smooth_mask)
    
    # Final valid mask: non-zero depth + covisible + depth consistent
    valid_mask = nonzero_mask & covisible_mask & consistent_mask

    if return_relative_depth_error:
        return relative_depth_error, w_kpts0_norm
    else:
        return valid_mask, w_kpts0_norm


def _solve_blender_poly_for_r(theta, k0, k1, k2, k3, k4, n_iter=10):
    """Solve |k0 + k1*r + k2*r^2 + k3*r^3 + k4*r^4| = theta for r via Newton's method.

    The unprojection model uses theta = |poly(r)|. When poly(r) is negative
    (k1 < 0), we negate all coefficients so that Newton solves the equivalent
    positive-polynomial equation.

    Args:
        theta: [N, L] target angle values
        k0..k4: [N, 1] polynomial coefficients (broadcastable to theta)
        n_iter: number of Newton iterations
    Returns:
        r: [N, L] solved physical radius values (clamped >= 0)
    """
    eps = 1e-8

    sign = torch.where(k1 >= 0, torch.ones_like(k1), -torch.ones_like(k1))
    k0, k1, k2, k3, k4 = k0 * sign, k1 * sign, k2 * sign, k3 * sign, k4 * sign

    r = (theta - k0) / (k1 + eps)
    r = r.clamp(min=0.0)

    for _ in range(n_iter):
        f = k0 + k1 * r + k2 * r**2 + k3 * r**3 + k4 * r**4 - theta
        f_prime = k1 + 2 * k2 * r + 3 * k3 * r**2 + 4 * k4 * r**3
        r = r - f / (f_prime + eps)
        r = r.clamp(min=0.0)

    return r


@torch.no_grad()
def warp_kpts_fisheye_blender(
    kpts0,
    depth0,
    depth1,
    T_0to1,
    K0,
    K1,
    k_coeffs0,
    k_coeffs1,
    sensor_size0,
    sensor_size1,
    crop_offset0=None,
    crop_offset1=None,
    smooth_mask=False,
    return_relative_depth_error=False,
    depth_interpolation_mode="bilinear",
    relative_depth_error_threshold=0.05,
    newton_iters=10,
):
    """Warp kpts0 from I0 to I1 using Blender polynomial fisheye model.

    Polynomial model: theta = k0 + k1*r + k2*r^2 + k3*r^3 + k4*r^4
    where r is the physical radius on the sensor (mm) and theta is the angle
    from the optical axis. Depth is radial (Euclidean distance from camera center).

    Args:
        kpts0: [N, L, 2] normalized coordinates in (-1, 1)
        depth0: [N, H, W] radial depth map
        depth1: [N, H, W] radial depth map
        T_0to1: [N, 3, 4] or [N, 4, 4] rigid transform from cam0 to cam1
        K0, K1: [N, 3, 3] intrinsics
        k_coeffs0: [N, 5] polynomial coefficients for source camera
        k_coeffs1: [N, 5] polynomial coefficients for target camera
        sensor_size0: [N, 2] sensor size (sensor_w, sensor_h) in mm for source
        sensor_size1: [N, 2] sensor size (sensor_w, sensor_h) in mm for target
        crop_offset0: [N, 2] or None, (offset_x, offset_y) in pixels for source
        crop_offset1: [N, 2] or None, (offset_x, offset_y) in pixels for target
        smooth_mask: bool or float
        return_relative_depth_error: bool
        depth_interpolation_mode: "bilinear", "nearest-exact", or "combined"
        relative_depth_error_threshold: float
        newton_iters: int, Newton iterations for polynomial inversion

    Returns:
        valid_mask: [N, L] boolean mask of valid warped points
        warped_kpts0: [N, L, 2] normalized coordinates in (-1, 1)
    """
    n, h, w = depth0.shape
    eps = 1e-8

    if depth_interpolation_mode == "combined":
        if smooth_mask:
            raise NotImplementedError("Combined mode not supported with smooth_mask")
        valid_bilinear, warp_bilinear = warp_kpts_fisheye_blender(
            kpts0, depth0, depth1, T_0to1, K0, K1,
            k_coeffs0, k_coeffs1, sensor_size0, sensor_size1,
            crop_offset0, crop_offset1,
            smooth_mask=False, return_relative_depth_error=False,
            depth_interpolation_mode="bilinear",
            relative_depth_error_threshold=relative_depth_error_threshold,
            newton_iters=newton_iters,
        )
        valid_nearest, warp_nearest = warp_kpts_fisheye_blender(
            kpts0, depth0, depth1, T_0to1, K0, K1,
            k_coeffs0, k_coeffs1, sensor_size0, sensor_size1,
            crop_offset0, crop_offset1,
            smooth_mask=False, return_relative_depth_error=False,
            depth_interpolation_mode="nearest-exact",
            relative_depth_error_threshold=relative_depth_error_threshold,
            newton_iters=newton_iters,
        )
        nearest_valid_bilinear_invalid = (~valid_bilinear) & valid_nearest
        warp = warp_bilinear.clone()
        warp[nearest_valid_bilinear_invalid] = warp_nearest[nearest_valid_bilinear_invalid]
        valid = valid_bilinear | valid_nearest
        return valid, warp

    # Step 1: Sample depth at kpts0
    kpts0_depth = F.grid_sample(
        depth0[:, None], kpts0[:, :, None],
        mode=depth_interpolation_mode, align_corners=False
    )[:, 0, :, 0]  # [N, L]

    kpts0_pixel = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2),
        dim=-1
    )  # [N, L, 2]

    nonzero_mask = kpts0_depth > eps

    # Step 2: Blender polynomial unprojection (pixel -> 3D)
    cx0 = K0[:, 0, 2].unsqueeze(1)  # [N, 1]
    cy0 = K0[:, 1, 2].unsqueeze(1)

    offset_x0 = crop_offset0[:, 0].unsqueeze(1) if crop_offset0 is not None else 0.0
    offset_y0 = crop_offset0[:, 1].unsqueeze(1) if crop_offset0 is not None else 0.0
    pixel_size_w0 = sensor_size0[:, 0].unsqueeze(1) / (w + 2 * offset_x0)
    pixel_size_h0 = sensor_size0[:, 1].unsqueeze(1) / (h + 2 * offset_y0)

    x_phys = (kpts0_pixel[..., 0] - cx0) * pixel_size_w0  # [N, L]
    y_phys = (kpts0_pixel[..., 1] - cy0) * pixel_size_h0
    r_phys = torch.sqrt(x_phys**2 + y_phys**2)

    k0_s = k_coeffs0[:, 0].unsqueeze(1)
    k1_s = k_coeffs0[:, 1].unsqueeze(1)
    k2_s = k_coeffs0[:, 2].unsqueeze(1)
    k3_s = k_coeffs0[:, 3].unsqueeze(1)
    k4_s = k_coeffs0[:, 4].unsqueeze(1)

    theta = torch.abs(
        k0_s + k1_s * r_phys + k2_s * r_phys**2 + k3_s * r_phys**3 + k4_s * r_phys**4
    )  # [N, L]
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    mask_valid_r = r_phys > eps
    factor = torch.zeros_like(r_phys)
    factor[mask_valid_r] = sin_theta[mask_valid_r] / r_phys[mask_valid_r]

    dir_x = x_phys * factor
    dir_y = y_phys * factor
    dir_z = cos_theta

    dir_x[~mask_valid_r] = 0.0
    dir_y[~mask_valid_r] = 0.0
    dir_z[~mask_valid_r] = 1.0

    norm = torch.sqrt(dir_x**2 + dir_y**2 + dir_z**2).clamp(min=eps)
    dir_x = dir_x / norm
    dir_y = dir_y / norm
    dir_z = dir_z / norm

    kpts0_cam = torch.stack([
        dir_x * kpts0_depth,
        dir_y * kpts0_depth,
        dir_z * kpts0_depth
    ], dim=1)  # [N, 3, L]

    # Step 3: Transform to target camera
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # [N, 3, L]
    w_kpts0_radial_depth = torch.norm(w_kpts0_cam, dim=1)  # [N, L]

    # Step 4: Blender polynomial projection (3D -> pixel in target)
    dir_w = w_kpts0_cam / (w_kpts0_radial_depth.unsqueeze(1) + eps)
    dir_x_w = dir_w[:, 0, :]
    dir_y_w = dir_w[:, 1, :]
    dir_z_w = dir_w[:, 2, :]

    theta_tgt = torch.acos(dir_z_w.clamp(-1.0 + eps, 1.0 - eps))  # [N, L]
    sin_theta_tgt = torch.sin(theta_tgt)

    k0_t = k_coeffs1[:, 0].unsqueeze(1)
    k1_t = k_coeffs1[:, 1].unsqueeze(1)
    k2_t = k_coeffs1[:, 2].unsqueeze(1)
    k3_t = k_coeffs1[:, 3].unsqueeze(1)
    k4_t = k_coeffs1[:, 4].unsqueeze(1)

    r_phys_tgt = _solve_blender_poly_for_r(
        theta_tgt, k0_t, k1_t, k2_t, k3_t, k4_t, n_iter=newton_iters
    )  # [N, L]

    mask_sin = sin_theta_tgt > eps
    proj_factor = torch.zeros_like(sin_theta_tgt)
    proj_factor[mask_sin] = r_phys_tgt[mask_sin] / sin_theta_tgt[mask_sin]

    x_phys_tgt = dir_x_w * proj_factor
    y_phys_tgt = dir_y_w * proj_factor

    cx1 = K1[:, 0, 2].unsqueeze(1)
    cy1 = K1[:, 1, 2].unsqueeze(1)

    offset_x1 = crop_offset1[:, 0].unsqueeze(1) if crop_offset1 is not None else 0.0
    offset_y1 = crop_offset1[:, 1].unsqueeze(1) if crop_offset1 is not None else 0.0
    pixel_size_w1 = sensor_size1[:, 0].unsqueeze(1) / (w + 2 * offset_x1)
    pixel_size_h1 = sensor_size1[:, 1].unsqueeze(1) / (h + 2 * offset_y1)

    w_kpts0_pixel = torch.stack([
        x_phys_tgt / pixel_size_w1 + cx1,
        y_phys_tgt / pixel_size_h1 + cy1
    ], dim=-1)  # [N, L, 2]

    # Step 5: Covisibility check
    covisible_mask = (
        (w_kpts0_pixel[..., 0] >= 0) &
        (w_kpts0_pixel[..., 0] <= w - 1) &
        (w_kpts0_pixel[..., 1] >= 0) &
        (w_kpts0_pixel[..., 1] <= h - 1)
    )

    w_kpts0_norm = torch.stack(
        (2 * w_kpts0_pixel[..., 0] / w - 1,
         2 * w_kpts0_pixel[..., 1] / h - 1),
        dim=-1
    )

    # Step 6: Depth consistency
    w_kpts0_depth_sampled = F.grid_sample(
        depth1[:, None], w_kpts0_norm[:, :, None],
        mode=depth_interpolation_mode, align_corners=False
    )[:, 0, :, 0]

    safe_depth = torch.where(
        w_kpts0_depth_sampled > eps,
        w_kpts0_depth_sampled,
        torch.ones_like(w_kpts0_depth_sampled) * eps
    )
    relative_depth_error = torch.abs(
        (w_kpts0_depth_sampled - w_kpts0_radial_depth) / safe_depth
    )

    if not smooth_mask:
        consistent_mask = relative_depth_error < relative_depth_error_threshold
    else:
        consistent_mask = torch.exp(-relative_depth_error / smooth_mask)

    valid_mask = nonzero_mask & covisible_mask & consistent_mask

    if return_relative_depth_error:
        return relative_depth_error, w_kpts0_norm
    else:
        return valid_mask, w_kpts0_norm


def get_gt_warp_fisheye_blender(
    depth1, depth2, T_1to2, K1, K2,
    k_coeffs1, k_coeffs2,
    sensor_size1, sensor_size2,
    crop_offset1=None, crop_offset2=None,
    depth_interpolation_mode='bilinear',
    relative_depth_error_threshold=0.05,
    H=None, W=None,
    newton_iters=10,
):
    """Compute GT warp between two Blender polynomial fisheye views.

    Args:
        depth1, depth2: [B, H_orig, W_orig] radial depth maps
        T_1to2: [B, 4, 4] transform from cam1 to cam2
        K1, K2: [B, 3, 3] intrinsics
        k_coeffs1, k_coeffs2: [B, 5] polynomial coefficients
        sensor_size1, sensor_size2: [B, 2] sensor size (sensor_w, sensor_h) in mm
        crop_offset1, crop_offset2: [B, 2] or None
        H, W: output grid resolution (defaults to depth1 spatial dims)
    Returns:
        x2: [B, H, W, 2] normalized warp coordinates
        prob: [B, H, W] validity mask
    """
    if H is None:
        B, H, W = depth1.shape
    else:
        B = depth1.shape[0]
    with torch.no_grad():
        x1_n = torch.meshgrid(
            *[
                torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=depth1.device)
                for n in (B, H, W)
            ],
            indexing='ij'
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        mask, x2 = warp_kpts_fisheye_blender(
            x1_n.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
            k_coeffs1.double(),
            k_coeffs2.double(),
            sensor_size1.double(),
            sensor_size2.double(),
            crop_offset1.double() if crop_offset1 is not None else None,
            crop_offset2.double() if crop_offset2 is not None else None,
            depth_interpolation_mode=depth_interpolation_mode,
            relative_depth_error_threshold=relative_depth_error_threshold,
            newton_iters=newton_iters,
        )
        prob = mask.float().reshape(B, H, W)
        x2 = x2.reshape(B, H, W, 2)
        return x2, prob


@torch.no_grad()
def compute_fisheye_to_pinhole_grid(K_fisheye, K_pinhole, H, W, device='cpu'):
    """Compute remap grid for converting a fisheye image to a pinhole image.

    For each pixel in the target pinhole image, computes the corresponding
    source coordinate in the fisheye image using the equidistant model.

    Args:
        K_fisheye: [3, 3] fisheye intrinsics (equidistant model)
        K_pinhole: [3, 3] pinhole intrinsics
        H, W: target image dimensions
        device: torch device

    Returns:
        grid: [1, H, W, 2] normalized coords (-1,1) for grid_sample
        cos_theta: [1, 1, H, W] factor to convert radial depth -> z-depth
        valid_mask: [1, 1, H, W] bool, True where fisheye source pixel is in bounds
    """
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float64) + 0.5,
        torch.arange(W, device=device, dtype=torch.float64) + 0.5,
        indexing='ij',
    )

    fx_pin = K_pinhole[0, 0].double()
    fy_pin = K_pinhole[1, 1].double()
    cx_pin = K_pinhole[0, 2].double()
    cy_pin = K_pinhole[1, 2].double()

    dx = (u_coords - cx_pin) / fx_pin
    dy = (v_coords - cy_pin) / fy_pin
    dz = torch.ones_like(dx)

    norm = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    dx, dy, dz = dx / norm, dy / norm, dz / norm

    cos_theta = dz

    eps = 1e-8
    theta = torch.acos(dz.clamp(-1.0 + eps, 1.0 - eps))
    sin_theta = torch.sin(theta)
    factor_proj = torch.ones_like(theta)
    mask_theta = theta > eps
    factor_proj[mask_theta] = theta[mask_theta] / sin_theta[mask_theta]

    x_prime = dx * factor_proj
    y_prime = dy * factor_proj

    fx_fish = K_fisheye[0, 0].double()
    fy_fish = K_fisheye[1, 1].double()
    cx_fish = K_fisheye[0, 2].double()
    cy_fish = K_fisheye[1, 2].double()

    u_fish = cx_fish + fx_fish * x_prime
    v_fish = cy_fish + fy_fish * y_prime

    valid_mask = (
        (u_fish >= 0) & (u_fish <= W - 1) &
        (v_fish >= 0) & (v_fish <= H - 1)
    )

    u_norm = 2 * u_fish / W - 1
    v_norm = 2 * v_fish / H - 1

    grid = torch.stack([u_norm, v_norm], dim=-1).float().unsqueeze(0)
    cos_theta = cos_theta.float().unsqueeze(0).unsqueeze(0)
    valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
    return grid, cos_theta, valid_mask


def remap_fisheye_to_pinhole(tensor, grid, mode='bilinear', depth_conversion_factor=None):
    """Apply a precomputed remap grid to warp a fisheye tensor to pinhole.

    Args:
        tensor: [B, C, H, W] input tensor in fisheye space
        grid: [1, H, W, 2] normalized grid from compute_fisheye_to_pinhole_grid
        mode: 'bilinear' or 'nearest'
        depth_conversion_factor: optional [1, 1, H, W] cos_theta to convert
            radial depth to z-depth after remapping

    Returns:
        [B, C, H, W] remapped tensor in pinhole space
    """
    remapped = F.grid_sample(
        tensor,
        grid.expand(tensor.shape[0], -1, -1, -1),
        mode=mode,
        padding_mode='zeros',
        align_corners=False,
    )
    if depth_conversion_factor is not None:
        remapped = remapped * depth_conversion_factor
    return remapped


@torch.no_grad()
def unproject_fisheye_to_pointmap(depth, K_fisheye):
    """Unproject fisheye radial depth to a dense 3D pointmap in camera coords.

    Args:
        depth: [B, 1, H, W] radial depth map (Euclidean distance)
        K_fisheye: [3, 3] fisheye intrinsics (equidistant model)

    Returns:
        pointmap: [B, 3, H, W] 3D points in camera coordinates
    """
    H, W = depth.shape[-2:]
    eps = 1e-8

    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype) + 0.5,
        torch.arange(W, device=depth.device, dtype=depth.dtype) + 0.5,
        indexing='ij',
    )

    fx = K_fisheye[0, 0].to(depth.dtype)
    fy = K_fisheye[1, 1].to(depth.dtype)
    cx = K_fisheye[0, 2].to(depth.dtype)
    cy = K_fisheye[1, 2].to(depth.dtype)

    x_cam = (u_coords - cx) / fx
    y_cam = (v_coords - cy) / fy

    r = torch.sqrt(x_cam ** 2 + y_cam ** 2)
    sin_r = torch.sin(r)
    cos_r = torch.cos(r)

    mask_valid = r > eps
    factor = torch.ones_like(r)
    factor[mask_valid] = sin_r[mask_valid] / r[mask_valid]

    dir_x = x_cam * factor
    dir_y = y_cam * factor
    dir_z = cos_r

    norm = torch.sqrt(dir_x ** 2 + dir_y ** 2 + dir_z ** 2).clamp(min=eps)
    dir_x = dir_x / norm
    dir_y = dir_y / norm
    dir_z = dir_z / norm

    directions = torch.stack([dir_x, dir_y, dir_z], dim=0)  # [3, H, W]
    pointmap = depth * directions.unsqueeze(0)  # [B, 3, H, W]
    return pointmap


def remap_fisheye_depth_to_pinhole(depth, grid, K_fisheye):
    """Remap fisheye radial depth to pinhole z-depth via pointmap reprojection.

    1. Unproject fisheye radial depth → 3D pointmap in camera coords
    2. Remap pointmap with grid_sample (bilinear on XYZ is geometrically correct)
    3. Extract Z component as pinhole z-depth

    Args:
        depth: [B, 1, H, W] fisheye radial depth
        grid: [1, H, W, 2] normalized grid from compute_fisheye_to_pinhole_grid
        K_fisheye: [3, 3] fisheye intrinsics

    Returns:
        z_depth: [B, 1, H, W] pinhole z-depth
    """
    pointmap = unproject_fisheye_to_pointmap(depth, K_fisheye)  # [B, 3, H, W]
    pointmap_pinhole = F.grid_sample(
        pointmap,
        grid.expand(depth.shape[0], -1, -1, -1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )
    z_depth = pointmap_pinhole[:, 2:3, :, :]  # [B, 1, H, W]
    return z_depth