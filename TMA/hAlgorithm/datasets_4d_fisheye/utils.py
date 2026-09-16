"""Fisheye polynomial projection and unprojection functions.

Blender Polynomial Fisheye Model
--------------------------------
The model relates the incident angle θ to the physical radius r on the sensor:

    θ = |k0 + k1*r + k2*r² + k3*r³ + k4*r⁴|

where:
    - r is the physical radius on the sensor (in mm)
    - θ is the incident angle from the optical axis (in radians)
    - k0, k1, k2, k3, k4 are the polynomial coefficients

To convert between pixel coordinates and physical coordinates:
    r_physical = r_pixel * pixel_size
    pixel_size = sensor_size / image_size

Depth Convention
----------------
Blender outputs **Z-depth** (distance along optical axis), NOT radial depth.
This is important for correct unprojection.
"""

import numpy as np
from typing import Tuple, Optional


def pixel_to_physical_radius(
    u: np.ndarray,
    v: np.ndarray,
    cx: float,
    cy: float,
    sensor_width: float,
    sensor_height: float,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert pixel coordinates to physical radius on sensor.

    Args:
        u, v: Pixel coordinates.
        cx, cy: Principal point in pixels.
        sensor_width, sensor_height: Sensor size in mm.
        image_width, image_height: Image size in pixels.

    Returns:
        r_phys: Physical radius in mm.
        x_phys: Physical x offset in mm.
        y_phys: Physical y offset in mm.
    """
    pixel_size_x = sensor_width / image_width
    pixel_size_y = sensor_height / image_height

    x_phys = (u - cx) * pixel_size_x
    y_phys = (v - cy) * pixel_size_y
    r_phys = np.sqrt(x_phys**2 + y_phys**2)

    return r_phys, x_phys, y_phys


def physical_to_pixel(
    x_phys: np.ndarray,
    y_phys: np.ndarray,
    cx: float,
    cy: float,
    sensor_width: float,
    sensor_height: float,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert physical sensor coordinates to pixel coordinates.

    Args:
        x_phys, y_phys: Physical coordinates in mm.
        cx, cy: Principal point in pixels.
        sensor_width, sensor_height: Sensor size in mm.
        image_width, image_height: Image size in pixels.

    Returns:
        u, v: Pixel coordinates.
    """
    pixel_size_x = sensor_width / image_width
    pixel_size_y = sensor_height / image_height

    u = x_phys / pixel_size_x + cx
    v = y_phys / pixel_size_y + cy

    return u, v


def poly_to_theta(r_phys: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Compute incident angle θ from physical radius using polynomial model.

    Blender polynomial fisheye model:
        θ = -(k0 + k1*r + k2*r² + k3*r³ + k4*r⁴)

    Note: The negative sign is important! With typical k1 < 0, this gives
    positive θ for positive r.

    Args:
        r_phys: Physical radius in mm.
        k: Polynomial coefficients [k0, k1, k2, k3, k4].

    Returns:
        theta: Incident angle in radians (always non-negative).
    """
    theta = -(
        k[0] + k[1] * r_phys + k[2] * r_phys**2 + k[3] * r_phys**3 + k[4] * r_phys**4
    )
    # Ensure non-negative (should be naturally non-negative with correct k values)
    theta = np.maximum(theta, 0.0)
    return theta


def solve_poly_for_r(
    theta: np.ndarray,
    k: np.ndarray,
    n_iter: int = 10,
) -> np.ndarray:
    """Solve polynomial for physical radius r given incident angle θ.

    Blender model: θ = -(k0 + k1*r + k2*r² + k3*r³ + k4*r⁴)
    Rearranged: k0 + k1*r + k2*r² + k3*r³ + k4*r⁴ = -θ

    Uses Newton's method to find r such that poly(r) = -θ.

    Args:
        theta: Target incident angle in radians.
        k: Polynomial coefficients [k0, k1, k2, k3, k4].
        n_iter: Number of Newton iterations.

    Returns:
        r_phys: Physical radius in mm.
    """
    eps = 1e-8

    # Initial guess: linear approximation
    # k0 + k1*r ≈ -θ  =>  r ≈ (-θ - k0) / k1
    r = (-theta - k[0]) / (k[1] + eps)
    r = np.maximum(r, 0.0)

    # Newton's method: find r where f(r) = poly(r) + θ = 0
    for _ in range(n_iter):
        poly_val = (
            k[0]
            + k[1] * r
            + k[2] * r**2
            + k[3] * r**3
            + k[4] * r**4
        )
        f = poly_val + theta  # Should be 0 when poly(r) = -θ
        f_prime = (
            k[1]
            + 2 * k[2] * r
            + 3 * k[3] * r**2
            + 4 * k[4] * r**3
        )
        r = r - f / (f_prime + eps)
        r = np.maximum(r, 0.0)

    return r


def fisheye_unproject(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    cx: float,
    cy: float,
    k: np.ndarray,
    sensor_width: float,
    sensor_height: float,
    image_width: int,
    image_height: int,
    depth_is_along_ray: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject fisheye pixel coordinates to 3D camera coordinates.

    Args:
        u, v: Pixel coordinates.
        depth: Depth values. By default, this is Z-depth (along optical axis),
               which is what Blender outputs. Set depth_is_along_ray=True if
               depth is radial (Euclidean distance from camera center).
        cx, cy: Principal point in pixels.
        k: Polynomial coefficients [k0, k1, k2, k3, k4].
        sensor_width, sensor_height: Sensor size in mm.
        image_width, image_height: Image size in pixels.
        depth_is_along_ray: If True, depth is radial distance. If False (default),
                            depth is Z-depth along optical axis.

    Returns:
        X, Y, Z: 3D coordinates in camera frame.
    """
    eps = 1e-8

    r_phys, x_phys, y_phys = pixel_to_physical_radius(
        u, v, cx, cy, sensor_width, sensor_height, image_width, image_height
    )

    theta = poly_to_theta(r_phys, k)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # For points at center (r_phys ≈ 0), direction is along optical axis
    mask_valid = r_phys > eps

    if depth_is_along_ray:
        # Depth is radial distance (along ray)
        # Direction vector: (sin(θ) * x_phys/r_phys, sin(θ) * y_phys/r_phys, cos(θ))
        factor = np.zeros_like(r_phys)
        factor[mask_valid] = sin_theta[mask_valid] / r_phys[mask_valid]

        dir_x = x_phys * factor
        dir_y = y_phys * factor
        dir_z = cos_theta

        dir_x[~mask_valid] = 0.0
        dir_y[~mask_valid] = 0.0
        dir_z[~mask_valid] = 1.0

        # Normalize direction
        norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        norm = np.maximum(norm, eps)
        dir_x = dir_x / norm
        dir_y = dir_y / norm
        dir_z = dir_z / norm

        X = dir_x * depth
        Y = dir_y * depth
        Z = dir_z * depth
    else:
        # Depth is Z-depth (along optical axis) - Blender convention
        # Given Z = depth, we need to find X and Y
        # tan(θ) = sqrt(X² + Y²) / Z
        # So sqrt(X² + Y²) = Z * tan(θ)
        Z = depth.copy()

        # Clamp theta to avoid tan(θ) explosion near 90 degrees
        # For θ > 80°, tan(θ) grows rapidly (tan(80°)≈5.7, tan(85°)≈11.4)
        # Typical fisheye FOV is ~180°, so max θ should be ~90°
        # But we clamp to 80° to avoid numerical issues at extreme angles
        max_theta = np.radians(80.0)
        theta_clamped = np.minimum(theta, max_theta)

        # Mark points with θ >= 90° as invalid (they would be behind the camera)
        invalid_theta = theta >= np.radians(90.0)

        tan_theta = np.tan(theta_clamped)
        xy_dist = Z * tan_theta  # sqrt(X² + Y²)

        # Compute azimuthal angle from physical coordinates
        phi = np.arctan2(y_phys, x_phys + eps)

        X = xy_dist * np.cos(phi)
        Y = xy_dist * np.sin(phi)

        # Handle center points and invalid points
        X[~mask_valid] = 0.0
        Y[~mask_valid] = 0.0

        # Set invalid theta points to NaN so they can be filtered later
        X[invalid_theta] = np.nan
        Y[invalid_theta] = np.nan
        Z[invalid_theta] = np.nan

    return X, Y, Z


def fisheye_project(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    cx: float,
    cy: float,
    k: np.ndarray,
    sensor_width: float,
    sensor_height: float,
    image_width: int,
    image_height: int,
    newton_iters: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D camera coordinates to fisheye pixel coordinates.

    Args:
        X, Y, Z: 3D coordinates in camera frame.
        cx, cy: Principal point in pixels.
        k: Polynomial coefficients [k0, k1, k2, k3, k4].
        sensor_width, sensor_height: Sensor size in mm.
        image_width, image_height: Image size in pixels.
        newton_iters: Number of Newton iterations for polynomial inversion.

    Returns:
        u, v: Pixel coordinates.
        z_depth: Z-depth (distance along optical axis), matching Blender convention.
    """
    eps = 1e-8

    radial_depth = np.sqrt(X**2 + Y**2 + Z**2)

    dir_x = X / (radial_depth + eps)
    dir_y = Y / (radial_depth + eps)
    dir_z = Z / (radial_depth + eps)

    theta = np.arccos(np.clip(dir_z, -1.0 + eps, 1.0 - eps))

    r_phys = solve_poly_for_r(theta, k, n_iter=newton_iters)

    sin_theta = np.sin(theta)
    mask_sin = sin_theta > eps
    proj_factor = np.zeros_like(sin_theta)
    proj_factor[mask_sin] = r_phys[mask_sin] / sin_theta[mask_sin]

    x_phys = dir_x * proj_factor
    y_phys = dir_y * proj_factor

    u, v = physical_to_pixel(
        x_phys, y_phys, cx, cy, sensor_width, sensor_height, image_width, image_height
    )

    # Return Z-depth to match Blender convention
    return u, v, Z


def detect_fisheye_valid_radius(
    rgb: np.ndarray,
    cx: float,
    cy: float,
    brightness_threshold: int = 5,
) -> float:
    """Auto-detect fisheye valid region radius from image content.

    Finds the radius of the valid (non-black) circular region by analyzing
    the image brightness.

    Args:
        rgb: RGB image array [H, W, 3].
        cx, cy: Estimated center of the fisheye circle.
        brightness_threshold: Minimum brightness to consider as valid.

    Returns:
        Detected radius in pixels.
    """
    import cv2

    H, W = rgb.shape[:2]

    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb

    non_zero_mask = gray > brightness_threshold
    non_zero_coords = np.where(non_zero_mask)

    if len(non_zero_coords[0]) == 0:
        return min(cx, cy, W - cx, H - cy)

    y_min, y_max = non_zero_coords[0].min(), non_zero_coords[0].max()
    x_min, x_max = non_zero_coords[1].min(), non_zero_coords[1].max()

    radius_x = (x_max - x_min) / 2
    radius_y = (y_max - y_min) / 2
    radius = min(radius_x, radius_y)

    max_radius = min(cx, cy, W - cx, H - cy)
    radius = min(radius, max_radius)

    return radius


def generate_fisheye_boundary_mask(
    height: int,
    width: int,
    cx: float,
    cy: float,
    radius: float,
) -> np.ndarray:
    """Generate a circular boundary mask for fisheye images.

    Args:
        height, width: Image dimensions.
        cx, cy: Center of the circle.
        radius: Radius of the valid circular region.

    Returns:
        Boolean mask where True indicates valid (inside circle) pixels.
    """
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return dist_from_center <= radius
