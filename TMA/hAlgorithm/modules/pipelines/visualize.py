import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from moviepy.editor import ImageSequenceClip
from PIL import Image

from hAlgorithm.utils import colorize_depth_maps
import cv2

# Helper function to save depth maps
def save_depth_map(
    depth, out_dir, file_prefix, data_idx, cmap="turbo", min_val=None, max_val=None, info=False
):
    """Save a normalized depth map as a colored image."""
    if min_val is None:
        min_val = depth.min()
    if max_val is None:
        max_val = depth.max()
    depth_norm = (depth - min_val) / (max_val - min_val + 1e-9)
    depth_colored = colorize_depth_maps(depth_norm, 0, 1, cmap=cmap)
    save_path = os.path.join(out_dir, f"{file_prefix}_{data_idx:06d}.jpg")
    depth_colored.save(save_path)
    if info:
        logging.info(f"visualize save: {save_path}")
    return save_path

def save_normal(
    normal, out_dir, file_prefix, data_idx,  mask = None, cmap="turbo", info=False
):
    """Save a normalized normal map as a colored image."""
    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    normal_colored = normal * [0.5, -0.5, -0.5] + 0.5
    normal_colored = (normal_colored.clip(0, 1) * 255).astype(np.uint8)
    save_path = os.path.join(out_dir, f"{file_prefix}_{data_idx:06d}.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(normal_colored, cv2.COLOR_RGB2BGR))
    if info:
        logging.info(f"visualize save: {save_path}")
    return save_path


def save_image(rgb, out_dir, file_prefix, data_idx, info=False):
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255).astype(np.uint8)
    rgb = Image.fromarray(rgb)
    if data_idx is not None:
        save_path = os.path.join(out_dir, f"{file_prefix}_{data_idx:06d}.jpg")
    else:
        save_path = os.path.join(out_dir, f"{file_prefix}.jpg")
    rgb.save(save_path)
    if info:
        logging.info(f"visualize save: {save_path}")
    return save_path


def save_video(video, out_dir, file_prefix, data_idx, info=False):
    if data_idx is not None:
        save_path = os.path.join(out_dir, f"{file_prefix}_{data_idx:06d}.mp4")
    else:
        save_path = os.path.join(out_dir, f"{file_prefix}.mp4")
    video = ImageSequenceClip(list(video), fps=24)
    video.write_videofile(save_path)
    if info:
        logging.info(f"visualize save: {save_path}")


# Helper function to save point clouds
def save_point_cloud(points, colors, out_dir, file_prefix, data_idx=None, info=False):
    """Save point cloud data to a .ply file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    if data_idx is not None:
        save_path = os.path.join(out_dir, f"{file_prefix}_{data_idx:06d}.ply")
    else:
        save_path = os.path.join(out_dir, f"{file_prefix}.ply")
    o3d.io.write_point_cloud(save_path, pcd)
    if info:
        logging.info(f"visualize save: {save_path}")
    return save_path


# Helper function to save gradient map
def save_gradient(depth_grad, out_dir, data_idx, info=False):
    """Save a gradient map as a colored image."""
    depth_grad_u = depth_grad[..., 0]
    depth_grad_v = depth_grad[..., 1]

    depth_grad_u = colorize_depth_maps(
        depth_grad_u, depth_grad_u.min(), depth_grad_u.max(), cmap="turbo"
    )
    depth_grad_v = colorize_depth_maps(
        depth_grad_v, depth_grad_v.min(), depth_grad_v.max(), cmap="turbo"
    )

    save_path = os.path.join(out_dir, f"gradu_{data_idx:06d}.jpg")
    depth_grad_u.save(save_path)
    if info:
        logging.info(f"visualize save: {save_path}")

    save_path = os.path.join(out_dir, f"gradv_{data_idx:06d}.jpg")
    depth_grad_v.save(save_path)
    if info:
        logging.info(f"visualize save: {save_path}")


# Helper function to revalue error map
def revalue(array, lower, upper, start, scale):
    """Rescale values in the array between [lower, upper] to [start, start+scale]."""
    mask = (array >= lower) & (array < upper)
    array[mask] = start + scale * (array[mask] - lower) / (upper - lower)
    return array


# Helper function to generate error bar
def generate_error_bar(w, cmap="jet"):
    """Generate a color bar for the error map."""
    error_bar_height = 50
    breakpoints = [0, 0.1, 0.5, 1.25, 2, 4]
    points = [0, 0.25, 0.38, 0.66, 0.83, 0.95]
    num_bins = [
        0,
        w // 8,
        w // 8,
        w // 4,
        w // 4,
        w - (w // 4 + w // 4 + w // 8 + w // 8 + w // 8),
    ]
    acc_num_bins = np.cumsum(num_bins)

    error_bar = np.array([])
    for i in range(1, len(breakpoints)):
        error_bar = np.concatenate((error_bar, np.linspace(points[i - 1], points[i], num_bins[i])))

    error_bar = np.repeat(error_bar, error_bar_height).reshape(w, error_bar_height).transpose(1, 0)
    error_bar_map = plt.cm.get_cmap(cmap)(error_bar)[:, :, :3]

    return error_bar_map, breakpoints, acc_num_bins


# Function to convert depth error to color map
def depth_err_to_colorbar(est, gt=None, with_bar=False, cmap="jet"):
    """
    Convert depth error map to a colorized error map.

    Parameters:
    - est: Predicted depth map (2D array).
    - gt: Ground truth depth map (2D array, optional).
    - with_bar: Whether to include a color bar (bool, default=False).
    - cmap: Color map name (str, default='jet').

    Returns:
    - Colored error map (3D array, shape [H, W, 3]).
    """
    if gt is None:
        gt = np.zeros_like(est)
        valid = est > 0
        max_depth = est.max()
    else:
        if est.shape != gt.shape:
            raise ValueError("Predicted and ground truth depth maps must have the same shape.")
        valid = gt > 0
        max_depth = gt.max()

    # Compute absolute error
    error_map = np.abs(est - gt) * valid
    h, w = error_map.shape
    max_value = error_map.max()

    # Define breakpoints based on max depth
    if max_depth < 30:
        breakpoints = np.array([0, 0.1, 0.5, 1.25, 2, 4, max(10, max_value)])
    else:
        breakpoints = np.array([0, 0.1, 0.5, 1.25, 2, 4, max(90, max_value)])

    # Normalize error map values
    points = np.array([0, 0.25, 0.38, 0.66, 0.83, 0.95, 1])
    for i in range(1, len(breakpoints)):
        error_map = revalue(
            error_map,
            breakpoints[i - 1],
            breakpoints[i],
            points[i - 1],
            points[i] - points[i - 1],
        )

    # Map normalized error to colors
    error_map = plt.cm.get_cmap(cmap)(error_map)[:, :, :3]
    error_map = error_map * valid[:, :, None]  # Mark invalid pixels as black

    if not with_bar:
        return error_map

    # Generate color bar
    error_bar_map, breakpoints, acc_num_bins = generate_error_bar(w, cmap=cmap)
    combined_map = np.concatenate((error_map, error_bar_map), axis=0)

    # Add labels for color bar
    fig, ax = plt.subplots(figsize=(w / 100, 0.5))
    ax.set_xticks(acc_num_bins)
    ax.set_xticklabels([str(f) for f in breakpoints])
    ax.axis("off")
    plt.close(fig)

    return combined_map


# Helper function to save error maps as images
def save_error(pred_depthmap, gt_depthmap, out_dir, file_prefix, data_idx, info=False):
    """
    Save error map visualization as an image.

    Parameters:
    - pred_depthmap: Predicted depth map (2D array).
    - gt_depthmap: Ground truth depth map (2D array).
    - out_dir: Output directory path.
    - file_prefix: File name prefix.
    - data_idx: Index of the data sample.
    """
    try:
        # Ensure depth maps have the same shape
        if pred_depthmap.shape != gt_depthmap.shape:
            raise ValueError("Predicted and ground truth depth maps must have the same shape.")

        # Generate error map
        error_map = depth_err_to_colorbar(pred_depthmap, gt_depthmap, with_bar=False)

        # Colorize ground truth depth map
        gt_depthmap_color = colorize_depth_maps(
            gt_depthmap, gt_depthmap.min(), gt_depthmap.max(), cmap="turbo"
        )
        gt_depthmap_color = np.array(gt_depthmap_color) / 255

        # Combine error map and ground truth depth map
        combined_map = np.concatenate((error_map, gt_depthmap_color), axis=0)

        # Save the combined map as an image
        save_path = os.path.join(out_dir, f"{file_prefix}_{data_idx:06d}.jpg")
        plt.imsave(save_path, combined_map)
        if info:
            logging.info(f"visualize save: {save_path}")
    except Exception as e:
        logging.error(f"Error generating error map: {e}")
