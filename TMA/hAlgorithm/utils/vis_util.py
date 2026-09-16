import math

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import torch
from PIL import Image


def visualize_batch(batch):
    # 从batch字典中获取图像、深度图、点云图以及对应的掩码信息
    image = batch["image"]  # 获取与输入相关的RGB图像数据
    gt_depth = batch["depth"]  # 获取真实深度图数据，形状为 [1, H, W]
    gt_pointmap = batch["pointmap"]  # 获取真实点云图数据，形状为 [3, H, W]
    depth_mask = batch["depth_mask"]  # 获取深度图的有效区域掩码，形状为 [1, H, W]
    sparse_pointmap = batch["sparse_pointmap"]  # 获取稀疏点云图数据，形状为 [3, H, W]
    sparse_pointmap_mask = batch[
        "sparse_pointmap_mask"
    ]  # 获取稀疏点云图的有效区域掩码，形状为 [1, H, W]

    # 可视化真实点云图（gt pointmap）
    # 使用深度掩码选择有效的点，并将这些点的位置和颜色信息转换为numpy数组
    points = (
        gt_pointmap[..., depth_mask.squeeze(0)].permute(1, 0).view(-1, 3).numpy()
    )  # 提取有效点位置
    colors = (
        image[..., depth_mask.squeeze(0)].permute(1, 0).view(-1, 3).numpy()
    )  # 提取对应点的颜色信息

    # 创建一个open3d点云对象并设置其点的位置和颜色
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)  # 设置点云的坐标
    point_cloud.colors = o3d.utility.Vector3dVector(colors)  # 设置点云的颜色

    # 可视化稀疏点云图
    # 使用稀疏点云掩码选择有效的点，并将这些点的位置信息转换为numpy数组
    sparse_points = (
        sparse_pointmap[..., sparse_pointmap_mask.squeeze(0)].permute(1, 0).numpy()
    )  # 提取有效点位置

    # 创建另一个open3d点云对象并仅设置其点的位置（没有颜色信息）
    sparse_point_cloud = o3d.geometry.PointCloud()
    sparse_point_cloud.points = o3d.utility.Vector3dVector(sparse_points)  # 设置稀疏点云的坐标

    # 使用open3d的图形界面显示两个点云对象
    o3d.visualization.draw_geometries([point_cloud, sparse_point_cloud])  # 显示稠密点云和稀疏点云


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    """
    Colorize depth maps.

    Args:
        depth_map (torch.Tensor or np.ndarray): Depth map, with shape [H, W] or [B, H, W].
        min_depth (float): Minimum valid depth value in the depth map.
        max_depth (float): Maximum valid depth value in the depth map.
        cmap (str, optional): Colormap to use for colorization. Defaults to "Spectral".
        valid_mask (torch.Tensor or np.ndarray, optional): Valid pixel mask, with shape [H, W] or [B, H, W].

    Returns:
        torch.Tensor or np.ndarray: Colorized depth map, with shape [3, H, W] or [B, 3, H, W].
    """

    # Ensure the input depth map has at least 2 dimensions (height and width)
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    # Convert the depth map to a NumPy array if it's a PyTorch tensor
    if isinstance(depth_map, torch.Tensor):
        depth = (
            depth_map.detach().squeeze().numpy()
        )  # Remove any singleton dimensions and convert to NumPy
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()  # Remove any singleton dimensions and create a copy

    # Reshape the depth map to ensure it has at least 3 dimensions [B, H, W]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]  # Add a batch dimension if missing

    # Get the colormap from Matplotlib
    cm = matplotlib.colormaps[cmap]

    # Normalize the depth values to the range [0, 1]
    depth_normalized = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)

    # Apply the colormap to the normalized depth values
    # The result is a 4D array with shape [B, H, W, 4] (including alpha channel), but we only need the RGB channels
    img_colored_np = cm(depth_normalized, bytes=False)[:, :, :, 0:3]

    # Move the color channel to the second dimension to match PyTorch's [C, H, W] format
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)  # Shape becomes [B, 3, H, W]

    # Apply the valid mask if provided
    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()  # Convert to NumPy if it's a PyTorch tensor
        valid_mask = valid_mask.squeeze()  # Remove any singleton dimensions

        # Ensure the valid mask has the correct shape [B, 1, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]

        # Repeat the valid mask across the color channels
        valid_mask = np.repeat(valid_mask, 3, axis=1)

        # Set invalid pixels to black (RGB = [0, 0, 0])
        img_colored_np[~valid_mask] = 0

    # Convert back to PyTorch tensor if the input was a PyTorch tensor
    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    img_colored = chw2hwc(img_colored.squeeze())

    img_colored = (img_colored * 255).astype(np.uint8)
    img_colored = Image.fromarray(img_colored)

    return img_colored


def grid_images(save_path, paths=None, images=None, col=4):
    assert paths is not None or images is not None

    if images is None:
        images = [cv2.imread(path) for path in paths]

    h, w = images[0].shape[:2]
    nums = len(images)
    raw = math.ceil(nums / col)
    if len(images[0].shape) == 3:
        merge_image = np.zeros([h * raw, w * col, 3], dtype=np.uint8)
    else:
        merge_image = np.zeros([h * raw, w * col], dtype=np.uint8)

    for ni in range(nums):
        ci = ni % col
        ri = ni // col
        merge_image[ri * h : (ri + 1) * h, ci * w : (ci + 1) * w] = images[ni]

    if save_path is not None:
        cv2.imwrite(save_path, merge_image)

    return merge_image


def apply_color_map(x, color_map):
    """
    Fast color mapping.
    x: Tensor of shape [B, 1, H, W], values should be in range [0, 1]
    color_map: Name of the color map
    """
    # 获取颜色映射表（256 色）
    cmap = matplotlib.cm.get_cmap(color_map, 256)

    if isinstance(x, torch.Tensor):
        color_lut = torch.tensor(cmap.colors, dtype=torch.float32, device=x.device)[:, :3]  # [256, 3]

        # 归一化到 0-255 并转换为索引
        x = (x * 255).clamp(0, 255).long()

        # 通过查找表获取颜色值
        image = color_lut[x.squeeze(1)]  # [B, H, W, 3]

        return image.permute(0, 3, 1, 2)  # 转换回 [B, 3, H, W]
    else:
        color_lut = cmap.colors[:, :3]  # [256, 3]

        # 归一化到 0-255 并转换为索引
        x = (x * 255).clip(0, 255).astype(np.uint8)

        # 通过查找表获取颜色值
        return color_lut[x]
