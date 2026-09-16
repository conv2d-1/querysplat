# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from https://github.com/facebookresearch/vggsfm
# and https://github.com/facebookresearch/co-tracker/tree/main


from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: Union[int, Tuple[int, int]], return_grid=False
) -> torch.Tensor:
    """
    This function initializes a grid and generates a 2D positional embedding using sine and cosine functions.
    It is a wrapper of get_2d_sincos_pos_embed_from_grid.
    Args:
    - embed_dim: The embedding dimension.
    - grid_size: The grid size.
    Returns:
    - pos_embed: The generated 2D positional embedding.
    """
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float)
    grid_w = torch.arange(grid_size_w, dtype=torch.float)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if return_grid:
        return (
            pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(0, 3, 1, 2),
            grid,
        )
    return pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(0, 3, 1, 2)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - grid: The grid to generate the embedding from.

    Returns:
    - emb: The generated 2D positional embedding.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=2)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()


def get_2d_embedding(xy: torch.Tensor, C: int, cat_coords: bool = True) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from given coordinates using sine and cosine functions.

    Args:
    - xy: The coordinates to generate the embedding from.
    - C: The size of the embedding.
    - cat_coords: A flag to indicate whether to concatenate the original coordinates to the embedding.

    Returns:
    - pe: The generated 2D positional embedding.
    """
    B, N, D = xy.shape
    assert D == 2

    x = xy[:, :, 0:1]
    y = xy[:, :, 1:2]
    div_term = (
        torch.arange(0, C, 2, device=xy.device, dtype=torch.float32) * (1000.0 / C)
    ).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe = torch.cat([pe_x, pe_y], dim=2)  # (B, N, C*3)
    if cat_coords:
        pe = torch.cat([xy, pe], dim=2)  # (B, N, C*3+3)
    return pe


def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        Tensor: sampled points.
    """
    coords = coords.detach().clone()
    ############################################################
    # IMPORTANT:
    coords = coords.to(input.device).to(input.dtype)
    ############################################################

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        scale = torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)],
            device=coords.device,
            dtype=coords.dtype,
        )
    else:
        scale = torch.tensor(
            [2 / size for size in reversed(sizes)], device=coords.device, dtype=coords.dtype
        )

    coords.mul_(scale)  # coords = coords * scale
    coords.sub_(1)  # coords = coords - 1

    return F.grid_sample(input, coords, align_corners=align_corners, padding_mode=padding_mode)


def sample_features4d(input, coords):
    r"""Sample spatial features

    `sample_features4d(input, coords)` samples the spatial features
    :attr:`input` represented by a 4D tensor :math:`(B, C, H, W)`.

    The field is sampled at coordinates :attr:`coords` using bilinear
    interpolation. :attr:`coords` is assumed to be of shape :math:`(B, R,
    2)`, where each sample has the format :math:`(x_i, y_i)`. This uses the
    same convention as :func:`bilinear_sampler` with `align_corners=True`.

    The output tensor has one feature per point, and has shape :math:`(B,
    R, C)`.

    Args:
        input (Tensor): spatial features.
        coords (Tensor): points.

    Returns:
        Tensor: sampled features.
    """

    B, _, _, _ = input.shape

    # B R 2 -> B R 1 2
    coords = coords.unsqueeze(2)

    # B C R 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 1, 3).view(
        B, -1, feats.shape[1] * feats.shape[3]
    )  # B C R 1 -> B R C

def project_query_to_points3d(
    query_points: torch.Tensor,
    local_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    query_frames: Optional[torch.Tensor] = None,
    extrinsics: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    将 2D query points 投影为 3D 点云（世界坐标系）。

    Args:
        query_points: [B, N, 2], (u, v) 像素坐标
        local_depth: [B, T, 1, H, W] 或 [B, 1, H, W]（若 T=1）
        intrinsics: [B, T, 3, 3]
        query_frames: [B, N] long tensor, 指定每个点来自哪个 t；None 表示 T=1，所有 t=0
        extrinsics: [B, T, 4, 4]，相机到世界的变换；None 表示使用单位阵

    Returns:
        points_world: [B, N, 3]，3D 点在世界坐标系中
    """
    B, N, _ = query_points.shape

    # 处理 T 维度
    if query_frames is None:
        # 默认 T=1，所有点来自 t=0
        query_frames = torch.zeros(B, N, dtype=torch.long, device=query_points.device)
        T_use = 1
    else:
        query_frames = query_frames.long()
        T_use = query_frames.max().item() + 1

    # 调整 intrinsics 形状到 [B, T_use, 3, 3]
    if intrinsics.shape[1] == 1 and T_use > 1:
        intrinsics = intrinsics.repeat(1, T_use, 1, 1)
    elif intrinsics.shape[1] >= T_use:
        intrinsics = intrinsics[:, :T_use]
    elif intrinsics.shape[1] < T_use:
        raise ValueError(f"intrinsics T dim {intrinsics.shape[1]} 不匹配 query_frames 所需 {T_use}")

    # 处理 extrinsics
    if extrinsics is None:
        # 构造单位阵 [B, T_use, 4, 4]
        extrinsics = torch.eye(4, device=query_points.device)
        extrinsics = extrinsics.reshape(1, 1, 4, 4).expand(B, T_use, 4, 4)
    else:
        if extrinsics.shape[1] == 1 and T_use > 1:
            extrinsics = extrinsics.repeat(1, T_use, 1, 1)
        elif extrinsics.shape[1] >= T_use:
            extrinsics = extrinsics[:, :T_use]
        elif extrinsics.shape[1] < T_use:
            raise ValueError(f"extrinsics T dim {extrinsics.shape[1]} 不匹配")

    # 确保 local_depth 是 [B, T, 1, H, W]
    if local_depth.dim() == 4:
        # 假设 T=1
        local_depth = local_depth.unsqueeze(1)  # -> [B, 1, 1, H, W]
    elif local_depth.dim() != 5:
        raise ValueError(f"local_depth 期望 4D 或 5D，得到 {local_depth.dim()}D")

    _, T_depth, _, H, W = local_depth.shape

    if T_depth == 1 and T_use > 1:
        local_depth = local_depth.repeat(1, T_use, 1, 1, 1)
    elif T_depth >= T_use:
        local_depth = local_depth[:, :T_use]
    elif T_depth < T_use:
        raise ValueError(f"local_depth T={T_depth} 与 query_frames 所需 T={T_use} 不匹配")

    # ================================
    # Step 1: 提取每个 (b,n) 在 t=query_frames[b,n] 时刻的深度
    # ================================
    # query_points: [B, N, 2] -> 归一化到 [-1,1] 用于 grid_sample
    # 但我们直接索引更高效

    # 将 (u, v) 转为整数坐标（假设是四舍五入）
    u = query_points[..., 0] + 0.5  # [B, N]
    v = query_points[..., 1] + 0.5  # [B, N]

    # 确保在范围内
    u = u.clamp(0, W - 1).long()  # [B, N]
    v = v.clamp(0, H - 1).long()  # [B, N]

    # 扩展 query_frames 到 [B, N]，每个 (b,n) 有 t_idx
    t_idx = query_frames  # [B, N]

    # 创建索引：对每个 (b,n)，取 depth[b, t_idx[b,n], 0, v[b,n], u[b,n]]
    # 使用高级索引
    b_idx = torch.arange(B, device=query_points.device).view(-1, 1)  # [B, 1]
    n_idx = torch.arange(N, device=query_points.device).view(1, -1)  # [1, N]

    # 提取深度: [B, N]
    depths = local_depth[b_idx, t_idx, 0, v, u]  # [B, N]

    # ================================
    # Step 2: 将 (u, v, depth) 转为相机坐标系下的 3D 点
    # ================================
    # 使用内参：x = (u - cx) * d / fx, y = (v - cy) * d / fy
    fx = intrinsics[..., 0, 0]  # [B, T_use]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]

    # 取对应 t 的内参
    K_fx = fx[b_idx, t_idx]  # [B, N]
    K_fy = fy[b_idx, t_idx]
    K_cx = cx[b_idx, t_idx]
    K_cy = cy[b_idx, t_idx]

    # 相机坐标
    z_cam = depths
    x_cam = (u.float() - K_cx) * z_cam / K_fx
    y_cam = (v.float() - K_cy) * z_cam / K_fy

    # 合并为 [B, N, 3]
    points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [B, N, 3]
    
    # ================================
    # Step 3: 使用齐次坐标转换到世界坐标系
    # ================================
    # 构造齐次坐标 [B, N, 4]
    ones = torch.ones(B, N, 1, device=points_cam.device)
    points_cam_h = torch.cat([points_cam, ones], dim=-1)  # [B, N, 4]

    # 获取对应 t 的 extrinsics: [B, N, 4, 4]
    # extrinsics: [B, T_use, 4, 4] -> 选择 t_idx[b,n]
    # 使用高级索引：[B, N, 4, 4]
    T_world2cam = extrinsics[b_idx, t_idx]  # [B, N, 4, 4]

    # 计算逆矩阵：T_cam2world = inv(T_world2cam)
    # 注意：如果 extrinsics 是从世界到相机的变换，则其逆就是从相机到世界
    T_cam2world = torch.inverse(T_world2cam)  # [B, N, 4, 4]

    # 齐次坐标变换：P_world_h = T_cam2world @ P_cam_h
    # points_cam_h: [B, N, 4] -> [B, N, 4, 1]
    points_cam_h_expanded = points_cam_h.unsqueeze(-1)  # [B, N, 4, 1]
    points_world_h = torch.matmul(T_cam2world, points_cam_h_expanded)  # [B, N, 4, 1]

    # 去掉最后两维，取前3维
    points_world = points_world_h.squeeze(-1)[..., :3]  # [B, N, 3]

    return points_world

def project_points3d_to_frames(
    pts3d: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: Optional[torch.Tensor] = None,
    image_size: Union[tuple, list, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将世界坐标系下的 3D 点投影到多个相机视图，得到 2D 像素坐标和可见性 mask。

    Args:
        pts3d: [B, N, 3], 世界坐标系下的 3D 点
        intrinsics: [B, T, 3, 3], 相机内参
        extrinsics: [B, T, 4, 4], 世界到相机的变换矩阵；None 表示单位阵
        image_size: (H, W) 用于判断是否在图像范围内；如果为 None，则不检查范围

    Returns:
        points_uv: [B, T, N, 2], 像素坐标 (u, v)
        vis_mask:  [B, T, N], bool tensor，表示该点是否可见
    """
    B, N, _ = pts3d.shape
    T = intrinsics.shape[1]
    device = pts3d.device

    # 获取图像尺寸
    if image_size is not None:
        H, W = image_size
    else:
        H = W = None  # 不做范围检查

    # ================================
    # 处理 extrinsics：默认为单位阵
    # ================================
    if extrinsics is None:
        extrinsics = torch.eye(4, device=device)
        extrinsics = extrinsics.reshape(1, 1, 4, 4).expand(B, T, 4, 4)
    else:
        if extrinsics.shape[1] == 1 and T > 1:
            extrinsics = extrinsics.repeat(1, T, 1, 1)
        elif extrinsics.shape[1] != T:
            raise ValueError(f"extrinsics T dim {extrinsics.shape[1]} 与 intrinsics 不匹配 T={T}")

    # ================================
    # Step 1: 转为齐次坐标并变换到相机坐标系
    # ================================
    pts3d_h = torch.cat([pts3d, torch.ones(B, N, 1, device=device)], dim=-1)  # [B, N, 4]
    pts3d_h = pts3d_h.unsqueeze(1)  # [B, 1, N, 4]
    extrinsics_expanded = extrinsics.unsqueeze(2)  # [B, T, 1, 4, 4]

    pts_cam_h = torch.matmul(extrinsics_expanded, pts3d_h.unsqueeze(-1)).squeeze(-1)  # [B, T, N, 4]
    pts_cam = pts_cam_h[..., :3]  # [B, T, N, 3]
    z_cam = pts_cam[..., 2]  # [B, T, N]

    # ================================
    # Step 2: 投影到图像平面
    # ================================
    fx = intrinsics[..., 0, 0].unsqueeze(-1)  # [B, T, 1]
    fy = intrinsics[..., 1, 1].unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1)

    eps = 1e-6
    x_proj = pts_cam[..., 0] / (z_cam + eps)
    y_proj = pts_cam[..., 1] / (z_cam + eps)

    u = fx * x_proj + cx  # [B, T, N]
    v = fy * y_proj + cy

    points_uv = torch.stack([u, v], dim=-1)  # [B, T, N, 2]

    # ================================
    # Step 3: 构建可见性 mask
    # ================================
    vis_mask = torch.ones(B, T, N, dtype=torch.bool, device=device)

    # 条件 1: 在相机前方
    vis_mask = vis_mask & (z_cam > 0)  # [B, T, N]

    # 条件 2: 在图像范围内（如果提供了 H, W）
    if H is not None and W is not None:
        vis_mask = vis_mask & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    return points_uv, vis_mask

def get_tracks_from_query(
    query_points,
    local_depth,
    intrinsics,
    query_frames=None,
    extrinsics=None,
):
    pts3d_world = project_query_to_points3d(query_points, local_depth, intrinsics, query_frames, extrinsics)
    tracks, vis_mask = project_points3d_to_frames(pts3d_world, intrinsics, extrinsics, image_size=local_depth.shape[-2:])
    B, T, N, _ = tracks.shape
    query_expanded = query_points.unsqueeze(1).expand(B, T, N, 2)
    tracks = torch.where(
        vis_mask.unsqueeze(-1),    # [B, T, N, 1] 作为条件
        tracks,                     # 条件为 True：保留原 track
        query_expanded             # 条件为 False：使用 query_points
    )
    return tracks, vis_mask
