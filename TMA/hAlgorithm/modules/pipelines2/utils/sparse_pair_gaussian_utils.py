"""Sparse pair-query utilities for Dynamic 4D Gaussian Splatting.

Builds Gaussians from WFM pair outputs (``warp3d``, ``warp3d_delta``).  Geometry
from the motion head lives in the **reference-camera / relative world frame**
(same convention as ``pair_tgt_trajs_3d`` / WorldTrack ``warp3d``).  Training,
validation, and infer visualization all rasterize via gsplat
(:func:`render_sparse_gaussians_at_frame`).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def find_pair_index(pair_idx: Sequence[Tuple[int, int]], src: int, tgt: int) -> Optional[int]:
    for i, (s, t) in enumerate(pair_idx):
        if int(s) == int(src) and int(t) == int(tgt):
            return i
    return None


def compute_gaussian_query_shape(
    input_width: int,
    input_height: int,
    gaussian_query_scale: float = 1.0,
    patch_size: int = 14,
) -> Tuple[int, int]:
    """Derive dense Gaussian query grid size from encoder input size and scale.

    UV coordinates remain normalized in ``[0, 1]``; only the number of samples
    along each axis grows.  Dimensions are rounded to multiples of ``patch_size``
    (same convention as ``ResizeKeepRatio``).
    """
    in_w = int(input_width)
    in_h = int(input_height)
    scale = float(gaussian_query_scale)
    if scale <= 1.0 + 1e-6:
        return in_w, in_h

    out_w = max(patch_size, int(round(in_w * scale / patch_size)) * patch_size)
    out_h = max(patch_size, int(round(in_h * scale / patch_size)) * patch_size)
    return out_w, out_h


def scale_bnv_intrinsics(
    intrinsics: Optional[Tensor],
    x_scale: float,
    y_scale: float,
) -> Optional[Tensor]:
    """Scale pinhole intrinsics when render resolution differs from encoder input."""
    if intrinsics is None:
        return None
    k = intrinsics.clone()
    k[..., 0, :] = k[..., 0, :] * float(x_scale)
    k[..., 1, :] = k[..., 1, :] * float(y_scale)
    return k


def resolve_gaussian_render_resolution(
    image: Tensor,
    intrinsics: Optional[Tensor],
    target_local_depth: Optional[Tensor],
    target_depth_mask: Optional[Tensor],
    gaussian_query,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], int, int]:
    """Upscale supervision tensors to match ``gaussian_query`` grid when denser than input."""
    batch_size, num_views, _, input_h, input_w = image.shape
    render_h, render_w = input_h, input_w
    if (
        gaussian_query is not None
        and getattr(gaussian_query, "full_uv", False)
        and gaussian_query.width is not None
        and gaussian_query.height is not None
    ):
        render_w = int(gaussian_query.width)
        render_h = int(gaussian_query.height)

    if render_h == input_h and render_w == input_w:
        return image, intrinsics, target_local_depth, target_depth_mask, render_h, render_w

    render_image = resize_bnv_geometry(
        image, batch_size, num_views, render_h, render_w, is_mask=False,
    )
    render_intrinsics = scale_bnv_intrinsics(
        intrinsics, render_w / input_w, render_h / input_h,
    )
    render_depth = resize_bnv_geometry(
        target_local_depth, batch_size, num_views, render_h, render_w, is_mask=False,
    )
    render_mask = resize_bnv_geometry(
        target_depth_mask, batch_size, num_views, render_h, render_w, is_mask=True,
    )
    return (
        render_image,
        render_intrinsics,
        render_depth,
        render_mask,
        render_h,
        render_w,
    )


def resolve_sparse_dgs_render_shape(
    meta_data: dict,
    results: Optional[dict],
    fallback_width: int,
    fallback_height: int,
) -> Tuple[int, int]:
    """Return ``(width, height)`` for sparse DGS rasterization."""
    gq = results.get("gaussian_query") if results else None
    if (
        gq is not None
        and getattr(gq, "full_uv", False)
        and gq.width is not None
        and gq.height is not None
    ):
        return int(gq.width), int(gq.height)
    return int(fallback_width), int(fallback_height)


def select_ref_frame(
    num_views: int,
    pair_idx: Sequence[Tuple[int, int]],
    random_src_frame: bool,
    training: bool,
    fixed_ref: int = 0,
) -> Optional[int]:
    """Pick a reference frame that has an identity pair ``(ref, ref)`` in ``pair_idx``."""
    if num_views <= 0:
        return None

    identity_refs = sorted({int(s) for s, t in pair_idx if int(s) == int(t)})
    if not identity_refs:
        return None

    if training and random_src_frame:
        ref = identity_refs[torch.randint(0, len(identity_refs), (1,)).item()]
    else:
        ref = fixed_ref if fixed_ref in identity_refs else identity_refs[0]
    return ref


def gather_pair_tensor(
    tensor: Tensor,
    pair_idx: Sequence[Tuple[int, int]],
    batch_size: int,
    pair_index: int,
) -> Tensor:
    """Gather one pair slice from a flattened ``[B*P, Q, ...]`` or ``[B*P, ...]`` tensor."""
    num_pair = len(pair_idx)
    if tensor.dim() == 3 and (tensor.shape[-1] == 3 or tensor.shape[1] == 3):
        tensor = flatten_pair_queries(tensor)
    elif tensor.shape[-1] == 3 and tensor.dim() == 4:
        tensor = flatten_pair_queries(tensor)
    rest = tensor.shape[1:]
    gathered = tensor.view(batch_size, num_pair, *rest)[:, pair_index]
    if gathered.shape[-1] == 3 or (gathered.dim() == 3 and gathered.shape[1] == 3):
        return squeeze_query_points(gathered)
    return gathered


def flatten_pair_queries(tensor: Tensor) -> Tensor:
    """Collapse optional singleton query dims to ``[B*P, Q, C]``."""
    if tensor.dim() == 3 and tensor.shape[-1] != 3 and tensor.shape[1] == 3:
        return tensor.transpose(1, 2).contiguous()
    if tensor.dim() == 4 and tensor.shape[-1] == 3:
        if tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        elif tensor.shape[2] == 1:
            tensor = tensor.squeeze(2)
    return tensor


def squeeze_query_points(tensor: Tensor) -> Tensor:
    """Normalize point tensors to ``[B, Q, 3]``."""
    if tensor.dim() == 4 and tensor.shape[-1] == 3:
        if tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        elif tensor.shape[2] == 1:
            tensor = tensor.squeeze(2)
        else:
            raise ValueError(f"Cannot squeeze query points from shape {tuple(tensor.shape)}")
    if tensor.dim() == 3:
        if tensor.shape[-1] == 3:
            return tensor
        if tensor.shape[1] == 3:
            return tensor.transpose(1, 2).contiguous()
    raise ValueError(f"Expected point tensor [B, Q, 3], got {tuple(tensor.shape)}")


def canonicalize_scene_flow(tensor: Tensor) -> Tensor:
    """Normalize per-frame displacements to ``[B, N, Q, 3]``."""
    if tensor.dim() == 5 and tensor.shape[-1] == 3:
        # Recover from bad scale broadcast: [B, N, N, Q, 3] or [B, N, 1, Q, 3].
        if tensor.shape[2] == 1:
            tensor = tensor.squeeze(2)
        elif tensor.shape[1] == tensor.shape[2]:
            tensor = tensor[:, :, 0]
        else:
            raise ValueError(f"Cannot canonicalize scene flow shape {tuple(tensor.shape)}")
    if tensor.dim() != 4:
        raise ValueError(f"Expected scene flow [B, N, Q, 3], got {tuple(tensor.shape)}")
    if tensor.shape[-1] == 3:
        return tensor
    if tensor.shape[-2] == 3:
        return tensor.permute(0, 1, 3, 2).contiguous()
    raise ValueError(f"Cannot canonicalize scene flow shape {tuple(tensor.shape)}")


def extract_view_scale(scale: Tensor, view_idx: int) -> Tensor:
    """Return per-view scale ``[B, 1, 1]`` for ``[B, Q, 3]`` tensors."""
    s = scale[:, view_idx]
    while s.dim() > 3:
        if s.shape[-1] == 1:
            s = s.squeeze(-1)
        else:
            raise ValueError(f"Unexpected scale shape {tuple(scale.shape)}")
    while s.dim() < 3:
        s = s.unsqueeze(-1)
    return s


def num_query_points(tensor: Tensor) -> int:
    if tensor.dim() in (3, 4):
        if tensor.shape[-1] == 3 or (tensor.dim() == 3 and tensor.shape[1] == 3):
            return squeeze_query_points(tensor).shape[1]
    if tensor.dim() == 2:
        return tensor.shape[0]
    raise ValueError(f"Cannot infer query count from shape {tuple(tensor.shape)}")


def align_sparse_query_geometry(
    means_ref: Tensor,
    displacements: Tensor,
    gs_opacity: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Ensure points, scene flow, and optional gs attrs share the same query count."""
    import logging

    means_ref = squeeze_query_points(means_ref)
    displacements = canonicalize_scene_flow(displacements)
    q = min(means_ref.shape[1], displacements.shape[2])
    if gs_opacity is not None and gs_opacity.dim() >= 2:
        q = min(q, gs_opacity.shape[1])
    if (
        means_ref.shape[1] != q
        or displacements.shape[2] != q
        or (gs_opacity is not None and gs_opacity.shape[1] != q)
    ):
        logging.getLogger(__name__).warning(
            "SparsePairDGS query mismatch: means=%d flow=%d gs=%s; using %d",
            means_ref.shape[1],
            displacements.shape[2],
            None if gs_opacity is None else gs_opacity.shape[1],
            q,
        )
    means_ref = means_ref[:, :q]
    displacements = displacements[:, :, :q]
    return means_ref, displacements


def compose_frame_means(
    means_ref: Tensor,
    displacements: Tensor,
    frame_idx: int,
) -> Tensor:
    """Add per-frame displacement to reference means as ``[B, Q, 3]``."""
    means_ref = squeeze_query_points(means_ref)
    displacements = canonicalize_scene_flow(displacements)
    q = min(means_ref.shape[1], displacements.shape[2])
    return means_ref[:, :q] + displacements[:, frame_idx, :q]


def align_query_feats(feats: Tensor, batch_size: int, num_queries: int) -> Tensor:
    """Ensure pair query features are ``[B, Q, C]``."""
    if feats.dim() == 4 and feats.shape[1] == 1:
        feats = feats.squeeze(1)
    if feats.dim() == 2:
        if feats.shape[0] == batch_size:
            feats = feats.unsqueeze(1)
        elif feats.shape[0] == num_queries:
            feats = feats.unsqueeze(0)
        else:
            raise ValueError(
                f"Cannot align query feats shape {tuple(feats.shape)} "
                f"to batch_size={batch_size}, num_queries={num_queries}"
            )
    if feats.dim() != 3:
        raise ValueError(f"Expected query feats to be 2D/3D, got {tuple(feats.shape)}")
    if feats.shape[0] != batch_size:
        raise ValueError(
            f"Query feats batch mismatch: feats={tuple(feats.shape)}, batch_size={batch_size}"
        )
    if feats.shape[1] == 1 and num_queries > 1:
        feats = feats.expand(-1, num_queries, -1)
    elif feats.shape[1] != num_queries:
        raise ValueError(
            f"Query feats/query count mismatch: feats={tuple(feats.shape)}, num_queries={num_queries}"
        )
    return feats


def ensure_bnv_cameras(
    tensor: Tensor,
    num_views: int,
    matrix_shape: tuple[int, int],
) -> Tensor:
    """Normalize camera tensors to ``[B, N, *matrix_shape]``."""
    if tensor.dim() == 4 and tensor.shape[-2:] == matrix_shape:
        return tensor
    if tensor.dim() == 3 and tensor.shape[-2:] == matrix_shape:
        if tensor.shape[0] == num_views:
            return tensor.unsqueeze(0)
        return tensor.unsqueeze(1)
    raise ValueError(
        f"Unexpected camera tensor shape {tuple(tensor.shape)}; "
        f"expected [B, N, {matrix_shape[0]}, {matrix_shape[1]}] with N={num_views}."
    )


def select_frame_camera(
    w2c: Tensor,
    intrinsics: Tensor,
    frame_idx: int,
    batch_idx: int = 0,
    num_views: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Return ``viewmats`` ``[1, 4, 4]`` and ``Ks`` ``[1, 3, 3]`` for gsplat."""
    if num_views is None:
        num_views = w2c.shape[1] if w2c.dim() == 4 else 1
    w2c = ensure_bnv_cameras(w2c, num_views, (4, 4))
    intrinsics = ensure_bnv_cameras(intrinsics, num_views, (3, 3))
    viewmats = w2c[batch_idx, frame_idx].view(1, 4, 4).float()
    ks = intrinsics[batch_idx, frame_idx].view(1, 3, 3).float()
    return viewmats, ks


def extract_gaussian_batch(gaussians, batch_idx: int = 0) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Extract per-point Gaussian attributes as ``[Q, ...]`` tensors."""
    def _take(field: str, last_dim: Optional[int] = None) -> Tensor:
        value = getattr(gaussians, field).float()
        if last_dim == 3 and value.dim() == 4 and value.shape[-1] == 3:
            value = squeeze_query_points(value)
        if value.dim() == 4 and last_dim is not None and value.shape[-1] == last_dim:
            while value.dim() > 3 and value.shape[1] == 1:
                value = value.squeeze(1)
        if value.dim() == 3:
            value = value[batch_idx]
        elif value.dim() == 2 and last_dim is not None and value.shape[-1] == last_dim:
            pass
        elif value.dim() == 2 and last_dim is None:
            value = value[batch_idx]
        elif value.dim() == 1 and last_dim is None:
            pass
        else:
            raise ValueError(f"Unexpected gaussian {field} shape {tuple(value.shape)}")
        return value

    means = _take("means", 3)
    rotations = _take("rotations", 4)
    scales = _take("scales", 3)
    opacities = _take("opacities")
    harmonics = gaussians.harmonics.float()
    while harmonics.dim() > 4 and harmonics.shape[1] == 1:
        harmonics = harmonics.squeeze(1)
    if harmonics.dim() == 4:
        harmonics = harmonics[batch_idx]

    if means.dim() != 2 or rotations.dim() != 2 or means.shape[0] != rotations.shape[0]:
        raise ValueError(
            "Gaussian attribute mismatch: "
            f"means={tuple(means.shape)}, rotations={tuple(rotations.shape)}"
        )
    if scales.shape[0] != means.shape[0] or opacities.shape[0] != means.shape[0]:
        raise ValueError(
            "Gaussian attribute mismatch: "
            f"means={tuple(means.shape)}, scales={tuple(scales.shape)}, "
            f"opacities={tuple(opacities.shape)}"
        )
    return means, rotations, scales, opacities, harmonics


def canonicalize_bnv_geometry(
    tensor: Optional[Tensor],
    batch_size: int,
    num_views: int,
) -> Optional[Tensor]:
    """Normalize GT depth / mask tensors to ``[B, N, C, H, W]`` (``C`` is 1 or 3)."""
    if tensor is None:
        return None
    b, n = batch_size, num_views
    if tensor.dim() == 5:
        if tensor.shape[0] == b:
            # [B, 1, N, H, W] from ``unsqueeze(-3)`` on [B, N, H, W]
            if (
                tensor.shape[1] == 1
                and tensor.shape[2] == n
                and tensor.shape[3] not in (1, 3)
            ):
                return tensor.squeeze(1).unsqueeze(2)
            if tensor.shape[1] == n and tensor.shape[2] in (1, 3):
                return tensor
            if tensor.shape[1] == 1 and tensor.shape[2] in (1, 3):
                return tensor
        if tensor.shape[0] == b * n:
            return tensor.view(b, n, *tensor.shape[1:])
    if tensor.dim() == 4:
        if tensor.shape[0] == b * n:
            if tensor.shape[1] in (1, 3):
                return tensor.view(b, n, tensor.shape[1], *tensor.shape[-2:])
            return tensor.view(b, n, 1, *tensor.shape[1:])
        if tensor.shape[0] == b and tensor.shape[1] == n:
            return tensor.unsqueeze(2)
    if tensor.dim() == 3 and tensor.shape[0] == b * n:
        return tensor.view(b, n, 1, *tensor.shape[-2:])
    return tensor


def extract_frame_bnv_mask(
    tensor: Optional[Tensor],
    frame_idx: int,
    batch_size: int,
    num_views: int,
    batch_idx: int = 0,
) -> Optional[Tensor]:
    """Extract ``[1, H, W]`` mask for one batch item / frame."""
    t = canonicalize_bnv_geometry(tensor, batch_size, num_views)
    if t is None or t.dim() != 5:
        return None
    return t[batch_idx, frame_idx, :1].float()


@torch.no_grad()
def compute_target_to_ref_visibility_masks(
    target_local_depth: Tensor,
    target_depth_mask: Optional[Tensor],
    w2c: Tensor,
    intrinsics: Tensor,
    ref_frame: int,
    target_frames: Sequence[int],
    depth_ratio_low: float = 0.9,
    depth_ratio_high: float = 1.1,
) -> Tensor:
    """FFGS-style reprojection visibility on target views w.r.t. the ref camera.

    For each target frame, transform its local XYZ into the ref camera, project
    into the ref image, and mark pixels whose depth is consistent with the ref
    depth map.  Supervision on dynamic frames uses this mask (intersected with
    gsplat alpha) to avoid hallucinating in disoccluded / inconsistent regions.

    Args:
        target_local_depth: ``(B, N, 3, H, W)`` XYZ in each view's camera frame.
        target_depth_mask:  ``(B, N, 1, H, W)`` or ``(B, N, H, W)`` valid mask.
        w2c:                ``(B, N, 4, 4)`` world-to-camera (same space as depth).
        intrinsics:         ``(B, N, 3, 3)``.
        ref_frame:          Reference / source view index.
        target_frames:      Target view indices to evaluate (typically ``t != ref``).

    Returns:
        ``(B, len(target_frames), 1, H, W)`` bool visibility masks on each target.
    """
    if not target_frames:
        raise ValueError("target_frames must be non-empty")

    if target_local_depth.dim() != 5:
        raise ValueError(
            f"target_local_depth must be [B, N, C, H, W], got {tuple(target_local_depth.shape)}"
        )

    B, _, _, H, W = target_local_depth.shape
    device = target_local_depth.device
    dtype = target_local_depth.dtype

    if target_local_depth.shape[2] == 1:
        z = target_local_depth[:, :, 0]
        ys = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, -1, H, W)
        xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, -1, H, W)
        if intrinsics.dim() == 4:
            fx = intrinsics[:, :, 0, 0].view(B, -1, 1, 1)
            fy = intrinsics[:, :, 1, 1].view(B, -1, 1, 1)
            cx = intrinsics[:, :, 0, 2].view(B, -1, 1, 1)
            cy = intrinsics[:, :, 1, 2].view(B, -1, 1, 1)
        else:
            fx = intrinsics[:, 0, 0].view(B, 1, 1, 1)
            fy = intrinsics[:, 1, 1].view(B, 1, 1, 1)
            cx = intrinsics[:, 0, 2].view(B, 1, 1, 1)
            cy = intrinsics[:, 1, 2].view(B, 1, 1, 1)
        x = (xs - cx) * z / fx.clamp(min=1e-6)
        y = (ys - cy) * z / fy.clamp(min=1e-6)
        target_local_depth = torch.stack([x, y, z], dim=2)
    elif target_local_depth.shape[2] > 3:
        target_local_depth = target_local_depth[:, :, :3]

    if target_depth_mask is None:
        target_depth_mask = torch.ones(
            B, target_local_depth.shape[1], 1, H, W,
            device=device, dtype=dtype,
        )
    if target_depth_mask.ndim == 4:
        target_depth_mask = target_depth_mask.unsqueeze(2)

    w2c_ref = w2c[:, ref_frame]
    k_ref = intrinsics[:, ref_frame]
    ref_z = target_local_depth[:, ref_frame, 2:3]
    ref_mask = target_depth_mask[:, ref_frame]

    vis_masks = []
    for t in target_frames:
        c2w_tgt = torch.inverse(w2c[:, t])
        tgt_to_ref = w2c_ref @ c2w_tgt

        pts_local = target_local_depth[:, t]
        tgt_mask = target_depth_mask[:, t]

        pts_flat = pts_local.reshape(B, 3, -1)
        ones = torch.ones(B, 1, H * W, device=device, dtype=dtype)
        pts_homo = torch.cat([pts_flat, ones], dim=1)
        pts_ref = (tgt_to_ref @ pts_homo)[:, :3]
        z_proj = pts_ref[:, 2:3]

        uv_homo = k_ref @ pts_ref
        uv = uv_homo[:, :2] / (z_proj + 1e-8)
        u, v = uv[:, 0], uv[:, 1]
        inside = (
            (u >= 0) & (u < W) & (v >= 0) & (v < H)
            & (z_proj.squeeze(1) > 0)
            & tgt_mask.reshape(B, -1).bool()
        )
        inside_mask = inside.reshape(B, 1, H, W)

        grid_u = 2 * uv[:, 0:1] / W - 1
        grid_v = 2 * uv[:, 1:2] / H - 1
        grid = torch.stack([grid_u, grid_v], dim=-1).reshape(B, H, W, 2)

        sampled_z = F.grid_sample(
            ref_z, grid, mode="bilinear", padding_mode="zeros", align_corners=False,
        )
        sampled_mask = F.grid_sample(
            ref_mask.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False,
        )

        z_proj_map = z_proj.reshape(B, 1, H, W)
        ratio = z_proj_map / (sampled_z + 1e-8)
        vis = (
            (ratio >= depth_ratio_low) & (ratio <= depth_ratio_high)
            & (sampled_z > 0) & (sampled_mask > 0.5)
        )
        vis_masks.append(vis & inside_mask)

    return torch.stack(vis_masks, dim=1)


@torch.no_grad()
def compute_depth_gradient(
    points_xyz: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-pixel XYZ gradient magnitude (dev_lx FFGS ``_compute_depth_gradient``).

    Args:
        points_xyz: ``(B, 3, H, W)`` local-frame point map.
        valid_mask: ``(B, 1, H, W)`` or ``None``.

    Returns:
        grad_mag_xyz, grad_mag_z, valid_grad_mask — each ``(B, H, W)``.
    """
    diff_y = points_xyz[:, :, 1:, :] - points_xyz[:, :, :-1, :]
    diff_x = points_xyz[:, :, :, 1:] - points_xyz[:, :, :, :-1]
    diff_y_padded = F.pad(diff_y, (0, 0, 0, 1))
    diff_x_padded = F.pad(diff_x, (0, 1, 0, 0))

    grad_mag_xyz = torch.sqrt(
        (diff_y_padded * diff_y_padded).sum(dim=1)
        + (diff_x_padded * diff_x_padded).sum(dim=1)
        + 1e-12
    )
    grad_mag_z = torch.sqrt(
        diff_y_padded[:, 2] ** 2 + diff_x_padded[:, 2] ** 2 + 1e-12
    )

    if valid_mask is not None:
        b_pts, _, h_pts, w_pts = points_xyz.shape
        m = valid_mask
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            if m.shape[0] == b_pts:
                m = m.unsqueeze(1)
            elif m.shape[0] == 1:
                m = m.unsqueeze(1)
            else:
                m = m[:1].unsqueeze(1)
        elif m.dim() == 4:
            m = m[:, :1]
        elif m.dim() == 5:
            m = m[:, 0, :1]
        if m.shape[-2:] != (h_pts, w_pts):
            m = F.interpolate(m.float(), size=(h_pts, w_pts), mode="nearest")
        mask_bool = m[:, 0].bool()
        pair_valid_y = mask_bool[:, 1:, :] & mask_bool[:, :-1, :]
        pair_valid_x = mask_bool[:, :, 1:] & mask_bool[:, :, :-1]
        pair_valid_y_padded = F.pad(pair_valid_y, (0, 0, 0, 1), value=False)
        pair_valid_x_padded = F.pad(pair_valid_x, (0, 1, 0, 0), value=False)
        valid_grad_mask = pair_valid_y_padded & pair_valid_x_padded
    else:
        valid_grad_mask = torch.ones_like(grad_mag_xyz, dtype=torch.bool)

    return grad_mag_xyz, grad_mag_z, valid_grad_mask


def resize_bnv_geometry(
    tensor: Optional[Tensor],
    batch_size: int,
    num_views: int,
    height: int,
    width: int,
    *,
    is_mask: bool = False,
) -> Optional[Tensor]:
    """Resize ``(B, N, C, H, W)`` GT geometry to match render resolution."""
    if tensor is None:
        return None
    tensor = canonicalize_bnv_geometry(tensor, batch_size, num_views)
    if tensor is None:
        return None
    if tensor.dim() != 5:
        raise ValueError(
            f"resize_bnv_geometry expects [B,N,C,H,W] after canonicalize, got {tuple(tensor.shape)}"
        )
    if tensor.shape[-2:] == (height, width):
        return tensor
    b, n, c = tensor.shape[:3]
    flat = tensor.reshape(b * n, c, tensor.shape[-2], tensor.shape[-1])
    if is_mask:
        flat = F.interpolate(
            flat.float(), size=(height, width), mode="nearest",
        )
    else:
        flat = F.interpolate(
            flat.float(), size=(height, width), mode="bilinear", align_corners=False,
        )
    return flat.view(b, n, c, height, width)


def render_sparse_gaussians_at_frame(
    gaussians,
    w2c: Tensor,
    intrinsics: Tensor,
    frame_idx: int,
    height: int,
    width: int,
    batch_idx: int = 0,
    num_views: Optional[int] = None,
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    return_alpha: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Render sparse Gaussians for one batch item / frame."""
    from gsplat.rendering import rasterization

    viewmats, ks = select_frame_camera(
        w2c, intrinsics, frame_idx=frame_idx, batch_idx=batch_idx, num_views=num_views,
    )
    means, rotations, scales, opacities, harmonics = extract_gaussian_batch(
        gaussians, batch_idx=batch_idx,
    )
    colors = harmonics.permute(0, 2, 1).contiguous()
    sh_degree = int(harmonics.shape[-1] ** 0.5) - 1
    bg = torch.tensor(background_color, device=means.device, dtype=torch.float32)

    with torch.autocast("cuda", enabled=False):
        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=rotations,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
            render_mode="RGB",
            packed=False,
            backgrounds=bg.unsqueeze(0),
        )
    rgb = render_colors[0].permute(2, 0, 1).clamp(0, 1)
    if not return_alpha:
        return rgb
    alpha = render_alphas[0]
    if alpha.dim() == 3 and alpha.shape[-1] == 1:
        alpha = alpha[..., 0]
    return rgb, alpha


def build_frame_displacements(
    warp3d: Tensor,
    warp3d_delta: Optional[Tensor],
    pair_idx: Sequence[Tuple[int, int]],
    ref_frame: int,
    batch_size: int,
    num_views: int,
) -> Tensor:
    """Build per-frame displacements ``[B, N, Q, 3]`` relative to ``ref_frame``.

    Priority for frame ``t``:
        1. ``warp3d_delta`` at pair ``(ref, t)``
        2. ``warp3d(ref,t) - warp3d(ref,ref)`` when cross pair exists
        3. zeros when ``t == ref``
    """
    num_pair = len(pair_idx)
    warp3d = flatten_pair_queries(warp3d)
    if warp3d_delta is not None:
        warp3d_delta = flatten_pair_queries(warp3d_delta)
    q_dim = warp3d.shape[-2]
    device, dtype = warp3d.device, warp3d.dtype

    w3d = warp3d.view(batch_size, num_pair, q_dim, 3)
    w3d_delta = None
    if warp3d_delta is not None:
        w3d_delta = warp3d_delta.view(batch_size, num_pair, q_dim, 3)

    ref_identity = find_pair_index(pair_idx, ref_frame, ref_frame)
    if ref_identity is None:
        raise ValueError(f"Identity pair ({ref_frame}, {ref_frame}) missing from pair_idx")

    ref_points = w3d[:, ref_identity]
    displacements = torch.zeros(batch_size, num_views, q_dim, 3, device=device, dtype=dtype)

    for t in range(num_views):
        if t == ref_frame:
            continue
        idx = find_pair_index(pair_idx, ref_frame, t)
        if idx is not None and w3d_delta is not None:
            displacements[:, t] = w3d_delta[:, idx]
        elif idx is not None:
            displacements[:, t] = w3d[:, idx] - ref_points
    return displacements


def compute_query_motion_descriptor(
    displacements: Tensor,
    ref_frame: int,
    enhanced_motion: bool = True,
) -> Tensor:
    """Aggregate per-frame displacements into a per-query motion descriptor.

    Args:
        displacements: ``[B, N, Q, 3]`` (normalized world space).
        ref_frame: Reference index to exclude from aggregation.

    Returns:
        ``[B, Q, C]`` with ``C=6`` (enhanced) or ``1`` (max magnitude only).
    """
    flow = displacements.clone()
    flow[:, ref_frame] = 0.0
    mag = flow.norm(dim=-1)  # [B, N, Q]

    if not enhanced_motion:
        max_mag = mag.max(dim=1).values.unsqueeze(-1)
        return max_mag

    max_mag = mag.max(dim=1).values.unsqueeze(-1)
    mean_mag = mag.mean(dim=1).unsqueeze(-1)
    std_mag = mag.std(dim=1).unsqueeze(-1).clamp(min=1e-6)
    std_mag = torch.nan_to_num(std_mag, nan=0.0)
    mean_flow = flow.mean(dim=1)
    return torch.cat([max_mag, mean_mag, std_mag, mean_flow], dim=-1)


def rgb_to_sh_dc(rgb: Tensor) -> Tensor:
    """Convert RGB in ``[0, 1]`` to the SH DC coefficient (per channel)."""
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def sample_rgb_at_query_uv(
    image: Tensor,
    query_uv: Tensor,
    ref_frame: int,
    batch_size: int = 1,
) -> Tensor:
    """Sample RGB at normalized query UVs on the reference frame.

    Args:
        image: ``[B, N, C, H, W]`` or ``[N, C, H, W]`` in ``[-1, 1]``.
        query_uv: ``[Q, 2]``, ``[B, Q, 2]``, or ``[B, N, Q, 2]`` in ``[0, 1]``.
        ref_frame: Reference frame index.
        batch_size: Batch size when ``image`` is 5-D.

    Returns:
        ``[B, Q, 3]`` RGB in ``[0, 1]``.
    """
    if image.dim() == 4:
        rgb = (image[ref_frame].float() + 1.0) * 0.5
        rgb = rgb.unsqueeze(0)
    elif image.dim() == 5:
        rgb = (image[:, ref_frame].float() + 1.0) * 0.5
    else:
        raise ValueError(f"Expected image [B,N,C,H,W] or [N,C,H,W], got {tuple(image.shape)}")

    uv = query_uv.float()
    if uv.dim() == 2:
        uv = uv.unsqueeze(0)
    if uv.dim() == 4:
        uv = uv[:, ref_frame]
    if uv.shape[0] == 1 and batch_size > 1:
        uv = uv.expand(batch_size, -1, -1)
    if uv.shape[0] != batch_size:
        raise ValueError(
            f"query_uv batch mismatch: got {uv.shape[0]}, expected {batch_size}",
        )

    grid = uv * 2.0 - 1.0
    grid = grid.unsqueeze(2)
    sampled = F.grid_sample(
        rgb,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(-1).permute(0, 2, 1).contiguous()


def bootstrap_sh_dc_from_rgb(
    gs_sh: Tensor,
    rgb_anchor: Tensor,
) -> Tensor:
    """Set SH DC from RGB anchor, preserving higher-order SH bands from ``gs_sh``."""
    sh = gs_sh.clone()
    sh[..., 0, :] = rgb_to_sh_dc(rgb_anchor)
    return sh


def normalize_quaternions(rotation: Tensor) -> Tensor:
    """Ensure unit quaternions for gsplat; zero-norm predictions use identity."""
    rot = rotation.float()
    norm = rot.norm(dim=-1, keepdim=True)
    identity = torch.zeros_like(rot)
    identity[..., 0] = 1.0
    unit = torch.where(norm > 1e-6, rot / norm.clamp(min=1e-8), identity)
    return unit.to(dtype=rotation.dtype)


def subsample_query_indices(num_queries: int, max_points: int, device: torch.device) -> Tensor:
    if max_points is None or max_points <= 0 or num_queries <= max_points:
        return torch.arange(num_queries, device=device)
    return torch.randperm(num_queries, device=device)[:max_points]


def estimate_wild_video_scale(
    results: dict,
    num_views: int,
    device: torch.device,
    depth_prior: float = 3.0,
) -> Tensor:
    """Heuristic ``sparse_pointmap_max_range`` for wild video (no GT in batch).

    Training stores metric scene scale separately from normalized geometry.
    Use predicted ``pair_depth`` median × ``depth_prior`` (typical batch scale is ~2–6).
    """
    batch_size = int(results["sparse_global_points"].shape[0])
    if "pair_depth" in results:
        pd = results["pair_depth"].float().reshape(batch_size, -1)
        scale_val = pd.median(dim=-1).values.clamp(min=0.1) * depth_prior
    else:
        pts = squeeze_query_points(results["sparse_global_points"].float())
        scale_val = pts.norm(dim=-1).mean(dim=1).clamp(min=0.1) * depth_prior
    scale_val = scale_val.clamp(min=0.5, max=20.0)
    return scale_val.view(batch_size, 1, 1, 1, 1).expand(batch_size, num_views, 1, 1, 1).to(device)


def build_sparse_dynamic_gaussians(
    gs_opacity: Tensor,
    gs_scale: Tensor,
    gs_rotation: Tensor,
    gs_sh: Tensor,
    means: Tensor,
    gs_opacity_delta: Optional[Tensor] = None,
    gs_scale_delta: Optional[Tensor] = None,
    gs_rotation_delta: Optional[Tensor] = None,
    target_frame_idx: Optional[int] = None,
):
    """Assemble a ``Gaussians`` dataclass from sparse query attributes.

    Args:
        gs_opacity:  ``[B, Q, 1]`` sigmoid-activated.
        gs_scale:    ``[B, Q, 3]`` log-space (with -4 bias applied upstream).
        gs_rotation: ``[B, Q, 4]`` unit quaternion.
        gs_sh:       ``[B, Q, d_sh, 3]``.
        means:       ``[B, Q, 3]`` world-space positions for the target frame.
    """
    from hAlgorithm.modules.models2.external.anysplat.model.encoder.common.gaussians import (
        build_covariance,
    )
    from hAlgorithm.modules.models2.external.anysplat.model.types import Gaussians

    means = squeeze_query_points(means)
    opacity = gs_opacity
    scale = gs_scale
    rotation = gs_rotation

    if target_frame_idx is not None:
        if gs_opacity_delta is not None and target_frame_idx < gs_opacity_delta.shape[1]:
            opacity = (opacity + gs_opacity_delta[:, target_frame_idx]).clamp(0.0, 1.0)
        if gs_scale_delta is not None and target_frame_idx < gs_scale_delta.shape[1]:
            scale = scale + gs_scale_delta[:, target_frame_idx]
        if gs_rotation_delta is not None and target_frame_idx < gs_rotation_delta.shape[1]:
            rotation = rotation + gs_rotation_delta[:, target_frame_idx]

    rotation = normalize_quaternions(rotation)
    scales = scale.exp().clamp(min=1e-6, max=1e3)
    covariances = build_covariance(scales.float(), rotation.float()).to(means.dtype)
    harmonics = gs_sh.permute(0, 1, 3, 2)
    opacities = opacity.squeeze(-1)

    return Gaussians(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
        scales=scales,
        rotations=rotation,
    )


def sparse_gaussians_to_dict(gaussians, batch_size: int) -> dict:
    """Convert ``Gaussians`` to the dict format used by ``WFMQueryPipeline.render_gaussians``."""
    return dict(
        means=gaussians.means,
        harmonics=gaussians.harmonics.permute(0, 1, 3, 2).contiguous(),
        opacities=gaussians.opacities,
        scales=gaussians.scales,
        rotations=gaussians.rotations,
    )


def sample_scalar_at_query_uv(
    values: Tensor,
    query_uv: Tensor,
    ref_frame: int = 0,
    batch_size: int = 1,
) -> Tensor:
    """Sample per-map values at normalized query UVs.

    Args:
        values: ``[B, N, C, H, W]``, ``[N, C, H, W]``, ``[B, C, H, W]``, or ``[C, H, W]``.
        query_uv: Same conventions as :func:`sample_rgb_at_query_uv`.

    Returns:
        ``[B, Q, C]`` (``C=1`` kept as last dim).
    """
    if values.dim() == 3:
        values = values.unsqueeze(0)
    if values.dim() == 4:
        if values.shape[0] == batch_size:
            pass
        else:
            values = values.unsqueeze(0)
    elif values.dim() == 5:
        values = values[:, ref_frame]
    else:
        raise ValueError(f"Unsupported values shape {tuple(values.shape)}")

    if values.shape[0] == 1 and batch_size > 1:
        values = values.expand(batch_size, -1, -1, -1)
    b, c, h, w = values.shape

    uv = query_uv.float()
    if uv.dim() == 2:
        uv = uv.unsqueeze(0)
    if uv.dim() == 4:
        uv = uv[:, ref_frame]
    if uv.shape[0] == 1 and batch_size > 1:
        uv = uv.expand(batch_size, -1, -1)
    if uv.shape[0] != batch_size:
        raise ValueError(
            f"query_uv batch mismatch: got {uv.shape[0]}, expected {batch_size}",
        )

    grid = uv * 2.0 - 1.0
    grid = grid.unsqueeze(2)
    sampled = F.grid_sample(
        values,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(-1).permute(0, 2, 1).contiguous()


def compute_focal_scale_multiplier(
    intrinsics: Tensor,
    width: int,
    height: int,
    batch_idx: int = 0,
    frame_idx: int = 0,
    multiplier: float = 0.1,
) -> Tensor:
    """Envision4D-style focal multiplier for depth-conditioned Gaussian scale."""
    num_views = intrinsics.shape[1] if intrinsics.dim() == 4 else max(intrinsics.shape[0], frame_idx + 1)
    intrinsics = ensure_bnv_cameras(intrinsics, num_views, (3, 3))
    if intrinsics.dim() == 3:
        k = intrinsics[frame_idx].float()
    else:
        k = intrinsics[batch_idx, frame_idx].float()
    pixel_size = torch.tensor(
        [1.0 / max(width, 1), 1.0 / max(height, 1)],
        device=k.device,
        dtype=k.dtype,
    )
    xy = multiplier * torch.matmul(k[:2, :2].inverse(), pixel_size)
    return xy.sum().clamp(min=1e-6)


def _nearest_motion_indices_cpu(muv: Tensor, dense_uv: Tensor) -> Tensor:
    """CPU KD-tree NN lookup for large dense grids (much faster than GPU brute force)."""
    from scipy.spatial import cKDTree

    b = dense_uv.shape[0]
    parts: list[Tensor] = []
    muv_cpu = muv.detach().float().cpu()
    dense_cpu = dense_uv.detach().float().cpu()
    for bi in range(b):
        tree = cKDTree(muv_cpu[bi].numpy())
        _, idx = tree.query(dense_cpu[bi].numpy(), k=1, workers=-1)
        parts.append(torch.as_tensor(idx, dtype=torch.long))
    return torch.stack(parts, dim=0).to(device=muv.device, non_blocking=True)


def propagate_motion_flow_to_dense_queries(
    motion_flow: Tensor,
    motion_uv: Tensor,
    dense_uv: Tensor,
    batch_size: int = 1,
    ref_frame: int = 0,
    chunk_size: int = 8192,
) -> Tensor:
    """Copy per-frame motion flow from sparse traj queries to dense query points.

    For large full-image grids (``Q_dense * Q_motion`` huge), uses a CPU KD-tree.
    Otherwise falls back to chunked GPU nearest-neighbour search.
    """
    motion_flow = canonicalize_scene_flow(motion_flow.float())
    dense_uv = dense_uv.float()
    if dense_uv.dim() == 2:
        dense_uv = dense_uv.unsqueeze(0)
    if dense_uv.dim() == 4:
        dense_uv = dense_uv[:, ref_frame]
    if dense_uv.shape[0] == 1 and batch_size > 1:
        dense_uv = dense_uv.expand(batch_size, -1, -1)

    muv = _motion_uv_at_ref(
        motion_uv,
        ref_frame=ref_frame,
        batch_size=batch_size,
        num_views=motion_flow.shape[1],
    )
    b, n_views, qm, _ = motion_flow.shape
    qg = dense_uv.shape[1]
    if qm == 0 or qg == 0:
        return motion_flow.new_zeros((b, n_views, qg, 3))

    with torch.no_grad():
        if qg >= 100_000 or (qg * qm >= 8_000_000):
            nn_idx = _nearest_motion_indices_cpu(muv, dense_uv)
        else:
            chunk_size = max(int(chunk_size), 1)
            nn_parts: list[Tensor] = []
            for q0 in range(0, qg, chunk_size):
                q1 = min(q0 + chunk_size, qg)
                diff = dense_uv[:, q0:q1].unsqueeze(2) - muv.unsqueeze(1)
                nn_parts.append(diff.pow(2).sum(dim=-1).argmin(dim=-1))
            nn_idx = torch.cat(nn_parts, dim=1)

    idx_exp = nn_idx.unsqueeze(1).unsqueeze(-1).expand(b, n_views, qg, 3)
    src = motion_flow.gather(2, idx_exp)
    return canonicalize_scene_flow(src)


def _motion_uv_at_ref(
    motion_uv: Tensor,
    ref_frame: int,
    batch_size: int,
    num_views: Optional[int] = None,
) -> Tensor:
    """Normalize motion UV to ``[B, Qm, 2]`` at the reference frame."""
    uv = motion_uv.float()
    if uv.dim() == 4:
        uv = uv[:, ref_frame]
    elif uv.dim() == 3:
        if uv.shape[0] != batch_size and uv.shape[-1] == 2:
            if batch_size == 1:
                uv = uv[ref_frame].unsqueeze(0)
            elif num_views is not None and uv.shape[0] == num_views:
                uv = uv[ref_frame].unsqueeze(0).expand(batch_size, -1, -1)
    elif uv.dim() == 2:
        uv = uv.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported motion_uv shape {tuple(uv.shape)}")

    if uv.dim() == 2:
        uv = uv.unsqueeze(0)
    if uv.shape[0] == 1 and batch_size > 1:
        uv = uv.expand(batch_size, -1, -1)
    if uv.shape[0] != batch_size:
        raise ValueError(
            f"motion_uv batch mismatch: got {uv.shape[0]}, expected {batch_size}",
        )
    return uv


def render_sparse_gaussians_rgb_depth_at_frame(
    gaussians,
    w2c: Tensor,
    intrinsics: Tensor,
    frame_idx: int,
    height: int,
    width: int,
    batch_idx: int = 0,
    num_views: Optional[int] = None,
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    return_alpha: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Render sparse Gaussians with RGB + expected depth."""
    from gsplat.rendering import rasterization

    viewmats, ks = select_frame_camera(
        w2c, intrinsics, frame_idx=frame_idx, batch_idx=batch_idx, num_views=num_views,
    )
    means, rotations, scales, opacities, harmonics = extract_gaussian_batch(
        gaussians, batch_idx=batch_idx,
    )
    colors = harmonics.permute(0, 2, 1).contiguous()
    sh_degree = int(harmonics.shape[-1] ** 0.5) - 1
    bg = torch.tensor(background_color, device=means.device, dtype=torch.float32)

    with torch.autocast("cuda", enabled=False):
        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=rotations,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
            render_mode="RGB+ED",
            packed=False,
            backgrounds=bg.unsqueeze(0),
        )
    rgb = render_colors[0, ..., :3].permute(2, 0, 1).clamp(0, 1)
    depth = render_colors[0, ..., 3]
    if not return_alpha:
        return rgb, depth
    alpha = render_alphas[0]
    if alpha.dim() == 3 and alpha.shape[-1] == 1:
        alpha = alpha[..., 0]
    return rgb, depth, alpha


def vis_sparse_4dgs_results(
    outputs_list: list,
    dgs_out_dir: str,
    data_idx: int,
) -> None:
    """Save GT | render | error strips for sparse 4DGS inference."""
    import logging
    import os

    import cv2
    import numpy as np

    logger = logging.getLogger(__name__)

    rows = []
    for out in outputs_list:
        gt_rgb = getattr(out, "rgb", None)
        ren_rgb = getattr(out, "dgs_render_rgb", None)
        if gt_rgb is None or ren_rgb is None:
            continue

        gt_bgr = (np.clip(gt_rgb[..., ::-1], 0, 1) * 255).astype(np.uint8)
        ren_bgr = (np.clip(ren_rgb[..., ::-1], 0, 1) * 255).astype(np.uint8)
        err_bgr = (np.clip(np.abs(gt_rgb - ren_rgb)[..., ::-1] * 3, 0, 1) * 255).astype(np.uint8)

        h, w = gt_bgr.shape[:2]
        strip = np.zeros((h, w * 3, 3), dtype=np.uint8)
        strip[:, :w] = gt_bgr
        strip[:, w:2 * w] = ren_bgr
        strip[:, 2 * w:] = err_bgr
        rows.append(strip)

    if not rows:
        return

    os.makedirs(dgs_out_dir, exist_ok=True)
    grid = np.concatenate(rows, axis=0)
    save_path = os.path.join(dgs_out_dir, f"4dgs_compare_{data_idx:06d}.jpg")
    cv2.imwrite(save_path, grid)
    logger.info("Sparse 4DGS vis saved → %s", save_path)
