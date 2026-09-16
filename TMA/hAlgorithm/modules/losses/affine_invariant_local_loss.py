import logging
import math
from typing import *

import torch
import utils3d
from torch import nn


def scatter_min(
    size: int, dim: int, index: torch.LongTensor, src: torch.Tensor
) -> torch.return_types.min:
    "Scatter the minimum value along the given dimension of `input` into `src` at the indices specified in `index`."
    shape = src.shape[:dim] + (size,) + src.shape[dim + 1 :]
    minimum = torch.full(shape, float("inf"), dtype=src.dtype, device=src.device).scatter_reduce(
        dim=dim, index=index, src=src, reduce="amin", include_self=False
    )
    minimum_where = torch.where(src == torch.gather(minimum, dim=dim, index=index))
    indices = torch.full(shape, -1, dtype=torch.long, device=src.device)
    indices[(*minimum_where[:dim], index[minimum_where], *minimum_where[dim + 1 :])] = (
        minimum_where[dim]
    )
    return torch.return_types.min((minimum, indices))


def split_batch_fwd(fn: Callable, chunk_size: int, *args, **kwargs):
    batch_size = next(x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)).shape[0]
    n_chunks = batch_size // chunk_size + (batch_size % chunk_size > 0)
    splited_args = tuple(
        arg.split(chunk_size, dim=0) if isinstance(arg, torch.Tensor) else [arg] * n_chunks
        for arg in args
    )
    splited_kwargs = {
        k: [v.split(chunk_size, dim=0) if isinstance(v, torch.Tensor) else [v] * n_chunks]
        for k, v in kwargs.items()
    }
    results = []
    for i in range(n_chunks):
        chunk_args = tuple(arg[i] for arg in splited_args)
        chunk_kwargs = {k: v[i] for k, v in splited_kwargs.items()}
        results.append(fn(*chunk_args, **chunk_kwargs))

    if isinstance(results[0], tuple):
        return tuple(torch.cat(r, dim=0) for r in zip(*results))
    else:
        return torch.cat(results, dim=0)


def _pad_cumsum(cumsum: torch.Tensor):
    return torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum, cumsum[..., -1:]], dim=-1)


def _pad_inf(x_: torch.Tensor):
    return torch.cat(
        [torch.full_like(x_[..., :1], -torch.inf), x_, torch.full_like(x_[..., :1], torch.inf)],
        dim=-1,
    )


def _compute_residual(a: torch.Tensor, xyw: torch.Tensor, trunc: float):
    return (
        a.mul(xyw[..., 0]).sub_(xyw[..., 1]).abs_().mul_(xyw[..., 2]).clamp_max_(trunc).sum(dim=-1)
    )


def align(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    trunc: Optional[Union[float, torch.Tensor]] = None,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """
    If trunc is None, solve `min sum_i w_i * |a * x_i - y_i|`, otherwise solve `min sum_i min(trunc, w_i * |a * x_i - y_i|)`.

    w_i must be >= 0.

    ### Parameters:
    - `x`: tensor of shape (..., n)
    - `y`: tensor of shape (..., n)
    - `w`: tensor of shape (..., n)
    - `trunc`: optional, float or tensor of shape (..., n) or None

    ### Returns:
    - `a`: tensor of shape (...), differentiable
    - `loss`: tensor of shape (...), value of loss function at `a`, detached
    - `index`: tensor of shape (...), where a = y[idx] / x[idx]
    """
    if trunc is None:
        x, y, w = torch.broadcast_tensors(x, y, w)
        sign = torch.sign(x)
        x, y = x * sign, y * sign
        y_div_x = y / x.clamp_min(eps)
        y_div_x, argsort = y_div_x.sort(dim=-1)

        wx = torch.gather(x * w, dim=-1, index=argsort)
        derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
        search = torch.searchsorted(
            derivatives, torch.zeros_like(derivatives[..., :1]), side="left"
        ).clamp_max(derivatives.shape[-1] - 1)

        a = y_div_x.gather(dim=-1, index=search).squeeze(-1)
        index = argsort.gather(dim=-1, index=search).squeeze(-1)
        loss = (w * (a[..., None] * x - y).abs()).sum(dim=-1)

    else:
        # Reshape to (batch_size, n) for simplicity
        x, y, w = torch.broadcast_tensors(x, y, w)
        batch_shape = x.shape[:-1]
        batch_size = math.prod(batch_shape)
        x, y, w = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), w.reshape(-1, w.shape[-1])

        sign = torch.sign(x)
        x, y = x * sign, y * sign
        wx, wy = w * x, w * y
        xyw = torch.stack([x, y, w], dim=-1)  # Stacked for convenient gathering

        y_div_x = A = y / x.clamp_min(eps)
        B = (wy - trunc) / wx.clamp_min(eps)
        C = (wy + trunc) / wx.clamp_min(eps)
        with torch.no_grad():
            # Caculate prefix sum by orders of A, B, C
            A, A_argsort = A.sort(dim=-1)
            Q_A = torch.cumsum(torch.gather(wx, dim=-1, index=A_argsort), dim=-1)
            A, Q_A = _pad_inf(A), _pad_cumsum(
                Q_A
            )  # Pad [-inf, A1, ..., An, inf] and [0, Q1, ..., Qn, Qn] to handle edge cases.

            B, B_argsort = B.sort(dim=-1)
            Q_B = torch.cumsum(torch.gather(wx, dim=-1, index=B_argsort), dim=-1)
            B, Q_B = _pad_inf(B), _pad_cumsum(Q_B)

            C, C_argsort = C.sort(dim=-1)
            Q_C = torch.cumsum(torch.gather(wx, dim=-1, index=C_argsort), dim=-1)
            C, Q_C = _pad_inf(C), _pad_cumsum(Q_C)

            # Caculate left and right derivative of A
            j_A = torch.searchsorted(A, y_div_x, side="left").sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side="left").sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side="left").sub_(1)
            left_derivative = (
                2 * torch.gather(Q_A, dim=-1, index=j_A)
                - torch.gather(Q_B, dim=-1, index=j_B)
                - torch.gather(Q_C, dim=-1, index=j_C)
            )
            j_A = torch.searchsorted(A, y_div_x, side="right").sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side="right").sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side="right").sub_(1)
            right_derivative = (
                2 * torch.gather(Q_A, dim=-1, index=j_A)
                - torch.gather(Q_B, dim=-1, index=j_B)
                - torch.gather(Q_C, dim=-1, index=j_C)
            )

            # Find extrema
            is_extrema = (left_derivative < 0) & (right_derivative >= 0)
            is_extrema[..., 0] |= ~is_extrema.any(
                dim=-1
            )  # In case all derivatives are zero, take the first one as extrema.
            where_extrema_batch, where_extrema_index = torch.where(is_extrema)

            # Calculate objective value at extrema
            extrema_a = y_div_x[where_extrema_batch, where_extrema_index]  # (num_extrema,)
            MAX_ELEMENTS = (
                4096**2
            )  # Split into small batches to avoid OOM in case there are too many extrema.(~1G)
            SPLIT_SIZE = MAX_ELEMENTS // x.shape[-1]
            extrema_value = torch.cat(
                [
                    _compute_residual(extrema_a_split[:, None], xyw[extrema_i_split, :, :], trunc)
                    for extrema_a_split, extrema_i_split in zip(
                        extrema_a.split(SPLIT_SIZE), where_extrema_batch.split(SPLIT_SIZE)
                    )
                ]
            )  # (num_extrema,)

            # Find minima among corresponding extrema
            minima, indices = scatter_min(
                size=batch_size, dim=0, index=where_extrema_batch, src=extrema_value
            )  # (batch_size,)
            index = where_extrema_index[indices]

        a = torch.gather(y, dim=-1, index=index[..., None]) / torch.gather(
            x, dim=-1, index=index[..., None]
        ).clamp_min(eps)
        a = a.reshape(batch_shape)
        loss = minima.reshape(batch_shape)
        index = index.reshape(batch_shape)

    return a, loss, index


def align_points_scale_xyz_shift(
    points_src: torch.Tensor,
    points_tgt: torch.Tensor,
    weight: Optional[torch.Tensor],
    trunc: Optional[Union[float, torch.Tensor]] = None,
    max_iters: int = 30,
    eps: float = 1e-6,
):
    """
    Align `points_src` to `points_tgt` with respect to a shared xyz scale and z shift.
    It is similar to `align_affine` but scale and shift are applied to different dimensions.

    ### Parameters:
    - `points_src: torch.Tensor` of shape (..., N, 3)
    - `points_tgt: torch.Tensor` of shape (..., N, 3)
    - `weights: torch.Tensor` of shape (..., N)

    ### Returns:
    - `scale: torch.Tensor` of shape (...).
    - `shift: torch.Tensor` of shape (..., 3)
    """
    dtype, device = points_src.dtype, points_src.device

    # Flatten batch dimensions for simplicity
    batch_shape, n = points_src.shape[:-2], points_src.shape[-2]
    batch_size = math.prod(batch_shape)
    points_src, points_tgt, weight = (
        points_src.reshape(batch_size, n, 3),
        points_tgt.reshape(batch_size, n, 3),
        weight.reshape(batch_size, n),
    )

    # Take anchors
    anchor_where_batch, anchor_where_n = torch.where(weight > 0)

    with torch.no_grad():
        points_src_anchor = points_src[anchor_where_batch, anchor_where_n]  # (anchors, 3)
        points_tgt_anchor = points_tgt[anchor_where_batch, anchor_where_n]  # (anchors, 3)

        points_src_anchored = (
            points_src[anchor_where_batch, :, :] - points_src_anchor[..., None, :]
        )  # (anchors, n, 3)
        points_tgt_anchored = (
            points_tgt[anchor_where_batch, :, :] - points_tgt_anchor[..., None, :]
        )  # (anchors, n, 3)
        weight_anchored = weight[anchor_where_batch, :, None].expand(-1, -1, 3)  # (anchors, n, 3)

        # Solve optimal scale and shift for each anchor
        MAX_ELEMENTS = 2**20
        scale, loss, index = split_batch_fwd(
            align,
            MAX_ELEMENTS // 2,
            points_src_anchored.flatten(-2),
            points_tgt_anchored.flatten(-2),
            weight_anchored.flatten(-2),
            trunc,
        )  # (anchors,)

        # Get optimal scale and shift for each batch element
        loss, index_anchor = scatter_min(
            size=batch_size, dim=0, index=anchor_where_batch, src=loss
        )  # (batch_size,)

    index_2 = index[index_anchor]  # (batch_size,) [0, 3n)
    index_1 = anchor_where_n[index_anchor] * 3 + index_2 % 3  # (batch_size,) [0, 3n)

    src_1, tgt_1 = torch.gather(points_src.flatten(-2), dim=1, index=index_1[..., None]).squeeze(
        -1
    ), torch.gather(points_tgt.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1)
    src_2, tgt_2 = torch.gather(points_src.flatten(-2), dim=1, index=index_2[..., None]).squeeze(
        -1
    ), torch.gather(points_tgt.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1)

    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1.0)
    shift = torch.gather(
        points_tgt, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)
    ).squeeze(-2) - scale[..., None] * torch.gather(
        points_src, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)
    ).squeeze(
        -2
    )

    scale, shift = scale.reshape(batch_shape), shift.reshape(*batch_shape, 3)

    return scale, shift


def mask_aware_nearest_resize(
    inputs: Union[torch.Tensor, Sequence[torch.Tensor], None],
    mask: torch.BoolTensor,
    size: Tuple[int, int],
    return_index: bool = False,
) -> Tuple[
    Union[torch.Tensor, Sequence[torch.Tensor], None],
    torch.BoolTensor,
    Tuple[torch.LongTensor, ...],
]:
    """
    Resize 2D map by nearest interpolation. Return the nearest neighbor index and mask of the resized map.

    ### Parameters
    - `inputs`: a single or a list of input 2D map(s) of shape (..., H, W, ...).
    - `mask`: input 2D mask of shape (..., H, W)
    - `size`: target size (target_width, target_height)

    ### Returns
    - `*resized_maps`: resized map(s) of shape (..., target_height, target_width, ...).
    - `resized_mask`: mask of the resized map of shape (..., target_height, target_width)
    - `nearest_idx`: if return_index is True, nearest neighbor index of the resized map of shape (..., target_height, target_width) for each dimension, .
    """
    height, width = mask.shape[-2:]
    target_width, target_height = size
    device = mask.device
    filter_h_f, filter_w_f = max(1, height / target_height), max(1, width / target_width)
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1

    # Window the original mask and uv
    uv = utils3d.torch.image_pixel_center(
        width=width, height=height, dtype=torch.float32, device=device
    )
    indices = torch.arange(height * width, dtype=torch.long, device=device).reshape(height, width)
    padded_uv = torch.full(
        (height + 2 * padding_h, width + 2 * padding_w, 2), 0, dtype=torch.float32, device=device
    )
    padded_uv[padding_h : padding_h + height, padding_w : padding_w + width] = uv
    padded_mask = torch.full(
        (*mask.shape[:-2], height + 2 * padding_h, width + 2 * padding_w),
        False,
        dtype=torch.bool,
        device=device,
    )
    padded_mask[..., padding_h : padding_h + height, padding_w : padding_w + width] = mask
    padded_indices = torch.full(
        (height + 2 * padding_h, width + 2 * padding_w), 0, dtype=torch.long, device=device
    )
    padded_indices[padding_h : padding_h + height, padding_w : padding_w + width] = indices
    windowed_uv = utils3d.torch.sliding_window_2d(
        padded_uv, (filter_h_i, filter_w_i), 1, dim=(0, 1)
    )
    windowed_mask = utils3d.torch.sliding_window_2d(
        padded_mask, (filter_h_i, filter_w_i), 1, dim=(-2, -1)
    )
    windowed_indices = utils3d.torch.sliding_window_2d(
        padded_indices, (filter_h_i, filter_w_i), 1, dim=(0, 1)
    )

    # Gather the target pixels's local window
    target_uv = utils3d.torch.image_uv(
        width=target_width, height=target_height, dtype=torch.float32, device=device
    ) * torch.tensor([width, height], dtype=torch.float32, device=device)
    target_lefttop = target_uv - torch.tensor(
        (filter_w_f / 2, filter_h_f / 2), dtype=torch.float32, device=device
    )
    target_window = torch.round(target_lefttop).long() + torch.tensor(
        (padding_w, padding_h), dtype=torch.long, device=device
    )

    target_window_uv = windowed_uv[target_window[..., 1], target_window[..., 0], :, :, :].reshape(
        target_height, target_width, 2, filter_size
    )  # (target_height, tgt_width, 2, filter_size)
    target_window_mask = windowed_mask[
        ..., target_window[..., 1], target_window[..., 0], :, :
    ].reshape(
        *mask.shape[:-2], target_height, target_width, filter_size
    )  # (..., target_height, tgt_width, filter_size)
    target_window_indices = windowed_indices[
        target_window[..., 1], target_window[..., 0], :, :
    ].reshape(
        target_height, target_width, filter_size
    )  # (target_height, tgt_width, filter_size)
    target_window_indices = target_window_indices.expand_as(target_window_mask)

    # Compute nearest neighbor in the local window for each pixel
    dist = torch.where(
        target_window_mask, torch.norm(target_window_uv - target_uv[..., None], dim=-2), torch.inf
    )  # (..., target_height, tgt_width, filter_size)
    nearest = torch.argmin(dist, dim=-1, keepdim=True)  # (..., target_height, tgt_width, 1)
    nearest_idx = torch.gather(target_window_indices, index=nearest, dim=-1).squeeze(
        -1
    )  # (..., target_height, tgt_width)
    target_mask = torch.any(target_window_mask, dim=-1)
    nearest_i, nearest_j = nearest_idx // width, nearest_idx % width
    batch_indices = [
        torch.arange(n, device=device).reshape([1] * i + [n] + [1] * (mask.dim() - i - 1))
        for i, n in enumerate(mask.shape[:-2])
    ]

    index = (*batch_indices, nearest_i, nearest_j)

    if inputs is None:
        outputs = None
    elif isinstance(inputs, torch.Tensor):
        outputs = inputs[index]
    elif isinstance(inputs, Sequence):
        outputs = tuple(x[index] for x in inputs)
    else:
        raise ValueError(f"Invalid input type: {type(inputs)}")

    if return_index:
        return outputs, target_mask, index
    else:
        return outputs, target_mask


def compute_anchor_sampling_weight(
    points: torch.Tensor,
    mask: torch.Tensor,
    radius_2d: torch.Tensor,
    radius_3d: torch.Tensor,
    num_test: int = 64,
) -> torch.Tensor:
    # Importance sampling to balance the sampled probability of fine strutures.
    # NOTE: MoGe-1 uses uniform random sampling instead of importance sampling.
    #       This is an incremental trick introduced later than the publication of MoGe-1 paper.

    height, width = points.shape[-3:-1]

    pixel_i, pixel_j = torch.meshgrid(
        torch.arange(height, device=points.device),
        torch.arange(width, device=points.device),
        indexing="ij",
    )

    test_delta_i = torch.randint(
        -radius_2d,
        radius_2d + 1,
        (
            height,
            width,
            num_test,
        ),
        device=points.device,
    )  # [num_test]
    test_delta_j = torch.randint(
        -radius_2d,
        radius_2d + 1,
        (
            height,
            width,
            num_test,
        ),
        device=points.device,
    )  # [num_test]
    test_i, test_j = (
        pixel_i[..., None] + test_delta_i,
        pixel_j[..., None] + test_delta_j,
    )  # [height, width, num_test]
    test_mask = (
        (test_i >= 0) & (test_i < height) & (test_j >= 0) & (test_j < width)
    )  # [height, width, num_test]
    test_i, test_j = test_i.clamp(0, height - 1), test_j.clamp(
        0, width - 1
    )  # [height, width, num_test]
    test_mask = test_mask & mask[..., test_i, test_j]  # [..., height, width, num_test]
    test_points = points[..., test_i, test_j, :]  # [..., height, width, num_test, 3]
    test_dist = (test_points - points[..., None, :]).norm(dim=-1)  # [..., height, width, num_test]

    weight = 1 / ((test_dist <= radius_3d[..., None]) & test_mask).float().sum(dim=-1).clamp_min(1)
    weight = torch.where(mask, weight, 0)
    weight = weight / weight.sum(dim=(-2, -1), keepdim=True).add(1e-7)  # [..., height, width]
    return weight


def weighted_mean(
    x: torch.Tensor,
    w: torch.Tensor = None,
    dim: Union[int, torch.Size] = None,
    keepdim: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)


def harmonic_mean(
    x: torch.Tensor,
    w: torch.Tensor = None,
    dim: Union[int, torch.Size] = None,
    keepdim: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    if w is None:
        return x.add(eps).reciprocal().mean(dim=dim, keepdim=keepdim).reciprocal()
    else:
        w = w.to(x.dtype)
        return (
            weighted_mean(x.add(eps).reciprocal(), w, dim=dim, keepdim=keepdim, eps=eps)
            .add(eps)
            .reciprocal()
        )


def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def affine_invariant_local_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    gt_mask: torch.Tensor,
    focal: torch.Tensor,
    global_scale: torch.Tensor,
    level: Literal[4, 16, 64],
    align_resolution: int = 32,
    num_patches: int = 16,
    beta: float = 0.0,
    trunc: float = 1.0,
    sparsity_aware: bool = False,
    anchor_mask: torch.Tensor = None,
):
    device, dtype = pred_points.device, pred_points.dtype
    *batch_shape, height, width, _ = pred_points.shape
    batch_size = math.prod(batch_shape)
    pred_points, gt_points, gt_mask, focal, global_scale, anchor_mask = (
        pred_points.reshape(-1, height, width, 3),
        gt_points.reshape(-1, height, width, 3),
        gt_mask.reshape(-1, height, width),
        focal.reshape(-1),
        global_scale.reshape(-1) if global_scale is not None else None,
        anchor_mask.reshape(-1, height, width) if anchor_mask is not None else None,
    )

    # Sample patch anchor points indices [num_total_patches]
    radius_2d = math.ceil(0.5 / level * (height**2 + width**2) ** 0.5)
    radius_3d = 0.5 / level / focal[:, None, None] * gt_points[..., 2]
    anchor_sampling_weights = compute_anchor_sampling_weight(
        gt_points, gt_mask, radius_2d, radius_3d, num_test=64
    )
    if anchor_mask is not None:
        anchor_mask = anchor_mask * gt_mask
        if anchor_mask.sum() < num_patches * batch_size:
            anchor_mask = gt_mask
    else:
        anchor_mask = gt_mask
    where_mask = torch.where(anchor_mask)
    random_selection = torch.multinomial(
        anchor_sampling_weights[where_mask], num_patches * batch_size, replacement=True
    )
    patch_batch_idx, patch_anchor_i, patch_anchor_j = [
        indices[random_selection] for indices in where_mask
    ]  # [num_total_patches]

    # Get patch indices [num_total_patches, patch_h, patch_w]
    patch_i, patch_j = torch.meshgrid(
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        torch.arange(-radius_2d, radius_2d + 1, device=device),
        indexing="ij",
    )
    patch_i, patch_j = (
        patch_i + patch_anchor_i[:, None, None],
        patch_j + patch_anchor_j[:, None, None],
    )
    patch_mask = (patch_i >= 0) & (patch_i < height) & (patch_j >= 0) & (patch_j < width)
    patch_i, patch_j = patch_i.clamp(0, height - 1), patch_j.clamp(0, width - 1)

    # Get patch mask and gt patch points
    gt_patch_anchor_points = gt_points[patch_batch_idx, patch_anchor_i, patch_anchor_j]
    gt_patch_radius_3d = 0.5 / level / focal[patch_batch_idx] * gt_patch_anchor_points[:, 2]
    gt_patch_points = gt_points[patch_batch_idx[:, None, None], patch_i, patch_j]
    gt_patch_dist = (gt_patch_points - gt_patch_anchor_points[:, None, None, :]).norm(dim=-1)
    patch_mask &= gt_mask[patch_batch_idx[:, None, None], patch_i, patch_j]
    patch_mask &= gt_patch_dist <= gt_patch_radius_3d[:, None, None]

    # Pick only non-empty patches
    MINIMUM_POINTS_PER_PATCH = 32
    nonempty = torch.where(patch_mask.sum(dim=(-2, -1)) >= MINIMUM_POINTS_PER_PATCH)
    num_nonempty_patches = nonempty[0].shape[0]
    if num_nonempty_patches == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), {}

    # Finalize all patch variables
    patch_batch_idx, patch_i, patch_j = (
        patch_batch_idx[nonempty],
        patch_i[nonempty],
        patch_j[nonempty],
    )
    patch_mask = patch_mask[nonempty]  # [num_nonempty_patches, patch_h, patch_w]
    gt_patch_points = gt_patch_points[nonempty]  # [num_nonempty_patches, patch_h, patch_w, 3]
    gt_patch_radius_3d = gt_patch_radius_3d[nonempty]  # [num_nonempty_patches]
    gt_patch_anchor_points = gt_patch_anchor_points[nonempty]  # [num_nonempty_patches, 3]
    pred_patch_points = pred_points[patch_batch_idx[:, None, None], patch_i, patch_j]

    # Align patch points
    (pred_patch_points_lr, gt_patch_points_lr), patch_lr_mask = mask_aware_nearest_resize(
        (pred_patch_points, gt_patch_points),
        mask=patch_mask,
        size=(align_resolution, align_resolution),
    )
    # local_scale, local_shift = align_points_scale_xyz_shift(pred_patch_points_lr.flatten(-3, -2), gt_patch_points_lr.flatten(-3, -2), patch_lr_mask.flatten(-2) / gt_patch_radius_3d[:, None].add(1e-7), trunc=trunc, max_iters=10)
    # if global_scale is not None:
    #     scale_differ = local_scale / global_scale[patch_batch_idx]
    #     patch_valid = (scale_differ > 0.1) & (scale_differ < 10.0) & (global_scale > 0)
    # else:
    #     patch_valid = local_scale > 0
    # # local_scale, local_shift = torch.where(patch_valid, local_scale, 0), torch.where(patch_valid[:, None], local_shift, 0)
    # patch_mask &= patch_valid[:, None, None]

    # pred_patch_points = local_scale[:, None, None, None] * pred_patch_points + local_shift[:, None, None, :]                   # [num_patches_nonempty, patch_h, patch_w, 3]

    # Compute loss
    gt_mean = harmonic_mean(gt_points[..., 2], gt_mask, dim=(-2, -1))
    patch_weight = patch_mask.float() / gt_patch_points[..., 2].clamp_min(
        0.1 * gt_mean[patch_batch_idx, None, None]
    )  # [num_patches_nonempty, patch_h, patch_w]
    loss = _smooth(
        (pred_patch_points - gt_patch_points).abs() * patch_weight[..., None], beta=beta
    ).mean(
        dim=(-3, -2, -1)
    )  # [num_patches_nonempty]

    if sparsity_aware:
        # Reweighting improves performance on sparse depth data. NOTE: this is not used in MoGe-1.
        sparsity = patch_mask.float().mean(dim=(-2, -1)) / patch_lr_mask.float().mean(dim=(-2, -1))
        loss = loss / (sparsity + 1e-7)
    loss = (
        torch.scatter_reduce(
            torch.zeros(batch_size, dtype=dtype, device=device),
            dim=0,
            index=patch_batch_idx,
            src=loss,
            reduce="sum",
        )
        / num_patches
    )
    loss = loss.reshape(batch_shape)

    err = (pred_patch_points.detach() - gt_patch_points).norm(dim=-1) / gt_patch_radius_3d[
        ..., None, None
    ]

    # Record any scalar metric
    misc = {
        "truncated_error": weighted_mean(err.clamp_max(1), patch_mask).item(),
        "delta": weighted_mean((err < 1).float(), patch_mask).item(),
    }

    return loss, misc


class AffineIvariantLocalLoss(nn.Module):
    """
    AffineIvariantLocalLoss of MoGe
    https://github.com/microsoft/MoGe/blob/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b/moge/train/losses.py#L111

    """

    def __init__(self, loss_weight=1, offset=0, threshold=3.0, sift_anchor=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.offset = offset
        self.threshold = threshold
        self.sift_anchor = sift_anchor
        self.eps = 1e-6

        if isinstance(self.loss_weight, dict):
            assert "default" in self.loss_weight

    def get_loss_weight(self, name=None):
        if name is not None and isinstance(self.loss_weight, dict) and name in self.loss_weight:
            loss_weight = self.loss_weight[name]
        elif isinstance(self.loss_weight, dict) and name not in self.loss_weight:
            loss_weight = self.loss_weight["default"]
        else:
            loss_weight = self.loss_weight
        return loss_weight

    def forward(self, prediction, target, mask, name=None, **kwargs):

        # prompt_scale = kwargs.get("prompt_scale", None)
        # prompt_center = kwargs.get("prompt_center", None)

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(prediction)

        intrinsics = kwargs["intrinsics"]
        focal = 1 / (1 / intrinsics[..., 0, 0] ** 2 + 1 / intrinsics[..., 1, 1] ** 2) ** 0.5
        focal = focal * 0 + 1

        if self.sift_anchor:
            sift_mask = kwargs.get('sift_point_mask', None)
        else:
            sift_mask = None
        loss = 0
        for level, align_resolution, num_patches in zip([4, 16, 64], [16, 8, 4], [16, 256, 4096]):
            loss_, misc = affine_invariant_local_loss(
                prediction, target, mask, focal, global_scale=None, level=level, anchor_mask=sift_mask,
                align_resolution=align_resolution, num_patches=num_patches
            )
            loss += loss_.mean()

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            logging.warning(f"Data {name}, AffineIvariantLocalLoss NAN error, {loss}")

        return loss * loss_weight
