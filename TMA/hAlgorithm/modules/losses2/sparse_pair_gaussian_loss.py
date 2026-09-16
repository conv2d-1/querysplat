"""Rendering loss for sparse pair-query Dynamic 4D Gaussian Splatting."""
from __future__ import annotations

import logging
import random
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hAlgorithm.modules.losses2.render_loss import GaussianDepthGradLoss
from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import (
    align_sparse_query_geometry,
    bootstrap_sh_dc_from_rgb,
    build_sparse_dynamic_gaussians,
    canonicalize_scene_flow,
    compose_frame_means,
    compute_depth_gradient,
    compute_target_to_ref_visibility_masks,
    canonicalize_bnv_geometry,
    extract_frame_bnv_mask,
    ensure_bnv_cameras,
    estimate_wild_video_scale,
    extract_gaussian_batch,
    extract_view_scale,
    find_pair_index,
    render_sparse_gaussians_at_frame,
    resize_bnv_geometry,
    resolve_gaussian_render_resolution,
    sample_rgb_at_query_uv,
    squeeze_query_points,
    subsample_query_indices,
)

logger = logging.getLogger(__name__)


class SparsePairDynamicGaussianRenderLoss(nn.Module):
    """Render sparse query Gaussians using pair ``warp3d`` / ``warp3d_delta`` geometry."""

    def __init__(
        self,
        l1_weight: float = 0.5,
        ssim_weight: float = 0.2,
        opacity_reg_weight: float = 0.0,
        scale_reg_weight: float = 0.0,
        num_render_frames: int = 1,
        max_render_points: int = 8192,
        background_color: tuple = (1.0, 1.0, 1.0),
        render_in_normalized_space: bool = False,
        supervise_covered_pixels_only: bool = False,
        coverage_alpha_threshold: float = 0.01,
        min_coverage_pixels: int = 64,
        ref_frame_loss_weight: float = 1.0,
        supervise_ref_full_image: bool = False,
        use_ref_reproject_visibility: bool = False,
        ref_visibility_depth_ratio_low: float = 0.9,
        ref_visibility_depth_ratio_high: float = 1.1,
        ffgs_alignment: bool = False,
        alpha_bce_weight: float = 0.0,
        grad_loss_weight: float = 0.0,
        grad_loss_sigma: float = 0.01,
        grad_loss_epsilon: float = 0.01,
        lpips_weight: float = 0.0,
        render_random_background: bool = False,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.opacity_reg_weight = opacity_reg_weight
        self.scale_reg_weight = scale_reg_weight
        self.num_render_frames = num_render_frames
        self.max_render_points = max_render_points
        self.background_color = background_color
        self.render_in_normalized_space = render_in_normalized_space
        self.supervise_covered_pixels_only = supervise_covered_pixels_only
        self.coverage_alpha_threshold = coverage_alpha_threshold
        self.min_coverage_pixels = min_coverage_pixels
        self.ref_frame_loss_weight = ref_frame_loss_weight
        self.supervise_ref_full_image = supervise_ref_full_image
        self.use_ref_reproject_visibility = use_ref_reproject_visibility
        self.ref_visibility_depth_ratio_low = ref_visibility_depth_ratio_low
        self.ref_visibility_depth_ratio_high = ref_visibility_depth_ratio_high
        self.ffgs_alignment = ffgs_alignment
        self.alpha_bce_weight = alpha_bce_weight
        self.grad_loss_weight = grad_loss_weight
        self.lpips_weight = lpips_weight
        self.render_random_background = render_random_background
        self._lpips = None
        if lpips_weight > 0:
            from lpips import LPIPS
            from hAlgorithm.modules.losses2.render_loss import convert_to_buffer

            self._lpips = LPIPS(net="vgg")
            convert_to_buffer(self._lpips, persistent=False)
        self._grad_loss = (
            GaussianDepthGradLoss(
                loss_weight=1.0,
                sigma=grad_loss_sigma,
                epsilon=grad_loss_epsilon,
            )
            if grad_loss_weight > 0
            else None
        )
        self.infer_vis_gs_log_scale_bias = None  # None → auto shrink for wild-video gsplat vis

    @staticmethod
    def _compute_infer_vis_log_scale_bias(
        gs_scale: Tensor,
        means_t: Tensor,
        w2c: Tensor,
        intrinsics: Tensor,
        frame_idx: int,
        width: int,
    ) -> float:
        """Shrink oversized per-pixel splats so gsplat vis retains spatial detail.

        Training tolerates huge kernels under RGB loss; wild-video vis needs
        world-scale roughly one pixel wide: ``z / fx``.
        """
        w2c_f = w2c[0, frame_idx].float()
        k = intrinsics[0, frame_idx].float()
        fx = k[0, 0].clamp(min=1.0)
        pts = means_t[0].float()
        ones = torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)
        z = (w2c_f @ torch.cat([pts, ones], dim=1).T).T[:, 2].clamp(min=1e-3)
        z_med = z.median()
        pixel_world = (z_med / fx) * 1.5
        target_log = torch.log(pixel_world.clamp(min=5e-4))
        current_log = gs_scale.float().median()
        bias = float((target_log - current_log).item())
        return max(min(bias, -2.0), -12.0)

    @staticmethod
    def _ssim_loss(
        pred: Tensor,
        target: Tensor,
        window_size: int = 11,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        kernel = SparsePairDynamicGaussianRenderLoss._gaussian_kernel(
            window_size, sigma=1.5, channels=pred.shape[1], device=pred.device,
        )
        mu_p = F.conv2d(pred, kernel, padding=window_size // 2, groups=pred.shape[1])
        mu_t = F.conv2d(target, kernel, padding=window_size // 2, groups=pred.shape[1])
        mu_pp = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=pred.shape[1])
        mu_tt = F.conv2d(target * target, kernel, padding=window_size // 2, groups=pred.shape[1])
        mu_pt = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=pred.shape[1])
        sigma_p = mu_pp - mu_p * mu_p
        sigma_t = mu_tt - mu_t * mu_t
        sigma_pt = mu_pt - mu_p * mu_t
        ssim_map = (
            (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
            / ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2))
        )
        if mask is None:
            return 1.0 - ssim_map.mean()
        m = mask.float()
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            m = m.unsqueeze(0)
        if m.shape[-2:] != ssim_map.shape[-2:]:
            m = F.interpolate(m, size=ssim_map.shape[-2:], mode="nearest")
        denom = m.sum().clamp(min=1.0)
        return 1.0 - (ssim_map * m).sum() / denom

    @staticmethod
    def _masked_l1_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        m = mask.float().unsqueeze(0)
        diff = (pred - target).abs() * m
        return diff.sum() / m.sum().clamp(min=1.0)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float, channels: int, device: torch.device) -> Tensor:
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        return kernel.expand(channels, 1, size, size).contiguous()

    def _render(
        self,
        gaussians,
        w2c: Tensor,
        intrinsics: Tensor,
        frame_idx: int,
        height: int,
        width: int,
        batch_idx: int = 0,
        num_views: Optional[int] = None,
        return_alpha: bool = False,
        background_color: Optional[tuple] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return render_sparse_gaussians_at_frame(
            gaussians=gaussians,
            w2c=w2c,
            intrinsics=intrinsics,
            frame_idx=frame_idx,
            height=height,
            width=width,
            batch_idx=batch_idx,
            num_views=num_views,
            background_color=tuple(background_color or self.background_color),
            return_alpha=return_alpha,
        )

    def _sample_background_color(self, device: torch.device) -> tuple[float, float, float]:
        if self.render_random_background and self.training:
            bg = torch.rand(3, device=device)
            return float(bg[0]), float(bg[1]), float(bg[2])
        return tuple(self.background_color)

    @staticmethod
    def _to_lpips_nchw(tensor: Tensor) -> Tensor:
        """Normalize RGB to ``[1, 3, H, W]`` for LPIPS."""
        x = tensor
        while x.dim() > 4:
            x = x.squeeze(0)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected RGB [1, 3, H, W], got {tuple(x.shape)}")
        return x

    def _lpips_frame_loss(
        self,
        render_rgb: Tensor,
        gt: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        pred = self._to_lpips_nchw(render_rgb.float())
        target = self._to_lpips_nchw(gt.float())
        if mask is not None:
            m = mask.float()
            while m.dim() < 4:
                m = m.unsqueeze(0)
            if m.shape[1] == 1:
                m = m.expand(-1, 3, -1, -1)
            pred = pred * m
            target = target * m
        with torch.autocast(device_type=pred.device.type, enabled=False):
            loss = self._lpips(pred * 2.0 - 1.0, target * 2.0 - 1.0, normalize=True)
        return torch.nan_to_num(loss).mean()

    def _available_render_frames(
        self,
        pair_idx: list,
        ref_frame: int,
        num_views: int,
    ) -> list[int]:
        frames = {ref_frame}
        for t in range(num_views):
            if t == ref_frame:
                continue
            if find_pair_index(pair_idx, ref_frame, t) is not None:
                frames.add(t)
        return sorted(frames)

    def _metricize_sparse_geometry(
        self,
        results: dict,
        w2c: Tensor,
        scale: Optional[Tensor],
        denormalize_fn: Callable,
        num_views: int,
    ) -> tuple[dict, Tensor]:
        """Put sparse geometry / cameras into metric space (matches training).

        ``warp3d`` lives in the ref-camera frame, which equals the relative world
        frame when extrinsics are ``normalize_cameras``-aligned.  Camera translation
        must be denormalized with ``scale[..., 0, 0]`` to match ``get_inputs()``.
        """
        ref_frame = int(results["src_frame_idx"])
        means_ref = squeeze_query_points(results["sparse_global_points"].float())
        displacements = canonicalize_scene_flow(results["sparse_scene_flow"].float())
        means_ref, displacements = align_sparse_query_geometry(
            means_ref, displacements, results["gs_opacity"],
        )
        gs_scale = results["gs_scale"]
        if scale is not None:
            s_ref = extract_view_scale(scale, ref_frame)
            while s_ref.dim() < means_ref.dim():
                s_ref = s_ref.unsqueeze(-1)
            means_ref = denormalize_fn(means_ref, scale=s_ref)
            while s_ref.dim() < displacements.dim():
                s_ref = s_ref.unsqueeze(-2)
            displacements = denormalize_fn(displacements, scale=s_ref)
            w2c = w2c.clone()
            cam_scale = scale[..., 0, 0]
            while cam_scale.dim() < w2c[..., :3, 3].dim():
                cam_scale = cam_scale.unsqueeze(-1)
            w2c[..., :3, 3] = denormalize_fn(w2c[..., :3, 3], scale=cam_scale)
            log_s = torch.log(extract_view_scale(scale, ref_frame).clamp(min=1e-6))
            gs_scale = gs_scale + log_s

        means_ref = squeeze_query_points(means_ref)
        displacements = canonicalize_scene_flow(displacements)
        means_ref, displacements = align_sparse_query_geometry(
            means_ref, displacements, results["gs_opacity"],
        )
        payload = dict(results)
        payload["sparse_global_points"] = means_ref
        payload["sparse_scene_flow"] = displacements
        payload["gs_scale"] = gs_scale
        return payload, w2c

    def _prepare_infer_metric_points(
        self,
        results: dict,
        scale: Optional[Tensor],
        denormalize_fn: Callable,
    ) -> tuple[Tensor, Tensor]:
        """Metric sparse points to match denormalized ``w2c`` (motion UV convention)."""
        ref_frame = int(results["src_frame_idx"])
        means_ref = squeeze_query_points(results["sparse_global_points"].float())
        displacements = canonicalize_scene_flow(results["sparse_scene_flow"].float())
        means_ref, displacements = align_sparse_query_geometry(
            means_ref, displacements, results["gs_opacity"],
        )
        if scale is not None:
            s_ref = extract_view_scale(scale, ref_frame)
            while s_ref.dim() < means_ref.dim():
                s_ref = s_ref.unsqueeze(-1)
            means_ref = denormalize_fn(means_ref, scale=s_ref)
            while s_ref.dim() < displacements.dim():
                s_ref = s_ref.unsqueeze(-2)
            displacements = denormalize_fn(displacements, scale=s_ref)
        return means_ref, displacements

    def _means_at_frame(
        self,
        means_ref: Tensor,
        displacements: Tensor,
        frame_idx: int,
    ) -> Tensor:
        """Compose per-frame means in the ref-relative world frame."""
        return compose_frame_means(means_ref, displacements, frame_idx)

    @staticmethod
    def _bootstrap_infer_sh(
        gs_sh: Tensor,
        query,
        image: Tensor,
        ref_frame: int,
        query_idx: Optional[Tensor] = None,
    ) -> Tensor:
        """Fill SH DC with GT RGB at query UVs for infer visualization."""
        if query is None or not hasattr(query, "uv"):
            return gs_sh
        uv = query.uv.float()
        if uv.dim() == 4:
            uv = uv[:, ref_frame]
        if query_idx is not None:
            if uv.dim() == 2:
                uv = uv[query_idx]
            elif uv.dim() == 3:
                uv = uv[:, query_idx]
        rgb_anchor = sample_rgb_at_query_uv(
            image=image,
            query_uv=uv,
            ref_frame=ref_frame,
            batch_size=gs_sh.shape[0],
        )
        return bootstrap_sh_dc_from_rgb(gs_sh, rgb_anchor)

    @staticmethod
    def _extract_frame_depth_mask(
        target_depth_mask: Optional[Tensor],
        frame_idx: int,
        batch_idx: int = 0,
        batch_size: int = 1,
        num_views: int = 1,
    ) -> Optional[Tensor]:
        return extract_frame_bnv_mask(
            target_depth_mask, frame_idx, batch_size, num_views, batch_idx,
        )

    @staticmethod
    def _ensure_bnv_geometry(
        tensor: Optional[Tensor],
        batch_size: int,
        num_views: int,
    ) -> Optional[Tensor]:
        return canonicalize_bnv_geometry(tensor, batch_size, num_views)

    def _precompute_ref_visibility(
        self,
        target_local_depth: Optional[Tensor],
        target_depth_mask: Optional[Tensor],
        w2c: Tensor,
        intrinsics: Tensor,
        ref_frame: int,
        render_frames: list[int],
        height: int,
        width: int,
        batch_size: int,
        num_views: int,
    ) -> dict[int, Tensor]:
        if not self.use_ref_reproject_visibility or target_local_depth is None:
            return {}
        non_ref = [t for t in render_frames if t != ref_frame]
        if not non_ref:
            return {}
        target_local_depth = resize_bnv_geometry(
            target_local_depth, batch_size, num_views, height, width,
        )
        target_depth_mask = resize_bnv_geometry(
            target_depth_mask, batch_size, num_views, height, width, is_mask=True,
        )
        vis_masks = compute_target_to_ref_visibility_masks(
            target_local_depth=target_local_depth,
            target_depth_mask=target_depth_mask,
            w2c=w2c,
            intrinsics=intrinsics,
            ref_frame=ref_frame,
            target_frames=non_ref,
            depth_ratio_low=self.ref_visibility_depth_ratio_low,
            depth_ratio_high=self.ref_visibility_depth_ratio_high,
        )
        return {t: vis_masks[:, i] for i, t in enumerate(non_ref)}

    def _needs_render_alpha(self, frame_idx: int, ref_frame: int) -> bool:
        if self.alpha_bce_weight > 0:
            return True
        if self.ffgs_alignment:
            return False
        if frame_idx == ref_frame and self.supervise_ref_full_image:
            return False
        if self.supervise_covered_pixels_only:
            return True
        return False

    @staticmethod
    def _alpha_bce_loss(render_alpha: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        pred = render_alpha.float().clamp(1e-6, 1.0 - 1e-6)
        while pred.dim() > 2:
            pred = pred.squeeze(0) if pred.shape[0] == 1 else pred[0]
        target = torch.ones_like(pred)
        if mask is None:
            return F.binary_cross_entropy(pred, target)

        m = mask.float()
        while m.dim() > 2:
            m = m.squeeze(0) if m.shape[0] == 1 else m[0]
        if m.dim() == 1:
            m = m.unsqueeze(0)
        if m.shape[-2:] != pred.shape[-2:]:
            m = F.interpolate(
                m.unsqueeze(0).unsqueeze(0),
                size=pred.shape[-2:],
                mode="nearest",
            ).squeeze(0).squeeze(0)
        loss_map = F.binary_cross_entropy(pred, target, reduction="none")
        return (loss_map * m).sum() / m.sum().clamp(min=1.0)

    def _build_alpha_supervision_mask(
        self,
        frame_idx: int,
        ref_frame: int,
        ref_vis_mask: Optional[Tensor],
    ) -> Optional[Tensor]:
        if self.alpha_bce_weight <= 0:
            return None
        if self.ffgs_alignment:
            if frame_idx == ref_frame:
                return None
            return ref_vis_mask
        return None

    def _build_frame_coverage_mask(
        self,
        frame_idx: int,
        ref_frame: int,
        render_alpha: Optional[Tensor],
        target_depth_mask: Optional[Tensor],
        ref_vis_mask: Optional[Tensor],
        batch_size: int = 1,
        num_views: int = 1,
    ) -> Optional[Tensor]:
        depth_m = self._extract_frame_depth_mask(
            target_depth_mask, frame_idx, batch_size=batch_size, num_views=num_views,
        )
        if depth_m is not None:
            depth_m = depth_m > 0.5

        if self.ffgs_alignment:
            if frame_idx == ref_frame:
                return None
            mask = ref_vis_mask
            if depth_m is not None:
                mask = depth_m if mask is None else (mask & depth_m)
            return mask

        if frame_idx == ref_frame and self.supervise_ref_full_image:
            return depth_m

        mask = None
        if self.supervise_covered_pixels_only and render_alpha is not None:
            mask = render_alpha > self.coverage_alpha_threshold
        if self.use_ref_reproject_visibility and ref_vis_mask is not None:
            mask = ref_vis_mask if mask is None else (mask & ref_vis_mask)
        if depth_m is not None:
            mask = depth_m if mask is None else (mask & depth_m)
        return mask

    def _compute_ref_grad_loss(
        self,
        name: str,
        means_ref: Tensor,
        gs_opacity: Tensor,
        gs_scale: Tensor,
        gs_rotation: Tensor,
        gs_sh: Tensor,
        gs_opacity_delta: Optional[Tensor],
        gs_scale_delta: Optional[Tensor],
        gs_rotation_delta: Optional[Tensor],
        target_local_depth: Tensor,
        target_depth_mask: Optional[Tensor],
        w2c: Tensor,
        intrinsics: Tensor,
        ref_frame: int,
        height: int,
        width: int,
        num_views: int,
    ) -> tuple[Tensor, dict]:
        if self._grad_loss is None:
            return torch.zeros((), device=means_ref.device), {}

        target_local_depth = canonicalize_bnv_geometry(
            target_local_depth, means_ref.shape[0], num_views,
        )
        if target_local_depth is None or target_local_depth.dim() != 5:
            return torch.zeros((), device=means_ref.device), {}

        batch_size = target_local_depth.shape[0]
        ref_depth = target_local_depth[:, ref_frame]
        if ref_depth.shape[1] == 1:
            z = ref_depth[:, 0]
            k = intrinsics[:, ref_frame]
            ys = torch.arange(height, device=z.device, dtype=z.dtype).view(1, 1, height, 1).expand(z.shape[0], 1, height, width)
            xs = torch.arange(width, device=z.device, dtype=z.dtype).view(1, 1, 1, width).expand(z.shape[0], 1, height, width)
            fx = k[:, 0, 0].view(-1, 1, 1, 1)
            fy = k[:, 1, 1].view(-1, 1, 1, 1)
            cx = k[:, 0, 2].view(-1, 1, 1, 1)
            cy = k[:, 1, 2].view(-1, 1, 1, 1)
            ref_depth = torch.stack([
                (xs - cx) * z.unsqueeze(1) / fx.clamp(min=1e-6),
                (ys - cy) * z.unsqueeze(1) / fy.clamp(min=1e-6),
                z.unsqueeze(1),
            ], dim=1)
        elif ref_depth.shape[1] > 3:
            ref_depth = ref_depth[:, :3]

        ref_mask = self._extract_frame_depth_mask(
            target_depth_mask, ref_frame, batch_size=batch_size, num_views=num_views,
        )
        if ref_mask is not None and ref_mask.shape[-2:] != (height, width):
            ref_mask = F.interpolate(
                ref_mask.unsqueeze(0), size=(height, width), mode="nearest",
            ).squeeze(0)

        grad_mag, _, valid_grad = compute_depth_gradient(ref_depth, ref_mask)
        grad_map = (grad_mag * valid_grad.to(grad_mag.dtype)).unsqueeze(1).unsqueeze(1)

        ref_gaussians = build_sparse_dynamic_gaussians(
            gs_opacity=gs_opacity,
            gs_scale=gs_scale,
            gs_rotation=gs_rotation,
            gs_sh=gs_sh,
            means=means_ref,
            gs_opacity_delta=gs_opacity_delta,
            gs_scale_delta=gs_scale_delta,
            gs_rotation_delta=gs_rotation_delta,
            target_frame_idx=ref_frame,
        )
        means, _, _, opacities, _ = extract_gaussian_batch(ref_gaussians, batch_idx=0)
        gs_dict = {
            "means": means.unsqueeze(0),
            "opacities": opacities.unsqueeze(0),
        }
        grad_loss = self._grad_loss(
            name=name,
            depth_grad=grad_map,
            gaussians=gs_dict,
            w2c=w2c[:, ref_frame : ref_frame + 1],
            intrinsics=intrinsics[:, ref_frame : ref_frame + 1],
        )
        return self.grad_loss_weight * grad_loss, {
            "dgs_grad_loss": round((self.grad_loss_weight * grad_loss).item(), 4),
        }

    def forward(
        self,
        name: str,
        results: dict,
        image: Tensor,
        intrinsics: Tensor,
        w2c: Tensor,
        scale: Optional[Tensor],
        denormalize_fn: Callable,
        meta_data: dict,
        global_step: int = 0,
        pair_idx: Optional[list] = None,
        target_local_depth: Optional[Tensor] = None,
        target_depth_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, dict]:
        def _zero():
            return torch.zeros((), device=image.device, requires_grad=False), {}

        required = ("gs_opacity", "gs_scale", "gs_rotation", "gs_sh",
                    "sparse_global_points", "sparse_scene_flow", "src_frame_idx")
        if not all(k in results for k in required):
            return _zero()

        pair_idx = pair_idx or results.get("pair_idx")
        if pair_idx is None:
            return _zero()

        (
            image,
            intrinsics,
            target_local_depth,
            target_depth_mask,
            H,
            W,
        ) = resolve_gaussian_render_resolution(
            image=image,
            intrinsics=intrinsics,
            target_local_depth=target_local_depth,
            target_depth_mask=target_depth_mask,
            gaussian_query=results.get("gaussian_query"),
        )
        B, N = image.shape[0], image.shape[1]
        ref_frame = int(results["src_frame_idx"])
        target_local_depth = self._ensure_bnv_geometry(target_local_depth, B, N)
        target_depth_mask = self._ensure_bnv_geometry(target_depth_mask, B, N)
        target_local_depth_resized = resize_bnv_geometry(
            target_local_depth, B, N, H, W,
        )
        target_depth_mask_resized = resize_bnv_geometry(
            target_depth_mask, B, N, H, W, is_mask=True,
        )
        results, w2c = self._metricize_sparse_geometry(
            results, w2c, scale, denormalize_fn, num_views=N,
        )

        w2c = ensure_bnv_cameras(w2c, N, (4, 4))
        intrinsics = ensure_bnv_cameras(intrinsics, N, (3, 3))

        (
            means_ref,
            displacements,
            gs_opacity,
            gs_scale,
            gs_rotation,
            gs_sh,
            gs_opacity_delta,
            gs_scale_delta,
            gs_rotation_delta,
            _query_idx,
        ) = self._subsample_gs_tensors(results, results["sparse_global_points"], results["sparse_scene_flow"])

        available = self._available_render_frames(pair_idx, ref_frame, N)
        if not available:
            return _zero()

        target_candidates = [t for t in available if t != ref_frame]
        if not target_candidates:
            target_candidates = [ref_frame]
        render_targets = random.sample(
            target_candidates,
            min(self.num_render_frames, len(target_candidates)),
        )
        render_frames = [ref_frame] + [t for t in render_targets if t not in (ref_frame,)]

        ref_vis_by_frame = self._precompute_ref_visibility(
            target_local_depth=target_local_depth_resized,
            target_depth_mask=target_depth_mask_resized,
            w2c=w2c,
            intrinsics=intrinsics,
            ref_frame=ref_frame,
            render_frames=render_frames,
            height=H,
            width=W,
            batch_size=B,
            num_views=N,
        )

        total_loss = torch.zeros((), device=image.device)
        rgb_loss_accum = torch.zeros((), device=image.device)
        loss_dict: dict = {}
        success = 0
        total_weight = 0.0

        if self.ffgs_alignment and self.grad_loss_weight > 0 and target_local_depth_resized is not None:
            grad_loss, grad_dict = self._compute_ref_grad_loss(
                name=name,
                means_ref=means_ref,
                gs_opacity=gs_opacity,
                gs_scale=gs_scale,
                gs_rotation=gs_rotation,
                gs_sh=gs_sh,
                gs_opacity_delta=gs_opacity_delta,
                gs_scale_delta=gs_scale_delta,
                gs_rotation_delta=gs_rotation_delta,
                target_local_depth=target_local_depth_resized,
                target_depth_mask=target_depth_mask_resized,
                w2c=w2c,
                intrinsics=intrinsics,
                ref_frame=ref_frame,
                height=H,
                width=W,
                num_views=N,
            )
            total_loss = total_loss + grad_loss
            loss_dict.update(grad_dict)

        for t in render_frames:
            means_t = self._means_at_frame(means_ref, displacements, t)
            need_alpha = self._needs_render_alpha(t, ref_frame)
            frame_bg = self._sample_background_color(image.device)
            try:
                gaussians = build_sparse_dynamic_gaussians(
                    gs_opacity=gs_opacity,
                    gs_scale=gs_scale,
                    gs_rotation=gs_rotation,
                    gs_sh=gs_sh,
                    means=means_t,
                    gs_opacity_delta=gs_opacity_delta,
                    gs_scale_delta=gs_scale_delta,
                    gs_rotation_delta=gs_rotation_delta,
                    target_frame_idx=t,
                )
                render_out = self._render(
                    gaussians,
                    w2c=w2c,
                    intrinsics=intrinsics,
                    frame_idx=t,
                    height=H,
                    width=W,
                    num_views=N,
                    return_alpha=need_alpha,
                    background_color=frame_bg,
                )
                if need_alpha:
                    render_rgb, render_alpha = render_out
                else:
                    render_rgb = render_out
                    render_alpha = None
            except Exception as exc:
                logger.warning(
                    "SparsePairDGS render failed frame %d: %s: %s",
                    t, type(exc).__name__, exc,
                )
                continue

            gt = (image[0, t].float() + 1.0) * 0.5
            ref_vis = ref_vis_by_frame.get(t)
            if ref_vis is not None and ref_vis.shape[0] == 1:
                ref_vis = ref_vis[0]
            coverage_mask = self._build_frame_coverage_mask(
                frame_idx=t,
                ref_frame=ref_frame,
                render_alpha=render_alpha,
                target_depth_mask=target_depth_mask_resized,
                ref_vis_mask=ref_vis,
                batch_size=B,
                num_views=N,
            )
            if coverage_mask is not None and int(coverage_mask.sum().item()) < self.min_coverage_pixels:
                logger.debug(
                    "SparsePairDGS skip frame %d: covered pixels %d < %d",
                    t,
                    int(coverage_mask.sum().item()),
                    self.min_coverage_pixels,
                )
                continue

            frame_loss = torch.tensor(0.0, device=image.device)
            if self.l1_weight > 0:
                if coverage_mask is not None:
                    frame_loss = frame_loss + self.l1_weight * self._masked_l1_loss(
                        render_rgb, gt, coverage_mask,
                    )
                else:
                    frame_loss = frame_loss + self.l1_weight * F.l1_loss(render_rgb, gt)
            if self.ssim_weight > 0:
                frame_loss = frame_loss + self.ssim_weight * self._ssim_loss(
                    render_rgb.unsqueeze(0),
                    gt.unsqueeze(0),
                    mask=coverage_mask,
                )
            if self.lpips_weight > 0 and self._lpips is not None:
                lpips_loss = self._lpips_frame_loss(render_rgb, gt, coverage_mask)
                frame_loss = frame_loss + self.lpips_weight * lpips_loss
                loss_dict[f"dgs_lpips_f{t}"] = round(lpips_loss.item(), 4)
            if self.alpha_bce_weight > 0 and render_alpha is not None:
                alpha_mask = self._build_alpha_supervision_mask(t, ref_frame, ref_vis)
                alpha_loss = self._alpha_bce_loss(render_alpha, alpha_mask)
                frame_loss = frame_loss + self.alpha_bce_weight * alpha_loss
                loss_dict[f"dgs_alpha_bce_f{t}"] = round(alpha_loss.item(), 4)
            frame_weight = self.ref_frame_loss_weight if t == ref_frame else 1.0
            rgb_loss_accum = rgb_loss_accum + frame_weight * torch.nan_to_num(frame_loss)
            total_weight += frame_weight
            loss_dict[f"dgs_rgb_loss_f{t}"] = round(frame_loss.item(), 4)
            if coverage_mask is not None:
                loss_dict[f"dgs_coverage_ratio_f{t}"] = round(
                    float(coverage_mask.float().mean().item()), 4,
                )
            if t in ref_vis_by_frame:
                loss_dict[f"dgs_ref_vis_ratio_f{t}"] = round(
                    float(ref_vis_by_frame[t].float().mean().item()), 4,
                )
            success += 1

        if success == 0 and "dgs_grad_loss" not in loss_dict:
            return _zero()

        if success > 0:
            total_loss = total_loss + rgb_loss_accum / max(total_weight, 1.0)
        loss_dict["dgs_rgb_loss"] = round(
            (rgb_loss_accum / max(total_weight, 1.0)).item() if success > 0 else 0.0, 4,
        )

        if self.opacity_reg_weight > 0:
            op = gs_opacity.clamp(1e-6, 1.0 - 1e-6)
            op_reg = -(op * op.log() + (1 - op) * (1 - op).log()).mean()
            total_loss = total_loss + self.opacity_reg_weight * op_reg
            loss_dict["dgs_opacity_reg"] = round(op_reg.item(), 4)

        if self.scale_reg_weight > 0:
            sc_reg = gs_scale.abs().mean()
            total_loss = total_loss + self.scale_reg_weight * sc_reg
            loss_dict["dgs_scale_reg"] = round(sc_reg.item(), 4)

        return total_loss, loss_dict

    def _subsample_gs_tensors(
        self,
        results: dict,
        means_ref: Tensor,
        displacements: Tensor,
        query_idx: Optional[Tensor] = None,
    ):
        """Subsample query points for rendering when Q exceeds ``max_render_points``."""
        means_ref = squeeze_query_points(means_ref)
        displacements = canonicalize_scene_flow(displacements)
        q = min(
            means_ref.shape[1],
            displacements.shape[2],
            results["gs_opacity"].shape[1],
            results["gs_scale"].shape[1],
            results["gs_rotation"].shape[1],
            results["gs_sh"].shape[1],
        )
        means_ref = means_ref[:, :q]
        displacements = displacements[:, :, :q]
        gs_opacity = results["gs_opacity"][:, :q]
        gs_scale = results["gs_scale"][:, :q]
        gs_rotation = results["gs_rotation"][:, :q]
        gs_sh = results["gs_sh"][:, :q]

        if query_idx is None:
            query_idx = subsample_query_indices(q, self.max_render_points, means_ref.device)

        gs_opacity = gs_opacity[:, query_idx]
        gs_scale = gs_scale[:, query_idx]
        gs_rotation = gs_rotation[:, query_idx]
        gs_sh = gs_sh[:, query_idx]
        means_ref = means_ref[:, query_idx]
        displacements = displacements[:, :, query_idx]

        gs_opacity_delta = results.get("gs_opacity_delta")
        gs_scale_delta = results.get("gs_scale_delta")
        gs_rotation_delta = results.get("gs_rotation_delta")
        if gs_opacity_delta is not None:
            gs_opacity_delta = gs_opacity_delta[:, :, :q][:, :, query_idx]
            gs_scale_delta = gs_scale_delta[:, :, :q][:, :, query_idx]
            gs_rotation_delta = gs_rotation_delta[:, :, :q][:, :, query_idx]

        return (
            means_ref,
            displacements,
            gs_opacity,
            gs_scale,
            gs_rotation,
            gs_sh,
            gs_opacity_delta,
            gs_scale_delta,
            gs_rotation_delta,
            query_idx,
        )

    @torch.no_grad()
    def _render_frames_infer(
        self,
        results: dict,
        intrinsics: Tensor,
        w2c: Tensor,
        image_shape: tuple[int, int],
        scale: Optional[Tensor],
        denormalize_fn: Callable,
        max_frames: Optional[int],
        infer_image: Tensor,
    ) -> dict:
        """Infer / val visualization via gsplat rasterization."""
        H, W = image_shape
        ref_frame = int(results.get("src_frame_idx", 0))
        N = canonicalize_scene_flow(results["sparse_scene_flow"].float()).shape[1]
        w2c = ensure_bnv_cameras(w2c, N, (4, 4))
        intrinsics = ensure_bnv_cameras(intrinsics, N, (3, 3))

        if scale is None and not self.render_in_normalized_space:
            scale = estimate_wild_video_scale(results, num_views=N, device=w2c.device)

        render_results, w2c = self._metricize_sparse_geometry(
            results, w2c, scale, denormalize_fn, num_views=N,
        )
        means_ref = squeeze_query_points(render_results["sparse_global_points"].float())
        displacements = canonicalize_scene_flow(
            render_results["sparse_scene_flow"].float(),
        )
        means_ref, displacements = align_sparse_query_geometry(
            means_ref, displacements, render_results["gs_opacity"],
        )

        (
            means_ref,
            displacements,
            gs_opacity,
            gs_scale,
            gs_rotation,
            gs_sh,
            gs_opacity_delta,
            gs_scale_delta,
            gs_rotation_delta,
            query_idx,
        ) = self._subsample_gs_tensors(
            render_results,
            means_ref,
            displacements,
        )

        gs_sh = self._bootstrap_infer_sh(
            gs_sh, results.get("query"), infer_image, ref_frame, query_idx=query_idx,
        )

        rgb_list = []
        for t in range(N):
            if max_frames is not None and t >= max_frames:
                break
            means_t = self._means_at_frame(means_ref, displacements, t)
            if self.infer_vis_gs_log_scale_bias is not None:
                log_bias = float(self.infer_vis_gs_log_scale_bias)
            elif self.render_in_normalized_space:
                # Match training render: no auto-shrink in normalized space.
                log_bias = 0.0
            else:
                log_bias = self._compute_infer_vis_log_scale_bias(
                    gs_scale, means_t, w2c, intrinsics, frame_idx=t, width=W,
                )
            gs_scale_vis = gs_scale + log_bias
            try:
                gaussians = build_sparse_dynamic_gaussians(
                    gs_opacity=gs_opacity,
                    gs_scale=gs_scale_vis,
                    gs_rotation=gs_rotation,
                    gs_sh=gs_sh,
                    means=means_t,
                    gs_opacity_delta=gs_opacity_delta,
                    gs_scale_delta=gs_scale_delta,
                    gs_rotation_delta=gs_rotation_delta,
                    target_frame_idx=t,
                )
                rgb = render_sparse_gaussians_at_frame(
                    gaussians=gaussians,
                    w2c=w2c,
                    intrinsics=intrinsics,
                    frame_idx=t,
                    height=H,
                    width=W,
                    num_views=N,
                    background_color=tuple(self.background_color),
                )
            except Exception as exc:
                logger.warning(
                    "SparsePairDGS gsplat infer failed frame %d: %s: %s",
                    t, type(exc).__name__, exc,
                )
                continue
            rgb_list.append(rgb.unsqueeze(0).unsqueeze(0))

        if not rgb_list:
            return {}
        return {"dgs_render_rgb": torch.cat(rgb_list, dim=1)}

    @torch.no_grad()
    def render_frames(
        self,
        results: dict,
        intrinsics: Tensor,
        w2c: Tensor,
        image_shape: tuple[int, int],
        scale: Optional[Tensor] = None,
        denormalize_fn: Optional[Callable] = None,
        max_frames: Optional[int] = None,
        infer_image: Optional[Tensor] = None,
    ) -> dict:
        """Infer / val visualization via gsplat (same backend as training render)."""
        if not all(k in results for k in ("gs_opacity", "sparse_global_points", "sparse_scene_flow")):
            return {}
        if infer_image is None or denormalize_fn is None:
            return {}

        return self._render_frames_infer(
            results=results,
            intrinsics=intrinsics,
            w2c=w2c,
            image_shape=image_shape,
            scale=scale,
            denormalize_fn=denormalize_fn,
            max_frames=max_frames,
            infer_image=infer_image,
        )
