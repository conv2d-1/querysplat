"""Render loss v2: RGB + target depth + floater suppression."""
from __future__ import annotations

import logging
import random
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from hAlgorithm.modules.losses2.sparse_pair_gaussian_loss import (
    SparsePairDynamicGaussianRenderLoss,
)
from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import (
    align_sparse_query_geometry,
    build_sparse_dynamic_gaussians,
    canonicalize_scene_flow,
    compose_frame_means,
    ensure_bnv_cameras,
    render_sparse_gaussians_rgb_depth_at_frame,
    resize_bnv_geometry,
    resolve_gaussian_render_resolution,
    squeeze_query_points,
    subsample_query_indices,
)

logger = logging.getLogger(__name__)


class SparsePairDynamicGaussianRenderLossV2(SparsePairDynamicGaussianRenderLoss):
    """Extends v1 render loss with supervised target depth and floater penalty.

    Uses a single ``RGB+ED`` gsplat pass per frame (no duplicate renders).
    """

    def __init__(
        self,
        target_depth_weight: float = 0.1,
        floater_weight: float = 0.01,
        floater_depth_margin: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_depth_weight = target_depth_weight
        self.floater_weight = floater_weight
        self.floater_depth_margin = floater_depth_margin

    @staticmethod
    def _extract_gt_depth(target_depth: Tensor, frame_idx: int) -> Tensor:
        depth = target_depth
        if depth.dim() == 5:
            depth = depth[0, frame_idx]
        elif depth.dim() == 4:
            depth = depth[frame_idx]
        if depth.shape[0] == 3:
            depth = depth[2:3]
        elif depth.shape[0] > 1:
            depth = depth[:1]
        return depth.float()

    @staticmethod
    def _extract_depth_mask(target_depth_mask: Optional[Tensor], frame_idx: int) -> Optional[Tensor]:
        if target_depth_mask is None:
            return None
        mask = target_depth_mask
        if mask.dim() == 5:
            mask = mask[0, frame_idx]
        elif mask.dim() == 4:
            mask = mask[frame_idx]
        if mask.shape[0] > 1:
            mask = mask[:1]
        return mask.float()

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
        target_depth: Optional[Tensor] = None,
        target_depth_mask: Optional[Tensor] = None,
        target_local_depth: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, dict]:
        def _zero():
            return torch.zeros((), device=image.device, requires_grad=False), {}

        required = (
            "gs_opacity", "gs_scale", "gs_rotation", "gs_sh",
            "sparse_global_points", "sparse_scene_flow", "src_frame_idx",
        )
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
        ) = self._subsample_gs_tensors(
            results, results["sparse_global_points"], results["sparse_scene_flow"],
        )

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

        use_depth = (
            target_depth is not None
            and (self.target_depth_weight > 0 or self.floater_weight > 0)
        )

        total_loss = torch.zeros((), device=image.device)
        loss_dict: dict = {}
        rgb_losses: list[Tensor] = []
        rgb_weights: list[float] = []
        depth_losses: list[Tensor] = []
        floater_losses: list[Tensor] = []
        success = 0

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
                if use_depth:
                    depth_out = render_sparse_gaussians_rgb_depth_at_frame(
                        gaussians=gaussians,
                        w2c=w2c,
                        intrinsics=intrinsics,
                        frame_idx=t,
                        height=H,
                        width=W,
                        num_views=N,
                        background_color=tuple(self.background_color),
                        return_alpha=need_alpha,
                    )
                    if need_alpha:
                        render_rgb, render_depth, render_alpha = depth_out
                    else:
                        render_rgb, render_depth = depth_out
                        render_alpha = None
                else:
                    render_out = self._render(
                        gaussians,
                        w2c=w2c,
                        intrinsics=intrinsics,
                        frame_idx=t,
                        height=H,
                        width=W,
                        num_views=N,
                        return_alpha=need_alpha,
                    )
                    if need_alpha:
                        render_rgb, render_alpha = render_out
                    else:
                        render_rgb = render_out
                        render_alpha = None
                    render_depth = None
            except Exception as exc:
                logger.warning(
                    "SparsePairDGS v2 render failed frame %d: %s: %s",
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
                    "SparsePairDGS v2 skip frame %d: covered pixels %d < %d",
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
            if self.alpha_bce_weight > 0 and render_alpha is not None:
                alpha_mask = self._build_alpha_supervision_mask(t, ref_frame, ref_vis)
                alpha_loss = self._alpha_bce_loss(render_alpha, alpha_mask)
                frame_loss = frame_loss + self.alpha_bce_weight * alpha_loss
                loss_dict[f"dgs_alpha_bce_f{t}"] = round(alpha_loss.item(), 4)
            frame_loss = torch.nan_to_num(frame_loss)
            frame_weight = self.ref_frame_loss_weight if t == ref_frame else 1.0
            rgb_losses.append(frame_loss)
            rgb_weights.append(frame_weight)
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

            if use_depth and render_depth is not None:
                gt_depth = self._extract_gt_depth(target_depth, t)
                if gt_depth.shape[-2:] != render_depth.shape[-2:]:
                    gt_depth = F.interpolate(
                        gt_depth.unsqueeze(0),
                        size=render_depth.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    ).squeeze(0)
                mask = self._extract_depth_mask(target_depth_mask, t)
                if mask is not None and mask.shape[-2:] != render_depth.shape[-2:]:
                    mask = F.interpolate(
                        mask.unsqueeze(0),
                        size=render_depth.shape[-2:],
                        mode="nearest",
                    ).squeeze(0)
                valid = render_depth > 1e-4
                if mask is not None:
                    valid = valid & (mask > 0.5)
                if valid.sum() >= 1:
                    if self.target_depth_weight > 0:
                        depth_losses.append(F.l1_loss(render_depth[valid], gt_depth[valid]))
                    if self.floater_weight > 0:
                        floater_mask = (render_depth - gt_depth) > self.floater_depth_margin
                        floater_mask = floater_mask & valid
                        if floater_mask.any():
                            floater_losses.append(
                                (render_depth[floater_mask] - gt_depth[floater_mask]).mean(),
                            )

        if success == 0:
            return _zero()

        weight_t = torch.tensor(rgb_weights, device=image.device, dtype=rgb_losses[0].dtype)
        rgb_term = (torch.stack(rgb_losses) * weight_t).sum() / weight_t.sum().clamp(min=1.0)
        total_loss = total_loss + rgb_term
        loss_dict["dgs_rgb_loss"] = round(rgb_term.item(), 4)

        if depth_losses:
            depth_term = torch.stack(depth_losses).mean() * self.target_depth_weight
            total_loss = total_loss + depth_term
            loss_dict["dgs_target_depth_loss"] = round(depth_term.item(), 4)
        if floater_losses:
            floater_term = torch.stack(floater_losses).mean() * self.floater_weight
            total_loss = total_loss + floater_term
            loss_dict["dgs_floater_loss"] = round(floater_term.item(), 4)

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
