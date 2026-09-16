import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fused_ssim import FusedSSIMMap
from lpips import LPIPS

from hAlgorithm.modules.utils.normal import get_surface_normalv2

from .base import Loss, check_and_fix_inf_nan


class RGBL1Loss(Loss):
    def __init__(self, loss_weight=1.0):
        super(RGBL1Loss, self).__init__(loss_weight=loss_weight)

    def forward(self, rgbs, render_rgbs, mask=None, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(render_rgbs))

        if mask is not None:
            loss = ((render_rgbs - rgbs) * mask).abs().sum() / (mask.sum() * rgbs.shape[-3])
        else:
            loss = (render_rgbs - rgbs).abs().mean()
        loss = torch.nan_to_num(loss)

        return loss * loss_weight


class RenderAlphaBCELoss(Loss):
    def __init__(self, loss_weight=0.1, eps=1e-6):
        super(RenderAlphaBCELoss, self).__init__(loss_weight=loss_weight)
        self.eps = eps

    def forward(self, prediction, mask=None, name=None, **kwargs):
        loss_weight = self.get_loss_weight(name)
        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(prediction))

        prediction = prediction.float().clamp(min=self.eps, max=1.0 - self.eps)
        target = torch.ones_like(prediction)

        if mask is not None:
            mask = mask.to(prediction.dtype)
            while mask.ndim < prediction.ndim:
                mask = mask.unsqueeze(1)
            mask = mask.expand_as(prediction)
            loss_map = F.binary_cross_entropy(prediction, target, reduction="none")
            denom = mask.sum().clamp(min=1.0)
            loss = (loss_map * mask).sum() / denom
        else:
            loss = F.binary_cross_entropy(prediction, target, reduction="mean")

        loss = torch.nan_to_num(loss)
        return loss * loss_weight


class SSIMLoss(Loss):
    def __init__(self, loss_weight=0.2):
        super(SSIMLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, rgbs, render_rgbs, mask=None, name=None, **kwargs):
        """
        Compute the SSIM loss between the rendered rgbs and the target rgbs.

        Args:
            rgbs (torch.Tensor): Target rgbs images of shape (B, C, H, W).
            render_rgbs (torch.Tensor): Rendered rgbs images of shape (B, C, H, W).
            mask (torch.Tensor, optional): Mask to apply to the loss. Shape (B, 1, H, W).
            name (str, optional): Name to retrieve specific loss weight from dictionary.
            **kwargs: Additional keyword arguments (unused here).

        Returns:
            torch.Tensor: The computed SSIM loss.
        """
        # Retrieve the loss weight
        loss_weight = self.get_loss_weight(name)
        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(render_rgbs))

        # Constants for SSIM calculation
        C1 = 0.01**2
        C2 = 0.03**2

        # Reshape tensors for SSIM calculation
        batch_size = rgbs.shape[0]
        rgbs_reshaped = rgbs.reshape(-1, 3, rgbs.shape[-2], rgbs.shape[-1])
        render_rgbs_reshaped = render_rgbs.reshape(-1, 3, render_rgbs.shape[-2], render_rgbs.shape[-1])

        # Compute SSIM map
        ssim_map = FusedSSIMMap.apply(C1, C2, render_rgbs_reshaped, rgbs_reshaped, "same", True)
        ssim_map = torch.nan_to_num(ssim_map)

        # Apply mask if provided
        if mask is not None:
            mask_reshaped = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])
            loss = (ssim_map * mask_reshaped).sum() / (mask.sum() * rgbs.shape[-3])
        else:
            loss = ssim_map.mean()

        # Convert SSIM map to loss value
        loss = 1 - loss
        loss = torch.nan_to_num(loss)

        return loss * loss_weight


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


class LpipsLoss(Loss):
    def __init__(self, loss_weight=0.05) -> None:
        super(LpipsLoss, self).__init__(loss_weight=loss_weight)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        rgbs,
        render_rgbs,
        mask=None,
        name=None,
        **kwargs,
    ):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(render_rgbs))

        if mask is not None:
            render_rgbs = render_rgbs * mask
            rgbs = rgbs * mask

        loss = self.lpips.forward(
            rearrange(render_rgbs, "b v c h w -> (b v) c h w"),
            rearrange(rgbs, "b v c h w -> (b v) c h w"),
            normalize=True,
        )

        loss = torch.nan_to_num(loss)
        loss = loss.mean()

        return loss * loss_weight


class NormalLoss(Loss):
    def __init__(self, loss_weight=0.1):
        super(NormalLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, depth, render_normal, mask=None, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(render_normal))

        depth = rearrange(depth, "b v c h w -> (b v) h w c")
        target_normal, normal_mask = get_surface_normalv2(depth)
        target_normal = rearrange(target_normal, "b c h w -> b h w c")
        render_normal = rearrange(render_normal, "b v c h w -> (b v) h w c")
        loss = ((render_normal[normal_mask] - target_normal[normal_mask]) ** 2).mean()
        loss = torch.nan_to_num(loss)

        return loss * loss_weight


def pearson_depth_loss(depth_src, depth_target, weight=0.05):
    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()
    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)
    co = (src * target).mean()
    assert not torch.any(torch.isnan(co))
    return (1 - co) * weight


def local_pearson_loss(depth_src, depth_target, box_p=128, p_corr=0.5, weight=0.25):
    # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
    num_box_h = (depth_src.shape[0] / box_p).floor()
    num_box_w = (depth_src.shape[1] / box_p).floor()
    max_h = depth_src.shape[0] - box_p
    max_w = depth_src.shape[1] - box_p
    _loss = torch.tensor(0.0, device="cuda")
    n_corr = int(p_corr * num_box_h * num_box_w)
    x_0 = torch.randint(0, max_h, size=(n_corr,), device="cuda")
    y_0 = torch.randint(0, max_w, size=(n_corr,), device="cuda")
    x_1 = x_0 + box_p
    y_1 = y_0 + box_p
    _loss = torch.tensor(0.0, device="cuda")
    for i in range(len(x_0)):
        _loss += pearson_depth_loss(depth_src[x_0[i] : x_1[i], y_0[i] : y_1[i]].reshape(-1), depth_target[x_0[i] : x_1[i], y_0[i] : y_1[i]].reshape(-1))
    return (_loss / n_corr) * weight


class RenderDpethLossV1(Loss):
    def __init__(
        self,
        loss_weight=1,
        threshold=None,
        zweighted=False,
        normal=False,
    ):
        super().__init__(loss_weight=loss_weight)

        self.threshold = threshold
        self.zweighted = zweighted
        self.normal = normal

    def forward(self, prediction, target, mask=None, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(prediction)

        if prediction.shape[-1] == 3:
            prediction = prediction[..., -1]
        if target.shape[-1] == 3:
            target = target[..., -1]

        if self.normal:
            prediction = prediction / prediction.abs().max()
            target = target / prediction.abs().max()

        if mask is not None:
            diff_raw = torch.abs(prediction[mask] - target[mask])
        else:
            diff_raw = torch.abs(prediction - target)

        diff_raw = diff_raw.squeeze()

        if self.zweighted:
            if mask is not None:
                diff = diff_raw / (target.abs() + 1e-6)
            else:
                diff = diff_raw / (target.abs() + 1e-6)
        else:
            diff = diff_raw

        diff = check_and_fix_inf_nan(diff, "diff")

        if self.threshold is not None:
            loss = diff[diff < self.threshold].mean()
        else:
            loss = diff.mean()

        return loss * loss_weight


class RenderDpethLossV2(Loss):
    def __init__(
        self,
        loss_weight=1,
        threshold=None,
        zweighted=False,
        normal=False,
    ):
        super().__init__(loss_weight=loss_weight)

        self.threshold = threshold
        self.zweighted = zweighted
        self.normal = normal

    def forward(self, prediction, target, mask=None, name=None, **kwargs):
        if target is None:
            return None

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(prediction)

        if prediction.shape[-1] == 3:
            prediction = prediction[..., -1]
        if target.shape[-1] == 3:
            target = target[..., -1]

        loss = pearson_depth_loss(prediction, target)
        return loss


class GaussianDepthGradLoss(Loss):
    """Edge-aware opacity regularizer from dev_lx FFGS (``sharp_dpt`` path)."""

    def __init__(self, loss_weight=1.0, sigma=0.01, epsilon=0.01, detach_depth_grad=True):
        super().__init__(loss_weight=loss_weight)
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)
        self.detach_depth_grad = detach_depth_grad

    def forward(
        self,
        depth_grad,
        gaussians,
        w2c,
        intrinsics,
        name=None,
        return_per_point=False,
        **kwargs,
    ):
        loss_weight = self.get_loss_weight(name)

        means = gaussians["means"]
        opacities = gaussians["opacities"]
        if opacities.ndim == 3 and opacities.shape[-1] == 1:
            opacities = opacities.squeeze(-1)
        if opacities.ndim != 2:
            raise ValueError(f"Unexpected opacities shape: {tuple(opacities.shape)}")

        B, P = means.shape[:2]
        device, dtype = means.device, means.dtype

        if loss_weight == 0:
            zero_ref = 0 * (means.sum() + opacities.sum() + depth_grad.sum())
            if return_per_point:
                zeros_p = torch.zeros(B, P, device=device, dtype=dtype)
                valid_p = torch.zeros(B, P, dtype=torch.bool, device=device)
                return zero_ref, zeros_p, valid_p
            return zero_ref

        if w2c.ndim == 3:
            w2c = w2c.unsqueeze(1)
        if intrinsics.ndim == 3:
            intrinsics = intrinsics.unsqueeze(1)

        N_cams = w2c.shape[1]

        if depth_grad.ndim == 3:
            depth_grad = depth_grad.unsqueeze(1).unsqueeze(2)
        elif depth_grad.ndim == 4:
            if depth_grad.shape[1] == N_cams:
                depth_grad = depth_grad.unsqueeze(2)
            else:
                depth_grad = depth_grad.unsqueeze(1)
        elif depth_grad.ndim != 5:
            raise ValueError(f"Unexpected depth_grad shape: {tuple(depth_grad.shape)}")

        if self.detach_depth_grad:
            depth_grad = depth_grad.detach()

        _, N, _, H, W = depth_grad.shape
        ones = torch.ones(B, P, 1, device=device, dtype=dtype)
        means_h = torch.cat([means, ones], dim=-1).transpose(1, 2)

        per_point_sum = torch.zeros(B, P, device=device, dtype=dtype)
        per_point_count = torch.zeros(B, P, device=device, dtype=dtype)

        for vi in range(N):
            w2c_v = w2c[:, vi].to(dtype)
            K_v = intrinsics[:, vi].to(dtype)

            pts_cam = torch.bmm(w2c_v, means_h)[:, :3]
            z = pts_cam[:, 2:3]
            uv_homo = torch.bmm(K_v, pts_cam)
            uv = uv_homo[:, :2] / uv_homo[:, 2:3].clamp(min=1e-8)
            u, v = uv[:, 0], uv[:, 1]

            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z.squeeze(1) > 0)
            inside_f = inside.to(dtype)

            grid_u = 2.0 * u / W - 1.0
            grid_v = 2.0 * v / H - 1.0
            grid = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(2)

            sampled = F.grid_sample(
                depth_grad[:, vi].to(dtype), grid,
                mode="bilinear", padding_mode="border", align_corners=False,
            ).squeeze(-1)

            if sampled.shape[1] == 1:
                grad_mag = sampled.squeeze(1).abs()
            else:
                grad_mag = torch.sqrt((sampled * sampled).sum(dim=1) + 1e-12)

            penalty = 1.0 - torch.exp(
                -torch.clamp(grad_mag - self.epsilon, min=0.0) / self.sigma
            )

            per_point_sum = per_point_sum + opacities * penalty * inside_f
            per_point_count = per_point_count + inside_f

        valid_any = per_point_count > 0
        per_point = per_point_sum / per_point_count.clamp(min=1e-8)
        per_point = torch.where(valid_any, per_point, torch.zeros_like(per_point))

        n_valid = valid_any.sum().to(dtype)
        loss = per_point.sum() / n_valid.clamp(min=1.0)
        loss = check_and_fix_inf_nan(loss, "GaussianDepthGradLoss")
        loss = loss * loss_weight

        if return_per_point:
            return loss, per_point.detach(), valid_any.detach()
        return loss
