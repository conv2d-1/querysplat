import logging

import torch
import torch.nn as nn
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
        render_rgbs_reshaped = render_rgbs.reshape(
            -1, 3, render_rgbs.shape[-2], render_rgbs.shape[-1]
        )

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
    _loss = torch.tensor(0.0,device='cuda')
    n_corr = int(p_corr * num_box_h * num_box_w)
    x_0 = torch.randint(0, max_h, size=(n_corr,), device = 'cuda')
    y_0 = torch.randint(0, max_w, size=(n_corr,), device = 'cuda')
    x_1 = x_0 + box_p
    y_1 = y_0 + box_p
    _loss = torch.tensor(0.0,device='cuda')
    for i in range(len(x_0)):
        _loss += pearson_depth_loss(depth_src[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1), depth_target[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1))
    return (_loss/n_corr) * weight


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
