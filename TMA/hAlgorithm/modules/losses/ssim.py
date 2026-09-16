import logging

import torch
import torch.nn as nn
from fused_ssim import FusedSSIMMap, fused_ssim


class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=0.2):
        super().__init__()
        self.loss_weight = loss_weight
        if isinstance(self.loss_weight, dict):
            assert (
                "default" in self.loss_weight
            ), "If loss_weight is a dictionary, it must contain 'default' key."

    def get_loss_weight(self, name=None):
        """Retrieve the appropriate loss weight based on the provided name."""
        if isinstance(self.loss_weight, dict):
            return self.loss_weight.get(name, self.loss_weight["default"])
        return self.loss_weight

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

        # Handle infinite values
        if torch.isinf(loss).item():
            logging.warning(f"Data {name}, SSIMLoss INF")
            return 0 * torch.sum(torch.nan_to_num(render_rgbs))

        return loss * loss_weight
