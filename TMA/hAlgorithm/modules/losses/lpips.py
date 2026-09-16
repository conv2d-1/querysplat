import logging

import torch
from einops import rearrange
from lpips import LPIPS
from torch import nn


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


class LpipsLoss(nn.Module):
    def __init__(self, loss_weight=0.05) -> None:
        super().__init__()

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

        self.loss_weight = loss_weight
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

        if torch.isinf(loss).item():
            loss = 0 * torch.sum(torch.nan_to_num(render_rgbs))
            logging.warning(f"Data {name}, LpipsLoss INF")

        return loss * loss_weight
