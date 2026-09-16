import logging

import torch
import torch.nn as nn


class RGBL1Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
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

    def forward(self, rgbs, render_rgbs, mask=None, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(render_rgbs))

        if mask is not None:
            loss = ((render_rgbs - rgbs) * mask).abs().sum() / (mask.sum() * rgbs.shape[-3])
        else:
            loss = (render_rgbs - rgbs).abs().mean()
        loss = torch.nan_to_num(loss)

        if torch.isinf(loss).item():
            loss = 0 * torch.sum(torch.nan_to_num(render_rgbs))
            logging.warning(f"Data {name}, rgbsL1Loss INF")

        return loss * loss_weight
