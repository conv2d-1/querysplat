import logging

import torch
import torch.nn as nn


class GlobalPointZLoss(nn.Module):
    """
    Compute Global point map supervision, eq.(3) off MoGe
    """

    def __init__(
        self,
        loss_weight=1,
        data_type=["lidar", "denselidar", "stereo", "denselidar_syn", "sfm"],
        **kwargs,
    ):
        super(GlobalPointZLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
        self.printed = False

    def forward(self, prediction, target, mask=None, **kwargs):

        diff = prediction[mask] - target[mask]
        diff = torch.abs(diff).sum(dim=-1, keepdim=True)
        diff = diff / (target[mask][:, 2:3] + self.eps)
        loss = diff[diff < 3.0].mean()
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            logging.warning(f"GlobalPointZLoss NAN error, {loss}")
        return loss * self.loss_weight
