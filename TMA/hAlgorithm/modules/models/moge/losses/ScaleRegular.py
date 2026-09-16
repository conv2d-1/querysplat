import logging

import torch
import torch.nn as nn


class ScaleRegularLoss(nn.Module):
    """
    Compute Global point map supervision, eq.(3) off MoGe
    """

    def __init__(
        self,
        loss_weight=1,
        data_type=["lidar", "denselidar", "stereo", "denselidar_syn", "sfm"],
        **kwargs,
    ):
        super(ScaleRegularLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
        self.printed = False

    def forward(self, prediction_raw, mask, **kwargs):
        # diff = torch.abs(prediction - target).sum(dim=-1, keepdim=True)*mask / (target[..., 2, None]+self.eps)
        # loss = torch.sum(diff) / (torch.sum(mask) + self.eps)
        prediction_raw = prediction_raw[mask]
        prediction_raw = prediction_raw - prediction_raw.mean(dim=0, keepdim=True)
        pred_norm = torch.norm(prediction_raw, dim=-1, keepdim=True)
        loss = torch.exp(-pred_norm.mean())
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction_raw)
            logging.warning(f"ScaleRegularLoss NAN error, {loss}")
        return loss * self.loss_weight
