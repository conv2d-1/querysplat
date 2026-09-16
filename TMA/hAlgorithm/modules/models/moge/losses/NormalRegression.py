import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .pointmap_to_normal import PointMap2Normal


######################################################
# EdgeguidedNormalRankingLoss
#####################################################
class NormalLoss(nn.Module):
    def __init__(
        self,
        point_pairs=10000,
        cos_theta1=0.25,
        cos_theta2=0.98,
        cos_theta3=0.5,
        cos_theta4=0.86,
        mask_value=1e-8,
        loss_weight=1.0,
        data_type=["lidar", "denselidar", "stereo", "denselidar_syn", "sfm"],
        **kwargs,
    ):
        super(NormalLoss, self).__init__()
        self.point_pairs = point_pairs  # number of point pairs
        self.mask_value = mask_value
        self.cos_theta1 = cos_theta1  # 75 degree
        self.cos_theta2 = cos_theta2  # 10 degree
        self.cos_theta3 = cos_theta3  # 60 degree
        self.cos_theta4 = cos_theta4  # 30 degree
        # self.kernel = torch.tensor(
        #     np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),
        #     requires_grad=False,
        # )[None, None, :, :].cuda()
        self.pointmap2normal = PointMap2Normal()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6

    def forward(self, prediction, target, mask, **kwargs):
        loss = self.get_loss(prediction, target, mask, **kwargs)
        return loss

    def get_loss(self, prediction, target, mask, **kwargs):
        """
        input and target: surface normal input
        input: rgb images
        """

        predictions_normals, _ = self.pointmap2normal(prediction, mask)
        targets_normals, targets_normals_masks = self.pointmap2normal(target, mask)
        masks_normals = mask & targets_normals_masks

        n, c, h, w = targets_normals.size()

        predictions_normals = predictions_normals.contiguous().view(n, c, -1).permute(0, 2, 1)
        targets_normals = targets_normals.contiguous().view(n, c, -1).permute(0, 2, 1)
        masks_normals = masks_normals.contiguous().view(n, -1)

        # angle between target and pred normal
        cos_angle = torch.einsum(
            "nc,nc->n", targets_normals[masks_normals], predictions_normals[masks_normals]
        )
        loss = (1 - cos_angle**2).mean()

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            logging.warning(
                f"Pair-wise Normal Regression Loss NAN error, {loss}"
            )  # , valid pix: {valid_samples}')
        return loss
