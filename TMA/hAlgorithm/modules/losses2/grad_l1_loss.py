import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Loss


class GradL1Loss(Loss):
    def __init__(self, scale_level, loss_weight=1.0, **kwargs):
        super(GradL1Loss, self).__init__(loss_weight=loss_weight)

        self.scale_level = scale_level
        self.loss_weight = loss_weight

    def forward(self, pred_depth, target_depth, valid_mask, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(pred_depth))

        valid_mask = valid_mask.type_as(pred_depth)
        mask_gt = target_depth * valid_mask
        loss = 0
        for i in range(self.scale_level):
            r = 2**i

            gt_this_res = F.avg_pool2d(mask_gt, r)
            pd_this_res = F.avg_pool2d(pred_depth, r)
            mask_this_res = F.avg_pool2d(valid_mask, r)

            # mask_this_res = (mask_this_res > 0.0).float()
            mask_this_res = (mask_this_res == 1.0).float()

            gt_grad_u = gt_this_res[:, :, 1:] - gt_this_res[:, :, :-1]
            pd_grad_u = pd_this_res[:, :, 1:] - pd_this_res[:, :, :-1]
            mask_u = mask_this_res[:, :, 1:] * mask_this_res[:, :, :-1]

            gt_grad_v = gt_this_res[:, 1:, :] - gt_this_res[:, :-1, :]
            pd_grad_v = pd_this_res[:, 1:, :] - pd_this_res[:, :-1, :]
            mask_v = mask_this_res[:, 1:, :] * mask_this_res[:, :-1, :]

            i_loss_u = ((pd_grad_u - gt_grad_u).abs() * mask_u).sum() / mask_u.sum().clip(1)
            i_loss_v = ((pd_grad_v - gt_grad_v).abs() * mask_v).sum() / mask_v.sum().clip(1)
            i_loss_u = torch.nan_to_num(i_loss_u)
            i_loss_v = torch.nan_to_num(i_loss_v)

            loss += i_loss_u + i_loss_v

        return loss * loss_weight
