import logging

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import check_and_fix_inf_nan


class GradL1Loss(nn.Module):
    """OMNI-DC/src/loss/submodule/seqgradl1loss.py"""

    def __init__(self, scale_level, loss_weight=1.0, debug=False, **kwargs):
        super(GradL1Loss, self).__init__()

        self.scale_level = scale_level
        self.loss_weight = loss_weight
        self.debug = debug

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

    def forward(self, pred, gt, mask, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(pred))

        mask = mask.type_as(pred)
        mask_gt = gt * mask
        loss = 0
        for i in range(self.scale_level):
            r = 2**i

            gt_this_res = F.avg_pool2d(mask_gt, r)
            pd_this_res = F.avg_pool2d(pred, r)
            mask_this_res = F.avg_pool2d(mask, r)

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

            if self.debug:
                self.debug_func(gt_grad_u, gt_grad_v, i, name="gt")
                self.debug_func(pd_grad_u, pd_grad_v, i, name="pd")

        if torch.isinf(loss).item():
            loss = 0 * torch.sum(torch.nan_to_num(pred))
            logging.warning(f"Data {name}, GradL1Loss NAN error, {loss}")

        return loss * loss_weight

    def debug_func(self, gradu, gradv, scale, name):
        import os

        import matplotlib.pyplot as plt

        save_dir = "debug/GradL1Loss/"
        os.makedirs(save_dir, exist_ok=True)

        for batch_idx in range(gradu.shape[0]):
            # Select the first batch item for plotting
            gradu_i = gradu[batch_idx].squeeze().detach().cpu().numpy()
            gradv_i = gradv[batch_idx].squeeze().detach().cpu().numpy()

            # Plot ground truth gradient
            fig, ax = plt.subplots(figsize=(5, 5))

            im = ax.imshow(gradu_i, cmap="turbo")
            ax.set_title(f"scale{scale}-gradu-{name}")
            ax.axis("off")
            fig.colorbar(im, ax=ax)

            filename = os.path.join(save_dir, f"gradu_{name}_scale{scale}_b{batch_idx}.png")
            plt.savefig(filename)
            plt.close(fig)

            # Plot predicted gradient
            fig, ax = plt.subplots(figsize=(5, 5))

            im = ax.imshow(gradv_i, cmap="turbo")
            ax.set_title(f"Scale{scale}-gradv-{name}")
            ax.axis("off")
            fig.colorbar(im, ax=ax)

            filename = os.path.join(save_dir, f"gradv_{name}_scale{scale}_b{batch_idx}.png")
            plt.savefig(filename)
            plt.close(fig)


class GradL1LossV2(GradL1Loss):
    """OMNI-DC/src/loss/submodule/seqgradl1loss.py"""
    def __init__(self, **kwargs):
        super(GradL1LossV2, self).__init__(**kwargs)

    def forward(self, pred, gt, mask, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(pred))

        mask = mask.type_as(pred)
        mask_gt = gt * mask
        loss = 0
        for i in range(self.scale_level):
            r = 2**i

            gt_this_res = F.avg_pool2d(mask_gt, r)
            pd_this_res = F.avg_pool2d(pred, r)
            mask_this_res = F.avg_pool2d(mask, r)

            # mask_this_res = (mask_this_res > 0.0).float()
            mask_this_res = (mask_this_res == 1.0).float()

            gt_grad_u = gt_this_res[:, :, 1:] - gt_this_res[:, :, :-1]
            pd_grad_u = pd_this_res[:, :, 1:] - pd_this_res[:, :, :-1]
            mask_u = mask_this_res[:, :, 1:] * mask_this_res[:, :, :-1]

            gt_grad_v = gt_this_res[:, 1:, :] - gt_this_res[:, :-1, :]
            pd_grad_v = pd_this_res[:, 1:, :] - pd_this_res[:, :-1, :]
            mask_v = mask_this_res[:, 1:, :] * mask_this_res[:, :-1, :]

            i_loss_u = ((pd_grad_u - gt_grad_u).abs() * mask_u)
            i_loss_u = check_and_fix_inf_nan(i_loss_u, "i_loss_u")
            i_loss_u = i_loss_u.sum() / mask_u.sum().clip(1)

            i_loss_v = ((pd_grad_v - gt_grad_v).abs() * mask_v)
            i_loss_v = check_and_fix_inf_nan(i_loss_v, "i_loss_v")
            i_loss_v = i_loss_v.sum() / mask_v.sum().clip(1)

            loss += i_loss_u + i_loss_v

            if self.debug:
                self.debug_func(gt_grad_u, gt_grad_v, i, name="gt")
                self.debug_func(pd_grad_u, pd_grad_v, i, name="pd")

        return loss * loss_weight
