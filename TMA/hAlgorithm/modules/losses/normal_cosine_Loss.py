import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.utils.normal import PointMap2Normal
from .base import check_and_fix_inf_nan


class NormalCosineLoss(nn.Module):
    def __init__(
        self,
        cos_theta1=0.25,
        cos_theta2=0.98,
        cos_theta3=0.5,
        cos_theta4=0.86,
        loss_weight=1.0,
        start_iter=None,
        debug=False,
        **kwargs,
    ):
        super().__init__()

        self.cos_theta1 = cos_theta1  # 75 degree
        self.cos_theta2 = cos_theta2  # 10 degree
        self.cos_theta3 = cos_theta3  # 60 degree
        self.cos_theta4 = cos_theta4  # 30 degree

        self.pointmap2normal = PointMap2Normal()
        self.loss_weight = loss_weight

        self.start_iter = start_iter
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

    def forward(self, prediction, target, mask, name=None, **kwargs):
        """
        input and target: surface normal input
        input: rgb images
        """

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(prediction))

        predictions_normals, _ = self.pointmap2normal(prediction, mask)
        targets_normals, targets_normals_masks = self.pointmap2normal(target, mask)
        masks_normals = mask & targets_normals_masks

        if self.debug:
            self.debug_func(predictions_normals, targets_normals, masks_normals)

        n, c, h, w = targets_normals.size()

        predictions_normals = predictions_normals.contiguous().view(n, c, -1).permute(0, 2, 1)
        targets_normals = targets_normals.contiguous().view(n, c, -1).permute(0, 2, 1)
        masks_normals = masks_normals.contiguous().view(n, -1)

        # angle between target and pred normal
        cos_angle = torch.einsum(
            "nc,nc->n",
            targets_normals[masks_normals],
            predictions_normals[masks_normals],
        )
        loss = (1 - cos_angle**2).mean()

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            logging.warning(
                f"Data {name}, Pair-wise Normal Regression Loss NAN error, {loss}"
            )  # , valid pix: {valid_samples}')

        return loss * loss_weight

    def debug_func(self, predictions_normals, targets_normals, masks_normals):
        import os

        import cv2
        import numpy as np

        outdir = "debug/NormalCosineLoss/"
        os.makedirs(outdir, exist_ok=True)

        for bi in range(predictions_normals.shape[0]):
            pnormal = predictions_normals[bi].detach().cpu().numpy().transpose(1, 2, 0)
            tnormal = targets_normals[bi].detach().cpu().numpy().transpose(1, 2, 0)
            masks_normal = masks_normals[bi].detach().cpu().numpy()

            pnormal = ((pnormal + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
            save_path = os.path.join(outdir, f"predict_normal_{bi}.jpg")
            cv2.imwrite(save_path, pnormal)

            tnormal = ((tnormal + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
            save_path = os.path.join(outdir, f"target_normal_{bi}.jpg")
            cv2.imwrite(save_path, tnormal)

            tnormal[~masks_normal] = 0
            save_path = os.path.join(outdir, f"target_mask_normal_{bi}.jpg")
            cv2.imwrite(save_path, tnormal)

        breakpoint()


class MultiScaleNormalCosineLoss(NormalCosineLoss):
    def __init__(
        self,
        scale_level=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_level = scale_level

    def forward(self, prediction, target, mask, name=None, **kwargs):
        """
        input and target: surface normal input
        input: rgb images
        """

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(prediction))

        mask = mask.type_as(prediction)
        loss = 0
        for i in range(self.scale_level):
            r = 2**i

            pred_this_res = F.avg_pool2d(prediction, r)
            gt_this_res = F.avg_pool2d(target, r)
            mask_this_res = F.avg_pool2d(mask, r)
            mask_this_res = (mask_this_res == 1.0).float()

            predictions_normals_this_res, _ = self.pointmap2normal(pred_this_res, mask_this_res)
            targets_normals_this_res, targets_normals_masks_this_res = self.pointmap2normal(
                gt_this_res, mask_this_res
            )
            masks_normals_this_res = mask_this_res & targets_normals_masks_this_res

            n, c, h, w = targets_normals_this_res.size()
            predictions_normals_this_res = (
                predictions_normals_this_res.contiguous().view(n, c, -1).permute(0, 2, 1)
            )
            targets_normals_this_res = (
                targets_normals_this_res.contiguous().view(n, c, -1).permute(0, 2, 1)
            )
            masks_normals_this_res = masks_normals_this_res.contiguous().view(n, -1)

            cos_angle_this_res = torch.einsum(
                "nc,nc->n",
                targets_normals_this_res[masks_normals_this_res],
                predictions_normals_this_res[masks_normals_this_res],
            )
            loss_this_res = (1 - cos_angle_this_res**2).mean()

            loss += loss_this_res

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            logging.warning(
                f"Data {name}, Pair-wise Normal Regression Loss NAN error, {loss}"
            )  # , valid pix: {valid_samples}')

        return loss * loss_weight

    def debug_func(self, predictions_normals, targets_normals, masks_normals):
        import os

        import cv2
        import numpy as np

        outdir = "debug/NormalCosineLoss/"
        os.makedirs(outdir, exist_ok=True)

        for bi in range(predictions_normals.shape[0]):
            pnormal = predictions_normals[bi].detach().cpu().numpy().transpose(1, 2, 0)
            tnormal = targets_normals[bi].detach().cpu().numpy().transpose(1, 2, 0)
            masks_normal = masks_normals[bi].detach().cpu().numpy()

            pnormal = ((pnormal + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
            save_path = os.path.join(outdir, f"predict_normal_{bi}.jpg")
            cv2.imwrite(save_path, pnormal)

            tnormal = ((tnormal + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
            save_path = os.path.join(outdir, f"target_normal_{bi}.jpg")
            cv2.imwrite(save_path, tnormal)

            tnormal[~masks_normal] = 0
            save_path = os.path.join(outdir, f"target_mask_normal_{bi}.jpg")
            cv2.imwrite(save_path, tnormal)

        breakpoint()


class NormalCosineLossV2(nn.Module):
    """NormalCosineLossV2 support sample_ratio and select_ratio"""

    def __init__(
        self,
        loss_weight=1.0,
        start_iter=None,
        sample_ratio=None,
        select_ratio=None,
        descending=False,
        predict_is_normals=False,
        target_is_normals=False,
        debug=False,
        **kwargs,
    ):
        super().__init__()
        self.pointmap2normal = PointMap2Normal()
        self.loss_weight = loss_weight

        self.start_iter = start_iter
        self.debug = debug

        self.sample_ratio = sample_ratio
        self.select_ratio = select_ratio
        self.descending = descending

        self.predict_is_normals = predict_is_normals
        self.target_is_normals = target_is_normals

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

    def select_index(self, mask):
        B, N = mask.shape
        pix_idx_mat = torch.arange(N, device=mask.device)
        new_masks = []
        for i in range(B):
            inputs_index = torch.masked_select(pix_idx_mat, mask[i])
            num_effect_pixels = len(inputs_index)

            intend_sample_num = int(N * self.sample_ratio)
            sample_num = (
                intend_sample_num if num_effect_pixels >= intend_sample_num else num_effect_pixels
            )

            shuffle_effect_pixels = torch.randperm(num_effect_pixels, device=mask.device)
            p1i = inputs_index[shuffle_effect_pixels[:sample_num]]

            new_mask = torch.zeros(N, dtype=torch.bool, device=mask.device)
            new_mask[p1i] = True

            new_masks.append(new_mask)

        new_masks = torch.stack(new_masks, dim=0)

        return new_masks

    def forward(self, prediction, target, mask, name=None, **kwargs):
        """
        input and target: surface normal input
        input: rgb images
        """

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(prediction))

        if self.predict_is_normals:
            predictions_normals = prediction
        else:
            predictions_normals, _ = self.pointmap2normal(prediction, mask)

        if self.target_is_normals:
            targets_normals = target
            masks_normals = mask
        else:
            targets_normals, targets_normals_masks = self.pointmap2normal(target, mask)
            masks_normals = mask & targets_normals_masks

        if self.debug:
            self.debug_func(predictions_normals, targets_normals, masks_normals)

        n, c, h, w = targets_normals.size()

        predictions_normals = predictions_normals.contiguous().view(n, c, -1).permute(0, 2, 1)
        targets_normals = targets_normals.contiguous().view(n, c, -1).permute(0, 2, 1)
        masks_normals = masks_normals.contiguous().view(n, -1)

        if self.sample_ratio is not None:
            masks_normals = self.select_index(masks_normals)

        # angle between target and pred normal
        cos_angle = torch.einsum(
            "nc,nc->n",
            targets_normals[masks_normals],
            predictions_normals[masks_normals],
        )
        loss = 1 - cos_angle**2
        loss = torch.nan_to_num(loss)

        if self.select_ratio is not None:
            loss, indices = torch.sort(loss, dim=0, descending=self.descending)
            loss = loss[int(loss.size(0) * self.select_ratio) :].mean()
        else:
            loss = loss.mean()

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(torch.nan_to_num(prediction))
            logging.warning(
                f"Data {name}, Pair-wise Normal Regression Loss NAN error, {loss}"
            )  # , valid pix: {valid_samples}')

        return loss * loss_weight

    def debug_func(self, predictions_normals, targets_normals, masks_normals):
        import os

        import cv2
        import numpy as np

        outdir = "debug/NormalCosineLoss/"
        os.makedirs(outdir, exist_ok=True)

        for bi in range(predictions_normals.shape[0]):
            pnormal = predictions_normals[bi].detach().cpu().numpy().transpose(1, 2, 0)
            tnormal = targets_normals[bi].detach().cpu().numpy().transpose(1, 2, 0)
            masks_normal = masks_normals[bi].detach().cpu().numpy()

            pnormal = ((pnormal + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
            save_path = os.path.join(outdir, f"predict_normal_{bi}.jpg")
            cv2.imwrite(save_path, pnormal)

            tnormal = ((tnormal + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
            save_path = os.path.join(outdir, f"target_normal_{bi}.jpg")
            cv2.imwrite(save_path, tnormal)

            tnormal[~masks_normal] = 0
            save_path = os.path.join(outdir, f"target_mask_normal_{bi}.jpg")
            cv2.imwrite(save_path, tnormal)

        breakpoint()


class NormalCosineLossV3(nn.Module):
    """NormalCosineLossV3 support sample_ratio and select_ratio"""

    def __init__(
        self,
        loss_weight=1.0,
        start_iter=None,
        sample_ratio=None,
        select_ratio=None,
        descending=False,
        predict_is_normals=False,
        target_is_normals=False,
        debug=False,
        **kwargs,
    ):
        super().__init__()
        self.pointmap2normal = PointMap2Normal()
        self.loss_weight = loss_weight

        self.start_iter = start_iter
        self.debug = debug

        self.sample_ratio = sample_ratio
        self.select_ratio = select_ratio
        self.descending = descending

        self.predict_is_normals = predict_is_normals
        self.target_is_normals = target_is_normals

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

    def select_index(self, mask):
        B, N = mask.shape
        pix_idx_mat = torch.arange(N, device=mask.device)
        new_masks = []
        for i in range(B):
            inputs_index = torch.masked_select(pix_idx_mat, mask[i])
            num_effect_pixels = len(inputs_index)

            intend_sample_num = int(N * self.sample_ratio)
            sample_num = (
                intend_sample_num if num_effect_pixels >= intend_sample_num else num_effect_pixels
            )

            shuffle_effect_pixels = torch.randperm(num_effect_pixels, device=mask.device)
            p1i = inputs_index[shuffle_effect_pixels[:sample_num]]

            new_mask = torch.zeros(N, dtype=torch.bool, device=mask.device)
            new_mask[p1i] = True

            new_masks.append(new_mask)

        new_masks = torch.stack(new_masks, dim=0)

        return new_masks

    def forward(self, prediction, target, mask, name=None, **kwargs):
        """
        input and target: surface normal input
        input: rgb images
        """

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(prediction))

        if self.predict_is_normals:
            predictions_normals = prediction
        else:
            predictions_normals, _ = self.pointmap2normal(prediction, mask)

        if self.target_is_normals:
            targets_normals = target
            masks_normals = mask
        else:
            targets_normals, targets_normals_masks = self.pointmap2normal(target, mask)
            masks_normals = mask & targets_normals_masks

        if self.debug:
            self.debug_func(predictions_normals, targets_normals, masks_normals)

        n, c, h, w = targets_normals.size()

        predictions_normals = predictions_normals.contiguous().view(n, c, -1).permute(0, 2, 1)
        targets_normals = targets_normals.contiguous().view(n, c, -1).permute(0, 2, 1)
        masks_normals = masks_normals.contiguous().view(n, -1)

        if self.sample_ratio is not None:
            masks_normals = self.select_index(masks_normals)

        # angle between target and pred normal
        cos_angle = torch.einsum(
            "nc,nc->n",
            targets_normals[masks_normals],
            predictions_normals[masks_normals],
        )
        loss = 1 - cos_angle**2
        # loss = torch.nan_to_num(loss)
        loss = check_and_fix_inf_nan(loss, "NormalCosineLossV3_loss")

        if self.select_ratio is not None:
            loss, indices = torch.sort(loss, dim=0, descending=self.descending)
            loss = loss[int(loss.size(0) * self.select_ratio) :].mean()
        else:
            loss = loss.mean()

        return loss * loss_weight