import torch
import torch.nn as nn

from hAlgorithm.modules.utils.normal import PointMap2Normal, PointMap2NormalSVD
import logging
from .base import Loss


class NormalAngleLoss(Loss):
    def __init__(
        self,
        loss_weight=1.0,
        predict_is_normals=False,
        target_is_normals=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(loss_weight=loss_weight)

        self.pointmap2normal = PointMap2Normal()
        self.loss_weight = loss_weight

        self.debug = debug

        self.predict_is_normals = predict_is_normals
        self.target_is_normals = target_is_normals

    def forward(self, pred_depth=None, target_depth=None, valid_mask=None, pred_normal=None, target_normal=None, name=None, **kwargs):
        """
        input and target_depth: surface normal input
        input: rgb images
        """

        loss_weight = self.get_loss_weight(name)
        
        valid_mask = valid_mask.squeeze(-1)

        if (loss_weight == 0) or (target_depth is None and target_normal is None):
            return 0 * torch.sum(torch.nan_to_num(pred_depth))

        if pred_normal is None:
            if pred_depth.shape[-1] != 3 :
                pred_depth = pred_depth.permute(0, 2, 3, 1)
            pred_normal, _ = self.pointmap2normal(pred_depth, valid_mask)
            n, c, h, w = pred_normal.size()
            pred_normal = pred_normal.view(n, c, -1).permute(0, 2, 1).contiguous()
        else:
            if pred_normal.shape[-1] == 3:
                n, h, w, c = pred_normal.size()
                pred_normal = pred_normal.view(n, -1, c)
            else:
                n, c, h, w = pred_normal.size()
                pred_normal = pred_normal.view(n, c, -1).permute(0, 2, 1)                

        masks_normals = valid_mask
        if target_normal is None:
            if target_depth.shape[-1] != 3 :
                target_depth = target_depth.permute(0, 2, 3, 1)
            target_normal, target_normal_masks = self.pointmap2normal(target_depth, valid_mask)
            
            masks_normals = masks_normals & target_normal_masks
            n, c, h, w = target_normal.size()
            target_normal = target_normal.view(n, c, -1).permute(0, 2, 1).contiguous()
        else:
            n, h, w, c = target_normal.size()
            target_normal = target_normal.view(n, -1, c)

        masks_normals = masks_normals.view(n, -1)

        # 点积，得到 cos(theta)
        cos_theta = (pred_normal[masks_normals] * target_normal[masks_normals]).sum(dim=-1)  # [N]
        
        # 夹紧防止数值溢出
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # 计算角度（弧度）
        angle = torch.acos(cos_theta)  # [N]
    
        loss = angle ** 2
        loss = torch.nan_to_num(loss).mean()

        return loss * loss_weight

class NormalAngleLossV2(nn.Module):

    def __init__(
        self,
        loss_weight=1.0,
        start_iter=None,
        sample_ratio=None,
        select_ratio=None,
        descending=False,
        predict_is_normals=False,
        target_is_normals=False,
        inverse_predict=False,
        point_to_normal="PointMap2Normal",
        debug=False,
        **kwargs,
    ):
        super().__init__()
        self.pointmap2normal = eval(point_to_normal)()
        self.loss_weight = loss_weight

        self.start_iter = start_iter
        self.debug = debug

        self.sample_ratio = sample_ratio
        self.select_ratio = select_ratio
        self.descending = descending

        self.predict_is_normals = predict_is_normals
        self.target_is_normals = target_is_normals
        
        self.inverse_predict = inverse_predict

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
            if self.inverse_predict:
                predictions_normals = -1.0 * predictions_normals
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

        cos_theta = (
            predictions_normals[masks_normals] * targets_normals[masks_normals]
        ).sum(dim=-1)  # [N]
        
        # 夹紧防止数值溢出
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # 计算角度（弧度）
        angle = torch.acos(cos_theta)  # [N]
    
        loss = angle ** 2
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

class NormalAngleLossV3(nn.Module):

    def __init__(
        self,
        loss_weight=1.0,
        start_iter=None,
        sample_ratio=None,
        select_ratio=None,
        descending=False,
        predict_is_normals=False,
        inverse_predict=False,
        point_to_normal="PointMap2Normal",
        debug=False,
        **kwargs,
    ):
        super().__init__()
        self.pointmap2normal = eval(point_to_normal)()
        self.loss_weight = loss_weight

        self.start_iter = start_iter
        self.debug = debug

        self.sample_ratio = sample_ratio
        self.select_ratio = select_ratio
        self.descending = descending

        self.predict_is_normals = predict_is_normals
        
        self.inverse_predict = inverse_predict

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

    def forward(self, prediction, target, mask, target_is_normals=False, name=None, **kwargs):
        """
        input and target: surface normal input
        input: rgb images
        """

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(prediction))

        if self.predict_is_normals:
            predictions_normals = prediction
            if self.inverse_predict:
                predictions_normals = -1.0 * predictions_normals
        else:
            predictions_normals, _ = self.pointmap2normal(prediction, mask)

        if target_is_normals:
            assert target.shape[1] == 3, f"Expect traget normals to have shape B,3,H,W, but got {target.shape}"
            targets_normals = torch.nn.functional.normalize(target, dim=1)
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

        cos_theta = (
            predictions_normals[masks_normals] * targets_normals[masks_normals]
        ).sum(dim=-1)  # [N]
        
        # 夹紧防止数值溢出
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # 计算角度（弧度）
        angle = torch.acos(cos_theta)  # [N]
    
        loss = angle ** 2
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

class NormalAngleL1LossV3(NormalAngleLossV3):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, prediction, target, mask, target_is_normals=False, name=None, **kwargs):
        """
        input and target: surface normal input
        input: rgb images
        """

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(prediction))

        if self.predict_is_normals:
            predictions_normals = prediction
            if self.inverse_predict:
                predictions_normals = -1.0 * predictions_normals
        else:
            predictions_normals, _ = self.pointmap2normal(prediction, mask)

        if target_is_normals:
            assert target.shape[1] == 3, f"Expect traget normals to have shape B,3,H,W, but got {target.shape}"
            targets_normals = torch.nn.functional.normalize(target, dim=1)
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

        cos_theta = (
            predictions_normals[masks_normals] * targets_normals[masks_normals]
        ).sum(dim=-1)  # [N]
        
        # 夹紧防止数值溢出
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # 计算角度（弧度）
        angle = torch.acos(cos_theta)  # [N]
    
        loss = angle.abs()
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
    
