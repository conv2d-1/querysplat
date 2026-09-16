import logging

import torch

from .base import Loss, filter_by_quantile
import torch.nn.functional as F
import math

class DenseMatchLoss(Loss):
    def __init__(
        self,
        loss_weight=1,
        scale_weight=1,
        alpha=0.5,
        scale_c=1e-4,
        conf_weight=0.01,
        coarse_scale=4,
        precision_weight=None,
    ):
        super().__init__(loss_weight=loss_weight)

        self.eps = 1e-6
        self.scale_weight = scale_weight
        self.conf_weight = conf_weight
        self.alpha = alpha
        self.scale_c = scale_c
        self.coarse_scale = coarse_scale
        self.precision_weight = precision_weight
        self.eps = 1e-6

    def get_scale_weight(self, scale):
        if isinstance(self.scale_weight, dict):
            return self.scale_weight.get(scale, 0)
        else:
            return self.scale_weight

    def forward(self, preds: dict, warp_gt_scales: dict, warp_mask_gt_scales: dict, pair_idx, name=None, **kwargs):
        B = next(iter(warp_gt_scales.values())).shape[0]
        num_pair = len(pair_idx)
        def reshape_pair_to_batch(value):
            return value.reshape(B*num_pair, *value.shape[2:]) if value is not None else None
        
        loss_dict = dict(ce_loss={}, reg_loss={}, prec_loss={})
        loss_weight = self.get_loss_weight(name)
        for key, val in preds.items():
            if key in ["final", "pair_idx"]:
                # final is the same as refiner_1
                continue
            if key in ["coarse"]:
                scale = self.coarse_scale
            elif key.startswith("refiner"):
                scale = int(key[-1])
            else:
                continue
            
            if scale not in warp_gt_scales:
                continue
            
            loss_weight_scale = self.get_scale_weight(scale)
            loss_scale = loss_weight * loss_weight_scale
            
            pred_warp_AB = reshape_pair_to_batch(val["warp_AB"]).float()
            pred_conf_AB = reshape_pair_to_batch(val["confidence_AB"]).float()
            
            warp_gt = reshape_pair_to_batch(warp_gt_scales[scale])
            warp_mask_gt = reshape_pair_to_batch(warp_mask_gt_scales[scale])
            epe = (pred_warp_AB - warp_gt).norm(dim=-1)
            epe = torch.clamp(epe, min=1e-6)
            ce_loss = F.binary_cross_entropy_with_logits(pred_conf_AB[..., 0], warp_mask_gt)

            if self.precision_weight is not None and key not in ["coarse"]:
                eps = self.eps
                residual = (pred_warp_AB - warp_gt).detach()
                
                pred_precision = pred_conf_AB[..., 1:4]
                z11, z21, z22 = pred_precision[..., 0], pred_precision[..., 1], pred_precision[..., 2]
                l11 = torch.nn.functional.softplus(z11) + eps
                l21 = z21
                l22 = torch.nn.functional.softplus(z22) + eps
                
                # 残差分量
                rx = residual[..., 0]  # (B, H, W)
                ry = residual[..., 1]  # (B, H, W)

                # 计算 ||L^T r||^2 = (l11 * rx)^2 + (l21 * rx + l22 * ry)^2
                term1 = l11 * rx
                term2 = l21 * rx + l22 * ry
                quadratic_term = 0.5 * (term1 ** 2 + term2 ** 2)  # (B, H, W)

                # 计算 -0.5 * log det(P) = -(log l11 + log l22)
                # 为数值稳定，加 eps 再 log
                log_det_term = -(torch.log(l11 + eps) + torch.log(l22 + eps))  # (B, H, W)

                # 常数项：log(2π)
                const_term = math.log(2 * math.pi)  # scalar ≈ 1.837877

                # 总 loss（逐像素）
                pixel_loss = quadratic_term + log_det_term + const_term  # (B, H, W)
                # 
                residual_mask = residual.norm(dim=-1) < 8

                # 可选：求平均或求和
                loss_conf = pixel_loss[residual_mask].mean()
                loss_dict["prec_loss"][key] = loss_conf * loss_scale * self.precision_weight

            a = self.alpha
            cs = self.scale_c * scale
            x = epe[warp_mask_gt > 0.99]
            reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
            
            if loss_scale == 0 or (not torch.any(reg_loss)):
                reg_loss = 0 * torch.sum(pred_warp_AB)
            
            loss_dict["ce_loss"][key] = ce_loss.mean() * self.conf_weight * loss_scale
            loss_dict["reg_loss"][key] = reg_loss.mean() * loss_scale
        
        loss_dict = {key: sum(val.values()) / len(val) for key, val in loss_dict.items() if len(val) > 0}
        loss = sum(loss_dict.values())
        return loss, loss_dict


class DenseMatchLossV2(Loss):
    '''
    Confidence Predict Pixel Confidence
    '''
    def __init__(
        self,
        loss_weight=1,
        scale_weight=1,
        alpha=0.5,
        scale_c=1e-4,
        conf_weight=0.01,
        coarse_scale=4,
        precision_weight=None,
        precision_thresh=1.0,
    ):
        super().__init__(loss_weight=loss_weight)

        self.eps = 1e-6
        self.scale_weight = scale_weight
        self.conf_weight = conf_weight
        self.alpha = alpha
        self.scale_c = scale_c
        self.coarse_scale = coarse_scale
        self.precision_weight = precision_weight
        self.precision_thresh = precision_thresh
        self.eps = 1e-6

    def get_scale_weight(self, scale):
        if isinstance(self.scale_weight, dict):
            return self.scale_weight.get(scale, 0)
        else:
            return self.scale_weight

    def forward(self, preds: dict, warp_gt_scales: dict, warp_mask_gt_scales: dict, pair_idx, name=None, **kwargs):
        B = next(iter(warp_gt_scales.values())).shape[0]
        num_pair = len(pair_idx)
        def reshape_pair_to_batch(value):
            return value.reshape(B*num_pair, *value.shape[2:]) if value is not None else None
        
        loss_dict = dict(ce_loss={}, reg_loss={}, prec_loss={})
        loss_weight = self.get_loss_weight(name)
        for key, val in preds.items():
            if key in ["final", "pair_idx"]:
                # final is the same as refiner_1
                continue
            if key in ["coarse"]:
                scale = self.coarse_scale
            elif key.startswith("refiner"):
                scale = int(key[-1])
            else:
                continue
            
            if scale not in warp_gt_scales:
                continue
            
            loss_weight_scale = self.get_scale_weight(scale)
            loss_scale = loss_weight * loss_weight_scale
            
            pred_warp_AB = reshape_pair_to_batch(val["warp_AB"]).float()
            pred_conf_AB = reshape_pair_to_batch(val["confidence_AB"]).float()
            
            warp_gt = reshape_pair_to_batch(warp_gt_scales[scale])
            warp_mask_gt = reshape_pair_to_batch(warp_mask_gt_scales[scale])
            epe = (pred_warp_AB - warp_gt).norm(dim=-1)
            epe = torch.clamp(epe, min=1e-6)
            ce_loss = F.binary_cross_entropy_with_logits(pred_conf_AB[..., 0], warp_mask_gt)

            if self.precision_weight is not None and pred_conf_AB.shape[-1] >= 4:
                if (warp_mask_gt > 0.99).sum() == 0:
                    loss_dict["prec_loss"][key] = 0 * torch.sum(pred_conf_AB)
                else:
                    eps = self.eps
                    H, W = warp_gt.shape[-3:-1]
                    residual = (pred_warp_AB - warp_gt).detach().abs()
                    # 残差分量
                    rx = residual[..., 0] * W / 2  # (B, H, W)
                    ry = residual[..., 1] * H / 2  # (B, H, W)
                    rnorm = torch.sqrt(rx**2 + ry**2).detach()
                    
                    confidence_gt_rx = (rx < self.precision_thresh).float()
                    pred_conf_rx = F.sigmoid(pred_conf_AB[..., 1])
                    loss_conf_rx = F.binary_cross_entropy(pred_conf_rx, confidence_gt_rx, reduction="none")[warp_mask_gt > 0.99]
                    loss_conf_rx = torch.nan_to_num(loss_conf_rx)
                    loss_conf_rx = loss_conf_rx.mean()
                    
                    
                    confidence_gt_ry = (ry < self.precision_thresh).float()
                    pred_conf_ry = F.sigmoid(pred_conf_AB[..., 2])
                    loss_conf_ry = F.binary_cross_entropy(pred_conf_ry, confidence_gt_ry, reduction="none")[warp_mask_gt > 0.99]
                    loss_conf_ry = torch.nan_to_num(loss_conf_ry)
                    loss_conf_ry = loss_conf_ry.mean()
                    
                    confidence_gt_rnorm = (rnorm < self.precision_thresh).float()
                    pred_conf_rnorm = F.sigmoid(pred_conf_AB[..., 3])
                    loss_conf_rnorm = F.binary_cross_entropy(pred_conf_rnorm, confidence_gt_rnorm, reduction="none")[warp_mask_gt > 0.99]
                    loss_conf_rnorm = torch.nan_to_num(loss_conf_rnorm)
                    loss_conf_rnorm = loss_conf_rnorm.mean()
                    
                    loss_conf = (loss_conf_rx + loss_conf_ry + loss_conf_rnorm)
                    
                    loss_dict["prec_loss"][key] = loss_conf * loss_scale * self.precision_weight

            a = self.alpha
            cs = self.scale_c * scale
            x = epe[warp_mask_gt > 0.99]
            reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
            
            if loss_scale == 0 or (not torch.any(reg_loss)):
                reg_loss = 0 * torch.sum(pred_warp_AB)
            
            loss_dict["ce_loss"][key] = ce_loss.mean() * self.conf_weight * loss_scale
            loss_dict["reg_loss"][key] = reg_loss.mean() * loss_scale
        
        loss_dict = {key: sum(val.values()) / len(val) for key, val in loss_dict.items() if len(val) > 0}
        loss = sum(loss_dict.values())
        return loss, loss_dict
