import torch
import torch.nn.functional as F

from .base import Loss



class QueryMatchLoss(Loss):
    """Match loss computed at sparse query locations.

    Same loss formulation as DenseMatchLossV2 (robust regression + BCE confidence)
    but operates on query-sampled warp predictions rather than dense maps.

    Args:
        pred_warp: (B_pair, Q, 2) predicted warp coordinates in [-1, 1]
        pred_conf: (B_pair, Q, C) predicted confidence logits (C=1 basic, C=4 with precision)
        gt_warp: (B_pair, Q, 2) ground truth warp coordinates in [-1, 1]
        gt_mask: (B_pair, Q) ground truth validity mask (float, 0 or 1)
    """

    def __init__(
        self,
        loss_weight=1,
        alpha=0.5,
        scale_c=1e-4,
        conf_weight=0.01,
        precision_weight=None,
        precision_thresh=1.0,
    ):
        super().__init__(loss_weight=loss_weight)

        self.alpha = alpha
        self.scale_c = scale_c
        self.conf_weight = conf_weight
        self.precision_weight = precision_weight
        self.precision_thresh = precision_thresh
        self.eps = 1e-6

    def forward(self, pred_warp, pred_conf, gt_warp, gt_mask, gt_h, gt_w, name=None, **kwargs):
        loss_weight = self.get_loss_weight(name)
        loss_dict = {}

        pred_warp = pred_warp.float()
        pred_conf = pred_conf.float()
        gt_warp = gt_warp.float()
        gt_mask = gt_mask.float()

        epe = (pred_warp - gt_warp).norm(dim=-1)
        epe = torch.clamp(epe, min=self.eps)

        ce_loss = F.binary_cross_entropy_with_logits(pred_conf[..., 0], gt_mask)
        loss_dict["ce_loss"] = ce_loss.mean() * self.conf_weight * loss_weight

        if self.precision_weight is not None:
            residual = (pred_warp - gt_warp).detach().abs()
            rx = residual[..., 0] * gt_w / 2
            ry = residual[..., 1] * gt_h / 2
            rnorm = torch.sqrt(rx ** 2 + ry ** 2).detach()

            confidence_gt_rx = (rx < self.precision_thresh).float()
            loss_conf_rx = F.binary_cross_entropy_with_logits(
                pred_conf[..., 1], confidence_gt_rx, reduction="none"
            )[gt_mask > 0.99]
            loss_conf_rx = torch.nan_to_num(loss_conf_rx).mean()

            confidence_gt_ry = (ry < self.precision_thresh).float()
            loss_conf_ry = F.binary_cross_entropy_with_logits(
                pred_conf[..., 2], confidence_gt_ry, reduction="none"
            )[gt_mask > 0.99]
            loss_conf_ry = torch.nan_to_num(loss_conf_ry).mean()

            confidence_gt_rnorm = (rnorm < self.precision_thresh).float()
            loss_conf_rnorm = F.binary_cross_entropy_with_logits(
                pred_conf[..., 3], confidence_gt_rnorm, reduction="none"
            )[gt_mask > 0.99]
            loss_conf_rnorm = torch.nan_to_num(loss_conf_rnorm).mean()

            loss_dict["prec_loss"] = (
                (loss_conf_rx + loss_conf_ry + loss_conf_rnorm) * loss_weight * self.precision_weight
            )

        a = self.alpha
        cs = self.scale_c
        x = epe[gt_mask > 0.99]
        reg_loss = cs ** a * ((x / cs) ** 2 + 1) ** (a / 2)

        if loss_weight == 0 or (not torch.any(reg_loss)):
            reg_loss = 0 * torch.sum(pred_warp)

        loss_dict["reg_loss"] = reg_loss.mean() * loss_weight

        loss = sum(loss_dict.values())
        return loss, loss_dict


class QueryMatchLoss2(QueryMatchLoss):
    def __init__(self, with_weight=False, **kwargs):
        super().__init__(**kwargs)

        self.with_weight = with_weight

    def forward(self, pred_warp, pred_conf, gt_warp, gt_mask, gt_h, gt_w, name=None, weight=None, **kwargs):
        loss_weight = self.get_loss_weight(name)
        loss_dict = {}

        pred_warp = pred_warp.float()
        pred_conf = pred_conf.float()
        gt_warp = gt_warp.float()
        gt_mask = gt_mask.float()

        epe = (pred_warp - gt_warp).norm(dim=-1)
        epe = torch.clamp(epe, min=self.eps)

        point_weight = None
        if self.with_weight and weight is not None:
            point_weight = weight.float().clamp_min(0.0)

        ce_loss = F.binary_cross_entropy_with_logits(pred_conf[..., 0], gt_mask, reduction="none")

        if point_weight is not None and point_weight.numel() > 0 and point_weight.sum() > self.eps:
            ce_loss = (ce_loss * point_weight).mean()
        else:
            ce_loss = ce_loss.mean()

        loss_dict["ce_loss"] = ce_loss.mean() * self.conf_weight * loss_weight

        valid_mask = gt_mask > 0.99
        valid_weight = point_weight[valid_mask] if point_weight is not None else None

        def reduce_loss(loss_tensor, selected_weight=None):
            loss_tensor = torch.nan_to_num(loss_tensor)
            if loss_tensor.numel() == 0:
                return 0 * torch.sum(pred_warp)
            if (
                selected_weight is not None
                and selected_weight.numel() == loss_tensor.numel()
                and selected_weight.sum() > self.eps
            ):
                return (loss_tensor * selected_weight).mean()
            return loss_tensor.mean()

        if self.precision_weight is not None:
            residual = (pred_warp - gt_warp).detach().abs()
            rx = residual[..., 0] * gt_w / 2
            ry = residual[..., 1] * gt_h / 2
            rnorm = torch.sqrt(rx ** 2 + ry ** 2).detach()

            confidence_gt_rx = (rx < self.precision_thresh).float()
            loss_conf_rx = F.binary_cross_entropy_with_logits(
                pred_conf[..., 1], confidence_gt_rx, reduction="none"
            )[valid_mask]
            loss_conf_rx = reduce_loss(loss_conf_rx, valid_weight)

            confidence_gt_ry = (ry < self.precision_thresh).float()
            loss_conf_ry = F.binary_cross_entropy_with_logits(
                pred_conf[..., 2], confidence_gt_ry, reduction="none"
            )[valid_mask]
            loss_conf_ry = reduce_loss(loss_conf_ry, valid_weight)

            confidence_gt_rnorm = (rnorm < self.precision_thresh).float()
            loss_conf_rnorm = F.binary_cross_entropy_with_logits(
                pred_conf[..., 3], confidence_gt_rnorm, reduction="none"
            )[valid_mask]
            loss_conf_rnorm = reduce_loss(loss_conf_rnorm, valid_weight)

            loss_dict["prec_loss"] = (
                (loss_conf_rx + loss_conf_ry + loss_conf_rnorm) * loss_weight * self.precision_weight
            )

        a = self.alpha
        cs = self.scale_c
        reg_loss_raw = cs ** a * ((epe / cs) ** 2 + 1) ** (a / 2)
        reg_loss = reg_loss_raw[valid_mask]
        reg_weight = valid_weight

        if loss_weight == 0 or (not torch.any(reg_loss)):
            reg_loss = 0 * torch.sum(pred_warp)
        else:
            reg_loss = reduce_loss(reg_loss, reg_weight)

        loss_dict["reg_loss"] = reg_loss * loss_weight

        loss = sum(loss_dict.values())
        return loss, loss_dict
