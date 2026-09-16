import logging
import torch

from .base import Loss, filter_by_quantile

class GlobalPointZWeightedLoss(Loss):
    def __init__(
        self,
        loss_weight=1,
        offset=0,
        threshold=3.0,
        zweighted=False,
        with_conf=True,
        reg_loss_scale=0.0,
        conf_loss_scale=1.0,
        conf_weight=0.2,
        valid_range=None,
    ):
        super().__init__(loss_weight=loss_weight)

        self.offset = offset
        self.threshold = threshold
        self.zweighted = zweighted
        self.with_conf = with_conf
        self.reg_loss_scale = reg_loss_scale
        self.conf_loss_scale = conf_loss_scale
        self.eps = 1e-6
        self.conf_weight = conf_weight
        self.valid_range = valid_range

    def forward(self, pred_depth, target_depth, valid_mask, pred_conf=None, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(pred_depth)

        # diff_raw = torch.abs(pred_depth[valid_mask] - target_depth[valid_mask])
        # diff_raw = diff_raw.sum(dim=-1, keepdim=True)
        diff_raw = torch.norm(pred_depth[valid_mask] - target_depth[valid_mask], dim=-1,keepdim=True)
    
        if self.zweighted:
            diff = (diff_raw + self.offset) / (target_depth[valid_mask][:, 2:3].abs() + self.offset + self.eps)
        else:
            diff = diff_raw

        valid_range = None
        if self.valid_range is not None:
            if isinstance(self.valid_range, dict):
                valid_range = self.valid_range.get(name, None)
            else:
                valid_range = self.valid_range

        if self.threshold is not None:
            if valid_range is not None:
                diff = filter_by_quantile(diff, valid_range)
            if isinstance(self.threshold, dict):
                threshold = self.threshold.get(name, self.threshold.get("default", 3.0))
                loss = diff[diff < threshold].mean()
            else:
                loss = diff[diff < self.threshold].mean()
        else:
            if valid_range is not None:
                diff = filter_by_quantile(diff, valid_range)
            loss = diff.mean()

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(pred_depth)
            logging.warning(f"Data {name}, GlobalPointZLoss NAN error, {loss}")

            loss_dict = dict(l1_loss=loss, conf_loss=loss)
            return loss_dict

        if self.with_conf and pred_conf is not None:
            pred_conf = pred_conf[valid_mask]

            conf, log_conf = pred_conf, torch.log(pred_conf)
            conf_loss = diff_raw * conf - self.conf_weight * log_conf
            if valid_range is not None:
                conf_loss = filter_by_quantile(conf_loss, valid_range)
            conf_loss = conf_loss.mean()

            if torch.isnan(conf_loss).item() | torch.isinf(conf_loss).item():
                conf_loss = 0 * torch.sum(pred_depth)
                logging.warning(f"Data {name}, GlobalPointZLoss NAN error, {conf_loss}")

            loss_dict = dict(
                l1_loss=loss * loss_weight * self.reg_loss_scale,
                conf_loss=conf_loss * loss_weight * self.conf_loss_scale,
            )

        else:
            loss_dict = dict(l1_loss=loss * loss_weight)
        
        return loss_dict