import logging

import torch

from .base import Loss, filter_by_quantile


class PointsL1Loss(Loss):
    def __init__(
        self,
        loss_weight=1,
        offset=0,
        threshold=3.0,
        zweighted=True,
        with_conf=False,
        l1_loss_scale=1.0,
        conf_loss_scale=1.0,
        conf_expp1=True,
        conf_weight=0.1,
        valid_range=None,
    ):
        super().__init__(loss_weight=loss_weight)

        self.offset = offset
        self.threshold = threshold
        self.zweighted = zweighted
        self.with_conf = with_conf
        self.l1_loss_scale = l1_loss_scale
        self.conf_loss_scale = conf_loss_scale
        self.eps = 1e-6
        self.conf_expp1 = conf_expp1
        self.conf_weight = conf_weight
        self.valid_range = valid_range

    def forward(self, pred_depth, target_depth, valid_mask, pred_conf=None, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return {"l1_loss": 0 * torch.nan_to_num(torch.sum(pred_depth))}

        diff_raw = torch.abs(pred_depth[valid_mask] - target_depth[valid_mask])
        diff_raw = diff_raw.sum(dim=-1, keepdim=True)

        if self.zweighted:
            reg_loss = (diff_raw + self.offset) / (target_depth[valid_mask][:, 2:3].abs() + self.offset + self.eps)
        else:
            reg_loss = diff_raw

        # reg_loss = check_and_fix_inf_nan(reg_loss, "PointsL1Loss_reg_loss")

        valid_range = None
        if self.valid_range is not None:
            if isinstance(self.valid_range, dict):
                valid_range = self.valid_range.get(name, None)
            else:
                valid_range = self.valid_range
            reg_loss = filter_by_quantile(reg_loss, valid_range)

        if self.threshold is not None:
            if isinstance(self.threshold, dict):
                threshold = self.threshold.get(name, self.threshold.get("default", 3.0))
                reg_loss = reg_loss[reg_loss < threshold]
            else:
                reg_loss = reg_loss[reg_loss < self.threshold]

        reg_loss = torch.nan_to_num(reg_loss.mean())  # NOTE: if len(loss) == 0, loss will be nan
        loss_dict = dict(l1_loss=reg_loss * loss_weight * self.l1_loss_scale)

        if self.with_conf and pred_conf is not None:
            pred_conf = pred_conf[valid_mask]

            if self.conf_expp1:
                pred_conf = 1 + torch.exp(pred_conf)

            conf, log_conf = pred_conf, torch.log(pred_conf)
            conf_loss = diff_raw * conf - self.conf_weight * log_conf

            # conf_loss = check_and_fix_inf_nan(conf_loss, "PointsL1Loss_conf_loss")

            if valid_range is not None:
                conf_loss = filter_by_quantile(conf_loss, valid_range)
            conf_loss = torch.nan_to_num(conf_loss.mean())

            loss_dict["conf_loss"] = conf_loss * loss_weight * self.conf_loss_scale

        return loss_dict


class GlobalPointZWeightedLoss(Loss):
    def __init__(
        self,
        loss_weight=1,
        offset=0,
        threshold=3.0,
        zweighted=True,
        with_conf=False,
        reg_loss_scale=1.0,
        conf_loss_scale=1.0,
        conf_expp1=True,
        conf_weight=0.1,
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
        self.conf_expp1 = conf_expp1
        self.conf_weight = conf_weight
        self.valid_range = valid_range

    def forward(self, pred_depth, target_depth, valid_mask, pred_conf=None, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return {"l1_loss": 0 * torch.sum(pred_depth)}

        diff_raw = torch.abs(pred_depth[valid_mask] - target_depth[valid_mask])
        diff_raw = diff_raw.sum(dim=-1, keepdim=True)

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

            if self.conf_expp1:
                pred_conf = 1 + torch.exp(pred_conf)

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


class ZWeightedLoss(Loss):
    def __init__(
        self,
        loss_weight=1,
        threshold=3.0,
        zweighted=True,
        with_conf=False,
        reg_loss_scale=1.0,
        conf_loss_scale=1.0,
        conf_expp1=True,
        conf_weight=0.1,
        valid_range=None,
    ):
        super().__init__(loss_weight=loss_weight)

        self.threshold = threshold
        self.zweighted = zweighted
        self.with_conf = with_conf
        self.reg_loss_scale = reg_loss_scale
        self.conf_loss_scale = conf_loss_scale
        self.eps = 1e-6
        self.conf_expp1 = conf_expp1
        self.conf_weight = conf_weight
        self.valid_range = valid_range

    def forward(self, pred_depth, target_depth, valid_mask, pred_conf=None, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return {"l1_loss": 0 * torch.sum(pred_depth)}

        diff_raw = torch.abs(pred_depth[valid_mask][:, -1] - target_depth[valid_mask][:, -1])

        if self.zweighted:
            diff = diff_raw / (target_depth[valid_mask][:, -1].abs() + self.eps)
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
            pred_conf = pred_conf.squeeze(-1)[valid_mask]

            if self.conf_expp1:
                pred_conf = 1 + torch.exp(pred_conf)
            # breakpoint()
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
