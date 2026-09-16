import logging

import torch
import torch.nn as nn

from .base import Loss, filter_by_quantile, check_and_fix_inf_nan


class GlobalPointZWeightedLoss(nn.Module):
    """
    Compute Global point map supervision, eq.(3) off MoGe
    """

    def __init__(self, loss_weight=1, offset=0, threshold=3.0, zweighted=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.offset = offset
        self.threshold = threshold
        self.zweighted = zweighted
        self.eps = 1e-6

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

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(prediction)

        diff = prediction[mask] - target[mask]
        diff = torch.abs(diff).sum(dim=-1, keepdim=True)
        if self.zweighted:
            diff = (diff + self.offset) / (target[mask][:, 2:3] + self.offset + self.eps)
        if self.threshold is not None:
            loss = diff[diff < self.threshold].mean()
        else:
            loss = diff.mean()

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            logging.warning(f"Data {name}, GlobalPointZLoss NAN error, {loss}")

        return loss * loss_weight


class GlobalPointZWeightedLossV2(nn.Module):
    """
    Compute Global point map supervision, eq.(3) off MoGe
    """

    def __init__(self, loss_weight=1, offset=0, threshold=3.0, zweighted=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.offset = offset
        self.threshold = threshold
        self.zweighted = zweighted
        self.eps = 1e-6

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

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(prediction)

        diff = prediction[mask] - target[mask]
        diff = torch.abs(diff).sum(dim=-1, keepdim=True)
        if self.zweighted:
            diff = (diff + self.offset) / (target[mask][:, 2:3].abs() + self.offset + self.eps)
        if self.threshold is not None:
            loss = diff[diff < self.threshold].mean()
        else:
            loss = diff.mean()

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            logging.warning(f"Data {name}, GlobalPointZLoss NAN error, {loss}")

        return loss * loss_weight


class GlobalPointZWeightedLossV3(nn.Module):
    """
    Compute Global point map supervision, eq.(3) off MoGe
    """

    def __init__(
        self,
        loss_weight=1,
        offset=0,
        threshold=3.0,
        zweighted=True,
        with_conf=False,
        conf_loss_scale=1.0,
        reg_loss_scale=1.0,
        conf_z=False,
        conf_expp1=True,
        conf_weight=0.1,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.offset = offset
        self.threshold = threshold
        self.zweighted = zweighted
        self.with_conf = with_conf
        self.conf_loss_scale = conf_loss_scale
        self.reg_loss_scale = reg_loss_scale
        self.conf_z = conf_z
        self.eps = 1e-6
        self.conf_expp1 = conf_expp1
        self.conf_weight = conf_weight

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

    def forward(self, prediction, target, mask, confidence=None, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(prediction)

        diff_raw = torch.abs(prediction[mask] - target[mask])
        if self.conf_z:
            diff_raw_z = diff_raw[..., 2:3]
        diff_raw = diff_raw.sum(dim=-1, keepdim=True)
        if self.zweighted:
            diff = (diff_raw + self.offset) / (target[mask][:, 2:3].abs() + self.offset + self.eps)
        else:
            diff = diff_raw

        if self.threshold is not None:
            if isinstance(self.threshold, dict):
                threshold = self.threshold.get(name, self.threshold.get("default", 3.0))
                loss = diff[diff < threshold].mean()
            else:
                loss = diff[diff < self.threshold].mean()
        else:
            loss = diff.mean()

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            logging.warning(f"Data {name}, GlobalPointZLoss NAN error, {loss}")

        if self.with_conf and confidence is not None:
            confidence = confidence[mask]
            if self.conf_expp1:
                confidence = 1 + torch.exp(confidence)
            conf, log_conf = confidence, torch.log(confidence)
            if self.conf_z:
                conf_loss = diff_raw_z * conf - self.conf_weight * log_conf
            else:
                conf_loss = diff_raw * conf - self.conf_weight * log_conf
            conf_loss = conf_loss.mean()

            if torch.isnan(conf_loss).item() | torch.isinf(conf_loss).item():
                conf_loss = 0 * torch.sum(prediction)
                logging.warning(f"Data {name}, GlobalPointZLoss NAN error, {conf_loss}")

            loss_dict = dict(
                l1_loss=loss * loss_weight * self.reg_loss_scale,
                conf_loss=conf_loss * loss_weight * self.conf_loss_scale,
            )
            
        else:
            loss_dict = dict(l1_loss=loss * loss_weight)

        return loss_dict


class GlobalPointZWeightedLossV4(Loss):
    """GlobalPointZWeightedLossV4, add valid_range"""

    def __init__(
        self,
        loss_weight=1,
        offset=0,
        threshold=3.0,
        zweighted=True,
        with_conf=False,
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
        self.conf_loss_scale = conf_loss_scale
        self.eps = 1e-6
        self.conf_expp1 = conf_expp1
        self.conf_weight = conf_weight
        self.valid_range = valid_range

    def forward(self, prediction, target, mask, confidence=None, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(prediction)

        diff_raw = torch.abs(prediction[mask] - target[mask])
        diff_raw = diff_raw.sum(dim=-1, keepdim=True)
    
        if self.zweighted:
            diff = (diff_raw + self.offset) / (target[mask][:, 2:3].abs() + self.offset + self.eps)
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
            loss = 0 * torch.sum(prediction)
            logging.warning(f"Data {name}, GlobalPointZLoss NAN error, {loss}")

        if self.with_conf and confidence is not None:
            confidence = confidence[mask]

            if self.conf_expp1:
                confidence = 1 + torch.exp(confidence)

            conf, log_conf = confidence, torch.log(confidence)
            conf_loss = diff_raw * conf - self.conf_weight * log_conf
            if valid_range is not None:
                conf_loss = filter_by_quantile(conf_loss, valid_range)
            conf_loss = conf_loss.mean()

            if torch.isnan(conf_loss).item() | torch.isinf(conf_loss).item():
                conf_loss = 0 * torch.sum(prediction)
                logging.warning(f"Data {name}, GlobalPointZLoss NAN error, {conf_loss}")

            loss_dict = dict(
                l1_loss=loss * loss_weight,
                conf_loss=conf_loss * loss_weight * self.conf_loss_scale,
            )
            return loss_dict
        else:
            return loss * loss_weight


class GlobalPointZWeightedLossV5(Loss):
    """GlobalPointZWeightedLossV5, add check_and_fix_inf_nan."""

    def __init__(
        self,
        loss_weight=1,
        offset=0,
        threshold=3.0,
        zweighted=True,
        with_conf=False,
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
        self.conf_loss_scale = conf_loss_scale
        self.eps = 1e-6
        self.conf_expp1 = conf_expp1
        self.conf_weight = conf_weight
        self.valid_range = valid_range

    def forward(self, prediction, target, mask, confidence=None, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(prediction)

        diff_raw = torch.abs(prediction[mask] - target[mask])
        diff_raw = diff_raw.sum(dim=-1, keepdim=True)
    
        if self.zweighted:
            diff = (diff_raw + self.offset) / (target[mask][:, 2:3].abs() + self.offset + self.eps)
        else:
            diff = diff_raw

        diff = check_and_fix_inf_nan(diff, "GlobalPointZWeightedLossV5_diff")

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

        if self.with_conf and confidence is not None:
            confidence = confidence[mask]

            if self.conf_expp1:
                confidence = 1 + torch.exp(confidence)

            conf, log_conf = confidence, torch.log(confidence)
            conf_loss = diff_raw * conf - self.conf_weight * log_conf
            conf_loss = check_and_fix_inf_nan(conf_loss, "GlobalPointZWeightedLossV5_conf_loss")

            if valid_range is not None:
                conf_loss = filter_by_quantile(conf_loss, valid_range)
            conf_loss = conf_loss.mean()

            loss_dict = dict(
                l1_loss=loss * loss_weight,
                conf_loss=conf_loss * loss_weight * self.conf_loss_scale,
            )
            return loss_dict
        else:
            return loss * loss_weight
