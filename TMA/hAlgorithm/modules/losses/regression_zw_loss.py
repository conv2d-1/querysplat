import torch
from .base import Loss, check_and_fix_inf_nan, filter_by_quantile


class RegressionZWeightedLoss(Loss):
    def __init__(
        self,
        loss_weight=1.0,
        zweighted=True,
        threshold=None,
        with_conf=False,
        conf_expp1=True,
        conf_gamma=1.0,
        conf_alpha=0.1,
        reg_weight=1.0,
        conf_weight=1.0,
        valid_range=None,
        use_dim=None,
    ):
        super().__init__(loss_weight=loss_weight)

        self.zweighted = zweighted

        self.with_conf = with_conf
        self.conf_expp1 = conf_expp1
        self.conf_gamma = conf_gamma
        self.conf_alpha = conf_alpha

        self.reg_weight = reg_weight
        self.conf_weight = conf_weight

        self.valid_range = valid_range
        self.use_dim = use_dim

        self.threshold = threshold

        self.eps = 1e-6

    def forward(self, prediction, target, mask, confidence=None, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return (0.0 * prediction).mean()

        target = check_and_fix_inf_nan(target, "target")

        if prediction.shape[-1] == 1 and target.shape[-1] == 3:
            target = target[..., -1:]

        if self.use_dim is None:
            reg_loss = torch.abs(prediction[mask] - target[mask]).sum(dim=-1)
        elif isinstance(self.use_dim, int):
            reg_loss = torch.abs(prediction[mask] - target[mask])[:, self.use_dim]
        elif isinstance(self.use_dim, (list, tuple)):
            reg_loss = torch.abs(prediction[mask] - target[mask])[:, self.use_dim].sum(dim=-1)
        
        if self.with_conf and confidence is not None:
            confidence = confidence[mask][:, 0]
            if self.conf_expp1:
                confidence = 1 + torch.exp(confidence)

            conf, log_conf = confidence, torch.log(confidence)
            conf_loss = self.conf_gamma * reg_loss * conf - self.conf_alpha * log_conf

            # Filter out outliers using quantile-based thresholding
            if self.valid_range is not None and self.valid_range > 0:
                conf_loss = filter_by_quantile(conf_loss, self.valid_range)

            conf_loss = check_and_fix_inf_nan(conf_loss, f"conf_loss")
            conf_loss = conf_loss.mean()
            
        else:
            conf_loss = (0.0 * prediction).mean()

        # Process regular regression loss
        if reg_loss.numel() > 0:
            if self.zweighted:
                reg_loss = reg_loss / (target[mask][:, -1].abs() + self.eps)
            
            # Filter out outliers using quantile-based thresholding
            if self.valid_range is not None and self.valid_range > 0:
                reg_loss = filter_by_quantile(reg_loss, self.valid_range)
            
            if self.threshold is not None:
                reg_loss = reg_loss[reg_loss < self.threshold]

            reg_loss = check_and_fix_inf_nan(reg_loss, f"reg_loss")
            reg_loss = reg_loss.mean()
        else:
            reg_loss = (0.0 * prediction).mean()

        loss_dict = dict(
            l1_loss=reg_loss * loss_weight * self.reg_weight,
            conf_loss=conf_loss * loss_weight * self.conf_weight,
        )
        return loss_dict
