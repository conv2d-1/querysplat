import torch

from .base import Loss, check_and_fix_inf_nan


class GaussianXYZOffsetLoss(Loss):
    """Regularize learned Gaussian xyz offsets toward zero (dev_lx ffgs_mv)."""

    def __init__(self, loss_weight=1.0, loss_type="l2"):
        super().__init__(loss_weight=loss_weight)
        if loss_type not in ("l1", "l2"):
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.loss_type = loss_type

    def forward(self, gs_offset_xyz=None, name=None, **kwargs):
        loss_weight = self.get_loss_weight(name)

        if gs_offset_xyz is None:
            return None

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(gs_offset_xyz))

        if self.loss_type == "l1":
            loss = gs_offset_xyz.abs().mean()
        else:
            loss = gs_offset_xyz.square().mean()

        loss = check_and_fix_inf_nan(loss, "GaussianXYZOffsetLoss")
        return loss * loss_weight
