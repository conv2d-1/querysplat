import torch

from .base import Loss


class FactoredLLoss(Loss):
    "Criterion that supports different L-norms for the factored loss functions"

    def __init__(
        self,
        reduction="mean",
        loss_type="l1",
        loss_weight=1.0,
        eps=1e-8,
    ):
        super().__init__(loss_weight)

        self.reduction = reduction
        self.loss_type = loss_type
        self.eps = eps

    def distance(self, a, b, loss_type):
        if loss_type == "l1":
            # L1 distance
            return torch.abs(a - b).sum(dim=-1)
        elif loss_type == "l2":
            # Euclidean (L2 norm) distance
            return torch.norm(a - b, dim=-1)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}.")

    def forward(self, name, target_ray_directions, pred_ray_directions, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if (loss_weight == 0) or (target_ray_directions is None):
            return 0 * torch.nan_to_num(torch.sum(pred_ray_directions))

        a = target_ray_directions
        b = pred_ray_directions

        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 4, f"Bad shape = {a.shape}"
        valid_mask = torch.any(torch.abs(a) > self.eps, dim=-1)

        dist = self.distance(a, b, self.loss_type)
        dist = torch.nan_to_num(dist)
        dist_masked = dist * valid_mask.to(dist.dtype)

        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist_masked
        if self.reduction == "sum":
            return dist_masked.sum()
        if self.reduction == "mean":
            valid_count = valid_mask.sum()
            if valid_count.item() == 0:
                return dist.new_zeros(())
            return dist_masked.sum() / valid_count.to(dist.dtype)

        raise ValueError(f"bad {self.reduction=} mode")
