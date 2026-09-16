import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Loss


class ConfidenceBceLoss(Loss):
    def __init__(
        self,
        loss_weight=1,
        threshold=0.05,
        norm_type="L1",
        **kwargs,
    ):
        super().__init__(loss_weight)

        self.eps = 1e-6
        self.threshold = threshold
        self.norm_type = norm_type

    def denormalize(self, depth, scale):
        if scale is not None:
            return depth * scale
        else:
            return depth

    def forward(
        self,
        pred_conf,
        pred_depth,
        target_depth,
        valid_mask=None,
        scale=None,
        name=None,
        **kwargs,
    ):
        """
        pred_conf: B,H,W
        prompt_diffmap: B,H,W,3
        valid_mask: B,H,W
        """

        with torch.no_grad():
            target = self.denormalize(target_depth, scale)
            pred = self.denormalize(pred_depth, scale)
            prompt_diffmap = torch.abs(target - pred)

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(pred_conf))
        
        if torch.isnan(pred_conf).sum() > 0 or torch.isinf(pred_conf).sum() > 0 :
            return 0 * torch.sum(torch.nan_to_num(pred_conf))

        if pred_conf.shape[3] == 1:
            pred_conf = pred_conf.squeeze(3).unsqueeze(1)  # for interpolate

        _, H_gt, W_gt, _ = prompt_diffmap.shape

        pred_conf = F.interpolate(pred_conf, (H_gt, W_gt)).squeeze(1)  # B,H,W
        predict_sig = nn.functional.sigmoid(pred_conf)  # B,H,W

        if self.norm_type in ["l1", "L1"]:
            diff = prompt_diffmap.sum(dim=-1, keepdim=True).squeeze(3)
            gt_noise_postive = (diff > self.threshold).float()  # B,H,W
        elif self.norm_type in ["l2", "L2"]:
            diff = torch.torch.pow(prompt_diffmap, 2)
            diff = diff.sum(dim=-1, keepdim=True).squeeze(3)
            gt_noise_postive = (diff > self.threshold * self.threshold).float()  # B,H,W
        else:
            raise NotImplementedError("norm_type in ConfInputLoss should be L1 or L2 !")

        gt_noise_negative = 1.0 - gt_noise_postive

        # noise-free areas should have a 1.0 confidence
        loss = F.binary_cross_entropy(predict_sig, gt_noise_negative, reduction="none")[valid_mask]
        loss = torch.nan_to_num(loss)
        loss = loss.mean()

        return loss * loss_weight
