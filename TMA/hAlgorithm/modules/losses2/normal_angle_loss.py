import torch

from hAlgorithm.modules.utils.normal import PointMap2Normal

from .base import Loss, filter_by_quantile


class NormalAngleLoss(Loss):
    def __init__(
        self,
        loss_weight=1.0,
        target_is_normals=False,
        valid_range=None,
        debug=False,
        **kwargs,
    ):
        super().__init__(loss_weight=loss_weight)

        self.pointmap2normal = PointMap2Normal()
        self.loss_weight = loss_weight

        self.debug = debug

        self.target_is_normals = target_is_normals

        self.valid_range = valid_range

    def forward(self, pred_depth=None, target_depth=None, valid_mask=None, pred_normal=None, target_normal=None, name=None, **kwargs):
        """
        input and target_depth: surface normal input
        input: rgb images
        """

        loss_weight = self.get_loss_weight(name)

        if (loss_weight == 0) or (target_depth is None and target_normal is None):
            if pred_depth is not None:
                return 0 * torch.sum(torch.nan_to_num(pred_depth))
            elif pred_normal is not None:
                return 0 * torch.sum(torch.nan_to_num(pred_normal))
            else:
                return 0

        if pred_normal is None:
            pred_normal, _ = self.pointmap2normal(pred_depth, valid_mask)
            n, c, h, w = pred_normal.size()
            pred_normal = pred_normal.view(n, c, -1).permute(0, 2, 1).contiguous()
        else:
            n, h, w, c = pred_normal.size()
            pred_normal = pred_normal.view(n, -1, c)

        masks_normals = valid_mask
        if target_normal is None or not self.target_is_normals:
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

        loss = angle**2

        if self.valid_range is not None:
            loss = filter_by_quantile(loss, self.valid_range)

        loss = torch.nan_to_num(loss).mean()

        return loss * loss_weight


class NormalAngleL1Loss(NormalAngleLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, pred_depth=None, target_depth=None, valid_mask=None, pred_normal=None, target_normal=None, name=None, **kwargs):
        """
        input and target_depth: surface normal input
        input: rgb images
        """

        loss_weight = self.get_loss_weight(name)

        if (loss_weight == 0) or (target_depth is None and target_normal is None):
            if pred_depth is not None:
                return 0 * torch.sum(torch.nan_to_num(pred_depth))
            elif pred_normal is not None:
                return 0 * torch.sum(torch.nan_to_num(pred_normal))
            else:
                return 0

        if pred_normal is None:
            pred_normal, _ = self.pointmap2normal(pred_depth, valid_mask)
            n, c, h, w = pred_normal.size()
            pred_normal = pred_normal.view(n, c, -1).permute(0, 2, 1).contiguous()
        else:
            n, h, w, c = pred_normal.size()
            pred_normal = pred_normal.view(n, -1, c)

        masks_normals = valid_mask
        if target_normal is None:
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

        loss = angle.abs()

        if self.valid_range is not None:
            loss = filter_by_quantile(loss, self.valid_range)

        loss = torch.nan_to_num(loss).mean()

        return loss * loss_weight
