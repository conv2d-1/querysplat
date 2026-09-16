import torch
import torch.nn as nn

from hAlgorithm.modules.utils.normal import PointMap2Normal

from .base import Loss, filter_by_quantile


class NormalCosineLoss(Loss):
    def __init__(
        self,
        loss_weight=1.0,
        start_iter=None,
        sample_ratio=None,
        select_ratio=None,
        descending=False,
        target_is_normals=False,
        valid_range=None,
        debug=False,
        **kwargs,
    ):
        super().__init__(loss_weight=loss_weight)

        self.pointmap2normal = PointMap2Normal()
        self.loss_weight = loss_weight

        self.start_iter = start_iter
        self.debug = debug

        self.sample_ratio = sample_ratio
        self.select_ratio = select_ratio
        self.descending = descending

        self.target_is_normals = target_is_normals
        self.valid_range = valid_range

    def select_index(self, valid_mask):
        B, N = valid_mask.shape
        pix_idx_mat = torch.arange(N, device=valid_mask.device)
        new_masks = []
        for i in range(B):
            inputs_index = torch.masked_select(pix_idx_mat, valid_mask[i])
            num_effect_pixels = len(inputs_index)

            intend_sample_num = int(N * self.sample_ratio)
            sample_num = intend_sample_num if num_effect_pixels >= intend_sample_num else num_effect_pixels

            shuffle_effect_pixels = torch.randperm(num_effect_pixels, device=valid_mask.device)
            p1i = inputs_index[shuffle_effect_pixels[:sample_num]]

            new_mask = torch.zeros(N, dtype=torch.bool, device=valid_mask.device)
            new_mask[p1i] = True

            new_masks.append(new_mask)

        new_masks = torch.stack(new_masks, dim=0)

        return new_masks

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

        if self.sample_ratio is not None:
            masks_normals = self.select_index(masks_normals)

        # angle between target_depth and pred normal
        cos_angle = torch.einsum(
            "nc,nc->n",
            target_normal[masks_normals],
            pred_normal[masks_normals],
        )
        loss = 1 - cos_angle**2
        loss = torch.nan_to_num(loss)

        if self.valid_range is not None:
            loss = filter_by_quantile(loss, self.valid_range)

        if self.select_ratio is not None:
            loss, indices = torch.sort(loss, dim=0, descending=self.descending)
            loss = loss[int(loss.size(0) * self.select_ratio) :].mean()
        else:
            loss = loss.mean()

        return loss * loss_weight
