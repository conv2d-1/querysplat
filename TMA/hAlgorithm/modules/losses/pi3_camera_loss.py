import torch

from hAlgorithm.modules.models.pi3.utils.pose_enc import intri_to_fov
from .base import Loss, check_and_fix_inf_nan, filter_by_quantile


class Pi3CameraLoss(Loss):
    def __init__(
        self,
        weight_T=1.0,
        weight_R=1.0,
        loss_weight=1.0,
        normalize=True,
        **kwargs,
    ):
        super().__init__(loss_weight=loss_weight)

        self.weight_T = weight_T
        self.weight_R = weight_R
        self.normalize = normalize

    def get_loss(self, pred, gt, scale):
        loss_T = (pred[..., :3, 3] * scale[..., 0, 0] - gt[..., :3, 3]).abs().mean()
        loss_R = (pred[..., :3, :3] - gt[..., :3, :3]).abs().mean()
        return loss_T, loss_R

    def normalize_cameras(self, base_index, c2w_pred, c2w_gt, scale):
        base_w2c_pred = c2w_pred[:, base_index : base_index + 1].inverse()
        base_w2c_gt = c2w_gt[:, base_index : base_index + 1].inverse()

        cur_c2w_pred = base_w2c_pred @ c2w_pred
        cur_c2w_gt = base_w2c_gt @ c2w_gt

        loss_T, loss_R = self.get_loss(cur_c2w_pred, cur_c2w_gt, scale=scale)

        return loss_T, loss_R

    def forward(
        self,
        c2w_pred,
        c2w_gt,
        mask=None,
        scale=None,
        name=None,
        **kwargs,
    ):
        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(c2w_pred)), dict()

        weight_T = self.weight_T
        weight_R = self.weight_R

        # Extract predicted and ground truth components
        mask_valid = mask

        batch_valid_mask = mask_valid[:, 0, 0].sum(dim=[-1, -2]) > 100

        loss_T = loss_R = 0

        if batch_valid_mask.sum() == 0:
            loss_T = (c2w_pred * 0).mean()
            loss_R = (c2w_pred * 0).mean()
        elif not self.normalize:
            loss_T, loss_R = self.get_loss(c2w_pred, c2w_gt, scale=scale)
        else:
            N = c2w_pred.shape[1]
            for i in range(N):
                loss_T_i, loss_R_i = self.normalize_cameras(
                    base_index=i, c2w_pred=c2w_pred, c2w_gt=c2w_gt, scale=scale
                )
                loss_T += loss_T_i
                loss_R += loss_R_i

            loss_T = loss_T / N
            loss_R = loss_R / N

        loss_T = loss_T * weight_T
        loss_R = loss_R * weight_R
        loss_camera = loss_T + loss_R

        loss_dict = {
            "loss_T": loss_T,
            "loss_R": loss_R,
        }

        return loss_camera, loss_dict


class W2CPi3CameraLoss(Loss):
    def __init__(
        self,
        weight_T=1.0,
        weight_R=1.0,
        loss_weight=1.0,
        weight_fl=0.5,
        normalize=True,
        loss_type="l1",
        **kwargs,
    ):
        super().__init__(loss_weight=loss_weight)

        self.weight_T = weight_T
        self.weight_R = weight_R
        self.weight_fl =  weight_fl
        self.normalize = normalize
        self.loss_type = loss_type

    def get_loss(self, pred, gt, scale):
        if self.loss_type == "l1":
            loss_T = (pred[..., :3, 3] * scale[..., 0, 0] - gt[..., :3, 3]).abs().mean()
            loss_R = (pred[..., :3, :3] - gt[..., :3, :3]).abs().mean()
        else:
            raise NotImplementedError
        return loss_T, loss_R

    def normalize_cameras(self, base_index, w2c_pred, w2c_gt, scale):
        base_c2w_pred = w2c_pred[:, base_index : base_index + 1].inverse()
        base_c2w_gt = w2c_gt[:, base_index : base_index + 1].inverse()

        cur_w2c_pred = w2c_pred @ base_c2w_pred
        cur_w2c_gt = w2c_gt @ base_c2w_gt

        loss_T, loss_R = self.get_loss(cur_w2c_pred, cur_w2c_gt, scale=scale)

        return loss_T, loss_R

    def forward(
        self,
        w2c_pred,
        w2c_gt,
        mask,
        scale,
        fov_pred=None,
        intrinsic_gt=None,
        image_size_hw=None,
        name=None,
        **kwargs,
    ):
        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(w2c_pred)), dict()

        # Extract predicted and ground truth components
        batch_valid_mask = mask[:, 0, 0].sum(dim=[-1, -2]) > 100

        loss_T = loss_R = loss_fl = 0

        # pose
        if batch_valid_mask.sum() == 0:
            loss_T = (w2c_pred * 0).mean()
            loss_R = (w2c_pred * 0).mean()
        elif not self.normalize:
            loss_T, loss_R = self.get_loss(w2c_pred.clone(), w2c_gt.clone(), scale=scale)
        else:
            N = w2c_pred.shape[1]
            for i in range(N):
                loss_T_i, loss_R_i = self.normalize_cameras(
                    base_index=i, w2c_pred=w2c_pred.clone(), w2c_gt=w2c_gt.clone(), scale=scale
                )
                loss_T += loss_T_i
                loss_R += loss_R_i

            loss_T = loss_T / N
            loss_R = loss_R / N
        
        # intrinsics fov
        if fov_pred is not None and intrinsic_gt is not None:
            if batch_valid_mask.sum() == 0:
                loss_fl = (w2c_pred * 0).mean()
            else:
                fov_w, fov_h = intri_to_fov(intrinsic_gt, image_size_hw=image_size_hw)
                loss_fl = (fov_pred - torch.cat([fov_h[..., None], fov_w[..., None]], dim=-1)).abs().mean()

        loss_T = loss_T * self.weight_T
        loss_R = loss_R * self.weight_R
        loss_fl = loss_fl * self.weight_fl
        loss_camera = loss_T + loss_R + loss_fl

        loss_dict = {
            "loss_T": loss_T,
            "loss_R": loss_R,
            "loss_fl": loss_fl,
        }

        return loss_camera, loss_dict
