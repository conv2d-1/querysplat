from math import ceil, floor

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    extri_intri_to_pose_encoding,
    pose_encoding_to_extri_intri,
)

from .base import Loss, check_and_fix_inf_nan


def huber_loss(x, y, delta=1.0, name=None):
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).float()
    # logging.info(f"{name}, mean{abs_diff.mean().detach().cpu().item()}, max{abs_diff.max().detach().cpu().item()}, min{abs_diff.min().detach().cpu().item()}, median{abs_diff.median().detach().cpu().item()}")
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1", delta=1.0, scale=None, r_scale=None):
    if loss_type == "l1":
        if scale is not None:
            loss_T = (cur_pred_pose_enc[..., :3] * scale[..., 0, 0] - gt_pose_encoding[..., :3] * scale[..., 0, 0]).abs()
        else:
            loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
        if r_scale is not None:
            loss_R = (cur_pred_pose_enc[..., 3:7] * scale[..., 0, 0] - gt_pose_encoding[..., 3:7] * scale[..., 0, 0]).abs()
        else:
            loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        if scale is not None:
            loss_T = (cur_pred_pose_enc[..., :3] * scale[..., 0, 0] - gt_pose_encoding[..., :3] * scale[..., 0, 0]).norm(dim=-1, keepdim=True)
        else:
            loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
        if r_scale is not None:
            loss_R = (cur_pred_pose_enc[..., 3:7] * scale[..., 0, 0] - gt_pose_encoding[..., 3:7] * scale[..., 0, 0]).norm(dim=-1)
        else:
            loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
    elif loss_type == "huber":
        loss_T = huber_loss(cur_pred_pose_enc[..., :3], gt_pose_encoding[..., :3], delta=delta, name="t")
        loss_R = huber_loss(cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7], delta=delta, name="r")
        loss_fl = huber_loss(cur_pred_pose_enc[..., 7:], gt_pose_encoding[..., 7:], delta=delta, name="fl")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

    loss_T = loss_T.clamp(max=100)  # TODO: remove this
    loss_T = loss_T.mean()
    loss_R = loss_R.mean()
    loss_fl = loss_fl.mean()

    return loss_T, loss_R, loss_fl


class VGGTCameraLoss(Loss):
    def __init__(
        self,
        loss_type="l1",
        gamma=0.6,
        pose_encoding_type="absT_quaR_FoV",
        weight_T=1.0,
        weight_R=1.0,
        weight_fl=0.5,
        loss_weight=1.0,
        delta=0.1,
        **kwargs,
    ):
        super().__init__(loss_weight=loss_weight)

        self.loss_type = loss_type
        self.gamma = gamma
        self.pose_encoding_type = pose_encoding_type
        self.weight_T = weight_T
        self.weight_R = weight_R
        self.weight_fl = weight_fl
        self.loss_weight = loss_weight
        self.delta = delta

    def forward(
        self,
        pose_enc,
        target_extrinsics,
        target_intrinsic,
        image_size_hw,
        valid_mask,
        name=None,
        **kwargs,
    ):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(pose_enc[0])), dict()

        if not isinstance(pose_enc, (list, tuple)):
            pose_enc = [pose_enc]

        loss_type = self.loss_type
        gamma = self.gamma
        pose_encoding_type = self.pose_encoding_type
        weight_T = self.weight_T
        weight_R = self.weight_R
        weight_fl = self.weight_fl

        # Extract predicted and ground truth components
        valid_mask = valid_mask

        batch_valid_mask = valid_mask[:, 0, 0].sum(dim=[-1, -2]) > 100
        num_predictions = len(pose_enc)

        gt_pose_encoding = extri_intri_to_pose_encoding(
            target_extrinsics,
            target_intrinsic,
            image_size_hw,
            pose_encoding_type=pose_encoding_type,
        )

        loss_T = loss_R = loss_fl = 0

        for i in range(num_predictions):
            i_weight = gamma ** (num_predictions - i - 1)

            cur_pred_pose_enc = pose_enc[i]

            if batch_valid_mask.sum() == 0:
                loss_T_i = (cur_pred_pose_enc * 0).mean()
                loss_R_i = (cur_pred_pose_enc * 0).mean()
                loss_fl_i = (cur_pred_pose_enc * 0).mean()
            else:
                loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(
                    cur_pred_pose_enc[batch_valid_mask].clone(),
                    gt_pose_encoding[batch_valid_mask].clone(),
                    loss_type=loss_type,
                    delta=self.delta,
                )
            loss_T += loss_T_i * i_weight
            loss_R += loss_R_i * i_weight
            loss_fl += loss_fl_i * i_weight

        loss_T = loss_T / num_predictions * weight_T
        loss_R = loss_R / num_predictions * weight_R
        loss_fl = loss_fl / num_predictions * weight_fl
        loss_camera = loss_T + loss_R + loss_fl

        loss_dict = {
            # "loss_camera": loss_camera,
            "loss_T": loss_T,
            "loss_R": loss_R,
            "loss_fl": loss_fl,
        }

        return loss_camera, loss_dict


class Pi3CameraLoss(VGGTCameraLoss):
    """Pi3CameraLoss, supervision for each coordinate system."""

    def __init__(self, w2c=True, **kwargs):
        super().__init__(**kwargs)
        self.w2c = w2c

    def normalize_cameras(self, base_index, w2c_pred, w2c_gt):
        input_dtype = w2c_pred.dtype
        w2c_pred = w2c_pred.float()
        w2c_gt = w2c_gt.float()

        base_c2w_pred = w2c_pred[:, base_index : base_index + 1].inverse()
        base_c2w_gt = w2c_gt[:, base_index : base_index + 1].inverse()

        cur_w2c_pred = w2c_pred @ base_c2w_pred
        cur_w2c_gt = w2c_gt @ base_c2w_gt

        return cur_w2c_pred.to(input_dtype), cur_w2c_gt.to(input_dtype)

    def normalize_cameras_c2w(self, base_index, c2w_pred, c2w_gt):
        input_dtype = c2w_pred.dtype
        c2w_pred = c2w_pred.float()
        c2w_gt = c2w_gt.float()

        base_w2c_pred = c2w_pred[:, base_index : base_index + 1].inverse()
        base_w2c_gt = c2w_gt[:, base_index : base_index + 1].inverse()

        cur_c2w_pred = base_w2c_pred @ c2w_pred
        cur_c2w_gt = base_w2c_gt @ c2w_gt

        return cur_c2w_pred.to(input_dtype), cur_c2w_gt.to(input_dtype)

    def forward(
        self,
        pose_enc,
        target_extrinsics,
        target_intrinsic,
        image_size_hw,
        valid_mask,
        scale=None,
        name=None,
        **kwargs,
    ):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(pose_enc[0])), dict()

        if not isinstance(pose_enc, (list, tuple)):
            pose_enc = [pose_enc]

        loss_type = self.loss_type
        gamma = self.gamma
        pose_encoding_type = self.pose_encoding_type
        weight_T = self.weight_T
        weight_R = self.weight_R
        weight_fl = self.weight_fl
        
        if valid_mask is not None:
            batch_valid_mask = valid_mask[:, 0, 0].sum(dim=[-1, -2]) > 100
        else:
            batch_valid_mask = None

        if scale is not None and batch_valid_mask is not None:
            scale = scale[batch_valid_mask]

        num_predictions = len(pose_enc)

        loss_T = loss_R = loss_fl = 0

        for i in range(num_predictions):
            i_weight = gamma ** (num_predictions - i - 1)

            cur_pred_pose_enc = pose_enc[i]
            N = cur_pred_pose_enc.shape[1]

            extrinsic_pred, intrinsics_pred = pose_encoding_to_extri_intri(
                pose_encoding=cur_pred_pose_enc,
                image_size_hw=image_size_hw,  # e.g., (256, 512)
                pose_encoding_type=pose_encoding_type,
                build_intrinsics=True,
            )

            for ni in range(N):
                if self.w2c:
                    cur_extrinsic_pred, cur_target_extrinsics = self.normalize_cameras(ni, extrinsic_pred, target_extrinsics)
                else:
                    cur_extrinsic_pred, cur_target_extrinsics = self.normalize_cameras_c2w(ni, extrinsic_pred, target_extrinsics)

                pred_pose_encoding = extri_intri_to_pose_encoding(
                    cur_extrinsic_pred,
                    intrinsics_pred,
                    image_size_hw,
                    pose_encoding_type=pose_encoding_type,
                )

                gt_pose_encoding = extri_intri_to_pose_encoding(
                    cur_target_extrinsics,
                    target_intrinsic,
                    image_size_hw,
                    pose_encoding_type=pose_encoding_type,
                )

                if batch_valid_mask is not None and batch_valid_mask.sum() == 0:
                    loss_T_i = (cur_pred_pose_enc * 0).mean()
                    loss_R_i = (cur_pred_pose_enc * 0).mean()
                    loss_fl_i = (cur_pred_pose_enc * 0).mean()
                elif batch_valid_mask is not None:
                    loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(
                        pred_pose_encoding[batch_valid_mask].clone(),
                        gt_pose_encoding[batch_valid_mask].clone(),
                        loss_type=loss_type,
                        delta=self.delta,
                        scale=scale,
                    )
                else:
                    loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(
                        pred_pose_encoding.clone(),
                        gt_pose_encoding.clone(),
                        loss_type=loss_type,
                        delta=self.delta,
                        scale=scale,
                    )

                loss_T += loss_T_i * i_weight / N
                loss_R += loss_R_i * i_weight / N
                loss_fl += loss_fl_i * i_weight / N

        loss_T = loss_T / num_predictions * weight_T
        loss_R = loss_R / num_predictions * weight_R
        loss_fl = loss_fl / num_predictions * weight_fl
        loss_camera = loss_T + loss_R + loss_fl

        loss_dict = {
            # "loss_camera": loss_camera,
            "loss_T": loss_T,
            "loss_R": loss_R,
            "loss_fl": loss_fl,
        }

        return loss_camera, loss_dict


class Pi3CameraLossV2(Pi3CameraLoss):
    """Pi3CameraLossV2, add r_scale."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self,
        pose_enc,
        target_extrinsics,
        target_intrinsic,
        image_size_hw,
        valid_mask,
        scale=None,
        name=None,
        **kwargs,
    ):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(pose_enc[0])), dict()

        if not isinstance(pose_enc, (list, tuple)):
            pose_enc = [pose_enc]

        loss_type = self.loss_type
        gamma = self.gamma
        pose_encoding_type = self.pose_encoding_type
        weight_T = self.weight_T
        weight_R = self.weight_R
        weight_fl = self.weight_fl

        batch_valid_mask = valid_mask[:, 0, 0].sum(dim=[-1, -2]) > 100

        if scale is not None:
            scale = scale[batch_valid_mask]

        num_predictions = len(pose_enc)

        loss_T = loss_R = loss_fl = 0

        for i in range(num_predictions):
            i_weight = gamma ** (num_predictions - i - 1)

            cur_pred_pose_enc = pose_enc[i]
            N = cur_pred_pose_enc.shape[1]

            extrinsic_pred, intrinsics_pred = pose_encoding_to_extri_intri(
                pose_encoding=cur_pred_pose_enc,
                image_size_hw=image_size_hw,  # e.g., (256, 512)
                pose_encoding_type=pose_encoding_type,
                build_intrinsics=True,
            )

            for ni in range(N):
                cur_extrinsic_pred, cur_target_extrinsics = self.normalize_cameras(ni, extrinsic_pred, target_extrinsics)

                pred_pose_encoding = extri_intri_to_pose_encoding(
                    cur_extrinsic_pred,
                    intrinsics_pred,
                    image_size_hw,
                    pose_encoding_type=pose_encoding_type,
                )

                gt_pose_encoding = extri_intri_to_pose_encoding(
                    cur_target_extrinsics,
                    target_intrinsic,
                    image_size_hw,
                    pose_encoding_type=pose_encoding_type,
                )

                if batch_valid_mask.sum() == 0:
                    loss_T_i = (cur_pred_pose_enc * 0).mean()
                    loss_R_i = (cur_pred_pose_enc * 0).mean()
                    loss_fl_i = (cur_pred_pose_enc * 0).mean()
                else:
                    loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(
                        pred_pose_encoding[batch_valid_mask].clone(),
                        gt_pose_encoding[batch_valid_mask].clone(),
                        loss_type=loss_type,
                        delta=self.delta,
                        scale=scale,
                        r_scale=scale,
                    )

                loss_T += loss_T_i * i_weight / N
                loss_R += loss_R_i * i_weight / N
                loss_fl += loss_fl_i * i_weight / N

        loss_T = loss_T / num_predictions * weight_T
        loss_R = loss_R / num_predictions * weight_R
        loss_fl = loss_fl / num_predictions * weight_fl
        loss_camera = loss_T + loss_R + loss_fl

        loss_dict = {
            # "loss_camera": loss_camera,
            "loss_T": loss_T,
            "loss_R": loss_R,
            "loss_fl": loss_fl,
        }

        return loss_camera, loss_dict
