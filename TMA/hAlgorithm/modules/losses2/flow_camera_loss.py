from math import ceil, floor

import torch
import torch.nn as nn
import torch.nn.functional as F
from .camera_loss import huber_loss

from hAlgorithm.modules.models2.external.cut3r.dust3r.utils.camera import (
    camera_to_pose_encoding,
    pose_encoding_to_camera,
)

from .base import Loss, check_and_fix_inf_nan

def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1", delta=1.0, **kwargs):
    gt_trans = gt_pose_encoding[..., :3]
    gt_quats = gt_pose_encoding[..., 3:7]
    
    pred_trans = cur_pred_pose_enc[..., :3]
    pred_quats = cur_pred_pose_enc[..., 3:7]
    pose_loss = (
        torch.norm(pred_trans - gt_trans, dim=-1).mean(),
        torch.norm(pred_quats - gt_quats, dim=-1).mean(),
        0
    )
    return pose_loss

def rel_camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1", delta=1.0, **kwargs):
    return 0, 0, 0

class Cut3rRelativeCameraLoss(Loss):
    def __init__(
        self,
        metric_space=True,
        loss_type="l1",
        gamma=0.6,
        pose_encoding_type="absT_quaR",
        weight_T=1.0,
        weight_R=1.0,
        weight_fl=0.5,
        loss_weight=1.0,
        delta=0.1,
        inv_gt=False,
        rel_pose_weight=0,
        **kwargs,
    ):
        super().__init__(loss_weight=loss_weight)
        self.metric_space=metric_space
        self.loss_type = loss_type
        self.gamma = gamma
        self.pose_encoding_type = pose_encoding_type
        self.weight_T = weight_T
        self.weight_R = weight_R
        self.weight_fl = weight_fl
        self.loss_weight = loss_weight
        self.delta = delta
        self.inv_gt = inv_gt
        self.rel_pose_weight = rel_pose_weight

    def forward(
        self,
        pose_enc : torch.Tensor,
        target_extrinsics : torch.Tensor,
        target_intrinsic : torch.Tensor,
        image_size_hw,
        valid_mask,
        scale=None,
        name=None,
        normalized_pred=False,
        **kwargs,
    ):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(pose_enc[0])), dict()

        if not isinstance(pose_enc, (list, tuple)):
            pose_enc = [pose_enc]

        if self.metric_space:
            assert scale is not None
            if not normalized_pred:
                # extrinsics[..., :3, 3] = self.normalize(extrinsics[..., :3, 3], scale[..., 0, 0])
                for i, _pose in enumerate(pose_enc):
                    pose_enc[i][..., :3] = pose_enc[i][..., :3] / scale[..., 0, 0]

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
        if self.inv_gt:
            target_extrinsics = target_extrinsics.inverse()

        gt_pose_encoding = camera_to_pose_encoding(
            target_extrinsics,
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
                rel_loss_T_i, rel_loss_R_i, rel_loss_fl_i = rel_camera_loss_single(
                    cur_pred_pose_enc[batch_valid_mask].clone(),
                    gt_pose_encoding[batch_valid_mask].clone(),
                    loss_type=loss_type,
                    delta=self.delta,
                )
                loss_T_i += rel_loss_T_i * self.rel_pose_weight
                loss_R_i += rel_loss_R_i * self.rel_pose_weight
                loss_fl_i += rel_loss_fl_i * self.rel_pose_weight
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