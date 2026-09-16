# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch

try:
    from pytorch3d.renderer.cameras import PerspectiveCameras
except:
    from .pytorch3d_custom.cameras import PerspectiveCameras

from pytorch3d.transforms import so3_relative_angle

from hAlgorithm.modules.models.vggt.utils.pose_enc import intri_to_fov


class PoseEvalMetricsV2:
    def __init__(self, skip_static_camera: bool = False, **kwargs):
        """
        Args:
            skip_static_camera: When True, translation-related metrics (tangle, auc_30,
                auc_1) are skipped for scenes where all GT camera positions are identical,
                since relative translations would be zero and the angle becomes undefined.
                Defaults to False to preserve the original evaluation behaviour.
        """
        self.skip_static_camera = skip_static_camera
        self.metrics = [
            "rangle",
            "tangle",
            "tscale",
            "terr",
            "auc_30",
            "auc_1",
            "fov_l1",
        ]

    def metrics_func(
        self,
        intrinsics_gts: list,
        extrinsics_gts: list,
        intrinsics_preds: list,
        extrinsics_preds: list,
    ):
        assert (
            len(intrinsics_gts)
            == len(intrinsics_preds)
            == len(extrinsics_gts)
            == len(extrinsics_preds)
        )
        nums = len(intrinsics_gts)

        focal_length_gt = torch.tensor([[m[0][0], m[1][1]] for m in intrinsics_gts])
        rotation_gt = torch.stack([m[:3, :3] for m in extrinsics_gts], dim=0)
        translation_gt = torch.stack([m[:3, 3] for m in extrinsics_gts], dim=0)
        gt_cameras = PerspectiveCameras(
            focal_length=focal_length_gt,
            R=rotation_gt,
            T=translation_gt,
        )

        focal_length_pred = torch.tensor([[m[0][0], m[1][1]] for m in intrinsics_preds])
        rotation_pred = torch.stack([m[:3, :3] for m in extrinsics_preds], dim=0)
        translation_pred = torch.stack([m[:3, 3] for m in extrinsics_preds], dim=0)
        pred_cameras = PerspectiveCameras(
            focal_length=focal_length_pred,
            R=rotation_pred,
            T=translation_pred,
        )

        # compute metrics
        try:
            rel_rangle_deg, rel_tangle_deg, rel_tangle_scale, rel_tangle_error = camera_to_rel_deg(
                pred_cameras,
                gt_cameras,
                device="cpu",
                batch_size=1,
                with_scale=True,
            )
        except Exception as e:
            logging.error(e)
            logging.error("PoseEvalMetricsV2, error in camera_to_rel_deg!!!")
            return dict()

        # focal_length_diff = (focal_length_gt - focal_length_pred).abs().sum() / (2.0 * nums)
        # rW, rH = intrinsics_gts[0][0][2] * 2, intrinsics_gts[0][1][2] * 2
        W, H = intrinsics_preds[0][0][2] * 2, intrinsics_preds[0][1][2] * 2
        # fov_w_gt, fov_h_gt = intri_to_fov(torch.tensor(intrinsics_gts).float(), image_size_hw=(rH, rW))
        fov_w_gt, fov_h_gt = intri_to_fov(
            torch.tensor(intrinsics_gts).float(), image_size_hw=(H, W)
        )
        fov_w, fov_h = intri_to_fov(torch.tensor(intrinsics_preds).float(), image_size_hw=(H, W))
        fov_l1 = (torch.cat([fov_w_gt, fov_h_gt]) - torch.cat([fov_w, fov_h])).abs().mean().item()

        results_dict = dict(
            rangle=rel_rangle_deg.mean().item(),
            fov_l1=fov_l1,
        )

        # Determine whether to include translation-dependent metrics.
        # When skip_static_camera is enabled, scenes where all GT camera positions are
        # identical are excluded because relative translations are zero and tangle is
        # undefined in that case.
        include_translation_metrics = True
        if self.skip_static_camera:
            static_camera = torch.allclose(
                translation_gt, translation_gt[:1].expand_as(translation_gt), atol=1e-6
            )
            if static_camera:
                include_translation_metrics = False

        if include_translation_metrics:
            # also called mAA in some literature
            Auc_30 = calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=30)
            Auc_1 = calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=1)
            results_dict["tangle"] = rel_tangle_deg.mean().item()
            results_dict["auc_30"] = Auc_30.item()
            results_dict["auc_1"] = Auc_1.item()

        return results_dict

    def eval_single_data(self, output):
        return dict()

    def eval_mf_data(self, outputs):
        if len(outputs) == 1:
            return dict()
        if outputs[0].intrinsics is None or outputs[0].extrinsics is None:
            return dict()

        intrinsics_preds = [getattr(output, "intrinsics_pred", None) for output in outputs]
        extrinsics_preds = [getattr(output, "extrinsics_pred", None) for output in outputs]

        if extrinsics_preds[0] is None:
            return dict()

        intrinsics_gts = [output.intrinsics.tolist() for output in outputs]
        extrinsics_gts = [torch.from_numpy(output.extrinsics) for output in outputs]

        if intrinsics_preds[0] is None:
            intrinsics_preds = intrinsics_gts
        else:
            intrinsics_preds = [intrinsics_pred.tolist() for intrinsics_pred in intrinsics_preds]

        extrinsics_preds = [
            torch.from_numpy(extrinsics_pred) for extrinsics_pred in extrinsics_preds
        ]

        results_dict = self.metrics_func(
            intrinsics_gts, extrinsics_gts, intrinsics_preds, extrinsics_preds
        )

        return results_dict

    def __call__(self, inputs, outputs):
        if isinstance(outputs, list):
            return self.eval_mf_data(outputs)
        else:
            return self.eval_single_data(outputs)


def camera_to_rel_deg(pred_cameras, gt_cameras, device, batch_size, with_scale=False):
    """
    Calculate relative rotation and translation angles between predicted and ground truth cameras.

    Args:
    - pred_cameras: Predicted camera.
    - gt_cameras: Ground truth camera.
    - accelerator: The device for moving tensors to GPU or others.
    - batch_size: Number of data samples in one batch.

    Returns:
    - rel_rotation_angle_deg, rel_translation_angle_deg: Relative rotation and translation angles in degrees.
    """

    with torch.no_grad():
        # Convert cameras to 4x4 SE3 transformation matrices
        gt_se3 = gt_cameras.get_world_to_view_transform().get_matrix()
        pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix()

        # Generate pairwise indices to compute relative poses
        pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, gt_se3.shape[0] // batch_size)
        pair_idx_i1 = pair_idx_i1.to(device)

        # Compute relative camera poses between pairs
        # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
        # This is possible because of SE3
        relative_pose_gt = closed_form_inverse(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])
        relative_pose_pred = closed_form_inverse(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])

        # Compute the difference in rotation and translation
        # between the ground truth and predicted relative camera poses
        rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])

        if with_scale:
            rel_tangle_deg, rel_tangle_scale, rel_tangle_error = translation(
                relative_pose_gt[:, 3, :3], relative_pose_pred[:, 3, :3]
            )
        else:
            rel_tangle_deg = translation_angle(
                relative_pose_gt[:, 3, :3], relative_pose_pred[:, 3, :3]
            )
            rel_tangle_scale = rel_tangle_error = None

    return rel_rangle_deg, rel_tangle_deg, rel_tangle_scale, rel_tangle_error


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays.

    :param r_error: numpy array representing R error values (Degree).
    :param t_error: numpy array representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error arrays along a new axis
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)

    # Compute the maximum error value for each pair
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram))


def calculate_auc(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

    :param r_error: torch.Tensor representing R error values (Degree).
    :param t_error: torch.Tensor representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error tensors along a new axis
    error_matrix = torch.stack((r_error, t_error), dim=1)

    # Compute the maximum error value for each pair
    max_errors, _ = torch.max(error_matrix, dim=1)

    # Define histogram bins
    # bins = torch.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram = torch.histc(max_errors, bins=max_threshold + 1, min=0, max=max_threshold)

    # Normalize the histogram
    num_pairs = float(max_errors.size(0))
    normalized_histogram = histogram / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return torch.cumsum(normalized_histogram, dim=0).mean()


def batched_all_pairs(B, N):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]

    return i1, i2


def closed_form_inverse(se3):
    """
    Computes the inverse of each 4x4 SE3 matrix in the batch.

    Args:
    - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

    Returns:
    - Tensor: Nx4x4 tensor of inverted SE3 matrices.
    """
    R = se3[:, :3, :3]
    T = se3[:, 3:, :3]

    # Compute the transpose of the rotation
    R_transposed = R.transpose(1, 2)

    # Compute the left part of the inverse transformation
    left_bottom = -T.bmm(R_transposed)
    left_combined = torch.cat((R_transposed, left_bottom), dim=1)

    # Keep the right-most column as it is
    right_col = se3[:, :, 3:].detach().clone()
    inverted_matrix = torch.cat((left_combined, right_col), dim=-1)

    return inverted_matrix


def rotation_angle(rot_gt, rot_pred, batch_size=None):
    # rot_gt, rot_pred (B, 3, 3)
    rel_angle_cos = so3_relative_angle(rot_gt, rot_pred, eps=1e-4)
    rel_rangle_deg = rel_angle_cos * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def compute_ARE(rotation1, rotation2):
    if isinstance(rotation1, torch.Tensor):
        rotation1 = rotation1.cpu().detach().numpy()
    if isinstance(rotation2, torch.Tensor):
        rotation2 = rotation2.cpu().detach().numpy()

    R_rel = np.einsum("Bij,Bjk ->Bik", rotation1.transpose(0, 2, 1), rotation2)
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    error = theta * 180 / np.pi
    return np.minimum(error, np.abs(180 - error))


def translation(tvec_gt, tvec_pred, batch_size=None):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg, rel_tangle_scale, rel_tangle_error = compare_translation(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)
        rel_tangle_scale = rel_tangle_scale.reshape(batch_size, -1)

    return rel_tangle_deg, rel_tangle_scale, rel_tangle_error


def compare_translation(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Compare two translation vectors in terms of both direction and scale.

    Parameters:
    - t_gt: Ground truth translation vector(s).
    - t: Predicted translation vector(s).
    - eps: Small epsilon value to avoid division by zero or numerical instability.
    - default_err: Default error value for handling NaN or Inf results.

    Returns:
    - angle_error: Angle difference between the normalized vectors.
    - scale_error: Scale difference (ratio) between the vectors.
    """
    # Compute norms
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)

    # Normalize vectors for direction comparison
    t_normalized = t / (t_norm + eps)
    t_gt_normalized = t_gt / (t_gt_norm + eps)

    # Compute cosine similarity (direction difference)
    cos_theta = torch.sum(t_normalized * t_gt_normalized, dim=1)
    # Clamp to avoid numerical instability
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)

    # Compute angle error
    angle_error = torch.acos(cos_theta)

    # Handle NaN or Inf values
    angle_error[torch.isnan(angle_error) | torch.isinf(angle_error)] = default_err

    # Compute scale error (ratio of lengths)
    scale_error = torch.abs(t_norm - t_gt_norm) / (t_gt_norm + eps)

    error = torch.norm(t - t_gt, p=2, dim=1, keepdim=True) / (t_gt_norm + eps)

    # Handle NaN or Inf values
    scale_error[torch.isnan(scale_error) | torch.isinf(scale_error)] = default_err

    return angle_error, scale_error, error
