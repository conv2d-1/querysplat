import numpy as np
import torch
from evo.core import metrics
from evo.core.units import Unit
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import so3_relative_angle

try:
    from .geo_helper import convert2Matrix
except:
    from geo_helper import convert2Matrix

######################### evo related ############################
def compute_evo_metrics(ref_traj, est_traj):
    metrics_results = {}
    # APE
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((ref_traj, est_traj))
    ape_stats = ape_metric.get_all_statistics()
    # print("Translation APE: \n",ape_stats)
    metrics_results["Translation APE"] = ape_stats

    ape_metric = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    ape_metric.process_data((ref_traj, est_traj))
    ape_stats = ape_metric.get_all_statistics()
    # print("Rotation APE: \n",ape_stats)
    metrics_results["Rotation APE Deg"] = ape_stats

    import matplotlib.pyplot as plt

    # 假设你已经有了 ape_metric.error，它是一个一维数组，表示每一帧的旋转 APE（单位：度）
    errors = ape_metric.error  # 这里是角度误差，不是平方误差
    # errors_fix = []
    # for i in range(0, len(errors), 50):
    #     if i-1 > 0:
    #         errors_fix.append(errors[i-1])
    #     errors_fix.append(errors[i])
    #     if i < len(errors):
    #         errors_fix.append(errors[i+1])
    # errors = errors_fix
    # errors = errors[49]
    # mask = errors < 1.5
    # errors *= mask
    # print(np.where(~mask))
    # 创建 x 轴（帧索引）
    # x = range(len(errors))

    # 绘制折线图
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, errors, label="Rotation APE (deg)", color='blue')
    # plt.xlabel("Frame Index")
    # plt.ylabel("Rotation APE (Degrees)")
    # plt.title("Absolute Pose Error (Rotation) Over Time")
    # plt.grid(True)
    # plt.legend()

    # # 保存图像（你可以自定义路径和文件名）
    # plt.savefig("rotation_ape_deg.png", dpi=300, bbox_inches='tight')

    # breakpoint()

    # 如果你还想在调试时显示（可选，但在非交互环境可能报错）
    # plt.show()

    # 如果你确实想要平方误差（比如为了观察误差幅度），可以这样：
    # squared_errors = errors ** 2
    # plt.plot(x, squared_errors, label="Squared Rotation APE", color='red')

    # RPE
    rpe_metric = metrics.RPE(pose_relation=metrics.PoseRelation.translation_part, delta=1, delta_unit=Unit.frames, all_pairs=False)
    rpe_metric.process_data((ref_traj, est_traj))
    rpe_stats = rpe_metric.get_all_statistics()
    # print("Translation RPE: \n", rpe_stats)
    metrics_results["Translation RPE"] = rpe_stats

    rpe_metric = metrics.RPE(pose_relation=metrics.PoseRelation.rotation_angle_deg, delta=1, delta_unit=Unit.frames, all_pairs=False)
    rpe_metric.process_data((ref_traj, est_traj))
    rpe_stats = rpe_metric.get_all_statistics()
    # print("Rotation RPE: \n", rpe_stats)
    metrics_results["Rotation RPE Deg"] = rpe_stats

    return metrics_results


######################### auc related ############################
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
            rel_tangle_deg, rel_tangle_scale, rel_tangle_error = translation(relative_pose_gt[:, 3, :3], relative_pose_pred[:, 3, :3])
        else:
            rel_tangle_deg = translation_angle(relative_pose_gt[:, 3, :3], relative_pose_pred[:, 3, :3])
            rel_tangle_scale = rel_tangle_error = None

    return rel_rangle_deg, rel_tangle_deg, rel_tangle_scale, rel_tangle_error


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


def compute_auc_metrics(ref_poses, est_poses):
    ref_pose_mat = convert2Matrix(ref_poses)
    est_pose_mat = convert2Matrix(est_poses)

    ref_cameras = PerspectiveCameras(
        R=ref_pose_mat[:, :3, :3],
        T=ref_pose_mat[:, :3, 3],
    )

    est_cameras = PerspectiveCameras(
        R=est_pose_mat[:, :3, :3],
        T=est_pose_mat[:, :3, 3],
    )

    rel_rangle_deg, rel_tangle_deg, rel_tangle_scale, rel_tangle_error = camera_to_rel_deg(
        est_cameras,
        ref_cameras,
        device="cpu",
        batch_size=1,
        with_scale=True,
    )

    Auc_30 = calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=30)
    Auc_1 = calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=1)

    return {"Auc_30": float(Auc_30), "Auc_1": float(Auc_1)}
