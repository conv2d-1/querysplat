import copy

import numpy as np
import torch
from evo.core import lie_algebra as lie
from evo.core.trajectory import PoseTrajectory3D


def mat_to_quat_trans_np(mat):
    # mat: 4*4
    trans = mat[:3, 3]
    rot = mat[:3, :3]

    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        # 迹 > 0，使用标准公式
        w = np.sqrt(1 + trace) / 2
        x = (rot[2, 1] - rot[1, 2]) / (4 * w)
        y = (rot[0, 2] - rot[2, 0]) / (4 * w)
        z = (rot[1, 0] - rot[0, 1]) / (4 * w)
    else:
        # 迹 <= 0，选择最大对角线元素计算
        if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            # R_{00} 最大
            s = 2 * np.sqrt(1 + rot[0, 0] - rot[1, 1] - rot[2, 2])
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            # R_{11} 最大
            s = 2 * np.sqrt(1 + rot[1, 1] - rot[0, 0] - rot[2, 2])
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            # R_{22} 最大
            s = 2 * np.sqrt(1 + rot[2, 2] - rot[0, 0] - rot[1, 1])
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s

    quat = np.array([w, x, y, z])
    return quat, trans


def quat_trans_to_mat_np(quat, trans):
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    quat_mat = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2],
        ]
    )
    mat = np.eye(4)
    mat[:3, :3] = quat_mat
    mat[:3, 3] = trans
    return mat


def convert2EvoTraj(poses):
    """
    poses: should be format as N*[ts, x, y, z, qw, qx, qy, qz]
    """
    xyzs = poses[:, 1:4]
    qwxyzs = poses[:, 4:]
    ts = poses[:, :1]
    return PoseTrajectory3D(xyzs, qwxyzs, ts)


def evo_align_traj(ref_poses, est_poses):
    ref_traj = convert2EvoTraj(ref_poses)
    est_traj = convert2EvoTraj(est_poses)

    est_traj_orig = copy.deepcopy(est_traj)

    R_ref_est, t_ref_est, scale = est_traj.align(ref_traj, correct_scale=True, correct_only_scale=False)

    return {"R_ref_est": R_ref_est, "t_ref_est": t_ref_est, "scale": scale, "ref_traj": ref_traj, "est_traj": est_traj, "est_traj_orig": est_traj_orig}


def convert2Matrix(poses):
    """
    poses: should be format as N*[ts, x, y, z, qw, qx, qy, qz]
    """
    P_mat = []
    for pose in poses:
        xyz = pose[1:4]
        qwxyz = pose[4:]
        mat = quat_trans_to_mat_np(qwxyz, xyz)
        P_mat.append(mat)
    P_mat = np.stack(P_mat, axis=0)
    return P_mat


def poseDictConvert2T(poses_dict):
    poses_ret = {}
    for k, item in poses_dict.items():
        xyz = item[:3]
        qwxyz = item[3:]
        T_c2w = quat_trans_to_mat_np(qwxyz, xyz)
        poses_ret[k] = T_c2w
    return poses_ret


def poseDictConvert2base(poses_dict, base_pose):
    poses_dict_ret = {}
    for k, T_c2w in poses_dict.items():
        poses_dict_ret[k] = np.linalg.inv(base_pose) @ T_c2w
    return poses_dict_ret


def poseDictInvPose(poses_dict):
    poses_dict_ret = {}
    for k, item in poses_dict.items():
        poses_dict_ret[k] = np.linalg.inv(item)
    return poses_dict_ret


def poseDictConvert2pvec(poses_dict):
    poses_ret = {}
    for k, item in poses_dict.items():
        qwxyz, trans = mat_to_quat_trans_np(item)
        poses_ret[k] = np.concatenate([trans, qwxyz])
    return poses_ret


######################## pointcloud operation ########################


def pca_align(points: torch.Tensor):
    """
    对点云做 PCA，并返回变换到 PCA 主轴坐标系的点云。

    Args:
        points: (N, 3) torch.Tensor 点云

    Returns:
        points_pca: (N, 3) torch.Tensor 变换到 PCA 主轴后的点云
        mean: (3,) 均值
        eigvecs: (3, 3) 主轴矩阵（正交矩阵，列是特征向量）
    """
    assert points.ndim == 2 and points.shape[1] == 3, "points 必须是 (N, 3)"

    # 1. 去均值
    mean = points.mean(dim=0, keepdim=True)  # (1, 3)
    centered = points - mean

    # 2. 协方差矩阵
    cov = centered.T @ centered / (points.shape[0] - 1)  # (3, 3)

    # 3. 特征分解
    eigvals, eigvecs = torch.linalg.eigh(cov)  # eigh 保证实对称矩阵
    # 按照特征值从大到小排序
    idx = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, idx]  # (3, 3)
    eigvals = eigvals[idx]

    # 4. 投影到 PCA 主轴坐标系
    points_pca = centered @ eigvecs

    return points_pca, mean.squeeze(0), eigvecs, eigvals
