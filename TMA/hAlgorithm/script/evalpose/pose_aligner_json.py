import argparse
import os
import sys

import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")  # 必须在 import pyplot 之前！

from evo.core import lie_algebra as lie

# from evo.tools import plot
# from evo.core import metrics
# from evo.core.units import Unit

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.getcwd())

try:
    from .geo_helper import evo_align_traj, mat_to_quat_trans_np, quat_trans_to_mat_np, convert2EvoTraj
    from .io_helper import load_kosmo_json, load_poses_bin_colmap, load_poses_txt_colmap, save_colmap_poses, write_1ddict2csv, write_2ddict2csv, write_tum_txt
    from .metrics_helper import compute_auc_metrics, compute_evo_metrics
    from .vis_helper import vis_trajs
    from .colmap_util import read_points3D_bin
except:
    from geo_helper import evo_align_traj, mat_to_quat_trans_np, quat_trans_to_mat_np, convert2EvoTraj
    from io_helper import load_kosmo_json, load_poses_bin_colmap, load_poses_txt_colmap, save_colmap_poses, write_1ddict2csv, write_2ddict2csv, write_tum_txt
    from metrics_helper import compute_auc_metrics, compute_evo_metrics
    from vis_helper import vis_trajs
    from colmap_util import read_points3D_bin


def get_split_method(key):
    # idx = key.split('.')[0].split('_')[0]
    # idx = key.split('.')[0][-4:]
    idx = key.split(".")[0]
    if idx.isdigit():
        return idx
    elif idx[-4:].isdigit():
        return idx[-4:]
    elif idx.split("_")[1].isdigit():
        return idx.split("_")[1]
    else:
        return idx


def get_poses(pose_file, pose_key, scene, view_id):
    ext = os.path.splitext(pose_file)[1]
    if ext == ".txt":
        poses_orig, _ = load_poses_txt_colmap(pose_file, None)
    elif ext == ".bin":
        poses_orig, _ = load_poses_bin_colmap(pose_file, None)
    elif ext == ".json":
        # assert scene is not None, "scene is required for json file"
        assert view_id is not None, "view_id is required for json file"
        poses_orig, _ = load_kosmo_json(pose_file, scene=scene, view_id=view_id, pose_key=pose_key)
    else:
        raise ValueError(f"Unsupported file extension: {ext} in {pose_file}")
    poses_orig = {k: poses_orig[k] for k in sorted(poses_orig)}
    return poses_orig


def convert_dict2list(pose_dict, get_split_method):
    pose_dict_tmp = {}
    for k in pose_dict:
        idx = int(get_split_method(k))
        pose_dict_tmp[idx] = pose_dict[k]
    poses = np.array([[k] + v for k, v in pose_dict_tmp.items()])
    return poses


def convert_colmap2tum(poses):
    tum_format = []
    for idx in range(len(poses)):
        pose = poses[idx]
        tum_format.append([pose[0], pose[1], pose[2], pose[3], pose[5], pose[6], pose[7], pose[4]])
    return tum_format


def eval_run(gt_pose_file, est_pose_file, scene=None, view_id=None, est_pose_key=None, gt_pose_key=None, output_dir="pose_eval_output", gt_pcd=None, est_pcd=None, vis=False):
    ## read poses
    est_poses_orig = get_poses(est_pose_file, est_pose_key, scene, view_id)
    est_poses_orig_list = convert_dict2list(est_poses_orig, get_split_method)

    gt_poses_orig = get_poses(gt_pose_file, gt_pose_key, scene, view_id)

    if view_id is not None:
        est_poses_orig = {name:val for name, val in est_poses_orig.items() if name.split("/")[0] ==f"camera_{view_id}"}
        gt_poses_orig = {name:val for name, val in gt_poses_orig.items() if name.split("/")[0] == f"camera_{view_id}"}

    remained_gt_poses = {}
    remained_est_poses = {}
    remained_names = []
    idx = 0
    for k in sorted(gt_poses_orig):
        if k in est_poses_orig:
            remained_gt_poses[idx] = gt_poses_orig[k]
            remained_est_poses[idx] = est_poses_orig[k]
            remained_names.append(k)
            idx += 1

    gt_poses = remained_gt_poses
    gt_poses = np.array([[k] + v for k, v in gt_poses.items()])
    est_poses = remained_est_poses
    est_poses = np.array([[k] + v for k, v in est_poses.items()])

    # gt_poses = gt_poses[:4]
    # est_poses = est_poses[:4]
    print(f"gt_poses:{gt_poses.shape}, est_poses:{est_poses.shape}")
    ## 原始数据写入tum txt
    output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    gt_tum_format = convert_colmap2tum(gt_poses)
    write_tum_txt(gt_tum_format, f"{output_dir}/orig_gt_poses_tum.txt")

    est_tum_format = convert_colmap2tum(est_poses_orig_list)
    write_tum_txt(est_tum_format, f"{output_dir}/orig_est_poses_tum.txt")

    ## align poses
    aligned_res = evo_align_traj(gt_poses, est_poses)
    t_ref_est = aligned_res["t_ref_est"]
    R_ref_est = aligned_res["R_ref_est"]
    scale = aligned_res["scale"]
    ref_traj = aligned_res["ref_traj"]
    est_traj = aligned_res["est_traj"]
    est_traj_orig = aligned_res["est_traj_orig"]

    ## compute metrics
    # ref_traj = convert2EvoTraj(gt_poses)
    # est_traj = convert2EvoTraj(est_poses)
    evo_metrics = compute_evo_metrics(ref_traj=ref_traj, est_traj=est_traj)
    for key, val in evo_metrics.items():
        if "APE" in key:
            print(f"{key}, " + " ".join([f"{k}:{v:.3f}" for k, v in val.items()]))

    ## write to csv
    write_2ddict2csv(evo_metrics, f"{output_dir}/evo_metrics.csv", key_name="EVO")

    ## vis poses
    # ref_xyzs = gt_poses[:, 1:4]
    # ref_xyzs_max = ref_xyzs.max()
    # ref_xyzs_min = ref_xyzs.min()

    traj_dict = {
        # "estimate (not aligned)": est_traj_orig,
        "estimate (aligned)": est_traj,
        "reference": ref_traj,
    }
    vis_trajs(traj_dict, xyz_ranges=None, save_path=f"{output_dir}/aligned_poses.png", vis=vis)

    ### get aligned poses ####
    T_ref_est = lie.se3(R_ref_est, t_ref_est)
    aligned_poses_c2w = []
    for item in est_poses:
        ts = item[0]
        xyz = item[1:4]
        qwxyz = item[4:]
        T_c2w = quat_trans_to_mat_np(qwxyz, xyz)
        scaled_T = lie.se3(T_c2w[:3, :3], scale * T_c2w[:3, 3])
        pose_est2ref_w = T_ref_est @ scaled_T
        quat_aligned, xyz_aligned = mat_to_quat_trans_np(pose_est2ref_w)
        aligned_poses_c2w.append([ts, xyz_aligned[0], xyz_aligned[1], xyz_aligned[2], quat_aligned[0], quat_aligned[1], quat_aligned[2], quat_aligned[3]])

    ## compute auc
    auc_metrics = compute_auc_metrics(gt_poses, aligned_poses_c2w)
    # print(auc_metrics)
    print(" ".join([f"{k}:{v:.3f}" for k, v in auc_metrics.items()]))
    
    ## write to csv
    write_1ddict2csv(auc_metrics, f"{output_dir}/auc_metrics.csv", key_name="AUC")

    ## 写入 aligned poses tum
    aligned_est_tum_format = convert_colmap2tum(aligned_poses_c2w)
    write_tum_txt(aligned_est_tum_format, f"{output_dir}/aligned_est_poses_tum.txt")

    ## 保存缩放后的gt和est点云和轨迹
    aligned_gt_w2c = {}
    T_est_ref = np.linalg.inv(lie.se3(R_ref_est, t_ref_est))
    for file_name, pvec_c2w in gt_poses_orig.items():
        xyz_c2w = pvec_c2w[:3]
        quat_c2w = pvec_c2w[3:]
        T_c2w = quat_trans_to_mat_np(quat_c2w, xyz_c2w)
        T_ref_c2est_w = T_est_ref @ T_c2w
        T_c2w_aligned = lie.se3(T_ref_c2est_w[:3, :3], T_ref_c2est_w[:3, 3] / scale)
        T_w2c_aligned = np.linalg.inv(T_c2w_aligned)
        quat_w2c_aligned, xyz_w2c_aligned = mat_to_quat_trans_np(T_w2c_aligned)
        aligned_gt_w2c[file_name] = np.array(xyz_w2c_aligned.tolist() + quat_w2c_aligned.tolist())
    save_colmap_poses(aligned_gt_w2c, f"{output_dir}/aligned_gt_colmap.txt")

    if gt_pcd is not None:
        gt_pcd = o3d.io.read_point_cloud(gt_pcd)
        gt_xyz = np.asarray(gt_pcd.points)
        gt_rgb = np.asarray(gt_pcd.colors)
        aligned_gt_xyz = (gt_xyz @ T_est_ref[:3, :3].T + T_est_ref[:3, 3]) / scale
        aligned_gt_pcd = o3d.geometry.PointCloud()
        aligned_gt_pcd.points = o3d.utility.Vector3dVector(aligned_gt_xyz)
        aligned_gt_pcd.colors = o3d.utility.Vector3dVector(gt_rgb)
        o3d.io.write_point_cloud(f"{output_dir}/aligned_pcd_gt.ply", aligned_gt_pcd)

    T_ref_est = lie.se3(R_ref_est, t_ref_est)
    aligned_est_w2c = {}
    for file_name, pvec_c2w in est_poses_orig.items():
        xyz_c2w = pvec_c2w[:3]
        quat_c2w = pvec_c2w[3:]
        T_c2w = quat_trans_to_mat_np(quat_c2w, xyz_c2w)
        aligned_T = lie.se3(T_c2w[:3, :3], scale * T_c2w[:3, 3])
        T_c2w_aligned = T_ref_est @ aligned_T
        T_w2c_aligned = np.linalg.inv(T_c2w_aligned)
        quat_w2c_aligned, xyz_w2c_aligned = mat_to_quat_trans_np(T_w2c_aligned)
        aligned_est_w2c[file_name] = np.array(xyz_w2c_aligned.tolist() + quat_w2c_aligned.tolist())
    save_colmap_poses(aligned_est_w2c, f"{output_dir}/aligned_est_colmap.txt")
    
    gt_w2c = {}
    for file_name, pvec_c2w in gt_poses_orig.items():
        xyz_c2w = pvec_c2w[:3]
        quat_c2w = pvec_c2w[3:]
        T_c2w = quat_trans_to_mat_np(quat_c2w, xyz_c2w)
        quat_w2c, xyz_w2c = mat_to_quat_trans_np(np.linalg.inv(T_c2w))
        gt_w2c[file_name] = np.array(xyz_w2c.tolist() + quat_w2c.tolist())
    save_colmap_poses(gt_w2c, f"{output_dir}/gt_colmap.txt")

    if est_pcd:
        if est_pcd.endswith(".ply"):
            est_pcd = o3d.io.read_point_cloud(est_pcd)
            est_xyz = np.asarray(est_pcd.points)
            est_rgb = np.asarray(est_pcd.colors)
        elif est_pcd.endswith(".bin"):
            est_xyz, est_rgb, ids, imgs_xyzs, imgs_rgbs = read_points3D_bin(est_pcd)
            est_rgb = est_rgb / 255.0
            # breakpoint()
        aligned_est_xyz = (est_xyz * scale) @ T_ref_est[:3, :3].T + T_ref_est[:3, 3]
        aligned_est_pcd = o3d.geometry.PointCloud()
        aligned_est_pcd.points = o3d.utility.Vector3dVector(aligned_est_xyz)
        aligned_est_pcd.colors = o3d.utility.Vector3dVector(est_rgb)
        o3d.io.write_point_cloud(f"{output_dir}/aligned_pcd_est.ply", aligned_est_pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_pose_file", "--gt", type=str, required=True, help="gt pose file")
    parser.add_argument("--est_pose_file", "--pred", type=str, required=True, help="estimated pose file")
    parser.add_argument("--scene", type=str, default=None, help="scene name")
    parser.add_argument("--view_id", type=int, default=None, help="view_id")
    parser.add_argument("--est_pose_key", type=str, default=None, help="est pose_key")
    parser.add_argument("--gt_pose_key", type=str, default=None, help="gt pose_key")
    parser.add_argument("--gt_pcd", type=str, default=None, help="gt pcd file")
    parser.add_argument("--est_pcd", type=str, default=None, help="est pcd file")
    parser.add_argument("--output_dir", type=str, default="pose_eval_output", help="result dir")
    parser.add_argument("--vis", action="store_true", default=False, help="visualize the trajs")

    args = parser.parse_args()

    eval_run(
        gt_pose_file=args.gt_pose_file,
        est_pose_file=args.est_pose_file,
        scene=args.scene,
        view_id=args.view_id,
        est_pose_key=args.est_pose_key,
        gt_pose_key=args.gt_pose_key,
        output_dir=args.output_dir,
        gt_pcd=args.gt_pcd,
        est_pcd=args.est_pcd,
        vis=args.vis,
    )
