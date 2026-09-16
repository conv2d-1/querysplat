import logging
import numpy as np
import poselib
import torch
import tqdm
import kornia

# Disable scientific notation
np.set_printoptions(suppress=True)


class MatchEvalMetrics:
    def __init__(self, metrics=None, ransac_threshold=2.5, matches_name="matches", min_matches=50, use_cov=False):
        self.metrics = ["auc@1", "auc@5", "auc@10", "mAcc@1", "mAcc@5", "mAcc@10", "R_err", "t_err", "inliers", "epipolar_distance"]
        self.ransac_threshold = ransac_threshold
        self.matches_name = matches_name
        self.min_matches = min_matches
        self.use_cov = use_cov
        if self.use_cov:
            self.metrics.extend(["auc@1_cov", "auc@5_cov", "auc@10_cov", "R_err_cov", "t_err_cov", "inliers_cov"])

    def eval_single_data(self, inputs, output, eval_idx):
        if not hasattr(output, self.matches_name):
            return dict()

        if output.intrinsics_image0 is None or output.intrinsics_image1 is None:
            return dict()

        if output.extrinsics_image0 is None or output.extrinsics_image1 is None:
            return dict()

        if output.matches_gt is not None and output.matches_gt.shape[0] < self.min_matches:
            return dict()

        fisheye = False
        if inputs is not None:
            camera_type = inputs.get("meta_data", {}).get("camera_type", ["PINHOLE"])[0]
            if isinstance(camera_type, (list, tuple)):
                camera_type = camera_type[0]
            fisheye = (camera_type == "FISHEYE_EQUIDISTANT")

        # NOTE: Xfeat
        # read and compute relative poses
        T0 = output.extrinsics_image0
        T1 = output.extrinsics_image1
        T_0to1 = np.matmul(T1, np.linalg.inv(T0))

        K0 = output.intrinsics_image0
        K1 = output.intrinsics_image1
        
        matches = getattr(output, self.matches_name)
        src_pts, dst_pts = matches[:, :2], matches[:, 2:]

        if self.use_cov:
            import copy
            pair_with_cov = {"K0": K0, "K1": K1, "T_0to1": T_0to1, "pts0": src_pts, "pts1": dst_pts, "ransac_thr": self.ransac_threshold}
            if output.cov0 is not None:
                pair_with_cov["cov0"] = output.cov0
            if output.cov1 is not None:
                pair_with_cov["cov1"] = output.cov1
            compute_pose_error(pair_with_cov, self.use_cov, fisheye=fisheye)
            results_dict_cov = compute_maa([pair_with_cov], thresholds=[1, 5, 10])
            results_dict_cov["R_err"] = pair_with_cov["R_err"]
            results_dict_cov["t_err"] = pair_with_cov["t_err"]
            results_dict_cov["inliers"] = np.array(pair_with_cov["inliers"]).sum()

        pair = {"K0": K0, "K1": K1, "T_0to1": T_0to1, "pts0": src_pts, "pts1": dst_pts, "ransac_thr": self.ransac_threshold}
        compute_pose_error(pair, fisheye=fisheye)

        results_dict = compute_maa([pair], thresholds=[1, 5, 10])
        results_dict["R_err"] = pair["R_err"]
        results_dict["t_err"] = pair["t_err"]
        results_dict["inliers"] = np.array(pair["inliers"]).sum()

        if np.isnan(results_dict["t_err"]):
            logging.error(f"MatchEvalMetrics, pts {len(src_pts)}, {results_dict}")
            return dict()

        # NOTE: 3RGS
        intrinsics_i_44_all = np.eye(4)
        intrinsics_i_44_all[:3, :3] = K0
        intrinsics_j_44_all = np.eye(4)
        intrinsics_j_44_all[:3, :3] = K1

        P_i = intrinsics_i_44_all @ output.extrinsics_image0
        P_j = intrinsics_j_44_all @ output.extrinsics_image1

        Fm = kornia.geometry.epipolar.fundamental_from_projections(torch.from_numpy(P_i[None, :3]).float(), torch.from_numpy(P_j[None, :3]).float())
        epipolar_distance = kornia.geometry.symmetrical_epipolar_distance(torch.from_numpy(src_pts[None]).float(), torch.from_numpy(dst_pts[None]).float(), Fm, squared=False, eps=1e-08)
        results_dict["epipolar_distance"] = epipolar_distance.mean()
        if self.use_cov:
            for key, val in results_dict_cov.items():
                results_dict[key+"_cov"] = val
        return results_dict

    def eval_mf_data(self, inputs, output):
        results_dict = dict()
        valid_result = 0
        for i, out in enumerate(output):
            if out is None:
                continue
            result = self.eval_single_data(inputs, out, eval_idx=i)
            for k, v in result.items():
                results_dict[k] = results_dict.get(k, 0) + v
            valid_result += 1
        if valid_result == 0:
            raise ValueError("Valid Result is zero!")
        results_dict = {k: v / valid_result for k, v in results_dict.items()}
        return results_dict

    def __call__(self, inputs, output):
        if isinstance(output, list):
            return self.eval_mf_data(inputs, output)
        else:
            return self.eval_single_data(inputs, output, eval_idx=None)


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def intrinsics_to_camera(K):
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }


def intrinsics_to_camera_fisheye(K):
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "OPENCV_FISHEYE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py, 0, 0, 0, 0],
    }


def estimate_pose_poselib(kpts0, kpts1, K0, K1, thresh, conf=0.99999, fisheye=False):
    camera_fn = intrinsics_to_camera_fisheye if fisheye else intrinsics_to_camera
    M, info = poselib.estimate_relative_pose(
        kpts0,
        kpts1,
        camera_fn(K0),
        camera_fn(K1),
        {"max_epipolar_error": thresh, "success_prob": conf, "min_iterations": 20, "max_iterations": 1_000},
    )

    R, t, inl = M.R, M.t, info["inliers"]
    inl = np.array(inl)
    ret = (R, t, inl)

    return ret, (kpts0, kpts1)


def estimate_pose_poselib_cov(kpts0, kpts1, cov0, cov1, K0, K1, thresh, conf=0.99999, fisheye=False):
    camera_fn = intrinsics_to_camera_fisheye if fisheye else intrinsics_to_camera
    M, info = poselib.estimate_relative_pose_cov(
        kpts0,
        kpts1,
        cov0,
        cov1,
        camera_fn(K0),
        camera_fn(K1),
        {"max_epipolar_error": thresh, "success_prob": conf, "min_iterations": 20, "max_iterations": 1_000},
    )

    R, t, inl = M.R, M.t, info["inliers"]
    inl = np.array(inl)
    ret = (R, t, inl)

    return ret, (kpts0, kpts1)

def tensor2bgr(t):
    return (t.cpu()[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def compute_pose_error(pair, use_cov=False, fisheye=False):
    """
    Input:
        pair (dict):{
            "pts0": ndrray(N,2)
            "pts1": ndrray(N,2)
            "K0": ndrray(3,3)
            "K1": ndrray(3,3)
            "T_0to1": ndrray(4,4)

        }
    Update:
        pair (dict):{
            "R_err" List[float]: [N]
            "t_err" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = 1.0 if "ransac_thr" not in pair else pair["ransac_thr"]
    conf = 0.99999
    pair.update({"R_err": np.inf, "t_err": np.inf, "inliers": []})

    pts0 = pair["pts0"]
    pts1 = pair["pts1"]
    K0 = pair["K0"]
    K1 = pair["K1"]
    T_0to1 = pair["T_0to1"]

    if use_cov and "cov0" in pair and "cov1" in pair:
        ret, corrs = estimate_pose_poselib_cov(pts0, pts1, pair["cov0"], pair["cov1"], K0, K1, pixel_thr, conf=conf, fisheye=fisheye)
    else:
        ret, corrs = estimate_pose_poselib(pts0, pts1, K0, K1, pixel_thr, conf=conf, fisheye=fisheye)

    if ret is not None:
        R, t, inliers = ret

        t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)
        # breakpoint()

        pair["R_err"] = R_err
        pair["t_err"] = t_err
        pair["inliers"] = inliers


def error_auc(errors, thresholds=[5, 10, 20]):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []

    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f"auc@{t}": auc for t, auc in zip(thresholds, aucs)}


def compute_maa(pairs, thresholds=[5, 10, 20]):
    # print("auc / mAcc on %d pairs" % (len(pairs)))
    outputs = dict()
    errors = []

    for p in pairs:
        et = p["t_err"]
        er = p["R_err"]
        errors.append(max(et, er))

    d_err_auc = error_auc(errors, thresholds=thresholds)

    for k, v in d_err_auc.items():
        # print(k, ": ", "%.1f" % (v * 100))
        outputs[k] = v * 100

    errors = np.array(errors)

    for t in thresholds:
        acc = (errors <= t).sum() / len(errors)
        # print("mAcc@%d: %.1f " % (t, acc * 100))
        outputs[f"mAcc@{t}"] = acc * 100

    return outputs


@torch.inference_mode()
def run_pose_benchmark(matcher_fn, loader, ransac_thr=2.5):
    """
    Run relative pose estimation benchmark using a specified matcher function and data loader.

    Parameters
    ----------
    matcher_fn : callable
        The matching function to be evaluated for pose estimation. It should accept two np.array RGB images (H,W,3)
        and return mkpts_0, mkpts_1 which are np.array(N,2) matching coordinates.

    loader : iterable
        Data loader that provides batches of data. Each batch should contain two images, along
        with their groundtruth camera poses.

    ransac_thr : float, optional, default=2.5
        The RANSAC threshold for considering a point as an inlier in pixels.
    """

    pairs = []
    cnt = 0
    for d in tqdm.tqdm(loader):
        d_error = {}
        src_pts, dst_pts = matcher_fn(tensor2bgr(d["image0"]), tensor2bgr(d["image1"]))

        # delete images to avoid OOM, happens in low mem machines
        del d["image0"]
        del d["image1"]

        # rescale kpts
        src_pts = src_pts * d["scale0"].numpy()
        dst_pts = dst_pts * d["scale1"].numpy()
        d.update({"pts0": src_pts, "pts1": dst_pts, "ransac_thr": ransac_thr})
        compute_pose_error(d)
        pairs.append(d)
        cnt += 1

    compute_maa(pairs)

