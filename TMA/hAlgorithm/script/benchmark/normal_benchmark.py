
import numpy as np
import os
import sys
sys.path.append(os.getcwd())

from hAlgorithm.modules.metrics.pointmap_eval_metrics import pointmap2normal
from hAlgorithm.modules.utils.normal import pointmap_to_normal_svd
import json
import argparse
import torch
import h5py
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
import cv2
from multiprocessing import Pool
from functools import partial

metric_keys = ["normal_cos", "normal_angle5", "normal_angle30", "normal_angle_mean"]

pointmap_normal_dir = 'point_normal_svd'

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    # 新增：获取原始 total 和 counts，用于聚合
    def get_raw(self):
        return {key: (self._data.loc[key, "total"], self._data.loc[key, "counts"]) for key in self._data.index}

    @staticmethod
    def aggregate(results_list):
        """聚合多个 MetricTracker 的结果"""
        from collections import defaultdict
        totals = defaultdict(float)
        counts = defaultdict(int)

        for raw in results_list:
            for key, (t, c) in raw.items():
                totals[key] += t
                counts[key] += c

        averages = {key: totals[key] / counts[key] if counts[key] > 0 else 0.0 for key in totals}
        return averages

class DepthToPoint:
    def __init__(self, device="cuda:0"):
        self.cache_dict = {}
        self.device = device

    def __call__(self, depth : torch.Tensor, K, device=None, cache=True):
        device = device or self.device
        squeeze_batch=False
        if depth.ndim == 3:
            depth = depth.unsqueeze(0)
            squeeze_batch=True
            if depth.shape[-1] == 1:
                depth = depth.permute(0, 3, 1, 2)
        elif depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
            squeeze_batch=True
        B, C, H, W = depth.shape
        cache_key = f"depth_to_point_{H}_{W}"
        if cache_key in self.cache_dict:
            points = self.cache_dict[cache_key]
            if points.device != device:
                points.to(device)
        else:
            grid_x, grid_y = torch.meshgrid(
                torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy"
            )
            points = (
                torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
                .reshape(3, -1)
                .float()
                .to(device)
            )
            if cache:
                self.cache_dict[cache_key] = points
        rays_d = K.inverse() @ points  # (B, 3, HW)
        pts = depth.flatten(2) * rays_d
        depth = pts.reshape(B, 3, H, W)
        if squeeze_batch:
            depth = depth.squeeze(0)
        return depth

def read_file(file_path, as_tensor=False):
    data_type = os.path.splitext(file_path)[-1].lower()
    if data_type in [".npy"]:
        data = np.load(file_path)
    elif data_type in [".hdf5", ".h5"]:
        # Open the HDF5 file and read the dataset.
        with h5py.File(file_path, "r") as f:
            data = np.array(f["dataset"])
    else:
        breakpoint()
        raise NotImplementedError(f"Supported Types are npy hdf5 h5, but got {data_type}")
    if as_tensor:
        return torch.as_tensor(data).cuda()
    return data

def prepare_data(pred_path, gt_path, depth_path, min_depth=1e-3, max_depth=200, depth_scale=1):
    pred_normal = read_file(pred_path, as_tensor=True) if pred_path else None
    gt_normal = read_file(gt_path, as_tensor=True) if gt_path else None
    gt_depth = read_file(depth_path, as_tensor=True) / depth_scale if depth_path else None
    gt_mask = torch.logical_and(
        gt_depth > min_depth,
        gt_depth < max_depth
    ) if depth_path else None
    return pred_normal, gt_normal, gt_depth, gt_mask

def eval_single_frame(
    pred : torch.Tensor,
    gt : torch.Tensor,
    eval_mask : torch.Tensor,
    metric_tracker : MetricTracker,
    gt_points_normal=None,
    gt_points_normal_mask=None,
):
    if pred is not None:
        pass
    else:
        return metric_tracker

    if pred.shape[-1] == 3:
        pred = pred.permute(2, 0, 1)
    
    if gt is not None and gt.shape[-1] == 3:
        gt = gt.permute(2, 0, 1)
    
    n = 1
    c, _, _ = pred.shape
    if eval_mask is not None:
        masks_normals = eval_mask.contiguous().view(n, -1)
    else:
        masks_normals = gt_points_normal_mask.contiguous().view(n, -1)
    
    def calc_metric(targets_normals, predictions_normals, masks_normals, point_normal=False):
        cos_angle = torch.einsum("nc,nc->n", targets_normals[masks_normals], predictions_normals[masks_normals])
        cos_angle = torch.clamp(cos_angle, min=-1.0, max=1.0)
        angle_error = torch.acos(cos_angle) * 180.0 / torch.pi
        angle_error[angle_error>90] = 180 - angle_error[angle_error>90]
        valid_pics = torch.sum(masks_normals, dtype=torch.float32) + 1e-6
        
        if point_normal:
            prefix = "point_"
        else:
            prefix = ""
        
        if "normal_cos" in metric_keys:
            metric = (1 - cos_angle**2).sum() / valid_pics
            metric_tracker.update(f"{prefix}normal_cos", metric.item())
        
        if "normal_angle5" in metric_keys:
            metric = 100.0 * (torch.sum(angle_error < 5) / valid_pics)
            metric_tracker.update(f"{prefix}normal_angle5", metric.item())

        if "normal_angle30" in metric_keys:
            metric = 100.0 * (torch.sum(angle_error < 30) / valid_pics)
            metric_tracker.update(f"{prefix}normal_angle30", metric.item())
        
        if "normal_angle_mean" in metric_keys:
            metric = angle_error.sum() / valid_pics
            metric_tracker.update(f"{prefix}normal_angle_mean", metric.item())

    if gt is not None:
        gt = torch.nn.functional.normalize(gt, dim=0)
        if pred.shape[-2:] != gt.shape[-2:]:
            predictions_normals = torch.nn.functional.interpolate(pred[None], size=gt.shape[-2:], mode='bilinear', align_corners=True)[0]
            predictions_normals = torch.nn.functional.normalize(predictions_normals, dim=0)
        else:
            predictions_normals = pred
        predictions_normals = predictions_normals.contiguous().view(n, c, -1).permute(0, 2, 1).float()
        targets_normals = gt.contiguous().view(n, c, -1).permute(0, 2, 1).float()
        calc_metric(targets_normals, predictions_normals, masks_normals)
    if gt_points_normal is not None:
        targets_normals_point, point_normal_mask = gt_points_normal, gt_points_normal_mask
        targets_normals_point = torch.nn.functional.normalize(targets_normals_point, dim=0)
        if pred.shape[-2:] != targets_normals_point.shape[-2:]:
            predictions_normals = torch.nn.functional.interpolate(pred[None], size=targets_normals_point.shape[-2:], mode='bilinear', align_corners=True)[0]
            predictions_normals = torch.nn.functional.normalize(predictions_normals, dim=0)
        else:
            predictions_normals = pred
        predictions_normals = predictions_normals.contiguous().view(n, c, -1).permute(0, 2, 1).float()
        targets_normals_point = targets_normals_point.contiguous().view(n, c, -1).permute(0, 2, 1).float()
        valid_normak_mask = ~torch.isnan(targets_normals_point).any(dim=-1)
        masks_normals_point = point_normal_mask.squeeze(0).contiguous().view(n, -1)
        masks_normals_point = masks_normals_point & valid_normak_mask & masks_normals
        calc_metric(targets_normals_point, predictions_normals, masks_normals_point, point_normal=True)
    return metric_tracker

def log_mrtrics(metric_tracker, log_file=None):
    eval_text = tabulate(
        [metric_tracker.result().keys(), metric_tracker.result().values()]
    )
    print(eval_text)
    final_text = ""
    for val in metric_tracker.result().values():
        final_text += f"{val:.4f}\t"
    print(final_text)
    if log_file is not None:
        try:
            with open(log_file, 'a') as f:
                f.write(eval_text+'\n')
                f.write(final_text+'\n')
            print(f"Eval Results are saved to {log_file}")
        except:
            log_file = './eval_results.log'
            with open(log_file, 'a') as f:
                f.write(eval_text+'\n')
                f.write(final_text+'\n')
            print(f"Eval Results are saved to {log_file}")

def eval_frame_worker(
    info,
    gt_path,
    metric_keys,
    point_to_normal,
    depth_scale_default=1.0
):
    """单帧评估，返回原始统计量用于聚合"""
    if point_to_normal:
        tracker = MetricTracker(*(metric_keys + ["point_" + m for m in metric_keys]))
    else:
        tracker = MetricTracker(*metric_keys)
    tracker.reset()

    # --- 构建 pred_normal_path ---
    pred_normal_path = info.get("pred_normal")
    if pred_normal_path is None and "pred_depth" in info:
        path_, name_ = os.path.dirname(info["pred_depth"]), os.path.basename(info["pred_depth"])
        pred_normal_path = os.path.join(path_, name_.replace('depth', 'normal'))
        if not os.path.exists(pred_normal_path):
            pred_normal_path = None

    # --- GT 路径 ---
    gt_normal_path = os.path.join(gt_path, info["normal"]) if "normal" in info else None
    gt_depth_path = os.path.join(gt_path, info["depth"])

    pred_normal, gt_normal, gt_depth, gt_mask = prepare_data(
        pred_normal_path, gt_normal_path, gt_depth_path,
        depth_scale=info.get("depth_scale", depth_scale_default)
    )

    if point_to_normal:
        gt_points_normal_path = os.path.join(gt_path, info["normal_svd"])
        gt_points_normal_mask_path = os.path.join(gt_path, info["normal_mask_svd"])
        if os.path.exists(gt_points_normal_path) and os.path.exists(gt_points_normal_mask_path):
            gt_points_normal = read_file(gt_points_normal_path, as_tensor=True)
            gt_points_normal_mask = read_file(gt_points_normal_mask_path, as_tensor=True)
        else:
            cam_in = info["cam_in"]
            K = torch.eye(3).cuda().float()
            K[0, 0] = cam_in[0]
            K[1, 1] = cam_in[1]
            K[0, 2] = cam_in[2]
            K[1, 2] = cam_in[3]
            gt_points = depth_to_point(gt_depth, K)
            gt_points_normal, gt_points_normal_mask = pointmap_to_normal_svd(
                gt_points.permute(1, 2, 0), gt_mask, patch_size=3
            )
            # 缓存计算结果（注意：多进程写同一文件需确保路径唯一）
            os.makedirs(os.path.dirname(gt_points_normal_path), exist_ok=True)
            np.save(gt_points_normal_path, gt_points_normal.cpu().numpy())
            os.makedirs(os.path.dirname(gt_points_normal_mask_path), exist_ok=True)
            np.save(gt_points_normal_mask_path, gt_points_normal_mask.bool().cpu().numpy())

            # 可视化
            normal_vis_path = os.path.join(gt_path, info["normal_vis_svd"])
            normal_svd = torch.where(gt_points_normal_mask, gt_points_normal, 0).permute(1, 2, 0)
            normal_colored = normal_svd.cpu().numpy() * [0.5, -0.5, -0.5] + 0.5
            normal_colored = (normal_colored.clip(0, 1) * 255).astype(np.uint8)
            os.makedirs(os.path.dirname(normal_vis_path), exist_ok=True)
            cv2.imwrite(normal_vis_path, cv2.cvtColor(normal_colored, cv2.COLOR_RGB2BGR))

        tracker = eval_single_frame(
            pred_normal, gt_normal, gt_mask, tracker,
            gt_points_normal, gt_points_normal_mask
        )
    else:
        tracker = eval_single_frame(pred_normal, gt_normal, gt_mask, tracker)

    return tracker.get_raw()

def eval_json_multiprocess(pred_path, gt_path, point_to_normal=False, num_workers=8):
    """多进程版本：处理 pred 是单个 JSON 文件的情况"""
    print(f"评估 JSON 文件: pred={pred_path}, gt={gt_path}")

    # === 1. 加载 pred_frames ===
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
        if "files" in pred_data:
            pred_frames = []
            for _, frames in pred_data.items():
                for frame in frames:
                    pred_frames.append(frame)
        elif "mf_files" in pred_data:
            pred_frames = []
            for scene, scene_infos in pred_data["mf_files"].items():
                for frame in scene_infos:
                    for view in frame["views"]:
                        if "normal_svd" not in view and point_to_normal:
                            view["normal_svd"] = view["rgb"].replace('images', 'point_normal_svd').replace('.png', '.npy')
                            view["normal_mask_svd"] = view["rgb"].replace('images', 'point_normal_mask_svd').replace('.png', '.npy')
                            view["normal_vis_svd"] = view["rgb"].replace('images', 'point_normal_vis_svd')
                        pred_frames.append(view)
        else:
            raise ValueError("Unsupported JSON format")

    # === 2. 并行处理每一帧 ===
    worker_fn = partial(
        eval_frame_worker,
        gt_path=gt_path,
        metric_keys=metric_keys,
        point_to_normal=point_to_normal,
        depth_scale_default=1.0
    )

    with Pool(processes=num_workers) as pool:
        raw_results = list(tqdm(
            pool.imap(worker_fn, pred_frames),
            total=len(pred_frames),
            desc="Eval Normal (MP)"
        ))

    # === 3. 聚合结果 ===
    final_avg = MetricTracker.aggregate(raw_results)

    # === 4. 写入日志（模拟原 MetricTracker 接口）===
    log_file = os.path.dirname(pred_path) + '/eval_results.log'

    # 创建一个 dummy MetricTracker 用于 log_mrtrics（只填 average）
    dummy_tracker = MetricTracker(*final_avg.keys())
    for key, avg_val in final_avg.items():
        dummy_tracker._data.loc[key, "average"] = avg_val
        # 可选：填 total/counts 为 0，因 log_mrtrics 可能只读 average
    log_mrtrics(dummy_tracker, log_file=log_file)

    return final_avg

def eval_dir(pred_dir, gt_dir, point_to_normal=False, scenes_metric_tracker=None):
    """处理 pred 是目录的情况"""
    print(f"评估目录: pred={pred_dir}, gt={gt_dir}")
    if not os.path.isdir(pred_dir):
        raise ValueError(f"预测路径 {pred_dir} 不是有效目录")
    if not os.path.isdir(gt_dir):
        raise ValueError(f"真值路径 {gt_dir} 不是有效目录")
    
    if scenes_metric_tracker is not None:
        metric_tracker = scenes_metric_tracker
    else:
        if point_to_normal:
            metric_tracker = MetricTracker(*(metric_keys + ["point_"+metric for metric in metric_keys]))
        else:
            metric_tracker = MetricTracker(*metric_keys)
        metric_tracker.reset()
    
    pred_frames = [frame for frame in os.listdir(pred_dir) if frame.endswith('npy')]

    for filename in tqdm(pred_frames, desc="Eval Normal", total=len(pred_frames)):
        pred_normal_path = os.path.join(pred_dir, filename)
        
        gt_normal_path = os.path.join(gt_dir, 'normal', filename[:-4]+'.npy')
        gt_normal_path = gt_normal_path if os.path.exists(gt_normal_path) else None
        gt_depth_path = None
        
        pred_normal, gt_normal, gt_depth, gt_mask = prepare_data(pred_normal_path, gt_normal_path, gt_depth_path)
        if point_to_normal:
            
            gt_points_normal_path = os.path.join(gt_dir, 'point_normal_svd', filename[:-4]+'.npy')
            gt_points_normal_mask_path = os.path.join(gt_dir, 'point_normal_mask_svd', filename[:-4]+'.npy')
            
            if os.path.exists(gt_points_normal_path):
                gt_points_normal = read_file(gt_points_normal_path, as_tensor=True)
                gt_points_normal_mask = read_file(gt_points_normal_mask_path, as_tensor=True)
            else:
                raise FileNotFoundError(f"{gt_points_normal_path} not Found")
            metric_tracker = eval_single_frame(pred_normal, gt_normal, gt_mask, metric_tracker, gt_points_normal, gt_points_normal_mask)
        else:
            metric_tracker = eval_single_frame(pred_normal, gt_normal, gt_mask, metric_tracker)
    if scenes_metric_tracker is not None:
        return metric_tracker
    log_file = os.path.dirname(pred_dir) + '/eval_results.log'
    log_mrtrics(metric_tracker, log_file=log_file)

depth_to_point = DepthToPoint()
def main():
    parser = argparse.ArgumentParser(description="根据 pred 类型选择评估方式")
    parser.add_argument('--pred', required=True, help='预测结果路径：可以是 .json 文件或包含 .json 文件的目录')
    parser.add_argument('--gt', required=False, default='/mnt/netdata/Team/AI/datasets/TMA_Data/BenchMark/SurfaceNormal/', help='真值（ground truth）路径：对应 pred 的 .json 文件或目录')
    parser.add_argument('--workers', required=False, default=8, help='num workers')
    parser.add_argument('--mv', action='store_true')
    args = parser.parse_args()

    pred_path = args.pred
    gt_path = args.gt
    if os.path.isfile(pred_path) and pred_path.endswith('.json'):
        eval_json_multiprocess(pred_path, gt_path, True, num_workers=args.workers)
    elif os.path.isdir(pred_path):
        if args.mv:
            scenes_metric_tracker = MetricTracker(*(metric_keys + ["point_"+metric for metric in metric_keys]))
            for scene in os.listdir(pred_path):
                pred_scene = os.path.join(pred_path, scene, "normal")
                gt_scene = os.path.join(gt_path, scene)
                scenes_metric_tracker = eval_dir(pred_scene, gt_scene, True, scenes_metric_tracker)
            log_file = os.path.dirname(pred_path) + '/eval_results.log'
            log_mrtrics(scenes_metric_tracker, log_file=log_file)
        else:
            eval_dir(pred_path, gt_path, True)
    else:
        raise ValueError(f"--pred 参数必须是一个 .json 文件或一个目录，当前输入为: {pred_path}")

if __name__ == "__main__":
    main()