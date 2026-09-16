import logging

import numpy as np
import torch

# Disable scientific notation
np.set_printoptions(suppress=True)

from hAlgorithm.modules.metrics.mask_eval_metrics import *
from hAlgorithm.modules.metrics.match_eval_metrics import *
from hAlgorithm.modules.pipelines2.utils.outputs import DenseMatchingOutput


def to_pixel(x: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
    return torch.stack(((x[..., 0] + 1) / 2 * W, (x[..., 1] + 1) / 2 * H), dim=-1)


def to_pixel_coordinates(warp: torch.Tensor, H_A: int, W_A: int, H_B: int, W_B: int):
    return torch.concat((to_pixel(warp[..., :2], H=H_A, W=W_A), to_pixel(warp[..., 2:], H=H_B, W=W_B)), dim=-1)


def get_normalized_grid(
    B: int,
    H: int,
    W: int,
    overload_device: torch.device | None = None,
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=overload_device or "cuda") for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n


def warp_to_match(warp: torch.Tensor, overlap_mask: torch.Tensor):
    _, H_A, W_A, _ = warp.shape
    grid = get_normalized_grid(1, H_A, W_A)  # (H, W, 2), normalized [-1, 1]
    matches = torch.cat((grid, warp), dim=-1).reshape(-1, 4)
    matches = to_pixel_coordinates(matches, H_A, W_A, H_A, W_A)
    overlap_mask = (overlap_mask.float() > 0.5).squeeze().reshape(-1)
    return matches[overlap_mask]


def covirance_to_match(cov: torch.Tensor, overlap_mask: torch.Tensor):
    _, H_A, W_A, _, _ = cov.shape
    cov = cov.reshape(-1, 2, 2)
    overlap_mask = (overlap_mask.float() > 0.5).squeeze().reshape(-1)
    return cov[overlap_mask]


class MatchMaskEvalMetrics(MaskEvalMetrics):
    def __init__(self, gt_mask_name, pred_mask_name, threshold=0.0):
        super().__init__(gt_mask_name, pred_mask_name, threshold)

    def eval_single_data(self, inputs, output, eval_idx):
        if self.gt_mask_name not in inputs and output.overlap_gt is None:
            return dict()
        if not hasattr(output, self.pred_mask_name):
            return dict()

        if eval_idx is not None:
            if self.gt_mask_name in inputs:
                gt_mask = inputs[self.gt_mask_name][:, eval_idx, ...].squeeze(0).long().numpy()
            else:
                gt_mask = output.overlap_gt.squeeze().long().cpu().numpy()
        else:
            if self.gt_mask_name in inputs:
                gt_mask = inputs[self.gt_mask_name].squeeze(0).long().numpy()
            else:
                gt_mask = output.overlap_gt.squeeze().long().cpu().numpy()

        if gt_mask.sum() == 0:
            logging.warning("MatchMaskEvalMetrics, gt_mask is empty")
            return dict()
        
        pred_mask = getattr(output, self.pred_mask_name)[None] > self.threshold

        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.squeeze().cpu().numpy().astype(int)
        else:
            pred_mask = pred_mask.astype(int)
        
        results_dict = mean_iou(pred_mask, gt_mask, 2, -1)
        IoU = results_dict.pop("IoU")
        Acc = results_dict.pop("Acc")
        for i in range(2):
            results_dict["IoU{}".format(i)] = IoU[i]
            results_dict["Acc{}".format(i)] = Acc[i]

        return results_dict

    def eval_mf_data(self, inputs, output):
        if not hasattr(output[0], "dense_matching"):
            return dict()
        if output[0].dense_matching is None or len(output[0].dense_matching) == 0 or output[0].dense_matching[0].warp_gt is None:
            return dict()

        results_dict = dict()
        valid_result = 0
        collected_output = []

        for out in output:
            for match_out in out.dense_matching:
                collected_output.append(match_out)

        for i, out in enumerate(collected_output):
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


class DenseMatchEvalMetrics:
    def __init__(self, metrics=None, with_pose_eval=False, with_coarse=False, coarse_only=False, min_pixels=50, **kwargs):
        self.metrics = ["epe", "acc1", "acc3"]
        self.with_coarse = with_coarse
        self.coarse_only = coarse_only
        if self.with_coarse and not coarse_only:
            self.metrics = self.metrics + [metric + "_coarse" for metric in self.metrics]
        self.min_pixels = min_pixels
        self.matches_name = "warp"
        self.with_pose_eval = with_pose_eval and not self.coarse_only
        if self.with_pose_eval:
            self.pose_eval = MatchEvalMetrics(**kwargs)
            self.metrics = self.metrics + self.pose_eval.metrics

    def eval_single_data(self, inputs, output: DenseMatchingOutput, eval_idx):
        if not hasattr(output, self.matches_name):
            return dict()
        warp_gt = output.warp_gt
        gt_mask = output.overlap_gt.bool().squeeze(-1)

        if gt_mask.sum() <= 0:
            return dict()

        B, V, C, H, W = inputs["image_show"].shape
        results_dict = {}
        warp = output.warp if not self.coarse_only else output.warp_coarse
        epe = epe_error(warp, warp_gt, H, W)
        acc1 = acc(epe, gt_mask, thresh=1)
        acc3 = acc(epe, gt_mask, thresh=3)
        results_dict["epe"] = epe[gt_mask].mean()
        results_dict["acc1"] = acc1
        results_dict["acc3"] = acc3

        if self.with_coarse and not self.coarse_only:
            warp = output.warp_coarse
            epe = epe_error(warp, warp_gt, H, W)
            acc1 = acc(epe, gt_mask, thresh=1)
            acc3 = acc(epe, gt_mask, thresh=3)
            results_dict["epe_coarse"] = epe[gt_mask].mean()
            results_dict["acc1_coarse"] = acc1
            results_dict["acc3_coarse"] = acc3

        if self.with_pose_eval:
            output.matches_gt = warp_to_match(warp_gt, gt_mask).cpu().numpy()  # N, 4
            output.matches = warp_to_match(output.warp if not self.coarse_only else output.warp_coarse, output.overlap).cpu().numpy()  # N, 4
            if output.pred_covariance is not None and self.pose_eval.use_cov:
                output.cov1 = covirance_to_match(output.pred_covariance, output.overlap).cpu().numpy()  # N, 2, 2
                output.cov0 = np.zeros_like(output.cov1)  # N, 2, 2
            pose_eval = self.pose_eval.eval_single_data(inputs, output, None)
            results_dict.update(pose_eval)
        return results_dict

    def eval_mf_data(self, inputs, output):
        results_dict = dict()
        valid_result = 0
        matching_outpus = []
        if hasattr(output[0], "dense_matching"):
            for out in output:
                for match_out in out.dense_matching:
                    matching_outpus.append(match_out)
        else:
            matching_outpus = output
        for i, out in enumerate(matching_outpus):
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


class DenseMatchEvalMetricsV2(DenseMatchEvalMetrics):
    """
    Supports Confidence Eval
    """

    def __init__(self, metrics=None, conf_thresh=None, with_pose_eval=False, with_coarse=False, coarse_only=False, min_pixels=50, **kwargs):
        metrics = ["epe", "acc1", "acc3"]
        self.metrics = metrics
        self.with_coarse = with_coarse
        self.coarse_only = coarse_only
        if self.with_coarse and not coarse_only:
            self.metrics = self.metrics + [metric + "_coarse" for metric in metrics]
        self.conf_thresh = conf_thresh
        if self.conf_thresh is not None:
            self.metrics = self.metrics + [metric + "_conf" for metric in metrics]
            self.metrics += ["conf_ratio"]
        self.min_pixels = min_pixels
        self.matches_name = "warp"
        self.with_pose_eval = with_pose_eval and not self.coarse_only
        if self.with_pose_eval:
            self.pose_eval = MatchEvalMetrics(**kwargs)
            self.pose_eval.metrics = ["auc@1", "auc@5", "auc@10", "R_err", "t_err", "inliers", "epipolar_distance"]
            self.metrics = self.metrics + self.pose_eval.metrics
            if self.conf_thresh is not None:
                self.metrics = self.metrics + [metric + "_conf" for metric in self.pose_eval.metrics]

    def eval_single_data(self, inputs, output: DenseMatchingOutput, eval_idx):
        if not hasattr(output, self.matches_name):
            return dict()
        warp_gt = output.warp_gt
        gt_mask = output.overlap_gt.bool().squeeze(-1)

        if gt_mask.sum() <= 0:
            return dict()

        B, V, C, H, W = inputs["image_show"].shape
        results_dict = {}
        warp = output.warp if not self.coarse_only else output.warp_coarse
        epe = epe_error(warp, warp_gt, H, W)
        acc1 = acc(epe, gt_mask, thresh=1)
        acc3 = acc(epe, gt_mask, thresh=3)
        results_dict["epe"] = epe[gt_mask].mean()
        results_dict["acc1"] = acc1
        results_dict["acc3"] = acc3

        if self.conf_thresh is not None:
            raw_conf_mask = output.pred_covariance[..., -1] > self.conf_thresh
            conf_mask = torch.logical_and(gt_mask, raw_conf_mask)
            results_dict["conf_ratio"] = conf_mask.sum() / gt_mask.sum()
            if conf_mask.sum() > 0:
                acc1 = acc(epe, conf_mask, thresh=1)
                acc3 = acc(epe, conf_mask, thresh=3)
                results_dict["epe_conf"] = epe[conf_mask].mean()
                results_dict["acc1_conf"] = acc1
                results_dict["acc3_conf"] = acc3

        if self.with_coarse and not self.coarse_only:
            warp = output.warp_coarse
            epe = epe_error(warp, warp_gt, H, W)
            acc1 = acc(epe, gt_mask, thresh=1)
            acc3 = acc(epe, gt_mask, thresh=3)
            results_dict["epe_coarse"] = epe[gt_mask].mean()
            results_dict["acc1_coarse"] = acc1
            results_dict["acc3_coarse"] = acc3

        if self.with_pose_eval:
            output.matches_gt = warp_to_match(warp_gt, gt_mask).cpu().numpy()  # N, 4
            output.matches = warp_to_match(output.warp if not self.coarse_only else output.warp_coarse, output.overlap).cpu().numpy()  # N, 4
            if output.pred_covariance is not None and self.pose_eval.use_cov:
                output.cov1 = covirance_to_match(output.pred_covariance, output.overlap).cpu().numpy()  # N, 2, 2
                output.cov0 = np.zeros_like(output.cov1)  # N, 2, 2
            pose_eval = self.pose_eval.eval_single_data(inputs, output, None)
            results_dict.update(pose_eval)
            if self.conf_thresh is not None:
                conf_mask = torch.logical_and(output.overlap, raw_conf_mask.unsqueeze(-1))
                output.matches_gt = warp_to_match(warp_gt, gt_mask).cpu().numpy()  # N, 4
                output.matches = warp_to_match(output.warp if not self.coarse_only else output.warp_coarse, conf_mask).cpu().numpy()  # N, 4
                pose_eval = self.pose_eval.eval_single_data(inputs, output, None)
                pose_eval = {k + "_conf": v for k, v in pose_eval.items()}
                results_dict.update(pose_eval)
        return results_dict

    def eval_mf_data(self, inputs, output):
        if not hasattr(output[0], "dense_matching"):
            return dict()
        if output[0].dense_matching is None or len(output[0].dense_matching) == 0 or output[0].dense_matching[0].warp_gt is None:
            return dict()

        results_dict = dict()
        valid_result = 0
        matching_outpus = []

        for out in output:
            for match_out in out.dense_matching:
                matching_outpus.append(match_out)

        for i, out in enumerate(matching_outpus):
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


class DenseMatchEvalMetricsV3(DenseMatchEvalMetrics):
    """
    Supports Confidence Eval
    """

    def __init__(self, metrics=[], conf_thresh=None, coarse_only=False, with_pose_eval=False, min_pixels=50, **kwargs):

        self.metrics = metrics

        self.coarse_only = coarse_only

        self.conf_thresh = conf_thresh
        if self.conf_thresh is not None:
            self.metrics = self.metrics + [metric + "_conf" for metric in metrics]
            self.metrics += ["conf_ratio"]

        self.min_pixels = min_pixels

        self.with_pose_eval = with_pose_eval

        if self.with_pose_eval:
            self.pose_eval = MatchEvalMetrics(**kwargs)
            self.pose_eval.metrics = ["auc@1", "auc@5", "auc@10", "R_err", "t_err", "inliers", "epipolar_distance"]
            self.metrics = self.metrics + self.pose_eval.metrics

            if self.conf_thresh is not None:
                self.metrics = self.metrics + [metric + "_conf" for metric in self.pose_eval.metrics]

    def eval_single_data(self, inputs, output: DenseMatchingOutput, eval_idx):
        warp_gt = output.warp_gt
        gt_mask = output.overlap_gt.bool().squeeze(-1)

        if gt_mask.sum() <= 0:
            return dict()

        B, V, C, H, W = inputs["image_show"].shape
        results_dict = {}

        warp = output.warp_coarse
        epe = epe_error(warp, warp_gt, H, W)
        acc1 = acc(epe, gt_mask, thresh=1)
        acc3 = acc(epe, gt_mask, thresh=3)
        results_dict["epe_coarse"] = epe[gt_mask].mean()
        results_dict["acc1_coarse"] = acc1
        results_dict["acc3_coarse"] = acc3

        if self.coarse_only:
            raw_conf_mask = output.pred_covariance[..., -1] > self.conf_thresh
            conf_mask = torch.logical_and(gt_mask, raw_conf_mask)
            results_dict["conf_ratio"] = conf_mask.sum() / gt_mask.sum()
            if conf_mask.sum() > 0:
                acc1 = acc(epe, conf_mask, thresh=1)
                acc3 = acc(epe, conf_mask, thresh=3)
                results_dict["epe_coarse_conf"] = epe[conf_mask].mean()
                results_dict["acc1_coarse_conf"] = acc1
                results_dict["acc3_coarse_conf"] = acc3

        if not self.coarse_only:
            warp = output.warp
            epe = epe_error(warp, warp_gt, H, W)
            acc1 = acc(epe, gt_mask, thresh=1)
            acc3 = acc(epe, gt_mask, thresh=3)
            results_dict["epe"] = epe[gt_mask].mean()
            results_dict["acc1"] = acc1
            results_dict["acc3"] = acc3

            if self.conf_thresh is not None:
                raw_conf_mask = output.pred_covariance[..., -1] > self.conf_thresh
                conf_mask = torch.logical_and(gt_mask, raw_conf_mask)
                results_dict["conf_ratio"] = conf_mask.sum() / gt_mask.sum()
                if conf_mask.sum() > 0:
                    acc1 = acc(epe, conf_mask, thresh=1)
                    acc3 = acc(epe, conf_mask, thresh=3)
                    results_dict["epe_conf"] = epe[conf_mask].mean()
                    results_dict["acc1_conf"] = acc1
                    results_dict["acc3_conf"] = acc3

        warp_refine = output.warp_refine
        if not self.coarse_only and warp_refine is not None:
            for s, warp in warp_refine.items():
                epe = epe_error(warp, warp_gt, H, W)
                acc1 = acc(epe, gt_mask, thresh=1)
                acc3 = acc(epe, gt_mask, thresh=3)
                results_dict[f"epe_refine_s{s}"] = epe[gt_mask].mean()
                results_dict[f"acc1_refine_s{s}"] = acc1
                results_dict[f"acc3_refine_s{s}"] = acc3

        if self.with_pose_eval:
            output.matches_gt = warp_to_match(warp_gt, gt_mask).cpu().numpy()  # N, 4
            output.matches = warp_to_match(output.warp if not self.coarse_only else output.warp_coarse, output.overlap).cpu().numpy()  # N, 4
            if output.pred_covariance is not None and self.pose_eval.use_cov:
                output.cov1 = covirance_to_match(output.pred_covariance, output.overlap).cpu().numpy()  # N, 2, 2
                output.cov0 = np.zeros_like(output.cov1)  # N, 2, 2
            pose_eval = self.pose_eval.eval_single_data(inputs, output, None)
            results_dict.update(pose_eval)

            if self.conf_thresh is not None:
                conf_mask = torch.logical_and(output.overlap, raw_conf_mask.unsqueeze(-1))
                output.matches_gt = warp_to_match(warp_gt, gt_mask).cpu().numpy()  # N, 4
                output.matches = warp_to_match(output.warp if not self.coarse_only else output.warp_coarse, conf_mask).cpu().numpy()  # N, 4
                pose_eval = self.pose_eval.eval_single_data(inputs, output, None)
                pose_eval = {k + "_conf": v for k, v in pose_eval.items()}
                results_dict.update(pose_eval)

        return results_dict

    def eval_mf_data(self, inputs, output):
        if not hasattr(output[0], "dense_matching"):
            return dict()
        if output[0].dense_matching is None or len(output[0].dense_matching) == 0 or output[0].dense_matching[0].warp_gt is None:
            return dict()

        results_dict = dict()
        valid_result = 0
        matching_outpus = []

        for out in output:
            for match_out in out.dense_matching:
                matching_outpus.append(match_out)

        for i, out in enumerate(matching_outpus):
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


def epe_error(warp, warp_gt, H, W):
    scale = torch.tensor([(W - 1) / 2, (H - 1) / 2], device=warp.device, dtype=warp.dtype)
    epe = ((warp - warp_gt) * scale).norm(dim=-1)
    return epe


def acc(error, mask, thresh):
    tot_num = torch.sum(mask)
    inlier_num = torch.sum(error[mask] < thresh)
    return inlier_num / tot_num * 100.0
