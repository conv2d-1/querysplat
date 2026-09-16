import logging

import numpy as np
import torch

from hAlgorithm.modules.utils.alignment import depth2disparity


class DepthEvalMetrics:
    def __init__(
        self,
        metrics,
        target_name,
        valid_mask_name,
        gt_min_depth=1e-6,
        gt_max_depth=200,
        conf_thresh=None,
        conf_ratio=None,
        conf_ext="conf",
        dist_thresh=None,
        dist_ext="dist",
    ):
        self.metrics = metrics

        self.target_name = target_name
        self.valid_mask_name = valid_mask_name
        self.gt_min_depth = gt_min_depth
        self.gt_max_depth = gt_max_depth
        self.conf_ext = conf_ext
        self.conf_thresh = conf_thresh
        self.conf_ratio = conf_ratio
        self.dist_ext = dist_ext
        self.dist_thresh = dist_thresh

        self.ext = ""
        if self.conf_thresh is not None or self.conf_ratio is not None:
            self.ext = self.ext + f"_{self.conf_ext}"
            self.metrics.append(f"valid_ratio_{self.conf_ext}")
        if self.dist_thresh is not None:
            self.ext = self.ext + f"_{self.dist_ext}"
            self.metrics.append(f"valid_ratio_{self.dist_ext}")

        self.metrics = [
            metric + f"{self.ext}" if not metric.startswith("valid_ratio") else metric
            for metric in metrics
        ]

    def eval_single_data(self, inputs, output, eval_idx=None):
        if self.target_name not in inputs:
            return dict()

        predict = output.depth_align
        if predict is None:
            return dict()

        predict = torch.from_numpy(predict)
        if eval_idx is not None:
            target = inputs[self.target_name][:, eval_idx, ...].squeeze().clone()
            valid_mask = inputs[self.valid_mask_name][:, eval_idx, ...].squeeze().clone()
        else:
            target = inputs[self.target_name].squeeze().clone()
            valid_mask = inputs[self.valid_mask_name].squeeze().clone()

        valid_mask = valid_mask & (target >= self.gt_min_depth) & (target <= self.gt_max_depth)

        predict = predict.to(target.device)
        results_dict = dict()

        if self.dist_thresh is not None:
            valid_nums = valid_mask.sum()
            valid_mask = valid_mask & (target >= self.gt_min_depth) & (target <= self.dist_thresh)
            results_dict[f"valid_ratio_{self.dist_ext}"] = valid_mask.sum() / valid_nums

        if self.conf_thresh is not None and output.confidence is not None:
            confidence = torch.from_numpy(output.confidence)
            confidence = confidence.to(target.device)
            valid_nums = valid_mask.sum()
            valid_mask = valid_mask & (confidence > self.conf_thresh)
            results_dict[f"valid_ratio_{self.conf_ext}"] = valid_mask.sum() / valid_nums
        elif self.conf_ratio is not None and output.confidence is not None:
            confidence = torch.from_numpy(output.confidence)
            confidence = confidence.to(target.device)
            conf_thresh = np.percentile(confidence.reshape(-1), self.conf_ratio * 100)
            valid_nums = valid_mask.sum()
            valid_mask = valid_mask & (confidence > conf_thresh)
            results_dict[f"valid_ratio_{self.conf_ext}"] = valid_mask.sum() / valid_nums

        if (~torch.isnan(predict)).sum() == 0:
            logging.warning("DepthEvalMetrics, Output is all NaN!")
            return dict()

        for metric in self.metrics:
            if metric.startswith("valid_ratio_"):
                continue
            if len(self.ext) > 0:
                results = eval(metric[: -len(self.ext)])(predict, target, valid_mask)
            else:
                results = eval(metric)(predict, target, valid_mask)
            if results is not None:
                results_dict[metric] = results
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


class RelDepthEvalMetrics(DepthEvalMetrics):
    def __init__(self, disp_space=True, **kwargs):
        super().__init__(**kwargs)
        self.metrics = [f"rel_{metric}" for metric in self.metrics]
        self.disp_space = disp_space

    def eval_single_data(self, inputs, output, eval_idx=None):
        predict = output.rel_depth
        predict = torch.from_numpy(predict)
        if eval_idx is not None:
            target = inputs[self.target_name][:, eval_idx, ...].squeeze().clone()
            valid_mask = inputs[self.valid_mask_name][:, eval_idx, ...].squeeze().clone()
        else:
            target = inputs[self.target_name].squeeze().clone()
            valid_mask = inputs[self.valid_mask_name].squeeze().clone()

        valid_mask = valid_mask & (target >= self.gt_min_depth) & (target <= self.gt_max_depth)

        if self.disp_space:
            valid_mask = valid_mask & (predict > 1e-1)
            predict = depth2disparity(predict)

        predict = predict.to(target.device)
        results_dict = dict()

        if self.dist_thresh is not None:
            valid_nums = valid_mask.sum()
            valid_mask = valid_mask & (target >= self.gt_min_depth) & (target <= self.dist_thresh)
            results_dict[f"rel_valid_ratio_{self.dist_ext}"] = valid_mask.sum() / valid_nums

        if self.conf_thresh is not None:
            confidence = torch.from_numpy(output.confidence)
            confidence = confidence.to(target.device)
            valid_nums = valid_mask.sum()
            valid_mask = valid_mask & (confidence > self.conf_thresh)
            results_dict[f"rel_valid_ratio_{self.conf_ext}"] = valid_mask.sum() / valid_nums
        elif self.conf_ratio is not None:
            confidence = torch.from_numpy(output.confidence)
            confidence = confidence.to(target.device)
            conf_thresh = np.percentile(confidence.reshape(-1), self.conf_ratio * 100)
            valid_nums = valid_mask.sum()
            valid_mask = valid_mask & (confidence > conf_thresh)
            results_dict[f"rel_valid_ratio_{self.conf_ext}"] = valid_mask.sum() / valid_nums

        for metric in self.metrics:
            rel_metric = metric[4:]
            if rel_metric.startswith("valid_ratio_"):
                continue
            if len(self.ext) > 0:
                results = eval(rel_metric[: -len(self.ext)])(predict, target, valid_mask)
            else:
                results = eval(rel_metric)(predict, target, valid_mask)

            # if metric in ['abs_relative_difference']:
            #     if results > 0.1:
            #         print('abs_relative_difference', results, "prompt num: ", inputs['sparse_pointmap_mask'].sum())
            #         print(inputs['meta_data']['data_idx'], inputs['meta_data']['data_info']['rgb'])
            if results is not None:
                results_dict[metric] = results
        return results_dict


class GlobalDepthEvalMetrics(DepthEvalMetrics):
    def __init__(
        self,
        mv_target_name=None,
        glb2local=False,
        local2glb=False,
        metric_add_distance=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.mv_target_name = mv_target_name
        self.glb2local = glb2local
        self.local2glb = local2glb
        self.metric_add_distance = metric_add_distance

        if self.glb2local:
            self.prefix = "glb2local"
        elif self.local2glb:
            self.prefix = "local2glb"
        else:
            self.prefix = "glb"

        self.metrics = [f"{self.prefix}_{metric}" for metric in self.metrics]

        if self.metric_add_distance:
            if self.conf_thresh is not None or self.conf_ratio is not None:
                valid_ratio = self.metrics.pop(-1)
                self.metrics.append(f"{self.prefix}_distance_{self.conf_ext}")
                self.metrics.append(valid_ratio)
            else:
                self.metrics.append(f"{self.prefix}_distance")

    def eval_single_data(self, inputs, output, eval_idx=None):
        if self.mv_target_name not in inputs:
            return dict()

        if self.glb2local:
            if output.glb2local_pointmap is None:
                return dict()

            predict = torch.from_numpy(output.glb2local_pointmap)
            target = inputs[self.target_name]
            valid_mask = inputs[self.valid_mask_name]
        elif self.local2glb:
            if output.local2glb_pointmap is None:
                return dict()

            predict = torch.from_numpy(output.local2glb_pointmap).reshape(
                output.pointmap_h, output.pointmap_w, 3
            )
            target = inputs[self.mv_target_name]
            valid_mask = inputs[self.valid_mask_name]
        else:
            if output.glb_mv_pointmap is None:
                return dict()

            predict = torch.from_numpy(output.glb_mv_pointmap)
            target = inputs[self.mv_target_name]
            valid_mask = inputs[self.valid_mask_name]

        if eval_idx is not None:
            valid_mask = valid_mask[:, eval_idx]
            target = target[:, eval_idx]

        if self.metric_add_distance:
            predict = predict[..., :3]
        else:
            predict = predict[..., -1]

        target = target.squeeze().clone()
        valid_mask = valid_mask.squeeze().clone()

        if self.metric_add_distance:
            target = target.permute(1, 2, 0)
            valid_mask = (
                valid_mask
                & (target[:, :, 2] >= self.gt_min_depth)
                & (target[:, :, 2] <= self.gt_max_depth)
            )
        else:
            if target.ndim == 3 and target.shape[0] == 3:
                target = target[-1]
            valid_mask = valid_mask & (target >= self.gt_min_depth) & (target <= self.gt_max_depth)

        results_dict = dict()

        if self.conf_thresh is not None and output.glb_mv_confidence is not None:
            confidence = torch.from_numpy(output.glb_mv_confidence)
            valid_nums = valid_mask.sum()
            valid_mask = valid_mask & (confidence > self.conf_thresh)
            results_dict[f"{self.prefix}_valid_ratio_{self.conf_ext}"] = (
                valid_mask.sum() / valid_nums
            )
        elif self.conf_ratio is not None and output.glb_mv_confidence is not None:
            confidence = torch.from_numpy(output.glb_mv_confidence)
            conf_thresh = np.percentile(confidence.reshape(-1), self.conf_ratio * 100)
            valid_nums = valid_mask.sum()
            valid_mask = valid_mask & (confidence > conf_thresh)
            results_dict[f"{self.prefix}_valid_ratio_{self.conf_ext}"] = (
                valid_mask.sum() / valid_nums
            )

        for metric in self.metrics:
            fix_metric = metric[len(self.prefix) + 1 :]
            if self.conf_thresh is not None or self.conf_ratio is not None:
                fix_metric = fix_metric[: -len(self.conf_ext) - 1]
            if self.metric_add_distance and fix_metric != "distance":
                results = eval(fix_metric)(predict[..., -1], target[..., -1], valid_mask)
            else:
                results = eval(fix_metric)(predict, target, valid_mask)
            if results is not None:
                results_dict[metric] = results
        return results_dict


def valid_ratio(output, target, valid_mask=None):
    return None


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / torch.abs(actual_target)
    
    if abs_relative_diff.ndim == 2:
        invalid_mask = torch.isinf(abs_relative_diff)
    elif abs_relative_diff.ndim == 3:
        invalid_mask = torch.isinf(abs_relative_diff).sum(-1) > 0

    if invalid_mask.sum() > 0:
        if valid_mask is not None:
            valid_mask = valid_mask & ~invalid_mask
        else:
            valid_mask = ~invalid_mask

    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    if n == 0:
        return 0 * output.sum()
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def mean_accuracy(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    z_diff = (actual_output - actual_target)[valid_mask]
    k = int(len(z_diff) * 0.02)
    z_diff, _ = torch.sort(z_diff)
    z_diff = z_diff[k:-k]
    return z_diff.abs().mean()


def mean_error(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    z_diff = (actual_output - actual_target)[valid_mask]
    k = int(len(z_diff) * 0.02)
    z_diff, _ = torch.sort(z_diff)
    z_diff = z_diff[k:-k]
    return z_diff.mean()


def std_error(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    z_diff = (actual_output - actual_target)[valid_mask]
    k = int(len(z_diff) * 0.02)
    z_diff, _ = torch.sort(z_diff)
    z_diff = z_diff[k:-k]
    return z_diff.std()


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = torch.pow(torch.abs(actual_output - actual_target), 2) / torch.abs(
        actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    if n == 0:
        return 0 * output.sum()
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    if n == 0:
        return 0 * output.sum()
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output.clip(min=1e-3)) - torch.log(target.clip(min=1e-3))
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    if n == 0:
        return 0 * output.sum()
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None):
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask].clip(min=1e-3))
            - torch.log10(target[valid_mask].clip(min=1e-3))
        )
    else:
        diff = torch.abs(torch.log10(output.clip(min=1e-3)) - torch.log10(target.clip(min=1e-3)))
    if len(diff) == 0:
        return 0 * output.sum()
    return diff.mean()


# adapt from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    if n == 0:
        return 0 * output.sum() + 1
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None):
    output_inv = 1.0 / output.clip(min=1e-3)
    target_inv = 1.0 / target.clip(min=1e-3)
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    if n == 0:
        return 0 * output.sum()
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse + 1e-6)
    return rmse.mean()


def silog_rmse(output, target, valid_mask=None):
    diff = torch.log(output.clip(min=1e-3)) - torch.log(target.clip(min=1e-3))
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = target.shape[-2] * target.shape[-1]
    if n == 0:
        return 0 * output.sum()
    diff2 = torch.pow(diff, 2)
    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term) + 1e-6) * 100
    return loss


def abs_difference(output, target, valid_mask=None):
    abs_diff = torch.abs(output - target)
    if valid_mask is not None:
        abs_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    if n == 0:
        return 0 * output.sum()
    abs_diff = torch.sum(abs_diff, (-1, -2)) / n
    return abs_diff.mean()


def distance(output, target, valid_mask=None):
    diff = output - target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = valid_mask.shape[-1] * valid_mask.shape[-2]
    if n == 0:
        return 0 * output.sum()
    diff = torch.norm(diff, dim=-1).sum((-1, -2)) / n
    return diff.mean()
