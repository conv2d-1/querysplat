import torch

from hAlgorithm.utils import instantiate_from_config


class LocalEvalMetrics:
    def __init__(self, local_loss):
        self.metrics = ["local_l1"]
        self.local_loss = instantiate_from_config(local_loss)

    def eval_single_data(self, inputs, output, eval_idx=None):
        target = inputs["pointmap"]
        valid_mask = inputs["depth_mask"]
        # prompt_scale = inputs["sparse_pointmap_max_range"]
        sift_mask = inputs["sift_mask"].bool()

        if eval_idx is not None:
            target = target[:, eval_idx]
            valid_mask = valid_mask[:, eval_idx]
            # prompt_scale = prompt_scale[:, eval_idx, None, None, None]
            sift_mask = sift_mask[:, eval_idx]
        # else:
        #     prompt_scale = prompt_scale[..., None, None, None]

        prompt_scale = 1

        target_norm = target / prompt_scale
        pointmap_pred = torch.from_numpy(
            output.pointmap.reshape(output.pointmap_h, output.pointmap_w, 3)
        ).permute(2, 0, 1)[None]
        pointmap_pred = pointmap_pred / prompt_scale

        local_loss = self.local_loss(
            prediction=pointmap_pred,
            target=target_norm,
            mask=valid_mask,
            sift_point_mask=sift_mask,
            name="",
        )
        results_dict = dict(local_l1=local_loss)
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


class GlobalLocalEvalMetrics(LocalEvalMetrics):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.metrics = ["glb_local_l1"]

    def __call__(self, inputs, output):
        mv_target = inputs["pointmap_reff"]
        valid_mask = inputs["depth_mask"]
        # prompt_scale = inputs["sparse_pointmap_max_range"]
        sift_mask = inputs["sift_mask"].bool()

        sift_track_points = inputs["sift_track_points"]
        sift_track_points_cam = inputs["sift_track_points_cam"]
        sift_track_points_uv = inputs["sift_track_points_uv"]
        sift_track_mask = inputs["sift_track_mask"].bool()
        sift_track_vis = inputs["sift_track_vis"].bool()

        # prompt_scale = prompt_scale[..., None, None, None]
        prompt_scale = 1

        mv_target_norm = mv_target / prompt_scale
        mv_pointmap_pred = [
            torch.from_numpy(
                output_i.glb_mv_pointmap.reshape(output_i.pointmap_h, output_i.pointmap_w, 3)
            )
            for output_i in output
        ]
        mv_pointmap_pred = torch.stack(mv_pointmap_pred).permute(0, 3, 1, 2)[None]
        mv_pointmap_pred = mv_pointmap_pred / prompt_scale

        local_loss = self.local_loss(
            prediction=mv_pointmap_pred,
            target=mv_target_norm,
            mask=valid_mask,
            sift_point_mask=sift_mask,
            sift_track_points=sift_track_points,
            sift_track_points_cam=sift_track_points_cam,
            sift_track_points_uv=sift_track_points_uv,
            sift_track_mask=sift_track_mask,
            sift_track_vis=sift_track_vis,
            name="",
        )
        results_dict = dict(glb_local_l1=local_loss)
        return results_dict


class SiftLocalDepthEvalMetrics:
    def __init__(
        self,
        metrics,
        target_name,
        valid_mask_name,
        gt_min_depth=1e-6,
        gt_max_depth=200,
        conf_thresh=None,
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
        self.dist_ext = dist_ext
        self.dist_thresh = dist_thresh

        self.ext = ""
        if self.conf_thresh is not None:
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
        predict = output.depth_align
        predict = torch.from_numpy(predict)
        if eval_idx is not None:
            sift_eval_mask = inputs["sift_eval_raw_mask"][:, eval_idx, ...].squeeze().clone()
            sift_point_mask = inputs["sift_raw_mask"][:, eval_idx, ...].squeeze().clone()
            target = inputs[self.target_name][:, eval_idx, ...].squeeze().clone()
            valid_mask = inputs[self.valid_mask_name][:, eval_idx, ...].squeeze().clone()
        else:
            sift_eval_mask = inputs["sift_eval_raw_mask"].squeeze().clone()
            sift_point_mask = inputs["sift_raw_mask"].squeeze().clone()
            target = inputs[self.target_name].squeeze().clone()
            valid_mask = inputs[self.valid_mask_name].squeeze().clone()

        valid_mask = valid_mask & (target >= self.gt_min_depth) & (target <= self.gt_max_depth)
        valid_mask = valid_mask & sift_eval_mask

        predict = predict.to(target.device)
        results_dict = dict()

        if self.dist_thresh is not None:
            valid_nums = valid_mask.sum()
            valid_mask = valid_mask & (target >= self.gt_min_depth) & (target <= self.dist_thresh)
            results_dict[f"valid_ratio_{self.dist_ext}"] = valid_mask.sum() / valid_nums

        if self.conf_thresh is not None:
            confidence = torch.from_numpy(output.confidence)
            confidence = confidence.to(target.device)
            valid_nums = valid_mask.sum()
            valid_mask = valid_mask & (confidence > self.conf_thresh)
            results_dict[f"valid_ratio_{self.conf_ext}"] = valid_mask.sum() / valid_nums

        for metric in self.metrics:
            if metric.startswith("valid_ratio_"):
                continue
            if len(self.ext) > 0:
                results = eval(metric[: -len(self.ext)])(
                    predict, target, valid_mask, sift_point_mask
                )
            else:
                results = eval(metric)(predict, target, valid_mask, sift_point_mask)
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


def local_abs_difference(output, target, valid_mask, sift_point_mask):
    return local_abs_difference_v2(output, target, valid_mask, sift_point_mask)
    actual_output = output.squeeze()
    actual_target = target.squeeze()
    H, W = actual_target.shape
    sift_point_mask = sift_point_mask & valid_mask
    points = torch.argwhere(sift_point_mask.bool())
    # local_error = torch.zeros_like(actual_target)
    local_error = []
    radius = 3
    for y, x in points:
        y_start = max(0, y - radius)
        y_end = min(H, y + radius + 1)
        x_start = max(0, x - radius)
        x_end = min(W, x + radius + 1)
        pt_mask = valid_mask[y_start:y_end, x_start:x_end]
        gt_area = actual_target[y_start:y_end, x_start:x_end][pt_mask]
        pred_area = actual_output[y_start:y_end, x_start:x_end][pt_mask]
        max_val = gt_area.max()
        min_val = gt_area.min()
        gt_area = (gt_area - min_val) / (max_val - min_val)
        pred_area = (pred_area - min_val) / (max_val - min_val)
        local_error.append(torch.abs(pred_area - gt_area).mean())
        # local_error[y_start:y_end, x_start:x_end][pt_mask] = torch.abs(pred_area - gt_area)

    # local_error = local_error[local_error>0]
    # return local_error.mean()
    return sum(local_error) / len(local_error)


def local_abs_difference_v2(output, target, valid_mask, sift_point_mask):
    prediction = output.squeeze() * valid_mask
    target = target.squeeze() * valid_mask

    prediction = prediction.unsqueeze(0)
    target = target.unsqueeze(0)

    sift_point_mask = (sift_point_mask & valid_mask).unsqueeze(0)

    if sift_point_mask.sum() < 1:
        return torch.tensor(0)

    radius = 3
    diameter = 2 * radius + 1
    range_index = prediction.new_tensor(
        [[i, j] for i in range(-radius, radius + 1) for j in range(-radius, radius + 1)]
    )

    bs, ys, xs = torch.where(sift_point_mask.bool())

    B, H, W = sift_point_mask.shape
    N = sift_point_mask.sum()
    range_index = range_index[None].repeat(N, 1, 1)
    range_index[:, :, 0] = torch.clip(range_index[:, :, 0] + ys[:, None], min=0, max=H - 1)
    range_index[:, :, 1] = torch.clip(range_index[:, :, 1] + xs[:, None], min=0, max=W - 1)
    bs = bs[:, None, None].repeat(1, diameter * diameter, 1)
    range_index = torch.cat([bs, range_index], dim=-1).long()
    anchor_gt = target[:][bs[:, 0, 0], ys, xs][:, None]
    anchor_pred = prediction[:][bs[:, 0, 0], ys, xs][:, None]
    range_gt = target[:][range_index[..., 0], range_index[..., 1], range_index[..., 2]]
    range_pred = prediction[:][range_index[..., 0], range_index[..., 1], range_index[..., 2]]

    dis_mask = (range_gt - anchor_gt).abs() <= (anchor_gt * 0.1)
    val_mask = (range_gt > 0) * dis_mask
    N = val_mask.sum(dim=-1)

    _max = (((range_gt - anchor_gt) * val_mask).abs().max(dim=-1)[0] + 1e-6).unsqueeze(-1)
    _max_mask = _max > 0.01
    val_mask = val_mask * _max_mask

    if val_mask.sum() < 1:
        return torch.tensor(0)

    range_gt_norm = (range_gt - anchor_gt) * val_mask / _max
    range_pred_norm = (range_pred - anchor_gt) * val_mask / _max
    local_error = (range_pred_norm - range_gt_norm) * val_mask
    local_error = (local_error.abs().sum(dim=-1) / N)[_max_mask[:, 0]].mean()

    if not torch.isfinite(local_error):
        print("Nan or inf in local_abs_difference_v2")
        local_error = torch.tensor(1)

    return local_error
