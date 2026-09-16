import torch


class VideoDepthEvalMetrics(object):
    def __init__(
        self,
        metrics,
        target_name,
        valid_mask_name,
        gt_min_depth=1e-6,
        gt_max_depth=200,
        conf_thresh=None,
        conf_ext="conf",
    ):
        self.metrics = metrics

        self.target_name = target_name
        self.valid_mask_name = valid_mask_name
        self.gt_min_depth = gt_min_depth
        self.gt_max_depth = gt_max_depth
        self.conf_ext = conf_ext

        self.conf_thresh = conf_thresh
        if self.conf_thresh is not None:
            self.metrics = [metric + f"_{self.conf_ext}" for metric in metrics]

    def __call__(self, inputs, outputs):
        if isinstance(outputs, list):
            frame_num = inputs["meta_data"]["frames"][0]
            view_num = inputs["meta_data"]["views"][0]
            assert frame_num * view_num == len(outputs)

            predict = torch.stack(
                [torch.from_numpy(output.depth_align) for output in outputs]
            ).squeeze()
            target = inputs[self.target_name].squeeze().clone()
            valid_mask = inputs[self.valid_mask_name].squeeze().clone()
            valid_mask = valid_mask & (target >= self.gt_min_depth) & (target <= self.gt_max_depth)
            predict = predict.to(target.device)

            assert predict.shape == target.shape
            results_dict = dict()

            if self.conf_thresh is not None:
                confidence = torch.stack(
                    [torch.from_numpy(output.confidence) for output in outputs]
                ).squeeze()
                confidence = confidence.to(target.device)
                valid_mask = valid_mask & (confidence > self.conf_thresh)

            for metric in self.metrics:
                # eval input shape: n, h, w
                if self.conf_thresh is not None:
                    results = eval(metric[: -len(self.conf_ext) - 1])(predict, target, valid_mask)
                else:
                    results = eval(metric)(predict, target, valid_mask)
                if results is not None:
                    results_dict[metric] = results
            return results_dict
        else:
            raise ValueError("output must be a list")


def endpoint_error(output, target, valid_mask):
    """
    Compute the endpoint error metric
    """
    endpoint_error = (valid_mask * (output - target) ** 2).sum(dim=1).sqrt()
    return endpoint_error.mean()


def temporal_endpoint_error(output, target, valid_mask):
    """
    Compute the temporal endpoint error metric
    """
    delta_mask = valid_mask[:-1] * valid_mask[1:]
    endpoint_error = (
        (delta_mask * ((output[:-1] - output[1:]) - (target[:-1] - target[1:])) ** 2)
        .sum(dim=1)
        .sqrt()
    )
    return endpoint_error.mean()
