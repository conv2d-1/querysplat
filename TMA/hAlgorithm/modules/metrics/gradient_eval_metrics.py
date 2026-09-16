import torch


class GradientEvalMetrics:
    def __init__(
        self,
        metrics,
        target_name,
        valid_mask_name,
        gt_min_depth=1e-6,
        gt_max_depth=200,
    ):
        self.metrics = metrics

        self.target_name = target_name
        self.valid_mask_name = valid_mask_name
        self.gt_min_depth = gt_min_depth
        self.gt_max_depth = gt_max_depth

    def depth2gradient(self, gt, mask):
        gt_grad_u = torch.zeros_like(gt)
        gt_grad_u[:, 1:] = gt[:, 1:] - gt[:, :-1]
        mask_u = torch.zeros_like(mask)
        mask_u[:, 1:] = mask[:, 1:] * mask[:, :-1]

        gt_grad_v = torch.zeros_like(gt)
        gt_grad_v[1:, :] = gt[1:, :] - gt[:-1, :]
        mask_v = torch.zeros_like(mask)
        mask_v[1:, :] = mask[1:, :] * mask[:-1, :]

        gt_grad = torch.stack([gt_grad_u, gt_grad_v], dim=0)
        mask = torch.stack([mask_u, mask_v], dim=0)

        return gt_grad, mask

    def __call__(self, inputs, output):
        predict = output.depth_grad
        if predict is None:
            return dict()

        predict = torch.from_numpy(predict)
        target = inputs[self.target_name].squeeze().clone()
        valid_mask = inputs[self.valid_mask_name].squeeze().clone()
        valid_mask = valid_mask & (target >= self.gt_min_depth) & (target <= self.gt_max_depth)
        target, valid_mask = self.depth2gradient(target, valid_mask)
        predict = predict.permute(2, 0, 1)  # [2, H, W]

        results_dict = dict()
        for metric in self.metrics:
            results = eval(metric)(predict, target, valid_mask)
            if results is not None:
                results_dict[metric] = results
        return results_dict


def gradient_difference(output, target, valid_mask=None):
    abs_diff = torch.abs(output - target)
    abs_diff[~valid_mask] = 0
    n = valid_mask.sum((-1, -2))
    if n.sum() == 0:
        return 0 * output.sum()
    else:
        abs_diff = abs_diff[n > 0]
        n = n[n > 0]
    abs_diff = torch.sum(abs_diff, (-1, -2)) / n
    return abs_diff.mean()
