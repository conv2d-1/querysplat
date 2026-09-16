import torch


class DispEvalMetrics:
    def __init__(
        self,
        target_name,
        valid_mask_name,
        gt_min=0.001,
        gt_max=500,
    ):

        self.target_name = target_name
        self.valid_mask_name = valid_mask_name
        self.gt_min = gt_min
        self.gt_max = gt_max

        metrics = ["mean_error", "acc1"]
        self.metrics = []
        self.metrics.extend([f"{metric}_init" for metric in metrics])
        self.metrics.extend([f"{metric}_final" for metric in metrics])
        self.metrics.extend([f"{metric}_seq" for metric in metrics])

    def eval_single_data(self, inputs, output, eval_idx=None):
        predict_init = output.disparity
        predict_init = torch.from_numpy(predict_init)
        predict_seq = output.seq_disparity
        predict_seq = torch.concat([torch.from_numpy(pred) for pred in predict_seq])

        if eval_idx is not None:
            target = inputs[self.target_name][:, eval_idx, ...].squeeze().clone()
            valid_mask = inputs[self.valid_mask_name][:, eval_idx, ...].squeeze().clone()
        else:
            target = inputs[self.target_name].squeeze().clone()
            valid_mask = inputs[self.valid_mask_name].squeeze().clone()

        valid_mask = valid_mask & (target >= self.gt_min) & (target <= self.gt_max)
        target = target * target.shape[-1]

        predict_init = predict_init.to(target.device)
        predict_seq = predict_seq.to(target.device)
        results_dict = dict()

        for metric in ["mean_error", "acc1"]:
            results_init = eval(metric)(predict_init, target, valid_mask)
            results_final = eval(metric)(predict_seq[-1], target, valid_mask)
            results_seq = eval(metric)(predict_seq, target, valid_mask)
            if results_init is not None:
                results_dict[f"{metric}_init"] = results_init
            if results_final is not None:
                results_dict[f"{metric}_final"] = results_final
            if results_seq is not None:
                results_dict[f"{metric}_seq"] = results_seq
        return results_dict

    def eval_mf_data(self, inputs, output):
        results_dict = dict()
        valid_result = 0
        for i, out in enumerate(output):
            if out.disparity is None:
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


def mean_error(pred, gt, valid):
    pred = pred.squeeze()
    gt = gt.squeeze()
    valid = valid.squeeze().float()
    pred = pred * valid
    gt = gt * valid

    if len(pred.shape) == 2:
        err_sum = torch.abs(pred - gt).sum()
        mean_error_total = err_sum / valid.sum()
    else:
        n, _, _ = pred.shape
        mean_error_total = 0
        for i in range(n):
            err_sum = torch.abs(pred[i] - gt).sum()
            mean_error_total += err_sum / valid.sum()
        mean_error_total = mean_error_total / n
    return mean_error_total


def acc1(pred, gt, valid):
    pred = pred.squeeze()
    gt = gt.squeeze()
    valid = valid.squeeze()
    pred = pred * valid
    gt = gt * valid
    if len(pred.shape) == 2:
        diff = torch.abs(pred - gt)[valid]
        acc1_total = (diff < 1.0).float().mean()
    else:
        n, _, _ = pred.shape
        acc1_total = 0
        for i in range(n):
            diff = torch.abs(pred[i] - gt)[valid]
            acc1_total += (diff < 1.0).float().mean()
        acc1_total = acc1_total / n
    return acc1_total
