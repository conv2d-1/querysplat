import torch


class PromptEvalMetrics:
    def __init__(
        self,
        metrics,
        target_name,
        valid_mask_name,
        predict_threshold=0,
        target_threshold=0.5,
    ):
        self.metrics = metrics

        self.target_name = target_name
        self.valid_mask_name = valid_mask_name

        self.predict_threshold = predict_threshold  # score
        self.target_threshold = target_threshold  # distance

    def __call__(self, inputs, output):
        predict = output.input_confidence
        if predict is None:
            return dict()

        predict = torch.from_numpy(predict)
        target = inputs[self.target_name].squeeze().clone()
        valid_mask = inputs[self.valid_mask_name].squeeze().clone()

        # 预处理：将预测值和目标值二值化
        predict_binary = (predict > self.predict_threshold).float()
        target_binary = (target <= self.target_threshold).float()

        results_dict = dict()
        for metric in self.metrics:
            results = eval(metric)(predict, target, predict_binary, target_binary, valid_mask)
            if results is not None:
                results_dict[metric] = results
        return results_dict


def prompt_accuracy(output, target, output_binary, target_binary, valid_mask):
    """
    计算准确率 (Accuracy)。
    衡量预测正确的像素点占总有效像素点的比例。
    """
    # 计算正确预测的像素点数
    correct_predictions = (output_binary == target_binary) & valid_mask
    total_valid = valid_mask.sum((-1, -2))
    correct_count = correct_predictions.sum((-1, -2))

    # 避免除以零
    accuracy_value = torch.where(
        total_valid > 0, correct_count / total_valid, torch.tensor(0.0, device=output.device)
    )
    return accuracy_value.mean()


def prompt_precision(output, target, output_binary, target_binary, valid_mask):
    """
    计算精确率 (Precision)。
    衡量预测为正类的像素中，实际为正类的比例。
    """
    # 计算真阳性 (TP) 和假阳性 (FP)
    true_positive = ((output_binary == 1) & (target_binary == 1) & valid_mask).sum((-1, -2))
    false_positive = ((output_binary == 1) & (target_binary == 0) & valid_mask).sum((-1, -2))

    # 避免除以零
    precision_value = torch.where(
        true_positive + false_positive > 0,
        true_positive / (true_positive + false_positive),
        torch.tensor(0.0, device=output.device),
    )
    return precision_value.mean()


def prompt_recall(output, target, output_binary, target_binary, valid_mask):
    """
    计算召回率 (Recall)。
    衡量实际为正类的像素中，被正确预测为正类的比例。
    """
    # 计算真阳性 (TP) 和假阴性 (FN)
    true_positive = ((output_binary == 1) & (target_binary == 1) & valid_mask).sum((-1, -2))
    false_negative = ((output_binary == 0) & (target_binary == 1) & valid_mask).sum((-1, -2))

    # 避免除以零
    recall_value = torch.where(
        true_positive + false_negative > 0,
        true_positive / (true_positive + false_negative),
        torch.tensor(0.0, device=output.device),
    )
    return recall_value.mean()


def prompt_f1_score(output, target, output_binary, target_binary, valid_mask):
    """
    计算 F1 分数 (F1 Score)。
    综合衡量精确率和召回率的调和平均值。
    """
    # 计算精确率和召回率
    prec = prompt_precision(output, target, output_binary, target_binary, valid_mask)
    rec = prompt_recall(output, target, output_binary, target_binary, valid_mask)

    # 避免除以零
    f1 = torch.where(
        prec + rec > 0, 2 * prec * rec / (prec + rec), torch.tensor(0.0, device=output.device)
    )
    return f1


def prompt_abs_difference(output, target, output_binary, target_binary, valid_mask):
    cur_valid_mask = output_binary.bool()
    abs_diff = torch.abs(target)
    n = cur_valid_mask.sum((-1, -2))
    if n.sum() == 0:
        return 0 * output.sum()
    else:
        abs_diff = abs_diff[n > 0]
        n = n[n > 0]
    abs_diff = torch.sum(abs_diff, (-1, -2)) / n
    return abs_diff.mean()
