# Copyright (c) OpenMMLab. All rights reserved.

from collections import OrderedDict

import numpy as np
import torch


def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return score


def total_intersect_and_union(results, gt_seg_maps, num_classes, ignore_index, label_map=dict(), reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes,), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(result, gt_seg_map, num_classes, ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, total_area_label


def intersect_and_union(pred_label, label, num_classes, ignore_index, label_map=dict(), reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        # label = torch.from_numpy(
        #     mmcv.imread(label, flag='unchanged', backend='pillow'))
        pass
    else:
        label = torch.from_numpy(label)

    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
    if label_map is not None:
        label_copy = label.clone()
        for old_id, new_id in label_map.items():
            label[label_copy == old_id] = new_id

    mask = label != ignore_index
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_area_to_metrics(total_area_intersect, total_area_union, total_area_pred_label, total_area_label, metrics=["mIoU"], nan_to_num=None, beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ["mIoU", "mDice", "mFscore"]
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError("metrics {} is not supported".format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({"aAcc": all_acc})
    for metric in metrics:
        if metric == "mIoU":
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics["IoU"] = iou
            ret_metrics["Acc"] = acc
        elif metric == "mDice":
            dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics["Dice"] = dice
            ret_metrics["Acc"] = acc
        elif metric == "mFscore":
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor([f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics["Fscore"] = f_value
            ret_metrics["Precision"] = precision
            ret_metrics["Recall"] = recall

    ret_metrics = {metric: value.numpy() for metric, value in ret_metrics.items()}
    if nan_to_num is not None:
        ret_metrics = OrderedDict({metric: np.nan_to_num(metric_value, nan=nan_to_num) for metric, metric_value in ret_metrics.items()})
    return ret_metrics


def eval_metrics(results, gt_seg_maps, num_classes, ignore_index, metrics=["mIoU"], nan_to_num=None, label_map=dict(), reduce_zero_label=False, beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    total_area_intersect, total_area_union, total_area_pred_label, total_area_label = total_intersect_and_union(results, gt_seg_maps, num_classes, ignore_index, label_map, reduce_zero_label)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union, total_area_pred_label, total_area_label, metrics, nan_to_num, beta)

    return ret_metrics


def mean_iou(results, gt_seg_maps, num_classes, ignore_index, nan_to_num=None, label_map=dict(), reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results, gt_seg_maps=gt_seg_maps, num_classes=num_classes, ignore_index=ignore_index, metrics=["mIoU"], nan_to_num=nan_to_num, label_map=label_map, reduce_zero_label=reduce_zero_label
    )
    return iou_result


def mean_dice(results, gt_seg_maps, num_classes, ignore_index, nan_to_num=None, label_map=dict(), reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results, gt_seg_maps=gt_seg_maps, num_classes=num_classes, ignore_index=ignore_index, metrics=["mDice"], nan_to_num=nan_to_num, label_map=label_map, reduce_zero_label=reduce_zero_label
    )
    return dice_result


def get_confusion_matrix(pred_label, label, num_classes, ignore_index):
    """Intersection over Union
    Args:
        pred_label (np.ndarray): 2D predict map
        label (np.ndarray): label 2D label map
        num_classes (int): number of categories
        ignore_index (int): index ignore in evaluation
    """

    mask = label != ignore_index
    pred_label = pred_label[mask]
    label = label[mask]

    n = num_classes
    inds = n * label + pred_label

    mat = np.bincount(inds, minlength=n**2).reshape(n, n)

    return mat


# This func is deprecated since it's not memory efficient
def legacy_mean_iou(results, gt_seg_maps, num_classes, ignore_index):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_mat = np.zeros((num_classes, num_classes), dtype=np.float)
    for i in range(num_imgs):
        mat = get_confusion_matrix(results[i], gt_seg_maps[i], num_classes, ignore_index=ignore_index)
        total_mat += mat
    all_acc = np.diag(total_mat).sum() / total_mat.sum()
    acc = np.diag(total_mat) / total_mat.sum(axis=1)
    iou = np.diag(total_mat) / (total_mat.sum(axis=1) + total_mat.sum(axis=0) - np.diag(total_mat))

    return all_acc, acc, iou


# This func is deprecated since it's not memory efficient
def legacy_mean_dice(results, gt_seg_maps, num_classes, ignore_index):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_mat = np.zeros((num_classes, num_classes), dtype=np.float)
    for i in range(num_imgs):
        mat = get_confusion_matrix(results[i], gt_seg_maps[i], num_classes, ignore_index=ignore_index)
        total_mat += mat
    all_acc = np.diag(total_mat).sum() / total_mat.sum()
    acc = np.diag(total_mat) / total_mat.sum(axis=1)
    dice = 2 * np.diag(total_mat) / (total_mat.sum(axis=1) + total_mat.sum(axis=0))

    return all_acc, acc, dice


def test_metrics():
    pred_size = (10, 30, 30)
    num_classes = 19
    ignore_index = 255
    results = np.random.randint(0, num_classes, size=pred_size)
    label = np.random.randint(0, num_classes, size=pred_size)
    label[:, 2, 5:10] = ignore_index
    all_acc, acc, iou = eval_metrics(results, label, num_classes, ignore_index, metrics="mIoU")
    all_acc_l, acc_l, iou_l = legacy_mean_iou(results, label, num_classes, ignore_index)
    assert all_acc == all_acc_l
    assert np.allclose(acc, acc_l)
    assert np.allclose(iou, iou_l)

    all_acc, acc, dice = eval_metrics(results, label, num_classes, ignore_index, metrics="mDice")
    all_acc_l, acc_l, dice_l = legacy_mean_dice(results, label, num_classes, ignore_index)
    assert all_acc == all_acc_l
    assert np.allclose(acc, acc_l)
    assert np.allclose(dice, dice_l)

    all_acc, acc, iou, dice = eval_metrics(results, label, num_classes, ignore_index, metrics=["mIoU", "mDice"])
    assert all_acc == all_acc_l
    assert np.allclose(acc, acc_l)
    assert np.allclose(iou, iou_l)
    assert np.allclose(dice, dice_l)

    results = np.random.randint(0, 5, size=pred_size)
    label = np.random.randint(0, 4, size=pred_size)
    all_acc, acc, iou = eval_metrics(results, label, num_classes, ignore_index=255, metrics="mIoU", nan_to_num=-1)
    assert acc[-1] == -1
    assert iou[-1] == -1

    all_acc, acc, dice = eval_metrics(results, label, num_classes, ignore_index=255, metrics="mDice", nan_to_num=-1)
    assert acc[-1] == -1
    assert dice[-1] == -1

    all_acc, acc, dice, iou = eval_metrics(results, label, num_classes, ignore_index=255, metrics=["mDice", "mIoU"], nan_to_num=-1)
    assert acc[-1] == -1
    assert dice[-1] == -1
    assert iou[-1] == -1


def test_mean_iou():
    pred_size = (10, 30, 30)
    num_classes = 19
    ignore_index = 255
    results = np.random.randint(0, num_classes, size=pred_size)
    label = np.random.randint(0, num_classes, size=pred_size)
    label[:, 2, 5:10] = ignore_index
    all_acc, acc, iou = mean_iou(results, label, num_classes, ignore_index)
    all_acc_l, acc_l, iou_l = legacy_mean_iou(results, label, num_classes, ignore_index)
    assert all_acc == all_acc_l
    assert np.allclose(acc, acc_l)
    assert np.allclose(iou, iou_l)

    results = np.random.randint(0, 5, size=pred_size)
    label = np.random.randint(0, 4, size=pred_size)
    all_acc, acc, iou = mean_iou(results, label, num_classes, ignore_index=255, nan_to_num=-1)
    assert acc[-1] == -1
    assert iou[-1] == -1


def test_mean_dice():
    pred_size = (10, 30, 30)
    num_classes = 19
    ignore_index = 255
    results = np.random.randint(0, num_classes, size=pred_size)
    label = np.random.randint(0, num_classes, size=pred_size)
    label[:, 2, 5:10] = ignore_index
    all_acc, acc, iou = mean_dice(results, label, num_classes, ignore_index)
    all_acc_l, acc_l, iou_l = legacy_mean_dice(results, label, num_classes, ignore_index)
    assert all_acc == all_acc_l
    assert np.allclose(acc, acc_l)
    assert np.allclose(iou, iou_l)

    results = np.random.randint(0, 5, size=pred_size)
    label = np.random.randint(0, 4, size=pred_size)
    all_acc, acc, iou = mean_dice(results, label, num_classes, ignore_index=255, nan_to_num=-1)
    assert acc[-1] == -1
    assert iou[-1] == -1


class MaskEvalMetrics:
    def __init__(self, gt_mask_name, pred_mask_name, threshold=0.0):

        self.metrics = ["aAcc"]
        for i in range(2):
            self.metrics.append("IoU{}".format(i))
            self.metrics.append("Acc{}".format(i))

        self.gt_mask_name = gt_mask_name
        self.pred_mask_name = pred_mask_name
        self.threshold = threshold

    def eval_single_data(self, inputs, output, eval_idx):
        if self.gt_mask_name not in inputs:
            return dict()
        if not hasattr(output, self.pred_mask_name):
            return dict()

        if eval_idx is not None:
            gt_mask = inputs[self.gt_mask_name][:, eval_idx, ...].squeeze(0).long().numpy()
        else:
            gt_mask = inputs[self.gt_mask_name].squeeze(0).long().numpy()

        pred_mask = (getattr(output, self.pred_mask_name)[None] > self.threshold).astype(int)

        results_dict = mean_iou(pred_mask, gt_mask, 2, -1)
        IoU = results_dict.pop("IoU")
        Acc = results_dict.pop("Acc")
        for i in range(2):
            results_dict["IoU{}".format(i)] = IoU[i]
            results_dict["Acc{}".format(i)] = Acc[i]
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
