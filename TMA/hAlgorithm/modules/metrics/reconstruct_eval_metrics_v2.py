import json
import os

import cv2
import torch

from hAlgorithm.modules.losses.lpips import LPIPS, convert_to_buffer
from hAlgorithm.modules.losses.pgsr import ssim
from hAlgorithm.modules.utils.image_utils import psnr


def rgb_psnr(output, target, **kwargs):
    if kwargs.get("mask", None) is not None:
        mask = kwargs["mask"]
        if mask.ndim == 2:
            output = output[:, mask]
            target = target[:, mask]
        elif mask.ndim == 3:
            output = output[:, mask[0]]
            target = target[:, mask[0]]
        else:
            raise NotImplementedError
        return psnr(output, target).mean()
    else:
        return psnr(output, target).mean()


def rgb_ssim(output, target, **kwargs):
    if len(output.shape) == 3:
        output = output.unsqueeze(0)
        target = target.unsqueeze(0)
    if kwargs.get("mask", None) is not None:
        return ssim(output, target, mask=kwargs["mask"]).mean()
    else:
        return ssim(output, target).mean()


def rgb_lpips(output, target, lpips, **kwargs):
    if len(output.shape) == 3:
        output = output.unsqueeze(0)
        target = target.unsqueeze(0)
    if kwargs.get("mask", None) is not None:
        mask = kwargs["mask"][None, None]
        return lpips.forward(output * mask, target * mask, normalize=False).mean()
    else:
        return lpips.forward(output, target, normalize=False).mean()
    


class ReconstructEvalMetricsWithNovelView:
    def __init__(
        self,
        metrics,
        target_name,
        target_normalize=True,
    ):
        novel_metrics = ["novel_" + metric for metric in metrics]
        self.metrics = metrics + novel_metrics
    
        self.target_name = target_name
        self.target_normalize = target_normalize

        if "rgb_psnr" in self.metrics:
            self.lpips = LPIPS(net="vgg").to("cuda")
            convert_to_buffer(self.lpips, persistent=False)
        else:
            self.lpips = None

    def eval_single_data(self, inputs, output, eval_idx):
        predict = getattr(output, "render_rgb", None)
        if predict is None:
            return dict()

        predict = torch.from_numpy(predict).permute(2, 0, 1).cuda()

        if isinstance(inputs, (list, tuple)):
            target = inputs[eval_idx][self.target_name][:, 0].squeeze().clone().cuda()
        else:
            target = inputs[self.target_name][:, eval_idx].squeeze().clone().cuda()
        
        if target.shape[2] == 3:
            target = target.permute(2, 0, 1).contiguous()

        if self.target_normalize:
            if target.max() > 1.5:
                target = target / 255.0
            else:
                target = (target + 1) * 0.5

        results_dict = dict()
        for metric in self.metrics:
            if "novel" in metric:
                continue
            if self.lpips is not None:
                results = eval(metric)(predict, target, lpips=self.lpips.cuda())
            if results is not None:
                results_dict[metric] = results
        return results_dict

    def eval_novel_view(self, inputs, outputs):
        novel_views = getattr(outputs[0], "novel_render_rgb", None)
        novel_mask = getattr(outputs[0], "novel_mask", None)
        if novel_views is None:
            return dict()
        novel_views = outputs[0].novel_render_rgb
        novel_views = torch.from_numpy(novel_views).permute(0, 3, 1, 2).cuda()
        novel_target = inputs[self.target_name].squeeze().clone().cuda()
        novel_target = (novel_target + 1) * 0.5

        novel_views[novel_mask == 0] = 0
        novel_target[novel_mask == 0] = 0

        results_dict = dict()
        for metric in self.metrics:
            if "novel" in metric:
                continue
            if self.lpips is not None:
                results = eval(metric)(novel_views, novel_target, lpips=self.lpips.cuda())
            if results is not None:
                results_dict["novel_" + metric] = results
        return results_dict

    def eval_mf_data(self, inputs, outputs):
        results_dict = dict()
        valid_result = 0
        for i, output in enumerate(outputs):
            if output is None:
                continue
            result = self.eval_single_data(inputs, output, eval_idx=i)
            for k, v in result.items():
                results_dict[k] = results_dict.get(k, 0) + v
            valid_result += 1
        if valid_result == 0:
            raise ValueError("Valid Result is zero!")
        results_dict = {k: v / valid_result for k, v in results_dict.items()}

        if "novel" in inputs:
            results_dict.update(self.eval_novel_view(inputs["novel"], outputs))

        return results_dict

    def __call__(self, inputs, outputs):
        if isinstance(outputs, list):
            return self.eval_mf_data(inputs, outputs)
        else:
            return self.eval_single_data(inputs, outputs, eval_idx=None)
