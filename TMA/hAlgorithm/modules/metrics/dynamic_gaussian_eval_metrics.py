from __future__ import annotations

from typing import Dict, Sequence

import torch

from hAlgorithm.modules.losses.lpips import LPIPS, convert_to_buffer
from hAlgorithm.modules.metrics.reconstruct_eval_metrics import (
    rgb_lpips,
    rgb_psnr,
    rgb_ssim,
)


class DynamicGaussianEvalMetrics:
    """Evaluate Dynamic 4D Gaussian renders against ground-truth RGB frames."""

    _SUPPORTED_METRICS = {
        "rgb_psnr": rgb_psnr,
        "rgb_ssim": rgb_ssim,
        "rgb_lpips": rgb_lpips,
    }

    def __init__(
        self,
        metrics: Sequence[str],
        target_name: str = "image",
        render_attr: str = "dgs_render_rgb",
        target_normalize: bool = True,
        metric_prefix: str = "dgs_",
    ) -> None:
        self.base_metrics = list(metrics)
        self.target_name = target_name
        self.render_attr = render_attr
        self.target_normalize = target_normalize
        self.metric_prefix = metric_prefix

        unsupported_metrics = [
            metric for metric in self.base_metrics
            if metric not in self._SUPPORTED_METRICS
        ]
        if unsupported_metrics:
            raise ValueError(
                f"Unsupported DynamicGaussianEvalMetrics metrics: {unsupported_metrics}. "
                f"Supported metrics are: {list(self._SUPPORTED_METRICS.keys())}."
            )

        self.metrics = [self._prefix_metric_name(metric) for metric in self.base_metrics]

        self.lpips = None
        self.lpips_device = None
        if "rgb_lpips" in self.base_metrics:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.lpips_device = torch.device(device)
            self.lpips = LPIPS(net="vgg").to(device)
            self.lpips.eval()
            convert_to_buffer(self.lpips, persistent=False)

    def _prefix_metric_name(self, metric_name: str) -> str:
        if self.metric_prefix and not metric_name.startswith(self.metric_prefix):
            return f"{self.metric_prefix}{metric_name}"
        return metric_name

    def _get_target(self, inputs, eval_idx=None):
        if isinstance(inputs, (list, tuple)):
            if eval_idx is None:
                eval_idx = 0
            target = inputs[eval_idx][self.target_name][:, 0].squeeze().clone()
        else:
            if self.target_name not in inputs:
                return None
            if eval_idx is None:
                target = inputs[self.target_name].squeeze().clone()
            else:
                target = inputs[self.target_name][:, eval_idx].squeeze().clone()

        target = target.float()
        if self.target_normalize:
            target = (target + 1.0) * 0.5
        return target.clamp(0.0, 1.0)

    def eval_single_data(self, inputs, output, eval_idx=None) -> Dict[str, torch.Tensor]:
        predict = getattr(output, self.render_attr, None)
        if predict is None:
            return dict()

        target = self._get_target(inputs, eval_idx=eval_idx)
        if target is None:
            return dict()

        predict = torch.from_numpy(predict).permute(2, 0, 1).contiguous().float()
        predict = predict.clamp(0.0, 1.0)

        metric_device = self.lpips_device if self.lpips is not None else target.device

        predict = predict.to(metric_device)
        target = target.to(metric_device)

        results_dict = dict()
        for base_metric, metric_name in zip(self.base_metrics, self.metrics):
            metric_fn = self._SUPPORTED_METRICS[base_metric]
            kwargs = {"lpips": self.lpips} if base_metric == "rgb_lpips" else {}
            result = metric_fn(predict, target, **kwargs)
            if result is not None:
                results_dict[metric_name] = result
        return results_dict

    def eval_mf_data(self, inputs, outputs) -> Dict[str, torch.Tensor]:
        results_dict = dict()
        valid_result = 0

        for eval_idx, output in enumerate(outputs):
            if output is None:
                continue

            result = self.eval_single_data(inputs, output, eval_idx=eval_idx)
            if not result:
                continue

            for key, value in result.items():
                results_dict[key] = results_dict.get(key, 0) + value
            valid_result += 1

        if valid_result == 0:
            return dict()

        return {key: value / valid_result for key, value in results_dict.items()}

    def __call__(self, inputs, outputs):
        if isinstance(outputs, list):
            return self.eval_mf_data(inputs, outputs)
        return self.eval_single_data(inputs, outputs, eval_idx=None)
