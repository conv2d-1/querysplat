import torch

from .dynamic_gaussian_eval_metrics import DynamicGaussianEvalMetrics


class FirstFrameDynamicGaussianEvalMetrics(DynamicGaussianEvalMetrics):
    """Evaluate only the first rendered frame/view of each multi-frame sample."""

    def eval_mf_data(self, inputs, outputs):
        if not outputs:
            return {}
        first_output = outputs[0]
        if first_output is None:
            return {}
        results = self.eval_single_data(inputs, first_output, eval_idx=0)
        # BaseTrainer's per-batch logging expects scalar Tensor values.
        return {
            key: value if hasattr(value, "item") else torch.as_tensor(value)
            for key, value in results.items()
        }
