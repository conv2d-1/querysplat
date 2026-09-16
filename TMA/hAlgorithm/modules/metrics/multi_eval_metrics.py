from hAlgorithm.utils import instantiate_from_config


class _CompositeMetricNames:
    """Expose derived metric names to the trainer's result aggregation."""

    def __init__(self, composites):
        self.metrics = [composite["name"] for composite in composites]

    def __call__(self, inputs, output):
        return {}


class MultiEvalMetrics:
    def __init__(self, metrics, debug=False, composite_metrics=None):
        assert isinstance(metrics, (list, tuple))
        self.metrics = []
        for metric in metrics:
            metric = instantiate_from_config(metric)
            if hasattr(metric, "debug"):
                metric.debug = debug or metric.debug
            self.metrics.append(metric)
        self.composite_metrics = composite_metrics or []
        if self.composite_metrics:
            self.metrics.append(_CompositeMetricNames(self.composite_metrics))

    def __call__(self, inputs, output):

        results_dict = dict()
        for metrics_obj in self.metrics:
            cur_results_dict = metrics_obj(inputs, output)
            results_dict.update(cur_results_dict)

        # Normalize heterogeneous metrics before combining them.  This keeps
        # the composite score bounded and makes the configured weights
        # meaningful across PSNR/SSIM/LPIPS and camera AUC metrics.
        for composite in self.composite_metrics:
            name = composite["name"]
            components = composite["components"]
            if not all(component["metric"] in results_dict for component in components):
                continue

            score = 0.0
            weight_sum = 0.0
            for component in components:
                value = results_dict[component["metric"]]
                if hasattr(value, "item"):
                    value = value.item()
                value = float(value)
                lower = float(component["lower"])
                upper = float(component["upper"])
                if upper <= lower:
                    raise ValueError(
                        f"Invalid normalization range for {component['metric']}: "
                        f"lower={lower}, upper={upper}"
                    )
                normalized = (value - lower) / (upper - lower)
                normalized = min(1.0, max(0.0, normalized))
                if component.get("invert", False):
                    normalized = 1.0 - normalized
                weight = float(component["weight"])
                score += weight * normalized
                weight_sum += weight

            if weight_sum <= 0:
                raise ValueError(f"Composite metric {name} has no positive weights")
            results_dict[name] = score / weight_sum

        return results_dict
