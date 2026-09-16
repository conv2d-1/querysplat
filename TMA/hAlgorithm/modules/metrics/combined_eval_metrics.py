from hAlgorithm.utils import instantiate_from_config


class CombinedEvalMetrics:
    """Unified evaluation class that runs multiple evaluators
    with shared default parameters, so they can be specified as a single config entry."""

    def __init__(self, eval_groups, task=None, **kwargs):
        self.task = task
        self.evaluators = []
        for group in eval_groups:
            group = dict(group)
            for k, v in kwargs.items():
                group.setdefault(k, v)
            self.evaluators.append(instantiate_from_config(group))

        self.metrics = []
        for evaluator in self.evaluators:
            for m in getattr(evaluator, "metrics", []):
                name = f"{self.task}|{m}" if self.task else m
                self.metrics.append(name)

    def __call__(self, inputs, output):
        results_dict = dict()
        for evaluator in self.evaluators:
            results = evaluator(inputs, output)
            results_dict.update(results)
        if self.task is not None:
            results_dict = {f"{self.task}|{k}": v for k, v in results_dict.items()}
        return results_dict
