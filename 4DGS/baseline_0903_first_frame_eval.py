"""First-frame metric evaluation config derived from baseline_0903.py."""

from pathlib import Path

_BASELINE = Path(__file__).with_name("baseline_0903.py")
with open(_BASELINE, encoding="utf-8") as _handle:
    exec(compile(_handle.read(), _BASELINE, "exec"), globals())

trainer["eval_metrics"]["metrics"][2]["eval_groups"][0]["type"] = (
    "hAlgorithm.modules.metrics.first_frame_dynamic_gaussian_eval_metrics."
    "FirstFrameDynamicGaussianEvalMetrics"
)
trainer["eval_metrics"]["metrics"][2]["eval_groups"][0]["metric_prefix"] = ""
trainer["logging_test_batch_results"] = True
trainer["in_evaluation"] = False
trainer["in_visualize"] = False
