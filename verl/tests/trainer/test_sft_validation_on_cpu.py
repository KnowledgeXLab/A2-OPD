from verl.trainer.sft_validation import (
    _resolve_ground_truth,
    _resolve_prompt,
    build_math_acc_metric_name,
    is_better_metric,
)


def test_is_better_metric_max_mode():
    assert is_better_metric(0.6, None, "max", 0.0)
    assert is_better_metric(0.6, 0.5, "max", 0.0)
    assert not is_better_metric(0.5, 0.6, "max", 0.0)
    assert not is_better_metric(0.51, 0.5, "max", 0.02)


def test_is_better_metric_min_mode():
    assert is_better_metric(0.4, None, "min", 0.0)
    assert is_better_metric(0.4, 0.5, "min", 0.0)
    assert not is_better_metric(0.5, 0.4, "min", 0.0)
    assert not is_better_metric(0.49, 0.5, "min", 0.02)


def test_resolve_math_test_total_row():
    import pandas as pd

    row = pd.Series(
        {
            "prompt": [{"role": "user", "content": "Compute 1+1"}],
            "reward_model": {"ground_truth": "2", "style": "rule"},
        }
    )
    assert _resolve_prompt(row, "prompt") == [{"role": "user", "content": "Compute 1+1"}]
    assert _resolve_ground_truth(row) == "2"


def test_build_math_acc_metric_name():
    assert build_math_acc_metric_name(1) == "val/math_acc"
    assert build_math_acc_metric_name(8) == "val/math_acc/pass@8"
    assert build_math_acc_metric_name(8, metric_kind="mean") == "val/math_acc/mean@8"
