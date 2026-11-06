import pytest

from experiments.eval_graph_policy import (
    _mean_exp_minus_graph_cost,
    _one_sided_success_rate_p,
    _single_bridge_landing_records,
)
from experiments.eval_bridges import _json_safe


def test_q3_success_test_uses_one_sided_binomial_against_v0_rate():
    assert _one_sided_success_rate_p(220, 500, 0.384) < 0.05
    assert _one_sided_success_rate_p(190, 500, 0.384) > 0.05


def test_unreachable_graph_cost_contributes_zero_to_discounted_bound():
    assert _mean_exp_minus_graph_cost([0.0, None], 0.6931471805599453) == pytest.approx(0.5)


def test_unreachable_matched_starts_are_skipped_by_single_bridge_diagnostic():
    outcomes = [
        {"planned_bridges": None, "bridge_landing": []},
        {"planned_bridges": 1, "bridge_landing": [{"bridge_type": 0, "landed": True}]},
        {"planned_bridges": 2, "bridge_landing": [{"bridge_type": 0, "landed": False}]},
    ]
    assert _single_bridge_landing_records(outcomes) == [{"bridge_type": 0, "landed": True}]


def test_bridge_result_json_sanitizes_infinite_improvement_from_unreachable_source():
    result = _json_safe({
        "source_cost_initially_unreachable": True,
        "improvement_steps": float("inf"),
    })
    assert result == {
        "source_cost_initially_unreachable": True,
        "improvement_steps": None,
    }


@pytest.mark.parametrize(
    ("successes", "n", "baseline"),
    [(-1, 10, 0.5), (11, 10, 0.5), (2, 10, 1.1)],
)
def test_q3_success_test_rejects_invalid_counts_or_baseline(successes, n, baseline):
    with pytest.raises(ValueError):
        _one_sided_success_rate_p(successes, n, baseline)
