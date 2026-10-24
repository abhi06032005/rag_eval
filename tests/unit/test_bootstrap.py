"""
Unit tests for paired query-level bootstrap evaluation.
"""

import numpy as np
import pytest
from ragbench.evaluation.bootstrap import paired_query_bootstrap


def test_paired_bootstrap_identical_systems():
    scores = {f"q{i}": 0.5 for i in range(100)}
    res = paired_query_bootstrap(scores, scores, n_resamples=1000, seed=42)
    assert res.mean_diff == 0.0
    assert res.ci_lower_95 == 0.0
    assert res.ci_upper_95 == 0.0
    assert res.tie_rate == 1.0
    assert res.improvement_rate == 0.0
    assert res.regression_rate == 0.0
    assert not res.is_significant


def test_paired_bootstrap_significant_superiority():
    scores_a = {f"q{i}": 0.8 for i in range(100)}
    scores_b = {f"q{i}": 0.5 for i in range(100)}
    res = paired_query_bootstrap(scores_a, scores_b, n_resamples=1000, seed=42)
    assert res.mean_diff == 0.3
    assert res.ci_lower_95 > 0.28
    assert res.ci_upper_95 < 0.32
    assert res.improvement_rate == 1.0
    assert res.regression_rate == 0.0
    assert res.is_significant
    assert res.p_value <= 0.001


def test_paired_bootstrap_regressions_tracked():
    scores_a = {"q1": 0.8, "q2": 0.2, "q3": 0.7}
    scores_b = {"q1": 0.5, "q2": 0.6, "q3": 0.7}
    res = paired_query_bootstrap(scores_a, scores_b, n_resamples=500, seed=42)
    assert res.regression_query_ids == ["q2"]
    assert pytest.approx(res.improvement_rate, 0.01) == 0.3333
    assert pytest.approx(res.regression_rate, 0.01) == 0.3333
    assert pytest.approx(res.tie_rate, 0.01) == 0.3333
    assert res.scientific_conclusion == "no statistically significant difference observed"


def test_non_inferiority_procedure():
    from ragbench.evaluation.bootstrap import test_non_inferiority

    # Test system is slightly lower (diff = 0.005) well within margin 0.025
    rng = np.random.RandomState(42)
    scores_ref = {f"q{i}": float(0.70 + rng.normal(0, 0.05)) for i in range(200)}
    scores_test = {f"q{i}": float(scores_ref[f"q{i}"] - 0.005 + rng.normal(0, 0.02)) for i in range(200)}

    ni_res = test_non_inferiority(scores_ref, scores_test, margin_delta=0.025, n_resamples=2000, seed=42)
    assert ni_res.is_non_inferior is True
    assert ni_res.ucl_95 < 0.025
    assert "Non-inferiority established" in ni_res.scientific_conclusion

    # Inferior system (loss = 0.05 > margin 0.025)
    scores_bad = {f"q{i}": float(scores_ref[f"q{i}"] - 0.05) for i in range(200)}
    ni_bad = test_non_inferiority(scores_ref, scores_bad, margin_delta=0.025, n_resamples=1000, seed=42)
    assert ni_bad.is_non_inferior is False
    assert "NOT established" in ni_bad.scientific_conclusion


def test_multiple_comparison_corrections():
    from ragbench.evaluation.bootstrap import adjust_p_values_bh, adjust_p_values_holm

    raw_p = [0.001, 0.012, 0.035, 0.048, 0.12, 0.45]
    holm_p = adjust_p_values_holm(raw_p)
    bh_p = adjust_p_values_bh(raw_p)

    # Holm should be monotonically non-decreasing and >= raw
    for r, h in zip(raw_p, holm_p):
        assert h >= r
    # BH should also control FDR and be <= Holm
    for h, b in zip(holm_p, bh_p):
        assert b <= h

