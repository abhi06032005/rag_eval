"""
Paired Query-Level Bootstrap Evaluation for Statistical Validation.
Computes non-parametric paired bootstrap differences, 95% confidence intervals,
win/loss/tie rates, and regression failure analysis.
"""

from dataclasses import asdict, dataclass
import numpy as np


@dataclass
class BootstrapResult:
    system_a: str
    system_b: str
    metric_name: str
    num_queries: int
    mean_a: float
    mean_b: float
    mean_diff: float  # mean(system_a - system_b)
    ci_lower_95: float
    ci_upper_95: float
    std_error: float
    p_value: float
    is_significant: bool  # 95% CI does not overlap 0
    improvement_rate: float  # % queries where A > B
    tie_rate: float          # % queries where A == B
    regression_rate: float   # % queries where A < B
    regression_query_ids: list[str]
    scientific_conclusion: str = "no statistically significant difference observed"
    p_value_holm_adjusted: float | None = None
    p_value_bh_adjusted: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NonInferiorityResult:
    reference_system: str
    test_system: str
    metric_name: str
    margin_delta: float
    observed_diff: float  # reference - test
    ucl_95: float          # 1-sided 95% Upper Confidence Limit
    p_value_non_inferiority: float
    is_non_inferior: bool
    scientific_conclusion: str
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


def adjust_p_values_holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction for Family-Wise Error Rate (FWER)."""
    n = len(p_values)
    if n == 0:
        return []
    sorted_indices = np.argsort(p_values)
    adjusted = np.zeros(n, dtype=float)
    running_max = 0.0
    for rank, orig_idx in enumerate(sorted_indices):
        multiplier = n - rank
        adj_p = min(1.0, p_values[orig_idx] * multiplier)
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = min(1.0, running_max)
    return [round(float(p), 4) for p in adjusted]


def adjust_p_values_bh(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg procedure for False Discovery Rate (FDR) control."""
    n = len(p_values)
    if n == 0:
        return []
    sorted_indices = np.argsort(p_values)[::-1]  # descending
    adjusted = np.zeros(n, dtype=float)
    running_min = 1.0
    for rank_from_top, orig_idx in enumerate(sorted_indices):
        rank = n - rank_from_top  # 1-based rank ascending
        adj_p = (p_values[orig_idx] * n) / rank
        running_min = min(running_min, adj_p)
        adjusted[orig_idx] = min(1.0, max(0.0, running_min))
    return [round(float(p), 4) for p in adjusted]


def paired_query_bootstrap(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    system_a_name: str = "System_A",
    system_b_name: str = "System_B",
    metric_name: str = "nDCG@10",
    n_resamples: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapResult:
    """
    Performs paired query-level bootstrap hypothesis testing between two systems.
    Computes delta_i = score_a[qid] - score_b[qid] and resamples delta_i with replacement.
    """
    common_qids = [qid for qid in scores_a if qid in scores_b]
    if not common_qids:
        raise ValueError("No common query IDs found between scores_a and scores_b.")

    qids = np.array(common_qids)
    arr_a = np.array([scores_a[qid] for qid in common_qids], dtype=np.float64)
    arr_b = np.array([scores_b[qid] for qid in common_qids], dtype=np.float64)

    deltas = arr_a - arr_b
    n_queries = len(common_qids)

    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    observed_mean_diff = float(np.mean(deltas))

    # Improvement / Tie / Regression counts
    tol = 1e-6
    improvements = deltas > tol
    regressions = deltas < -tol
    ties = np.abs(deltas) <= tol

    improvement_rate = float(np.mean(improvements))
    regression_rate = float(np.mean(regressions))
    tie_rate = float(np.mean(ties))

    regression_qids = qids[regressions].tolist()

    # Paired bootstrap resampling
    rng = np.random.RandomState(seed)
    boot_indices = rng.randint(0, n_queries, size=(n_resamples, n_queries))
    boot_means = np.mean(deltas[boot_indices], axis=1)

    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    std_error = float(np.std(boot_means, ddof=1))

    # Empirical p-value: proportion of bootstrap samples that cross zero
    if observed_mean_diff >= 0:
        p_val = 2.0 * float(np.mean(boot_means <= 0.0))
    else:
        p_val = 2.0 * float(np.mean(boot_means >= 0.0))
    p_val = min(1.0, max(1.0 / n_resamples, p_val))

    # Significant if 95% CI does not contain zero
    is_significant = (ci_lower > 0.0) or (ci_upper < 0.0)
    conclusion = (
        "statistically significant difference observed"
        if is_significant
        else "no statistically significant difference observed"
    )

    return BootstrapResult(
        system_a=system_a_name,
        system_b=system_b_name,
        metric_name=metric_name,
        num_queries=n_queries,
        mean_a=round(mean_a, 4),
        mean_b=round(mean_b, 4),
        mean_diff=round(observed_mean_diff, 4),
        ci_lower_95=round(ci_lower, 4),
        ci_upper_95=round(ci_upper, 4),
        std_error=round(std_error, 4),
        p_value=round(p_val, 4),
        is_significant=is_significant,
        improvement_rate=round(improvement_rate, 4),
        tie_rate=round(tie_rate, 4),
        regression_rate=round(regression_rate, 4),
        regression_query_ids=regression_qids,
        scientific_conclusion=conclusion,
    )


def test_non_inferiority(
    scores_reference: dict[str, float],
    scores_test: dict[str, float],
    reference_name: str = "Fixed_Hybrid",
    test_name: str = "Adaptive_Router",
    metric_name: str = "nDCG@10",
    margin_delta: float = 0.025,
    alpha: float = 0.05,
    n_resamples: int = 10000,
    seed: int = 42,
) -> NonInferiorityResult:
    """
    Formal non-inferiority testing.
    H0: mu_reference - mu_test >= margin_delta (test system is inferior by margin delta or more)
    H1: mu_reference - mu_test < margin_delta  (test system is non-inferior within margin delta)
    
    A one-sided (1 - alpha) Upper Confidence Limit (UCL) is computed.
    If UCL < margin_delta, H0 is rejected at level alpha.
    """
    common_qids = [qid for qid in scores_reference if qid in scores_test]
    if not common_qids:
        raise ValueError("No common query IDs between reference and test scores.")

    arr_ref = np.array([scores_reference[qid] for qid in common_qids], dtype=np.float64)
    arr_test = np.array([scores_test[qid] for qid in common_qids], dtype=np.float64)

    # Loss delta: positive means reference was better than test
    deltas = arr_ref - arr_test
    n_queries = len(common_qids)
    observed_loss = float(np.mean(deltas))

    rng = np.random.RandomState(seed)
    boot_indices = rng.randint(0, n_queries, size=(n_resamples, n_queries))
    boot_losses = np.mean(deltas[boot_indices], axis=1)

    # 1-sided (1 - alpha) UCL = (1 - alpha) percentile of bootstrap loss distribution
    ucl = float(np.percentile(boot_losses, 100.0 * (1.0 - alpha)))

    # Empirical p-value for H0: proportion of bootstrap samples >= margin_delta
    p_non_inf = float(np.mean(boot_losses >= margin_delta))
    p_non_inf = min(1.0, max(1.0 / n_resamples, p_non_inf))

    is_non_inferior = ucl < margin_delta
    conclusion = (
        f"Non-inferiority established at margin delta={margin_delta} (UCL_95={ucl:.4f} < {margin_delta}, p={p_non_inf:.4f})"
        if is_non_inferior
        else f"Non-inferiority NOT established at margin delta={margin_delta} (UCL_95={ucl:.4f} >= {margin_delta}, p={p_non_inf:.4f})"
    )

    rationale = (
        f"Pre-specified margin delta={margin_delta} corresponds to the standard minimally clinically/practically "
        f"important difference in information retrieval evaluation (Voorhees 2000, Carterette 2007). "
        f"Testing verifies whether cost reduction occurs without unacceptable quality degradation."
    )

    return NonInferiorityResult(
        reference_system=reference_name,
        test_system=test_name,
        metric_name=metric_name,
        margin_delta=margin_delta,
        observed_diff=round(observed_loss, 4),
        ucl_95=round(ucl, 4),
        p_value_non_inferiority=round(p_non_inf, 4),
        is_non_inferior=is_non_inferior,
        scientific_conclusion=conclusion,
        rationale=rationale,
    )
