"""Statistical validation for quantum vs. classical comparison.

Provides bootstrap confidence intervals, Wilcoxon signed-rank tests,
Cohen's d effect sizes, and multiple-testing correction.  Designed for
the paired-sample setting where the *same* proteins are predicted by
both the quantum-enhanced and classical-only models.

References:
    - Efron & Tibshirani, *An Introduction to the Bootstrap* (1993).
    - Cohen, *Statistical Power Analysis* (1988).
    - Holm, Scand. J. Statist. 6, 65–70 (1979).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    """Container for a paired statistical comparison."""

    metric_name: str
    quantum_mean: float
    classical_mean: float
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value_wilcoxon: float
    p_value_bootstrap: float
    cohens_d: float
    effect_interpretation: str
    significant_wilcoxon: bool
    significant_bootstrap: bool


# ---------------------------------------------------------------------------
# Bootstrap utilities
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Non-parametric bootstrap confidence interval.

    Returns:
        (estimate, lower, upper)
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_stats = np.array(
        [statistic(rng.choice(data, size=n, replace=True)) for _ in range(n_bootstrap)]
    )
    alpha = 1 - ci
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(statistic(data)), float(lo), float(hi)


def paired_bootstrap_test(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Tuple[float, np.ndarray]:
    """Two-sided paired bootstrap test for H0: mean(a) == mean(b).

    Returns:
        (p_value, bootstrap_diffs)
    """
    rng = np.random.RandomState(seed)
    observed_diff = np.mean(a) - np.mean(b)
    diffs = a - b
    n = len(diffs)

    boot_diffs = np.array(
        [np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(n_bootstrap)]
    )

    # Two-sided p-value
    p = float(np.mean(np.abs(boot_diffs - np.mean(boot_diffs)) >= np.abs(observed_diff)))
    return max(p, 1 / n_bootstrap), boot_diffs


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d = mean(a - b) / std(a - b)."""
    diff = a - b
    sd = np.std(diff, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def cohens_d_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI on Cohen's d."""
    rng = np.random.RandomState(seed)
    n = len(a)
    ds = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        ds.append(cohens_d(a[idx], b[idx]))
    ds = np.array(ds)
    alpha = 1 - ci
    return float(np.mean(ds)), float(np.percentile(ds, 100 * alpha / 2)), float(np.percentile(ds, 100 * (1 - alpha / 2)))


def interpret_effect(d: float) -> str:
    """Cohen's benchmarks."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Multiple-testing correction (Holm–Bonferroni)
# ---------------------------------------------------------------------------

def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm–Bonferroni step-down correction.

    Returns list of booleans: True = reject H0.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    reject = [False] * m
    for rank, idx in enumerate(order):
        adjusted_alpha = alpha / (m - rank)
        if p_values[idx] <= adjusted_alpha:
            reject[idx] = True
        else:
            break  # stop rejecting
    return reject


# ---------------------------------------------------------------------------
# Full comparison pipeline
# ---------------------------------------------------------------------------

def compare_quantum_classical(
    quantum_metrics: Dict[str, np.ndarray],
    classical_metrics: Dict[str, np.ndarray],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> List[ComparisonResult]:
    """Run full statistical comparison across all metrics.

    Args:
        quantum_metrics: {metric_name: array_of_per_protein_scores}
        classical_metrics: same keys.
        n_bootstrap: Bootstrap resamples.
        alpha: Significance level.
        seed: Random seed.

    Returns:
        List of ComparisonResult for each metric.
    """
    results: List[ComparisonResult] = []
    p_values_w: List[float] = []
    p_values_b: List[float] = []

    for metric in quantum_metrics:
        q = np.asarray(quantum_metrics[metric], dtype=float)
        c = np.asarray(classical_metrics[metric], dtype=float)
        assert len(q) == len(c), f"Mismatched lengths for {metric}"

        # Wilcoxon signed-rank (needs ≥6 non-zero diffs ideally)
        try:
            _, p_w = stats.wilcoxon(q, c, alternative="two-sided")
        except ValueError:
            p_w = 1.0
        p_values_w.append(p_w)

        # Bootstrap
        p_b, boot_diffs = paired_bootstrap_test(q, c, n_bootstrap, seed)
        p_values_b.append(p_b)

        d = cohens_d(q, c)
        diff = q - c
        _, ci_lo, ci_hi = bootstrap_ci(diff, n_bootstrap=n_bootstrap, seed=seed)

        results.append(ComparisonResult(
            metric_name=metric,
            quantum_mean=float(q.mean()),
            classical_mean=float(c.mean()),
            mean_diff=float(diff.mean()),
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            p_value_wilcoxon=p_w,
            p_value_bootstrap=p_b,
            cohens_d=d,
            effect_interpretation=interpret_effect(d),
            significant_wilcoxon=False,  # filled after correction
            significant_bootstrap=False,
        ))

    # Holm–Bonferroni correction
    reject_w = holm_bonferroni(p_values_w, alpha)
    reject_b = holm_bonferroni(p_values_b, alpha)
    for i, r in enumerate(results):
        r.significant_wilcoxon = reject_w[i]
        r.significant_bootstrap = reject_b[i]

    return results


def format_comparison_report(results: List[ComparisonResult]) -> str:
    """Pretty-print comparison results as a text report."""
    lines = [
        "=" * 80,
        "QUANTUM vs. CLASSICAL — STATISTICAL COMPARISON REPORT",
        "=" * 80,
        "",
    ]
    for r in results:
        lines.append(f"Metric: {r.metric_name}")
        lines.append(f"  Quantum mean:   {r.quantum_mean:.4f}")
        lines.append(f"  Classical mean: {r.classical_mean:.4f}")
        lines.append(f"  Mean diff:      {r.mean_diff:+.4f}  "
                      f"95% CI [{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]")
        lines.append(f"  Cohen's d:      {r.cohens_d:+.3f}  ({r.effect_interpretation})")
        lines.append(f"  Wilcoxon p:     {r.p_value_wilcoxon:.4g}  "
                      f"{'*' if r.significant_wilcoxon else 'n.s.'}")
        lines.append(f"  Bootstrap p:    {r.p_value_bootstrap:.4g}  "
                      f"{'*' if r.significant_bootstrap else 'n.s.'}")
        lines.append("")
    lines.append("=" * 80)
    lines.append("* significant after Holm–Bonferroni correction (α = 0.05)")
    return "\n".join(lines)
