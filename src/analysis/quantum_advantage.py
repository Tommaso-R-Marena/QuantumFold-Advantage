"""Quantum Advantage Statistical Analysis.

Rigorous statistical framework for demonstrating quantum advantage in
protein structure prediction.

Implements:
- Paired hypothesis testing
- Effect size calculation
- Multiple comparison correction
- Power analysis
- Bootstrap confidence intervals
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import scipy.stats as stats


@dataclass
class AdvantageTestResult:
    """Results from quantum advantage statistical test."""

    # Hypothesis test results
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    ttest_statistic: float
    ttest_pvalue: float

    # Effect sizes
    cohens_d: float
    rank_biserial: float
    cliff_delta: float

    # Confidence intervals
    mean_diff_ci_lower: float
    mean_diff_ci_upper: float
    median_diff_ci_lower: float
    median_diff_ci_upper: float

    # Summary statistics
    quantum_mean: float
    quantum_std: float
    classical_mean: float
    classical_std: float
    mean_difference: float
    median_difference: float

    # Power analysis
    statistical_power: float
    required_sample_size: float

    # Multiple comparison correction
    bonferroni_pvalue: float
    fdr_pvalue: float

    # Metadata
    n_samples: int
    metric_name: str
    higher_is_better: bool
    significance_level: float

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result shows significant quantum advantage."""
        return self.fdr_pvalue < alpha and self.mean_difference > 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "hypothesis_tests": {
                "wilcoxon_statistic": self.wilcoxon_statistic,
                "wilcoxon_pvalue": self.wilcoxon_pvalue,
                "ttest_statistic": self.ttest_statistic,
                "ttest_pvalue": self.ttest_pvalue,
            },
            "effect_sizes": {
                "cohens_d": self.cohens_d,
                "rank_biserial": self.rank_biserial,
                "cliff_delta": self.cliff_delta,
            },
            "confidence_intervals": {
                "mean_diff_95ci": [self.mean_diff_ci_lower, self.mean_diff_ci_upper],
                "median_diff_95ci": [self.median_diff_ci_lower, self.median_diff_ci_upper],
            },
            "summary_statistics": {
                "quantum": {"mean": self.quantum_mean, "std": self.quantum_std},
                "classical": {"mean": self.classical_mean, "std": self.classical_std},
                "mean_difference": self.mean_difference,
                "median_difference": self.median_difference,
            },
            "power_analysis": {
                "statistical_power": self.statistical_power,
                "required_sample_size": self.required_sample_size,
            },
            "corrected_pvalues": {
                "bonferroni": self.bonferroni_pvalue,
                "fdr": self.fdr_pvalue,
            },
            "metadata": {
                "n_samples": self.n_samples,
                "metric_name": self.metric_name,
                "higher_is_better": self.higher_is_better,
                "alpha": self.significance_level,
                "is_significant": self.is_significant(),
            },
        }


class QuantumAdvantageAnalyzer:
    """Statistical analyzer for quantum advantage claims.

    Implements rigorous statistical methodology for comparing quantum-enhanced
    and classical models, following best practices for computational biology
    benchmarking.

    References:
        - Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets",
          JMLR (2006)
        - Benavoli et al., "Time for a Change: a Tutorial for Comparing Multiple
          Classifiers Through Bayesian Analysis", JMLR (2017)
    """

    def __init__(self, n_bootstrap: int = 10000, random_state: int = 42):
        """Initialize analyzer.

        Args:
            n_bootstrap: Number of bootstrap samples for CI estimation
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_state)

    def cohens_d(self, quantum_scores: np.ndarray, classical_scores: np.ndarray) -> float:
        """Calculate Cohen's d effect size.

        Cohen's d measures the standardized difference between two means.

        Interpretation:
            |d| < 0.2: negligible
            0.2 ≤ |d| < 0.5: small
            0.5 ≤ |d| < 0.8: medium
            |d| ≥ 0.8: large

        Args:
            quantum_scores: Quantum model scores
            classical_scores: Classical baseline scores

        Returns:
            Cohen's d value
        """
        mean_diff = np.mean(quantum_scores) - np.mean(classical_scores)
        pooled_std = np.sqrt(
            (np.var(quantum_scores, ddof=1) + np.var(classical_scores, ddof=1)) / 2
        )
        return mean_diff / pooled_std if pooled_std > 0 else 0.0

    def rank_biserial_correlation(
        self, quantum_scores: np.ndarray, classical_scores: np.ndarray
    ) -> float:
        """Calculate rank-biserial correlation (effect size for Wilcoxon test).

        Args:
            quantum_scores: Quantum model scores
            classical_scores: Classical baseline scores

        Returns:
            Rank-biserial correlation in [-1, 1]
        """
        diffs = quantum_scores - classical_scores
        n_pos = np.sum(diffs > 0)
        n_neg = np.sum(diffs < 0)
        n_total = len(diffs)

        return (n_pos - n_neg) / n_total

    def cliff_delta(self, quantum_scores: np.ndarray, classical_scores: np.ndarray) -> float:
        """Calculate Cliff's Delta (non-parametric effect size).

        Interpretation:
            |δ| < 0.147: negligible
            0.147 ≤ |δ| < 0.33: small
            0.33 ≤ |δ| < 0.474: medium
            |δ| ≥ 0.474: large

        Args:
            quantum_scores: Quantum model scores
            classical_scores: Classical baseline scores

        Returns:
            Cliff's Delta in [-1, 1]
        """
        n_q = len(quantum_scores)
        n_c = len(classical_scores)

        dominance = 0
        for q in quantum_scores:
            dominance += np.sum(q > classical_scores)
            dominance -= np.sum(q < classical_scores)

        return dominance / (n_q * n_c)

    def bootstrap_ci(
        self, data: np.ndarray, statistic_func, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval.

        Args:
            data: Input data array
            statistic_func: Function to compute statistic
            alpha: Significance level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_stats = []
        n = len(data)

        for _ in range(self.n_bootstrap):
            sample = self.rng.choice(data, size=n, replace=True)
            stat = statistic_func(sample)
            bootstrap_stats.append(stat)

        bootstrap_stats = np.array(bootstrap_stats)
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return lower, upper

    def paired_ttest(
        self, quantum_scores: np.ndarray, classical_scores: np.ndarray
    ) -> Tuple[float, float]:
        """Paired t-test for dependent samples.

        Args:
            quantum_scores: Quantum model scores
            classical_scores: Classical baseline scores

        Returns:
            Tuple of (t_statistic, p_value)
        """
        return stats.ttest_rel(quantum_scores, classical_scores)

    def wilcoxon_test(
        self, quantum_scores: np.ndarray, classical_scores: np.ndarray
    ) -> Tuple[float, float]:
        """Wilcoxon signed-rank test (non-parametric alternative to t-test).

        Args:
            quantum_scores: Quantum model scores
            classical_scores: Classical baseline scores

        Returns:
            Tuple of (test_statistic, p_value)
        """
        return stats.wilcoxon(
            quantum_scores, classical_scores, alternative="greater", zero_method="zsplit"
        )

    def statistical_power(self, effect_size: float, n: int, alpha: float = 0.05) -> float:
        """Estimate statistical power of test.

        Args:
            effect_size: Cohen's d effect size
            n: Sample size
            alpha: Significance level

        Returns:
            Estimated power (probability of correctly rejecting H0)
        """
        from scipy.stats import nct

        # Non-centrality parameter
        delta = effect_size * np.sqrt(n)

        # Critical value
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

        # Power calculation
        power = 1 - nct.cdf(t_crit, df=n - 1, nc=delta)

        return power

    def required_sample_size(
        self, effect_size: float, power: float = 0.8, alpha: float = 0.05
    ) -> int:
        """Calculate required sample size for desired power.

        Args:
            effect_size: Expected Cohen's d
            power: Desired statistical power
            alpha: Significance level

        Returns:
            Required sample size
        """
        from scipy.optimize import brentq

        def power_diff(n):
            return self.statistical_power(effect_size, int(n), alpha) - power

        try:
            n_required = brentq(power_diff, 2, 10000)
            return int(np.ceil(n_required))
        except BaseException:
            return -1  # Cannot achieve desired power

    def bonferroni_correction(self, pvalue: float, n_comparisons: int) -> float:
        """Apply Bonferroni correction for multiple comparisons.

        Args:
            pvalue: Uncorrected p-value
            n_comparisons: Number of comparisons

        Returns:
            Bonferroni-corrected p-value
        """
        return min(1.0, pvalue * n_comparisons)

    def fdr_correction(self, pvalues: List[float], alpha: float = 0.05) -> List[float]:
        """Benjamini-Hochberg FDR correction.

        Args:
            pvalues: List of uncorrected p-values
            alpha: Significance level

        Returns:
            List of FDR-corrected p-values
        """
        pvalues = np.array(pvalues)
        n = len(pvalues)
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]

        # Compute FDR-adjusted p-values
        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[i] = sorted_pvals[i] * n / (i + 1)

        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        # Restore original order
        fdr_pvals = np.zeros(n)
        fdr_pvals[sorted_idx] = adjusted

        return fdr_pvals.tolist()

    def analyze_advantage(
        self,
        quantum_scores: np.ndarray,
        classical_scores: np.ndarray,
        metric_name: str = "TM-score",
        higher_is_better: bool = True,
        alpha: float = 0.05,
        n_comparisons: int = 1,
    ) -> AdvantageTestResult:
        """Comprehensive quantum advantage analysis.

        Args:
            quantum_scores: Quantum model performance scores
            classical_scores: Classical baseline performance scores
            metric_name: Name of evaluation metric
            higher_is_better: Whether higher scores indicate better performance
            alpha: Significance level for hypothesis tests
            n_comparisons: Number of comparisons (for correction)

        Returns:
            AdvantageTestResult with complete statistical analysis
        """
        # Ensure arrays
        quantum_scores = np.asarray(quantum_scores)
        classical_scores = np.asarray(classical_scores)

        if len(quantum_scores) != len(classical_scores):
            raise ValueError("Quantum and classical scores must have same length")

        n = len(quantum_scores)

        # Flip scores if lower is better
        if not higher_is_better:
            quantum_scores = -quantum_scores
            classical_scores = -classical_scores

        # Hypothesis tests
        wilcoxon_stat, wilcoxon_p = self.wilcoxon_test(quantum_scores, classical_scores)
        ttest_stat, ttest_p = self.paired_ttest(quantum_scores, classical_scores)

        # Effect sizes
        cohens_d = self.cohens_d(quantum_scores, classical_scores)
        rank_biserial = self.rank_biserial_correlation(quantum_scores, classical_scores)
        cliff_delta = self.cliff_delta(quantum_scores, classical_scores)

        # Confidence intervals for differences
        diffs = quantum_scores - classical_scores
        mean_diff_ci = self.bootstrap_ci(diffs, np.mean, alpha)
        median_diff_ci = self.bootstrap_ci(diffs, np.median, alpha)

        # Summary statistics
        quantum_mean = np.mean(quantum_scores)
        quantum_std = np.std(quantum_scores, ddof=1)
        classical_mean = np.mean(classical_scores)
        classical_std = np.std(classical_scores, ddof=1)
        mean_diff = quantum_mean - classical_mean
        median_diff = np.median(quantum_scores) - np.median(classical_scores)

        # Power analysis
        power = self.statistical_power(cohens_d, n, alpha)
        req_n = self.required_sample_size(cohens_d, 0.8, alpha)

        # Multiple comparison correction
        bonf_p = self.bonferroni_correction(wilcoxon_p, n_comparisons)
        fdr_p = self.fdr_correction([wilcoxon_p] * n_comparisons)[0]

        return AdvantageTestResult(
            wilcoxon_statistic=float(wilcoxon_stat),
            wilcoxon_pvalue=float(wilcoxon_p),
            ttest_statistic=float(ttest_stat),
            ttest_pvalue=float(ttest_p),
            cohens_d=float(cohens_d),
            rank_biserial=float(rank_biserial),
            cliff_delta=float(cliff_delta),
            mean_diff_ci_lower=float(mean_diff_ci[0]),
            mean_diff_ci_upper=float(mean_diff_ci[1]),
            median_diff_ci_lower=float(median_diff_ci[0]),
            median_diff_ci_upper=float(median_diff_ci[1]),
            quantum_mean=float(quantum_mean),
            quantum_std=float(quantum_std),
            classical_mean=float(classical_mean),
            classical_std=float(classical_std),
            mean_difference=float(mean_diff),
            median_difference=float(median_diff),
            statistical_power=float(power),
            required_sample_size=float(req_n),
            bonferroni_pvalue=float(bonf_p),
            fdr_pvalue=float(fdr_p),
            n_samples=int(n),
            metric_name=metric_name,
            higher_is_better=higher_is_better,
            significance_level=alpha,
        )

    def save_results(self, result: AdvantageTestResult, output_path: Union[str, Path]) -> None:
        """Save analysis results to JSON.

        Args:
            result: AdvantageTestResult object
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
