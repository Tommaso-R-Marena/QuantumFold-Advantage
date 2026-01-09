"""Statistical validation framework for quantum advantage claims.

Implements rigorous statistical tests required for publication:
- Paired hypothesis tests (Wilcoxon, t-tests)
- Bootstrap confidence intervals
- Effect size calculations (Cohen's d, Cliff's delta)
- Multiple comparison correction (Bonferroni, Benjamini-Hochberg)
- Statistical power analysis
- Cross-validation protocols
- Significance testing with proper assumptions

References:
    - Wilcoxon test: Wilcoxon, Biometrics Bulletin (1945)
    - Bootstrap: Efron & Tibshirani, "An Introduction to the Bootstrap" (1993)
    - Cohen's d: Cohen, "Statistical Power Analysis" (1988)
    - Benjamini-Hochberg: Benjamini & Hochberg, J. Royal Stat. Soc. (1995)
    - Power analysis: Cohen (1992) DOI: 10.1037/0033-2909.112.1.155
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy import stats
from scipy.stats import friedmanchisquare, mannwhitneyu, ttest_rel, wilcoxon


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""

    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    significant: bool
    interpretation: str

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "effect_size": float(self.effect_size),
            "confidence_interval": [
                float(self.confidence_interval[0]),
                float(self.confidence_interval[1]),
            ],
            "significant": self.significant,
            "interpretation": self.interpretation,
        }


class StatisticalValidator:
    """Comprehensive statistical validation for model comparisons."""

    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000):
        """
        Args:
            alpha: Significance level (default 0.05)
            n_bootstrap: Number of bootstrap samples
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap

    def paired_wilcoxon_test(
        self, method_a: np.ndarray, method_b: np.ndarray, alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """Paired Wilcoxon signed-rank test.

        Non-parametric test for paired samples. Use when data may not be normally distributed.

        Args:
            method_a: Performance metrics for method A (n_samples,)
            method_b: Performance metrics for method B (n_samples,)
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Statistical test result
        """
        assert len(method_a) == len(method_b), "Samples must be paired"

        # Wilcoxon test
        statistic, p_value = wilcoxon(method_a, method_b, alternative=alternative)

        # Effect size (rank-biserial correlation)
        differences = method_a - method_b
        n = len(differences)
        r = 1 - (2 * statistic) / (n * (n + 1))

        # Bootstrap confidence interval
        ci = self._bootstrap_ci(differences)

        # Interpretation
        significant = p_value < self.alpha
        if alternative == "greater":
            interpretation = (
                f"Method A {'is' if significant else 'is not'} significantly better than Method B"
            )
        elif alternative == "less":
            interpretation = (
                f"Method A {'is' if significant else 'is not'} significantly worse than Method B"
            )
        else:
            interpretation = f"Methods {'differ' if significant else 'do not differ'} significantly"

        return StatisticalTestResult(
            test_name="Wilcoxon Signed-Rank Test",
            statistic=statistic,
            p_value=p_value,
            effect_size=r,
            confidence_interval=ci,
            significant=significant,
            interpretation=interpretation,
        )

    def paired_t_test(
        self, method_a: np.ndarray, method_b: np.ndarray, alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """Paired t-test.

        Parametric test for paired samples. Assumes normal distribution of differences.

        Args:
            method_a: Performance metrics for method A
            method_b: Performance metrics for method B
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Statistical test result
        """
        assert len(method_a) == len(method_b), "Samples must be paired"

        # Check normality of differences
        differences = method_a - method_b
        _, normality_p = stats.shapiro(differences)
        if normality_p < 0.05:
            warnings.warn("Differences may not be normally distributed. Consider Wilcoxon test.")

        # t-test
        statistic, p_value = ttest_rel(method_a, method_b, alternative=alternative)

        # Cohen's d effect size
        cohens_d = self._compute_cohens_d(method_a, method_b, paired=True)

        # Confidence interval
        ci = self._bootstrap_ci(differences)

        significant = p_value < self.alpha
        interpretation = self._interpret_effect_size(cohens_d)

        return StatisticalTestResult(
            test_name="Paired t-Test",
            statistic=statistic,
            p_value=p_value,
            effect_size=cohens_d,
            confidence_interval=ci,
            significant=significant,
            interpretation=f"{interpretation} (p={'<' if significant else '='}{p_value:.4f})",
        )

    def _compute_cohens_d(
        self, group_a: np.ndarray, group_b: np.ndarray, paired: bool = True
    ) -> float:
        """Compute Cohen's d effect size.

        Args:
            group_a: First group
            group_b: Second group
            paired: Whether samples are paired

        Returns:
            Cohen's d value
        """
        if paired:
            # For paired samples, use difference scores
            diff = group_a - group_b
            return np.mean(diff) / np.std(diff, ddof=1)
        else:
            # For independent samples
            n1, n2 = len(group_a), len(group_b)
            var1, var2 = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            return (np.mean(group_a) - np.mean(group_b)) / pooled_std

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible effect"
        elif abs_d < 0.5:
            return "Small effect"
        elif abs_d < 0.8:
            return "Medium effect"
        else:
            return "Large effect"

    def _bootstrap_ci(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute bootstrap confidence interval.

        Args:
            data: Data array
            confidence: Confidence level (default 0.95)

        Returns:
            (lower_bound, upper_bound) tuple
        """
        n = len(data)
        bootstrap_means = []

        for _ in range(self.n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        # Percentile method
        alpha_half = (1 - confidence) / 2
        lower = np.percentile(bootstrap_means, alpha_half * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha_half) * 100)

        return (lower, upper)

    def multiple_comparison_correction(
        self, p_values: List[float], method: str = "bonferroni"
    ) -> Tuple[List[float], List[bool]]:
        """Correct for multiple comparisons.

        Args:
            p_values: List of p-values from multiple tests
            method: 'bonferroni' or 'fdr' (Benjamini-Hochberg)

        Returns:
            (corrected_p_values, rejected) tuple
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)

        if method == "bonferroni":
            # Bonferroni correction
            corrected = p_values * n_tests
            corrected = np.minimum(corrected, 1.0)
            rejected = corrected < self.alpha

        elif method == "fdr":
            # Benjamini-Hochberg FDR
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]

            # Compute adjusted p-values
            adjusted = np.zeros(n_tests)
            for i in range(n_tests):
                adjusted[sorted_idx[i]] = min(1.0, sorted_p[i] * n_tests / (i + 1))

            # Ensure monotonicity
            for i in range(n_tests - 1, 0, -1):
                adjusted[sorted_idx[i - 1]] = min(
                    adjusted[sorted_idx[i - 1]], adjusted[sorted_idx[i]]
                )

            corrected = adjusted
            rejected = corrected < self.alpha

        else:
            raise ValueError(f"Unknown method: {method}")

        return corrected.tolist(), rejected.tolist()

    def compute_statistical_power(
        self,
        effect_size: float,
        n_samples: int,
        alpha: float = 0.05,
        test_type: str = "paired-ttest",
    ) -> float:
        """Compute statistical power.

        Args:
            effect_size: Expected effect size (Cohen's d)
            n_samples: Sample size
            alpha: Significance level
            test_type: Type of test

        Returns:
            Statistical power (0-1)
        """
        from scipy.stats import nct, t

        # Non-centrality parameter
        if test_type == "paired-ttest":
            ncp = effect_size * np.sqrt(n_samples)
        else:
            ncp = effect_size * np.sqrt(n_samples / 2)

        # Critical value
        df = n_samples - 1
        critical_value = t.ppf(1 - alpha / 2, df)

        # Power = P(reject H0 | H1 is true)
        power = 1 - nct.cdf(critical_value, df, ncp) + nct.cdf(-critical_value, df, ncp)

        return float(power)

    def sample_size_calculation(
        self, effect_size: float, power: float = 0.8, alpha: float = 0.05
    ) -> int:
        """Calculate required sample size for desired power.

        Args:
            effect_size: Expected effect size (Cohen's d)
            power: Desired statistical power
            alpha: Significance level

        Returns:
            Required sample size
        """
        # Binary search for required n
        n_low, n_high = 2, 10000

        while n_high - n_low > 1:
            n_mid = (n_low + n_high) // 2
            current_power = self.compute_statistical_power(effect_size, n_mid, alpha)

            if current_power < power:
                n_low = n_mid
            else:
                n_high = n_mid

        return n_high


class ComprehensiveBenchmark:
    """Comprehensive benchmarking with statistical validation."""

    def __init__(self, output_dir: str = "statistical_results", alpha: float = 0.05):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validator = StatisticalValidator(alpha=alpha)
        self.results = {"tests": [], "summary": {}}

    def compare_methods(
        self,
        quantum_scores: np.ndarray,
        classical_scores: np.ndarray,
        metric_name: str = "TM-score",
        higher_is_better: bool = True,
    ) -> Dict:
        """Comprehensive statistical comparison of two methods.

        Args:
            quantum_scores: Scores for quantum method
            classical_scores: Scores for classical method
            metric_name: Name of the metric
            higher_is_better: Whether higher scores are better

        Returns:
            Dictionary with all test results
        """
        print(f"\n{'='*80}")
        print(f"Statistical Comparison: Quantum vs Classical ({metric_name})")
        print(f"{'='*80}\n")

        # Descriptive statistics
        print("Descriptive Statistics:")
        print(f"  Quantum:  mean={np.mean(quantum_scores):.4f}, std={np.std(quantum_scores):.4f}")
        print(
            f"  Classical: mean={np.mean(classical_scores):.4f}, std={np.std(classical_scores):.4f}"
        )
        print(f"  Difference: {np.mean(quantum_scores - classical_scores):.4f}\n")

        # Determine alternative hypothesis
        alternative = "greater" if higher_is_better else "less"

        # Wilcoxon test (non-parametric)
        wilcoxon_result = self.validator.paired_wilcoxon_test(
            quantum_scores, classical_scores, alternative=alternative
        )
        print(f"Wilcoxon Signed-Rank Test:")
        print(f"  Statistic: {wilcoxon_result.statistic:.4f}")
        print(f"  P-value: {wilcoxon_result.p_value:.4e}")
        print(f"  Effect size (r): {wilcoxon_result.effect_size:.4f}")
        print(f"  Significant: {wilcoxon_result.significant}")
        print(f"  {wilcoxon_result.interpretation}\n")

        # Paired t-test (parametric)
        ttest_result = self.validator.paired_t_test(
            quantum_scores, classical_scores, alternative=alternative
        )
        print(f"Paired t-Test:")
        print(f"  Statistic: {ttest_result.statistic:.4f}")
        print(f"  P-value: {ttest_result.p_value:.4e}")
        print(f"  Cohen's d: {ttest_result.effect_size:.4f}")
        print(f"  {ttest_result.interpretation}\n")

        # Bootstrap confidence interval
        differences = quantum_scores - classical_scores
        ci = self.validator._bootstrap_ci(differences)
        print(f"Bootstrap 95% CI for difference: [{ci[0]:.4f}, {ci[1]:.4f}]\n")

        # Statistical power
        power = self.validator.compute_statistical_power(
            ttest_result.effect_size, len(quantum_scores)
        )
        print(f"Statistical Power: {power:.4f}")
        if power < 0.8:
            print(f"  WARNING: Low statistical power. Consider more samples.")
            required_n = self.validator.sample_size_calculation(ttest_result.effect_size)
            print(f"  Required sample size for 80% power: {required_n}\n")
        else:
            print(f"  Sufficient statistical power.\n")

        # Store results
        comparison_result = {
            "metric": metric_name,
            "quantum_mean": float(np.mean(quantum_scores)),
            "classical_mean": float(np.mean(classical_scores)),
            "quantum_std": float(np.std(quantum_scores)),
            "classical_std": float(np.std(classical_scores)),
            "wilcoxon": wilcoxon_result.to_dict(),
            "ttest": ttest_result.to_dict(),
            "bootstrap_ci": {"lower": ci[0], "upper": ci[1]},
            "statistical_power": float(power),
            "n_samples": len(quantum_scores),
        }

        self.results["tests"].append(comparison_result)

        return comparison_result

    def plot_comparison(
        self,
        quantum_scores: np.ndarray,
        classical_scores: np.ndarray,
        metric_name: str = "Score",
        save_path: Optional[str] = None,
    ):
        """Create visualization of method comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Box plot comparison
        ax = axes[0, 0]
        data = [quantum_scores, classical_scores]
        ax.boxplot(data, labels=["Quantum", "Classical"])
        ax.set_ylabel(metric_name)
        ax.set_title("Distribution Comparison")
        ax.grid(True, alpha=0.3)

        # 2. Paired differences
        ax = axes[0, 1]
        differences = quantum_scores - classical_scores
        ax.hist(differences, bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", label="No difference")
        ax.set_xlabel(f"Quantum - Classical ({metric_name})")
        ax.set_ylabel("Frequency")
        ax.set_title("Paired Differences")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Scatter plot
        ax = axes[1, 0]
        ax.scatter(classical_scores, quantum_scores, alpha=0.6)
        lim = [
            min(classical_scores.min(), quantum_scores.min()),
            max(classical_scores.max(), quantum_scores.max()),
        ]
        ax.plot(lim, lim, "r--", label="Equal performance")
        ax.set_xlabel(f"Classical {metric_name}")
        ax.set_ylabel(f"Quantum {metric_name}")
        ax.set_title("Paired Samples")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Violin plot
        ax = axes[1, 1]
        parts = ax.violinplot(
            [quantum_scores, classical_scores], positions=[1, 2], showmeans=True, showmedians=True
        )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Quantum", "Classical"])
        ax.set_ylabel(metric_name)
        ax.set_title("Distribution Shape")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(
                self.output_dir / f"{metric_name}_comparison.png", dpi=300, bbox_inches="tight"
            )

        plt.close()

    def save_results(self, filename: str = "statistical_validation_results.json"):
        """Save all statistical results to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    def generate_report(self, filename: str = "statistical_report.txt"):
        """Generate human-readable statistical report."""
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("STATISTICAL VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            for test in self.results["tests"]:
                f.write(f"Metric: {test['metric']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Sample size: {test['n_samples']}\n")
                f.write(f"Quantum mean: {test['quantum_mean']:.4f} ± {test['quantum_std']:.4f}\n")
                f.write(
                    f"Classical mean: {test['classical_mean']:.4f} ± {test['classical_std']:.4f}\n\n"
                )

                f.write("Wilcoxon Test:\n")
                f.write(f"  P-value: {test['wilcoxon']['p_value']:.4e}\n")
                f.write(f"  Significant: {test['wilcoxon']['significant']}\n")
                f.write(f"  {test['wilcoxon']['interpretation']}\n\n")

                f.write("Paired t-Test:\n")
                f.write(f"  P-value: {test['ttest']['p_value']:.4e}\n")
                f.write(f"  Cohen's d: {test['ttest']['effect_size']:.4f}\n")
                f.write(f"  {test['ttest']['interpretation']}\n\n")

                f.write(f"Statistical Power: {test['statistical_power']:.4f}\n")
                f.write("\n" + "=" * 80 + "\n\n")

        print(f"Report saved to {output_path}")
