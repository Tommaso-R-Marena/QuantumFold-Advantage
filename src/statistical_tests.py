"""Rigorous statistical validation and hypothesis testing."""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""


class StatisticalValidator:
    """Perform rigorous statistical validation."""

    def __init__(self, alpha: float = 0.05):
        """
        Initialize validator.

        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha

    def paired_t_test(self, method_a: List[float], method_b: List[float],
                     method_a_name: str = "Method A",
                     method_b_name: str = "Method B") -> StatisticalTestResult:
        """Perform paired t-test to compare two methods.

        Args:
            method_a: Metric values for method A
            method_b: Metric values for method B
            method_a_name: Name of method A
            method_b_name: Name of method B

        Returns:
            StatisticalTestResult object
        """
        statistic, p_value = stats.ttest_rel(method_a, method_b)

        # Compute Cohen's d for effect size
        diff = np.array(method_a) - np.array(method_b)
        effect_size = np.mean(diff) / np.std(diff)

        # Confidence interval for mean difference
        ci = stats.t.interval(1 - self.alpha, len(diff) - 1,
                             loc=np.mean(diff), scale=stats.sem(diff))

        significant = p_value < self.alpha

        if significant:
            if np.mean(method_a) < np.mean(method_b):
                interpretation = f"{method_a_name} is significantly better than {method_b_name} (p={p_value:.4f})"
            else:
                interpretation = f"{method_b_name} is significantly better than {method_a_name} (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference between {method_a_name} and {method_b_name} (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation
        )

    def wilcoxon_test(self, method_a: List[float], method_b: List[float],
                     method_a_name: str = "Method A",
                     method_b_name: str = "Method B") -> StatisticalTestResult:
        """Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

        Args:
            method_a: Metric values for method A
            method_b: Metric values for method B
            method_a_name: Name of method A
            method_b_name: Name of method B

        Returns:
            StatisticalTestResult object
        """
        statistic, p_value = stats.wilcoxon(method_a, method_b)

        significant = p_value < self.alpha

        if significant:
            if np.median(method_a) < np.median(method_b):
                interpretation = f"{method_a_name} is significantly better than {method_b_name} (p={p_value:.4f}, non-parametric)"
            else:
                interpretation = f"{method_b_name} is significantly better than {method_a_name} (p={p_value:.4f}, non-parametric)"
        else:
            interpretation = f"No significant difference between {method_a_name} and {method_b_name} (p={p_value:.4f}, non-parametric)"

        return StatisticalTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            interpretation=interpretation
        )

    def bootstrap_confidence_interval(self, data: List[float], n_bootstrap: int = 10000,
                                     confidence: float = 0.95) -> Tuple[float, float, float]:
        """Compute bootstrap confidence interval for mean.

        Args:
            data: Data points
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            (mean, lower_bound, upper_bound)
        """
        data = np.array(data)
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)
        mean = np.mean(data)
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return mean, lower, upper

    def cross_validation_stats(self, cv_scores: List[float]) -> Dict:
        """Compute statistics from cross-validation scores.

        Args:
            cv_scores: Scores from k-fold cross-validation

        Returns:
            Dictionary with statistics
        """
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        sem_score = stats.sem(cv_scores)

        # 95% confidence interval
        ci = stats.t.interval(0.95, len(cv_scores) - 1, loc=mean_score, scale=sem_score)

        return {
            'mean': mean_score,
            'std': std_score,
            'sem': sem_score,
            'min': np.min(cv_scores),
            'max': np.max(cv_scores),
            'cv_scores': cv_scores,
            'confidence_interval_95': ci
        }

    def multiple_testing_correction(self, p_values: List[float],
                                   method: str = 'bonferroni') -> List[bool]:
        """Apply multiple testing correction.

        Args:
            p_values: List of p-values from multiple tests
            method: Correction method ('bonferroni', 'holm', 'fdr_bh')

        Returns:
            List of booleans indicating significance after correction
        """
        from statsmodels.stats.multitest import multipletests

        reject, corrected_p, _, _ = multipletests(p_values, alpha=self.alpha, method=method)

        logger.info(f"Applied {method} correction to {len(p_values)} tests")
        logger.info(f"Significant tests: {sum(reject)}/{len(p_values)}")

        return reject.tolist()

    def effect_size_interpretation(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size.

        Args:
            cohens_d: Cohen's d value

        Returns:
            Interpretation string
        """
        d = abs(cohens_d)

        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def comprehensive_comparison(self, method_results: Dict[str, List[float]],
                                baseline_name: str) -> Dict:
        """Perform comprehensive statistical comparison against baseline.

        Args:
            method_results: Dictionary mapping method names to metric values
            baseline_name: Name of baseline method

        Returns:
            Dictionary with all test results
        """
        if baseline_name not in method_results:
            raise ValueError(f"Baseline method '{baseline_name}' not found in results")

        baseline = method_results[baseline_name]
        comparisons = {}

        for method_name, values in method_results.items():
            if method_name == baseline_name:
                continue

            # Paired t-test
            t_test = self.paired_t_test(values, baseline, method_name, baseline_name)

            # Wilcoxon test
            w_test = self.wilcoxon_test(values, baseline, method_name, baseline_name)

            # Bootstrap CI
            mean, lower, upper = self.bootstrap_confidence_interval(values)

            comparisons[method_name] = {
                't_test': t_test,
                'wilcoxon_test': w_test,
                'bootstrap_ci': (mean, lower, upper),
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values)
            }

        return comparisons

    def generate_statistical_report(self, comparisons: Dict) -> str:
        """Generate comprehensive statistical report.

        Args:
            comparisons: Output from comprehensive_comparison

        Returns:
            Formatted report string
        """
        report = []
        report.append("\n" + "=" * 80)
        report.append("STATISTICAL VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"\nSignificance level: α = {self.alpha}\n")

        for method_name, results in comparisons.items():
            report.append(f"\n{method_name}:")
            report.append("-" * 80)

            # Summary statistics
            report.append(f"  Mean: {results['mean']:.4f}")
            report.append(f"  Std:  {results['std']:.4f}")
            report.append(f"  Median: {results['median']:.4f}")

            # Bootstrap CI
            mean, lower, upper = results['bootstrap_ci']
            report.append(f"  95% Bootstrap CI: [{lower:.4f}, {upper:.4f}]")

            # t-test
            t_test = results['t_test']
            report.append(f"\n  Paired t-test:")
            report.append(f"    {t_test.interpretation}")
            report.append(f"    Effect size (Cohen's d): {t_test.effect_size:.3f} ({self.effect_size_interpretation(t_test.effect_size)})")

            # Wilcoxon
            w_test = results['wilcoxon_test']
            report.append(f"\n  Wilcoxon test:")
            report.append(f"    {w_test.interpretation}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)
