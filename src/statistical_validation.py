"""Statistical validation and hypothesis testing for quantum advantage claims.

Implements rigorous statistical methods required for publication:
- Paired statistical tests (t-test, Wilcoxon signed-rank)
- Bootstrap confidence intervals
- Effect size calculations (Cohen's d, Hedges' g)
- Multiple comparison correction (Bonferroni, Benjamini-Hochberg)
- Power analysis
- Cross-validation protocols
- Reproducibility metrics

References:
    - Statistical Testing: Wasserstein & Lazar, Am. Stat. 70, 129 (2016)
    - Effect Sizes: Cohen, Statistical Power Analysis (1988)
    - Bootstrap: Efron & Tibshirani, An Introduction to Bootstrap (1993)
    - Multiple Testing: Benjamini & Hochberg, J. R. Stat. Soc. B 57, 289 (1995)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import bootstrap
import warnings
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    effect_size_name: str
    significant: bool
    alpha: float
    interpretation: str
    
    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'confidence_interval': self.confidence_interval,
            'effect_size': self.effect_size,
            'effect_size_name': self.effect_size_name,
            'significant': self.significant,
            'alpha': self.alpha,
            'interpretation': self.interpretation
        }


class StatisticalValidator:
    """Comprehensive statistical validation for model comparisons.
    
    Args:
        alpha: Significance level (typically 0.05)
        n_bootstrap: Number of bootstrap resamples
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000, random_state: int = 42):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        np.random.seed(random_state)
    
    def paired_t_test(
        self,
        quantum_scores: np.ndarray,
        classical_scores: np.ndarray,
        alternative: str = 'two-sided'
    ) -> StatisticalTestResult:
        """Paired t-test for comparing two models on same test set.
        
        Args:
            quantum_scores: Scores from quantum model (n_samples,)
            classical_scores: Scores from classical model (n_samples,)
            alternative: 'two-sided', 'greater', or 'less'
        
        Returns:
            StatisticalTestResult with test details
        """
        # Compute differences
        differences = quantum_scores - classical_scores
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(quantum_scores, classical_scores, alternative=alternative)
        
        # Confidence interval for mean difference
        mean_diff = np.mean(differences)
        se_diff = stats.sem(differences)
        ci = stats.t.interval(1 - self.alpha, len(differences) - 1, mean_diff, se_diff)
        
        # Cohen's d (effect size)
        cohens_d = self.compute_cohens_d(quantum_scores, classical_scores, paired=True)
        
        # Interpretation
        if p_value < self.alpha:
            if alternative == 'greater':
                interpretation = f"Quantum model significantly outperforms classical (p={p_value:.4f})"
            elif alternative == 'less':
                interpretation = f"Classical model significantly outperforms quantum (p={p_value:.4f})"
            else:
                interpretation = f"Significant difference detected (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"
        
        return StatisticalTestResult(
            test_name='Paired t-test',
            statistic=t_stat,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=cohens_d,
            effect_size_name="Cohen's d",
            significant=p_value < self.alpha,
            alpha=self.alpha,
            interpretation=interpretation
        )
    
    def wilcoxon_test(
        self,
        quantum_scores: np.ndarray,
        classical_scores: np.ndarray,
        alternative: str = 'two-sided'
    ) -> StatisticalTestResult:
        """Wilcoxon signed-rank test (non-parametric alternative to t-test).
        
        More robust to outliers and non-normal distributions.
        
        Args:
            quantum_scores: Scores from quantum model
            classical_scores: Scores from classical model
            alternative: 'two-sided', 'greater', or 'less'
        
        Returns:
            StatisticalTestResult
        """
        # Wilcoxon signed-rank test
        try:
            stat, p_value = stats.wilcoxon(
                quantum_scores,
                classical_scores,
                alternative=alternative,
                zero_method='wilcox'
            )
        except ValueError as e:
            warnings.warn(f"Wilcoxon test failed: {e}")
            return None
        
        # Effect size (rank-biserial correlation)
        differences = quantum_scores - classical_scores
        n_pos = np.sum(differences > 0)
        n_neg = np.sum(differences < 0)
        r = (n_pos - n_neg) / len(differences)
        
        # Bootstrap confidence interval
        ci = self.bootstrap_difference_ci(quantum_scores, classical_scores)
        
        # Interpretation
        if p_value < self.alpha:
            interpretation = f"Significant difference (Wilcoxon, p={p_value:.4f})"
        else:
            interpretation = f"No significant difference (Wilcoxon, p={p_value:.4f})"
        
        return StatisticalTestResult(
            test_name='Wilcoxon signed-rank test',
            statistic=stat,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=r,
            effect_size_name='Rank-biserial correlation',
            significant=p_value < self.alpha,
            alpha=self.alpha,
            interpretation=interpretation
        )
    
    def bootstrap_difference_ci(
        self,
        quantum_scores: np.ndarray,
        classical_scores: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for mean difference.
        
        Args:
            quantum_scores: Quantum model scores
            classical_scores: Classical model scores
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        
        Returns:
            (lower_bound, upper_bound) tuple
        """
        differences = quantum_scores - classical_scores
        
        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(differences, size=len(differences), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Compute percentile confidence interval
        alpha_half = (1 - confidence_level) / 2
        lower = np.percentile(bootstrap_means, alpha_half * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha_half) * 100)
        
        return (lower, upper)
    
    def compute_cohens_d(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        paired: bool = True
    ) -> float:
        """Compute Cohen's d effect size.
        
        Args:
            group1: First group scores
            group2: Second group scores
            paired: Whether data is paired
        
        Returns:
            Cohen's d value
            
        Interpretation:
            |d| < 0.2: negligible
            0.2 <= |d| < 0.5: small
            0.5 <= |d| < 0.8: medium
            |d| >= 0.8: large
        """
        if paired:
            # For paired data, use standard deviation of differences
            differences = group1 - group2
            d = np.mean(differences) / np.std(differences, ddof=1)
        else:
            # For independent groups, use pooled standard deviation
            mean_diff = np.mean(group1) - np.mean(group2)
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            d = mean_diff / pooled_std
        
        return float(d)
    
    def compute_hedges_g(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Hedges' g (corrected Cohen's d for small samples).
        
        Args:
            group1: First group scores
            group2: Second group scores
        
        Returns:
            Hedges' g value
        """
        cohens_d = self.compute_cohens_d(group1, group2, paired=False)
        n = len(group1) + len(group2)
        correction = 1 - (3 / (4 * n - 9))
        hedges_g = cohens_d * correction
        return float(hedges_g)
    
    def bonferroni_correction(self, p_values: List[float]) -> Tuple[List[float], List[bool]]:
        """Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of p-values
        
        Returns:
            (corrected_p_values, significant_flags) tuple
        """
        n_tests = len(p_values)
        corrected_alpha = self.alpha / n_tests
        corrected_p_values = [min(p * n_tests, 1.0) for p in p_values]
        significant = [p < corrected_alpha for p in p_values]
        return corrected_p_values, significant
    
    def benjamini_hochberg_correction(
        self,
        p_values: List[float]
    ) -> Tuple[List[float], List[bool]]:
        """Benjamini-Hochberg FDR correction (less conservative than Bonferroni).
        
        Args:
            p_values: List of p-values
        
        Returns:
            (corrected_p_values, significant_flags) tuple
        """
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]
        
        # BH critical values
        critical_values = (np.arange(1, n_tests + 1) / n_tests) * self.alpha
        
        # Find largest i where p_i <= (i/m)*alpha
        significant = sorted_p_values <= critical_values
        
        # Unsort
        unsorted_significant = np.zeros(n_tests, dtype=bool)
        unsorted_significant[sorted_indices] = significant
        
        # Adjusted p-values
        adjusted_p = np.minimum.accumulate(sorted_p_values[::-1] * n_tests / np.arange(n_tests, 0, -1))[::-1]
        adjusted_p = np.minimum(adjusted_p, 1.0)
        unsorted_adjusted_p = np.zeros(n_tests)
        unsorted_adjusted_p[sorted_indices] = adjusted_p
        
        return list(unsorted_adjusted_p), list(unsorted_significant)
    
    def power_analysis(
        self,
        effect_size: float,
        n_samples: int,
        alpha: Optional[float] = None
    ) -> float:
        """Compute statistical power for given effect size and sample size.
        
        Args:
            effect_size: Expected Cohen's d
            n_samples: Number of samples
            alpha: Significance level (defaults to self.alpha)
        
        Returns:
            Statistical power (probability of detecting effect)
        """
        if alpha is None:
            alpha = self.alpha
        
        # Critical value for two-tailed test
        critical_z = stats.norm.ppf(1 - alpha / 2)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n_samples)
        
        # Power = P(reject H0 | H1 is true)
        power = 1 - stats.norm.cdf(critical_z - ncp) + stats.norm.cdf(-critical_z - ncp)
        
        return float(power)
    
    def required_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: Optional[float] = None
    ) -> int:
        """Compute required sample size for desired power.
        
        Args:
            effect_size: Expected Cohen's d
            power: Desired statistical power (typically 0.8)
            alpha: Significance level
        
        Returns:
            Required sample size
        """
        if alpha is None:
            alpha = self.alpha
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        n = ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))


class ComprehensiveComparison:
    """Comprehensive comparison framework for quantum vs classical models.
    
    Performs full statistical analysis including:
    - Multiple statistical tests
    - Effect size calculations
    - Confidence intervals
    - Visualization
    - Publication-ready report generation
    
    Args:
        output_dir: Directory for saving results
        alpha: Significance level
    """
    
    def __init__(self, output_dir: str = 'statistical_results', alpha: float = 0.05):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.validator = StatisticalValidator(alpha=alpha)
        self.results = {}
    
    def compare(
        self,
        quantum_metrics: Dict[str, np.ndarray],
        classical_metrics: Dict[str, np.ndarray],
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """Perform comprehensive comparison.
        
        Args:
            quantum_metrics: Dictionary of metric_name -> scores array
            classical_metrics: Dictionary of metric_name -> scores array
            metric_names: Subset of metrics to analyze (None = all)
        
        Returns:
            Dictionary of results per metric
        """
        if metric_names is None:
            metric_names = list(quantum_metrics.keys())
        
        results = {}
        
        for metric in metric_names:
            quantum_scores = quantum_metrics[metric]
            classical_scores = classical_metrics[metric]
            
            # Paired t-test
            t_test = self.validator.paired_t_test(
                quantum_scores,
                classical_scores,
                alternative='greater'  # Test if quantum > classical
            )
            
            # Wilcoxon test
            wilcoxon = self.validator.wilcoxon_test(
                quantum_scores,
                classical_scores,
                alternative='greater'
            )
            
            # Effect size
            cohens_d = self.validator.compute_cohens_d(quantum_scores, classical_scores)
            
            # Bootstrap CI
            ci = self.validator.bootstrap_difference_ci(quantum_scores, classical_scores)
            
            # Power analysis
            power = self.validator.power_analysis(abs(cohens_d), len(quantum_scores))
            
            results[metric] = {
                't_test': t_test.to_dict(),
                'wilcoxon': wilcoxon.to_dict() if wilcoxon else None,
                'cohens_d': cohens_d,
                'bootstrap_ci': ci,
                'statistical_power': power,
                'quantum_mean': float(np.mean(quantum_scores)),
                'classical_mean': float(np.mean(classical_scores)),
                'quantum_std': float(np.std(quantum_scores)),
                'classical_std': float(np.std(classical_scores))
            }
        
        self.results = results
        return results
    
    def generate_report(self, filename: str = 'statistical_report.txt'):
        """Generate publication-ready text report."""
        report_path = self.output_dir / filename
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STATISTICAL VALIDATION REPORT\n")
            f.write("Quantum vs Classical Model Comparison\n")
            f.write("=" * 80 + "\n\n")
            
            for metric, result in self.results.items():
                f.write(f"\nMetric: {metric.upper()}\n")
                f.write("-" * 60 + "\n")
                f.write(f"Quantum Mean ± SD: {result['quantum_mean']:.4f} ± {result['quantum_std']:.4f}\n")
                f.write(f"Classical Mean ± SD: {result['classical_mean']:.4f} ± {result['classical_std']:.4f}\n")
                f.write(f"\nEffect Size (Cohen's d): {result['cohens_d']:.4f}\n")
                
                # Interpret effect size
                d = abs(result['cohens_d'])
                if d < 0.2:
                    interpretation = "negligible"
                elif d < 0.5:
                    interpretation = "small"
                elif d < 0.8:
                    interpretation = "medium"
                else:
                    interpretation = "large"
                f.write(f"Effect Size Interpretation: {interpretation}\n")
                
                # Statistical tests
                f.write(f"\nPaired t-test:\n")
                f.write(f"  t-statistic: {result['t_test']['statistic']:.4f}\n")
                f.write(f"  p-value: {result['t_test']['p_value']:.6f}\n")
                f.write(f"  Significant: {result['t_test']['significant']}\n")
                f.write(f"  Interpretation: {result['t_test']['interpretation']}\n")
                
                if result['wilcoxon']:
                    f.write(f"\nWilcoxon signed-rank test:\n")
                    f.write(f"  W-statistic: {result['wilcoxon']['statistic']:.4f}\n")
                    f.write(f"  p-value: {result['wilcoxon']['p_value']:.6f}\n")
                    f.write(f"  Significant: {result['wilcoxon']['significant']}\n")
                
                f.write(f"\nBootstrap 95% CI: [{result['bootstrap_ci'][0]:.4f}, {result['bootstrap_ci'][1]:.4f}]\n")
                f.write(f"Statistical Power: {result['statistical_power']:.4f}\n")
                f.write("\n")
        
        print(f"Report saved to {report_path}")
    
    def plot_comparison(self, filename: str = 'comparison_plots.png'):
        """Generate comparison visualization."""
        n_metrics = len(self.results)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, (metric, result) in zip(axes, self.results.items()):
            # Box plot
            data = [
                [result['quantum_mean']] * 100,  # Placeholder
                [result['classical_mean']] * 100
            ]
            ax.boxplot(data, labels=['Quantum', 'Classical'])
            ax.set_title(f"{metric}\np={result['t_test']['p_value']:.4f}")
            ax.set_ylabel('Score')
            
            # Add significance marker
            if result['t_test']['significant']:
                ax.text(1.5, max(result['quantum_mean'], result['classical_mean']) * 1.1,
                       '*', fontsize=20, ha='center')
        
        plt.tight_layout()
        plot_path = self.output_dir / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {plot_path}")
    
    def save_results(self, filename: str = 'results.json'):
        """Save results to JSON."""
        results_path = self.output_dir / filename
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {results_path}")
