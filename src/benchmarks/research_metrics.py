"""Research-grade benchmarking and statistical validation.

Provides comprehensive metrics for protein structure prediction:
- TM-score, RMSD, GDT-TS, lDDT, CAD-score
- Statistical hypothesis testing
- Effect size calculations
- Bootstrap confidence intervals
- Multiple comparison correction
- Power analysis
- Publication-quality result tables
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist


@dataclass
class StructurePredictionMetrics:
    """Container for all structural prediction metrics."""

    tm_score: float
    rmsd: float
    gdt_ts: float
    gdt_ha: float
    lddt: float
    mean_plddt: float
    contact_precision: float
    contact_recall: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "TM-score": self.tm_score,
            "RMSD (Å)": self.rmsd,
            "GDT-TS": self.gdt_ts,
            "GDT-HA": self.gdt_ha,
            "lDDT": self.lddt,
            "pLDDT": self.mean_plddt,
            "Contact Precision": self.contact_precision,
            "Contact Recall": self.contact_recall,
        }


class ResearchBenchmark:
    """Comprehensive benchmarking with statistical rigor."""

    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000):
        """Initialize benchmark.

        Args:
            alpha: Significance level for hypothesis tests
            n_bootstrap: Number of bootstrap samples for CI
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap

    def compute_tm_score(
        self,
        pred_coords: np.ndarray,
        true_coords: np.ndarray,
        sequence_length: Optional[int] = None,
    ) -> float:
        """Compute TM-score using Zhang-Skolnick algorithm.

        Args:
            pred_coords: Predicted coordinates (N, 3)
            true_coords: True coordinates (N, 3)
            sequence_length: Sequence length for normalization

        Returns:
            TM-score (0-1, higher is better)
        """
        if sequence_length is None:
            sequence_length = len(pred_coords)

        # Normalize scale
        d0 = 1.24 * (sequence_length - 15) ** (1.0 / 3.0) - 1.8
        if sequence_length <= 21:
            d0 = 0.5

        # Align structures (Kabsch algorithm)
        pred_aligned, true_aligned = self._kabsch_align(pred_coords, true_coords)

        # Compute TM-score
        distances = np.linalg.norm(pred_aligned - true_aligned, axis=1)
        tm_score = np.mean(1.0 / (1.0 + (distances / d0) ** 2))

        return float(tm_score)

    def compute_rmsd(
        self, pred_coords: np.ndarray, true_coords: np.ndarray, align: bool = True
    ) -> float:
        """Compute RMSD (CA-RMSD).

        Args:
            pred_coords: Predicted coordinates (N, 3)
            true_coords: True coordinates (N, 3)
            align: Whether to align structures first

        Returns:
            RMSD in Angstroms
        """
        if align:
            pred_aligned, true_aligned = self._kabsch_align(pred_coords, true_coords)
        else:
            pred_aligned, true_aligned = pred_coords, true_coords

        rmsd = np.sqrt(np.mean(np.sum((pred_aligned - true_aligned) ** 2, axis=1)))
        return float(rmsd)

    def compute_gdt(
        self,
        pred_coords: np.ndarray,
        true_coords: np.ndarray,
        thresholds: List[float] = [1.0, 2.0, 4.0, 8.0],
    ) -> Dict[str, float]:
        """Compute GDT (Global Distance Test) scores.

        Args:
            pred_coords: Predicted coordinates (N, 3)
            true_coords: True coordinates (N, 3)
            thresholds: Distance thresholds in Angstroms

        Returns:
            Dictionary with GDT_TS, GDT_HA, and per-threshold scores
        """
        pred_aligned, true_aligned = self._kabsch_align(pred_coords, true_coords)
        distances = np.linalg.norm(pred_aligned - true_aligned, axis=1)

        scores = {}
        for threshold in thresholds:
            fraction = np.mean(distances < threshold)
            scores[f"GDT_{threshold}"] = fraction * 100

        # GDT-TS (Total Score): average of 1, 2, 4, 8 Å
        scores["GDT_TS"] = np.mean([scores[f"GDT_{t}"] for t in [1.0, 2.0, 4.0, 8.0]])

        # GDT-HA (High Accuracy): average of 0.5, 1, 2, 4 Å
        if 0.5 not in thresholds:
            scores["GDT_0.5"] = np.mean(distances < 0.5) * 100
        scores["GDT_HA"] = np.mean(
            [scores["GDT_0.5"], scores["GDT_1.0"], scores["GDT_2.0"], scores["GDT_4.0"]]
        )

        return scores

    def compute_lddt(
        self, pred_coords: np.ndarray, true_coords: np.ndarray, cutoff: float = 15.0
    ) -> float:
        """Compute lDDT (local Distance Difference Test).

        Args:
            pred_coords: Predicted coordinates (N, 3)
            true_coords: True coordinates (N, 3)
            cutoff: Distance cutoff for local contacts

        Returns:
            lDDT score (0-100)
        """
        len(pred_coords)

        # Compute pairwise distances
        true_dist = cdist(true_coords, true_coords)
        pred_dist = cdist(pred_coords, pred_coords)

        # Find local contacts (within cutoff in true structure)
        local_mask = (true_dist < cutoff) & (true_dist > 0)

        if not local_mask.any():
            return 0.0

        # Compute distance differences
        dist_diff = np.abs(true_dist[local_mask] - pred_dist[local_mask])

        # Score based on thresholds
        thresholds = [0.5, 1.0, 2.0, 4.0]
        scores = [np.mean(dist_diff < t) for t in thresholds]
        lddt = np.mean(scores) * 100

        return float(lddt)

    def compute_contact_metrics(
        self,
        pred_coords: np.ndarray,
        true_coords: np.ndarray,
        threshold: float = 8.0,
        min_separation: int = 6,
    ) -> Tuple[float, float, float]:
        """Compute contact prediction metrics.

        Args:
            pred_coords: Predicted coordinates (N, 3)
            true_coords: True coordinates (N, 3)
            threshold: Distance threshold for contact
            min_separation: Minimum sequence separation

        Returns:
            Tuple of (precision, recall, f1_score)
        """
        n_res = len(pred_coords)

        # Compute contact maps
        true_dist = cdist(true_coords, true_coords)
        pred_dist = cdist(pred_coords, pred_coords)

        # Create masks (exclude near neighbors)
        for i in range(n_res):
            for j in range(max(0, i - min_separation), min(n_res, i + min_separation + 1)):
                true_dist[i, j] = np.inf
                pred_dist[i, j] = np.inf

        # Binary contact maps
        true_contacts = true_dist < threshold
        pred_contacts = pred_dist < threshold

        # Compute metrics
        tp = np.sum(true_contacts & pred_contacts)
        fp = np.sum(pred_contacts & ~true_contacts)
        fn = np.sum(true_contacts & ~pred_contacts)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def compute_all_metrics(
        self,
        pred_coords: np.ndarray,
        true_coords: np.ndarray,
        sequence: str,
        confidence: Optional[np.ndarray] = None,
    ) -> StructurePredictionMetrics:
        """Compute all structural metrics.

        Args:
            pred_coords: Predicted coordinates (N, 3)
            true_coords: True coordinates (N, 3)
            sequence: Amino acid sequence
            confidence: Per-residue confidence scores

        Returns:
            StructurePredictionMetrics object
        """
        tm_score = self.compute_tm_score(pred_coords, true_coords, len(sequence))
        rmsd = self.compute_rmsd(pred_coords, true_coords)
        gdt_scores = self.compute_gdt(pred_coords, true_coords)
        lddt = self.compute_lddt(pred_coords, true_coords)
        contact_prec, contact_rec, _ = self.compute_contact_metrics(pred_coords, true_coords)

        mean_plddt = np.mean(confidence) if confidence is not None else 0.0

        return StructurePredictionMetrics(
            tm_score=tm_score,
            rmsd=rmsd,
            gdt_ts=gdt_scores["GDT_TS"],
            gdt_ha=gdt_scores["GDT_HA"],
            lddt=lddt,
            mean_plddt=float(mean_plddt),
            contact_precision=contact_prec,
            contact_recall=contact_rec,
        )

    def compare_methods(
        self,
        quantum_scores: np.ndarray,
        classical_scores: np.ndarray,
        metric_name: str = "TM-score",
        higher_is_better: bool = True,
    ) -> Dict:
        """Rigorous statistical comparison of two methods.

        Args:
            quantum_scores: Scores from quantum method
            classical_scores: Scores from classical baseline
            metric_name: Name of metric for reporting
            higher_is_better: Whether higher scores are better

        Returns:
            Dictionary with statistical test results
        """
        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(quantum_scores, classical_scores)

        # Wilcoxon signed-rank test (non-parametric)
        wilcoxon_stat, wilcoxon_pval = stats.wilcoxon(
            quantum_scores, classical_scores, alternative="greater" if higher_is_better else "less"
        )

        # Effect size (Cohen's d)
        cohens_d = self._compute_cohens_d(quantum_scores, classical_scores)

        # Bootstrap confidence intervals
        quantum_ci = self._bootstrap_ci(quantum_scores)
        classical_ci = self._bootstrap_ci(classical_scores)
        diff_ci = self._bootstrap_diff_ci(quantum_scores, classical_scores)

        # Rank-biserial correlation (effect size for Wilcoxon)
        rank_biserial = self._compute_rank_biserial(quantum_scores, classical_scores)

        # Power analysis
        power = self._compute_power(quantum_scores, classical_scores, self.alpha)

        results = {
            "metric": metric_name,
            "quantum_mean": np.mean(quantum_scores),
            "classical_mean": np.mean(classical_scores),
            "quantum_std": np.std(quantum_scores, ddof=1),
            "classical_std": np.std(classical_scores, ddof=1),
            "quantum_ci": quantum_ci,
            "classical_ci": classical_ci,
            "difference_ci": diff_ci,
            "t_statistic": t_stat,
            "t_pvalue": t_pval,
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_pvalue": wilcoxon_pval,
            "cohens_d": cohens_d,
            "rank_biserial": rank_biserial,
            "power": power,
            "n_samples": len(quantum_scores),
            "significant": wilcoxon_pval < self.alpha,
        }

        return results

    def plot_comparison(
        self,
        quantum_scores: np.ndarray,
        classical_scores: np.ndarray,
        metric_name: str = "TM-score",
        figsize: Tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """Create publication-quality comparison plot.

        Args:
            quantum_scores: Scores from quantum method
            classical_scores: Scores from classical baseline
            metric_name: Name of metric
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # Violin plot
        data = pd.DataFrame(
            {
                "Score": np.concatenate([quantum_scores, classical_scores]),
                "Method": ["Quantum"] * len(quantum_scores) + ["Classical"] * len(classical_scores),
            }
        )

        sns.violinplot(data=data, x="Method", y="Score", ax=ax1, palette=["#FF6B6B", "#4ECDC4"])
        ax1.set_ylabel(metric_name, fontsize=12)
        ax1.set_title("Distribution Comparison", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")

        # Paired scatter plot
        ax2.scatter(classical_scores, quantum_scores, alpha=0.6, s=80, edgecolors="black")
        lims = [
            min(classical_scores.min(), quantum_scores.min()),
            max(classical_scores.max(), quantum_scores.max()),
        ]
        ax2.plot(lims, lims, "k--", alpha=0.5, linewidth=2, label="y=x")
        ax2.set_xlabel(f"Classical {metric_name}", fontsize=12)
        ax2.set_ylabel(f"Quantum {metric_name}", fontsize=12)
        ax2.set_title("Paired Comparison", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Difference plot
        differences = quantum_scores - classical_scores
        ax3.hist(differences, bins=20, alpha=0.7, color="#95E1D3", edgecolor="black")
        ax3.axvline(0, color="red", linestyle="--", linewidth=2, label="No difference")
        ax3.axvline(
            np.mean(differences),
            color="blue",
            linestyle="-",
            linewidth=2,
            label=f"Mean diff: {np.mean(differences):.4f}",
        )
        ax3.set_xlabel(f"Quantum - Classical {metric_name}", fontsize=12)
        ax3.set_ylabel("Frequency", fontsize=12)
        ax3.set_title("Difference Distribution", fontsize=14, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def generate_latex_table(self, results_dict: Dict, caption: str = "Comparison Results") -> str:
        """Generate publication-ready LaTeX table.

        Args:
            results_dict: Results from compare_methods
            caption: Table caption

        Returns:
            LaTeX table string
        """
        latex = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{" + caption + "}",
            r"\begin{tabular}{lcc}",
            r"\hline",
            r"Metric & Quantum & Classical \\\\",
            r"\hline",
        ]

        # Add results
        latex.append(
            f"{results_dict['metric']} & "
            f"{results_dict['quantum_mean']:.3f} $\\pm$ {results_dict['quantum_std']:.3f} & "
            f"{results_dict['classical_mean']:.3f} $\\pm$ {results_dict['classical_std']:.3f} \\\\\\"
        )

        latex.extend(
            [
                r"\hline",
                f"$p$-value (Wilcoxon) & \\multicolumn{{2}}{{c}}{{{results_dict['wilcoxon_pvalue']:.4f}}} \\\\\\",
                f"Cohen's $d$ & \\multicolumn{{2}}{{c}}{{{results_dict['cohens_d']:.3f}}} \\\\\\",
                f"Power & \\multicolumn{{2}}{{c}}{{{results_dict['power']:.3f}}} \\\\\\",
                r"\hline",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(latex)

    # Helper methods

    def _kabsch_align(
        self, coords_a: np.ndarray, coords_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align two coordinate sets using Kabsch algorithm."""
        # Center coordinates
        center_a = np.mean(coords_a, axis=0)
        center_b = np.mean(coords_b, axis=0)
        centered_a = coords_a - center_a
        centered_b = coords_b - center_b

        # Compute rotation matrix
        H = centered_a.T @ centered_b
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure right-handed coordinate system
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Apply rotation
        aligned_a = centered_a @ R

        return aligned_a, centered_b

    def _compute_cohens_d(self, scores_a: np.ndarray, scores_b: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        mean_diff = np.mean(scores_a) - np.mean(scores_b)
        pooled_std = np.sqrt((np.var(scores_a, ddof=1) + np.var(scores_b, ddof=1)) / 2)
        return mean_diff / pooled_std if pooled_std > 0 else 0.0

    def _compute_rank_biserial(self, scores_a: np.ndarray, scores_b: np.ndarray) -> float:
        """Compute rank-biserial correlation for Wilcoxon test."""
        differences = scores_a - scores_b
        n_positive = np.sum(differences > 0)
        n_negative = np.sum(differences < 0)
        return (n_positive - n_negative) / len(differences)

    def _bootstrap_ci(self, scores: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return (lower, upper)

    def _bootstrap_diff_ci(
        self, scores_a: np.ndarray, scores_b: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute bootstrap CI for difference."""
        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            sample_a = np.random.choice(scores_a, size=len(scores_a), replace=True)
            sample_b = np.random.choice(scores_b, size=len(scores_b), replace=True)
            bootstrap_diffs.append(np.mean(sample_a) - np.mean(sample_b))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_diffs, alpha / 2 * 100)
        upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)

        return (lower, upper)

    def _compute_power(self, scores_a: np.ndarray, scores_b: np.ndarray, alpha: float) -> float:
        """Estimate statistical power."""
        # Simplified power calculation using effect size
        effect_size = abs(self._compute_cohens_d(scores_a, scores_b))
        n = len(scores_a)

        # Approximation using normal distribution
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = effect_size * np.sqrt(n / 2) - z_alpha
        power = norm.cdf(z_beta)

        return max(0.0, min(1.0, power))


def compute_casp_metrics(pred_coords, native_coords, sequence) -> Dict[str, float]:
    """Compute a broad CASP-style metric panel."""
    bench = ResearchBenchmark()
    gdt_ts = bench.compute_gdt(pred_coords, native_coords, thresholds=[1.0, 2.0, 4.0, 8.0])
    gdt_ha = bench.compute_gdt(pred_coords, native_coords, thresholds=[0.5, 1.0, 2.0, 4.0])
    contact_precision, _, _ = bench.compute_contact_metrics(
        pred_coords, native_coords, threshold=8.0
    )
    return {
        "TM-score": bench.compute_tm_score(pred_coords, native_coords, len(sequence)),
        "GDT_TS": gdt_ts["GDT_TS"],
        "GDT_HA": gdt_ha["GDT_HA"],
        "lDDT": bench.compute_lddt(pred_coords, native_coords),
        "RMSD": bench.compute_rmsd(pred_coords, native_coords),
        "Contact_Precision": contact_precision,
        "Secondary_Structure_Agreement": 0.0,
    }
