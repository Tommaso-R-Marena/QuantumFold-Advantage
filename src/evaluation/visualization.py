"""Visualization utilities for quantum vs. classical comparison.

Generates publication-quality plots for comparing model performance.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.evaluation.statistical_tests import ComparisonResult

sns.set_theme(style="whitegrid", font_scale=1.1)
QUANTUM_COLOR = "#7B2FBE"
CLASSICAL_COLOR = "#2196F3"


def plot_metric_comparison(
    quantum_vals: np.ndarray,
    classical_vals: np.ndarray,
    metric_name: str,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side violin/box plot for a single metric."""
    fig, ax = plt.subplots(figsize=(6, 5))
    data = {
        "Quantum-Enhanced": quantum_vals,
        "Classical Baseline": classical_vals,
    }
    positions = [0, 1]
    colors = [QUANTUM_COLOR, CLASSICAL_COLOR]

    parts = ax.violinplot(
        [quantum_vals, classical_vals], positions=positions, showmeans=True, showmedians=True
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Quantum-Enhanced", "Classical Baseline"])
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Comparison")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(
    quantum_history: Dict[str, List[float]],
    classical_history: Dict[str, List[float]],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Training loss and TM-score curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs_q = range(1, len(quantum_history.get("train_loss", [])) + 1)
    epochs_c = range(1, len(classical_history.get("train_loss", [])) + 1)

    # Loss
    ax = axes[0]
    if "train_loss" in quantum_history:
        ax.plot(epochs_q, quantum_history["train_loss"], color=QUANTUM_COLOR, label="Quantum (train)")
    if "test_loss" in quantum_history:
        ax.plot(epochs_q, quantum_history["test_loss"], color=QUANTUM_COLOR, linestyle="--", label="Quantum (val)")
    if "train_loss" in classical_history:
        ax.plot(epochs_c, classical_history["train_loss"], color=CLASSICAL_COLOR, label="Classical (train)")
    if "test_loss" in classical_history:
        ax.plot(epochs_c, classical_history["test_loss"], color=CLASSICAL_COLOR, linestyle="--", label="Classical (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend(fontsize=8)

    # TM-score
    ax = axes[1]
    if "test_tm_score" in quantum_history:
        ax.plot(epochs_q, quantum_history["test_tm_score"], color=QUANTUM_COLOR, label="Quantum")
    if "test_tm_score" in classical_history:
        ax.plot(epochs_c, classical_history["test_tm_score"], color=CLASSICAL_COLOR, label="Classical")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("TM-score")
    ax.set_title("Validation TM-score")
    ax.legend()

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_protein_improvement(
    protein_names: List[str],
    quantum_scores: np.ndarray,
    classical_scores: np.ndarray,
    metric_name: str = "TM-score",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart showing per-target improvement."""
    diff = quantum_scores - classical_scores
    order = np.argsort(diff)[::-1]

    fig, ax = plt.subplots(figsize=(max(8, len(protein_names) * 0.5), 5))
    colors = [QUANTUM_COLOR if d > 0 else CLASSICAL_COLOR for d in diff[order]]
    ax.bar(range(len(diff)), diff[order], color=colors, alpha=0.8)
    ax.set_xticks(range(len(diff)))
    ax.set_xticklabels([protein_names[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"Δ {metric_name} (Quantum − Classical)")
    ax.set_title(f"Per-Target {metric_name} Improvement")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_effect_size_forest(
    results: List[ComparisonResult],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Forest plot of Cohen's d effect sizes with bootstrap CIs."""
    fig, ax = plt.subplots(figsize=(8, max(3, len(results) * 0.8)))

    names = [r.metric_name for r in results]
    ds = [r.cohens_d for r in results]
    ci_lo = [r.ci_lower for r in results]
    ci_hi = [r.ci_upper for r in results]

    y_pos = range(len(results))
    ax.barh(y_pos, ds, color=QUANTUM_COLOR, alpha=0.6, height=0.5)
    ax.errorbar(ds, y_pos, xerr=[
        [d - lo for d, lo in zip(ds, ci_lo)],
        [hi - d for d, hi in zip(ds, ci_hi)],
    ], fmt="o", color="black", capsize=4)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names)
    ax.set_xlabel("Cohen's d  (positive = quantum better)")
    ax.set_title("Effect Size Forest Plot")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_bootstrap_distribution(
    boot_diffs: np.ndarray,
    observed_diff: float,
    metric_name: str = "Metric",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of bootstrapped mean differences."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(boot_diffs, bins=60, color=QUANTUM_COLOR, alpha=0.6, edgecolor="white")
    ax.axvline(observed_diff, color="red", linewidth=2, label=f"Observed = {observed_diff:.4f}")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(f"Δ {metric_name}")
    ax.set_ylabel("Count")
    ax.set_title(f"Bootstrap Distribution of Mean Difference ({metric_name})")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
