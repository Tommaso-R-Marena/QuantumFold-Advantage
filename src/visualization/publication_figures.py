"""Publication-quality figure generation for papers and presentations."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("colorblind")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,  # TrueType fonts for publication
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)


class PublicationFigures:
    """Generate publication-ready figures."""

    def __init__(self, output_dir: str = "./figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Figures will be saved to {self.output_dir}")

    def plot_benchmark_comparison(
        self, results: Dict[str, Dict], save_name: str = "benchmark_comparison"
    ):
        """Create comprehensive benchmark comparison figure.

        Args:
            results: Dictionary with method names as keys and metrics as values
            save_name: Filename for saving
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        methods = list(results.keys())
        metrics = ["rmsd", "tm_score", "gdt_ts", "gdt_ha", "lddt"]
        metric_labels = ["RMSD (Å)", "TM-score", "GDT_TS", "GDT_HA", "lDDT"]

        # Extract data
        plot_data = {}
        for metric in metrics:
            plot_data[metric] = {
                "means": [results[m]["mean"][metric] for m in methods],
                "stds": [results[m]["std"][metric] for m in methods],
            }

        # Plot each metric
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = fig.add_subplot(gs[positions[idx][0], positions[idx][1]])

            means = plot_data[metric]["means"]
            stds = plot_data[metric]["stds"]

            x = np.arange(len(methods))
            bars = ax.bar(
                x,
                means,
                yerr=stds,
                capsize=5,
                alpha=0.8,
                color=sns.color_palette("colorblind", len(methods)),
            )

            ax.set_ylabel(label, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3, linestyle="--")

            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + std,
                    f"{mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Add summary table
        ax_table = fig.add_subplot(gs[1, 2])
        ax_table.axis("off")

        table_data = []
        for method in methods:
            row = [method]
            for metric in metrics:
                val = results[method]["mean"][metric]
                row.append(f"{val:.2f}")
            table_data.append(row)

        table = ax_table.table(
            cellText=table_data,
            colLabels=["Method"] + metric_labels,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)

        # Style table
        for i in range(len(metric_labels) + 1):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        plt.suptitle(
            "Performance Comparison Across Methods", fontsize=16, fontweight="bold", y=0.98
        )

        # Save
        for ext in ["png", "pdf", "svg"]:
            plt.savefig(self.output_dir / f"{save_name}.{ext}")

        logger.info(f"Benchmark comparison saved to {self.output_dir}/{save_name}")
        plt.close()

    def plot_training_curves(
        self, history: Dict[str, List[float]], save_name: str = "training_curves"
    ):
        """Plot training and validation curves.

        Args:
            history: Dictionary with metric histories
            save_name: Filename for saving
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Training loss
        if "train_loss" in history:
            axes[0].plot(history["train_loss"], label="Training Loss", linewidth=2)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training Loss", fontweight="bold")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

        # Validation metrics
        if "val_rmsd" in history:
            axes[1].plot(history["val_rmsd"], label="Validation RMSD", linewidth=2, color="orange")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("RMSD (Å)")
            axes[1].set_title("Validation RMSD", fontweight="bold")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

        if "val_tm_score" in history:
            axes[2].plot(
                history["val_tm_score"], label="Validation TM-score", linewidth=2, color="green"
            )
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("TM-score")
            axes[2].set_title("Validation TM-score", fontweight="bold")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

        # Learning rate
        if "learning_rate" in history:
            axes[3].plot(history["learning_rate"], label="Learning Rate", linewidth=2, color="red")
            axes[3].set_xlabel("Step")
            axes[3].set_ylabel("Learning Rate")
            axes[3].set_title("Learning Rate Schedule", fontweight="bold")
            axes[3].set_yscale("log")
            axes[3].grid(True, alpha=0.3)
            axes[3].legend()

        plt.suptitle("Training Dynamics", fontsize=16, fontweight="bold")
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            plt.savefig(self.output_dir / f"{save_name}.{ext}")

        logger.info(f"Training curves saved to {self.output_dir}/{save_name}")
        plt.close()

    def plot_structure_comparison(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        pdb_id: str,
        save_name: Optional[str] = None,
    ):
        """Plot 3D structure comparison.

        Args:
            predicted: Predicted CA coordinates (N, 3)
            target: Target CA coordinates (N, 3)
            pdb_id: PDB identifier
            save_name: Optional custom filename
        """
        fig = plt.figure(figsize=(18, 6))

        # Predicted structure
        ax1 = fig.add_subplot(131, projection="3d")
        ax1.plot(
            predicted[:, 0],
            predicted[:, 1],
            predicted[:, 2],
            "b-",
            alpha=0.6,
            linewidth=2,
            label="Backbone",
        )
        ax1.scatter(
            predicted[:, 0],
            predicted[:, 1],
            predicted[:, 2],
            c=np.arange(len(predicted)),
            cmap="viridis",
            s=30,
            alpha=0.7,
        )
        ax1.set_title("Predicted Structure", fontweight="bold")
        ax1.set_xlabel("X (Å)")
        ax1.set_ylabel("Y (Å)")
        ax1.set_zlabel("Z (Å)")
        ax1.legend()

        # Target structure
        ax2 = fig.add_subplot(132, projection="3d")
        ax2.plot(
            target[:, 0], target[:, 1], target[:, 2], "r-", alpha=0.6, linewidth=2, label="Backbone"
        )
        ax2.scatter(
            target[:, 0],
            target[:, 1],
            target[:, 2],
            c=np.arange(len(target)),
            cmap="plasma",
            s=30,
            alpha=0.7,
        )
        ax2.set_title("Target Structure", fontweight="bold")
        ax2.set_xlabel("X (Å)")
        ax2.set_ylabel("Y (Å)")
        ax2.set_zlabel("Z (Å)")
        ax2.legend()

        # Overlay
        ax3 = fig.add_subplot(133, projection="3d")
        ax3.plot(
            predicted[:, 0],
            predicted[:, 1],
            predicted[:, 2],
            "b-",
            alpha=0.5,
            linewidth=2,
            label="Predicted",
        )
        ax3.plot(
            target[:, 0], target[:, 1], target[:, 2], "r-", alpha=0.5, linewidth=2, label="Target"
        )
        ax3.scatter(predicted[:, 0], predicted[:, 1], predicted[:, 2], c="blue", s=20, alpha=0.4)
        ax3.scatter(target[:, 0], target[:, 1], target[:, 2], c="red", s=20, alpha=0.4)
        ax3.set_title("Superposition", fontweight="bold")
        ax3.set_xlabel("X (Å)")
        ax3.set_ylabel("Y (Å)")
        ax3.set_zlabel("Z (Å)")
        ax3.legend()

        plt.suptitle(f"Structure Comparison: {pdb_id}", fontsize=16, fontweight="bold")
        plt.tight_layout()

        save_name = save_name or f"structure_{pdb_id}"
        for ext in ["png", "pdf"]:
            plt.savefig(self.output_dir / f"{save_name}.{ext}")

        logger.info(f"Structure comparison saved to {self.output_dir}/{save_name}")
        plt.close()

    def plot_per_residue_error(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        sequence: str,
        pdb_id: str,
        save_name: Optional[str] = None,
    ):
        """Plot per-residue error analysis.

        Args:
            predicted: Predicted CA coordinates
            target: Target CA coordinates
            sequence: Amino acid sequence
            pdb_id: PDB identifier
            save_name: Optional custom filename
        """
        # Compute per-residue distances
        distances = np.sqrt(np.sum((predicted - target) ** 2, axis=1))

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Error plot
        axes[0].plot(distances, linewidth=2, color="steelblue")
        axes[0].axhline(
            y=np.mean(distances),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(distances):.2f}Å",
            linewidth=2,
        )
        axes[0].fill_between(range(len(distances)), 0, distances, alpha=0.3)
        axes[0].set_ylabel("Cα Distance (Å)", fontweight="bold")
        axes[0].set_title(f"Per-Residue Error: {pdb_id}", fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Heatmap of error by residue type
        residue_errors = {}
        for i, aa in enumerate(sequence):
            if aa not in residue_errors:
                residue_errors[aa] = []
            residue_errors[aa].append(distances[i])

        aa_types = sorted(residue_errors.keys())
        mean_errors = [np.mean(residue_errors[aa]) for aa in aa_types]

        axes[1].bar(aa_types, mean_errors, color="coral", alpha=0.7)
        axes[1].set_xlabel("Residue Type", fontweight="bold")
        axes[1].set_ylabel("Mean Cα Distance (Å)", fontweight="bold")
        axes[1].set_title("Average Error by Residue Type", fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()

        save_name = save_name or f"per_residue_error_{pdb_id}"
        for ext in ["png", "pdf"]:
            plt.savefig(self.output_dir / f"{save_name}.{ext}")

        logger.info(f"Per-residue error plot saved to {self.output_dir}/{save_name}")
        plt.close()

    def plot_distance_matrix_comparison(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        pdb_id: str,
        save_name: Optional[str] = None,
    ):
        """Plot distance matrix comparison.

        Args:
            predicted: Predicted CA coordinates
            target: Target CA coordinates
            pdb_id: PDB identifier
            save_name: Optional custom filename
        """
        from scipy.spatial.distance import pdist, squareform

        pred_dmat = squareform(pdist(predicted))
        targ_dmat = squareform(pdist(target))
        diff_dmat = np.abs(pred_dmat - targ_dmat)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Predicted distance matrix
        im1 = axes[0].imshow(pred_dmat, cmap="viridis", aspect="auto")
        axes[0].set_title("Predicted Distance Matrix", fontweight="bold")
        axes[0].set_xlabel("Residue Index")
        axes[0].set_ylabel("Residue Index")
        plt.colorbar(im1, ax=axes[0], label="Distance (Å)")

        # Target distance matrix
        im2 = axes[1].imshow(targ_dmat, cmap="viridis", aspect="auto")
        axes[1].set_title("Target Distance Matrix", fontweight="bold")
        axes[1].set_xlabel("Residue Index")
        axes[1].set_ylabel("Residue Index")
        plt.colorbar(im2, ax=axes[1], label="Distance (Å)")

        # Difference
        im3 = axes[2].imshow(diff_dmat, cmap="RdYlBu_r", aspect="auto")
        axes[2].set_title("Absolute Difference", fontweight="bold")
        axes[2].set_xlabel("Residue Index")
        axes[2].set_ylabel("Residue Index")
        plt.colorbar(im3, ax=axes[2], label="|Difference| (Å)")

        plt.suptitle(f"Distance Matrix Analysis: {pdb_id}", fontsize=16, fontweight="bold")
        plt.tight_layout()

        save_name = save_name or f"distance_matrices_{pdb_id}"
        for ext in ["png", "pdf"]:
            plt.savefig(self.output_dir / f"{save_name}.{ext}")

        logger.info(f"Distance matrix comparison saved to {self.output_dir}/{save_name}")
        plt.close()

    def plot_metric_distributions(
        self, results: Dict[str, List[float]], save_name: str = "metric_distributions"
    ):
        """Plot distributions of metrics across test set.

        Args:
            results: Dictionary mapping metric names to lists of values
            save_name: Filename for saving
        """
        n_metrics = len(results)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(14, 8))
        axes = axes.flatten()

        for idx, (metric_name, values) in enumerate(results.items()):
            ax = axes[idx]

            # Histogram
            ax.hist(values, bins=30, alpha=0.7, color="steelblue", edgecolor="black")

            # Add statistics
            mean_val = np.mean(values)
            median_val = np.median(values)
            np.std(values)

            ax.axvline(
                mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}"
            )
            ax.axvline(
                median_val,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Median: {median_val:.2f}",
            )

            ax.set_xlabel(metric_name, fontweight="bold")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{metric_name} Distribution", fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)

        # Hide extra subplots
        for idx in range(len(results), len(axes)):
            axes[idx].axis("off")

        plt.suptitle("Metric Distributions Across Test Set", fontsize=16, fontweight="bold")
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            plt.savefig(self.output_dir / f"{save_name}.{ext}")

        logger.info(f"Metric distributions saved to {self.output_dir}/{save_name}")
        plt.close()
