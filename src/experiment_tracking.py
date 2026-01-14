"""Comprehensive experiment tracking with Weights & Biases integration.

Provides automatic logging of metrics, hyperparameters, model artifacts,
and system information for full reproducibility.
"""

import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for reproducible experiments."""

    # Model architecture
    model_name: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    use_gradient_checkpointing: bool

    # Training
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    grad_accum_steps: int
    warmup_steps: int
    max_grad_norm: float
    mixed_precision: bool

    # Data
    dataset_name: str
    train_size: int
    val_size: int
    test_size: int
    max_seq_length: int
    augmentation: bool

    # Embedding
    embedding_model: str
    embedding_dim: int
    cache_embeddings: bool

    # Loss configuration
    coord_loss_weight: float
    fape_loss_weight: float
    dist_loss_weight: float
    conf_loss_weight: float

    # System
    seed: int
    device: str
    num_workers: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)

    def save(self, path: str):
        """Save configuration to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def get_hash(self) -> str:
        """Generate unique hash for this configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class ExperimentTracker:
    """Comprehensive experiment tracking and logging."""

    def __init__(
        self,
        config: ExperimentConfig,
        project_name: str = "QuantumFold-Advantage",
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        enable_wandb: bool = True,
    ):
        """
        Initialize experiment tracker.

        Args:
            config: Experiment configuration
            project_name: W&B project name
            entity: W&B entity (username/team)
            tags: List of tags for organization
            notes: Experiment notes
            enable_wandb: Whether to use W&B (disable for local testing)
        """
        self.config = config
        self.enable_wandb = enable_wandb
        self.best_metrics = {}

        # Create experiment directory
        exp_name = f"{config.model_name}_{config.get_hash()}"
        self.exp_dir = Path(f"experiments/{exp_name}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config.save(self.exp_dir / "config.json")

        # Initialize W&B
        if self.enable_wandb:
            self.run = wandb.init(
                project=project_name,
                entity=entity,
                config=config.to_dict(),
                name=exp_name,
                tags=tags or [],
                notes=notes,
                dir=str(self.exp_dir),
            )

            # Log code
            wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
        else:
            self.run = None
            logger.info("Running without W&B tracking")

        logger.info(f"Experiment directory: {self.exp_dir}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True,
        prefix: str = "",
    ):
        """Log metrics to W&B and local file.

        Args:
            metrics: Dictionary of metric names to values
            step: Training step number
            commit: Whether to commit the metrics immediately
            prefix: Prefix to add to metric names (e.g., 'train/', 'val/')
        """
        prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        if self.enable_wandb:
            wandb.log(prefixed_metrics, step=step, commit=commit)

        # Also save to local JSON
        metrics_file = self.exp_dir / f"{prefix.replace('/', '_')}metrics.jsonl"
        with open(metrics_file, "a") as f:
            log_entry = {"step": step, **prefixed_metrics}
            f.write(json.dumps(log_entry) + "\n")

    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log additional hyperparameters."""
        if self.enable_wandb:
            wandb.config.update(hyperparams)

    def log_model(
        self, model: torch.nn.Module, name: str = "model", metadata: Optional[Dict] = None
    ):
        """Save and log model checkpoint.

        Args:
            model: PyTorch model
            name: Model name/version
            metadata: Additional metadata to save
        """
        checkpoint_path = self.exp_dir / f"{name}.pt"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": self.config.to_dict(),
            "metadata": metadata or {},
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")

        if self.enable_wandb:
            artifact = wandb.Artifact(name, type="model", metadata=metadata)
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)

    def log_best_model(
        self,
        model: torch.nn.Module,
        metric_value: float,
        metric_name: str,
        higher_is_better: bool = False,
    ):
        """Save model if it achieves best metric.

        Args:
            model: PyTorch model
            metric_value: Current metric value
            metric_name: Metric name
            higher_is_better: Whether higher values are better
        """
        is_best = False

        if metric_name not in self.best_metrics:
            is_best = True
        else:
            if higher_is_better:
                is_best = metric_value > self.best_metrics[metric_name]
            else:
                is_best = metric_value < self.best_metrics[metric_name]

        if is_best:
            self.best_metrics[metric_name] = metric_value
            metadata = {
                "metric_name": metric_name,
                "metric_value": metric_value,
                "higher_is_better": higher_is_better,
            }
            self.log_model(model, f"best_{metric_name}", metadata)
            logger.info(f"New best {metric_name}: {metric_value:.4f}")

    def log_system_info(self):
        """Log system and environment information."""
        system_info = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": (
                torch.cuda.get_device_properties(0).total_memory / 1e9
                if torch.cuda.is_available()
                else None
            ),
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        if self.enable_wandb:
            wandb.config.update({"system": system_info})

        with open(self.exp_dir / "system_info.json", "w") as f:
            json.dump(system_info, f, indent=2)

        return system_info

    def log_dataset_info(self, train_dataset, val_dataset, test_dataset):
        """Log dataset statistics."""
        dataset_info = {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "total_size": len(train_dataset) + len(val_dataset) + len(test_dataset),
        }

        if self.enable_wandb:
            wandb.config.update({"dataset": dataset_info})

        with open(self.exp_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

    def watch_model(self, model: torch.nn.Module, log_freq: int = 100, log_graph: bool = True):
        """Watch model for gradient and parameter tracking.

        Args:
            model: PyTorch model
            log_freq: How often to log (in steps)
            log_graph: Whether to log computational graph
        """
        if self.enable_wandb:
            wandb.watch(model, log="all", log_freq=log_freq, log_graph=log_graph)

    def log_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        pdb_ids: List[str],
        step: int,
        max_samples: int = 10,
    ):
        """Log prediction examples with visualizations.

        Args:
            predictions: Predicted structures
            targets: Target structures
            pdb_ids: PDB identifiers
            step: Training step
            max_samples: Maximum number of samples to log
        """
        if not self.enable_wandb:
            return

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        n_samples = min(len(predictions), max_samples)

        for i in range(n_samples):
            fig = plt.figure(figsize=(15, 5))

            # Plot predicted structure
            ax1 = fig.add_subplot(131, projection="3d")
            pred = predictions[i]
            ax1.plot(pred[:, 0], pred[:, 1], pred[:, 2], "b-", alpha=0.6, linewidth=2)
            ax1.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c="blue", s=20)
            ax1.set_title(f"Predicted: {pdb_ids[i]}")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")

            # Plot target structure
            ax2 = fig.add_subplot(132, projection="3d")
            targ = targets[i]
            ax2.plot(targ[:, 0], targ[:, 1], targ[:, 2], "r-", alpha=0.6, linewidth=2)
            ax2.scatter(targ[:, 0], targ[:, 1], targ[:, 2], c="red", s=20)
            ax2.set_title(f"Target: {pdb_ids[i]}")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Z")

            # Plot overlay
            ax3 = fig.add_subplot(133, projection="3d")
            ax3.plot(
                pred[:, 0], pred[:, 1], pred[:, 2], "b-", alpha=0.4, linewidth=2, label="Predicted"
            )
            ax3.plot(
                targ[:, 0], targ[:, 1], targ[:, 2], "r-", alpha=0.4, linewidth=2, label="Target"
            )
            ax3.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c="blue", s=15, alpha=0.5)
            ax3.scatter(targ[:, 0], targ[:, 1], targ[:, 2], c="red", s=15, alpha=0.5)
            ax3.set_title("Overlay")
            ax3.set_xlabel("X")
            ax3.set_ylabel("Y")
            ax3.set_zlabel("Z")
            ax3.legend()

            plt.tight_layout()

            wandb.log({f"predictions/{pdb_ids[i]}": wandb.Image(fig)}, step=step)
            plt.close(fig)

    def log_attention_maps(
        self,
        attention_weights: torch.Tensor,
        sequence: str,
        step: int,
        layer_idx: int = 0,
        head_idx: int = 0,
    ):
        """Log attention heatmaps.

        Args:
            attention_weights: Attention tensor (num_heads, seq_len, seq_len)
            sequence: Amino acid sequence
            step: Training step
            layer_idx: Which layer to visualize
            head_idx: Which attention head to visualize
        """
        if not self.enable_wandb:
            return

        import matplotlib.pyplot as plt
        import seaborn as sns

        attn = attention_weights[head_idx].cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(attn, cmap="viridis", square=True, cbar_kws={"label": "Attention Weight"})
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title(
            f"Attention Map - Layer {layer_idx}, Head {head_idx}\nSequence: {sequence[:50]}..."
        )

        wandb.log({f"attention/layer{layer_idx}_head{head_idx}": wandb.Image(fig)}, step=step)
        plt.close(fig)

    def finish(self):
        """Finish experiment and cleanup."""
        if self.enable_wandb:
            wandb.finish()

        logger.info(f"Experiment finished. Results saved to {self.exp_dir}")

    def create_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        report = []
        report.append("=" * 80)
        report.append("EXPERIMENT SUMMARY")
        report.append("=" * 80)
        report.append(f"\nExperiment Directory: {self.exp_dir}")
        report.append(f"Configuration Hash: {self.config.get_hash()}")
        report.append("\nBest Metrics:")

        for metric_name, value in self.best_metrics.items():
            report.append(f"  {metric_name}: {value:.4f}")

        report.append("\nConfiguration:")
        for key, value in self.config.to_dict().items():
            report.append(f"  {key}: {value}")

        report.append("=" * 80)

        summary_text = "\n".join(report)

        # Save to file
        with open(self.exp_dir / "summary.txt", "w") as f:
            f.write(summary_text)

        return summary_text
