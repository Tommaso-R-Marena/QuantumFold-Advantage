"""Systematic Ablation Study Framework.

Provides tools for conducting comprehensive ablation studies to isolate
the contribution of individual model components.
"""

import itertools
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""

    name: str
    description: str

    # Model configuration changes
    config_modifications: Dict[str, Any] = field(default_factory=dict)

    # Training modifications
    training_modifications: Dict[str, Any] = field(default_factory=dict)

    # Expected behavior
    expected_effect: str = ""  # "improve", "degrade", "neutral"

    # Metadata
    category: str = "architecture"  # "architecture", "training", "data"

    def __repr__(self) -> str:
        return f"AblationConfig(name='{self.name}', category='{self.category}')"


@dataclass
class AblationResult:
    """Results from a single ablation experiment."""

    config: AblationConfig

    # Performance metrics
    metrics: Dict[str, float]

    # Comparison to baseline
    baseline_metrics: Dict[str, float]
    relative_change: Dict[str, float]  # Percentage change

    # Statistical significance
    pvalues: Dict[str, float]
    significant: bool

    # Training statistics
    training_time: float
    n_parameters: int
    convergence_step: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "config": {
                "name": self.config.name,
                "description": self.config.description,
                "category": self.config.category,
                "modifications": self.config.config_modifications,
            },
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "relative_change": self.relative_change,
            "pvalues": self.pvalues,
            "significant": self.significant,
            "training_stats": {
                "time_seconds": self.training_time,
                "n_parameters": self.n_parameters,
                "convergence_step": self.convergence_step,
            },
        }


class AblationStudy:
    """Framework for systematic ablation studies.

    Automates the process of:
    1. Defining ablation configurations
    2. Training ablated models
    3. Evaluating performance
    4. Statistical comparison to baseline
    5. Generating reports and visualizations
    """

    def __init__(
        self,
        baseline_config: Dict[str, Any],
        baseline_metrics: Dict[str, float],
        model_factory: Callable,
        train_function: Callable,
        eval_function: Callable,
        output_dir: Path = Path("ablation_results"),
    ):
        """Initialize ablation framework.

        Args:
            baseline_config: Configuration for baseline model
            baseline_metrics: Baseline model performance metrics
            model_factory: Function to create model from config
            train_function: Function to train model
            eval_function: Function to evaluate model
            output_dir: Directory for saving results
        """
        self.baseline_config = baseline_config
        self.baseline_metrics = baseline_metrics
        self.model_factory = model_factory
        self.train_function = train_function
        self.eval_function = eval_function
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ablation_configs: List[AblationConfig] = []
        self.results: List[AblationResult] = []

    def add_ablation(self, config: AblationConfig) -> None:
        """Add an ablation configuration.

        Args:
            config: AblationConfig object
        """
        self.ablation_configs.append(config)

    def add_quantum_ablations(self) -> None:
        """Add standard quantum architecture ablations."""

        # No quantum (baseline)
        self.add_ablation(
            AblationConfig(
                name="no_quantum",
                description="Remove all quantum layers",
                config_modifications={"use_quantum": False},
                expected_effect="degrade",
                category="quantum",
            )
        )

        # Entanglement topology
        for entangle in ["linear", "circular", "all_to_all"]:
            self.add_ablation(
                AblationConfig(
                    name=f"entanglement_{entangle}",
                    description=f"Use {entangle} entanglement",
                    config_modifications={"entanglement": entangle},
                    expected_effect="vary",
                    category="quantum",
                )
            )

        # Number of quantum layers
        for n_layers in [1, 2, 3, 5, 8]:
            self.add_ablation(
                AblationConfig(
                    name=f"quantum_layers_{n_layers}",
                    description=f"Use {n_layers} quantum layers",
                    config_modifications={"n_quantum_layers": n_layers},
                    expected_effect="vary",
                    category="quantum",
                )
            )

        # Number of qubits
        for n_qubits in [4, 6, 8, 10, 12]:
            self.add_ablation(
                AblationConfig(
                    name=f"qubits_{n_qubits}",
                    description=f"Use {n_qubits} qubits",
                    config_modifications={"n_qubits": n_qubits},
                    expected_effect="improve",
                    category="quantum",
                )
            )

        # Noise levels
        for noise_prob in [0.0, 0.001, 0.01, 0.05, 0.1]:
            self.add_ablation(
                AblationConfig(
                    name=f"noise_{noise_prob:.3f}",
                    description=f"Depolarizing noise p={noise_prob}",
                    config_modifications={"noise_prob": noise_prob},
                    expected_effect="degrade" if noise_prob > 0 else "improve",
                    category="quantum",
                )
            )

    def add_architecture_ablations(self) -> None:
        """Add standard architecture ablations."""

        # Hidden dimension
        for hidden_dim in [256, 384, 512, 768, 1024]:
            self.add_ablation(
                AblationConfig(
                    name=f"hidden_dim_{hidden_dim}",
                    description=f"Hidden dimension = {hidden_dim}",
                    config_modifications={"hidden_dim": hidden_dim},
                    expected_effect="improve" if hidden_dim > 512 else "degrade",
                    category="architecture",
                )
            )

        # Number of transformer layers
        for n_layers in [2, 4, 6, 8, 12]:
            self.add_ablation(
                AblationConfig(
                    name=f"transformer_layers_{n_layers}",
                    description=f"{n_layers} transformer encoder layers",
                    config_modifications={"n_transformer_layers": n_layers},
                    expected_effect="vary",
                    category="architecture",
                )
            )

        # Attention heads
        for n_heads in [4, 8, 12, 16]:
            self.add_ablation(
                AblationConfig(
                    name=f"attention_heads_{n_heads}",
                    description=f"{n_heads} attention heads",
                    config_modifications={"n_heads": n_heads},
                    expected_effect="vary",
                    category="architecture",
                )
            )

        # Structure refinement layers
        for n_struct_layers in [2, 4, 6, 8, 12]:
            self.add_ablation(
                AblationConfig(
                    name=f"structure_layers_{n_struct_layers}",
                    description=f"{n_struct_layers} structure module layers",
                    config_modifications={"n_structure_layers": n_struct_layers},
                    expected_effect="improve" if n_struct_layers >= 8 else "degrade",
                    category="architecture",
                )
            )

        # Remove components
        self.add_ablation(
            AblationConfig(
                name="no_structure_module",
                description="Remove iterative structure refinement",
                config_modifications={"n_structure_layers": 0},
                expected_effect="degrade",
                category="architecture",
            )
        )

        self.add_ablation(
            AblationConfig(
                name="no_confidence_head",
                description="Remove confidence prediction head",
                config_modifications={"confidence_head": False},
                expected_effect="neutral",
                category="architecture",
            )
        )

    def add_training_ablations(self) -> None:
        """Add standard training procedure ablations."""

        # Learning rates
        for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
            self.add_ablation(
                AblationConfig(
                    name=f"lr_{lr:.0e}",
                    description=f"Learning rate = {lr}",
                    training_modifications={"learning_rate": lr},
                    expected_effect="vary",
                    category="training",
                )
            )

        # Optimizers
        for optimizer in ["adam", "adamw", "sgd"]:
            self.add_ablation(
                AblationConfig(
                    name=f"optimizer_{optimizer}",
                    description=f"Optimizer: {optimizer}",
                    training_modifications={"optimizer": optimizer},
                    expected_effect="vary",
                    category="training",
                )
            )

        # Mixed precision
        self.add_ablation(
            AblationConfig(
                name="no_mixed_precision",
                description="Train in FP32 instead of FP16",
                training_modifications={"use_amp": False},
                expected_effect="neutral",
                category="training",
            )
        )

        # EMA
        self.add_ablation(
            AblationConfig(
                name="no_ema",
                description="Disable exponential moving average",
                training_modifications={"use_ema": False},
                expected_effect="degrade",
                category="training",
            )
        )

        # Gradient accumulation
        for grad_accum in [1, 2, 4, 8]:
            self.add_ablation(
                AblationConfig(
                    name=f"grad_accum_{grad_accum}",
                    description=f"Gradient accumulation steps = {grad_accum}",
                    training_modifications={"gradient_accumulation_steps": grad_accum},
                    expected_effect="vary",
                    category="training",
                )
            )

    def run_ablation(self, config: AblationConfig, n_seeds: int = 3) -> AblationResult:
        """Run a single ablation experiment.

        Args:
            config: Ablation configuration
            n_seeds: Number of random seeds to average over

        Returns:
            AblationResult with performance metrics
        """
        print(f"\nRunning ablation: {config.name}")
        print(f"Description: {config.description}")

        all_metrics = []
        training_times = []

        for seed in range(n_seeds):
            # Create modified config
            model_config = deepcopy(self.baseline_config)
            model_config.update(config.config_modifications)

            training_config = deepcopy(self.baseline_config.get("training", {}))
            training_config.update(config.training_modifications)
            training_config["seed"] = seed

            # Create model
            model = self.model_factory(model_config)

            # Train
            import time

            start_time = time.time()
            trained_model, train_info = self.train_function(model, training_config)
            training_time = time.time() - start_time

            # Evaluate
            metrics = self.eval_function(trained_model)

            all_metrics.append(metrics)
            training_times.append(training_time)

            print(f"  Seed {seed}: TM-score = {metrics.get('tm_score', 0):.4f}")

        # Aggregate across seeds
        aggregated_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated_metrics[key] = np.mean(values)
            aggregated_metrics[f"{key}_std"] = np.std(values)

        # Compute relative change
        relative_change = {}
        for key in self.baseline_metrics.keys():
            if key in aggregated_metrics:
                baseline = self.baseline_metrics[key]
                ablation = aggregated_metrics[key]
                rel_change = 100 * (ablation - baseline) / baseline
                relative_change[key] = rel_change

        # Statistical significance (paired t-test)
        from scipy import stats

        pvalues = {}
        for key in self.baseline_metrics.keys():
            if key in aggregated_metrics:
                # Assuming we have baseline values for each seed
                baseline_values = [self.baseline_metrics[key]] * n_seeds
                ablation_values = [m[key] for m in all_metrics]
                _, pval = stats.ttest_rel(ablation_values, baseline_values)
                pvalues[key] = pval

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())

        # Determine significance
        significant = any(p < 0.05 for p in pvalues.values())

        result = AblationResult(
            config=config,
            metrics=aggregated_metrics,
            baseline_metrics=self.baseline_metrics,
            relative_change=relative_change,
            pvalues=pvalues,
            significant=significant,
            training_time=np.mean(training_times),
            n_parameters=n_params,
            convergence_step=train_info.get("convergence_step"),
        )

        self.results.append(result)
        return result

    def run_all_ablations(
        self, n_seeds: int = 3, categories: Optional[List[str]] = None
    ) -> List[AblationResult]:
        """Run all defined ablation experiments.

        Args:
            n_seeds: Number of random seeds per ablation
            categories: List of categories to run (None = all)

        Returns:
            List of AblationResult objects
        """
        configs_to_run = self.ablation_configs
        if categories is not None:
            configs_to_run = [c for c in configs_to_run if c.category in categories]

        print(f"Running {len(configs_to_run)} ablations...")

        results = []
        for config in configs_to_run:
            result = self.run_ablation(config, n_seeds=n_seeds)
            results.append(result)

        return results

    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate markdown ablation study report.

        Args:
            output_file: Optional file to save report

        Returns:
            Markdown formatted report string
        """
        report = []
        report.append("# Ablation Study Report\n")
        report.append(
            f"**Baseline Configuration:**\n```json\n{json.dumps(self.baseline_config, indent=2)}\n```\n"
        )
        report.append(f"**Baseline Metrics:**\n")
        for key, val in self.baseline_metrics.items():
            report.append(f"- {key}: {val:.4f}\n")
        report.append("\n---\n")

        # Group by category
        by_category = {}
        for result in self.results:
            cat = result.config.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result)

        for category, results in by_category.items():
            report.append(f"\n## {category.capitalize()} Ablations\n")

            # Create table
            report.append(
                "| Ablation | TM-score | Δ TM | GDT-TS | Δ GDT | Significant | Params |\n"
            )
            report.append(
                "|----------|----------|---------|--------|---------|-------------|--------|\n"
            )

            for result in results:
                name = result.config.name
                tm = result.metrics.get("tm_score", 0)
                tm_change = result.relative_change.get("tm_score", 0)
                gdt = result.metrics.get("gdt_ts", 0)
                gdt_change = result.relative_change.get("gdt_ts", 0)
                sig = "✅" if result.significant else "❌"
                params = f"{result.n_parameters/1e6:.1f}M"

                report.append(
                    f"| {name} | {tm:.4f} | {tm_change:+.2f}% | {gdt:.2f} | {gdt_change:+.2f}% | {sig} | {params} |\n"
                )

        report_text = "".join(report)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")

        return report_text

    def save_results(self, output_file: Optional[Path] = None) -> None:
        """Save all results to JSON.

        Args:
            output_file: Output JSON file path
        """
        if output_file is None:
            output_file = self.output_dir / "ablation_results.json"

        output_data = {
            "baseline_config": self.baseline_config,
            "baseline_metrics": self.baseline_metrics,
            "results": [r.to_dict() for r in self.results],
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to {output_file}")
