"""Automatic improvement and benchmarking helpers.

This module provides lightweight, deterministic utilities that can be used
from the CLI to:
- suggest improved training settings from prior metric history
- benchmark candidate settings on a synthetic-but-structural proxy task
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .benchmarks import ResearchBenchmark


@dataclass
class BenchmarkCandidate:
    """Training/runtime settings candidate for auto-benchmarking."""

    name: str
    learning_rate: float
    batch_size: int
    quantum_depth: int
    use_amp: bool
    use_ema: bool


class AutoImprovementEngine:
    """Suggest parameter updates from learning history."""

    def suggest(
        self,
        history: Dict[str, List[float]],
        current: Dict[str, float],
    ) -> Dict[str, float]:
        """Suggest improved settings from metric trends.

        Expected history keys: ``train_loss`` and ``val_loss``.
        """
        train = history.get("train_loss", [])
        val = history.get("val_loss", [])
        if len(train) < 3 or len(val) < 3:
            return dict(current)

        updated = dict(current)

        recent_train_delta = train[-1] - train[-3]
        recent_val_delta = val[-1] - val[-3]

        # Plateau: reduce LR.
        if abs(recent_train_delta) < 1e-3 and abs(recent_val_delta) < 1e-3:
            updated["learning_rate"] = max(float(current["learning_rate"]) * 0.5, 1e-6)

        # Overfitting signal: validation worsens while train improves.
        if recent_train_delta < -1e-3 and recent_val_delta > 1e-3:
            updated["weight_decay"] = min(float(current.get("weight_decay", 1e-4)) * 2.0, 1e-2)

        # Underfitting: both train/val still high and improving slowly.
        if train[-1] > 0.5 and val[-1] > 0.5 and recent_train_delta > -5e-3:
            updated["model_width_multiplier"] = min(
                float(current.get("model_width_multiplier", 1.0)) * 1.25, 2.0
            )

        return updated


class AutoBenchmarkRunner:
    """Run proxy structure-quality benchmarks for candidate settings."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.evaluator = ResearchBenchmark(n_bootstrap=100)

    def _simulate_prediction_error(self, candidate: BenchmarkCandidate) -> float:
        """Map candidate settings to simulated coordinate noise (lower is better)."""
        noise = 0.75

        if candidate.use_amp:
            noise *= 0.95
        if candidate.use_ema:
            noise *= 0.9

        # Mild preference for small learning rates and moderate depth.
        noise *= min(max(candidate.learning_rate / 1e-3, 0.7), 1.3)
        noise *= 1.0 + abs(candidate.quantum_depth - 3) * 0.04
        noise *= 1.0 - min(candidate.batch_size, 64) / 1024
        return max(noise, 0.05)

    def benchmark(self, candidates: List[BenchmarkCandidate], n_samples: int = 8) -> Dict[str, object]:
        """Benchmark multiple candidates and return ranked results."""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if not candidates:
            raise ValueError("candidates must contain at least one benchmark candidate")

        results = []

        for candidate in candidates:
            start = time.perf_counter()
            rmsd_values = []
            tm_values = []

            pred_noise = self._simulate_prediction_error(candidate)
            for _ in range(n_samples):
                coords_true = self.rng.normal(0.0, 8.0, size=(128, 3))
                coords_pred = coords_true + self.rng.normal(0.0, pred_noise, size=(128, 3))

                rmsd_values.append(self.evaluator.compute_rmsd(coords_pred, coords_true, align=True))
                tm_values.append(self.evaluator.compute_tm_score(coords_pred, coords_true))

            elapsed = time.perf_counter() - start
            mean_tm = float(np.mean(tm_values))
            mean_rmsd = float(np.mean(rmsd_values))

            composite = (mean_tm * 100.0) - (mean_rmsd * 5.0) - elapsed
            results.append(
                {
                    "name": candidate.name,
                    "learning_rate": candidate.learning_rate,
                    "batch_size": candidate.batch_size,
                    "quantum_depth": candidate.quantum_depth,
                    "use_amp": candidate.use_amp,
                    "use_ema": candidate.use_ema,
                    "avg_tm_score": mean_tm,
                    "avg_rmsd": mean_rmsd,
                    "elapsed_sec": elapsed,
                    "composite_score": float(composite),
                }
            )

        ranked = sorted(results, key=lambda x: x["composite_score"], reverse=True)
        return {
            "best": ranked[0],
            "ranked": ranked,
            "n_candidates": len(ranked),
            "n_samples": n_samples,
        }


def load_history_json(path: Optional[Path]) -> Dict[str, List[float]]:
    """Load training history from JSON file."""
    if path is None or not path.exists():
        return {"train_loss": [], "val_loss": []}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "train_loss": list(payload.get("train_loss", [])),
        "val_loss": list(payload.get("val_loss", [])),
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Save JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
