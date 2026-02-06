"""Benchmarking utilities for QuantumFold-Advantage.

Provides research-grade metrics and statistical validation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .research_metrics import ResearchBenchmark, StructurePredictionMetrics


def _to_numpy(coords: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor-like coordinates to a NumPy array of shape (N, 3)."""
    if isinstance(coords, torch.Tensor):
        arr = coords.detach().cpu().numpy()
    else:
        arr = np.asarray(coords)

    if arr.ndim == 4:
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
    elif arr.ndim == 3:
        arr = arr[0]

    if arr.ndim != 2 or arr.shape[-1] != 3:
        raise ValueError(f"Expected coordinates with final shape (N, 3), got {arr.shape}")

    return arr.astype(np.float64)


def calculate_rmsd(
    predicted: torch.Tensor | np.ndarray, ground_truth: torch.Tensor | np.ndarray
) -> float:
    """Calculate RMSD for compatibility with legacy tests/notebooks."""
    benchmark = ResearchBenchmark(n_bootstrap=100)
    pred = _to_numpy(predicted)
    true = _to_numpy(ground_truth)
    return benchmark.compute_rmsd(pred, true, align=False)


def calculate_tm_score(
    predicted: torch.Tensor | np.ndarray, ground_truth: torch.Tensor | np.ndarray
) -> float:
    """Calculate TM-score for compatibility with legacy tests/notebooks."""
    benchmark = ResearchBenchmark(n_bootstrap=100)
    pred = _to_numpy(predicted)
    true = _to_numpy(ground_truth)
    return benchmark.compute_tm_score(pred, true)


class BenchmarkMetrics:
    """Compatibility wrapper exposing simple metric dictionary outputs."""

    def __init__(self) -> None:
        self.benchmark = ResearchBenchmark(n_bootstrap=500)

    def calculate_all(
        self,
        predicted: torch.Tensor | np.ndarray,
        ground_truth: torch.Tensor | np.ndarray,
    ) -> Dict[str, float]:
        pred = _to_numpy(predicted)
        true = _to_numpy(ground_truth)

        gdt = self.benchmark.compute_gdt(pred, true)
        return {
            "rmsd": self.benchmark.compute_rmsd(pred, true, align=False),
            "tm_score": self.benchmark.compute_tm_score(pred, true),
            "gdt_ts": gdt["GDT_TS"],
            "gdt_ha": gdt["GDT_HA"],
            "lddt": self.benchmark.compute_lddt(pred, true),
        }


__all__ = [
    "ResearchBenchmark",
    "StructurePredictionMetrics",
    "BenchmarkMetrics",
    "calculate_rmsd",
    "calculate_tm_score",
]
