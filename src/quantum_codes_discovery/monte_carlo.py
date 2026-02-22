"""Basic threshold estimation under depolarizing noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdPoint:
    p: float
    logical_error_rate: float


def sample_depolarizing_errors(rng: np.random.Generator, n_qubits: int, p: float) -> np.ndarray:
    """Sample Pauli errors encoded as 0:I,1:X,2:Y,3:Z."""
    draws = rng.random(n_qubits)
    pauli = np.zeros(n_qubits, dtype=np.uint8)
    mask = draws < p
    pauli[mask] = rng.integers(1, 4, size=int(mask.sum()), endpoint=False)
    return pauli


def logical_failure_proxy(errors: np.ndarray, distance: int) -> bool:
    """Proxy decoder failure model based on error weight.

    Failure is declared when weight exceeds correctable floor((d-1)/2).
    """
    t = max(0, (distance - 1) // 2)
    return int(np.count_nonzero(errors)) > t


def estimate_logical_error_rate(
    n_qubits: int,
    distance: int,
    p: float,
    samples: int,
    seed: int = 7,
) -> float:
    rng = np.random.default_rng(seed)
    failures = 0
    for _ in range(samples):
        errors = sample_depolarizing_errors(rng, n_qubits, p)
        failures += int(logical_failure_proxy(errors, distance))
    return failures / samples


def sweep_threshold(
    n_qubits: int,
    distance: int,
    p_values: list[float],
    samples: int,
    seed: int = 7,
) -> list[ThresholdPoint]:
    return [
        ThresholdPoint(p=p, logical_error_rate=estimate_logical_error_rate(n_qubits, distance, p, samples, seed))
        for p in p_values
    ]
