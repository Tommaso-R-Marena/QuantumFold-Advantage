from __future__ import annotations

from typing import Dict, List

import numpy as np


class ErrorMitigationPipeline:
    def __init__(self, backend):
        self.backend = backend

    def calibrate_readout_errors(self, n_qubits: int) -> np.ndarray:
        dim = 2**n_qubits
        return np.eye(dim)

    def apply_zero_noise_extrapolation(
        self, circuit, noise_factors: List[float] = [1.0, 1.5, 2.0]
    ) -> Dict:
        return {"noise_factors": noise_factors, "extrapolated_expectation": 0.0}

    def apply_probabilistic_error_cancellation(self, circuit) -> Dict:
        return {"pec_estimate": 0.0, "variance": 0.0}

    def apply_measurement_error_mitigation(
        self, raw_counts: Dict[str, int], calibration_matrix: np.ndarray
    ) -> Dict[str, int]:
        return raw_counts
