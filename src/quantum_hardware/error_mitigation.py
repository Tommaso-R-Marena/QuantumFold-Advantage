from typing import Dict, List

import numpy as np


class ErrorMitigator:
    def __init__(self, method: str = "zne"):
        self.method = method

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
