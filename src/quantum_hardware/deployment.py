from __future__ import annotations

from typing import Dict


class QuantumDeploymentPipeline:
    def deploy_model(self, model, backend_name: str, num_proteins_per_batch: int = 10) -> None:
        return None

    def estimate_cost(self, num_proteins: int, shots_per_protein: int = 1024) -> Dict:
        total_shots = num_proteins * shots_per_protein
        return {
            "ibm_qpu_seconds": total_shots * 1e-4,
            "ionq_usd": total_shots * 0.30 / 1000,
            "rigetti_usd": total_shots * 0.10 / 1000,
            "aws_braket_usd": num_proteins * 0.01 + total_shots * 0.00035,
        }
