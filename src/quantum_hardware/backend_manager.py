from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class JobHandle:
    provider: str
    device_name: str
    job_id: str


class QuantumBackendManager:
    def __init__(
        self, provider: str = "ibm", credentials_path: Path = Path("~/.quantum_credentials.json")
    ):
        self.provider = provider
        self.credentials = self.load_credentials(credentials_path.expanduser())

    def load_credentials(self, credentials_path: Path) -> Dict:
        if credentials_path.exists():
            return json.loads(credentials_path.read_text())
        return {}

    def list_available_devices(self) -> List[Dict]:
        return [
            {
                "name": "ibm_brisbane",
                "num_qubits": 127,
                "connectivity": "heavy-hex",
                "error_rates": {"2q": 0.01},
                "queue_depth": 5,
                "cost": 0.0,
            },
            {
                "name": "ionq_aria",
                "num_qubits": 25,
                "connectivity": "all-to-all",
                "error_rates": {"2q": 0.02},
                "queue_depth": 2,
                "cost": 1.0,
            },
        ]

    def select_best_device(self, n_qubits_required: int, min_fidelity: float = 0.99) -> str:
        candidates = [
            d for d in self.list_available_devices() if d["num_qubits"] >= n_qubits_required
        ]
        return sorted(candidates, key=lambda d: (d["queue_depth"], d["error_rates"]["2q"]))[0][
            "name"
        ]

    def submit_job(
        self, circuit, shots: int = 1024, device_name: Optional[str] = None
    ) -> JobHandle:
        return JobHandle(
            provider=self.provider,
            device_name=device_name or self.select_best_device(4),
            job_id="mock-job-001",
        )

    def retrieve_results(self, job_handle: JobHandle, apply_error_mitigation: bool = True) -> Dict:
        return {
            "job_id": job_handle.job_id,
            "counts": {"0000": 512, "1111": 512},
            "mitigated": apply_error_mitigation,
        }
