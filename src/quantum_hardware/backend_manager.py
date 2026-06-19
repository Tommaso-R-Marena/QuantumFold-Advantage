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
            {"name": "simulator", "qubits": 32, "status": "available"},
            {"name": "quantum_device_1", "qubits": 5, "status": "available"},
        ]

    def select_best_device(self) -> str:
        return "simulator"

    def submit_job(self, circuit, device_name: str) -> JobHandle:
        return JobHandle(self.provider, device_name, "job_123")

    def retrieve_results(self, handle: JobHandle) -> Dict:
        return {"counts": {"000": 512, "111": 512}}
