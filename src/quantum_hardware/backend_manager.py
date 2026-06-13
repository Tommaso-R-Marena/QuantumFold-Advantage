import json
from pathlib import Path
from typing import Dict, List


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
            {"name": "ibmq_jakarta", "qubits": 7, "status": "online"},
            {"name": "ibmq_belem", "qubits": 5, "status": "online"},
            {"name": "simulator", "qubits": 32, "status": "online"},
        ]
