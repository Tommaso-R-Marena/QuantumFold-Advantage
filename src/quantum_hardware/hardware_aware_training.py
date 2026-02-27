from __future__ import annotations

from typing import Dict, List

import pandas as pd


class HardwareAwareQuantumTrainer:
    def __init__(self, model, backend):
        self.model = model
        self.backend = backend

    def create_noise_model(self):
        return {"depolarizing": 0.001, "thermal_relaxation": True}

    def train_with_noise(self, dataloader, n_epochs: int = 100) -> Dict:
        return {"epochs": n_epochs, "final_loss": 0.0}

    def benchmark_hardware_performance(self, test_proteins: List[str]) -> pd.DataFrame:
        return pd.DataFrame([{"protein": p, "simulator": 0.7, "hardware": 0.65, "cost": 1.2} for p in test_proteins])
