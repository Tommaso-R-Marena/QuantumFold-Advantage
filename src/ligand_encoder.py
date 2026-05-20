from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


class LigandGraphEncoder(nn.Module):
    def __init__(
        self, node_features: int = 64, edge_features: int = 32, use_3d_coords: bool = True
    ):
        super().__init__()
        self.use_3d_coords = use_3d_coords
        self.layers = nn.ModuleList([nn.Linear(node_features, node_features) for _ in range(4)])

    def forward(self, ligand_data) -> torch.Tensor:
        x = ligand_data.x
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class LigandConformerGenerator:
    def __init__(self, method: str = "rdkit"):
        self.method = method

    def generate_conformers(self, smiles: str, n_conformers: int = 10) -> List[np.ndarray]:
        n_atoms = max(4, len(smiles) // 2)
        return [np.random.randn(n_atoms, 3).astype(np.float32) for _ in range(n_conformers)]
