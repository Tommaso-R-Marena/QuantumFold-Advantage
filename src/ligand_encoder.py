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
            x = layer(x)
        return x
