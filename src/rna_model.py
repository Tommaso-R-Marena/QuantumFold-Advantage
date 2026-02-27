from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from src.advanced_model import InvariantPointAttention


class RNAStructureModule(nn.Module):
    def __init__(self, c_s: int = 384, c_z: int = 128, use_base_pair_constraints: bool = True):
        super().__init__()
        self.use_base_pair_constraints = use_base_pair_constraints
        self.ipa_layers = nn.ModuleList(
            [InvariantPointAttention(c_s=c_s, c_z=c_z) for _ in range(8)]
        )
        self.base_pair_predictor = nn.Sequential(
            nn.Linear(c_z, c_z // 2), nn.ReLU(), nn.Linear(c_z // 2, 1), nn.Sigmoid()
        )
        self.torsion_predictor = nn.Linear(c_s, 7)

    def apply_base_pair_constraints(self, z: Tensor, bp_probs: Tensor) -> Tensor:
        return z + bp_probs.unsqueeze(-1)

    def predict_secondary_structure(self, z: Tensor) -> Dict:
        probs = self.base_pair_predictor(z).squeeze(-1)
        return {"dot_bracket": "." * z.shape[1], "pairing_scores": probs, "pseudoknot": False}
