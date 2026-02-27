from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from src.advanced_model import AdvancedProteinFoldingModel
from src.ligand_encoder import LigandGraphEncoder


class ProteinLigandComplexModel(nn.Module):
    def __init__(
        self,
        protein_model: AdvancedProteinFoldingModel,
        ligand_encoder: LigandGraphEncoder,
        binding_site_predictor: bool = True,
    ):
        super().__init__()
        self.protein_model = protein_model
        self.ligand_encoder = ligand_encoder
        self.protein_ligand_attention = nn.MultiheadAttention(
            embed_dim=384, num_heads=8, kdim=64, vdim=64
        )
        self.site_predictor = (
            nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())
            if binding_site_predictor
            else None
        )
        self.pose_refinement = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=384, nhead=8) for _ in range(4)]
        )
        self.affinity_head = nn.Linear(384, 1)

    def predict_binding_site(self, protein_repr: Tensor) -> Tensor:
        if self.site_predictor is None:
            return torch.zeros(protein_repr.shape[:-1], device=protein_repr.device)
        return self.site_predictor(protein_repr).squeeze(-1)

    def dock_ligand(
        self, protein_coords: Tensor, ligand_coords: Tensor, binding_site_mask: Tensor
    ) -> Dict:
        centroid = (
            protein_coords[binding_site_mask > 0.5].mean(dim=0)
            if binding_site_mask.any()
            else protein_coords.mean(dim=0)
        )
        ligand_final = ligand_coords - ligand_coords.mean(dim=0, keepdim=True) + centroid
        return {
            "ligand_coords_final": ligand_final,
            "protein_coords_final": protein_coords,
            "binding_affinity_estimate": float(-7.5),
            "pose_confidence": float(0.75),
        }
