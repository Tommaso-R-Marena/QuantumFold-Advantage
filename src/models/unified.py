"""Unified QuantumFold model for multiple modalities.

Supports Protein, RNA, and Protein-Ligand complex structure prediction.
Integrates advanced geometric attention and optional quantum enhancement.
"""

from typing import Dict, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.advanced_model import StructureModule, ConfidenceHead
from src.quantum_layers import HybridQuantumClassicalBlock
from src.rna_embeddings import RNATokenEmbedder
from src.ligand_encoder import LigandGraphEncoder

class UnifiedQuantumFold(nn.Module):
    """Unified hybrid model for protein, RNA, and complex prediction."""

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        n_structure_layers: int = 8,
        use_quantum: bool = True,
        n_qubits: Optional[int] = None,
        quantum_depth: Optional[int] = None
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.use_quantum = use_quantum

        # Hardware-aware quantum scaling
        if use_quantum:
            self.n_qubits, self.quantum_depth = self._auto_scale_quantum_resources(n_qubits, quantum_depth)

        # Encoders for different modalities
        self.protein_encoder = None # Lazy-loaded or passed during training
        self.rna_encoder = RNATokenEmbedder(d_model=c_s)
        self.ligand_encoder = LigandGraphEncoder(node_features=c_s)

        # Unified projection layers
        self.protein_proj = nn.Linear(20, c_s) # Default to one-hot dim
        self.rna_proj = nn.Linear(c_s, c_s)

        # Quantum Block (shared across modalities)
        if use_quantum:
            self.quantum_block = HybridQuantumClassicalBlock(
                in_channels=c_s,
                out_channels=c_s,
                n_qubits=self.n_qubits,
                quantum_depth=self.quantum_depth,
                use_gated_fusion=True
            )

        # Core Structure Module
        self.structure_module = StructureModule(
            c_s=c_s,
            c_z=c_z,
            n_layers=n_structure_layers
        )

        self.confidence_head = ConfidenceHead(c_s=c_s)

        # Pair representation embedding
        self.pair_embed = nn.Sequential(
            nn.Linear(c_s * 2 + 1, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z)
        )

    def _init_pair_repr(self, s: Tensor) -> Tensor:
        b, n, _ = s.shape
        s_i = s.unsqueeze(2).expand(-1, -1, n, -1)
        s_j = s.unsqueeze(1).expand(-1, n, -1, -1)
        pos = torch.arange(n, device=s.device).float()
        rel_pos = (pos.unsqueeze(0) - pos.unsqueeze(1)).unsqueeze(0).unsqueeze(-1).expand(b, -1, -1, -1)
        return self.pair_embed(torch.cat([s_i, s_j, rel_pos], dim=-1))

    def _auto_scale_quantum_resources(self, n_qubits: Optional[int], quantum_depth: Optional[int]) -> Tuple[int, int]:
        """Automatically scale quantum resources based on hardware."""
        import psutil

        # Defaults
        default_qubits = 8
        default_depth = 4

        if torch.cuda.is_available():
            # GPU present: can handle more complex simulations
            # Use provided values or higher defaults
            final_qubits = n_qubits if n_qubits is not None else 8
            final_depth = quantum_depth if quantum_depth is not None else 4
        else:
            # CPU only: need to be more conservative
            mem = psutil.virtual_memory()
            if mem.total < 16 * 1024**3: # Less than 16GB RAM
                final_qubits = min(n_qubits if n_qubits is not None else 4, 4)
                final_depth = min(quantum_depth if quantum_depth is not None else 2, 2)
            else:
                final_qubits = min(n_qubits if n_qubits is not None else 6, 6)
                final_depth = min(quantum_depth if quantum_depth is not None else 3, 3)

        return final_qubits, final_depth

    def _init_coordinates(self, b: int, n: int, device: torch.device) -> Tensor:
        coords = torch.zeros(b, n, 3, device=device)
        coords[:, :, 0] = torch.arange(n, device=device).float() * 3.8
        return coords

    def forward(
        self,
        protein_embeddings: Optional[Tensor] = None,
        rna_sequence: Optional[str] = None,
        ligand_data: Optional[Dict] = None,
        mask: Optional[Tensor] = None,
        modality: str = "protein"
    ) -> Dict[str, Tensor]:
        """
        Forward pass for specified modality.
        """
        device = protein_embeddings.device if protein_embeddings is not None else next(self.parameters()).device

        if modality == "protein":
            if protein_embeddings is None:
                raise ValueError("protein_embeddings required for protein modality")
            s = self.protein_proj(protein_embeddings)
        elif modality == "rna":
            if rna_sequence is None:
                raise ValueError("rna_sequence required for rna modality")
            s = self.rna_encoder(rna_sequence).unsqueeze(0).to(device)
            s = self.rna_proj(s)
        elif modality == "complex":
            # Simplified complex handling: combine protein and ligand features
            # Real implementation would need cross-attention
            s = self.protein_proj(protein_embeddings)
            # Just use protein for now in this demonstration
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Ensure 3D shape (B, L, D)
        if s.ndim == 2:
            s = s.unsqueeze(0)

        b, n, _ = s.shape

        # Apply Quantum Enhancement
        if self.use_quantum:
            s = self.quantum_block(s)

        # Structure Prediction Pipeline
        z = self._init_pair_repr(s)
        init_coords = self._init_coordinates(b, n, device)

        structure_out = self.structure_module(s, z, init_coords, mask)

        confidence_logits = self.confidence_head(structure_out["final_repr"])
        plddt = self.confidence_head.compute_plddt(confidence_logits)

        results = {
            "coordinates": structure_out["final_coords"],
            "trajectory": structure_out["trajectory"],
            "plddt": plddt,
            "single_repr": structure_out["final_repr"],
            "pair_repr": z,
            "clash_penalty": structure_out["clash_penalty"],
            "bond_length_loss": structure_out["bond_length_loss"]
        }

        return results
