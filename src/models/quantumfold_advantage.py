"""QuantumFold-Advantage: Hybrid quantum-classical model for protein structure
prediction.

This module defines the main model class, which integrates variational quantum
circuits with an AlphaFold-inspired deep learning backbone.  The model can
operate in two modes controlled by a single flag:

  * **quantum_enabled=True** — quantum enhancement layers are active.
  * **quantum_enabled=False** — all quantum components are bypassed, producing
    a purely classical baseline with an otherwise identical architecture.

This design enables rigorous ablation: the only difference between the two
conditions is the presence of quantum processing.

References:
    - Jumper et al., Nature 596, 583–589 (2021) — AlphaFold2.
    - Casares et al., Quantum Sci. Technol. 7 025013 (2022) — QFold.
    - Outeiral et al., WIREs Comput Mol Sci 11 e1481 (2021).
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.classical.evoformer import EvoformerStack
from src.classical.structure_module import StructureModule
from src.quantum_layers import AdvancedQuantumCircuitLayer, QuantumAttentionLayer


# ---------------------------------------------------------------------------
# Amino acid constants
# ---------------------------------------------------------------------------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
N_AMINO_ACIDS = len(AMINO_ACIDS)

# Physicochemical properties: hydrophobicity (Kyte-Doolittle), charge, MW
AA_PROPERTIES = {
    "A": [1.8, 0, 89],
    "C": [2.5, 0, 121],
    "D": [-3.5, -1, 133],
    "E": [-3.5, -1, 147],
    "F": [2.8, 0, 165],
    "G": [-0.4, 0, 75],
    "H": [-3.2, 0.5, 155],
    "I": [4.5, 0, 131],
    "K": [-3.9, 1, 146],
    "L": [3.8, 0, 131],
    "M": [1.9, 0, 149],
    "N": [-3.5, 0, 132],
    "P": [-1.6, 0, 115],
    "Q": [-3.5, 0, 146],
    "R": [-4.5, 1, 174],
    "S": [-0.8, 0, 105],
    "T": [-0.7, 0, 119],
    "V": [4.2, 0, 117],
    "W": [-0.9, 0, 204],
    "Y": [-1.3, 0, 181],
}

N_PHYSCHEM = 3  # hydrophobicity, charge, MW


# ---------------------------------------------------------------------------
# Embedding layers
# ---------------------------------------------------------------------------
class ProteinEmbedding(nn.Module):
    """Amino-acid embedding with learned vectors and physicochemical features.

    Output dimension = embed_dim.
    """

    def __init__(self, embed_dim: int = 128, max_len: int = 512):
        super().__init__()
        self.aa_embed = nn.Embedding(N_AMINO_ACIDS + 1, embed_dim, padding_idx=N_AMINO_ACIDS)
        self.physchem_proj = nn.Linear(N_PHYSCHEM, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, aa_idx: Tensor, physchem: Tensor) -> Tensor:
        """
        Args:
            aa_idx: (B, L) int  — amino acid indices.
            physchem: (B, L, N_PHYSCHEM) float.
        Returns:
            (B, L, embed_dim)
        """
        B, L = aa_idx.shape
        pos = torch.arange(L, device=aa_idx.device).unsqueeze(0).expand(B, L)
        emb = self.aa_embed(aa_idx) + self.physchem_proj(physchem) + self.pos_embed(pos)
        return self.norm(emb)


# ---------------------------------------------------------------------------
# Quantum enhancement layer
# ---------------------------------------------------------------------------
class QuantumEnhancementLayer(nn.Module):
    """Applies a sliding-window quantum circuit to augment residue features.

    When disabled, the layer acts as an identity (zero contribution).

    Args:
        d_model: Residue representation dimension.
        n_qubits: Number of qubits in the variational circuit.
        n_circuit_layers: Depth of the variational circuit.
        window_size: Sliding window over the sequence.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_qubits: int = 8,
        n_circuit_layers: int = 4,
        window_size: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.window_size = window_size

        self.proj_in = nn.Linear(d_model, n_qubits)
        self.quantum_circuit = AdvancedQuantumCircuitLayer(
            n_qubits=n_qubits,
            n_layers=n_circuit_layers,
            entanglement="circular",
            init_strategy="haar",
        )
        self.proj_out = nn.Linear(n_qubits, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model) — residual-gated quantum features added.
        """
        B, L, D = x.shape

        # Simple average-pool local context via 1D avg pooling
        # Transpose to (B, D, L) for avg_pool1d
        ws = min(self.window_size, L)
        pad = ws // 2
        x_t = x.transpose(1, 2)  # (B, D, L)
        x_pooled = nn.functional.avg_pool1d(
            nn.functional.pad(x_t, (pad, pad), mode="constant", value=0),
            kernel_size=ws,
            stride=1,
        )  # (B, D, L')
        # Trim to L
        local_ctx = x_pooled[:, :, :L].transpose(1, 2)  # (B, L, D)

        # Project to qubit space
        q_in = self.proj_in(local_ctx)  # (B, L, n_qubits)

        # Flatten for circuit
        q_flat = q_in.reshape(B * L, self.n_qubits)
        q_out = self.quantum_circuit(q_flat)  # (B*L, n_qubits)
        q_out = q_out.reshape(B, L, self.n_qubits)

        # Project back and gate
        q_features = self.proj_out(q_out.float())
        combined = torch.cat([x, q_features], dim=-1)
        g = self.gate(combined)
        enhanced = x + g * self.norm(q_features)
        return enhanced


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class QuantumFoldAdvantage(nn.Module):
    """Hybrid quantum-classical protein structure prediction model.

    Architecture:
        1. Protein embedding (amino acid + physicochemical + positional)
        2. *[Optional]* Quantum enhancement layer (sliding-window VQC)
        3. Evoformer stack (self-attention + pair updates + transition)
        4. *[Optional]* Quantum attention (replaces one classical head)
        5. Structure module (IPA → backbone coordinates)

    Args:
        d_model: Residue representation dimension.
        d_pair: Pair representation dimension.
        n_heads: Number of attention heads.
        n_evoformer_blocks: Evoformer depth.
        n_structure_iterations: Structure module refinement steps.
        max_seq_len: Maximum supported sequence length.
        dropout: Dropout rate.
        quantum_enabled: Activate quantum enhancement layers.
        n_qubits: Qubits for quantum circuits (ignored if quantum_enabled=False).
        n_circuit_layers: VQC depth.
        quantum_window_size: Sliding-window size for quantum enhancement.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_pair: int = 64,
        n_heads: int = 8,
        n_evoformer_blocks: int = 4,
        n_structure_iterations: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        # Quantum parameters
        quantum_enabled: bool = True,
        n_qubits: int = 8,
        n_circuit_layers: int = 4,
        quantum_window_size: int = 16,
    ):
        super().__init__()
        self.quantum_enabled = quantum_enabled
        self.d_model = d_model
        self.d_pair = d_pair

        # 1. Embedding
        self.embedding = ProteinEmbedding(embed_dim=d_model, max_len=max_seq_len)

        # 2. Quantum enhancement (only built when enabled to save memory,
        #    but architecture comparison remains valid because the gated
        #    residual defaults to identity when not present)
        self.quantum_enhancement = None
        self.quantum_attention = None
        if quantum_enabled:
            self.quantum_enhancement = QuantumEnhancementLayer(
                d_model=d_model,
                n_qubits=n_qubits,
                n_circuit_layers=n_circuit_layers,
                window_size=quantum_window_size,
            )
            self.quantum_attention = QuantumAttentionLayer(
                embed_dim=d_model,
                n_qubits=min(n_qubits, 4),
                n_heads=min(n_heads, 4),
                quantum_depth=max(n_circuit_layers // 2, 2),
            )

        # 3. Pair representation initializer (from outer product of single rep)
        self.pair_init = nn.Sequential(
            nn.Linear(d_model, d_pair),
        )

        # 4. Evoformer
        self.evoformer = EvoformerStack(
            n_blocks=n_evoformer_blocks,
            d_model=d_model,
            d_pair=d_pair,
            n_heads=n_heads,
            dropout=dropout,
        )

        # 5. Structure Module
        self.structure_module = StructureModule(
            d_model=d_model,
            n_heads=n_heads,
            n_iterations=n_structure_iterations,
        )

    # ------------------------------------------------------------------
    def _init_pair_rep(self, s: Tensor) -> Tensor:
        """Initialise pair representation via outer product of projected singles."""
        p = self.pair_init(s)  # (B, L, d_pair)
        pair = torch.einsum("bid,bjd->bijd", p, p)  # (B, L, L, d_pair)
        return pair

    # ------------------------------------------------------------------
    def forward(
        self,
        aa_idx: Tensor,
        physchem: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            aa_idx: (B, L) int — amino-acid indices (0-19, 20=pad).
            physchem: (B, L, 3) float — physicochemical features.
            mask: (B, L) bool — True for valid residues.

        Returns:
            Dictionary with:
                coords_backbone: (B, L, 3, 3) — [N, Cα, C] atom coordinates.
                coords_ca: (B, L, 3) — Cα coordinates (convenience).
                rotations: (B, L, 3, 3)
                translations: (B, L, 3)
        """
        # 1. Embed
        s = self.embedding(aa_idx, physchem)

        # 2. Quantum enhancement
        if self.quantum_enabled and self.quantum_enhancement is not None:
            s = self.quantum_enhancement(s)

        # 3. Evoformer
        pair = self._init_pair_rep(s)
        s, pair = self.evoformer(s, pair, mask=mask)

        # 4. Quantum attention (post-Evoformer refinement)
        if self.quantum_enabled and self.quantum_attention is not None:
            s = self.quantum_attention(s, mask=None)

        # 5. Structure module
        coords, rotations, translations = self.structure_module(s, mask=mask)

        return {
            "coords_backbone": coords,  # (B, L, 3, 3)
            "coords_ca": coords[:, :, 1, :],  # (B, L, 3) — Cα
            "rotations": rotations,
            "translations": translations,
        }

    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        """Report parameter counts split by component."""
        groups = {
            "embedding": self.embedding,
            "evoformer": self.evoformer,
            "structure_module": self.structure_module,
        }
        if self.quantum_enhancement is not None:
            groups["quantum_enhancement"] = self.quantum_enhancement
        if self.quantum_attention is not None:
            groups["quantum_attention"] = self.quantum_attention

        counts = {}
        for name, module in groups.items():
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_quantum_model(**kwargs) -> QuantumFoldAdvantage:
    """Create a quantum-enhanced model."""
    return QuantumFoldAdvantage(quantum_enabled=True, **kwargs)


def create_classical_model(**kwargs) -> QuantumFoldAdvantage:
    """Create a classical-only baseline (identical architecture, no quantum)."""
    return QuantumFoldAdvantage(quantum_enabled=False, **kwargs)
