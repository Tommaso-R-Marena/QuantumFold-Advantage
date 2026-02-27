"""Advanced protein folding model architecture.

Implements state-of-the-art components:
- Invariant Point Attention (IPA) from AlphaFold-3
- Equivariant Graph Neural Networks (EGNN)
- Structure Module with iterative refinement
- Confidence prediction head (pLDDT)
- Hybrid quantum-classical integration

References:
    - AlphaFold-3: Abramson et al., Nature (2024) DOI: 10.1038/s41586-024-07487-w
    - IPA: Jumper et al., Nature (2021) DOI: 10.1038/s41586-021-03819-2
    - EGNN: Satorras et al., ICML (2021) arXiv:2102.09844
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InvariantPointAttention(nn.Module):
    """Invariant Point Attention from AlphaFold.

    IPA is rotation and translation equivariant, making it ideal for
    processing protein structures in 3D space.

    Args:
        c_s: Single representation dimension
        c_z: Pair representation dimension
        c_hidden: Hidden dimension
        n_heads: Number of attention heads
        n_query_points: Number of query points per head
        n_point_values: Number of value points per head
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_hidden: int = 16,
        n_heads: int = 12,
        n_query_points: int = 4,
        n_point_values: int = 8,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.n_heads = n_heads
        self.n_query_points = n_query_points
        self.n_point_values = n_point_values

        # Scalar projections
        self.linear_q = nn.Linear(c_s, n_heads * c_hidden)
        self.linear_k = nn.Linear(c_s, n_heads * c_hidden)
        self.linear_v = nn.Linear(c_s, n_heads * c_hidden)

        # Point projections
        self.linear_q_points = nn.Linear(c_s, n_heads * n_query_points * 3)
        self.linear_k_points = nn.Linear(c_s, n_heads * n_query_points * 3)
        self.linear_v_points = nn.Linear(c_s, n_heads * n_point_values * 3)

        # Pair bias
        self.linear_b = nn.Linear(c_z, n_heads)

        # Output projection
        output_dim = n_heads * (c_hidden + n_point_values * 4 + c_z)
        self.linear_out = nn.Linear(output_dim, c_s)

        # Learnable weights
        self.head_weights = nn.Parameter(torch.ones(n_heads))

        # Softplus for positive weights
        self.softplus = nn.Softplus()

    def forward(
        self, s: Tensor, z: Tensor, rigids: Tuple[Tensor, Tensor], mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass through IPA.

        Args:
            s: Single representation (batch, N, c_s)
            z: Pair representation (batch, N, N, c_z)
            rigids: (translations, rotations) defining local frames
                   translations: (batch, N, 3)
                   rotations: (batch, N, 3, 3)
            mask: Residue mask (batch, N)

        Returns:
            Updated single representation (batch, N, c_s)
        """
        batch_size, n_res, _ = s.shape
        translations, rotations = rigids

        # Scalar attention
        q_scalar = self.linear_q(s).view(batch_size, n_res, self.n_heads, self.c_hidden)
        k_scalar = self.linear_k(s).view(batch_size, n_res, self.n_heads, self.c_hidden)
        v_scalar = self.linear_v(s).view(batch_size, n_res, self.n_heads, self.c_hidden)

        # Point attention queries/keys
        q_points = self.linear_q_points(s).view(
            batch_size, n_res, self.n_heads, self.n_query_points, 3
        )
        k_points = self.linear_k_points(s).view(
            batch_size, n_res, self.n_heads, self.n_query_points, 3
        )

        # Transform points to global frame
        q_points_global = torch.einsum(
            "bnij,bnhpj->bnhpi", rotations, q_points
        ) + translations.unsqueeze(2).unsqueeze(2)
        k_points_global = torch.einsum(
            "bnij,bnhpj->bnhpi", rotations, k_points
        ) + translations.unsqueeze(2).unsqueeze(2)

        # Compute attention scores
        # Scalar component
        attn_scalar = torch.einsum("bnhc,bmhc->bnmh", q_scalar, k_scalar) / math.sqrt(self.c_hidden)

        # Point component (distance-based)
        # Shape: (batch, N, N, heads, points)
        point_distances = torch.sum(
            (q_points_global.unsqueeze(2) - k_points_global.unsqueeze(1)) ** 2, dim=-1
        )  # (batch, N, N, heads, points)

        # Average over points
        attn_points = -0.5 * torch.sum(point_distances, dim=-1)  # (batch, N, N, heads)
        attn_points = attn_points.permute(0, 1, 2, 3)  # Already correct shape

        # Pair bias
        b = self.linear_b(z)  # (batch, N, N, heads)

        # Combine attention components
        attn = attn_scalar.permute(0, 1, 2, 3) + attn_points + b

        # Apply mask
        if mask is not None:
            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # (batch, N, N)
            attn = attn.masked_fill(mask_2d.unsqueeze(-1) == 0, float("-inf"))

        # Softmax
        attn_weights = F.softmax(attn, dim=2)  # (batch, N, N, heads)

        # Apply attention to values
        # Scalar values
        v_out_scalar = torch.einsum("bnmh,bmhc->bnhc", attn_weights, v_scalar)

        # Point values
        v_points = self.linear_v_points(s).view(
            batch_size, n_res, self.n_heads, self.n_point_values, 3
        )
        v_points_global = torch.einsum(
            "bnij,bnhpj->bnhpi", rotations, v_points
        ) + translations.unsqueeze(2).unsqueeze(2)

        v_out_points = torch.einsum("bnmh,bmhpi->bnhpi", attn_weights, v_points_global)

        # Transform back to local frame
        v_out_points_local = torch.einsum(
            "bnij,bnhpj->bnhpi",
            rotations.transpose(-2, -1),
            v_out_points - translations.unsqueeze(2).unsqueeze(2),
        )

        # Point features: concatenate coordinates and norms
        v_out_points_flat = v_out_points_local.reshape(batch_size, n_res, self.n_heads, -1)
        point_norms = torch.norm(v_out_points_local, dim=-1)  # (batch, N, heads, points)

        # Pair features (sum over attended positions)
        pair_features = torch.einsum("bnmh,bnmc->bnhc", attn_weights, z)

        # Concatenate all features
        features = torch.cat([v_out_scalar, v_out_points_flat, point_norms, pair_features], dim=-1)

        # Flatten heads
        features = features.reshape(batch_size, n_res, -1)

        # Output projection
        output = self.linear_out(features)

        return output


class StructureModule(nn.Module):
    """Structure prediction module with iterative refinement.

    Args:
        c_s: Single representation dimension
        c_z: Pair representation dimension
        n_layers: Number of refinement iterations
    """

    def __init__(self, c_s: int = 384, c_z: int = 128, n_layers: int = 8):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.n_layers = n_layers

        # IPA layers
        self.ipa_layers = nn.ModuleList(
            [InvariantPointAttention(c_s=c_s, c_z=c_z) for _ in range(n_layers)]
        )

        # Transition layers
        self.transitions = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(c_s, c_s * 4), nn.ReLU(), nn.Linear(c_s * 4, c_s))
                for _ in range(n_layers)
            ]
        )

        # Update layers for backbone
        self.backbone_update = nn.ModuleList(
            [
                nn.Linear(c_s, 6)  # 3 for translation, 3 for rotation (axis-angle)
                for _ in range(n_layers)
            ]
        )

        # Normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(c_s) for _ in range(n_layers * 2)])

    def _compute_rigids(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute local coordinate frames from CA positions.

        Args:
            coords: CA coordinates (batch, N, 3)

        Returns:
            (translations, rotations)
        """
        batch_size, n_res, _ = coords.shape

        # Use CA positions as translations
        translations = coords

        # Compute rotation matrices from consecutive CAs
        rotations = (
            torch.eye(3, device=coords.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, n_res, 1, 1)
        )

        for i in range(n_res - 2):
            # Local x-axis: CA(i) -> CA(i+1)
            x_axis = coords[:, i + 1] - coords[:, i]
            x_axis = F.normalize(x_axis, dim=-1)

            # Local y-axis: perpendicular to x in plane with CA(i+2)
            v = coords[:, i + 2] - coords[:, i]
            y_axis = v - torch.sum(v * x_axis, dim=-1, keepdim=True) * x_axis
            y_axis = F.normalize(y_axis, dim=-1)

            # Local z-axis: cross product
            z_axis = torch.cross(x_axis, y_axis, dim=-1)

            # Stack into rotation matrix
            rotations[:, i] = torch.stack([x_axis, y_axis, z_axis], dim=-1)

        return translations, rotations

    def forward(
        self, s: Tensor, z: Tensor, initial_coords: Tensor, mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Iterative structure refinement.

        Args:
            s: Single representation (batch, N, c_s)
            z: Pair representation (batch, N, N, c_z)
            initial_coords: Initial CA coordinates (batch, N, 3)
            mask: Residue mask

        Returns:
            Dictionary with final coordinates and intermediate predictions
        """
        coords = initial_coords
        trajectory = [coords]

        for i in range(self.n_layers):
            # Compute current rigids
            translations, rotations = self._compute_rigids(coords)

            # IPA layer
            s_ipa = self.ipa_layers[i](s, z, (translations, rotations), mask)
            s = self.layer_norms[i * 2](s + s_ipa)

            # Transition
            s_trans = self.transitions[i](s)
            s = self.layer_norms[i * 2 + 1](s + s_trans)

            # Update backbone
            updates = self.backbone_update[i](s)  # (batch, N, 6)

            # Apply updates (translation + rotation)
            translation_update = updates[..., :3]
            rotation_update = updates[..., 3:]

            # Update coordinates
            coords = coords + translation_update

            trajectory.append(coords)

        return {
            "final_coords": coords,
            "trajectory": torch.stack(trajectory, dim=1),  # (batch, n_layers+1, N, 3)
            "final_repr": s,
        }


class ConfidenceHead(nn.Module):
    """Predict per-residue confidence scores (pLDDT-style).

    Args:
        c_s: Single representation dimension
        n_bins: Number of distance bins
    """

    def __init__(self, c_s: int = 384, n_bins: int = 50):
        super().__init__()
        self.n_bins = n_bins

        self.predictor = nn.Sequential(nn.Linear(c_s, c_s), nn.ReLU(), nn.Linear(c_s, n_bins))

    def forward(self, s: Tensor) -> Tensor:
        """Predict confidence scores.

        Args:
            s: Single representation (batch, N, c_s)

        Returns:
            Confidence logits (batch, N, n_bins)
        """
        return self.predictor(s)

    def compute_plddt(self, logits: Tensor) -> Tensor:
        """Convert logits to pLDDT scores (0-100).

        Args:
            logits: Confidence logits (batch, N, n_bins)

        Returns:
            pLDDT scores (batch, N)
        """
        probs = F.softmax(logits, dim=-1)

        # Distance bins (0 to 50 Angstroms)
        bins = torch.linspace(0, 50, self.n_bins, device=logits.device)

        # Expected distance error
        expected_error = torch.sum(probs * bins.view(1, 1, -1), dim=-1)

        # Convert to pLDDT (100 - scaled error)
        plddt = 100.0 * torch.clamp(1.0 - expected_error / 15.0, 0, 1)

        return plddt


class AdvancedProteinFoldingModel(nn.Module):
    """Advanced protein folding model with all components.

    Integrates:
    - Pre-trained embeddings
    - Invariant Point Attention
    - Structure module with iterative refinement
    - Confidence prediction
    - Optional quantum enhancement

    Args:
        input_dim: Input feature dimension
        c_s: Single representation dimension
        c_z: Pair representation dimension
        n_structure_layers: Structure module depth
        use_quantum: Enable quantum enhancement
    """

    def __init__(
        self,
        input_dim: int = 1280,  # ESM-2 embedding size
        c_s: int = 384,
        c_z: int = 128,
        n_structure_layers: int = 8,
        use_quantum: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.c_s = c_s
        self.c_z = c_z
        self.use_quantum = use_quantum

        # Input embedding
        self.input_proj = nn.Linear(input_dim, c_s)

        # Pair representation initialization
        self.pair_embed = nn.Sequential(
            nn.Linear(c_s * 2 + 1, c_z), nn.ReLU(), nn.Linear(c_z, c_z)  # +1 for relative position
        )

        # Quantum enhancement (optional)
        if use_quantum:
            from .quantum_layers import HybridQuantumClassicalBlock

            self.quantum_block = HybridQuantumClassicalBlock(
                in_channels=c_s,
                out_channels=c_s,
                n_qubits=8,
                quantum_depth=4,
                use_gated_fusion=True,
            )

        # Structure module - pass c_z parameter
        self.structure_module = StructureModule(c_s=c_s, c_z=c_z, n_layers=n_structure_layers)

        # Confidence head
        self.confidence_head = ConfidenceHead(c_s=c_s)

    def _init_pair_repr(self, s: Tensor) -> Tensor:
        """Initialize pair representation.

        Args:
            s: Single representation (batch, N, c_s)

        Returns:
            Pair representation (batch, N, N, c_z)
        """
        batch_size, n_res, _ = s.shape

        # Outer product
        s_i = s.unsqueeze(2).expand(-1, -1, n_res, -1)
        s_j = s.unsqueeze(1).expand(-1, n_res, -1, -1)

        # Relative position encoding
        pos = torch.arange(n_res, device=s.device).float()
        rel_pos = (pos.unsqueeze(0) - pos.unsqueeze(1)).unsqueeze(0).unsqueeze(-1)
        rel_pos = rel_pos.expand(batch_size, -1, -1, -1)

        # Combine
        pair_features = torch.cat([s_i, s_j, rel_pos], dim=-1)

        # Project
        z = self.pair_embed(pair_features)

        return z

    def _init_coordinates(self, batch_size: int, n_res: int, device: torch.device) -> Tensor:
        """Initialize coordinates as extended chain."""
        # Simple extended structure (3.8 Å spacing)
        coords = torch.zeros(batch_size, n_res, 3, device=device)
        coords[:, :, 0] = torch.arange(n_res, device=device).float() * 3.8
        return coords

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input features (batch, N, input_dim)
            mask: Residue mask (batch, N)

        Returns:
            Dictionary with predictions
        """
        batch_size, n_res, _ = x.shape

        # Project input
        s = self.input_proj(x)

        # Quantum enhancement
        if self.use_quantum:
            s = self.quantum_block(s)

        # Initialize pair representation
        z = self._init_pair_repr(s)

        # Initialize coordinates
        initial_coords = self._init_coordinates(batch_size, n_res, x.device)

        # Structure module
        structure_out = self.structure_module(s, z, initial_coords, mask)

        # Confidence prediction
        confidence_logits = self.confidence_head(structure_out["final_repr"])
        plddt = self.confidence_head.compute_plddt(confidence_logits)

        return {
            "coordinates": structure_out["final_coords"],
            "trajectory": structure_out["trajectory"],
            "confidence_logits": confidence_logits,
            "plddt": plddt,
            "single_repr": structure_out["final_repr"],
            "pair_repr": z,
        }


class InterChainIPA(nn.Module):
    """IPA variant that handles chain boundaries."""

    def __init__(self, c_s: int, c_z: int):
        super().__init__()
        self.ipa = InvariantPointAttention(c_s=c_s, c_z=c_z)

    def forward(self, s, z, rigids, chain_breaks, mask=None):
        n = s.shape[1]
        chain_ids = torch.zeros(n, device=s.device, dtype=torch.long)
        for i, brk in enumerate(chain_breaks):
            chain_ids[brk:] = i + 1
        chain_mask = (chain_ids.unsqueeze(0) == chain_ids.unsqueeze(1)).float()
        if mask is None:
            mask = torch.ones(s.shape[0], n, device=s.device)
        z = z * chain_mask.unsqueeze(0).unsqueeze(-1)
        return self.ipa(s, z, rigids, mask=mask)


class MultiChainStructureModule(nn.Module):
    """Structure module with inter-chain modeling."""

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        n_layers: int = 8,
        enable_inter_chain_attention: bool = True,
    ):
        super().__init__()
        self.intra = StructureModule(c_s=c_s, c_z=c_z, n_layers=n_layers)
        self.enable_inter_chain_attention = enable_inter_chain_attention
        self.inter_chain_ipa = nn.ModuleList([InterChainIPA(c_s=c_s, c_z=c_z) for _ in range(n_layers)])
        self.chain_break_embeddings = nn.Embedding(100, c_z)
        self.interface_predictor = nn.Sequential(
            nn.Linear(c_z, c_z // 2), nn.ReLU(), nn.Linear(c_z // 2, 1), nn.Sigmoid()
        )

    def forward(self, s: Tensor, z: Tensor, chain_breaks, mask: Optional[Tensor] = None) -> Dict:
        coords = torch.zeros(s.shape[0], s.shape[1], 3, device=s.device)
        rigids = self.intra._compute_rigids(coords)
        s_updated = s
        if self.enable_inter_chain_attention:
            for layer in self.inter_chain_ipa:
                s_updated = s_updated + layer(s_updated, z, rigids, chain_breaks=chain_breaks, mask=mask)
        interface_logits = self.interface_predictor(z).squeeze(-1)
        return {"single": s_updated, "pair": z, "interface_contacts": interface_logits}
