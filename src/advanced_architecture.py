"""Advanced neural network architectures for protein structure prediction.

Implements state-of-the-art components:
- Invariant Point Attention (IPA) from AlphaFold-3
- Equivariant Graph Neural Networks (EGNN)
- Structure Module with iterative refinement
- pLDDT confidence prediction
- Multi-scale feature aggregation
- Quantum-classical hybrid integration

References:
    - AlphaFold-3: Abramson et al., Nature 630, 493 (2024)
    - IPA: Jumper et al., Nature 596, 583 (2021)
    - EGNN: Satorras et al., ICML 2021
    - pLDDT: Jumper et al., Nature 596, 583 (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class ProteinStructure:
    """Container for protein structure representations."""
    coordinates: torch.Tensor  # (n_residues, 3)
    frames: torch.Tensor  # (n_residues, 3, 3) local rotation matrices
    translations: torch.Tensor  # (n_residues, 3) local translations
    confidence: torch.Tensor  # (n_residues,) pLDDT scores


class InvariantPointAttention(nn.Module):
    """Invariant Point Attention from AlphaFold.
    
    IPA operates on points in 3D space and is invariant to global rotations
    and translations, making it ideal for protein structure prediction.
    
    Args:
        dim: Feature dimension
        n_heads: Number of attention heads
        n_query_points: Number of query points per head
        n_value_points: Number of value points per head
        
    References:
        Jumper et al., "Highly accurate protein structure prediction with AlphaFold",
        Nature 596, 583-589 (2021), Supplementary Section 1.6
    """
    
    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 8,
        n_query_points: int = 4,
        n_value_points: int = 8
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_query_points = n_query_points
        self.n_value_points = n_value_points
        self.head_dim = dim // n_heads
        
        # Projections for features
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Projections for points
        self.q_points_proj = nn.Linear(dim, n_heads * n_query_points * 3)
        self.k_points_proj = nn.Linear(dim, n_heads * n_query_points * 3)
        self.v_points_proj = nn.Linear(dim, n_heads * n_value_points * 3)
        
        # Output projection
        self.out_proj = nn.Linear(dim + n_heads * n_value_points * 3, dim)
        
        # Learnable parameters for attention logits
        self.w_c = nn.Parameter(torch.ones(n_heads))
        self.w_l = nn.Parameter(torch.ones(n_heads))
        
    def forward(
        self,
        features: torch.Tensor,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            features: Node features (batch, n_residues, dim)
            frames: Local coordinate frames (batch, n_residues, 3, 3)
            mask: Attention mask (batch, n_residues)
        
        Returns:
            (updated_features, updated_points) tuple
        """
        batch_size, n_residues, _ = features.shape
        
        # Project features for attention
        Q = self.q_proj(features).view(batch_size, n_residues, self.n_heads, self.head_dim)
        K = self.k_proj(features).view(batch_size, n_residues, self.n_heads, self.head_dim)
        V = self.v_proj(features).view(batch_size, n_residues, self.n_heads, self.head_dim)
        
        # Project features to points in local frames
        q_points = self.q_points_proj(features).view(
            batch_size, n_residues, self.n_heads, self.n_query_points, 3
        )
        k_points = self.k_points_proj(features).view(
            batch_size, n_residues, self.n_heads, self.n_query_points, 3
        )
        v_points = self.v_points_proj(features).view(
            batch_size, n_residues, self.n_heads, self.n_value_points, 3
        )
        
        # Transform points to global frame
        q_points_global = torch.einsum('bnhpc,bnic->bnhpi', q_points, frames)
        k_points_global = torch.einsum('bmhpc,bmic->bmhpi', k_points, frames)
        
        # Compute attention logits
        # Feature-based attention
        attn_logits_features = torch.einsum('bnhd,bmhd->bhnm', Q, K) / np.sqrt(self.head_dim)
        
        # Point-based attention (invariant to global transformations)
        point_diff = q_points_global.unsqueeze(3) - k_points_global.unsqueeze(2)
        # (batch, n_heads, n_residues, n_residues, n_query_points, 3)
        
        point_distances = torch.sqrt(torch.sum(point_diff ** 2, dim=-1) + 1e-8)
        # (batch, n_heads, n_residues, n_residues, n_query_points)
        
        attn_logits_points = -torch.sum(point_distances, dim=-1)
        # (batch, n_heads, n_residues, n_residues)
        
        # Combine attention logits
        attn_logits = (
            self.w_c.view(1, -1, 1, 1) * attn_logits_features +
            self.w_l.view(1, -1, 1, 1) * attn_logits_points
        )
        
        # Apply mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(-1)
            attn_logits = attn_logits.masked_fill(~mask_expanded.bool(), -1e9)
        
        # Softmax attention weights
        attn_weights = torch.softmax(attn_logits, dim=-1)
        # (batch, n_heads, n_residues, n_residues)
        
        # Apply attention to features
        attn_features = torch.einsum('bhnm,bmhd->bnhd', attn_weights, V)
        attn_features = attn_features.reshape(batch_size, n_residues, -1)
        
        # Apply attention to value points
        v_points_global = torch.einsum('bmhpc,bmic->bmhpi', v_points, frames)
        attn_points = torch.einsum('bhnm,bmhpi->bnhpi', attn_weights, v_points_global)
        attn_points_flat = attn_points.reshape(batch_size, n_residues, -1)
        
        # Combine and project
        combined = torch.cat([attn_features, attn_points_flat], dim=-1)
        output = self.out_proj(combined)
        
        return output, attn_points


class EquivariantGraphConv(nn.Module):
    """Equivariant Graph Convolutional layer.
    
    Maintains SE(3) equivariance for coordinates while processing features.
    
    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden dimension
        out_dim: Output feature dimension
        
    References:
        Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021
    """
    
    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        
        # Edge feature network
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node feature network
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Coordinate update network
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            features: Node features (batch, n_nodes, in_dim)
            coords: Node coordinates (batch, n_nodes, 3)
            edge_index: Edge connectivity (optional, uses fully connected if None)
        
        Returns:
            (updated_features, updated_coords) tuple
        """
        batch_size, n_nodes, _ = features.shape
        
        # Compute pairwise features
        coord_diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        # (batch, n_nodes, n_nodes, 3)
        
        distances = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1, keepdim=True) + 1e-8)
        # (batch, n_nodes, n_nodes, 1)
        
        # Edge features
        feat_i = features.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        feat_j = features.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        edge_features = torch.cat([feat_i, feat_j, distances], dim=-1)
        # (batch, n_nodes, n_nodes, in_dim*2+1)
        
        # Process edges
        edge_hidden = self.edge_mlp(edge_features)
        # (batch, n_nodes, n_nodes, hidden_dim)
        
        # Aggregate edge features
        aggregated = torch.mean(edge_hidden, dim=2)
        # (batch, n_nodes, hidden_dim)
        
        # Update node features
        node_input = torch.cat([features, aggregated], dim=-1)
        updated_features = self.node_mlp(node_input)
        
        # Update coordinates (equivariant)
        coord_weights = self.coord_mlp(edge_hidden)
        # (batch, n_nodes, n_nodes, 1)
        
        coord_updates = coord_weights * coord_diff
        # (batch, n_nodes, n_nodes, 3)
        
        updated_coords = coords + torch.mean(coord_updates, dim=2)
        # (batch, n_nodes, 3)
        
        return updated_features, updated_coords


class StructureModule(nn.Module):
    """Structure prediction module with iterative refinement.
    
    Args:
        dim: Feature dimension
        n_layers: Number of refinement layers
        use_ipa: Use Invariant Point Attention
        use_egnn: Use Equivariant GNN
    """
    
    def __init__(
        self,
        dim: int = 256,
        n_layers: int = 8,
        use_ipa: bool = True,
        use_egnn: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.use_ipa = use_ipa
        self.use_egnn = use_egnn
        
        # Iterative refinement layers
        self.refinement_layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict()
            
            if use_ipa:
                layer['ipa'] = InvariantPointAttention(dim=dim)
            
            if use_egnn:
                layer['egnn'] = EquivariantGraphConv(in_dim=dim, out_dim=dim)
            
            layer['norm'] = nn.LayerNorm(dim)
            layer['ffn'] = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
            
            self.refinement_layers.append(layer)
        
        # Coordinate prediction head
        self.coord_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3)
        )
        
        # Confidence prediction (pLDDT)
        self.confidence_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
    
    def initialize_frames(self, n_residues: int, device: torch.device) -> torch.Tensor:
        """Initialize local coordinate frames."""
        frames = torch.eye(3, device=device).unsqueeze(0).expand(n_residues, 3, 3)
        return frames.unsqueeze(0)  # Add batch dimension
    
    def forward(
        self,
        features: torch.Tensor,
        initial_coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> ProteinStructure:
        """Iterative structure prediction.
        
        Args:
            features: Input features (batch, n_residues, dim)
            initial_coords: Initial coordinates (optional)
            mask: Residue mask (optional)
        
        Returns:
            ProteinStructure with predictions
        """
        batch_size, n_residues, _ = features.shape
        device = features.device
        
        # Initialize coordinates if not provided
        if initial_coords is None:
            coords = torch.zeros(batch_size, n_residues, 3, device=device)
        else:
            coords = initial_coords
        
        # Initialize frames
        frames = self.initialize_frames(n_residues, device).expand(batch_size, -1, -1, -1)
        
        # Iterative refinement
        for layer in self.refinement_layers:
            # Invariant Point Attention
            if self.use_ipa and 'ipa' in layer:
                ipa_out, _ = layer['ipa'](features, frames, mask)
                features = features + ipa_out
            
            # Equivariant GNN
            if self.use_egnn and 'egnn' in layer:
                egnn_features, egnn_coords = layer['egnn'](features, coords)
                features = features + egnn_features
                coords = coords + (egnn_coords - coords) * 0.1  # Small step
            
            # Normalization
            features = layer['norm'](features)
            
            # Feed-forward
            features = features + layer['ffn'](features)
        
        # Final coordinate prediction
        coord_updates = self.coord_head(features)
        final_coords = coords + coord_updates
        
        # Confidence prediction (pLDDT-like)
        confidence = self.confidence_head(features).squeeze(-1)
        
        return ProteinStructure(
            coordinates=final_coords,
            frames=frames,
            translations=torch.zeros_like(final_coords),
            confidence=confidence
        )


class QuantumEnhancedProteinFolder(nn.Module):
    """Complete protein folding model with quantum enhancement.
    
    Integrates:
    - Pre-trained embeddings (ESM-2)
    - Quantum layers
    - Advanced structure prediction (IPA + EGNN)
    - Confidence prediction
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        n_quantum_qubits: Qubits for quantum layers
        n_structure_layers: Structure refinement layers
        use_quantum: Enable quantum enhancement
    """
    
    def __init__(
        self,
        input_dim: int = 1280,  # ESM-2 embedding size
        hidden_dim: int = 256,
        n_quantum_qubits: int = 4,
        n_structure_layers: int = 8,
        use_quantum: bool = True
    ):
        super().__init__()
        self.use_quantum = use_quantum
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Quantum enhancement (optional)
        if use_quantum:
            from quantum_layers import HybridQuantumClassicalBlock
            self.quantum_block = HybridQuantumClassicalBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                n_qubits=n_quantum_qubits,
                use_gated_fusion=True
            )
        
        # Structure prediction module
        self.structure_module = StructureModule(
            dim=hidden_dim,
            n_layers=n_structure_layers,
            use_ipa=True,
            use_egnn=True
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> ProteinStructure:
        """Forward pass.
        
        Args:
            embeddings: Pre-trained embeddings (batch, n_residues, input_dim)
            mask: Residue mask (optional)
        
        Returns:
            ProteinStructure with predictions
        """
        # Project embeddings
        features = self.input_proj(embeddings)
        
        # Quantum enhancement
        if self.use_quantum:
            features = self.quantum_block(features)
        
        # Structure prediction
        structure = self.structure_module(features, mask=mask)
        
        return structure
