"""AlphaFold-inspired attention mechanisms for protein structure prediction.

Implements Multi-Head Self-Attention and a simplified Invariant Point Attention
(IPA) module that operates on residue representations and backbone frames.

References:
    - Jumper et al., "Highly accurate protein structure prediction with
      AlphaFold," Nature 596, 583–589 (2021).
    - Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with pre-LayerNorm.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate applied to attention weights.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) bool, True = keep, False = pad.
        Returns:
            (batch, seq_len, d_model)
        """
        B, L, _ = x.shape
        h = self.norm(x)

        Q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # mask: (B, L) -> (B, 1, 1, L)
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return x + self.out_proj(out)


class InvariantPointAttention(nn.Module):
    """Simplified Invariant Point Attention inspired by AlphaFold2.

    Computes attention that is equivariant to SE(3) transformations by
    operating on local point coordinates in addition to scalar features.

    Args:
        d_model: Residue representation dimension.
        n_heads: Number of attention heads.
        n_query_points: Number of 3D query points per head.
        n_value_points: Number of 3D value points per head.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_query_points: int = 4,
        n_value_points: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_query_points = n_query_points
        self.n_value_points = n_value_points

        # Scalar projections
        self.q_scalar = nn.Linear(d_model, n_heads * self.head_dim)
        self.k_scalar = nn.Linear(d_model, n_heads * self.head_dim)
        self.v_scalar = nn.Linear(d_model, n_heads * self.head_dim)

        # Point projections (3D coordinates)
        self.q_points = nn.Linear(d_model, n_heads * n_query_points * 3)
        self.k_points = nn.Linear(d_model, n_heads * n_query_points * 3)
        self.v_points = nn.Linear(d_model, n_heads * n_value_points * 3)

        # Learnable per-head weighting for point attention
        self.head_weights = nn.Parameter(torch.zeros(n_heads))

        # Output
        out_dim = n_heads * (self.head_dim + n_value_points * 3 + n_value_points)
        self.out_proj = nn.Linear(out_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        s: Tensor,
        translations: Tensor,
        rotations: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            s: Single representations (B, L, d_model).
            translations: Backbone translations (B, L, 3).
            rotations: Backbone rotations (B, L, 3, 3).
            mask: Padding mask (B, L), True = valid.

        Returns:
            Updated representations (B, L, d_model).
        """
        B, L, _ = s.shape
        h = self.norm(s)

        # Scalar attention components
        q_s = self.q_scalar(h).view(B, L, self.n_heads, self.head_dim)
        k_s = self.k_scalar(h).view(B, L, self.n_heads, self.head_dim)
        v_s = self.v_scalar(h).view(B, L, self.n_heads, self.head_dim)

        # Point attention components — transform to global frame
        q_pts = self.q_points(h).view(B, L, self.n_heads, self.n_query_points, 3)
        k_pts = self.k_points(h).view(B, L, self.n_heads, self.n_query_points, 3)
        v_pts = self.v_points(h).view(B, L, self.n_heads, self.n_value_points, 3)

        # Apply rotations to points -> global frame
        # rotations: (B, L, 3, 3), pts: (B, L, H, P, 3)
        R = rotations.unsqueeze(2).unsqueeze(3)  # (B, L, 1, 1, 3, 3)
        T = translations.unsqueeze(2).unsqueeze(3)  # (B, L, 1, 1, 3)

        q_pts_global = (
            torch.einsum(
                "blhpc,blhpcd->blhpd",
                q_pts,
                R.expand(-1, -1, self.n_heads, self.n_query_points, -1, -1),
            )
            + T
        )
        k_pts_global = (
            torch.einsum(
                "blhpc,blhpcd->blhpd",
                k_pts,
                R.expand(-1, -1, self.n_heads, self.n_query_points, -1, -1),
            )
            + T
        )
        v_pts_global = (
            torch.einsum(
                "blhpc,blhpcd->blhpd",
                v_pts,
                R.expand(-1, -1, self.n_heads, self.n_value_points, -1, -1),
            )
            + T
        )

        # Scalar attention scores
        scalar_attn = torch.einsum("bihd,bjhd->bhij", q_s, k_s) / math.sqrt(self.head_dim)

        # Point attention scores
        # Squared distances between query and key points
        q_expand = q_pts_global.unsqueeze(3)  # (B, L_q, H, 1, P, 3)
        k_expand = k_pts_global.unsqueeze(2)  # (B, 1, L_k, H, P, 3)
        # Rearrange for broadcasting
        q_for_dist = q_pts_global.permute(0, 2, 1, 3, 4)  # (B, H, L, P, 3)
        k_for_dist = k_pts_global.permute(0, 2, 1, 3, 4)  # (B, H, L, P, 3)

        pt_dists = torch.sum(
            (q_for_dist.unsqueeze(3) - k_for_dist.unsqueeze(2)) ** 2, dim=(-1, -2)
        )  # (B, H, L, L)

        w_h = torch.nn.functional.softplus(self.head_weights).view(1, self.n_heads, 1, 1)
        point_attn = -0.5 * w_h * pt_dists

        # Combined attention
        attn_logits = scalar_attn + point_attn

        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attn_logits = attn_logits.masked_fill(~attn_mask, float("-inf"))

        attn = torch.softmax(attn_logits, dim=-1)

        # Aggregate scalar values
        result_scalar = torch.einsum("bhij,bjhd->bihd", attn, v_s)
        result_scalar = result_scalar.reshape(B, L, self.n_heads * self.head_dim)

        # Aggregate point values
        v_pts_perm = v_pts_global.permute(0, 2, 1, 3, 4)  # (B, H, L, Pv, 3)
        result_pts = torch.einsum("bhij,bhjpc->bhipc", attn, v_pts_perm)
        result_pts = result_pts.permute(0, 2, 1, 3, 4)  # (B, L, H, Pv, 3)

        # Transform back to local frame
        R_inv = rotations.transpose(-1, -2).unsqueeze(2).unsqueeze(3)
        result_pts_local = torch.einsum(
            "blhpc,blhpcd->blhpd",
            result_pts - T,
            R_inv.expand(-1, -1, self.n_heads, self.n_value_points, -1, -1),
        )

        # Point norms
        result_pts_norm = torch.norm(result_pts_local, dim=-1)  # (B, L, H, Pv)

        # Concatenate all features
        result_pts_flat = result_pts_local.reshape(B, L, self.n_heads * self.n_value_points * 3)
        result_norms_flat = result_pts_norm.reshape(B, L, self.n_heads * self.n_value_points)

        output = torch.cat([result_scalar, result_pts_flat, result_norms_flat], dim=-1)
        return s + self.out_proj(output)
