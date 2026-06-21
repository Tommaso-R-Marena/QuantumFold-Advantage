"""Simplified Evoformer-inspired processing block.

Processes single (per-residue) representations with self-attention
and transition feed-forward layers.  A lightweight analogue of the
Evoformer stack in AlphaFold2.

Reference:
    Jumper et al., Nature 596, 583–589 (2021).
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.classical.attention import MultiHeadSelfAttention


class TransitionBlock(nn.Module):
    """Two-layer feed-forward with GELU and residual connection."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


class PairUpdate(nn.Module):
    """Outer-product-mean update for pair representation from single rep.

    Given single representations s of shape (B, L, d_model), produces a
    pair-representation update of shape (B, L, L, d_pair) and applies it
    via a residual to an existing pair representation.
    """

    def __init__(self, d_model: int = 128, d_pair: int = 64, d_hidden: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.left_proj = nn.Linear(d_model, d_hidden)
        self.right_proj = nn.Linear(d_model, d_hidden)
        self.out_proj = nn.Linear(d_hidden * d_hidden, d_pair)

    def forward(self, s: Tensor, pair: Tensor) -> Tensor:
        """
        Args:
            s: (B, L, d_model)
            pair: (B, L, L, d_pair)
        Returns:
            Updated pair: (B, L, L, d_pair)
        """
        h = self.norm(s)
        left = self.left_proj(h)   # (B, L, d_hidden)
        right = self.right_proj(h)  # (B, L, d_hidden)

        # Outer product: (B, L, d_hidden) x (B, L, d_hidden) -> (B, L, L, d_hidden^2)
        outer = torch.einsum("bid,bjc->bijdc", left, right)
        B, L, _, d1, d2 = outer.shape
        outer = outer.reshape(B, L, L, d1 * d2)

        return pair + self.out_proj(outer)


class EvoformerBlock(nn.Module):
    """Single Evoformer-inspired block.

    Sequence: MSA-row attention -> pair update -> transition.

    Args:
        d_model: Residue representation dimension.
        d_pair: Pair representation dimension.
        n_heads: Attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_pair: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.pair_update = PairUpdate(d_model, d_pair)
        self.transition = TransitionBlock(d_model, expansion=4, dropout=dropout)

    def forward(
        self,
        s: Tensor,
        pair: Tensor,
        mask: Optional[Tensor] = None,
    ) -> tuple:
        """
        Args:
            s: (B, L, d_model)
            pair: (B, L, L, d_pair)
            mask: (B, L) bool
        Returns:
            (updated_s, updated_pair)
        """
        s = self.self_attn(s, mask=mask)
        pair = self.pair_update(s, pair)
        s = self.transition(s)
        return s, pair


class EvoformerStack(nn.Module):
    """Stack of Evoformer blocks.

    Args:
        n_blocks: Number of Evoformer blocks.
        d_model: Residue representation dimension.
        d_pair: Pair representation dimension.
        n_heads: Attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        d_model: int = 128,
        d_pair: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [EvoformerBlock(d_model, d_pair, n_heads, dropout) for _ in range(n_blocks)]
        )

    def forward(
        self, s: Tensor, pair: Tensor, mask: Optional[Tensor] = None
    ) -> tuple:
        for block in self.blocks:
            s, pair = block(s, pair, mask=mask)
        return s, pair
