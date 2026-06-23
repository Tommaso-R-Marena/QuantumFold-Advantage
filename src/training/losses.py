"""Loss functions for protein structure prediction.

Provides FAPE (Frame Aligned Point Error), distance-matrix loss, and a
combined loss used to train QuantumFold-Advantage models.

Reference:
    Jumper et al., Nature 596, 583–589 (2021), Supplementary §1.9.
"""

import torch
import torch.nn as nn
from torch import Tensor


class FAPELoss(nn.Module):
    """Frame Aligned Point Error.

    Compares predicted and true atom positions after transforming into
    local residue frames, making the loss invariant to global
    rotation/translation.

    Args:
        d_clamp: Upper bound on per-atom loss (Å).
        eps: Numerical stability.
    """

    def __init__(self, d_clamp: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.d_clamp = d_clamp
        self.eps = eps

    def forward(
        self,
        pred_coords: Tensor,
        true_coords: Tensor,
        pred_rotations: Tensor,
        true_rotations: Tensor,
        pred_translations: Tensor,
        true_translations: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            pred/true_coords: (B, L, A, 3) — A atoms per residue.
            pred/true_rotations: (B, L, 3, 3)
            pred/true_translations: (B, L, 3)
            mask: (B, L) bool.
        Returns:
            Scalar FAPE loss.
        """
        B, L, A, _ = pred_coords.shape

        # Transform atoms into each residue's local frame
        # For computational efficiency, sample a subset of frames
        n_frames = min(L, 32)
        frame_idx = torch.linspace(0, L - 1, n_frames, device=pred_coords.device).long()

        total_loss = torch.tensor(0.0, device=pred_coords.device)
        count = 0

        for fi in frame_idx:
            # Local frame: R^T @ (x - t)
            R_pred = pred_rotations[:, fi]  # (B, 3, 3)
            t_pred = pred_translations[:, fi]  # (B, 3)
            R_true = true_rotations[:, fi]
            t_true = true_translations[:, fi]

            # Transform all atoms into frame fi
            pred_local = torch.einsum(
                "bij,blaj->blai",
                R_pred.transpose(-1, -2),
                pred_coords - t_pred.unsqueeze(1).unsqueeze(2),
            )
            true_local = torch.einsum(
                "bij,blaj->blai",
                R_true.transpose(-1, -2),
                true_coords - t_true.unsqueeze(1).unsqueeze(2),
            )

            # Per-atom distance, clamped
            dist = torch.sqrt(torch.sum((pred_local - true_local) ** 2, dim=-1) + self.eps)
            dist = torch.clamp(dist, max=self.d_clamp)

            if mask is not None:
                dist = dist * mask.unsqueeze(-1).float()
                total_loss = total_loss + dist.sum() / (mask.sum() * A + self.eps)
            else:
                total_loss = total_loss + dist.mean()
            count += 1

        return total_loss / max(count, 1)


class DistanceMatrixLoss(nn.Module):
    """MSE loss on Cα pairwise distance matrices."""

    def forward(
        self,
        pred_ca: Tensor,
        true_ca: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            pred_ca, true_ca: (B, L, 3)
            mask: (B, L) bool
        """
        pred_dm = torch.cdist(pred_ca, pred_ca)  # (B, L, L)
        true_dm = torch.cdist(true_ca, true_ca)

        if mask is not None:
            pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            diff = (pred_dm - true_dm) ** 2
            return (diff * pair_mask.float()).sum() / (pair_mask.float().sum() + 1e-8)
        return nn.functional.mse_loss(pred_dm, true_dm)


class CombinedLoss(nn.Module):
    """Weighted combination of FAPE + distance matrix losses.

    Args:
        fape_weight: Weight for FAPE loss.
        dm_weight: Weight for distance-matrix loss.
        coord_weight: Weight for direct coordinate MSE.
    """

    def __init__(
        self, fape_weight: float = 1.0, dm_weight: float = 0.5, coord_weight: float = 0.25
    ):
        super().__init__()
        self.fape = FAPELoss()
        self.dm = DistanceMatrixLoss()
        self.fape_weight = fape_weight
        self.dm_weight = dm_weight
        self.coord_weight = coord_weight

    def forward(
        self,
        pred: dict,
        true_coords: Tensor,
        true_rotations: Tensor,
        true_translations: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        loss = torch.tensor(0.0, device=true_coords.device)

        if self.fape_weight > 0:
            loss = loss + self.fape_weight * self.fape(
                pred["coords_backbone"],
                true_coords,
                pred["rotations"],
                true_rotations,
                pred["translations"],
                true_translations,
                mask,
            )
        if self.dm_weight > 0:
            true_ca = true_coords[:, :, 1, :]  # Cα
            loss = loss + self.dm_weight * self.dm(pred["coords_ca"], true_ca, mask)

        if self.coord_weight > 0:
            true_ca = true_coords[:, :, 1, :]
            if mask is not None:
                diff = ((pred["coords_ca"] - true_ca) ** 2).sum(-1)
                loss = loss + self.coord_weight * (diff * mask.float()).sum() / (
                    mask.float().sum() + 1e-8
                )
            else:
                loss = loss + self.coord_weight * nn.functional.mse_loss(pred["coords_ca"], true_ca)
        return loss
