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
        batch_size, n_res, n_atoms, _ = pred_coords.shape

        # Transform atoms into each residue's local frame
        # For computational efficiency, sample a subset of frames
        n_frames = min(n_res, 32)
        frame_idx = torch.linspace(0, n_res - 1, n_frames, device=pred_coords.device).long()

        # Vectorized implementation: process all sampled frames at once
        # Select sampled frames
        R_pred = pred_rotations[:, frame_idx]  # (B, F, 3, 3)
        t_pred = pred_translations[:, frame_idx]  # (B, F, 3)
        R_true = true_rotations[:, frame_idx]
        t_true = true_translations[:, frame_idx]

        # R^T: (B, F, 3, 3)
        R_pred_T = R_pred.transpose(-1, -2)
        R_true_T = R_true.transpose(-1, -2)

        # Local frame transform: x_local = R^T @ x - R^T @ t
        # (B, F, 3, 3) @ (B, L, A, 3) -> (B, F, L, A, 3)
        pred_local = torch.einsum("bfij,blaj->bflai", R_pred_T, pred_coords)
        # (B, F, 3, 3) @ (B, F, 3) -> (B, F, 3) -> (B, F, 1, 1, 3)
        pred_offset = torch.einsum("bfij,bfj->bfi", R_pred_T, t_pred).view(batch_size, -1, 1, 1, 3)
        pred_local = pred_local - pred_offset

        true_local = torch.einsum("bfij,blaj->bflai", R_true_T, true_coords)
        true_offset = torch.einsum("bfij,bfj->bfi", R_true_T, t_true).view(batch_size, -1, 1, 1, 3)
        true_local = true_local - true_offset

        # Per-atom distance, clamped
        # dist: (B, F, L, A)
        dist = torch.sqrt(torch.sum((pred_local - true_local) ** 2, dim=-1) + self.eps)
        dist = torch.clamp(dist, max=self.d_clamp)

        if mask is not None:
            # mask: (B, L) -> (B, 1, L, 1)
            dist = dist * mask.view(batch_size, 1, n_res, 1).float()
            # Average over frames of (sum over atoms / active atoms)
            loss_per_frame = dist.sum(dim=(0, 2, 3)) / (mask.sum() * n_atoms + self.eps)
            return loss_per_frame.mean()
        else:
            return dist.mean()


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
