"""Structure Module: converts residue representations to 3D coordinates.

Implements an iterative refinement procedure inspired by AlphaFold2's
Structure Module.  Starting from identity backbone frames, the module
repeatedly applies Invariant Point Attention and feed-forward updates
to refine per-residue translations and rotations, then predicts
backbone atom positions (N, Cα, C).

Reference:
    Jumper et al., Nature 596, 583–589 (2021), Algorithm 20.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.classical.attention import InvariantPointAttention


class BackboneUpdate(nn.Module):
    """Predict rotation and translation updates from single rep."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 6),  # 3 for rotation (axis-angle), 3 for translation
        )

    def forward(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            s: (B, L, d_model)
        Returns:
            rot_update: (B, L, 3, 3) rotation update
            trans_update: (B, L, 3) translation update
        """
        out = self.net(s)
        angle_axis = out[..., :3]   # (B, L, 3)
        trans = out[..., 3:]        # (B, L, 3)

        # Convert axis-angle to rotation matrices
        rot = self._axis_angle_to_rotation(angle_axis)
        return rot, trans

    @staticmethod
    def _axis_angle_to_rotation(aa: Tensor) -> Tensor:
        """Rodrigues' formula: axis-angle -> rotation matrix.

        Args:
            aa: (*, 3)
        Returns:
            R: (*, 3, 3)
        """
        theta = torch.norm(aa, dim=-1, keepdim=True).clamp(min=1e-8)  # (*, 1)
        axis = aa / theta  # (*, 3)
        cos_t = torch.cos(theta).unsqueeze(-1)  # (*, 1, 1)
        sin_t = torch.sin(theta).unsqueeze(-1)  # (*, 1, 1)

        # Skew-symmetric matrix
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        K = torch.stack([
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ], dim=-2)  # (*, 3, 3)

        eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
        # K^2 via batch matmul — flatten all leading dims
        orig_shape = K.shape
        K_flat = K.reshape(-1, 3, 3)
        K2 = torch.bmm(K_flat, K_flat).reshape(orig_shape)
        R = eye + sin_t * K + (1 - cos_t) * K2
        return R


class TorsionAngleHead(nn.Module):
    """Predict per-residue torsion angles (phi, psi, omega)."""

    def __init__(self, d_model: int = 128, n_angles: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_angles * 2),  # sin, cos for each angle
        )
        self.n_angles = n_angles

    def forward(self, s: Tensor) -> Tensor:
        """
        Args:
            s: (B, L, d_model)
        Returns:
            angles: (B, L, n_angles, 2)  — (sin, cos) pairs
        """
        out = self.net(s)
        out = out.view(*s.shape[:2], self.n_angles, 2)
        # Normalize to unit circle
        return out / (torch.norm(out, dim=-1, keepdim=True) + 1e-8)


class StructureModule(nn.Module):
    """Iterative structure prediction module.

    Args:
        d_model: Residue representation dimension.
        n_heads: IPA attention heads.
        n_iterations: Number of refinement iterations.
        n_query_points: IPA query points per head.
        n_value_points: IPA value points per head.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_iterations: int = 4,
        n_query_points: int = 4,
        n_value_points: int = 4,
    ):
        super().__init__()
        self.n_iterations = n_iterations

        # Shared IPA layer applied at each iteration
        self.ipa = InvariantPointAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_query_points=n_query_points,
            n_value_points=n_value_points,
        )

        # Feed-forward after IPA
        self.transition = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.backbone_update = BackboneUpdate(d_model)
        self.torsion_head = TorsionAngleHead(d_model)

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

    def _init_frames(self, B: int, L: int, device: torch.device, dtype: torch.dtype):
        """Initialize backbone frames to identity."""
        rotations = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3).clone()
        translations = torch.zeros(B, L, 3, device=device, dtype=dtype)
        return rotations, translations

    def _atoms_from_frames(
        self, rotations: Tensor, translations: Tensor, torsions: Tensor
    ) -> Tensor:
        """Reconstruct backbone atoms (N, Cα, C) from frames.

        Uses idealized bond geometry:
        - N  at (-0.527, 1.359, 0.0) in local frame
        - Cα at (0.0, 0.0, 0.0) — origin
        - C  at (1.526, 0.0, 0.0) in local frame

        Args:
            rotations: (B, L, 3, 3)
            translations: (B, L, 3)
            torsions: (B, L, 3, 2) — sin/cos of phi, psi, omega

        Returns:
            atoms: (B, L, 3, 3)  — 3 backbone atoms × 3 coords each
        """
        B, L = translations.shape[:2]
        device = translations.device
        dtype = translations.dtype

        # Ideal atom positions in local frame (Angstroms)
        ideal_N = torch.tensor([-0.527, 1.359, 0.0], device=device, dtype=dtype)
        ideal_CA = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        ideal_C = torch.tensor([1.526, 0.0, 0.0], device=device, dtype=dtype)

        ideal_atoms = torch.stack([ideal_N, ideal_CA, ideal_C], dim=0)  # (3, 3)

        # Transform to global frame
        # atoms_global = R @ ideal_atoms^T + T
        atoms_local = ideal_atoms.unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)  # (B, L, 3_atoms, 3_xyz)
        atoms_global = torch.einsum("blij,blaj->blai", rotations, atoms_local) + translations.unsqueeze(2)

        return atoms_global

    def forward(
        self,
        s: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            s: Single representations (B, L, d_model).
            mask: Padding mask (B, L).

        Returns:
            coords: Backbone atom coordinates (B, L, 3, 3) — [N, Cα, C].
            rotations: Final backbone rotations (B, L, 3, 3).
            translations: Final backbone translations (B, L, 3).
        """
        B, L, _ = s.shape
        rotations, translations = self._init_frames(B, L, s.device, s.dtype)

        for _ in range(self.n_iterations):
            # IPA update
            s = self.ipa(s, translations, rotations, mask=mask)
            s = s + self.transition(s)

            # Backbone frame update
            rot_update, trans_update = self.backbone_update(s)

            # Compose with existing frames
            rotations = torch.einsum("blij,bljk->blik", rot_update, rotations)
            translations = translations + torch.einsum(
                "blij,blj->bli", rotations, trans_update
            )

        s = self.final_norm(s)
        torsions = self.torsion_head(s)
        coords = self._atoms_from_frames(rotations, translations, torsions)

        return coords, rotations, translations
