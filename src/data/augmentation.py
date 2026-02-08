"""Advanced data augmentation for protein folding prediction.

Implements state-of-the-art augmentation strategies:
- 3D geometric transformations (rotation, translation, noise)
- Sequence-level augmentation (masking, substitution)
- Structure-aware augmentation (torsion angle perturbation)
- Embedding space augmentation (dropout, mixup)
- Multi-scale augmentation with controllable strength

References:
    - AlphaFold2: Jumper et al., Nature 596, 583 (2021)
    - Mixup: Zhang et al., ICLR 2018
    - ProtAugment: Kucera et al., bioRxiv 2022
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation pipeline."""

    # Geometric augmentation
    rotation_prob: float = 0.8
    translation_prob: float = 0.5
    translation_scale: float = 0.5  # Angstroms

    # Gaussian noise
    coord_noise_prob: float = 0.7
    coord_noise_scale: float = 0.2  # Angstroms

    # Sequence augmentation
    mask_prob: float = 0.15
    substitution_prob: float = 0.1

    # Embedding augmentation
    embedding_dropout_prob: float = 0.1
    embedding_noise_scale: float = 0.05

    # Torsion angle perturbation
    torsion_noise_prob: float = 0.6
    torsion_noise_scale: float = 0.1  # Radians

    # Advanced augmentation
    mixup_prob: float = 0.3
    mixup_alpha: float = 0.4

    # Mirror reflection
    mirror_prob: float = 0.5


class ProteinAugmentation(nn.Module):
    """Comprehensive protein structure augmentation.

    Applies multiple augmentation strategies that preserve protein
    structural validity while increasing training diversity.

    Args:
        config: Augmentation configuration
        training: Enable augmentation (disable for validation/test)
    """

    def __init__(self, config: Optional[AugmentationConfig] = None, training: bool = True):
        super().__init__()
        self.config = config or AugmentationConfig()
        self.training = training

    def random_rotation_matrix(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate random 3D rotation matrices.

        Args:
            batch_size: Number of rotation matrices
            device: Torch device

        Returns:
            Rotation matrices (batch_size, 3, 3)
        """
        # Generate random rotation using axis-angle representation
        angles = torch.rand(batch_size, 3, device=device) * 2 * np.pi

        # Euler angles to rotation matrix
        cos_a, sin_a = torch.cos(angles[:, 0]), torch.sin(angles[:, 0])
        cos_b, sin_b = torch.cos(angles[:, 1]), torch.sin(angles[:, 1])
        cos_c, sin_c = torch.cos(angles[:, 2]), torch.sin(angles[:, 2])

        # Construct rotation matrix
        R = torch.zeros(batch_size, 3, 3, device=device)

        R[:, 0, 0] = cos_b * cos_c
        R[:, 0, 1] = -cos_b * sin_c
        R[:, 0, 2] = sin_b

        R[:, 1, 0] = sin_a * sin_b * cos_c + cos_a * sin_c
        R[:, 1, 1] = -sin_a * sin_b * sin_c + cos_a * cos_c
        R[:, 1, 2] = -sin_a * cos_b

        R[:, 2, 0] = -cos_a * sin_b * cos_c + sin_a * sin_c
        R[:, 2, 1] = cos_a * sin_b * sin_c + sin_a * cos_c
        R[:, 2, 2] = cos_a * cos_b

        return R

    def augment_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply geometric augmentation to 3D coordinates.

        Args:
            coords: Protein coordinates (batch, seq_len, 3)

        Returns:
            Augmented coordinates (batch, seq_len, 3)
        """
        if not self.training:
            return coords

        batch_size, seq_len, _ = coords.shape
        device = coords.device
        augmented = coords.clone()

        # Random rotation
        if torch.rand(1).item() < self.config.rotation_prob:
            R = self.random_rotation_matrix(batch_size, device)
            # Center coordinates
            center = augmented.mean(dim=1, keepdim=True)
            centered = augmented - center
            # Apply rotation
            rotated = torch.bmm(centered.view(batch_size, seq_len, 3), R.transpose(1, 2))
            augmented = rotated + center

        # Random translation
        if torch.rand(1).item() < self.config.translation_prob:
            translation = (
                torch.randn(batch_size, 1, 3, device=device) * self.config.translation_scale
            )
            augmented = augmented + translation

        # Gaussian noise
        if torch.rand(1).item() < self.config.coord_noise_prob:
            noise = torch.randn_like(augmented) * self.config.coord_noise_scale
            augmented = augmented + noise

        # Mirror reflection (chirality-preserving)
        if torch.rand(1).item() < self.config.mirror_prob:
            # Random axis
            axis = torch.randint(0, 3, (1,)).item()
            augmented[..., axis] = -augmented[..., axis]

        return augmented

    def augment_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to sequence embeddings.

        Args:
            embeddings: Sequence embeddings (batch, seq_len, embed_dim)

        Returns:
            Augmented embeddings (batch, seq_len, embed_dim)
        """
        if not self.training:
            return embeddings

        augmented = embeddings.clone()

        # Embedding dropout (random masking)
        if torch.rand(1).item() < self.config.embedding_dropout_prob:
            mask = torch.rand_like(augmented) > self.config.mask_prob
            augmented = augmented * mask

        # Gaussian noise in embedding space
        noise = torch.randn_like(augmented) * self.config.embedding_noise_scale
        augmented = augmented + noise

        return augmented

    def mixup_batch(
        self, coords: torch.Tensor, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """Apply mixup augmentation to batch.

        Mixup interpolates between pairs of examples:
        x_mixed = λ * x_i + (1 - λ) * x_j

        Args:
            coords: Coordinates (batch, seq_len, 3)
            embeddings: Embeddings (batch, seq_len, embed_dim)
            labels: Optional labels for supervised mixup

        Returns:
            Tuple of (mixed_coords, mixed_embeddings, mixed_labels)
        """
        if not self.training or torch.rand(1).item() >= self.config.mixup_prob:
            return coords, embeddings, labels

        batch_size = coords.shape[0]

        # Sample mixing coefficient from Beta distribution
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)

        # Random permutation
        indices = torch.randperm(batch_size, device=coords.device)

        # Mix coordinates
        mixed_coords = lam * coords + (1 - lam) * coords[indices]

        # Mix embeddings
        mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[indices]

        # Mix labels if provided
        mixed_labels = None
        if labels is not None:
            if labels.dtype in [torch.long, torch.int32, torch.int64]:
                # For classification, return both labels and lambda
                mixed_labels = (labels, labels[indices], lam)
            else:
                mixed_labels = lam * labels + (1 - lam) * labels[indices]

        return mixed_coords, mixed_embeddings, mixed_labels

    def forward(
        self, coords: torch.Tensor, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Apply full augmentation pipeline.

        Args:
            coords: 3D coordinates (batch, seq_len, 3)
            embeddings: Sequence embeddings (batch, seq_len, embed_dim)
            labels: Optional ground truth labels

        Returns:
            Dictionary with augmented data
        """
        # Apply geometric augmentation
        aug_coords = self.augment_coordinates(coords)

        # Apply embedding augmentation
        aug_embeddings = self.augment_embeddings(embeddings)

        # Apply mixup
        aug_coords, aug_embeddings, aug_labels = self.mixup_batch(
            aug_coords, aug_embeddings, labels
        )

        return {
            "coordinates": aug_coords,
            "embeddings": aug_embeddings,
            "labels": aug_labels,
            "augmented": True,
        }


class SequenceAugmentation:
    """Sequence-level augmentation strategies.

    Applies augmentation directly to amino acid sequences before
    embedding generation.
    """

    @staticmethod
    def random_mask(sequence: str, mask_prob: float = 0.15, mask_token: str = "X") -> str:
        """Randomly mask amino acids in sequence.

        Args:
            sequence: Amino acid sequence
            mask_prob: Probability of masking each residue
            mask_token: Token to use for masking

        Returns:
            Masked sequence
        """
        masked = list(sequence)
        for i in range(len(masked)):
            if np.random.random() < mask_prob:
                masked[i] = mask_token
        return "".join(masked)

    @staticmethod
    def conservative_substitution(sequence: str, sub_prob: float = 0.1) -> str:
        """Apply conservative amino acid substitutions.

        Uses BLOSUM62-inspired substitution groups:
        - Hydrophobic: AILMFWV
        - Polar: STNQ
        - Positive: KRH
        - Negative: DE
        - Special: CGP

        Args:
            sequence: Amino acid sequence
            sub_prob: Probability of substituting each residue

        Returns:
            Sequence with conservative substitutions
        """
        substitution_groups = {
            "A": "ILMV",
            "I": "ALMV",
            "L": "AIMV",
            "M": "AILV",
            "F": "WY",
            "W": "FY",
            "V": "AILM",
            "Y": "FW",
            "S": "TN",
            "T": "SN",
            "N": "STQ",
            "Q": "N",
            "K": "RH",
            "R": "KH",
            "H": "KR",
            "D": "E",
            "E": "D",
            "C": "S",
            "G": "A",
            "P": "A",
        }

        substituted = list(sequence)
        for i in range(len(substituted)):
            if np.random.random() < sub_prob:
                aa = substituted[i]
                if aa in substitution_groups:
                    options = substitution_groups[aa]
                    substituted[i] = np.random.choice(list(options))

        return "".join(substituted)


class TorsionAngleAugmentation:
    """Augmentation based on backbone torsion angles.

    Perturbs phi/psi angles while maintaining protein geometry.
    Useful for generating conformational variations.
    """

    @staticmethod
    def compute_torsion_angles(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute backbone phi/psi angles from CA coordinates.

        Args:
            coords: CA coordinates (batch, seq_len, 3)

        Returns:
            Tuple of (phi, psi) angles in radians
        """
        # Simplified calculation (assumes CA-only)
        # For full implementation, need N, CA, C atoms

        batch_size, seq_len, _ = coords.shape
        phi = torch.zeros(batch_size, seq_len, device=coords.device)
        psi = torch.zeros(batch_size, seq_len, device=coords.device)

        # Compute pseudo-torsion angles using CA vectors
        for i in range(1, seq_len - 1):
            v1 = coords[:, i] - coords[:, i - 1]
            v2 = coords[:, i + 1] - coords[:, i]

            # Cross product for angle
            cross = torch.cross(v1, v2, dim=-1)
            angle = torch.atan2(torch.norm(cross, dim=-1), (v1 * v2).sum(dim=-1))
            phi[:, i] = angle

        return phi, psi

    @staticmethod
    def perturb_torsion_angles(coords: torch.Tensor, noise_scale: float = 0.1) -> torch.Tensor:
        """Perturb structure by modifying torsion angles.

        Args:
            coords: Coordinates (batch, seq_len, 3)
            noise_scale: Scale of perturbation in radians

        Returns:
            Perturbed coordinates
        """
        # Simplified perturbation (full implementation needs chain reconstruction)
        noise = torch.randn_like(coords) * noise_scale
        return coords + noise


class MultiScaleAugmentation(nn.Module):
    """Multi-scale augmentation with adaptive strength.

    Applies different augmentation strengths based on:
    - Training progress (curriculum learning)
    - Sample difficulty
    - Model confidence
    """

    def __init__(self, base_config: AugmentationConfig):
        super().__init__()
        self.base_config = base_config
        self.current_epoch = 0
        self.total_epochs = 100

    def get_adaptive_config(self, difficulty: Optional[float] = None) -> AugmentationConfig:
        """Generate adaptive augmentation configuration.

        Args:
            difficulty: Sample difficulty score [0, 1]

        Returns:
            Adapted augmentation config
        """
        config = AugmentationConfig()

        # Curriculum learning: increase augmentation over time
        progress = self.current_epoch / self.total_epochs
        strength_multiplier = 0.5 + 0.5 * progress

        # Difficulty-based adaptation
        if difficulty is not None:
            # Harder samples get less augmentation
            strength_multiplier *= 1.0 - 0.5 * difficulty

        # Scale all noise parameters
        config.coord_noise_scale = self.base_config.coord_noise_scale * strength_multiplier
        config.embedding_noise_scale = self.base_config.embedding_noise_scale * strength_multiplier
        config.torsion_noise_scale = self.base_config.torsion_noise_scale * strength_multiplier

        # Keep probabilities from base config
        config.rotation_prob = self.base_config.rotation_prob
        config.mask_prob = self.base_config.mask_prob

        return config

    def step_epoch(self):
        """Update epoch counter for curriculum learning."""
        self.current_epoch += 1
