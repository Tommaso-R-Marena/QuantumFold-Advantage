"""Advanced data augmentation for protein structures.

Provides:
- 3D rotation equivariant transforms
- Sequence masking strategies
- Structure perturbations
- Embedding noise injection
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
import numpy as np


class ProteinAugmentation:
    """Protein-specific data augmentation."""

    @staticmethod
    def random_rotation_3d(
        coords: Tensor,
        max_angle: float = np.pi,
    ) -> Tensor:
        """Apply random 3D rotation to coordinates.

        Args:
            coords: Coordinates (batch, n_res, 3)
            max_angle: Maximum rotation angle in radians

        Returns:
            Rotated coordinates
        """
        batch_size = coords.shape[0]
        device = coords.device

        # Random rotation angles
        angles = torch.rand(batch_size, 3, device=device) * max_angle

        # Rotation matrices
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)

        # X-axis rotation
        R_x = torch.zeros(batch_size, 3, 3, device=device)
        R_x[:, 0, 0] = 1
        R_x[:, 1, 1] = cos_a[:, 0]
        R_x[:, 1, 2] = -sin_a[:, 0]
        R_x[:, 2, 1] = sin_a[:, 0]
        R_x[:, 2, 2] = cos_a[:, 0]

        # Y-axis rotation
        R_y = torch.zeros(batch_size, 3, 3, device=device)
        R_y[:, 0, 0] = cos_a[:, 1]
        R_y[:, 0, 2] = sin_a[:, 1]
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -sin_a[:, 1]
        R_y[:, 2, 2] = cos_a[:, 1]

        # Z-axis rotation
        R_z = torch.zeros(batch_size, 3, 3, device=device)
        R_z[:, 0, 0] = cos_a[:, 2]
        R_z[:, 0, 1] = -sin_a[:, 2]
        R_z[:, 1, 0] = sin_a[:, 2]
        R_z[:, 1, 1] = cos_a[:, 2]
        R_z[:, 2, 2] = 1

        # Combined rotation
        R = torch.bmm(torch.bmm(R_z, R_y), R_x)

        # Apply rotation
        rotated_coords = torch.bmm(coords, R.transpose(1, 2))

        return rotated_coords

    @staticmethod
    def random_translation(
        coords: Tensor,
        max_translation: float = 5.0,
    ) -> Tensor:
        """Apply random translation to coordinates.

        Args:
            coords: Coordinates (batch, n_res, 3)
            max_translation: Maximum translation distance

        Returns:
            Translated coordinates
        """
        batch_size = coords.shape[0]
        device = coords.device

        # Random translation vector
        translation = (torch.rand(batch_size, 1, 3, device=device) - 0.5) * 2 * max_translation

        return coords + translation

    @staticmethod
    def add_gaussian_noise(
        tensor: Tensor,
        noise_std: float = 0.1,
    ) -> Tensor:
        """Add Gaussian noise to tensor.

        Args:
            tensor: Input tensor
            noise_std: Standard deviation of noise

        Returns:
            Noisy tensor
        """
        noise = torch.randn_like(tensor) * noise_std
        return tensor + noise

    @staticmethod
    def sequence_masking(
        embeddings: Tensor,
        mask_prob: float = 0.15,
        mask_value: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """Apply random masking to sequence embeddings.

        Args:
            embeddings: Sequence embeddings (batch, seq_len, dim)
            mask_prob: Probability of masking each position
            mask_value: Value to use for masked positions

        Returns:
            Tuple of (masked_embeddings, mask)
        """
        batch_size, seq_len, _ = embeddings.shape
        device = embeddings.device

        # Random mask
        mask = torch.rand(batch_size, seq_len, device=device) < mask_prob

        # Apply mask
        masked_embeddings = embeddings.clone()
        masked_embeddings[mask] = mask_value

        return masked_embeddings, mask.float()

    @staticmethod
    def crop_sequence(
        embeddings: Tensor,
        coords: Optional[Tensor] = None,
        min_length: int = 20,
        max_length: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Randomly crop sequence to random length.

        Args:
            embeddings: Sequence embeddings (batch, seq_len, dim)
            coords: Optional coordinates (batch, seq_len, 3)
            min_length: Minimum crop length
            max_length: Maximum crop length (None for seq_len)

        Returns:
            Tuple of (cropped_embeddings, cropped_coords)
        """
        batch_size, seq_len, _ = embeddings.shape

        if max_length is None:
            max_length = seq_len

        # Random crop length
        crop_len = torch.randint(min_length, max_length + 1, (1,)).item()

        # Random start position
        max_start = seq_len - crop_len
        start_idx = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0

        # Crop
        cropped_embeddings = embeddings[:, start_idx:start_idx + crop_len]
        cropped_coords = coords[:, start_idx:start_idx + crop_len] if coords is not None else None

        return cropped_embeddings, cropped_coords


class AugmentationPipeline:
    """Composable augmentation pipeline."""

    def __init__(
        self,
        rotation_prob: float = 0.5,
        translation_prob: float = 0.3,
        noise_prob: float = 0.3,
        noise_std: float = 0.1,
        mask_prob: float = 0.0,
    ):
        self.rotation_prob = rotation_prob
        self.translation_prob = translation_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.aug = ProteinAugmentation()

    def __call__(
        self,
        embeddings: Tensor,
        coords: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Apply augmentation pipeline.

        Args:
            embeddings: Sequence embeddings
            coords: Optional coordinates

        Returns:
            Augmented embeddings and coordinates
        """
        # Coordinate augmentations
        if coords is not None:
            if torch.rand(1).item() < self.rotation_prob:
                coords = self.aug.random_rotation_3d(coords)

            if torch.rand(1).item() < self.translation_prob:
                coords = self.aug.random_translation(coords)

        # Embedding augmentations
        if torch.rand(1).item() < self.noise_prob:
            embeddings = self.aug.add_gaussian_noise(embeddings, self.noise_std)

        if self.mask_prob > 0:
            embeddings, _ = self.aug.sequence_masking(embeddings, self.mask_prob)

        return embeddings, coords
