"""Advanced data preprocessing for protein folding.

Provides:
- Sequence cleaning and normalization
- Structure alignment and superposition
- Data augmentation
- Feature engineering
- Batching and collation
"""

import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation


class SequenceProcessor:
    """Process and normalize protein sequences."""

    # Amino acid mappings
    AA_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}

    # Ambiguous amino acid resolution
    AMBIGUOUS_AA = {
        "B": "N",  # Asn or Asp
        "Z": "Q",  # Gln or Glu
        "X": "G",  # Unknown -> Gly (most common)
        "U": "C",  # Selenocysteine -> Cys
        "O": "K",  # Pyrrolysine -> Lys
    }

    @staticmethod
    def clean_sequence(sequence: str, replace_unknown: bool = True) -> str:
        """Clean and normalize sequence.

        Args:
            sequence: Raw amino acid sequence
            replace_unknown: Replace ambiguous amino acids

        Returns:
            Cleaned sequence
        """
        # Convert to uppercase
        sequence = sequence.upper().strip()

        # Remove whitespace and special characters
        sequence = "".join(c for c in sequence if c.isalpha())

        # Replace ambiguous amino acids
        if replace_unknown:
            for ambiguous, replacement in SequenceProcessor.AMBIGUOUS_AA.items():
                sequence = sequence.replace(ambiguous, replacement)

        # Remove invalid characters
        valid_aa = set(SequenceProcessor.AA_TO_IDX.keys())
        sequence = "".join(c for c in sequence if c in valid_aa)

        return sequence

    @staticmethod
    def encode_sequence(sequence: str) -> torch.Tensor:
        """Encode sequence as integer tensor.

        Args:
            sequence: Amino acid sequence

        Returns:
            Encoded sequence (L,)
        """
        encoded = [SequenceProcessor.AA_TO_IDX.get(aa, 0) for aa in sequence]
        return torch.tensor(encoded, dtype=torch.long)

    @staticmethod
    def decode_sequence(encoded: torch.Tensor) -> str:
        """Decode integer tensor to sequence.

        Args:
            encoded: Encoded sequence (L,)

        Returns:
            Amino acid sequence
        """
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().numpy()

        return "".join(SequenceProcessor.IDX_TO_AA.get(int(idx), "X") for idx in encoded)


class StructureAugmenter:
    """Augment protein structures for training."""

    @staticmethod
    def random_rotation(coordinates: np.ndarray) -> np.ndarray:
        """Apply random 3D rotation.

        Args:
            coordinates: Input coordinates (N, 3)

        Returns:
            Rotated coordinates (N, 3)
        """
        rotation = Rotation.random()
        return rotation.apply(coordinates)

    @staticmethod
    def random_translation(coordinates: np.ndarray, max_shift: float = 5.0) -> np.ndarray:
        """Apply random translation.

        Args:
            coordinates: Input coordinates (N, 3)
            max_shift: Maximum shift in Angstroms

        Returns:
            Translated coordinates (N, 3)
        """
        shift = np.random.uniform(-max_shift, max_shift, size=3)
        return coordinates + shift

    @staticmethod
    def add_noise(coordinates: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to coordinates.

        Args:
            coordinates: Input coordinates (N, 3)
            noise_scale: Standard deviation of noise

        Returns:
            Noisy coordinates (N, 3)
        """
        noise = np.random.normal(0, noise_scale, size=coordinates.shape)
        return coordinates + noise

    @staticmethod
    def center_coordinates(coordinates: np.ndarray) -> np.ndarray:
        """Center coordinates at origin.

        Args:
            coordinates: Input coordinates (N, 3)

        Returns:
            Centered coordinates (N, 3)
        """
        centroid = np.mean(coordinates, axis=0)
        return coordinates - centroid

    @staticmethod
    def augment_structure(
        coordinates: np.ndarray,
        rotate: bool = True,
        translate: bool = False,
        add_noise: bool = True,
        noise_scale: float = 0.1,
    ) -> np.ndarray:
        """Apply multiple augmentations.

        Args:
            coordinates: Input coordinates (N, 3)
            rotate: Apply random rotation
            translate: Apply random translation
            add_noise: Add Gaussian noise
            noise_scale: Noise standard deviation

        Returns:
            Augmented coordinates (N, 3)
        """
        coords = coordinates.copy()

        # Always center first
        coords = StructureAugmenter.center_coordinates(coords)

        if rotate:
            coords = StructureAugmenter.random_rotation(coords)

        if translate:
            coords = StructureAugmenter.random_translation(coords)

        if add_noise:
            coords = StructureAugmenter.add_noise(coords, noise_scale)

        return coords


class BatchCollator:
    """Collate batches with variable-length sequences."""

    def __init__(self, pad_value: float = 0.0, return_mask: bool = True):
        self.pad_value = pad_value
        self.return_mask = return_mask

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch items.

        Args:
            batch: List of dictionaries with 'sequence', 'coordinates', etc.

        Returns:
            Collated batch dictionary
        """
        # Find maximum length
        max_len = max(len(item["sequence"]) for item in batch)
        batch_size = len(batch)

        collated = {}

        # Pad sequences
        if "sequence" in batch[0]:
            sequences = [item["sequence"] for item in batch]
            collated["sequences"] = sequences
            collated["seq_lengths"] = torch.tensor(
                [len(seq) for seq in sequences], dtype=torch.long
            )

        # Pad embeddings
        if "embeddings" in batch[0]:
            emb_dim = batch[0]["embeddings"].shape[-1]
            padded_emb = torch.full(
                (batch_size, max_len, emb_dim), self.pad_value, dtype=torch.float32
            )

            for i, item in enumerate(batch):
                length = len(item["sequence"])
                padded_emb[i, :length] = item["embeddings"]

            collated["embeddings"] = padded_emb

        # Pad coordinates
        if "coordinates" in batch[0]:
            padded_coords = torch.full(
                (batch_size, max_len, 3), self.pad_value, dtype=torch.float32
            )

            for i, item in enumerate(batch):
                length = len(item["sequence"])
                padded_coords[i, :length] = torch.from_numpy(item["coordinates"]).float()

            collated["coordinates"] = padded_coords

        # Create attention mask
        if self.return_mask:
            mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

            for i, item in enumerate(batch):
                length = len(item["sequence"])
                mask[i, :length] = True

            collated["mask"] = mask

        return collated


def length_based_bucketing(
    items: List[Dict], bucket_size: int = 32, length_key: str = "sequence"
) -> List[List[Dict]]:
    """Group items into buckets by length for efficient batching.

    Args:
        items: List of data items
        bucket_size: Items per bucket
        length_key: Key to use for length calculation

    Returns:
        List of buckets (lists of items)
    """
    # Sort by length
    sorted_items = sorted(
        items,
        key=lambda x: (
            len(x[length_key]) if isinstance(x[length_key], (str, list)) else x[length_key].shape[0]
        ),
    )

    # Create buckets
    buckets = []
    for i in range(0, len(sorted_items), bucket_size):
        buckets.append(sorted_items[i : i + bucket_size])

    return buckets
