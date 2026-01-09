"""
Data loading and synthetic generation for protein folding experiments.

Key functions:
- fetch_pdb_structures: Download small proteins from RCSB PDB
- generate_synthetic_data: Create random valid protein sequences and toy coordinates
- ProteinDataset: PyTorch Dataset for training

References:
    - RCSB PDB API: https://www.rcsb.org/docs/programmatic-access
    - PDB file format: https://www.wwpdb.org/documentation/file-format
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Standard amino acids (20 canonical)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class ProteinDataset(Dataset):
    """
    PyTorch Dataset for protein sequence-structure pairs.

    Attributes:
        sequences: List of amino acid sequences (strings)
        coordinates: List of 3D coordinate arrays (N_residues x 3)
        labels: Optional labels for supervised learning
    """

    def __init__(
        self, sequences: List[str], coordinates: List[np.ndarray], labels: List[int] = None
    ):
        """
        Initialize dataset.

        Args:
            sequences: Amino acid sequences
            coordinates: Corresponding 3D coordinates (Å)
            labels: Optional integer labels
        """
        assert len(sequences) == len(coordinates), "Sequence and coordinate count mismatch"
        self.sequences = sequences
        self.coordinates = coordinates
        self.labels = labels if labels is not None else [0] * len(sequences)
        self.logger = logging.getLogger(__name__)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get single data point.

        Returns:
            Dictionary with keys: sequence_encoded, coordinates, label, length
        """
        seq = self.sequences[idx]
        coords = self.coordinates[idx]
        label = self.labels[idx]

        # Encode sequence: one-hot encoding (20 amino acids)
        seq_encoded = self._encode_sequence(seq)

        # Ensure coordinates are float32 tensors
        coords_tensor = torch.tensor(coords, dtype=torch.float32)

        return {
            "sequence": seq_encoded,
            "coordinates": coords_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "length": len(seq),
        }

    def _encode_sequence(self, seq: str) -> torch.Tensor:
        """
        One-hot encode amino acid sequence.

        Args:
            seq: Amino acid sequence string

        Returns:
            Tensor of shape (length, 20)
        """
        encoding = np.zeros((len(seq), 20), dtype=np.float32)
        for i, aa in enumerate(seq):
            if aa in AMINO_ACIDS:
                encoding[i, AMINO_ACIDS.index(aa)] = 1.0
        return torch.tensor(encoding, dtype=torch.float32)


def generate_synthetic_data(
    num_samples: int = 10, min_length: int = 10, max_length: int = 50, seed: int = 42
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Generate synthetic protein sequences and toy 3D coordinates.

    Args:
        num_samples: Number of proteins to generate
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        seed: Random seed

    Returns:
        (sequences, coordinates) tuple

    Notes:
        Coordinates are random walks in 3D space with bond length ~3.8 Å
        (approximate C-alpha distance in proteins). This is a TOY fallback
        for offline testing, not biologically realistic.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {num_samples} synthetic proteins (lengths {min_length}-{max_length})")

    rng = np.random.RandomState(seed)
    sequences = []
    coordinates = []

    for i in range(num_samples):
        # Random sequence length
        length = rng.randint(min_length, max_length + 1)

        # Random amino acid sequence
        seq = "".join(rng.choice(list(AMINO_ACIDS), size=length))
        sequences.append(seq)

        # Generate toy coordinates as random walk
        # Start at origin
        coords = np.zeros((length, 3), dtype=np.float32)
        for j in range(1, length):
            # Random direction, fixed step size ~3.8 Å (C-alpha distance)
            direction = rng.randn(3)
            direction /= np.linalg.norm(direction)
            coords[j] = coords[j - 1] + direction * 3.8

        coordinates.append(coords)

    logger.info(f"Generated {len(sequences)} synthetic proteins")
    return sequences, coordinates


def fetch_pdb_structures(
    pdb_ids: List[str] = None, max_length: int = 50, output_dir: str = "outputs"
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Download small protein structures from RCSB PDB.

    Args:
        pdb_ids: List of PDB IDs to fetch (e.g., ["1CRN", "1L2Y"])
        max_length: Maximum residue count
        output_dir: Directory to save PDB files

    Returns:
        (sequences, coordinates) tuple

    Notes:
        If network fails, returns empty lists. Caller should use synthetic fallback.
        Downloads PDB files via RCSB REST API:
        https://files.rcsb.org/download/{pdb_id}.pdb

    References:
        RCSB PDB File Download: https://www.rcsb.org/docs/programmatic-access/file-download-services
    """
    logger = logging.getLogger(__name__)

    if pdb_ids is None:
        # Default: small well-characterized proteins
        pdb_ids = ["1CRN", "1L2Y"]  # Crambin (46 res), Trp-cage (20 res)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sequences = []
    coordinates = []

    try:
        import requests
        from Bio import PDB

        parser = PDB.PDBParser(QUIET=True)

        for pdb_id in pdb_ids:
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            pdb_file = output_path / f"{pdb_id}.pdb"

            # Download if not cached
            if not pdb_file.exists():
                logger.info(f"Downloading {pdb_id} from {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                pdb_file.write_bytes(response.content)

            # Parse PDB
            structure = parser.get_structure(pdb_id, str(pdb_file))

            for model in structure:
                for chain in model:
                    residues = [res for res in chain if PDB.is_aa(res)]

                    if len(residues) > max_length:
                        logger.warning(
                            f"{pdb_id} chain {chain.id}: {len(residues)} residues > {max_length}, skipping"
                        )
                        continue

                    # Extract sequence
                    seq = "".join(
                        [PDB.Polypeptide.three_to_one(res.get_resname()) for res in residues]
                    )

                    # Extract C-alpha coordinates
                    coords = []
                    for res in residues:
                        if "CA" in res:
                            coords.append(res["CA"].get_coord())

                    if len(coords) == len(seq):
                        sequences.append(seq)
                        coordinates.append(np.array(coords, dtype=np.float32))
                        logger.info(f"Loaded {pdb_id} chain {chain.id}: {len(seq)} residues")

    except ImportError:
        logger.warning("Biopython not available; cannot fetch PDB structures")
    except Exception as e:
        logger.warning(f"Failed to fetch PDB structures: {e}")

    return sequences, coordinates


def load_data(
    use_real_data: bool = True, num_synthetic: int = 20, seed: int = 42
) -> Tuple[ProteinDataset, ProteinDataset]:
    """
    Load training and test datasets.

    Args:
        use_real_data: Attempt to download real PDB structures
        num_synthetic: Number of synthetic samples if real data unavailable
        seed: Random seed

    Returns:
        (train_dataset, test_dataset) tuple

    Notes:
        Splits data 80/20 train/test. Uses synthetic fallback if real data fails.
    """
    logger = logging.getLogger(__name__)

    sequences = []
    coordinates = []

    if use_real_data:
        seq_real, coord_real = fetch_pdb_structures()
        if len(seq_real) > 0:
            sequences.extend(seq_real)
            coordinates.extend(coord_real)
            logger.info(f"Loaded {len(seq_real)} real protein structures from PDB")

    # Use synthetic fallback if no real data
    if len(sequences) == 0:
        logger.info("Using synthetic data fallback")
        sequences, coordinates = generate_synthetic_data(num_samples=num_synthetic, seed=seed)

    # Split train/test
    n_total = len(sequences)
    n_train = int(0.8 * n_total)

    train_sequences = sequences[:n_train]
    train_coordinates = coordinates[:n_train]
    test_sequences = sequences[n_train:]
    test_coordinates = coordinates[n_train:]

    train_dataset = ProteinDataset(train_sequences, train_coordinates)
    test_dataset = ProteinDataset(test_sequences, test_coordinates)

    logger.info(f"Dataset split: {len(train_dataset)} train, {len(test_dataset)} test")

    return train_dataset, test_dataset
