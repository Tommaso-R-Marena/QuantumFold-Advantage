"""Dataset for QuantumFold-Advantage models.

Provides a PyTorch Dataset that yields amino-acid indices,
physicochemical features, backbone coordinates (N/Cα/C), and
padding masks.  Includes a synthetic data generator for end-to-end
testing without downloading CASP targets.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = len(AMINO_ACIDS)  # 20

# Physicochemical properties: hydrophobicity, charge, normalised MW
AA_PROPS = {
    "A": [1.8, 0.0, 0.42], "C": [2.5, 0.0, 0.57], "D": [-3.5, -1.0, 0.63],
    "E": [-3.5, -1.0, 0.69], "F": [2.8, 0.0, 0.78], "G": [-0.4, 0.0, 0.35],
    "H": [-3.2, 0.5, 0.73], "I": [4.5, 0.0, 0.62], "K": [-3.9, 1.0, 0.69],
    "L": [3.8, 0.0, 0.62], "M": [1.9, 0.0, 0.70], "N": [-3.5, 0.0, 0.62],
    "P": [-1.6, 0.0, 0.54], "Q": [-3.5, 0.0, 0.69], "R": [-4.5, 1.0, 0.82],
    "S": [-0.8, 0.0, 0.50], "T": [-0.7, 0.0, 0.56], "V": [4.2, 0.0, 0.55],
    "W": [-0.9, 0.0, 0.96], "Y": [-1.3, 0.0, 0.85],
}


def encode_sequence(seq: str) -> Tuple[np.ndarray, np.ndarray]:
    """Encode amino-acid string to index array and physicochemical matrix."""
    idx = np.array([AA_TO_IDX.get(aa, PAD_IDX) for aa in seq], dtype=np.int64)
    props = np.array([AA_PROPS.get(aa, [0.0, 0.0, 0.0]) for aa in seq], dtype=np.float32)
    return idx, props


class ProteinStructureDataset(Dataset):
    """Dataset yielding (aa_idx, physchem, coords, mask) tuples.

    Args:
        sequences: List of amino-acid strings.
        coords_list: List of Cα coordinate arrays (N, 3).
        max_len: Pad/truncate sequences to this length.
    """

    def __init__(
        self,
        sequences: List[str],
        coords_list: List[np.ndarray],
        max_len: int = 256,
    ):
        assert len(sequences) == len(coords_list)
        self.sequences = sequences
        self.coords_list = coords_list
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx][:self.max_len]
        ca_coords = self.coords_list[idx][:self.max_len]
        L = len(seq)

        aa_idx, physchem = encode_sequence(seq)

        # Build backbone coords (N, Cα, C) from Cα using idealised offsets
        backbone = np.zeros((L, 3, 3), dtype=np.float32)
        backbone[:, 1, :] = ca_coords  # Cα
        backbone[:, 0, :] = ca_coords + np.array([-0.527, 1.359, 0.0])  # N
        backbone[:, 2, :] = ca_coords + np.array([1.526, 0.0, 0.0])      # C

        # Pad
        aa_padded = np.full(self.max_len, PAD_IDX, dtype=np.int64)
        aa_padded[:L] = aa_idx
        physchem_padded = np.zeros((self.max_len, 3), dtype=np.float32)
        physchem_padded[:L] = physchem
        coords_padded = np.zeros((self.max_len, 3, 3), dtype=np.float32)
        coords_padded[:L] = backbone
        mask = np.zeros(self.max_len, dtype=bool)
        mask[:L] = True

        return {
            "aa_idx": torch.from_numpy(aa_padded),
            "physchem": torch.from_numpy(physchem_padded),
            "coords": torch.from_numpy(coords_padded),
            "mask": torch.from_numpy(mask),
            "length": L,
        }


# ---------------------------------------------------------------------------
# Synthetic data generator (for testing without real PDB files)
# ---------------------------------------------------------------------------

def generate_synthetic_proteins(
    n_proteins: int = 50,
    min_len: int = 20,
    max_len: int = 80,
    seed: int = 42,
) -> Tuple[List[str], List[np.ndarray]]:
    """Generate synthetic proteins with random-walk Cα coordinates.

    Bond length fixed at 3.8 Å (average Cα–Cα distance).
    """
    rng = np.random.RandomState(seed)
    sequences: List[str] = []
    coords_list: List[np.ndarray] = []

    for _ in range(n_proteins):
        L = rng.randint(min_len, max_len + 1)
        seq = "".join(rng.choice(list(AMINO_ACIDS), size=L))
        sequences.append(seq)

        coords = np.zeros((L, 3), dtype=np.float32)
        for j in range(1, L):
            direction = rng.randn(3).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-8
            coords[j] = coords[j - 1] + direction * 3.8
        coords_list.append(coords)

    return sequences, coords_list


def create_dataloaders(
    sequences: List[str],
    coords_list: List[np.ndarray],
    max_len: int = 256,
    batch_size: int = 4,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Split data and create DataLoaders."""
    n = len(sequences)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    splits = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }

    loaders = {}
    for name, idxs in splits.items():
        ds = ProteinStructureDataset(
            [sequences[i] for i in idxs],
            [coords_list[i] for i in idxs],
            max_len=max_len,
        )
        loaders[name] = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"), drop_last=False
        )
    return loaders["train"], loaders["val"], loaders["test"]
