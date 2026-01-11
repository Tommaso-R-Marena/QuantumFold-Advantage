"""Advanced data processing utilities for protein structures.

This module provides tools for processing PDB files, MSA generation,
feature engineering, and data augmentation for protein folding.
"""

import gzip
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa, three_to_one
except ImportError:
    print("Warning: Biopython not installed. Install with: pip install biopython")


class ProteinFeatureExtractor:
    """Extract features from protein structures for model input."""

    def __init__(self, max_seq_len: int = 512):
        """
        Args:
            max_seq_len: Maximum sequence length to process
        """
        self.max_seq_len = max_seq_len
        self.aa_types = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_types)}

    def extract_from_pdb(
        self, pdb_path: str, chain_id: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Extract features from PDB file.

        Args:
            pdb_path: Path to PDB file
            chain_id: Specific chain to extract (None for first chain)

        Returns:
            Dictionary containing:
                - sequence: Amino acid sequence
                - coordinates: CA atom coordinates (N, 3)
                - residue_types: One-hot encoded residue types (N, 20)
                - distances: Pairwise distance matrix (N, N)
                - angles: Backbone angles (phi, psi, omega) (N, 3)
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)

        # Get chain
        if chain_id is None:
            chain = list(structure.get_chains())[0]
        else:
            chain = structure[0][chain_id]

        # Extract sequence and coordinates
        sequence = []
        coordinates = []
        residue_types = []

        for residue in chain:
            if is_aa(residue, standard=True):
                try:
                    # Get amino acid type
                    aa = three_to_one(residue.get_resname())
                    sequence.append(aa)

                    # Get CA coordinates
                    ca_atom = residue["CA"]
                    coordinates.append(ca_atom.get_coord())

                    # One-hot encode residue type
                    one_hot = np.zeros(len(self.aa_types))
                    if aa in self.aa_to_idx:
                        one_hot[self.aa_to_idx[aa]] = 1.0
                    residue_types.append(one_hot)

                except KeyError:
                    continue

        sequence = "".join(sequence)
        coordinates = np.array(coordinates)
        residue_types = np.array(residue_types)

        # Truncate if too long
        if len(sequence) > self.max_seq_len:
            sequence = sequence[: self.max_seq_len]
            coordinates = coordinates[: self.max_seq_len]
            residue_types = residue_types[: self.max_seq_len]

        # Calculate pairwise distances
        distances = self._calculate_distance_matrix(coordinates)

        # Calculate backbone angles
        angles = self._calculate_backbone_angles(coordinates)

        return {
            "sequence": sequence,
            "coordinates": coordinates,
            "residue_types": residue_types,
            "distances": distances,
            "angles": angles,
            "sequence_length": len(sequence),
        }

    def _calculate_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Calculate pairwise distance matrix."""
        diff = coordinates[:, None, :] - coordinates[None, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        return distances

    def _calculate_backbone_angles(self, coordinates: np.ndarray) -> np.ndarray:
        """Calculate backbone dihedral angles (phi, psi, omega).

        Note: This is a simplified version. Full implementation would
        require N, CA, C atoms for accurate angle calculation.
        """
        n_residues = len(coordinates)
        angles = np.zeros((n_residues, 3))

        # Calculate approximate angles from CA positions
        for i in range(1, n_residues - 1):
            # Vectors between consecutive CA atoms
            v1 = coordinates[i] - coordinates[i - 1]
            v2 = coordinates[i + 1] - coordinates[i]

            # Calculate angle (simplified)
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

            angles[i, 0] = angle  # Approximate phi
            angles[i, 1] = angle  # Approximate psi

        return angles

    def create_contact_map(self, coordinates: np.ndarray, threshold: float = 8.0) -> np.ndarray:
        """Create binary contact map from coordinates.

        Args:
            coordinates: CA coordinates
            threshold: Distance threshold for contact (Å)

        Returns:
            Binary contact map (N, N)
        """
        distances = self._calculate_distance_matrix(coordinates)
        contacts = (distances < threshold).astype(np.float32)
        # Remove diagonal
        np.fill_diagonal(contacts, 0)
        return contacts


class MSAProcessor:
    """Process multiple sequence alignments for model input."""

    def __init__(self, max_sequences: int = 128, max_seq_len: int = 512):
        self.max_sequences = max_sequences
        self.max_seq_len = max_seq_len

    def load_a3m(self, a3m_path: str) -> Tuple[List[str], List[str]]:
        """Load MSA from A3M file format.

        Args:
            a3m_path: Path to A3M file

        Returns:
            Tuple of (sequences, identifiers)
        """
        sequences = []
        identifiers = []

        if a3m_path.endswith(".gz"):
            open_fn = gzip.open
            mode = "rt"
        else:
            open_fn = open
            mode = "r"

        with open_fn(a3m_path, mode) as f:
            current_id = None
            current_seq = []

            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        identifiers.append(current_id)
                    current_id = line[1:]
                    current_seq = []
                else:
                    # Remove lowercase letters (insertions)
                    current_seq.append("".join(c for c in line if c.isupper()))

            if current_seq:
                sequences.append("".join(current_seq))
                identifiers.append(current_id)

        return sequences, identifiers

    def process_msa(
        self, sequences: List[str], truncate_length: Optional[int] = None
    ) -> np.ndarray:
        """Process MSA into model input format.

        Args:
            sequences: List of aligned sequences
            truncate_length: Length to truncate sequences

        Returns:
            MSA matrix (num_sequences, seq_len, 21) with one-hot encoding
            including gap character
        """
        if truncate_length is None:
            truncate_length = self.max_seq_len

        # Limit number of sequences
        if len(sequences) > self.max_sequences:
            # Keep query + most diverse sequences
            sequences = sequences[: self.max_sequences]

        # Truncate sequences
        sequences = [seq[:truncate_length] for seq in sequences]

        # Pad sequences to same length
        max_len = max(len(seq) for seq in sequences)
        sequences = [seq.ljust(max_len, "-") for seq in sequences]

        # Encode MSA
        aa_types = "ACDEFGHIKLMNPQRSTVWY-"  # Include gap
        aa_to_idx = {aa: i for i, aa in enumerate(aa_types)}

        msa_encoded = np.zeros((len(sequences), max_len, len(aa_types)))

        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                if aa in aa_to_idx:
                    msa_encoded[i, j, aa_to_idx[aa]] = 1.0

        return msa_encoded

    def calculate_msa_statistics(self, msa_encoded: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate statistics from MSA for features.

        Returns:
            Dictionary with:
                - conservation: Per-position conservation scores
                - coevolution: Pairwise coevolution scores
                - gap_frequency: Per-position gap frequency
        """
        n_seqs, seq_len, n_aa = msa_encoded.shape

        # Conservation (Shannon entropy)
        aa_freq = msa_encoded.mean(axis=0) + 1e-9  # Add pseudocount
        entropy = -np.sum(aa_freq * np.log(aa_freq), axis=1)
        conservation = 1.0 - (entropy / np.log(n_aa))  # Normalize

        # Gap frequency
        gap_frequency = msa_encoded[:, :, -1].mean(axis=0)

        # Simplified coevolution (mutual information approximation)
        # Full implementation would use MI or DCA
        coevolution = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # Correlation between positions
                corr = np.corrcoef(msa_encoded[:, i, :].T, msa_encoded[:, j, :].T)[
                    :n_aa, n_aa:
                ].mean()
                coevolution[i, j] = coevolution[j, i] = abs(corr)

        return {
            "conservation": conservation,
            "coevolution": coevolution,
            "gap_frequency": gap_frequency,
        }


class DataAugmentation:
    """Data augmentation for protein structures."""

    @staticmethod
    def random_rotation(coordinates: np.ndarray) -> np.ndarray:
        """Apply random 3D rotation."""
        # Generate random rotation matrix
        angles = np.random.uniform(0, 2 * np.pi, 3)

        # Rotation around x-axis
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])],
            ]
        )

        # Rotation around y-axis
        Ry = np.array(
            [
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])],
            ]
        )

        # Rotation around z-axis
        Rz = np.array(
            [
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )

        R = Rz @ Ry @ Rx
        return coordinates @ R.T

    @staticmethod
    def random_translation(coordinates: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Apply random translation."""
        translation = np.random.normal(0, scale, size=3)
        return coordinates + translation

    @staticmethod
    def add_coordinate_noise(coordinates: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to coordinates."""
        noise = np.random.normal(0, noise_level, size=coordinates.shape)
        return coordinates + noise
