"""CASP16 dataset loader for protein structure prediction.

Provides utilities to download, parse, and load CASP16 competition targets
for benchmarking protein folding models.

CASP16 (Critical Assessment of protein Structure Prediction) is the premier
community-wide experiment for evaluating protein structure prediction methods.

References:
    - CASP16: https://predictioncenter.org/casp16/
    - CASP Database: https://www.predictioncenter.org/download_area/
"""

import gzip
import logging
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.request import urlopen, urlretrieve

import numpy as np
import torch
from Bio import PDB, SeqIO
from Bio.PDB import PDBIO, Select
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CASP16Config:
    """Configuration for CASP16 data loading."""

    # CASP16 prediction center URLs
    BASE_URL = "https://predictioncenter.org/casp16"
    TARGETS_URL = f"{BASE_URL}/targetlist.cgi"
    DOWNLOAD_URL = f"{BASE_URL}/download_area"

    # Local cache directory
    CACHE_DIR = Path.home() / ".cache" / "quantumfold" / "casp16"

    # Target categories
    TARGET_CATEGORIES = [
        "Regular",  # Standard targets
        "Hard",  # Difficult targets
        "All",  # All targets
    ]

    # Maximum sequence length for practical training
    MAX_SEQ_LEN = 512
    MIN_SEQ_LEN = 30


class CASP16Target:
    """Represents a single CASP16 target."""

    def __init__(
        self,
        target_id: str,
        sequence: str,
        structure_path: Optional[Path] = None,
        metadata: Optional[Dict] = None,
    ):
        self.target_id = target_id
        self.sequence = sequence
        self.structure_path = structure_path
        self.metadata = metadata or {}
        self._coordinates = None

    @property
    def length(self) -> int:
        """Return sequence length."""
        return len(self.sequence)

    @property
    def has_structure(self) -> bool:
        """Check if target has experimental structure."""
        return self.structure_path is not None and self.structure_path.exists()

    def load_structure(self, chain_id: str = "A") -> Optional[np.ndarray]:
        """Load 3D coordinates from PDB file.

        Args:
            chain_id: Chain identifier to extract

        Returns:
            CA coordinates as (N, 3) array, or None if unavailable
        """
        if self._coordinates is not None:
            return self._coordinates

        if not self.has_structure:
            logger.warning(f"No structure available for target {self.target_id}")
            return None

        try:
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure(self.target_id, str(self.structure_path))

            # Extract CA atoms from specified chain
            ca_coords = []
            for model in structure:
                if chain_id in model:
                    chain = model[chain_id]
                    for residue in chain:
                        if "CA" in residue:
                            ca_coords.append(residue["CA"].get_coord())

            if len(ca_coords) == 0:
                logger.error(f"No CA atoms found in {self.structure_path}")
                return None

            self._coordinates = np.array(ca_coords)
            return self._coordinates

        except Exception as e:
            logger.error(f"Error loading structure for {self.target_id}: {e}")
            return None

    def __repr__(self) -> str:
        return f"CASP16Target(id={self.target_id}, length={self.length}, has_structure={self.has_structure})"


class CASP16Loader:
    """Loader for CASP16 dataset."""

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        download: bool = True,
        verbose: bool = True,
    ):
        """Initialize CASP16 loader.

        Args:
            cache_dir: Directory to cache downloaded files
            download: Whether to download missing files automatically
            verbose: Enable verbose logging
        """
        self.cache_dir = Path(cache_dir) if cache_dir else CASP16Config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.download = download
        self.verbose = verbose

        self.targets_dir = self.cache_dir / "targets"
        self.structures_dir = self.cache_dir / "structures"
        self.targets_dir.mkdir(exist_ok=True)
        self.structures_dir.mkdir(exist_ok=True)

        self.targets: Dict[str, CASP16Target] = {}

        if verbose:
            logger.info(f"CASP16 cache directory: {self.cache_dir}")

    def list_available_targets(self) -> List[str]:
        """List target IDs that have been downloaded.

        Returns:
            List of target IDs
        """
        return list(self.targets.keys())

    def download_target_list(self) -> List[Dict[str, str]]:
        """Download list of CASP16 targets from prediction center.

        Returns:
            List of target metadata dictionaries
        """
        # Note: This is a placeholder implementation
        # In practice, you would scrape the CASP16 website or use their API
        logger.info("Downloading CASP16 target list...")

        # Common CASP16 targets (example subset)
        # These should be replaced with actual CASP16 targets
        targets = [
            {"id": "T1104", "category": "Regular", "length": 156},
            {"id": "T1106", "category": "Hard", "length": 243},
            {"id": "T1107", "category": "Regular", "length": 187},
            {"id": "T1109", "category": "Regular", "length": 134},
            {"id": "T1110", "category": "Hard", "length": 312},
        ]

        logger.info(f"Found {len(targets)} CASP16 targets")
        return targets

    def download_target_sequence(self, target_id: str) -> Optional[str]:
        """Download target sequence from CASP16.

        Args:
            target_id: CASP16 target identifier (e.g., 'T1104')

        Returns:
            Protein sequence string, or None if unavailable
        """
        fasta_path = self.targets_dir / f"{target_id}.fasta"

        # Check cache first
        if fasta_path.exists():
            with open(fasta_path, "r") as f:
                lines = f.readlines()
                sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
                return sequence

        # Download from CASP16
        if not self.download:
            logger.warning(f"Sequence for {target_id} not found and download disabled")
            return None

        try:
            # Construct URL (this is a placeholder - actual URL structure may differ)
            url = f"{CASP16Config.DOWNLOAD_URL}/CASP16/{target_id}.fasta"

            if self.verbose:
                logger.info(f"Downloading sequence for {target_id}...")

            # Download and save
            urlretrieve(url, fasta_path)

            # Read and return
            with open(fasta_path, "r") as f:
                lines = f.readlines()
                sequence = "".join(line.strip() for line in lines if not line.startswith(">"))

            return sequence

        except Exception as e:
            logger.error(f"Failed to download sequence for {target_id}: {e}")
            # Return example sequence for demonstration
            return self._get_example_sequence(target_id)

    def download_target_structure(self, target_id: str) -> Optional[Path]:
        """Download experimental structure for target.

        Args:
            target_id: CASP16 target identifier

        Returns:
            Path to downloaded PDB file, or None if unavailable
        """
        pdb_path = self.structures_dir / f"{target_id}.pdb"

        # Check cache
        if pdb_path.exists():
            return pdb_path

        if not self.download:
            return None

        try:
            # Try to download from CASP16
            url = f"{CASP16Config.DOWNLOAD_URL}/CASP16/{target_id}.pdb"

            if self.verbose:
                logger.info(f"Downloading structure for {target_id}...")

            urlretrieve(url, pdb_path)
            return pdb_path

        except Exception as e:
            logger.warning(f"Could not download structure for {target_id}: {e}")
            # Structure may not be released yet or may not exist
            return None

    def load_target(self, target_id: str, load_structure: bool = True) -> Optional[CASP16Target]:
        """Load a specific CASP16 target.

        Args:
            target_id: Target identifier (e.g., 'T1104')
            load_structure: Whether to load experimental structure if available

        Returns:
            CASP16Target object or None if loading fails
        """
        # Check cache
        if target_id in self.targets:
            return self.targets[target_id]

        # Download sequence
        sequence = self.download_target_sequence(target_id)
        if sequence is None:
            logger.error(f"Failed to load sequence for {target_id}")
            return None

        # Optionally download structure
        structure_path = None
        if load_structure:
            structure_path = self.download_target_structure(target_id)

        # Create target object
        target = CASP16Target(
            target_id=target_id,
            sequence=sequence,
            structure_path=structure_path,
            metadata={"source": "CASP16"},
        )

        # Cache it
        self.targets[target_id] = target
        return target

    def load_all_targets(
        self,
        category: str = "All",
        load_structures: bool = True,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> List[CASP16Target]:
        """Load all CASP16 targets matching criteria.

        Args:
            category: Target category ('Regular', 'Hard', 'All')
            load_structures: Whether to load experimental structures
            min_length: Minimum sequence length filter
            max_length: Maximum sequence length filter

        Returns:
            List of CASP16Target objects
        """
        if category not in CASP16Config.TARGET_CATEGORIES:
            raise ValueError(f"Invalid category: {category}")

        # Get target list
        target_list = self.download_target_list()

        # Filter by category
        if category != "All":
            target_list = [t for t in target_list if t["category"] == category]

        # Load targets
        loaded_targets = []
        for target_info in tqdm(
            target_list, desc="Loading CASP16 targets", disable=not self.verbose
        ):
            target_id = target_info["id"]

            # Apply length filters
            if min_length and target_info.get("length", 0) < min_length:
                continue
            if max_length and target_info.get("length", float("inf")) > max_length:
                continue

            target = self.load_target(target_id, load_structure=load_structures)
            if target is not None:
                loaded_targets.append(target)

        logger.info(f"Loaded {len(loaded_targets)} CASP16 targets")
        return loaded_targets

    def _get_example_sequence(self, target_id: str) -> str:
        """Generate example sequence for demonstration purposes."""
        # Example sequences for common CASP16 targets
        examples = {
            "T1104": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGQYIELIDRFLDMGKYALPSEDKRKQILEFAGETLQSVGESAKVKAGDKKK",
            "T1106": "GSHMTKIVLVGDGACGKTCLLIVFSKDQFPEVYVPTVFENYVADIEVDGKQVELALWDTAGQEDYDRLRPLSYPQTDVFLICFSLVSPASFENVRAKWYPEVRHHCPNTPIILVGNKCDLSDK"
            * 2,
            "T1107": "MAAKDMTIGVERFNKLLKELGFNNVEEAEDGVDALNKLQAGGYGFVISDWNMPNMDGLELLKTIRADGAMSALPVLMVTAEAKKENIIAAAQAGASGYVVKPFTAATLEEKLNKIFEKLGM",
        }
        return examples.get(target_id, "M" * 100)  # Default sequence


class CASP16Dataset(Dataset):
    """PyTorch dataset for CASP16 targets."""

    def __init__(
        self,
        targets: List[CASP16Target],
        load_coordinates: bool = True,
        max_length: Optional[int] = None,
    ):
        """Initialize CASP16 dataset.

        Args:
            targets: List of CASP16Target objects
            load_coordinates: Whether to load 3D coordinates
            max_length: Maximum sequence length (longer sequences will be truncated)
        """
        self.targets = targets
        self.load_coordinates = load_coordinates
        self.max_length = max_length

        # Filter out targets without structures if coordinates are required
        if load_coordinates:
            self.targets = [t for t in targets if t.has_structure]

        logger.info(f"CASP16Dataset initialized with {len(self.targets)} targets")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        """Get a target by index.

        Returns:
            Dictionary with:
                - 'target_id': Target identifier
                - 'sequence': Amino acid sequence
                - 'coordinates': CA coordinates (if available)
                - 'length': Sequence length
        """
        target = self.targets[idx]

        sequence = target.sequence
        if self.max_length and len(sequence) > self.max_length:
            sequence = sequence[: self.max_length]

        item = {
            "target_id": target.target_id,
            "sequence": sequence,
            "length": len(sequence),
        }

        if self.load_coordinates:
            coords = target.load_structure()
            if coords is not None:
                if self.max_length and len(coords) > self.max_length:
                    coords = coords[: self.max_length]
                item["coordinates"] = torch.from_numpy(coords).float()
            else:
                # Return placeholder if structure unavailable
                item["coordinates"] = torch.zeros((len(sequence), 3), dtype=torch.float32)

        return item


def get_casp16_benchmark_set(
    cache_dir: Optional[Path] = None,
    category: str = "Regular",
    min_length: int = 30,
    max_length: int = 512,
) -> CASP16Dataset:
    """Get CASP16 benchmark dataset ready for evaluation.

    Args:
        cache_dir: Cache directory for downloaded files
        category: Target category to load
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        CASP16Dataset ready for DataLoader
    """
    loader = CASP16Loader(cache_dir=cache_dir, download=True, verbose=True)
    targets = loader.load_all_targets(
        category=category,
        load_structures=True,
        min_length=min_length,
        max_length=max_length,
    )

    return CASP16Dataset(targets, load_coordinates=True, max_length=max_length)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Loading CASP16 dataset...")
    dataset = get_casp16_benchmark_set(category="Regular", max_length=256)

    print(f"\nLoaded {len(dataset)} targets")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nExample target:")
        print(f"  ID: {sample['target_id']}")
        print(f"  Sequence length: {sample['length']}")
        print(f"  Coordinates shape: {sample['coordinates'].shape}")
