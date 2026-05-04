"""Data loading and processing utilities for protein structure prediction.

This module provides:
- CASP16 benchmark dataset loading
- Synthetic protein dataset generation
- Data validation and preprocessing
- Custom collate functions for DataLoader
"""

try:
    from .casp16_loader import (
        CASP16Config,
        CASP16Dataset,
        CASP16Loader,
        CASP16Target,
        get_casp16_benchmark_set,
    )

    _CASP16_AVAILABLE = True
except ImportError as e:
    _CASP16_AVAILABLE = False
    _CASP16_ERROR = str(e)

__all__ = [
    "CASP16Config",
    "CASP16Target",
    "CASP16Loader",
    "CASP16Dataset",
    "get_casp16_benchmark_set",
]

if not _CASP16_AVAILABLE:
    import warnings

    warnings.warn(f"CASP16 loader not available: {_CASP16_ERROR}")

from .loader import ProteinDataset, generate_synthetic_data, fetch_pdb_structures, load_data as load_or_generate_data
from .hybrid_dataset import create_dataloaders, generate_synthetic_proteins, ProteinStructureDataset

__all__.extend([
    "ProteinDataset",
    "generate_synthetic_data",
    "fetch_pdb_structures",
    "load_or_generate_data",
    "create_dataloaders",
    "generate_synthetic_proteins",
    "ProteinStructureDataset",
])
