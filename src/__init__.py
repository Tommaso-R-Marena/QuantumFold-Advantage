"""QuantumFold-Advantage: Quantum-classical hybrid protein structure prediction.

This package provides:
- Quantum-enhanced deep learning models
- Advanced protein folding architectures
- Statistical validation tools
- CASP benchmark utilities
"""

__version__ = '0.1.0'
__author__ = 'Tommaso R. Marena'
__email__ = 'marena@cua.edu'

# Core imports
try:
    from .quantum_layers import (
        QuantumLayer,
        QuantumHybridLayer,
        EntanglementType,
    )
except ImportError:
    pass

try:
    from .advanced_model import AdvancedProteinFoldingModel
except ImportError:
    pass

try:
    from .protein_embeddings import ESM2Embedder, ProtT5Embedder
except ImportError:
    pass

try:
    from .advanced_training import (
        AdvancedTrainer,
        TrainingConfig,
        FrameAlignedPointError,
    )
except ImportError:
    pass

__all__ = [
    '__version__',
    '__author__',
    '__email__',
    'QuantumLayer',
    'QuantumHybridLayer',
    'EntanglementType',
    'AdvancedProteinFoldingModel',
    'ESM2Embedder',
    'ProtT5Embedder',
    'AdvancedTrainer',
    'TrainingConfig',
    'FrameAlignedPointError',
]
