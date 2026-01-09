"""QuantumFold-Advantage: Quantum-classical hybrid protein structure prediction.

This package provides:
- Quantum-enhanced deep learning models
- Advanced protein folding architectures
- Statistical validation tools
- CASP benchmark utilities
"""

__version__ = "0.1.0"
__author__ = "Tommaso R. Marena"
__email__ = "marena@cua.edu"

# Core imports with graceful error handling
# These imports are optional and tests should handle their absence

_QUANTUM_AVAILABLE = False
_MODEL_AVAILABLE = False
_EMBEDDINGS_AVAILABLE = False
_TRAINING_AVAILABLE = False

try:
    from .quantum_layers import EntanglementType, QuantumHybridLayer, QuantumLayer

    _QUANTUM_AVAILABLE = True
except (ImportError, AttributeError) as e:
    # PennyLane/autoray compatibility issue - tests will skip quantum features
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    QuantumLayer = None
    QuantumHybridLayer = None
    EntanglementType = None

try:
    from .advanced_model import AdvancedProteinFoldingModel

    _MODEL_AVAILABLE = True
except ImportError:
    AdvancedProteinFoldingModel = None

try:
    from .protein_embeddings import ESM2Embedder, ProtT5Embedder

    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    ESM2Embedder = None
    ProtT5Embedder = None

try:
    from .advanced_training import AdvancedTrainer, FrameAlignedPointError, TrainingConfig

    _TRAINING_AVAILABLE = True
except ImportError:
    AdvancedTrainer = None
    TrainingConfig = None
    FrameAlignedPointError = None

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "QuantumLayer",
    "QuantumHybridLayer",
    "EntanglementType",
    "AdvancedProteinFoldingModel",
    "ESM2Embedder",
    "ProtT5Embedder",
    "AdvancedTrainer",
    "TrainingConfig",
    "FrameAlignedPointError",
    # Availability flags
    "_QUANTUM_AVAILABLE",
    "_MODEL_AVAILABLE",
    "_EMBEDDINGS_AVAILABLE",
    "_TRAINING_AVAILABLE",
]
