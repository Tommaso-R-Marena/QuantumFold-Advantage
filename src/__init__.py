"""QuantumFold-Advantage: Quantum-enhanced protein folding prediction."""

__version__ = "0.1.0"
__author__ = "Tommaso R. Marena"

# Graceful imports with proper error handling
import sys
import warnings

# Core modules that should always be available
try:
    from . import data
    from . import utils
except ImportError as e:
    warnings.warn(f"Some core modules could not be imported: {e}")
    data = None
    utils = None

# Optional modules - fail gracefully
try:
    from . import model
except ImportError:
    model = None

try:
    from . import advanced_model
except ImportError:
    advanced_model = None

try:
    from . import quantum_layers
except ImportError:
    quantum_layers = None

try:
    from . import training
except ImportError:
    training = None

try:
    from . import benchmarks
except ImportError:
    benchmarks = None

# Export public API
__all__ = [
    "__version__",
    "__author__",
    "data",
    "utils",
    "model",
    "advanced_model",
    "quantum_layers",
    "training",
    "benchmarks",
]

# Helpful error message if imports failed
def _check_installation():
    """Check if package is properly installed."""
    missing = [name for name in __all__[2:] if globals()[name] is None]
    if missing:
        warnings.warn(
            f"Some modules could not be imported: {', '.join(missing)}. "
            "This may be due to missing dependencies. "
            "Run: pip install -e .[dev] to install all dependencies."
        )

_check_installation()
