"""Utility modules for QuantumFold-Advantage."""

from .checkpoint import CheckpointManager, CheckpointMetadata
from .config import Config, load_config, save_config
from .validation import (
    ValidationError,
    clip_gradients,
    safe_divide,
    validate_config,
    validate_coordinates,
    validate_protein_sequence,
    validate_range,
    validate_tensor,
    validate_type,
)
from .common import *

__all__ = [
    "Config",
    "load_config",
    "save_config",
    "CheckpointManager",
    "CheckpointMetadata",
    "ValidationError",
    "validate_tensor",
    "validate_range",
    "validate_type",
    "validate_config",
    "validate_protein_sequence",
    "validate_coordinates",
    "safe_divide",
    "clip_gradients",
    "set_seed",
    "detect_device",
    "setup_logging",
    "count_parameters",
]

# Optional utilities: imported lazily when dependencies are available.
try:
    pass

    __all__.extend(
        [
            "setup_distributed",
            "cleanup_distributed",
            "get_rank",
            "get_world_size",
            "is_main_process",
            "barrier",
            "convert_to_ddp",
            "GradientAccumulator",
        ]
    )
except ImportError:
    pass

try:
    pass

    __all__.extend(["HyperparameterTuner", "create_training_objective"])
except ImportError:
    pass

try:
    pass

    __all__.extend(["get_logger"])
except ImportError:
    pass

try:
    pass

    __all__.extend(
        [
            "MemoryTracker",
            "clear_gpu_memory",
            "memory_efficient_mode",
            "get_model_memory_usage",
            "estimate_batch_size",
            "MemoryLeakDetector",
        ]
    )
except ImportError:
    pass

try:
    pass

    __all__.extend(["Profiler", "profile_function", "profile_memory", "GPUProfiler"])
except ImportError:
    pass
