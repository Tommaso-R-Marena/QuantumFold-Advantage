"""Utility modules for QuantumFold-Advantage.

Provides:
- Configuration management (config)
- Structured logging (logging_config)
- Performance profiling (profiling)
- Checkpoint management (checkpoint)
- Distributed training (distributed)
- Memory optimization (memory)
- Input validation (validation)
- Hyperparameter tuning (hyperparameter_tuning)
"""

from .checkpoint import CheckpointManager, CheckpointMetadata
from .config import Config, load_config, save_config
from .distributed import (
    GradientAccumulator,
    barrier,
    cleanup_distributed,
    convert_to_ddp,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
)
from .hyperparameter_tuning import HyperparameterTuner, create_training_objective
from .logging_config import get_logger, setup_logging
from .memory import (
    MemoryLeakDetector,
    MemoryTracker,
    clear_gpu_memory,
    estimate_batch_size,
    get_model_memory_usage,
    memory_efficient_mode,
)
from .profiling import GPUProfiler, Profiler, profile_function, profile_memory
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

__all__ = [
    # Configuration
    "Config",
    "load_config",
    "save_config",
    # Logging
    "setup_logging",
    "get_logger",
    # Profiling
    "Profiler",
    "profile_function",
    "profile_memory",
    "GPUProfiler",
    # Checkpointing
    "CheckpointManager",
    "CheckpointMetadata",
    # Distributed
    "setup_distributed",
    "cleanup_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "convert_to_ddp",
    "GradientAccumulator",
    # Memory
    "MemoryTracker",
    "clear_gpu_memory",
    "memory_efficient_mode",
    "get_model_memory_usage",
    "estimate_batch_size",
    "MemoryLeakDetector",
    # Validation
    "ValidationError",
    "validate_tensor",
    "validate_range",
    "validate_type",
    "validate_config",
    "validate_protein_sequence",
    "validate_coordinates",
    "safe_divide",
    "clip_gradients",
    # Hyperparameter Tuning
    "HyperparameterTuner",
    "create_training_objective",
]
