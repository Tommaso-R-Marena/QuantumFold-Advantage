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

from .config import Config, load_config, save_config
from .logging_config import setup_logging, get_logger
from .profiling import (
    Profiler,
    profile_function,
    profile_memory,
    GPUProfiler
)
from .checkpoint import CheckpointManager, CheckpointMetadata
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    convert_to_ddp,
    GradientAccumulator
)
from .memory import (
    MemoryTracker,
    clear_gpu_memory,
    memory_efficient_mode,
    get_model_memory_usage,
    estimate_batch_size,
    MemoryLeakDetector
)
from .validation import (
    ValidationError,
    validate_tensor,
    validate_range,
    validate_type,
    validate_config,
    validate_protein_sequence,
    validate_coordinates,
    safe_divide,
    clip_gradients
)
from .hyperparameter_tuning import (
    HyperparameterTuner,
    create_training_objective
)

__all__ = [
    # Configuration
    'Config',
    'load_config',
    'save_config',
    
    # Logging
    'setup_logging',
    'get_logger',
    
    # Profiling
    'Profiler',
    'profile_function',
    'profile_memory',
    'GPUProfiler',
    
    # Checkpointing
    'CheckpointManager',
    'CheckpointMetadata',
    
    # Distributed
    'setup_distributed',
    'cleanup_distributed',
    'get_rank',
    'get_world_size',
    'is_main_process',
    'barrier',
    'convert_to_ddp',
    'GradientAccumulator',
    
    # Memory
    'MemoryTracker',
    'clear_gpu_memory',
    'memory_efficient_mode',
    'get_model_memory_usage',
    'estimate_batch_size',
    'MemoryLeakDetector',
    
    # Validation
    'ValidationError',
    'validate_tensor',
    'validate_range',
    'validate_type',
    'validate_config',
    'validate_protein_sequence',
    'validate_coordinates',
    'safe_divide',
    'clip_gradients',
    
    # Hyperparameter Tuning
    'HyperparameterTuner',
    'create_training_objective',
]
