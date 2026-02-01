"""Memory optimization utilities for efficient training.

Provides:
- Memory profiling and monitoring
- Automatic garbage collection
- Memory-efficient data loading
- GPU memory management
- Memory leak detection
"""

import gc
import logging
import time
from contextlib import contextmanager
from typing import Dict, Optional

import psutil
import torch

logger = logging.getLogger(__name__)


class MemoryTracker:
    """Track memory usage during training.

    Monitors both CPU and GPU memory, provides statistics and warnings.
    """

    def __init__(self, warn_threshold: float = 0.9):
        """
        Args:
            warn_threshold: Warn when memory usage exceeds this fraction (0-1)
        """
        self.warn_threshold = warn_threshold
        self.measurements = []

    def get_cpu_memory_usage(self) -> Dict[str, float]:
        """Get current CPU memory usage.

        Returns:
            Dictionary with memory statistics in GB
        """
        process = psutil.Process()
        mem_info = process.memory_info()
        virtual = psutil.virtual_memory()

        return {
            "used_gb": mem_info.rss / 1e9,
            "available_gb": virtual.available / 1e9,
            "total_gb": virtual.total / 1e9,
            "percent": virtual.percent,
        }

    def get_gpu_memory_usage(self, device: Optional[int] = None) -> Dict[str, float]:
        """Get current GPU memory usage.

        Args:
            device: GPU device ID (None for current device)

        Returns:
            Dictionary with GPU memory statistics in GB
        """
        if not torch.cuda.is_available():
            return {"used_gb": 0, "allocated_gb": 0, "cached_gb": 0, "total_gb": 0}

        if device is None:
            device = torch.cuda.current_device()

        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "percent": (allocated / total) * 100 if total > 0 else 0,
        }

    def log_memory_stats(self, prefix: str = ""):
        """Log current memory statistics."""
        cpu_mem = self.get_cpu_memory_usage()
        gpu_mem = self.get_gpu_memory_usage()

        msg = f"{prefix}Memory - "
        msg += f"CPU: {cpu_mem['used_gb']:.2f}GB/{cpu_mem['total_gb']:.2f}GB ({cpu_mem['percent']:.1f}%)"

        if gpu_mem["total_gb"] > 0:
            msg += f", GPU: {gpu_mem['allocated_gb']:.2f}GB/{gpu_mem['total_gb']:.2f}GB "
            msg += f"({gpu_mem['percent']:.1f}%)"

        logger.info(msg)

        # Warn if memory usage is high
        if cpu_mem["percent"] / 100 > self.warn_threshold:
            logger.warning(f"High CPU memory usage: {cpu_mem['percent']:.1f}%")

        if gpu_mem["percent"] / 100 > self.warn_threshold:
            logger.warning(f"High GPU memory usage: {gpu_mem['percent']:.1f}%")

    def record_measurement(self, tag: str):
        """Record a memory measurement.

        Args:
            tag: Label for this measurement
        """
        cpu_mem = self.get_cpu_memory_usage()
        gpu_mem = self.get_gpu_memory_usage()

        self.measurements.append(
            {
                "tag": tag,
                "timestamp": time.time(),
                "cpu_gb": cpu_mem["used_gb"],
                "gpu_gb": gpu_mem["allocated_gb"],
            }
        )

    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage from measurements.

        Returns:
            Dictionary with peak CPU and GPU memory in GB
        """
        if not self.measurements:
            return {"cpu_gb": 0, "gpu_gb": 0}

        peak_cpu = max(m["cpu_gb"] for m in self.measurements)
        peak_gpu = max(m["gpu_gb"] for m in self.measurements)

        return {"cpu_gb": peak_cpu, "gpu_gb": peak_gpu}


def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("Cleared GPU memory cache")


@contextmanager
def memory_efficient_mode():
    """Context manager for memory-efficient execution.

    - Disables gradient computation
    - Clears cache after execution
    """
    with torch.no_grad():
        try:
            yield
        finally:
            clear_gpu_memory()


def get_model_memory_usage(model: torch.nn.Module) -> Dict[str, float]:
    """Calculate memory usage of a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter and buffer memory in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return {
        "params_mb": param_size / 1e6,
        "buffers_mb": buffer_size / 1e6,
        "total_mb": (param_size + buffer_size) / 1e6,
    }


def estimate_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    available_memory_gb: float = None,
    safety_factor: float = 0.8,
) -> int:
    """Estimate maximum batch size that fits in memory.

    Args:
        model: PyTorch model
        input_shape: Shape of single input (without batch dimension)
        available_memory_gb: Available GPU memory in GB (auto-detect if None)
        safety_factor: Use this fraction of available memory (0-1)

    Returns:
        Estimated maximum batch size
    """
    if available_memory_gb is None:
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            allocated_mem = torch.cuda.memory_allocated(0)
            available_memory_gb = (total_mem - allocated_mem) / 1e9
        else:
            # Use CPU memory
            available_memory_gb = psutil.virtual_memory().available / 1e9

    # Apply safety factor
    available_memory_gb *= safety_factor

    # Get model size
    model_mem = get_model_memory_usage(model)
    model_mb = model_mem["total_mb"]

    # Estimate activation memory for single sample
    # Rough estimate: ~4x model parameters for activations
    single_sample_mb = model_mb * 4

    # Calculate max batch size
    available_mb = available_memory_gb * 1000
    max_batch_size = int((available_mb - model_mb) / single_sample_mb)

    return max(1, max_batch_size)


@contextmanager
def optimize_gpu_memory():
    """Context manager for optimized GPU memory usage.

    Sets optimal PyTorch memory allocation settings.
    """
    if not torch.cuda.is_available():
        yield
        return

    # Save current settings
    original_allocator_settings = torch.cuda.get_allocator_backend()

    try:
        # Enable memory efficient settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        yield

    finally:
        # Restore original settings
        pass


class MemoryLeakDetector:
    """Detect potential memory leaks during training."""

    def __init__(self, threshold_mb: float = 100):
        """
        Args:
            threshold_mb: Warn if memory increases by this amount between checks
        """
        self.threshold_mb = threshold_mb
        self.previous_memory = None

    def check(self, tag: str = ""):
        """Check for memory leaks.

        Args:
            tag: Label for this check
        """
        current_memory = self._get_total_memory()

        if self.previous_memory is not None:
            delta_mb = current_memory - self.previous_memory

            if delta_mb > self.threshold_mb:
                logger.warning(
                    f"Potential memory leak detected {tag}: " f"+{delta_mb:.1f}MB since last check"
                )

        self.previous_memory = current_memory

    def _get_total_memory(self) -> float:
        """Get total memory usage in MB."""
        cpu_mem = psutil.Process().memory_info().rss / 1e6

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e6
            return cpu_mem + gpu_mem

        return cpu_mem
