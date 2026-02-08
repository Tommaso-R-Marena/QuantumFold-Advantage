"""Performance profiling utilities.

Provides tools for:
- Function timing and profiling
- Memory usage tracking
- GPU utilization monitoring
- Computational bottleneck detection
- Performance regression detection
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Timer", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        self.logger.info(f"{self.name}: {self.elapsed:.4f}s")

    def __str__(self):
        if self.elapsed is not None:
            return f"{self.name}: {self.elapsed:.4f}s"
        return f"{self.name}: not completed"


def profile_function(func: Callable = None, *, name: Optional[str] = None) -> Callable:
    """Decorator to profile function execution time.

    Usage:
        @profile_function
        def my_function():
            pass

        @profile_function(name="Custom Name")
        def another_function():
            pass
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            func_name = name or f.__name__
            start = time.perf_counter()
            result = f(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"Function '{func_name}' took {elapsed:.4f}s")
            return result

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


class MemoryTracker:
    """Track CPU and GPU memory usage."""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()

    def get_cpu_memory_mb(self) -> float:
        """Get current CPU memory usage in MB."""
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024**2

    def get_gpu_memory_mb(self, device: int = 0) -> Dict[str, float]:
        """Get GPU memory usage in MB."""
        if not self.gpu_available:
            return {"allocated": 0, "reserved": 0, "free": 0}

        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2

        # Get total GPU memory
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / 1024**2
        free = total - allocated

        return {"allocated": allocated, "reserved": reserved, "free": free, "total": total}

    def log_memory_stats(self):
        """Log current memory statistics."""
        cpu_mem = self.get_cpu_memory_mb()
        logger.info(f"CPU Memory: {cpu_mem:.2f} MB")

        if self.gpu_available:
            for i in range(torch.cuda.device_count()):
                gpu_mem = self.get_gpu_memory_mb(i)
                logger.info(
                    f"GPU {i} Memory - "
                    f"Allocated: {gpu_mem['allocated']:.2f} MB, "
                    f"Reserved: {gpu_mem['reserved']:.2f} MB, "
                    f"Free: {gpu_mem['free']:.2f} MB"
                )

    @contextmanager
    def track_memory(self, name: str = "Operation"):
        """Context manager to track memory delta."""
        cpu_before = self.get_cpu_memory_mb()
        gpu_before = self.get_gpu_memory_mb() if self.gpu_available else None

        yield

        cpu_after = self.get_cpu_memory_mb()
        cpu_delta = cpu_after - cpu_before

        logger.info(f"{name} - CPU Memory Delta: {cpu_delta:+.2f} MB")

        if gpu_before:
            gpu_after = self.get_gpu_memory_mb()
            gpu_delta = gpu_after["allocated"] - gpu_before["allocated"]
            logger.info(f"{name} - GPU Memory Delta: {gpu_delta:+.2f} MB")


class ThroughputMonitor:
    """Monitor training throughput."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.samples_processed = 0
        self.start_time = time.perf_counter()
        self.step_times = []

    def step(self, batch_size: int):
        """Record a training step."""
        current_time = time.perf_counter()
        elapsed = current_time - self.start_time

        self.samples_processed += batch_size
        self.step_times.append(elapsed)

        # Keep only recent history
        if len(self.step_times) > self.window_size:
            self.step_times.pop(0)

        self.start_time = current_time

    def get_throughput(self) -> Dict[str, float]:
        """Get current throughput statistics."""
        if not self.step_times:
            return {"samples_per_sec": 0, "avg_step_time": 0}

        avg_step_time = np.mean(self.step_times)
        samples_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0

        return {
            "samples_per_sec": samples_per_sec,
            "avg_step_time": avg_step_time,
            "total_samples": self.samples_processed,
        }

    def log_throughput(self):
        """Log throughput statistics."""
        stats = self.get_throughput()
        logger.info(
            f"Throughput: {stats['samples_per_sec']:.2f} samples/sec, "
            f"Avg step time: {stats['avg_step_time']:.4f}s"
        )


class ModelProfiler:
    """Profile model forward/backward pass."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.forward_times = []
        self.backward_times = []

    @contextmanager
    def profile_forward(self):
        """Profile forward pass."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.forward_times.append(elapsed)

    @contextmanager
    def profile_backward(self):
        """Profile backward pass."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.backward_times.append(elapsed)

    def get_stats(self) -> Dict[str, float]:
        """Get profiling statistics."""
        return {
            "avg_forward_time": np.mean(self.forward_times) if self.forward_times else 0,
            "avg_backward_time": np.mean(self.backward_times) if self.backward_times else 0,
            "forward_std": np.std(self.forward_times) if self.forward_times else 0,
            "backward_std": np.std(self.backward_times) if self.backward_times else 0,
        }

    def reset(self):
        """Reset profiling data."""
        self.forward_times = []
        self.backward_times = []


def profile_model_memory(model: torch.nn.Module, input_size: tuple) -> Dict[str, Any]:
    """Profile model memory footprint.

    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, seq_len, dim)

    Returns:
        Dictionary with memory statistics
    """
    device = next(model.parameters()).device

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate parameter memory (in MB)
    param_memory_mb = total_params * 4 / 1024**2  # Assuming float32

    # Profile forward pass
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

        dummy_input = torch.randn(input_size, device=device)
        with torch.no_grad():
            _ = model(dummy_input)

        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_memory_mb = None

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_memory_mb": param_memory_mb,
        "peak_memory_mb": peak_memory_mb,
    }
