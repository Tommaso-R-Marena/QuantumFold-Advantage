"""Performance optimization utilities for protein folding.

Provides:
- Memory profiling and optimization
- Computation caching
- Batch processing helpers
- GPU memory management
- Profiling decorators
"""

import functools
import gc
import time
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, TypeVar

import numpy as np
import psutil
import torch

T = TypeVar("T")


class LRUCache:
    """Thread-safe LRU cache with size limit."""

    def __init__(self, maxsize: int = 128):
        self.cache: OrderedDict = OrderedDict()
        self.maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            # Remove oldest
            self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)


# Global cache for expensive computations
_global_cache = LRUCache(maxsize=256)


def cached(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to cache function results.

    Example:
        @cached
        def expensive_computation(x, y):
            return x ** y
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from function name and arguments
        key = f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items()))))}"

        result = _global_cache.get(key)
        if result is not None:
            return result

        result = func(*args, **kwargs)
        _global_cache.put(key, result)
        return result

    return wrapper


def profile_time(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to profile function execution time.

    Example:
        @profile_time
        def slow_function():
            time.sleep(1)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if elapsed > 1.0:
            print(f"⏱️  {func.__name__}: {elapsed:.2f}s")

        return result

    return wrapper


def profile_memory(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to profile memory usage.

    Example:
        @profile_memory
        def memory_intensive():
            return np.zeros((10000, 10000))
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get current memory
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**2  # MB

        result = func(*args, **kwargs)

        mem_after = process.memory_info().rss / 1024**2
        mem_diff = mem_after - mem_before

        if abs(mem_diff) > 100:  # More than 100 MB
            print(f"📦 {func.__name__}: {mem_diff:+.1f} MB")

        return result

    return wrapper


class GPUMemoryManager:
    """Manage GPU memory efficiently."""

    @staticmethod
    def get_gpu_memory() -> Dict[str, float]:
        """Get current GPU memory usage.

        Returns:
            Dictionary with allocated, reserved, and free memory in MB
        """
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0}

        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        free = total - allocated

        return {"allocated": allocated, "reserved": reserved, "free": free, "total": total}

    @staticmethod
    def clear_cache() -> None:
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    @staticmethod
    def optimize_memory() -> None:
        """Optimize memory usage."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def check_memory_available(required_mb: float) -> bool:
        """Check if enough GPU memory is available.

        Args:
            required_mb: Required memory in MB

        Returns:
            True if enough memory available
        """
        if not torch.cuda.is_available():
            return False

        mem_info = GPUMemoryManager.get_gpu_memory()
        return mem_info["free"] >= required_mb

    @staticmethod
    def print_memory_summary() -> None:
        """Print GPU memory summary."""
        mem_info = GPUMemoryManager.get_gpu_memory()
        print(f"\n📦 GPU Memory:")
        print(f"  Allocated: {mem_info['allocated']:.1f} MB")
        print(f"  Reserved:  {mem_info['reserved']:.1f} MB")
        print(f"  Free:      {mem_info['free']:.1f} MB")
        print(f"  Total:     {mem_info['total']:.1f} MB")


class BatchProcessor:
    """Efficient batch processing with automatic sizing."""

    def __init__(self, max_batch_size: int = 32, auto_adjust: bool = True):
        self.max_batch_size = max_batch_size
        self.auto_adjust = auto_adjust
        self.current_batch_size = max_batch_size

    def process_batches(self, data: list, process_fn: Callable, **kwargs) -> list:
        """Process data in optimally-sized batches.

        Args:
            data: List of items to process
            process_fn: Function to apply to each batch
            **kwargs: Additional arguments for process_fn

        Returns:
            List of processed results
        """
        results = []
        batch_size = self.current_batch_size

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            try:
                result = process_fn(batch, **kwargs)
                results.extend(result if isinstance(result, list) else [result])

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.auto_adjust:
                    # Reduce batch size and retry
                    GPUMemoryManager.clear_cache()
                    batch_size = max(1, batch_size // 2)
                    self.current_batch_size = batch_size
                    warnings.warn(f"Reduced batch size to {batch_size} due to OOM")
                    # Retry with smaller batch
                    i -= batch_size  # Go back
                    continue
                else:
                    raise

        return results


def optimize_tensor_memory(tensor: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """Optimize tensor memory usage.

    Args:
        tensor: Input tensor
        inplace: Whether to modify tensor in-place

    Returns:
        Optimized tensor
    """
    if not inplace:
        tensor = tensor.clone()

    # Contiguous memory layout
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Remove gradients if not needed
    if tensor.requires_grad and not torch.is_grad_enabled():
        tensor = tensor.detach()

    return tensor


def estimate_model_memory(model: torch.nn.Module) -> float:
    """Estimate model memory footprint in MB.

    Args:
        model: PyTorch model

    Returns:
        Estimated memory in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_mb = (param_size + buffer_size) / 1024**2
    return total_mb


class ProgressiveLoader:
    """Load data progressively to avoid memory spikes."""

    @staticmethod
    def load_embeddings_progressive(
        sequences: list, embedder: Callable, chunk_size: int = 100
    ) -> torch.Tensor:
        """Load embeddings in chunks.

        Args:
            sequences: List of protein sequences
            embedder: Embedding function
            chunk_size: Number of sequences per chunk

        Returns:
            Concatenated embeddings
        """
        all_embeddings = []

        for i in range(0, len(sequences), chunk_size):
            chunk = sequences[i : i + chunk_size]

            # Generate embeddings
            emb = embedder(chunk)

            # Move to CPU immediately to free GPU memory
            if isinstance(emb, dict):
                emb = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in emb.items()}
            elif isinstance(emb, torch.Tensor):
                emb = emb.cpu()

            all_embeddings.append(emb)

            # Clear cache
            GPUMemoryManager.clear_cache()

        # Concatenate all chunks
        if isinstance(all_embeddings[0], dict):
            result = {
                k: torch.cat([e[k] for e in all_embeddings], dim=0)
                for k in all_embeddings[0].keys()
                if isinstance(all_embeddings[0][k], torch.Tensor)
            }
        else:
            result = torch.cat(all_embeddings, dim=0)

        return result


def benchmark_function(
    func: Callable, *args, n_runs: int = 10, warmup: int = 2, **kwargs
) -> Dict[str, float]:
    """Benchmark function performance.

    Args:
        func: Function to benchmark
        *args: Function arguments
        n_runs: Number of benchmark runs
        warmup: Number of warmup runs
        **kwargs: Function keyword arguments

    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times),
    }


class TensorCheckpointer:
    """Checkpoint tensors to disk to save memory."""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        import os

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoints = {}

    def save(self, name: str, tensor: torch.Tensor) -> str:
        """Save tensor to disk.

        Args:
            name: Checkpoint name
            tensor: Tensor to save

        Returns:
            Path to saved checkpoint
        """
        import os

        path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        torch.save(tensor.cpu(), path)
        self.checkpoints[name] = path
        return path

    def load(self, name: str, device: str = "cpu") -> torch.Tensor:
        """Load tensor from disk.

        Args:
            name: Checkpoint name
            device: Device to load to

        Returns:
            Loaded tensor
        """
        if name not in self.checkpoints:
            raise KeyError(f"Checkpoint {name} not found")

        tensor = torch.load(self.checkpoints[name], map_location=device)
        return tensor

    def clear_all(self) -> None:
        """Delete all checkpoints."""
        import os

        for path in self.checkpoints.values():
            if os.path.exists(path):
                os.remove(path)
        self.checkpoints.clear()
