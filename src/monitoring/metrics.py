"""Real-time monitoring and metrics collection."""

import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional

import torch
import torch.nn as nn


class MetricsCollector:
    """Collect and aggregate training metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.step_times: Deque[float] = deque(maxlen=window_size)
        self.last_step_time: Optional[float] = None

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        self.metrics[name].append(value)

    def log_step_time(self) -> None:
        """Log time for current step."""
        current_time = time.time()
        if self.last_step_time is not None:
            self.step_times.append(current_time - self.last_step_time)
        self.last_step_time = current_time

    def get_average(self, name: str) -> Optional[float]:
        """Get average of a metric over window.

        Args:
            name: Metric name

        Returns:
            Average value or None if no data
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None
        return sum(self.metrics[name]) / len(self.metrics[name])

    def get_throughput(self) -> Optional[float]:
        """Get training throughput (steps/second).

        Returns:
            Throughput or None if no data
        """
        if len(self.step_times) == 0:
            return None
        avg_time = sum(self.step_times) / len(self.step_times)
        return 1.0 / avg_time if avg_time > 0 else None

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics.

        Returns:
            Dictionary of metric averages
        """
        summary = {}
        for name in self.metrics:
            avg = self.get_average(name)
            if avg is not None:
                summary[name] = avg

        throughput = self.get_throughput()
        if throughput is not None:
            summary["throughput_steps_per_sec"] = throughput

        return summary

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.step_times.clear()
        self.last_step_time = None


class GPUMonitor:
    """Monitor GPU utilization and memory."""

    @staticmethod
    def get_gpu_memory_usage() -> Dict[str, float]:
        """Get current GPU memory usage.

        Returns:
            Dictionary with memory statistics in MB
        """
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }

    @staticmethod
    def get_gpu_utilization() -> Optional[float]:
        """Get GPU utilization percentage.

        Returns:
            Utilization percentage or None if unavailable
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except:
            return None


class GradientMonitor:
    """Monitor gradient statistics during training."""

    @staticmethod
    def compute_gradient_norm(model: nn.Module) -> float:
        """Compute total gradient norm.

        Args:
            model: Model with gradients

        Returns:
            Total gradient norm
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    @staticmethod
    def get_gradient_stats(model: nn.Module) -> Dict[str, float]:
        """Get comprehensive gradient statistics.

        Args:
            model: Model with gradients

        Returns:
            Dictionary of gradient statistics
        """
        grad_norms = []
        for p in model.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.data.norm(2).item())

        if not grad_norms:
            return {}

        return {
            "grad_norm": sum(g**2 for g in grad_norms) ** 0.5,
            "grad_mean": sum(grad_norms) / len(grad_norms),
            "grad_max": max(grad_norms),
            "grad_min": min(grad_norms),
        }
