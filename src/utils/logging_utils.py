"""Enhanced logging utilities for protein folding experiments.

Provides:
- Structured logging with context
- Experiment tracking integration
- Progress bars and status updates
- Error reporting and debugging
- Performance metrics logging
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class ColoredFormatter(logging.Formatter):
    """Colored console output formatter."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class ExperimentLogger:
    """Comprehensive experiment logging."""

    def __init__(
        self,
        name: str = "QuantumFold",
        log_dir: Optional[str] = None,
        level: int = logging.INFO,
        use_colors: bool = True,
        log_to_file: bool = True,
    ):
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if use_colors:
            console_fmt = ColoredFormatter(
                "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
            )
        else:
            console_fmt = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
            )

        console_handler.setFormatter(console_fmt)
        self.logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{name}_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_fmt = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_fmt)
            self.logger.addHandler(file_handler)

            self.log_file = log_file

        # Metrics storage
        self.metrics: Dict[str, list] = {}
        self.experiment_start = time.time()

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.debug(message)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.error(message)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.critical(message)

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(
            {"value": value, "step": step, "timestamp": time.time() - self.experiment_start}
        )

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)

        # Also log as info
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.info(f"Metrics: {metrics_str}")

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of hyperparameter names and values
        """
        self.info("🛠️  Hyperparameters:")
        for name, value in params.items():
            self.info(f"  {name}: {value}")

        # Save to JSON
        params_file = self.log_dir / "hyperparameters.json"
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)

    def log_model_summary(self, model: Any) -> None:
        """Log model architecture summary.

        Args:
            model: PyTorch model
        """
        try:
            import torch

            if isinstance(model, torch.nn.Module):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                self.info("🎯 Model Summary:")
                self.info(f"  Total parameters: {total_params:,}")
                self.info(f"  Trainable parameters: {trainable_params:,}")
                self.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        except Exception as e:
            self.warning(f"Could not log model summary: {e}")

    def save_metrics(self, filename: Optional[str] = None) -> Path:
        """Save metrics to JSON file.

        Args:
            filename: Optional filename (default: metrics.json)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = "metrics.json"

        metrics_file = self.log_dir / filename
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

        self.info(f"Metrics saved to {metrics_file}")
        return metrics_file

    def progress_bar(self, iterable, desc: str = "", **kwargs):
        """Create a progress bar.

        Args:
            iterable: Iterable to wrap
            desc: Description
            **kwargs: Additional tqdm arguments

        Returns:
            Progress bar iterator
        """
        if HAS_TQDM:
            return tqdm(iterable, desc=desc, **kwargs)
        else:
            # Fallback without progress bar
            return iterable


# Global logger instance
_global_logger: Optional[ExperimentLogger] = None


def get_logger(name: str = "QuantumFold", **kwargs) -> ExperimentLogger:
    """Get or create global logger.

    Args:
        name: Logger name
        **kwargs: Additional logger arguments

    Returns:
        Logger instance
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = ExperimentLogger(name=name, **kwargs)

    return _global_logger


class TimingContext:
    """Context manager for timing code blocks.

    Example:
        with TimingContext("Training epoch"):
            train_one_epoch()
    """

    def __init__(self, name: str, logger: Optional[ExperimentLogger] = None):
        self.name = name
        self.logger = logger or get_logger()
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.info(f"⏳ Starting: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time

        if exc_type is None:
            self.logger.info(f"✅ Completed: {self.name} ({elapsed:.2f}s)")
        else:
            self.logger.error(f"❌ Failed: {self.name} ({elapsed:.2f}s)")

        return False  # Don't suppress exceptions
