"""Comprehensive logging configuration for QuantumFold-Advantage.

Provides:
- Structured logging with JSON output
- Performance profiling hooks
- Experiment tracking integration
- Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
"""

import json
import logging
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class StructuredLogger:
    """Structured logger with JSON output and performance tracking."""

    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_file: Optional[Path] = None,
        json_output: bool = True,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.json_output = json_output

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        if json_output:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            if json_output:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
                    )
                )
            self.logger.addHandler(file_handler)

    def _log_structured(self, level: str, message: str, **kwargs):
        """Log with structured data."""
        extra = {"structured_data": kwargs} if kwargs else {}
        getattr(self.logger, level)(message, extra=extra)

    def debug(self, message: str, **kwargs):
        self._log_structured("debug", message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log_structured("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log_structured("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log_structured("error", message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log_structured("critical", message, **kwargs)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add structured data if available
        if hasattr(record, "structured_data"):
            log_data.update(record.structured_data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def profile_function(logger: Optional[StructuredLogger] = None):
    """Decorator to profile function execution time.

    Args:
        logger: Optional logger instance

    Example:
        @profile_function()
        def train_model(data):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                if logger:
                    logger.info(
                        f"Function {func.__name__} completed",
                        duration_seconds=elapsed,
                        function=func.__name__,
                    )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                if logger:
                    logger.error(
                        f"Function {func.__name__} failed",
                        duration_seconds=elapsed,
                        function=func.__name__,
                        error=str(e),
                    )
                raise
        return wrapper
    return decorator


def get_logger(
    name: str,
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    json_output: bool = True,
) -> StructuredLogger:
    """Get or create a structured logger.

    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        json_output: Use JSON formatting

    Returns:
        StructuredLogger instance
    """
    log_file = None
    if log_dir:
        log_dir = Path(log_dir)
        log_file = log_dir / f"{name}.log"

    return StructuredLogger(name, level, log_file, json_output)
