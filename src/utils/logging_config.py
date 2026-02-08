"""Structured logging configuration for production.

Provides:
- JSON structured logging
- Multiple output handlers (console, file, remote)
- Log levels and filtering
- Performance metrics logging
- Integration with experiment trackers
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add custom fields
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console output for better readability."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    json_logging: bool = False,
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """Setup comprehensive logging configuration.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        json_logging: Use JSON format for logs
        console_output: Enable console logging
        file_output: Enable file logging

    Returns:
        Configured logger instance
    """
    # Get root logger
    logger = logging.getLogger("quantumfold")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []  # Clear existing handlers

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        if json_logging:
            console_handler.setFormatter(JSONFormatter())
        else:
            formatter = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    # File handler
    if file_output and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"quantumfold_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets all logs

        if json_logging:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
                )
            )

        logger.addHandler(file_handler)

        # Error log file (errors and above only)
        error_file = log_dir / f"errors_{timestamp}.log"
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s")
        )
        logger.addHandler(error_handler)

    return logger


def get_logger(name: str = "quantumfold") -> logging.Logger:
    """Get or create a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logging()
        logger = logging.getLogger(name)
    return logger


class MetricsLogger:
    """Logger for training metrics and statistics."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics_history: Dict[str, list] = {}

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a single metric value."""
        if name not in self.metrics_history:
            self.metrics_history[name] = []

        self.metrics_history[name].append(value)

        extra_data = {"metric": name, "value": value}
        if step is not None:
            extra_data["step"] = step

        # Create log record with extra data
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0, f"Metric: {name} = {value:.4f}", (), None
        )
        record.extra_data = extra_data
        self.logger.handle(record)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def get_history(self, metric_name: str) -> list:
        """Get history of a metric."""
        return self.metrics_history.get(metric_name, [])

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value of a metric."""
        history = self.get_history(metric_name)
        return history[-1] if history else None
