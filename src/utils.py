"""
Utility functions for reproducibility, device detection, and logging.

Key functions:
- set_seed: Set random seeds for NumPy, PyTorch, and Python random
- get_device: Detect available compute device (GPU/CPU)
- setup_logging: Configure logging with specified level
"""

import logging
import random
import sys

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    
    References:
        Ensures reproducible results across NumPy, PyTorch, and Python random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Additional reproducibility settings for CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(mode: str = "auto") -> torch.device:
    """
    Detect and return compute device.
    
    Args:
        mode: "auto", "cpu", or "gpu"
    
    Returns:
        torch.device object
    
    Notes:
        In "auto" mode, prefers CUDA GPU if available, else CPU.
        Logs device selection for provenance tracking.
    """
    logger = logging.getLogger(__name__)
    
    if mode == "cpu":
        device = torch.device("cpu")
        logger.info("Device mode forced to CPU")
    elif mode == "gpu":
        if not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA not available; falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-detected GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Auto-detected device: CPU")
    
    return device


def setup_logging(level: str = "INFO"):
    """
    Configure logging with specified level.
    
    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def adjust_config_for_device(config: dict, device: torch.device) -> dict:
    """
    Adjust training configuration based on device capabilities.
    
    Args:
        config: Configuration dictionary
        device: torch.device
    
    Returns:
        Adjusted configuration dictionary
    
    Notes:
        CPU fallback: reduces batch size, epochs, and model capacity to
        ensure reasonable runtime on CPU-only environments (e.g., Colab free tier).
    """
    logger = logging.getLogger(__name__)
    adjusted = config.copy()
    
    if device.type == "cpu":
        # Reduce resource requirements for CPU
        if adjusted.get("batch_size", 4) > 2:
            adjusted["batch_size"] = 2
            logger.info("CPU mode: Reduced batch_size to 2")
        
        if adjusted.get("epochs", 10) > 5:
            adjusted["epochs"] = 5
            logger.info("CPU mode: Reduced epochs to 5")
        
        if adjusted.get("hidden_dim", 64) > 32:
            adjusted["hidden_dim"] = 32
            logger.info("CPU mode: Reduced hidden_dim to 32")
    
    return adjusted
