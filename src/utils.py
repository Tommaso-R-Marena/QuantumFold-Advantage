"""Utility functions for reproducibility and device management."""
import random
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def detect_device():
    """
    Detect available compute device.
    
    Returns:
        str: 'cuda' if available, else 'cpu'
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info("CUDA not available, using CPU")
    return device

def setup_logging(level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)