"""Tools for ensuring reproducibility across runs."""

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Use deterministic algorithms where possible
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    # Set environment variable for CUDA
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")


def save_reproducibility_info(config_dict: dict, save_path: str):
    """Save all information needed to reproduce experiment.

    Args:
        config_dict: Experiment configuration
        save_path: Where to save the information
    """
    import json
    import sys

    import torch

    repro_info = {
        "config": config_dict,
        "environment": {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        },
        "packages": {},
    }

    # Try to get package versions
    try:
        import pkg_resources

        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        repro_info["packages"] = installed_packages
    except:
        logger.warning("Could not retrieve package versions")

    with open(save_path, "w") as f:
        json.dump(repro_info, f, indent=2)

    logger.info(f"Reproducibility information saved to {save_path}")
