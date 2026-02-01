"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def set_seed():
    """Set random seeds for reproducibility."""
    def _set_seed(seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    return _set_seed


@pytest.fixture
def sample_sequence():
    """Sample protein sequence for testing."""
    return "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"


@pytest.fixture
def sample_embedding(device):
    """Sample embedding tensor."""
    batch_size, seq_len, embed_dim = 2, 50, 128
    return torch.randn(batch_size, seq_len, embed_dim, device=device)


@pytest.fixture
def sample_coordinates(device):
    """Sample 3D coordinates."""
    batch_size, n_res = 2, 50
    return torch.randn(batch_size, n_res, 3, device=device)


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir
