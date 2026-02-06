"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project package imports are stable regardless of invocation directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


@pytest.fixture
def sample_embeddings(sample_embedding):
    """Backward-compatible fixture alias."""
    return sample_embedding


@pytest.fixture
def sample_sequences(sample_sequence):
    """Sample batch of protein sequences."""
    return [sample_sequence, sample_sequence[:50]]


@pytest.fixture
def benchmark():
    """Fallback benchmark fixture when pytest-benchmark plugin is unavailable."""

    def _run(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    return _run
