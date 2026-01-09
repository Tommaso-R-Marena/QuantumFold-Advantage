"""Pytest configuration and fixtures for QuantumFold-Advantage tests."""

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Get test device (prefer CPU for CI)."""
    return torch.device("cpu")


@pytest.fixture
def sample_sequence():
    """Sample protein sequence for testing."""
    return "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"


@pytest.fixture
def sample_sequences():
    """Batch of sample sequences."""
    return [
        "MKTAYIAKQRQISFVK",
        "SHFSRQLEERLGLIE",
        "VQAPILSRVGDGTQD",
    ]


@pytest.fixture
def sample_coordinates():
    """Sample 3D coordinates for testing."""
    batch_size = 2
    seq_len = 20
    return torch.randn(batch_size, seq_len, 3)


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    batch_size = 2
    seq_len = 20
    embed_dim = 128
    return torch.randn(batch_size, seq_len, embed_dim)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def config_dict():
    """Sample configuration dictionary."""
    return {
        "epochs": 5,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "use_quantum": True,
        "n_qubits": 4,
        "quantum_depth": 2,
    }
