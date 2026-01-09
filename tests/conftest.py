"""Pytest configuration and fixtures for QuantumFold-Advantage tests."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import warnings

# Suppress warnings during tests
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ====================
# Pytest Configuration
# ====================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "performance: marks performance/benchmark tests")
    config.addinivalue_line("markers", "stress: marks stress tests")


# ====================
# Basic Fixtures
# ====================

@pytest.fixture(scope="session")
def device():
    """Get test device (prefer CPU for CI)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


# ====================
# Sequence Fixtures
# ====================

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
def short_sequence():
    """Short sequence for quick tests."""
    return "ACDEFGHIKLMNPQRSTVWY"  # All 20 amino acids


@pytest.fixture
def long_sequence():
    """Longer sequence for scaling tests."""
    return "MKTAYIAK" * 20  # 160 residues


# ====================
# Tensor Fixtures
# ====================

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
def large_embeddings():
    """Larger embeddings for stress tests."""
    batch_size = 8
    seq_len = 100
    embed_dim = 256
    return torch.randn(batch_size, seq_len, embed_dim)


@pytest.fixture
def sample_pair_embeddings():
    """Sample pair embeddings for testing."""
    batch_size = 2
    seq_len = 20
    pair_dim = 64
    return torch.randn(batch_size, seq_len, seq_len, pair_dim)


# ====================
# Directory Fixtures
# ====================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


# ====================
# Configuration Fixtures
# ====================

@pytest.fixture
def config_dict():
    """Sample configuration dictionary."""
    return {
        'epochs': 5,
        'batch_size': 4,
        'learning_rate': 1e-3,
        'use_quantum': False,  # Disable for faster tests
        'n_qubits': 4,
        'quantum_depth': 2,
    }


@pytest.fixture
def training_config():
    """Training configuration for tests."""
    try:
        from src.advanced_training import TrainingConfig
        return TrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            use_amp=False,  # Disable AMP for testing
            use_ema=False,
        )
    except ImportError:
        return None


# ====================
# Model Fixtures
# ====================

@pytest.fixture
def simple_model(sample_embeddings):
    """Simple model for testing."""
    try:
        from src.advanced_model import AdvancedProteinFoldingModel
        return AdvancedProteinFoldingModel(
            input_dim=sample_embeddings.shape[-1],
            c_s=32,
            c_z=16,
            use_quantum=False,
        )
    except ImportError:
        return None


@pytest.fixture
def quantum_model(sample_embeddings):
    """Quantum-enabled model for testing."""
    try:
        from src.advanced_model import AdvancedProteinFoldingModel
        return AdvancedProteinFoldingModel(
            input_dim=sample_embeddings.shape[-1],
            c_s=32,
            c_z=16,
            use_quantum=True,
            n_qubits=4,
        )
    except ImportError:
        return None


# ====================
# Skip Conditions
# ====================

@pytest.fixture(scope="session")
def skip_if_no_gpu():
    """Skip test if GPU not available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


@pytest.fixture(scope="session")
def skip_if_no_esm():
    """Skip test if ESM not available."""
    try:
        import esm
    except ImportError:
        pytest.skip("ESM not installed")


# ====================
# Benchmark Fixtures
# ====================

@pytest.fixture
def benchmark_data():
    """Data for benchmark tests."""
    return {
        'sequences': [f"MKTAYIAK{i}" * 10 for i in range(10)],
        'expected_rmsd': [1.5, 2.0, 1.8, 2.2, 1.9, 2.1, 1.7, 2.3, 1.6, 2.4],
        'expected_tm': [0.85, 0.80, 0.83, 0.78, 0.82, 0.79, 0.84, 0.77, 0.86, 0.76],
    }


# ====================
# Cleanup
# ====================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Clear any cached data
    import gc
    gc.collect()
