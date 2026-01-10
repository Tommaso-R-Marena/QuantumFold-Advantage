"""Tests for protein folding model."""

import pytest
import torch

try:
    from src.advanced_model import AdvancedProteinFoldingModel

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not available")
class TestAdvancedProteinFoldingModel:
    """Tests for AdvancedProteinFoldingModel."""

    def test_initialization_quantum(self):
        """Test quantum model initialization."""
        model = AdvancedProteinFoldingModel(
            input_dim=128, c_s=64, c_z=32, use_quantum=True, n_qubits=4
        )
        assert model.use_quantum

    def test_initialization_classical(self):
        """Test classical model initialization."""
        model = AdvancedProteinFoldingModel(
            input_dim=128, c_s=64, c_z=32, use_quantum=False
        )
        assert not model.use_quantum

    def test_forward_pass(self, sample_embeddings):
        """Test model forward pass."""
        model = AdvancedProteinFoldingModel(
            input_dim=sample_embeddings.shape[-1],
            c_s=64,
            c_z=32,
            use_quantum=False,  # Use classical for faster testing
        )

        output = model(sample_embeddings)

        assert "coordinates" in output
        assert output["coordinates"].shape[0] == sample_embeddings.shape[0]
        assert output["coordinates"].shape[1] == sample_embeddings.shape[1]
        assert output["coordinates"].shape[2] == 3  # x, y, z

    def test_output_contains_required_keys(self, sample_embeddings):
        """Test that output contains all required keys."""
        model = AdvancedProteinFoldingModel(
            input_dim=sample_embeddings.shape[-1], c_s=64, c_z=32
        )

        output = model(sample_embeddings)

        assert "coordinates" in output
        # Other keys may be optional depending on configuration
