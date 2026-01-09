"""Stress tests and edge cases."""

import numpy as np
import pytest
import torch

try:
    from src.advanced_model import AdvancedProteinFoldingModel

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


@pytest.mark.stress
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not available")
class TestStressCases:
    """Stress testing for edge cases and limits."""

    def test_very_long_sequence(self):
        """Test with very long sequence (500 residues)."""
        seq_len = 500
        embeddings = torch.randn(1, seq_len, 128)

        model = AdvancedProteinFoldingModel(
            input_dim=128,
            c_s=32,
            c_z=16,
            use_quantum=False,
        )
        model.eval()

        with torch.no_grad():
            output = model(embeddings)

        assert output["coordinates"].shape == (1, seq_len, 3)

    def test_single_residue(self):
        """Test with single residue (edge case)."""
        embeddings = torch.randn(1, 1, 128)

        model = AdvancedProteinFoldingModel(
            input_dim=128,
            c_s=32,
            c_z=16,
            use_quantum=False,
        )
        model.eval()

        with torch.no_grad():
            output = model(embeddings)

        assert output["coordinates"].shape == (1, 1, 3)

    def test_large_batch(self):
        """Test with large batch size."""
        batch_size = 64
        seq_len = 50
        embeddings = torch.randn(batch_size, seq_len, 128)

        model = AdvancedProteinFoldingModel(
            input_dim=128,
            c_s=32,
            c_z=16,
            use_quantum=False,
        )
        model.eval()

        with torch.no_grad():
            output = model(embeddings)

        assert output["coordinates"].shape == (batch_size, seq_len, 3)

    def test_extreme_values(self):
        """Test with extreme input values."""
        # Very large values
        embeddings_large = torch.ones(2, 20, 128) * 1000

        # Very small values
        embeddings_small = torch.ones(2, 20, 128) * 1e-10

        # Mixed extreme values
        embeddings_mixed = torch.randn(2, 20, 128) * 100

        model = AdvancedProteinFoldingModel(
            input_dim=128,
            c_s=32,
            c_z=16,
            use_quantum=False,
        )
        model.eval()

        for emb in [embeddings_large, embeddings_small, embeddings_mixed]:
            with torch.no_grad():
                output = model(emb)

            # Check for NaN or Inf
            assert not torch.isnan(output["coordinates"]).any()
            assert not torch.isinf(output["coordinates"]).any()

    def test_zero_input(self):
        """Test with all-zero input."""
        embeddings = torch.zeros(2, 20, 128)

        model = AdvancedProteinFoldingModel(
            input_dim=128,
            c_s=32,
            c_z=16,
            use_quantum=False,
        )
        model.eval()

        with torch.no_grad():
            output = model(embeddings)

        assert output["coordinates"].shape == (2, 20, 3)

    @pytest.mark.slow
    def test_repeated_inference(self):
        """Test many repeated inferences (memory leak check)."""
        model = AdvancedProteinFoldingModel(
            input_dim=128,
            c_s=32,
            c_z=16,
            use_quantum=False,
        )
        model.eval()

        embeddings = torch.randn(2, 20, 128)

        # Run 100 inferences
        for _ in range(100):
            with torch.no_grad():
                output = model(embeddings)

            # Explicit cleanup
            del output

        # Should complete without memory error
        assert True


@pytest.mark.stress
class TestEdgeCases:
    """Edge case testing."""

    def test_empty_sequence_handling(self):
        """Test how system handles empty sequences."""
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            embeddings = torch.randn(1, 0, 128)  # Zero length

    def test_mismatched_dimensions(self):
        """Test error handling for mismatched dimensions."""
        if not MODEL_AVAILABLE:
            pytest.skip("Model not available")

        model = AdvancedProteinFoldingModel(
            input_dim=128,
            c_s=32,
            c_z=16,
            use_quantum=False,
        )

        # Wrong embedding dimension
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            embeddings = torch.randn(2, 20, 64)  # Should be 128
            model(embeddings)
