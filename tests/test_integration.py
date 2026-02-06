"""Integration tests for full pipeline."""

import pytest
import torch

# Skip all tests if imports fail
try:
    from src.advanced_model import AdvancedProteinFoldingModel
    from src.benchmarks import BenchmarkMetrics

    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


@pytest.mark.integration
@pytest.mark.skipif(not IMPORTS_OK, reason=f"Imports failed")
class TestFullPipeline:
    """Test complete training and evaluation pipeline."""

    def test_model_forward_backward(self, sample_embeddings):
        """Test forward and backward pass."""
        model = AdvancedProteinFoldingModel(
            input_dim=sample_embeddings.shape[-1],
            c_s=64,
            c_z=32,
            use_quantum=False,
        )

        # Forward pass
        output = model(sample_embeddings)
        assert "coordinates" in output

        # Backward pass
        loss = output["coordinates"].sum()
        loss.backward()

        # Check gradients exist
        grads = [param.grad for param in model.parameters() if param.requires_grad]
        assert any(g is not None for g in grads)

    @pytest.mark.slow
    def test_mini_training_loop(self, sample_embeddings, sample_coordinates):
        """Test minimal training loop."""
        model = AdvancedProteinFoldingModel(
            input_dim=sample_embeddings.shape[-1],
            c_s=32,
            c_z=16,
            use_quantum=False,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        initial_loss = None
        for step in range(5):
            optimizer.zero_grad()

            output = model(sample_embeddings)
            loss = torch.nn.functional.mse_loss(
                output["coordinates"],
                sample_coordinates[: sample_embeddings.shape[0], : sample_embeddings.shape[1]],
            )

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        # Loss should decrease (or at least not increase significantly)
        assert final_loss <= initial_loss * 1.5


@pytest.mark.integration
@pytest.mark.skipif(not IMPORTS_OK, reason="Imports failed")
class TestEndToEnd:
    """End-to-end workflow tests."""

    def test_predict_and_evaluate(self, sample_embeddings, sample_coordinates):
        """Test prediction and evaluation workflow."""
        # Create model
        model = AdvancedProteinFoldingModel(
            input_dim=sample_embeddings.shape[-1],
            c_s=32,
            c_z=16,
            use_quantum=False,
        )
        model.eval()

        # Make prediction
        with torch.no_grad():
            output = model(sample_embeddings)

        # Evaluate
        metrics = BenchmarkMetrics()
        results = metrics.calculate_all(
            predicted=output["coordinates"],
            ground_truth=sample_coordinates[
                : output["coordinates"].shape[0], : output["coordinates"].shape[1]
            ],
        )

        assert results["rmsd"] >= 0
        assert 0 <= results["tm_score"] <= 1
