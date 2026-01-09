"""Performance and benchmark tests."""

import time

import pytest
import torch

try:
    from src.advanced_model import AdvancedProteinFoldingModel

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


@pytest.mark.performance
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not available")
class TestPerformance:
    """Performance benchmarks."""

    def test_forward_pass_speed(self, benchmark, sample_embeddings):
        """Benchmark forward pass speed."""
        model = AdvancedProteinFoldingModel(
            input_dim=sample_embeddings.shape[-1],
            c_s=64,
            c_z=32,
            use_quantum=False,
        )
        model.eval()

        def forward():
            with torch.no_grad():
                return model(sample_embeddings)

        result = benchmark(forward)
        assert "coordinates" in result

    def test_memory_usage(self, sample_embeddings):
        """Test memory usage is reasonable."""
        model = AdvancedProteinFoldingModel(
            input_dim=sample_embeddings.shape[-1],
            c_s=64,
            c_z=32,
            use_quantum=False,
        )

        # Get initial memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            model = model.cuda()
            sample_embeddings = sample_embeddings.cuda()

        # Forward pass
        with torch.no_grad():
            output = model(sample_embeddings)

        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"Peak memory: {memory_mb:.2f} MB")
            # Should be reasonable (less than 2GB for small models)
            assert memory_mb < 2000

    @pytest.mark.slow
    def test_scaling_with_sequence_length(self):
        """Test how performance scales with sequence length."""
        timings = {}

        for seq_len in [10, 20, 50, 100]:
            model = AdvancedProteinFoldingModel(
                input_dim=128,
                c_s=32,
                c_z=16,
                use_quantum=False,
            )
            model.eval()

            embeddings = torch.randn(2, seq_len, 128)

            start = time.time()
            with torch.no_grad():
                output = model(embeddings)
            elapsed = time.time() - start

            timings[seq_len] = elapsed
            print(f"Seq length {seq_len}: {elapsed:.4f}s")

        # Should scale roughly quadratically or better
        # (due to attention mechanisms)
        assert timings[100] < timings[10] * 200  # Not worse than O(n^2)
