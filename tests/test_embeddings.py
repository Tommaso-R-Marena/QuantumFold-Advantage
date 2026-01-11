"""Tests for protein embeddings."""

import pytest

try:
    from src.protein_embeddings import ESM2Embedder

    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False


@pytest.mark.skipif(not ESM_AVAILABLE, reason="ESM not available")
@pytest.mark.slow
class TestESM2Embedder:
    """Tests for ESM2Embedder (slow tests, require model download)."""

    def test_initialization(self):
        """Test embedder initialization."""
        # Use smallest model for testing
        embedder = ESM2Embedder(model_name="esm2_t6_8M_UR50D")
        assert embedder.embed_dim > 0
        assert embedder.num_layers > 0

    def test_forward_pass(self, sample_sequences):
        """Test embedding generation."""
        embedder = ESM2Embedder(model_name="esm2_t6_8M_UR50D")

        output = embedder(sample_sequences)

        assert "embeddings" in output
        assert "mean_embedding" in output
        assert output["embeddings"].shape[0] == len(sample_sequences)
        assert output["embeddings"].shape[2] == embedder.embed_dim

    def test_single_sequence(self, sample_sequence):
        """Test with single sequence."""
        embedder = ESM2Embedder(model_name="esm2_t6_8M_UR50D")

        output = embedder([sample_sequence])

        assert output["embeddings"].shape[0] == 1
        assert output["embeddings"].shape[1] == len(sample_sequence)
