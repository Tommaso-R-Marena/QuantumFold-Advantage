"""Property-based tests using Hypothesis."""

import pytest

torch = pytest.importorskip("torch")

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


try:
    from src.benchmarks import calculate_rmsd

    BENCHMARKS_AVAILABLE = True
except ImportError:
    BENCHMARKS_AVAILABLE = False


@pytest.mark.property
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
@pytest.mark.skipif(not BENCHMARKS_AVAILABLE, reason="Benchmarks not available")
class TestRMSDProperties:
    """Property-based tests for RMSD calculation."""

    @given(
        coords=st.lists(
            st.lists(
                st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=3, max_size=3
            ),
            min_size=5,
            max_size=50,
        )
    )
    @settings(max_examples=50, deadline=1000)
    def test_rmsd_identical_is_zero(self, coords):
        """RMSD of identical structures should be zero."""
        coords_tensor = torch.tensor([coords], dtype=torch.float32)
        rmsd = calculate_rmsd(coords_tensor, coords_tensor)
        assert rmsd < 1e-5

    @given(
        coords1=st.lists(
            st.lists(
                st.floats(min_value=-50, max_value=50, allow_nan=False), min_size=3, max_size=3
            ),
            min_size=5,
            max_size=20,
        )
    )
    @settings(max_examples=30, deadline=1000)
    def test_rmsd_non_negative(self, coords1):
        """RMSD should always be non-negative."""
        coords1_tensor = torch.tensor([coords1], dtype=torch.float32)
        coords2_tensor = coords1_tensor + torch.randn_like(coords1_tensor)

        rmsd = calculate_rmsd(coords1_tensor, coords2_tensor)
        assert rmsd >= 0

    @given(
        coords=st.lists(
            st.lists(
                st.floats(min_value=-50, max_value=50, allow_nan=False), min_size=3, max_size=3
            ),
            min_size=5,
            max_size=20,
        )
    )
    @settings(max_examples=30, deadline=1000)
    def test_rmsd_symmetry(self, coords):
        """RMSD(A, B) should equal RMSD(B, A)."""
        coords1 = torch.tensor([coords], dtype=torch.float32)
        coords2 = coords1 + torch.randn_like(coords1) * 0.5

        rmsd1 = calculate_rmsd(coords1, coords2)
        rmsd2 = calculate_rmsd(coords2, coords1)

        assert abs(rmsd1 - rmsd2) < 1e-5


@pytest.mark.property
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
class TestTensorProperties:
    """Property-based tests for tensor operations."""

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=5, max_value=50),
        embed_dim=st.integers(min_value=32, max_value=256),
    )
    @settings(max_examples=20, deadline=2000)
    def test_tensor_shapes(self, batch_size, seq_len, embed_dim):
        """Test that tensor operations preserve expected shapes."""
        embeddings = torch.randn(batch_size, seq_len, embed_dim)

        # Test shape preservation
        assert embeddings.shape == (batch_size, seq_len, embed_dim)

        # Test that operations don't create NaN
        normalized = torch.nn.functional.normalize(embeddings, dim=-1)
        assert not torch.isnan(normalized).any()
