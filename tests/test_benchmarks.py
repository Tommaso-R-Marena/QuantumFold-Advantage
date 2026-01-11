"""Tests for benchmarking utilities."""

import numpy as np
import pytest
import torch

try:
    from src.benchmarks import (
        BenchmarkMetrics,
        calculate_rmsd,
    )

    BENCHMARKS_AVAILABLE = True
except ImportError:
    BENCHMARKS_AVAILABLE = False


@pytest.mark.skipif(not BENCHMARKS_AVAILABLE, reason="Benchmarks module not available")
class TestRMSD:
    """Tests for RMSD calculation."""

    def test_identical_structures(self, sample_coordinates):
        """Test RMSD of identical structures is zero."""
        rmsd = calculate_rmsd(sample_coordinates, sample_coordinates)
        assert rmsd < 1e-6

    def test_known_rmsd(self):
        """Test RMSD calculation with known values."""
        pred = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        true = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

        rmsd = calculate_rmsd(pred, true)
        expected_rmsd = np.sqrt(2.0) / np.sqrt(2)  # sqrt(2)/sqrt(N_atoms)

        assert abs(rmsd - expected_rmsd) < 0.01


@pytest.mark.skipif(not BENCHMARKS_AVAILABLE, reason="Benchmarks module not available")
@pytest.mark.integration
class TestBenchmarkMetrics:
    """Integration tests for benchmark metrics."""

    def test_metrics_calculation(self, sample_coordinates):
        """Test comprehensive metrics calculation."""
        metrics = BenchmarkMetrics()

        pred_coords = sample_coordinates
        true_coords = sample_coordinates + torch.randn_like(sample_coordinates) * 0.5

        results = metrics.calculate_all(predicted=pred_coords, ground_truth=true_coords)

        assert "rmsd" in results
        assert "tm_score" in results
        assert "gdt_ts" in results
        assert all(v >= 0 for v in results.values())
