"""Regression tests to catch breaking changes."""

import pytest
import torch
import json
from pathlib import Path

try:
    from src.benchmarks import calculate_rmsd, calculate_tm_score
    BENCHMARKS_AVAILABLE = True
except ImportError:
    BENCHMARKS_AVAILABLE = False


@pytest.mark.regression
@pytest.mark.skipif(not BENCHMARKS_AVAILABLE, reason="Benchmarks not available")
class TestRegressionBenchmarks:
    """Regression tests for benchmark calculations."""
    
    def test_rmsd_known_values(self):
        """Test RMSD against known reference values."""
        # Reference structure
        ref_coords = torch.tensor([[[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]]])
        
        # Test structure (known displacement)
        test_coords = torch.tensor([[[
            [0.1, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [0.1, 1.1, 0.1],
            [0.1, 0.1, 1.1],
        ]]])
        
        rmsd = calculate_rmsd(test_coords, ref_coords)
        
        # Expected RMSD for uniform 0.1 displacement in each dimension
        expected_rmsd = (0.1 ** 2 * 3) ** 0.5
        
        assert abs(rmsd - expected_rmsd) < 0.01, f"RMSD changed: got {rmsd}, expected {expected_rmsd}"
    
    def test_tm_score_bounds(self):
        """Test TM-score stays within [0, 1] bounds."""
        # Random structures
        coords1 = torch.randn(1, 20, 3)
        coords2 = torch.randn(1, 20, 3)
        
        tm_score = calculate_tm_score(coords1, coords2)
        
        assert 0 <= tm_score <= 1, f"TM-score out of bounds: {tm_score}"


@pytest.mark.regression
class TestAPIStability:
    """Test that public APIs remain stable."""
    
    def test_module_imports(self):
        """Test that expected modules can be imported."""
        # These should always be importable
        try:
            import src
            assert hasattr(src, '__version__')
        except ImportError:
            pytest.fail("Core src module import failed")
    
    def test_config_structure(self):
        """Test that config files maintain expected structure."""
        # Check pyproject.toml exists and is valid
        pyproject_path = Path("pyproject.toml")
        assert pyproject_path.exists(), "pyproject.toml missing"
        
        # Check requirements.txt exists
        req_path = Path("requirements.txt")
        assert req_path.exists(), "requirements.txt missing"
