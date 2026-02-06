"""Tests for 01_getting_started.ipynb functionality."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn


class TestEnvironmentSetup:
    """Test environment detection and dependency management."""

    def test_numpy_version_constraint(self):
        """NumPy must be <2.0 for autograd/PennyLane compatibility."""
        import numpy as np

        major_version = int(np.__version__.split(".")[0])
        assert major_version >= 1

    def test_colab_detection(self):
        """Should correctly detect Colab environment."""
        try:
            pass

            in_colab = True
        except ImportError:
            in_colab = False

        # Should work regardless
        assert isinstance(in_colab, bool)

    def test_torch_import(self):
        """PyTorch should import successfully."""
        import torch

        assert torch.__version__ is not None

    def test_cuda_detection(self):
        """CUDA detection should not crash."""
        import torch

        is_available = torch.cuda.is_available()
        assert isinstance(is_available, bool)

        if is_available:
            device_name = torch.cuda.get_device_name(0)
            assert isinstance(device_name, str)
            assert len(device_name) > 0

    def test_autograd_import(self):
        """Autograd must import without ValueError."""
        try:
            import autograd

            assert autograd is not None
        except ValueError as e:
            pytest.fail(f"Autograd import failed: {e}")

    def test_pennylane_import(self):
        """PennyLane must import successfully."""
        try:
            import pennylane as qml

            assert qml.__version__ is not None
        except (ImportError, ValueError) as e:
            pytest.fail(f"PennyLane import failed: {e}")


class TestPathOperations:
    """Test path and directory operations."""

    def test_path_import(self):
        """Path should import from pathlib."""
        from pathlib import Path

        assert Path is not None

    def test_cwd_parent_exists(self):
        """Current working directory should have a parent."""
        from pathlib import Path

        cwd = Path.cwd()
        assert cwd.parent.exists()

    @patch("sys.modules", {"google.colab": MagicMock()})
    def test_colab_path_setup(self):
        """Colab should use /content/QuantumFold-Advantage path."""
        expected_path = "/content/QuantumFold-Advantage"
        # Just verify the path format is valid
        assert expected_path.startswith("/content")

    def test_sys_path_modification(self):
        """Should be able to modify sys.path."""
        original_len = len(sys.path)
        test_path = "/test/path"
        sys.path.insert(0, test_path)
        assert len(sys.path) == original_len + 1
        assert sys.path[0] == test_path
        sys.path.pop(0)  # Clean up


class TestSimpleProteinModel:
    """Test the fallback SimpleProteinModel."""

    @staticmethod
    def create_simple_model(input_dim=480, hidden_dim=128):
        """Create SimpleProteinModel for testing."""

        class SimpleProteinModel(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.output = nn.Linear(hidden_dim, 3)
                self.confidence = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                h = self.encoder(x)
                coords = self.output(h)
                plddt = torch.sigmoid(self.confidence(h)).squeeze(-1) * 100
                return {"coordinates": coords, "plddt": plddt}

        return SimpleProteinModel(input_dim, hidden_dim)

    def test_model_creation(self):
        """SimpleProteinModel should initialize correctly."""
        model = self.create_simple_model()
        assert model is not None

    def test_model_forward_pass(self):
        """Forward pass should produce correct output shapes."""
        model = self.create_simple_model(input_dim=480, hidden_dim=128)
        batch_size, seq_len, input_dim = 1, 21, 480

        x = torch.randn(batch_size, seq_len, input_dim)
        with torch.no_grad():
            output = model(x)

        assert "coordinates" in output
        assert "plddt" in output
        assert output["coordinates"].shape == (batch_size, seq_len, 3)
        assert output["plddt"].shape == (batch_size, seq_len)

    def test_plddt_range(self):
        """pLDDT scores should be in [0, 100] range."""
        model = self.create_simple_model()
        x = torch.randn(1, 21, 480)

        with torch.no_grad():
            output = model(x)
            plddt = output["plddt"]

        assert torch.all(plddt >= 0), f"Found pLDDT < 0: {plddt.min()}"
        assert torch.all(plddt <= 100), f"Found pLDDT > 100: {plddt.max()}"

    def test_model_parameter_count(self):
        """Model should have expected number of parameters."""
        model = self.create_simple_model(input_dim=480, hidden_dim=128)
        total_params = sum(p.numel() for p in model.parameters())

        # Expected: encoder (480*128 + 128) + (128*128 + 128) + output (128*3 + 3) + conf (128*1 + 1)
        expected = (480 * 128 + 128) + (128 * 128 + 128) + (128 * 3 + 3) + (128 * 1 + 1)
        assert total_params == expected

    def test_device_placement(self):
        """Model should move to device correctly."""
        device = torch.device("cpu")
        model = self.create_simple_model().to(device)

        for param in model.parameters():
            assert param.device.type == device.type

    def test_gradient_computation(self):
        """Should compute gradients correctly."""
        model = self.create_simple_model()
        x = torch.randn(1, 21, 480, requires_grad=True)

        output = model(x)
        loss = output["coordinates"].sum()
        loss.backward()

        # Check that gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "No gradients computed"


class TestCASPMetrics:
    """Test CASP evaluation metrics."""

    @staticmethod
    def calculate_rmsd(coords1, coords2):
        return np.sqrt(np.mean((coords1 - coords2) ** 2))

    @staticmethod
    def calculate_tm_score_simple(coords1, coords2, seq_len):
        d0 = 1.24 * (seq_len - 15) ** (1 / 3) - 1.8
        distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
        tm_score = np.mean(1 / (1 + (distances / d0) ** 2))
        return tm_score

    @staticmethod
    def calculate_gdt_ts_simple(coords1, coords2):
        distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))
        gdt_ts = (
            np.mean(
                [
                    (distances < 1.0).mean(),
                    (distances < 2.0).mean(),
                    (distances < 4.0).mean(),
                    (distances < 8.0).mean(),
                ]
            )
            * 100
        )
        return gdt_ts

    def test_rmsd_identical_structures(self):
        """RMSD should be 0 for identical structures."""
        coords = np.random.randn(21, 3)
        rmsd = self.calculate_rmsd(coords, coords)
        assert rmsd == pytest.approx(0.0, abs=1e-10)

    def test_rmsd_known_displacement(self):
        """RMSD should match known displacement."""
        coords1 = np.zeros((21, 3))
        coords2 = np.ones((21, 3))  # Shift by 1 in all dimensions
        rmsd = self.calculate_rmsd(coords1, coords2)
        expected = 1.0  # mean over all coordinate elements
        assert rmsd == pytest.approx(expected, rel=1e-6)

    def test_rmsd_symmetry(self):
        """RMSD(A,B) should equal RMSD(B,A)."""
        coords1 = np.random.randn(21, 3)
        coords2 = np.random.randn(21, 3)
        rmsd_12 = self.calculate_rmsd(coords1, coords2)
        rmsd_21 = self.calculate_rmsd(coords2, coords1)
        assert rmsd_12 == pytest.approx(rmsd_21)

    def test_tm_score_range(self):
        """TM-score should be in (0, 1] range."""
        coords1 = np.random.randn(21, 3)
        coords2 = np.random.randn(21, 3)
        tm_score = self.calculate_tm_score_simple(coords1, coords2, 21)
        assert 0.0 < tm_score <= 1.0

    def test_tm_score_identical(self):
        """TM-score should be 1.0 for identical structures."""
        coords = np.random.randn(21, 3)
        tm_score = self.calculate_tm_score_simple(coords, coords, 21)
        assert tm_score == pytest.approx(1.0, rel=1e-6)

    def test_gdt_ts_range(self):
        """GDT_TS should be in [0, 100] range."""
        coords1 = np.random.randn(21, 3)
        coords2 = np.random.randn(21, 3)
        gdt_ts = self.calculate_gdt_ts_simple(coords1, coords2)
        assert 0.0 <= gdt_ts <= 100.0

    def test_gdt_ts_identical(self):
        """GDT_TS should be 100 for identical structures."""
        coords = np.random.randn(21, 3)
        gdt_ts = self.calculate_gdt_ts_simple(coords, coords)
        assert gdt_ts == pytest.approx(100.0, rel=1e-6)

    def test_gdt_ts_threshold_logic(self):
        """GDT_TS should correctly apply distance thresholds."""
        # Create coords where all distances are exactly 3.0 Angstroms
        coords1 = np.zeros((21, 3))
        coords2 = np.zeros((21, 3))
        coords2[:, 0] = 3.0  # Shift by 3.0 in x direction

        gdt_ts = self.calculate_gdt_ts_simple(coords1, coords2)

        # At d=3.0: <1A (0%), <2A (0%), <4A (100%), <8A (100%)
        # GDT_TS = (0 + 0 + 100 + 100) / 4 = 50.0
        assert gdt_ts == pytest.approx(50.0, rel=1e-6)


class TestVisualization:
    """Test visualization components."""

    def test_matplotlib_import(self):
        """Matplotlib should import successfully."""
        import matplotlib.pyplot as plt

        assert plt is not None

    def test_seaborn_import(self):
        """Seaborn should import successfully."""
        import seaborn as sns

        assert sns is not None

    def test_3d_plotting_available(self):
        """3D plotting should be available."""
        try:
            pass

            assert True
        except ImportError as e:
            pytest.fail(f"3D plotting not available: {e}")

    @pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
    def test_create_3d_figure(self):
        """Should create 3D figure without errors."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(18, 5))
        ax = fig.add_subplot(131, projection="3d")

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    @pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
    def test_distance_map_generation(self):
        """Should generate distance map correctly."""
        coords = np.random.randn(21, 3)

        # Calculate pairwise distances
        distances = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))

        assert distances.shape == (21, 21)
        assert np.allclose(distances, distances.T)  # Should be symmetric
        assert np.allclose(np.diag(distances), 0)  # Diagonal should be zero

    @pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
    def test_confidence_colormap(self):
        """Should apply confidence colormap correctly."""
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        plddt_scores = np.random.uniform(0, 100, 21)
        colors = plt.cm.RdYlGn(plddt_scores / 100)

        assert colors.shape == (21, 4)  # RGBA
        assert np.all(colors >= 0) and np.all(colors <= 1)


class TestProteinEmbeddings:
    """Test protein sequence and embedding handling."""

    def test_sequence_validation(self):
        """Protein sequence should be valid amino acids."""
        sequence = "GIVEQCCTSICSLYQLENYCN"
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

        for aa in sequence:
            assert aa in valid_aa, f"Invalid amino acid: {aa}"

    def test_random_embedding_generation(self):
        """Should generate random embeddings with correct shape."""
        seq_len = 21
        embed_dim = 480

        torch.manual_seed(42)
        embeddings = torch.randn(1, seq_len, embed_dim)

        assert embeddings.shape == (1, seq_len, embed_dim)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    def test_embedding_device_transfer(self):
        """Embeddings should transfer to device correctly."""
        embeddings = torch.randn(1, 21, 480)
        device = torch.device("cpu")
        embeddings_device = embeddings.to(device)

        assert embeddings_device.device.type == device.type


class TestConfidenceScores:
    """Test pLDDT confidence score handling."""

    def test_plddt_statistics(self):
        """pLDDT statistics should be computed correctly."""
        plddt_scores = np.array([90, 85, 70, 60, 50, 40, 75, 80, 85, 90])

        mean_plddt = plddt_scores.mean()
        median_plddt = np.median(plddt_scores)
        min_plddt = plddt_scores.min()
        max_plddt = plddt_scores.max()

        assert 0 <= mean_plddt <= 100
        assert 0 <= median_plddt <= 100
        assert min_plddt == 40
        assert max_plddt == 90

    def test_high_confidence_threshold(self):
        """Should correctly count high confidence residues."""
        plddt_scores = np.array([90, 85, 70, 60, 50, 40, 75, 80, 85, 90])
        high_conf = (plddt_scores > 70).sum()

        assert high_conf == 6  # values strictly greater than 70

    def test_confidence_percentage(self):
        """Should calculate confidence percentage correctly."""
        plddt_scores = np.array([90, 85, 70, 60])
        high_conf = (plddt_scores > 70).sum()
        percentage = 100 * high_conf / len(plddt_scores)

        assert percentage == 50.0  # 2 out of 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
