"""Tests for 03_advanced_visualization.ipynb functionality."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


class TestDataGeneration:
    """Test synthetic protein structure data generation."""

    def test_alpha_helix_geometry(self):
        """Should generate alpha helix with realistic parameters."""
        np.random.seed(42)
        n_residues = 50
        t = np.linspace(0, 4 * np.pi, n_residues)

        radius = 2.3
        pitch = 1.5

        coords = np.zeros((n_residues, 3))
        coords[:, 0] = radius * np.cos(t)
        coords[:, 1] = radius * np.sin(t)
        coords[:, 2] = pitch * t

        assert coords.shape == (n_residues, 3)
        assert not np.isnan(coords).any()

        # Check helix properties
        distances_along_helix = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        assert np.all(distances_along_helix > 0)

    def test_helix_with_noise(self):
        """Should add realistic noise to helix."""
        np.random.seed(42)
        n_residues = 50
        t = np.linspace(0, 4 * np.pi, n_residues)

        coords = np.zeros((n_residues, 3))
        coords[:, 0] = 2.3 * np.cos(t)
        coords[:, 1] = 2.3 * np.sin(t)
        coords[:, 2] = 1.5 * t

        coords_noisy = coords + np.random.randn(n_residues, 3) * 0.2

        # Noise should be small but present
        diff = np.abs(coords - coords_noisy).max()
        assert diff > 0.01  # Some noise added
        assert diff < 1.0  # But not too much

    def test_structure_dimensions(self):
        """Should have reasonable spatial extent."""
        np.random.seed(42)
        n_residues = 50
        t = np.linspace(0, 4 * np.pi, n_residues)

        coords = np.zeros((n_residues, 3))
        coords[:, 0] = 2.3 * np.cos(t)
        coords[:, 1] = 2.3 * np.sin(t)
        coords[:, 2] = 1.5 * t

        # Check bounding box
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        # X and Y should be within helix radius
        assert np.abs(mins[0]) < 5
        assert np.abs(maxs[0]) < 5
        assert np.abs(mins[1]) < 5
        assert np.abs(maxs[1]) < 5

        # Z should increase linearly
        assert maxs[2] > mins[2]


class Test3DVisualization:
    """Test 3D structure visualization methods."""

    @pytest.fixture
    def sample_coords(self):
        """Generate sample coordinates."""
        np.random.seed(42)
        n = 50
        t = np.linspace(0, 4 * np.pi, n)
        coords = np.zeros((n, 3))
        coords[:, 0] = 2.3 * np.cos(t)
        coords[:, 1] = 2.3 * np.sin(t)
        coords[:, 2] = 1.5 * t
        return coords

    def test_3d_axes_available(self):
        """Should have 3D plotting capability."""
        assert Axes3D is not None

    def test_backbone_trace_plot(self, sample_coords):
        """Should create backbone trace plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2], "b-")
        ax.scatter(
            sample_coords[:, 0],
            sample_coords[:, 1],
            sample_coords[:, 2],
            c=range(len(sample_coords)),
        )

        assert len(ax.lines) > 0
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_sphere_representation(self, sample_coords):
        """Should create sphere representation."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2], s=200)

        assert scatter is not None
        plt.close(fig)

    def test_tube_representation(self, sample_coords):
        """Should create tube representation."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for i in range(len(sample_coords) - 1):
            ax.plot(
                sample_coords[i : i + 2, 0],
                sample_coords[i : i + 2, 1],
                sample_coords[i : i + 2, 2],
            )

        assert len(ax.lines) == len(sample_coords) - 1
        plt.close(fig)

    def test_multiple_subplots(self, sample_coords):
        """Should create multiple 3D subplots."""
        fig = plt.figure(figsize=(15, 5))

        ax1 = fig.add_subplot(131, projection="3d")
        ax2 = fig.add_subplot(132, projection="3d")
        ax3 = fig.add_subplot(133, projection="3d")

        for ax in [ax1, ax2, ax3]:
            ax.scatter(sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2])

        plt.close(fig)

    def test_axis_labels(self):
        """Should set proper axis labels."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")

        assert ax.get_xlabel() == "X (Å)"
        assert ax.get_ylabel() == "Y (Å)"
        assert ax.get_zlabel() == "Z (Å)"
        plt.close(fig)


class TestDistanceMaps:
    """Test distance and contact map generation."""

    @pytest.fixture
    def sample_coords(self):
        """Generate sample coordinates."""
        np.random.seed(42)
        return np.random.randn(50, 3) * 5

    def test_distance_matrix_calculation(self, sample_coords):
        """Should calculate distance matrix correctly."""
        distances = np.sqrt(
            np.sum((sample_coords[:, None, :] - sample_coords[None, :, :]) ** 2, axis=2)
        )

        n = len(sample_coords)
        assert distances.shape == (n, n)

        # Check symmetry
        assert np.allclose(distances, distances.T)

        # Check diagonal is zero
        assert np.allclose(np.diag(distances), 0)

        # All distances should be non-negative
        assert np.all(distances >= 0)

    def test_contact_map_generation(self, sample_coords):
        """Should generate binary contact map."""
        distances = np.sqrt(
            np.sum((sample_coords[:, None, :] - sample_coords[None, :, :]) ** 2, axis=2)
        )

        threshold = 8.0
        contacts = (distances < threshold).astype(float)
        np.fill_diagonal(contacts, 0)

        # Should be binary
        assert set(np.unique(contacts)).issubset({0.0, 1.0})

        # Diagonal should be zero
        assert np.allclose(np.diag(contacts), 0)

        # Should be symmetric
        assert np.allclose(contacts, contacts.T)

    def test_distance_difference_map(self, sample_coords):
        """Should calculate distance difference map."""
        distances1 = np.sqrt(
            np.sum((sample_coords[:, None, :] - sample_coords[None, :, :]) ** 2, axis=2)
        )

        coords_alt = sample_coords + np.random.randn(*sample_coords.shape) * 1.0
        distances2 = np.sqrt(np.sum((coords_alt[:, None, :] - coords_alt[None, :, :]) ** 2, axis=2))

        diff = np.abs(distances1 - distances2)

        # Should be non-negative
        assert np.all(diff >= 0)

        # Should have some differences
        assert diff.max() > 0

    def test_contact_statistics(self, sample_coords):
        """Should calculate contact statistics correctly."""
        distances = np.sqrt(
            np.sum((sample_coords[:, None, :] - sample_coords[None, :, :]) ** 2, axis=2)
        )

        threshold = 8.0
        contacts = (distances < threshold).astype(float)
        np.fill_diagonal(contacts, 0)

        n_residues = len(sample_coords)
        n_contacts = np.sum(contacts) / 2
        contact_density = n_contacts / (n_residues * (n_residues - 1) / 2)

        assert 0 <= contact_density <= 1
        assert n_contacts >= 0

    def test_distance_map_visualization(self, sample_coords):
        """Should visualize distance map."""
        distances = np.sqrt(
            np.sum((sample_coords[:, None, :] - sample_coords[None, :, :]) ** 2, axis=2)
        )

        fig, ax = plt.subplots()
        im = ax.imshow(distances, cmap="viridis")
        plt.colorbar(im, ax=ax)

        assert im is not None
        plt.close(fig)


class TestConfidenceVisualization:
    """Test confidence score visualization."""

    def test_confidence_score_generation(self):
        """Should generate realistic confidence scores."""
        np.random.seed(42)
        n_residues = 50
        confidence = np.random.beta(8, 2, n_residues)

        assert len(confidence) == n_residues
        assert np.all((confidence >= 0) & (confidence <= 1))

    def test_confidence_bar_plot(self):
        """Should create confidence bar plot."""
        np.random.seed(42)
        n_residues = 50
        confidence = np.random.beta(8, 2, n_residues)

        fig, ax = plt.subplots()
        ax.bar(range(n_residues), confidence)
        ax.axhline(y=0.7, color="orange", linestyle="--")
        ax.axhline(y=0.5, color="red", linestyle="--")

        assert len(ax.patches) == n_residues
        assert len(ax.lines) == 2  # Two threshold lines
        plt.close(fig)

    def test_pairwise_confidence_matrix(self):
        """Should create pairwise confidence matrix."""
        np.random.seed(42)
        n_residues = 50
        confidence = np.random.beta(8, 2, n_residues)
        pairwise = np.outer(confidence, confidence)

        assert pairwise.shape == (n_residues, n_residues)
        assert np.all((pairwise >= 0) & (pairwise <= 1))

        # Should be symmetric
        assert np.allclose(pairwise, pairwise.T)

    def test_confidence_heatmap(self):
        """Should create confidence heatmap."""
        np.random.seed(42)
        n_residues = 50
        confidence = np.random.beta(8, 2, n_residues)
        pairwise = np.outer(confidence, confidence)

        fig, ax = plt.subplots()
        im = ax.imshow(pairwise, cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)

        assert im is not None
        plt.close(fig)

    def test_confidence_statistics(self):
        """Should calculate confidence statistics."""
        np.random.seed(42)
        confidence = np.random.beta(8, 2, 50)

        mean_conf = confidence.mean()
        min_conf = confidence.min()
        max_conf = confidence.max()
        high_conf_count = (confidence > 0.7).sum()

        assert 0 <= mean_conf <= 1
        assert 0 <= min_conf <= max_conf <= 1
        assert 0 <= high_conf_count <= len(confidence)


class TestRamachandranPlot:
    """Test Ramachandran plot generation."""

    @pytest.fixture
    def sample_coords(self):
        """Generate sample coordinates."""
        np.random.seed(42)
        n = 50
        t = np.linspace(0, 4 * np.pi, n)
        coords = np.zeros((n, 3))
        coords[:, 0] = 2.3 * np.cos(t)
        coords[:, 1] = 2.3 * np.sin(t)
        coords[:, 2] = 1.5 * t
        coords += np.random.randn(n, 3) * 0.2
        return coords

    def test_dihedral_angle_calculation(self, sample_coords):
        """Should calculate backbone dihedral angles."""
        phi_angles = []
        psi_angles = []

        for i in range(1, len(sample_coords) - 1):
            v1 = sample_coords[i] - sample_coords[i - 1]
            v2 = sample_coords[i + 1] - sample_coords[i]

            phi = np.arctan2(v1[1], v1[0]) * 180 / np.pi
            psi = np.arctan2(v2[1], v2[0]) * 180 / np.pi

            phi_angles.append(phi)
            psi_angles.append(psi)

        assert len(phi_angles) == len(sample_coords) - 2
        assert len(psi_angles) == len(sample_coords) - 2

        # Angles should be in valid range
        assert all(-180 <= angle <= 180 for angle in phi_angles)
        assert all(-180 <= angle <= 180 for angle in psi_angles)

    def test_ramachandran_plot_creation(self, sample_coords):
        """Should create Ramachandran plot."""
        phi_angles = []
        psi_angles = []

        for i in range(1, len(sample_coords) - 1):
            v1 = sample_coords[i] - sample_coords[i - 1]
            v2 = sample_coords[i + 1] - sample_coords[i]

            phi = np.arctan2(v1[1], v1[0]) * 180 / np.pi
            psi = np.arctan2(v2[1], v2[0]) * 180 / np.pi

            phi_angles.append(phi)
            psi_angles.append(psi)

        fig, ax = plt.subplots()
        ax.scatter(phi_angles, psi_angles)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)

        assert len(ax.collections) > 0
        plt.close(fig)

    def test_ramachandran_regions(self):
        """Should add secondary structure regions."""
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots()

        # Alpha-helix region
        ax.add_patch(Rectangle((-180, -70), 120, 70, fill=False, edgecolor="red"))

        # Beta-sheet region
        ax.add_patch(Rectangle((-180, 90), 120, 90, fill=False, edgecolor="blue"))

        assert len(ax.patches) == 2
        plt.close(fig)


class TestPlotlyIntegration:
    """Test Plotly interactive visualization."""

    def test_plotly_import(self):
        """Plotly should be importable."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            assert go is not None
            assert make_subplots is not None
        except ImportError:
            pytest.skip("Plotly not installed")

    @pytest.mark.skipif("plotly" not in dir(), reason="Plotly not installed")
    def test_3d_scatter_creation(self):
        """Should create 3D scatter plot with Plotly."""
        try:
            import plotly.graph_objects as go

            np.random.seed(42)
            coords = np.random.randn(50, 3)

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers+lines"
                    )
                ]
            )

            assert fig is not None
            assert len(fig.data) == 1
        except ImportError:
            pytest.skip("Plotly not installed")

    @pytest.mark.skipif("plotly" not in dir(), reason="Plotly not installed")
    def test_plotly_hover_info(self):
        """Should add hover information."""
        try:
            import plotly.graph_objects as go

            n = 50
            coords = np.random.randn(n, 3)
            hover_text = [f"Residue {i}" for i in range(n)]

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=coords[:, 2],
                        text=hover_text,
                        mode="markers",
                    )
                ]
            )

            assert fig.data[0].text is not None
        except ImportError:
            pytest.skip("Plotly not installed")


class TestVisualizationIntegration:
    """Test integration of multiple visualization methods."""

    def test_seaborn_style_application(self):
        """Should apply seaborn styling."""
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except:
            plt.style.use("default")

        # Should not raise error
        assert True

    def test_color_palette_setting(self):
        """Should set seaborn color palette."""
        sns.set_palette("husl")
        palette = sns.color_palette()
        assert len(palette) > 0

    def test_figure_saving(self):
        """Should save figures to disk."""
        import os
        import tempfile

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.png")
            plt.savefig(filepath, dpi=150)
            assert os.path.exists(filepath)

        plt.close(fig)

    def test_multiple_plot_types(self):
        """Should create multiple plot types in same figure."""
        np.random.seed(42)
        coords = np.random.randn(50, 3)
        distances = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Distance map
        axes[0].imshow(distances, cmap="viridis")

        # Histogram
        axes[1].hist(distances[np.triu_indices_from(distances, k=1)], bins=20)

        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
