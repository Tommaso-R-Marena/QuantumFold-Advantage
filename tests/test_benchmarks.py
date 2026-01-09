"""Unit tests for benchmarking utilities."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks import BenchmarkComparison, ProteinStructureEvaluator, StructureMetrics


class TestProteinStructureEvaluator(unittest.TestCase):
    """Test structure evaluation metrics."""

    def setUp(self):
        self.evaluator = ProteinStructureEvaluator()

        # Create sample coordinates
        np.random.seed(42)
        self.n_residues = 50
        self.coords_true = np.random.randn(self.n_residues, 3) * 10

        # Perfect prediction
        self.coords_perfect = self.coords_true.copy()

        # Noisy prediction
        self.coords_noisy = self.coords_true + np.random.randn(self.n_residues, 3) * 2.0

        # Very bad prediction
        self.coords_bad = np.random.randn(self.n_residues, 3) * 20

    def test_rmsd_perfect(self):
        """Test RMSD for perfect prediction."""
        rmsd = self.evaluator.calculate_rmsd(self.coords_perfect, self.coords_true)
        self.assertAlmostEqual(rmsd, 0.0, places=5)

    def test_rmsd_noisy(self):
        """Test RMSD for noisy prediction."""
        rmsd = self.evaluator.calculate_rmsd(self.coords_noisy, self.coords_true)
        self.assertGreater(rmsd, 0.0)
        self.assertLess(rmsd, 5.0)  # Should be reasonable

    def test_rmsd_symmetry(self):
        """Test RMSD is symmetric."""
        rmsd1 = self.evaluator.calculate_rmsd(self.coords_noisy, self.coords_true, align=False)
        rmsd2 = self.evaluator.calculate_rmsd(self.coords_true, self.coords_noisy, align=False)
        self.assertAlmostEqual(rmsd1, rmsd2, places=5)

    def test_tm_score_range(self):
        """Test TM-score is in valid range."""
        tm_perfect = self.evaluator.calculate_tm_score(self.coords_perfect, self.coords_true)
        tm_noisy = self.evaluator.calculate_tm_score(self.coords_noisy, self.coords_true)
        tm_bad = self.evaluator.calculate_tm_score(self.coords_bad, self.coords_true)

        # Check range
        self.assertGreaterEqual(tm_perfect, 0.0)
        self.assertLessEqual(tm_perfect, 1.0)
        self.assertGreaterEqual(tm_noisy, 0.0)
        self.assertLessEqual(tm_noisy, 1.0)
        self.assertGreaterEqual(tm_bad, 0.0)
        self.assertLessEqual(tm_bad, 1.0)

        # Check ordering
        self.assertGreater(tm_perfect, tm_noisy)
        self.assertGreater(tm_noisy, tm_bad)

    def test_gdt_scores(self):
        """Test GDT_TS and GDT_HA scores."""
        gdt_ts, gdt_ha = self.evaluator.calculate_gdt(self.coords_perfect, self.coords_true)

        # Perfect prediction should have 100% GDT
        self.assertAlmostEqual(gdt_ts, 100.0, places=0)
        self.assertAlmostEqual(gdt_ha, 100.0, places=0)

        # Noisy prediction
        gdt_ts_noisy, gdt_ha_noisy = self.evaluator.calculate_gdt(
            self.coords_noisy, self.coords_true
        )
        self.assertLess(gdt_ts_noisy, 100.0)
        self.assertGreater(gdt_ts_noisy, 0.0)

    def test_lddt_score(self):
        """Test lDDT calculation."""
        lddt_perfect = self.evaluator.calculate_lddt(self.coords_perfect, self.coords_true)
        lddt_noisy = self.evaluator.calculate_lddt(self.coords_noisy, self.coords_true)
        lddt_bad = self.evaluator.calculate_lddt(self.coords_bad, self.coords_true)

        # Check range [0, 1]
        self.assertGreaterEqual(lddt_perfect, 0.0)
        self.assertLessEqual(lddt_perfect, 1.0)

        # Perfect should have high lDDT
        self.assertGreater(lddt_perfect, 0.95)

        # Check ordering
        self.assertGreater(lddt_perfect, lddt_noisy)
        self.assertGreater(lddt_noisy, lddt_bad)

    def test_clash_score(self):
        """Test clash score calculation."""
        # Perfect structure should have few/no clashes
        clash_perfect = self.evaluator.calculate_clash_score(self.coords_true)
        self.assertGreaterEqual(clash_perfect, 0.0)

        # Create structure with obvious clashes
        coords_clash = np.zeros((10, 3))
        coords_clash[:, 0] = np.arange(10) * 0.5  # Very close atoms
        clash_bad = self.evaluator.calculate_clash_score(coords_clash)
        self.assertGreater(clash_bad, clash_perfect)

    def test_evaluate_structure(self):
        """Test comprehensive structure evaluation."""
        metrics = self.evaluator.evaluate_structure(
            self.coords_noisy, self.coords_true, sequence_length=self.n_residues
        )

        self.assertIsInstance(metrics, StructureMetrics)
        self.assertIsInstance(metrics.rmsd, float)
        self.assertIsInstance(metrics.tm_score, float)
        self.assertIsInstance(metrics.gdt_ts, float)
        self.assertIsInstance(metrics.gdt_ha, float)
        self.assertIsInstance(metrics.lddt, float)
        self.assertIsInstance(metrics.clash_score, float)


class TestBenchmarkComparison(unittest.TestCase):
    """Test benchmark comparison functionality."""

    def setUp(self):
        self.benchmark = BenchmarkComparison(output_dir="test_outputs")

        np.random.seed(42)
        self.n_residues = 50
        self.coords_true = np.random.randn(self.n_residues, 3) * 10
        self.coords_pred1 = self.coords_true + np.random.randn(self.n_residues, 3) * 1.0
        self.coords_pred2 = self.coords_true + np.random.randn(self.n_residues, 3) * 2.0

    def test_compare_predictions(self):
        """Test comparing multiple predictions."""
        comparison = self.benchmark.compare_predictions(
            protein_id="test_protein",
            coords_true=self.coords_true,
            coords_quantumfold=self.coords_pred1,
            coords_alphafold=self.coords_pred2,
            sequence_length=self.n_residues,
        )

        self.assertIn("QuantumFold", comparison)
        self.assertIn("AlphaFold-3", comparison)
        self.assertIsInstance(comparison["QuantumFold"], StructureMetrics)

    def test_results_storage(self):
        """Test that results are stored correctly."""
        initial_count = len(self.benchmark.results)

        self.benchmark.compare_predictions(
            protein_id="test_protein_1",
            coords_true=self.coords_true,
            coords_quantumfold=self.coords_pred1,
        )

        self.assertEqual(len(self.benchmark.results), initial_count + 1)
        self.assertEqual(self.benchmark.results[-1]["protein_id"], "test_protein_1")

    def tearDown(self):
        """Clean up test outputs."""
        import shutil

        output_dir = Path("test_outputs")
        if output_dir.exists():
            shutil.rmtree(output_dir)


class TestAlignmentFunction(unittest.TestCase):
    """Test coordinate alignment (Kabsch algorithm)."""

    def setUp(self):
        self.evaluator = ProteinStructureEvaluator()
        np.random.seed(42)
        self.coords = np.random.randn(20, 3) * 5

    def test_alignment_rotation(self):
        """Test alignment handles rotation."""
        # Create rotated version
        angle = np.pi / 4
        R = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        coords_rotated = self.coords @ R.T

        # Align
        coords_aligned = self.evaluator._align_structures(coords_rotated, self.coords)

        # Should be close after alignment
        rmsd = np.sqrt(np.mean(np.sum((coords_aligned - self.coords) ** 2, axis=1)))
        self.assertLess(rmsd, 1e-6)

    def test_alignment_translation(self):
        """Test alignment handles translation."""
        # Create translated version
        coords_translated = self.coords + np.array([10, 20, 30])

        # Align
        coords_aligned = self.evaluator._align_structures(coords_translated, self.coords)

        # Should be close after alignment
        rmsd = np.sqrt(np.mean(np.sum((coords_aligned - self.coords) ** 2, axis=1)))
        self.assertLess(rmsd, 1e-6)


if __name__ == "__main__":
    unittest.main()
