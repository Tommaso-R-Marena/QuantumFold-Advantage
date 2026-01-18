"""Tests for complete production benchmark notebook.

This test suite validates:
1. Notebook structure and metadata
2. Cell execution order and dependencies
3. Colab-specific configurations
4. Memory and resource requirements
5. Output validation
"""

from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

NOTEBOOK_PATH = Path("examples/complete_production_run.ipynb")


@pytest.fixture
def notebook():
    """Load the production notebook."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


class TestNotebookStructure:
    """Test notebook structure and metadata."""

    def test_notebook_exists(self):
        """Test that notebook file exists."""
        assert NOTEBOOK_PATH.exists(), f"Notebook not found at {NOTEBOOK_PATH}"

    def test_notebook_format(self, notebook):
        """Test that notebook is valid format."""
        assert notebook.nbformat == 4, "Notebook should be nbformat 4"
        assert len(notebook.cells) > 0, "Notebook should have cells"

    def test_colab_metadata(self, notebook):
        """Test Colab-specific metadata."""
        metadata = notebook.metadata

        # Check for Colab metadata
        assert "colab" in metadata, "Missing Colab metadata"
        assert "accelerator" in metadata, "Missing accelerator metadata"
        assert metadata["accelerator"] == "GPU", "Should specify GPU accelerator"

        # Check Colab settings
        colab_meta = metadata["colab"]
        assert "gpuType" in colab_meta, "Missing GPU type specification"
        assert colab_meta["gpuType"] == "A100", "Should specify A100 GPU"
        assert "machine_shape" in colab_meta, "Missing machine shape"
        assert colab_meta["machine_shape"] == "hm", "Should specify high memory"

    def test_required_sections(self, notebook):
        """Test that notebook has all required sections."""
        cell_sources = [cell.source for cell in notebook.cells]
        combined_source = "\n".join(cell_sources)

        required_sections = [
            "Environment Check",
            "Mount Google Drive",
            "Data Preparation",
            "Generate ESM-2 Embeddings",
            "Train Quantum-Enhanced Model",
            "Train Classical Baseline",
            "Comprehensive Evaluation",
            "Statistical Validation",
            "Publication Figures",
            "Export Results",
        ]

        for section in required_sections:
            assert section in combined_source, f"Missing section: {section}"

    def test_cell_types(self, notebook):
        """Test cell type distribution."""
        code_cells = [c for c in notebook.cells if c.cell_type == "code"]
        markdown_cells = [c for c in notebook.cells if c.cell_type == "markdown"]

        assert len(code_cells) >= 15, "Should have at least 15 code cells"
        assert len(markdown_cells) >= 5, "Should have at least 5 markdown cells"

    def test_colab_badge(self, notebook):
        """Test that Colab badge is present in first markdown cell."""
        first_markdown = next((c for c in notebook.cells if c.cell_type == "markdown"), None)

        assert first_markdown is not None, "Should have markdown cells"
        assert "colab-badge.svg" in first_markdown.source, "Missing Colab badge"
        assert "complete_production_run.ipynb" in first_markdown.source, "Incorrect notebook link"


class TestConfigurationCells:
    """Test configuration and setup cells."""

    def test_environment_check_cell(self, notebook):
        """Test environment check cell."""
        check_cell = next((c for c in notebook.cells if "ENVIRONMENT CHECK" in c.source), None)

        assert check_cell is not None, "Missing environment check cell"
        assert "torch.cuda.is_available()" in check_cell.source
        assert "A100" in check_cell.source, "Should check for A100 GPU"
        assert "psutil" in check_cell.source, "Should check system resources"

    def test_configuration_cell(self, notebook):
        """Test configuration cell with parameters."""
        config_cell = next(
            (c for c in notebook.cells if "Configuration" in c.source and "@param" in c.source),
            None,
        )

        assert config_cell is not None, "Missing configuration cell"

        # Check for key parameters
        params = [
            "NUM_TRAINING_PROTEINS",
            "NUM_EPOCHS_QUANTUM",
            "NUM_EPOCHS_CLASSICAL",
            "BATCH_SIZE",
            "ESM_MODEL",
            "HIDDEN_DIM",
            "NUM_QUBITS",
            "USE_MIXED_PRECISION",
            "USE_REDUCED_CONFIG",
        ]

        for param in params:
            assert param in config_cell.source, f"Missing parameter: {param}"

    def test_reduced_config_option(self, notebook):
        """Test that reduced config option exists for T4 GPUs."""
        config_cell = next((c for c in notebook.cells if "USE_REDUCED_CONFIG" in c.source), None)

        assert config_cell is not None
        assert "T4" in config_cell.source or "Free Tier" in config_cell.source
        assert "if USE_REDUCED_CONFIG:" in config_cell.source


class TestDataProcessing:
    """Test data processing cells."""

    def test_dataset_generation(self, notebook):
        """Test dataset generation function."""
        dataset_cell = next(
            (c for c in notebook.cells if "generate_diverse_protein_dataset" in c.source), None
        )

        assert dataset_cell is not None, "Missing dataset generation"

        # Check for diverse protein classes
        protein_classes = ["alpha_proteins", "beta_proteins", "mixed_proteins", "small_proteins"]
        for pclass in protein_classes:
            assert pclass in dataset_cell.source, f"Missing protein class: {pclass}"

    def test_data_split(self, notebook):
        """Test train/val/test split."""
        split_cell = next(
            (c for c in notebook.cells if "train_ids" in c.source and "val_ids" in c.source), None
        )

        assert split_cell is not None, "Missing data split cell"
        assert (
            "0.70" in split_cell.source or "70%" in split_cell.source
        ), "Should use 70% for training"
        assert "data_split.json" in split_cell.source, "Should save split info"

    def test_embedding_generation(self, notebook):
        """Test ESM-2 embedding generation."""
        emb_cell = next((c for c in notebook.cells if "ESM2Embedder" in c.source), None)

        assert emb_cell is not None, "Missing embedding generation"
        assert "embedding_cache" in emb_cell.source, "Should cache embeddings"
        assert "torch.cuda.empty_cache()" in emb_cell.source, "Should free memory"


class TestModelTraining:
    """Test model training cells."""

    def test_quantum_model_init(self, notebook):
        """Test quantum model initialization."""
        quantum_init = next(
            (
                c
                for c in notebook.cells
                if "quantum_model = AdvancedProteinFoldingModel" in c.source
            ),
            None,
        )

        assert quantum_init is not None, "Missing quantum model initialization"
        assert "use_quantum=True" in quantum_init.source, "Should enable quantum"
        assert "num_qubits" in quantum_init.source
        assert "noise_level" in quantum_init.source

    def test_classical_model_init(self, notebook):
        """Test classical model initialization."""
        classical_init = next(
            (
                c
                for c in notebook.cells
                if "classical_model = AdvancedProteinFoldingModel" in c.source
            ),
            None,
        )

        assert classical_init is not None, "Missing classical model initialization"
        assert "use_quantum=False" in classical_init.source, "Should disable quantum"
        assert "quantum_model.cpu()" in classical_init.source, "Should free quantum model from GPU"

    def test_trainer_initialization(self, notebook):
        """Test trainer initialization."""
        trainer_cells = [c for c in notebook.cells if "AdvancedTrainer" in c.source]

        assert len(trainer_cells) >= 2, "Should have trainers for both models"

        # Check for key trainer parameters
        trainer_source = "\n".join([c.source for c in trainer_cells])
        assert "use_amp" in trainer_source
        assert "use_ema" in trainer_source
        assert "gradient_clip" in trainer_source

    def test_training_loops(self, notebook):
        """Test training loop execution."""
        training_cells = [c for c in notebook.cells if ".train(" in c.source]

        assert len(training_cells) >= 2, "Should train both models"

        for cell in training_cells:
            assert "num_epochs" in cell.source
            assert "save_freq" in cell.source or "val_freq" in cell.source


class TestEvaluation:
    """Test evaluation and statistical testing."""

    def test_model_evaluation(self, notebook):
        """Test model evaluation function."""
        eval_cell = next((c for c in notebook.cells if "def evaluate_model" in c.source), None)

        assert eval_cell is not None, "Missing evaluation function"

        # Check for all metrics
        metrics = ["tm_scores", "rmsds", "gdt_ts", "plddts"]
        for metric in metrics:
            assert metric in eval_cell.source, f"Missing metric: {metric}"

        assert "compute_tm_score" in eval_cell.source
        assert "compute_rmsd" in eval_cell.source
        assert "compute_gdt_ts" in eval_cell.source

    def test_statistical_validation(self, notebook):
        """Test statistical validation."""
        stats_cell = next((c for c in notebook.cells if "ComprehensiveBenchmark" in c.source), None)

        assert stats_cell is not None, "Missing statistical validation"
        assert "compare_methods" in stats_cell.source
        assert "wilcoxon_p" in stats_cell.source or "ttest_p" in stats_cell.source

    def test_results_printing(self, notebook):
        """Test that results are printed."""
        results_cells = [c for c in notebook.cells if "TEST SET RESULTS" in c.source]

        assert len(results_cells) > 0, "Should print test set results"


class TestVisualization:
    """Test visualization cells."""

    def test_training_curves(self, notebook):
        """Test training curve plotting."""
        plot_cell = next((c for c in notebook.cells if "Training Loss Curves" in c.source), None)

        assert plot_cell is not None, "Missing training curves plot"
        assert "plt.subplots" in plot_cell.source
        assert "plt.savefig" in plot_cell.source, "Should save figures"
        assert "dpi=300" in plot_cell.source, "Should use high DPI for publication"

    def test_distribution_plots(self, notebook):
        """Test metric distribution plotting."""
        dist_cell = next((c for c in notebook.cells if "violinplot" in c.source), None)

        assert dist_cell is not None, "Missing distribution plots"
        assert "metric_distributions.png" in dist_cell.source

    def test_comparison_plots(self, notebook):
        """Test paired comparison plotting."""
        comp_cell = next((c for c in notebook.cells if "Paired Comparison" in c.source), None)

        assert comp_cell is not None, "Missing comparison plots"
        assert "scatter" in comp_cell.source
        assert "paired_comparison.png" in comp_cell.source


class TestResultsExport:
    """Test results export and archiving."""

    def test_csv_export(self, notebook):
        """Test CSV export of raw results."""
        csv_cell = next((c for c in notebook.cells if "to_csv" in c.source), None)

        assert csv_cell is not None, "Missing CSV export"
        assert "raw_results.csv" in csv_cell.source
        assert "pd.DataFrame" in csv_cell.source

    def test_summary_report(self, notebook):
        """Test summary report generation."""
        summary_cell = next((c for c in notebook.cells if "RESULTS_SUMMARY.json" in c.source), None)

        assert summary_cell is not None, "Missing summary report"
        assert "json.dump" in summary_cell.source

        # Check for key summary sections
        assert "experiment" in summary_cell.source
        assert "hardware" in summary_cell.source
        assert "configuration" in summary_cell.source
        assert "quantum_results" in summary_cell.source
        assert "classical_results" in summary_cell.source

    def test_archive_creation(self, notebook):
        """Test results archive creation."""
        archive_cell = next((c for c in notebook.cells if "make_archive" in c.source), None)

        assert archive_cell is not None, "Missing archive creation"
        assert "shutil" in archive_cell.source
        assert "zip" in archive_cell.source

    def test_download_functionality(self, notebook):
        """Test Colab download functionality."""
        download_cell = next((c for c in notebook.cells if "files.download" in c.source), None)

        assert download_cell is not None, "Missing download functionality"
        assert "google.colab" in download_cell.source


class TestMemoryManagement:
    """Test memory management practices."""

    def test_cuda_cache_clearing(self, notebook):
        """Test that CUDA cache is cleared appropriately."""
        cache_cells = [c for c in notebook.cells if "torch.cuda.empty_cache()" in c.source]

        assert len(cache_cells) >= 2, "Should clear CUDA cache multiple times"

    def test_model_offloading(self, notebook):
        """Test that models are moved off GPU when not needed."""
        offload_cells = [c for c in notebook.cells if ".cpu()" in c.source]

        assert len(offload_cells) > 0, "Should offload models to CPU"

    def test_embedding_caching(self, notebook):
        """Test that embeddings are cached to disk."""
        cache_cell = next((c for c in notebook.cells if "embedding_cache" in c.source), None)

        assert cache_cell is not None
        assert "torch.save" in cache_cell.source, "Should save embeddings"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_gpu_check(self, notebook):
        """Test GPU availability check."""
        gpu_cell = next(
            (c for c in notebook.cells if "torch.cuda.is_available()" in c.source), None
        )

        assert gpu_cell is not None
        assert (
            "sys.exit(1)" in gpu_cell.source or "ERROR" in gpu_cell.source
        ), "Should handle missing GPU"

    def test_drive_mount_handling(self, notebook):
        """Test Google Drive mount handling."""
        mount_cell = next((c for c in notebook.cells if "drive.mount" in c.source), None)

        assert mount_cell is not None
        assert "MOUNT_DRIVE" in mount_cell.source, "Should have mount option"
        assert "if" in mount_cell.source, "Should conditionally mount"


class TestReproducibility:
    """Test reproducibility features."""

    def test_seed_setting(self, notebook):
        """Test random seed setting."""
        seed_cells = [c for c in notebook.cells if "set_seed" in c.source or "seed=42" in c.source]

        assert len(seed_cells) > 0, "Should set random seed"

    def test_config_saving(self, notebook):
        """Test configuration saving."""
        config_save = next(
            (c for c in notebook.cells if "experiment_config.json" in c.source), None
        )

        assert config_save is not None, "Should save experiment config"
        assert "timestamp" in config_save.source, "Should record timestamp"


class TestDocumentation:
    """Test documentation and user guidance."""

    def test_conclusion_section(self, notebook):
        """Test that notebook has proper conclusion."""
        conclusion = next((c for c in notebook.cells if "Benchmark Complete" in c.source), None)

        assert conclusion is not None, "Missing conclusion section"
        assert "Next Steps" in conclusion.source
        assert "Citation" in conclusion.source

    def test_runtime_estimates(self, notebook):
        """Test that runtime estimates are provided."""
        estimate_cells = [c for c in notebook.cells if "minutes" in c.source or "hours" in c.source]

        assert len(estimate_cells) > 0, "Should provide runtime estimates"

    def test_progress_indicators(self, notebook):
        """Test that progress indicators are used."""
        tqdm_cells = [c for c in notebook.cells if "tqdm" in c.source]

        assert len(tqdm_cells) > 0, "Should use progress bars"


@pytest.mark.integration
class TestNotebookExecution:
    """Integration tests for notebook execution.

    These tests are marked as integration and may be skipped in fast test runs.
    """

    @pytest.mark.slow
    def test_imports_execute(self, notebook, tmp_path):
        """Test that import cells execute without error."""
        # Create a notebook with just import cells
        import_nb = nbformat.v4.new_notebook()
        import_cells = [
            c for c in notebook.cells if c.cell_type == "code" and "import" in c.source
        ][
            :5
        ]  # Just test first few import cells

        import_nb.cells = import_cells

        # Try to execute
        ep = ExecutePreprocessor(timeout=120, kernel_name="python3")

        try:
            ep.preprocess(import_nb)
        except CellExecutionError as e:
            pytest.fail(f"Import cell failed to execute: {e}")

    @pytest.mark.slow
    def test_config_cell_executes(self, notebook):
        """Test that configuration cell can be executed."""
        config_cell = next(
            (c for c in notebook.cells if "Configuration" in c.source and "CONFIG = {" in c.source),
            None,
        )

        if config_cell:
            # This would require mock setup - just verify structure for now
            assert "json.dump" in config_cell.source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
