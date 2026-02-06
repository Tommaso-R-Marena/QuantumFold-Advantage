"""Production notebook validation with broad capability coverage."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK_PATH = Path(__file__).parent.parent / "examples" / "complete_production_run.ipynb"


def _source(cell) -> str:
    src = getattr(cell, "source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def _combined_source(nb) -> str:
    return "\n".join(_source(c) for c in nb.cells)


def _find_cell(nb, *patterns):
    for cell in nb.cells:
        source = _source(cell)
        if all(p in source for p in patterns):
            return cell
    return None


def _load_notebook():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


class TestNotebookStructure:
    def test_notebook_exists(self):
        assert NOTEBOOK_PATH.exists()

    def test_has_markdown_and_code_cells(self):
        nb = _load_notebook()
        assert any(c.cell_type == "markdown" for c in nb.cells)
        assert any(c.cell_type == "code" for c in nb.cells)

    def test_colab_badge_in_intro(self):
        nb = _load_notebook()
        first_markdown = next(c for c in nb.cells if c.cell_type == "markdown")
        src = _source(first_markdown)
        assert "colab-badge.svg" in src
        assert "complete_production_run.ipynb" in src

    def test_key_sections_appear_in_expected_order(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        markers = [
            "## 📊 Step 1: Data Preparation",
            "## 🧬 Step 2: Generate ESM-2 Embeddings",
            "## ⚛️ Step 3: Train Quantum-Enhanced Model",
            "## 🔬 Step 4: Train Classical Baseline",
            "## 📈 Step 5: Comprehensive Evaluation",
            "## 📊 Step 6: Statistical Validation",
            "## 📈 Step 7: Generate Publication Figures",
            "## 💾 Step 8: Export Results",
        ]
        positions = [combined.find(m) for m in markers if combined.find(m) != -1]
        assert positions, "Expected at least one section marker"
        assert positions == sorted(positions)


class TestConfigurationAndSetup:
    def test_environment_check_and_hardware_paths(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        assert "torch.cuda.is_available()" in combined
        assert "A100" in combined
        assert "psutil" in combined

    def test_configuration_parameters_present(self):
        nb = _load_notebook()
        config_cell = _find_cell(nb, "Configuration")
        assert config_cell is not None
        src = _source(config_cell)
        for param in [
            "NUM_TRAINING_PROTEINS",
            "NUM_EPOCHS_QUANTUM",
            "NUM_EPOCHS_CLASSICAL",
            "BATCH_SIZE",
            "ESM_MODEL",
            "HIDDEN_DIM",
            "NUM_QUBITS",
            "USE_REDUCED_CONFIG",
        ]:
            assert param in src

    def test_reduced_config_logic_present(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        assert "USE_REDUCED_CONFIG" in combined
        assert "T4" in combined or "Free Tier" in combined


class TestDataTrainingEvaluationFlow:
    def test_data_processing_capabilities(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "generate_diverse_protein_dataset",
            "train_ids",
            "val_ids",
            "test_ids",
            "data_split.json",
            "ESM2Embedder",
            "embedding_cache",
        ]:
            assert token in combined

    def test_model_initialization_capabilities(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "quantum_model = AdvancedProteinFoldingModel",
            "classical_model = AdvancedProteinFoldingModel",
            "use_quantum=True",
            "use_quantum=False",
            "quantum_model.cpu()",
        ]:
            assert token in combined

    def test_training_capabilities(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "AdvancedTrainer",
            "use_amp",
            "use_ema",
            "gradient_clip",
            ".train(",
            "num_epochs",
            "save_freq",
        ]:
            assert token in combined

    def test_evaluation_and_stats_capabilities(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "def evaluate_model",
            "compute_tm_score",
            "compute_rmsd",
            "compute_gdt_ts",
            "ComprehensiveBenchmark",
            "compare_methods",
            "wilcoxon_p",
            "ttest_p",
            "TEST SET RESULTS",
        ]:
            assert token in combined


class TestVisualizationAndExport:
    def test_visualization_capabilities(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "Training Loss Curves",
            "plt.subplots",
            "plt.savefig",
            "violinplot",
            "metric_distributions.png",
            "Paired Comparison",
            "paired_comparison.png",
        ]:
            assert token in combined

    def test_export_and_archive_capabilities(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "to_csv",
            "raw_results.csv",
            "RESULTS_SUMMARY.json",
            "json.dump",
            "make_archive",
            "zip",
            "files.download",
        ]:
            assert token in combined


class TestReliabilityAndReproducibility:
    def test_memory_and_error_handling_capabilities(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "torch.cuda.empty_cache()",
            "embedding_cache",
            "torch.cuda.is_available()",
            "drive.mount",
            "MOUNT_DRIVE",
        ]:
            assert token in combined

    def test_reproducibility_and_docs_capabilities(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "seed",
            "experiment_config.json",
            "Benchmark Complete",
            "Next Steps",
            "Citation",
            "minutes",
            "hours",
            "tqdm",
        ]:
            assert token in combined


class TestProductionExecution:
    def test_notebook_executes_smoke(self):
        nb = _load_notebook()
        ep = ExecutePreprocessor(timeout=180, kernel_name="python3", allow_errors=False)
        ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOK_PATH.parent)}})
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert code_cells
        assert any(getattr(c, "outputs", []) for c in code_cells)
