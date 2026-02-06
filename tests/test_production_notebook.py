"""Production notebook validation with broad capability and flow coverage."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK_PATH = Path(__file__).parent.parent / "examples" / "complete_production_run.ipynb"


def _source(cell) -> str:
    src = cell.get("source", "") if isinstance(cell, dict) else getattr(cell, "source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def _cells(nb):
    return nb.get("cells", []) if isinstance(nb, dict) else getattr(nb, "cells", [])


def _metadata(nb):
    return nb.get("metadata", {}) if isinstance(nb, dict) else getattr(nb, "metadata", {})


def _combined_source(nb) -> str:
    return "\n".join(_source(c) for c in _cells(nb))


def _find_cell(nb, *patterns):
    for cell in _cells(nb):
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
        assert any(
            (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None))
            == "markdown"
            for c in _cells(nb)
        )
        assert any(
            (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "code"
            for c in _cells(nb)
        )

    def test_colab_badge_in_intro(self):
        nb = _load_notebook()
        first_markdown = next(
            c
            for c in _cells(nb)
            if (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None))
            == "markdown"
        )
        src = _source(first_markdown)
        assert "colab-badge.svg" in src
        assert "complete_production_run.ipynb" in src

    def test_notebook_metadata_has_runtime_context(self):
        nb = _load_notebook()
        metadata = _metadata(nb)
        assert metadata
        if isinstance(metadata, dict) and "kernelspec" in metadata:
            assert "name" in metadata["kernelspec"]

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
            "## ✅ Benchmark Complete!",
        ]
        positions = [combined.find(m) for m in markers if combined.find(m) != -1]
        assert positions, "Expected at least one section marker"
        assert positions == sorted(positions)


class TestConfigurationAndSetup:
    def test_environment_check_and_hardware_paths(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in ["torch.cuda.is_available()", "A100", "psutil"]:
            assert token in combined

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

    def test_saved_artifact_names_are_declared(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "training_curves.png",
            "metric_distributions.png",
            "paired_comparison.png",
            "raw_results.csv",
            "RESULTS_SUMMARY.json",
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
        code_cells = [
            c
            for c in _cells(nb)
            if (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None))
            == "code"
        ]
        assert code_cells
        outputs = [
            c.get("outputs", []) if isinstance(c, dict) else getattr(c, "outputs", [])
            for c in code_cells
        ]
        assert any(output_list for output_list in outputs)


class TestProductionNotebookLegacyParity:
    """Restore broad legacy-style assertions over production notebook capabilities."""

    def test_environment_and_mount_workflow_tokens(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in ["torch.cuda.is_available()", "drive.mount", "MOUNT_DRIVE", "psutil"]:
            assert token in combined

    def test_training_both_models_is_described(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "quantum_model = AdvancedProteinFoldingModel",
            "classical_model = AdvancedProteinFoldingModel",
            "use_quantum=True",
            "use_quantum=False",
            ".train(",
        ]:
            assert token in combined

    def test_statistical_outputs_and_comparisons_present(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "ComprehensiveBenchmark",
            "compare_methods",
            "wilcoxon_p",
            "ttest_p",
            "TEST SET RESULTS",
        ]:
            assert token in combined

    def test_visual_outputs_and_saved_files_present(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "Training Loss Curves",
            "metric_distributions.png",
            "paired_comparison.png",
            "training_curves.png",
        ]:
            assert token in combined

    def test_final_documentation_sections_present(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in ["Benchmark Complete", "Next Steps", "Citation", "minutes", "hours"]:
            assert token in combined


class TestProductionNotebookCompleteLegacySurface:
    """Additional restored legacy-style checks for the full production workflow."""

    def test_required_section_markers_present(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for section in [
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
        ]:
            assert section in combined

    def test_configuration_and_training_hyperparams_present(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "NUM_TRAINING_PROTEINS",
            "NUM_EPOCHS_QUANTUM",
            "NUM_EPOCHS_CLASSICAL",
            "BATCH_SIZE",
            "ESM_MODEL",
            "HIDDEN_DIM",
            "NUM_QUBITS",
            "num_epochs",
            "save_freq",
            "val_freq",
        ]:
            assert token in combined

    def test_training_pipeline_components_present(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "generate_diverse_protein_dataset",
            "ESM2Embedder",
            "embedding_cache",
            "AdvancedProteinFoldingModel",
            "AdvancedTrainer",
            "use_amp",
            "use_ema",
            "gradient_clip",
        ]:
            assert token in combined

    def test_evaluation_statistical_pipeline_present(self):
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

    def test_export_reproducibility_and_docs_pipeline_present(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "raw_results.csv",
            "RESULTS_SUMMARY.json",
            "json.dump",
            "make_archive",
            "files.download",
            "experiment_config.json",
            "Benchmark Complete",
            "Next Steps",
            "Citation",
            "minutes",
            "hours",
        ]:
            assert token in combined


class TestProductionNotebookFlowIntegrity:
    """Additional flow-integrity checks for full production lifecycle."""

    def test_data_split_and_cache_lifecycle_tokens_present(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in ["train_ids", "val_ids", "test_ids", "data_split.json", "embedding_cache"]:
            assert token in combined

    def test_model_comparison_and_reporting_tokens_present(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "quantum_model = AdvancedProteinFoldingModel",
            "classical_model = AdvancedProteinFoldingModel",
            "TEST SET RESULTS",
            "paired_comparison.png",
            "RESULTS_SUMMARY.json",
        ]:
            assert token in combined

    def test_runtime_and_hardware_documentation_tokens_present(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in ["A100", "T4", "minutes", "hours", "torch.cuda.is_available()"]:
            assert token in combined


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestProductionNotebookHelperFunctions:
    """Unit checks for production notebook helper utilities."""

    def test_source_and_cells_handle_dict_and_object_inputs(self):
        assert _source({"source": ["a", "b"]}) == "ab"
        assert _source(_Obj(source="text")) == "text"
        assert len(_cells({"cells": [{"cell_type": "code"}]})) == 1
        assert len(_cells(_Obj(cells=[_Obj(cell_type="markdown")]))) == 1

    def test_metadata_and_combined_source_work_for_dict_notebook(self):
        nb = {
            "metadata": {"kernelspec": {"name": "python3"}},
            "cells": [
                {"cell_type": "markdown", "source": "# Title"},
                {"cell_type": "code", "source": "print('x')"},
            ],
        }
        assert _metadata(nb)["kernelspec"]["name"] == "python3"
        combined = _combined_source(nb)
        assert "# Title" in combined and "print('x')" in combined

    def test_find_cell_returns_match_and_none_when_missing(self):
        nb = {
            "cells": [
                {"cell_type": "code", "source": "alpha beta"},
                {"cell_type": "code", "source": "gamma delta"},
            ]
        }
        assert _find_cell(nb, "alpha", "beta") is not None
        assert _find_cell(nb, "nope") is None


class TestProductionNotebookAdditionalInvariants:
    """Extra invariants for production notebook comprehensiveness."""

    def test_contains_all_expected_output_artifact_extensions(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for ext in [".png", ".csv", ".json"]:
            assert ext in combined
        assert "zip" in combined

    def test_contains_both_quantum_and_classical_paths(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in ["use_quantum=True", "use_quantum=False", "quantum_model", "classical_model"]:
            assert token in combined

    def test_contains_reproducibility_and_cleanup_tokens(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "seed",
            "experiment_config.json",
            "torch.cuda.empty_cache()",
            "embedding_cache",
        ]:
            assert token in combined

    def test_contains_statistical_and_visual_reporting_tokens(self):
        nb = _load_notebook()
        combined = _combined_source(nb)
        for token in [
            "wilcoxon_p",
            "ttest_p",
            "metric_distributions.png",
            "paired_comparison.png",
            "TEST SET RESULTS",
        ]:
            assert token in combined
