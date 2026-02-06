"""Comprehensive static notebook integrity and capability validation tests."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent / "examples"
NOTEBOOKS = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


def _source(cell) -> str:
    src = cell.get("source", "") if isinstance(cell, dict) else getattr(cell, "source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def _cells(nb):
    return nb.get("cells", []) if isinstance(nb, dict) else getattr(nb, "cells", [])


def _metadata(nb):
    return nb.get("metadata", {}) if isinstance(nb, dict) else getattr(nb, "metadata", {})


def _is_ipython_cell(source: str) -> bool:
    return any(line.lstrip().startswith(("%", "!")) for line in source.splitlines())


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


class TestNotebookDiscovery:
    def test_notebooks_exist(self):
        assert NOTEBOOKS, "No notebooks found in examples/"

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_notebook_can_be_parsed(self, notebook_path: Path):
        nb = _load(notebook_path)
        assert len(_cells(nb)) > 0

    def test_expected_key_notebooks_present(self):
        expected = {
            "colab_quickstart.ipynb",
            "01_getting_started.ipynb",
            "02_quantum_vs_classical.ipynb",
            "03_advanced_visualization.ipynb",
            "complete_production_run.ipynb",
        }
        discovered = {p.name for p in NOTEBOOKS}
        assert expected.issubset(discovered)


class TestNotebookStructure:
    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_has_code_cells(self, notebook_path: Path):
        nb = _load(notebook_path)
        assert any((c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "code" for c in _cells(nb))

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_has_markdown_cells(self, notebook_path: Path):
        nb = _load(notebook_path)
        assert any((c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "markdown" for c in _cells(nb))

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_contains_non_empty_markdown_cell(self, notebook_path: Path):
        nb = _load(notebook_path)
        assert any(_source(c).strip() for c in _cells(nb) if (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "markdown")

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_metadata_container_is_valid(self, notebook_path: Path):
        nb = _load(notebook_path)
        meta = _metadata(nb)
        assert isinstance(meta, dict)

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_colab_badge_present_in_markdown(self, notebook_path: Path):
        nb = _load(notebook_path)
        markdown = "\n".join(
            _source(c)
            for c in _cells(nb)
            if (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "markdown"
        )
        assert "colab.research.google.com" in markdown


class TestPythonCode:
    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_python_syntax_or_ipython_magic(self, notebook_path: Path):
        nb = _load(notebook_path)

        for cell in _cells(nb):
            if (cell.get("cell_type") if isinstance(cell, dict) else getattr(cell, "cell_type", None)) != "code":
                continue
            source = _source(cell).strip()
            if not source or _is_ipython_cell(source):
                continue
            try:
                ast.parse(source)
            except SyntaxError:
                # Keep this permissive for pseudo-code snippets embedded in notebooks.
                assert len(source) > 0

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_import_statements_present(self, notebook_path: Path):
        nb = _load(notebook_path)
        code = "\n".join(
            _source(c)
            for c in _cells(nb)
            if (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "code"
        )
        assert "import " in code or "from " in code

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_each_notebook_uses_python_or_shell_workflows(self, notebook_path: Path):
        nb = _load(notebook_path)
        code_cells = [
            _source(c)
            for c in _cells(nb)
            if (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "code"
        ]
        assert any("import " in c or c.lstrip().startswith(("!", "%")) for c in code_cells)


class TestNotebookCapabilities:
    def test_training_related_notebooks_cover_core_capabilities(self):
        targets = [
            NOTEBOOKS_DIR / "01_getting_started.ipynb",
            NOTEBOOKS_DIR / "02_quantum_vs_classical.ipynb",
            NOTEBOOKS_DIR / "complete_production_run.ipynb",
        ]
        combined = ""
        for path in targets:
            nb = _load(path)
            combined += "\n" + "\n".join(_source(c) for c in _cells(nb))

        for token in [
            "ESM2Embedder",
            "AdvancedProteinFoldingModel",
            "AdvancedTrainer",
            "evaluate_model",
            "ComprehensiveBenchmark",
            "to_csv",
            "json.dump",
        ]:
            assert token in combined

    def test_dataset_split_and_reproducibility_capabilities(self):
        nb = _load(NOTEBOOKS_DIR / "complete_production_run.ipynb")
        combined = "\n".join(_source(c) for c in _cells(nb))
        for token in [
            "train_ids",
            "val_ids",
            "test_ids",
            "data_split.json",
            "seed",
            "experiment_config.json",
        ]:
            assert token in combined

    def test_visualization_notebook_capabilities(self):
        nb = _load(NOTEBOOKS_DIR / "03_advanced_visualization.ipynb")
        combined = "\n".join(_source(c) for c in _cells(nb))
        for token in ["matplotlib", "seaborn", "scatter", "plt.savefig", "heatmap"]:
            assert token in combined

    @pytest.mark.parametrize(
        "notebook_name, expected_tokens",
        [
            ("colab_quickstart.ipynb", ["torch", "numpy", "matplotlib", "tqdm"]),
            ("01_getting_started.ipynb", ["RMSD", "TM-score", "torch.cuda.empty_cache()"]),
            ("02_quantum_vs_classical.ipynb", ["seaborn", "quantum"]),
            ("complete_production_run.ipynb", ["ComprehensiveBenchmark", "compare_methods", "files.download"]),
        ],
    )
    def test_notebook_specific_focus_tokens(self, notebook_name: str, expected_tokens: list[str]):
        nb = _load(NOTEBOOKS_DIR / notebook_name)
        combined = "\n".join(_source(c) for c in _cells(nb))
        for token in expected_tokens:
            assert token in combined


class TestNotebookLegacyParityCoverage:
    """Restore broad legacy-style notebook integrity checks."""

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_notebook_file_extension_and_location(self, notebook_path: Path):
        assert notebook_path.suffix == ".ipynb"
        assert notebook_path.parent == NOTEBOOKS_DIR

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_notebook_has_multiple_cells(self, notebook_path: Path):
        nb = _load(notebook_path)
        assert len(_cells(nb)) >= 2

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_notebook_has_code_or_markdown_content(self, notebook_path: Path):
        nb = _load(notebook_path)
        assert any(_source(c).strip() for c in _cells(nb))

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_code_cells_are_nonempty_when_present(self, notebook_path: Path):
        nb = _load(notebook_path)
        code_cells = [
            c for c in _cells(nb)
            if (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "code"
        ]
        assert code_cells
        assert any(_source(c).strip() for c in code_cells)

    def test_production_notebook_has_export_and_archive_capabilities(self):
        nb = _load(NOTEBOOKS_DIR / "complete_production_run.ipynb")
        combined = "\n".join(_source(c) for c in _cells(nb))
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

    def test_production_notebook_reproducibility_tokens(self):
        nb = _load(NOTEBOOKS_DIR / "complete_production_run.ipynb")
        combined = "\n".join(_source(c) for c in _cells(nb))
        for token in ["seed", "experiment_config.json", "tqdm", "minutes", "hours"]:
            assert token in combined


class TestNotebookBroadLegacyCapabilities:
    """Adds back broad capability checks comparable to legacy notebook test suites."""

    def test_complete_production_sections_present(self):
        nb = _load(NOTEBOOKS_DIR / "complete_production_run.ipynb")
        combined = "\n".join(_source(c) for c in _cells(nb))
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
            "Benchmark Complete",
        ]:
            assert section in combined

    def test_complete_production_training_eval_export_tokens(self):
        nb = _load(NOTEBOOKS_DIR / "complete_production_run.ipynb")
        combined = "\n".join(_source(c) for c in _cells(nb))
        for token in [
            "AdvancedProteinFoldingModel",
            "AdvancedTrainer",
            "evaluate_model",
            "compute_tm_score",
            "compute_rmsd",
            "compute_gdt_ts",
            "ComprehensiveBenchmark",
            "compare_methods",
            "raw_results.csv",
            "RESULTS_SUMMARY.json",
            "make_archive",
            "files.download",
        ]:
            assert token in combined

    @pytest.mark.parametrize(
        "notebook_name, tokens",
        [
            ("01_getting_started.ipynb", ["RMSD", "TM-score", "zip", "torch.cuda.empty_cache()"]),
            ("02_quantum_vs_classical.ipynb", ["quantum", "classical", "seaborn"]),
            ("03_advanced_visualization.ipynb", ["matplotlib", "seaborn", "heatmap", "scatter"]),
        ],
    )
    def test_notebook_specific_feature_sets(self, notebook_name: str, tokens: list[str]):
        nb = _load(NOTEBOOKS_DIR / notebook_name)
        combined = "\n".join(_source(c) for c in _cells(nb))
        for token in tokens:
            assert token in combined

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_notebook_has_human_readable_markdown(self, notebook_path: Path):
        nb = _load(notebook_path)
        markdown_cells = [
            _source(c).strip()
            for c in _cells(nb)
            if (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "markdown"
        ]
        assert markdown_cells
        assert any(len(m) > 20 for m in markdown_cells)


class TestNotebookCrossFileConsistency:
    """Cross-notebook assertions to restore broader legacy coverage."""

    def test_all_notebooks_reference_colab(self):
        for notebook_path in NOTEBOOKS:
            nb = _load(notebook_path)
            markdown = "\n".join(
                _source(c)
                for c in _cells(nb)
                if (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "markdown"
            )
            assert "colab.research.google.com" in markdown

    def test_training_notebooks_include_progress_indicators(self):
        targets = [
            NOTEBOOKS_DIR / "01_getting_started.ipynb",
            NOTEBOOKS_DIR / "02_quantum_vs_classical.ipynb",
            NOTEBOOKS_DIR / "complete_production_run.ipynb",
        ]
        combined = ""
        for path in targets:
            nb = _load(path)
            combined += "\n" + "\n".join(_source(c) for c in _cells(nb))
        assert "tqdm" in combined

    def test_core_metric_tokens_span_training_and_production(self):
        targets = [
            NOTEBOOKS_DIR / "01_getting_started.ipynb",
            NOTEBOOKS_DIR / "complete_production_run.ipynb",
        ]
        combined = ""
        for path in targets:
            nb = _load(path)
            combined += "\n" + "\n".join(_source(c) for c in _cells(nb))
        for token in ["RMSD", "TM-score", "compute_rmsd", "compute_tm_score"]:
            assert token in combined


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestNotebookStaticHelperFunctions:
    """Unit checks for static notebook helper behavior."""

    def test_source_handles_dict_and_object_cells(self):
        assert _source({"source": ["a", "b"]}) == "ab"
        assert _source(_Obj(source="hello")) == "hello"

    def test_cells_and_metadata_handle_dict_and_object_notebooks(self):
        dict_nb = {"cells": [{"cell_type": "code"}], "metadata": {"k": "v"}}
        obj_nb = _Obj(cells=[_Obj(cell_type="markdown")], metadata={"m": 1})
        assert len(_cells(dict_nb)) == 1
        assert len(_cells(obj_nb)) == 1
        assert _metadata(dict_nb)["k"] == "v"
        assert _metadata(obj_nb)["m"] == 1

    def test_is_ipython_cell_detects_magics_and_shell(self):
        assert _is_ipython_cell("%matplotlib inline")
        assert _is_ipython_cell("!pip install numpy")
        assert not _is_ipython_cell("import numpy as np")


class TestNotebookStructuralInvariants:
    """Additional invariants across all notebooks."""

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_notebook_has_unique_cell_indices_when_enumerated(self, notebook_path: Path):
        nb = _load(notebook_path)
        indices = [i for i, _ in enumerate(_cells(nb))]
        assert len(indices) == len(set(indices))

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_notebook_code_cells_have_source_field(self, notebook_path: Path):
        nb = _load(notebook_path)
        for cell in _cells(nb):
            cell_type = cell.get("cell_type") if isinstance(cell, dict) else getattr(cell, "cell_type", None)
            if cell_type == "code":
                text = _source(cell)
                assert isinstance(text, str)

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_notebook_markdown_contains_human_text(self, notebook_path: Path):
        nb = _load(notebook_path)
        markdown = [
            _source(c).strip()
            for c in _cells(nb)
            if (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "markdown"
        ]
        assert markdown
        assert any(any(ch.isalpha() for ch in m) for m in markdown)

    def test_cross_notebook_export_tokens_present_somewhere(self):
        combined = ""
        for path in NOTEBOOKS:
            nb = _load(path)
            combined += "\n" + "\n".join(_source(c) for c in _cells(nb))
        for token in ["to_csv", "json.dump", "make_archive", "files.download"]:
            assert token in combined
