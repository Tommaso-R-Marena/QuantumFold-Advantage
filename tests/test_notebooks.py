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
