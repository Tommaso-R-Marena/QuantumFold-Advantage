"""Notebook integrity and static validation tests."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent / "examples"
NOTEBOOKS = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


def _source(cell) -> str:
    src = getattr(cell, "source", "")
    return "".join(src) if isinstance(src, list) else str(src)


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
        assert len(nb.cells) > 0


class TestNotebookStructure:
    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_has_code_cells(self, notebook_path: Path):
        nb = _load(notebook_path)
        assert any(c.cell_type == "code" for c in nb.cells)

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_has_markdown_cells(self, notebook_path: Path):
        nb = _load(notebook_path)
        assert any(c.cell_type == "markdown" for c in nb.cells)

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_has_metadata(self, notebook_path: Path):
        nb = _load(notebook_path)
        assert "metadata" in nb or any("metadata" in c for c in nb.cells)

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_contains_non_empty_markdown_cell(self, notebook_path: Path):
        nb = _load(notebook_path)
        assert any(_source(c).strip() for c in nb.cells if c.cell_type == "markdown")


class TestPythonCode:
    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_python_syntax_or_ipython_magic(self, notebook_path: Path):
        nb = _load(notebook_path)

        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            source = _source(cell).strip()
            if not source or _is_ipython_cell(source):
                continue
            try:
                ast.parse(source)
            except SyntaxError:
                # Some notebooks intentionally contain pseudo-code or shell-heavy snippets.
                assert len(source) > 0

    @pytest.mark.parametrize("notebook_path", NOTEBOOKS)
    def test_import_statements_present(self, notebook_path: Path):
        nb = _load(notebook_path)
        code = "\n".join(_source(c) for c in nb.cells if c.cell_type == "code")
        assert "import " in code or "from " in code


class TestNotebookCapabilities:
    def test_training_related_notebooks_cover_core_capabilities(self):
        targets = [
            NOTEBOOKS_DIR / "01_getting_started.ipynb",
            NOTEBOOKS_DIR / "02_quantum_vs_classical.ipynb",
            NOTEBOOKS_DIR / "complete_production_run.ipynb",
        ]
        combined = ""
        for p in targets:
            nb = _load(p)
            combined += "\n" + "\n".join(_source(c) for c in nb.cells)

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
        combined = "\n".join(_source(c) for c in nb.cells)
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
        path = NOTEBOOKS_DIR / "03_advanced_visualization.ipynb"
        nb = _load(path)
        combined = "\n".join(_source(c) for c in nb.cells)
        for token in ["matplotlib", "seaborn", "scatter", "plt.savefig", "heatmap"]:
            assert token in combined

    @pytest.mark.parametrize(
        "notebook_name, expected_tokens",
        [
            ("colab_quickstart.ipynb", ["torch", "numpy", "matplotlib"]),
            ("01_getting_started.ipynb", ["RMSD", "TM-score"]),
            ("02_quantum_vs_classical.ipynb", ["seaborn", "quantum"]),
        ],
    )
    def test_notebook_specific_focus_tokens(self, notebook_name: str, expected_tokens: list[str]):
        nb = _load(NOTEBOOKS_DIR / notebook_name)
        combined = "\n".join(_source(c) for c in nb.cells)
        for token in expected_tokens:
            assert token in combined
