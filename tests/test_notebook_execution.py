"""End-to-end notebook execution and structural validation tests."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS_DIR = Path(__file__).parent.parent / "examples"
EXECUTION_TARGETS = [
    "colab_quickstart.ipynb",
    "01_getting_started.ipynb",
    "02_quantum_vs_classical.ipynb",
    "03_advanced_visualization.ipynb",
]
TIMEOUT = 600


class NotebookExecutionError(Exception):
    """Raised when notebook execution fails."""


def _source(cell) -> str:
    src = cell.get("source", "") if isinstance(cell, dict) else getattr(cell, "source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def _output_type(output) -> str:
    if isinstance(output, dict):
        return output.get("output_type", "")
    return getattr(output, "output_type", "")


def _cells(nb):
    return nb.get("cells", []) if isinstance(nb, dict) else getattr(nb, "cells", [])


def _read_notebook(notebook_path: Path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


def execute_notebook(notebook_path: Path, timeout: int = TIMEOUT):
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    nb = _read_notebook(notebook_path)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3", allow_errors=False)
    try:
        ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
    except Exception as exc:  # pragma: no cover - exercised in failure mode
        raise NotebookExecutionError(f"Failed to execute {notebook_path.name}: {exc}") from exc
    return nb


def notebook_stats(nb) -> dict:
    stats = {
        "total_cells": len(_cells(nb)),
        "code_cells": 0,
        "markdown_cells": 0,
        "cells_with_output": 0,
        "cells_with_errors": 0,
        "executed_code_cells": 0,
    }
    for cell in _cells(nb):
        cell_type = cell.get("cell_type") if isinstance(cell, dict) else getattr(cell, "cell_type", None)
        if cell_type == "code":
            stats["code_cells"] += 1
            outputs = cell.get("outputs", []) if isinstance(cell, dict) else getattr(cell, "outputs", [])
            outputs = outputs or []
            if outputs:
                stats["cells_with_output"] += 1
            if any(_output_type(out) == "error" for out in outputs):
                stats["cells_with_errors"] += 1
            execution_count = (
                cell.get("execution_count") if isinstance(cell, dict) else getattr(cell, "execution_count", None)
            )
            if execution_count is not None:
                stats["executed_code_cells"] += 1
        elif cell_type == "markdown":
            stats["markdown_cells"] += 1
    return stats


class TestNotebookExecution:
    @pytest.mark.parametrize("name", EXECUTION_TARGETS)
    def test_notebook_file_exists(self, name):
        assert (NOTEBOOKS_DIR / name).exists(), f"Missing notebook: {name}"

    def test_execute_notebook_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            execute_notebook(NOTEBOOKS_DIR / "missing_notebook.ipynb", timeout=30)

    @pytest.mark.parametrize("name", EXECUTION_TARGETS)
    def test_notebook_has_expected_kernel_metadata(self, name):
        nb = _read_notebook(NOTEBOOKS_DIR / name)
        metadata = nb.get("metadata", {}) if isinstance(nb, dict) else getattr(nb, "metadata", {})
        kernelspec = metadata.get("kernelspec", {}) if isinstance(metadata, dict) else {}
        if kernelspec:
            assert "name" in kernelspec

    @pytest.mark.slow
    @pytest.mark.parametrize("name", EXECUTION_TARGETS)
    def test_notebook_executes_without_errors(self, name):
        nb = execute_notebook(NOTEBOOKS_DIR / name, timeout=120 if "quickstart" in name else 300)
        stats = notebook_stats(nb)
        assert stats["code_cells"] > 0
        assert stats["cells_with_errors"] == 0
        assert stats["cells_with_output"] > 0
        assert stats["executed_code_cells"] > 0

    @pytest.mark.slow
    @pytest.mark.parametrize("name", EXECUTION_TARGETS)
    def test_executed_notebook_has_monotonic_execution_counts(self, name):
        nb = execute_notebook(NOTEBOOKS_DIR / name, timeout=120 if "quickstart" in name else 300)
        execution_counts = []
        for cell in _cells(nb):
            cell_type = cell.get("cell_type") if isinstance(cell, dict) else getattr(cell, "cell_type", None)
            if cell_type != "code":
                continue
            execution_count = (
                cell.get("execution_count") if isinstance(cell, dict) else getattr(cell, "execution_count", None)
            )
            if execution_count is not None:
                execution_counts.append(execution_count)

        assert execution_counts, "Expected at least one executed code cell"
        assert execution_counts == sorted(execution_counts), "Execution counts should be monotonic"


class TestNotebookContentCoverage:
    @pytest.mark.parametrize("name", EXECUTION_TARGETS)
    def test_has_markdown_and_code_cells(self, name):
        nb = _read_notebook(NOTEBOOKS_DIR / name)
        assert any((c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "markdown" for c in _cells(nb))
        assert any((c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "code" for c in _cells(nb))

    @pytest.mark.parametrize("name", EXECUTION_TARGETS)
    def test_each_notebook_has_colab_link(self, name):
        nb = _read_notebook(NOTEBOOKS_DIR / name)
        markdown = "\n".join(
            _source(c)
            for c in _cells(nb)
            if (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "markdown"
        )
        assert "colab.research.google.com" in markdown
        assert "colab-badge.svg" in markdown

    def test_key_workflow_topics_are_present(self):
        combined = ""
        for name in EXECUTION_TARGETS:
            nb = _read_notebook(NOTEBOOKS_DIR / name)
            combined += "\n".join(_source(c) for c in _cells(nb))

        for required in ["torch", "numpy", "matplotlib", "RMSD", "TM-score"]:
            assert required in combined

    def test_training_and_benchmarking_capabilities_are_covered(self):
        combined = ""
        for name in [
            "01_getting_started.ipynb",
            "02_quantum_vs_classical.ipynb",
            "complete_production_run.ipynb",
        ]:
            nb = _read_notebook(NOTEBOOKS_DIR / name)
            combined += "\n" + "\n".join(_source(c) for c in _cells(nb))

        for required in [
            "AdvancedProteinFoldingModel",
            "AdvancedTrainer",
            "ComprehensiveBenchmark",
            "compare_methods",
            "compute_tm_score",
            "compute_rmsd",
        ]:
            assert required in combined

    def test_visualization_capabilities_are_covered(self):
        combined = ""
        for name in ["03_advanced_visualization.ipynb", "complete_production_run.ipynb"]:
            nb = _read_notebook(NOTEBOOKS_DIR / name)
            combined += "\n" + "\n".join(_source(c) for c in _cells(nb))

        for required in ["seaborn", "scatter", "plt.savefig", "violinplot"]:
            assert required in combined

    def test_colab_notebook_has_colab_badge(self):
        nb = _read_notebook(NOTEBOOKS_DIR / "colab_quickstart.ipynb")
        first_markdown = next(
            c
            for c in _cells(nb)
            if (c.get("cell_type") if isinstance(c, dict) else getattr(c, "cell_type", None)) == "markdown"
        )
        source = _source(first_markdown)
        assert "colab-badge.svg" in source
        assert "colab.research.google.com" in source
