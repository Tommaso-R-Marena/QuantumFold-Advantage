"""Notebook execution smoke tests."""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS_DIR = Path(__file__).parent.parent / "examples"
NOTEBOOKS = [
    "colab_quickstart.ipynb",
    "01_getting_started.ipynb",
    "02_quantum_vs_classical.ipynb",
    "03_advanced_visualization.ipynb",
]


def _run_notebook(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=120, kernel_name="python3", allow_errors=False)
    ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})
    return nb


def test_notebook_files_exist():
    for name in NOTEBOOKS:
        assert (NOTEBOOKS_DIR / name).exists()


def test_notebooks_execute_without_errors():
    for name in NOTEBOOKS:
        nb = _run_notebook(NOTEBOOKS_DIR / name)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert code_cells
        assert any(getattr(c, "outputs", []) for c in code_cells)
