"""Production notebook smoke tests."""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def test_production_notebook_exists():
    assert (Path(__file__).parent.parent / "examples" / "complete_production_run.ipynb").exists()


def test_production_notebook_has_structure():
    path = Path(__file__).parent.parent / "examples" / "complete_production_run.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    markdown = [c for c in nb.cells if c.cell_type == "markdown"]
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert markdown
    assert code


def test_production_notebook_executes_smoke():
    path = Path(__file__).parent.parent / "examples" / "complete_production_run.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=120, kernel_name="python3", allow_errors=False)
    ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})
    assert any(getattr(c, "outputs", []) for c in nb.cells if c.cell_type == "code")
