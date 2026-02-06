"""General notebook integrity tests."""

from pathlib import Path

import nbformat

NOTEBOOKS = list((Path(__file__).parent.parent / "examples").glob("*.ipynb"))


def _cell_source(cell):
    source = getattr(cell, "source", "")
    if isinstance(source, list):
        return "".join(source)
    return source


def test_notebook_collection_not_empty():
    assert NOTEBOOKS


def test_all_notebooks_parse():
    for path in NOTEBOOKS:
        with open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        assert len(nb.cells) > 0


def test_code_cells_have_text_content():
    for path in NOTEBOOKS:
        with open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert code_cells
        assert all(isinstance(_cell_source(c), str) for c in code_cells)
