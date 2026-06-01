from .core import NotebookNode

def new_notebook():
    return NotebookNode(cells=[], metadata={}, nbformat=4, nbformat_minor=5)

def new_code_cell(source=""):
    return NotebookNode(cell_type="code", source=source, metadata={}, outputs=[], execution_count=None)
