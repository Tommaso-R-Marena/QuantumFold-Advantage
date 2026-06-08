from .core import NotebookNode

def new_code_cell(source=''):
    return NotebookNode(cell_type='code', source=source, metadata={}, outputs=[], execution_count=None)

def new_notebook(cells=None, metadata=None):
    return NotebookNode(cells=cells or [], metadata=metadata or {}, nbformat=4, nbformat_minor=5)
