import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def _run(source, allow_errors=False):
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(source))
    ep = ExecutePreprocessor(allow_errors=allow_errors, kernel_name='python3')
    return ep.preprocess(nb, {'metadata': {'path': '.'}})[0]

def test_placeholder():
    # Placeholder for unit tests while debugging CI
    pass
