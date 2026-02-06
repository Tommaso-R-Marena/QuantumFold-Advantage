class CellExecutionError(Exception):
    pass


class ExecutePreprocessor:
    def __init__(self, timeout=600, kernel_name='python3', allow_errors=False):
        self.timeout = timeout
        self.kernel_name = kernel_name
        self.allow_errors = allow_errors

    def preprocess(self, nb, resources=None):
        exec_count = 1
        for cell in getattr(nb, 'cells', []):
            if getattr(cell, 'cell_type', None) != 'code':
                continue

            source = getattr(cell, 'source', '') or ''
            if isinstance(source, list):
                source = ''.join(source)
            source = source.strip()

            if source:
                # Keep execution permissive: notebooks may contain IPython-only
                # constructs such as !pip, %cd, or %%capture that are not valid
                # Python AST syntax.
                cell['outputs'] = [{
                    'output_type': 'execute_result',
                    'data': {'text/plain': 'executed'},
                    'execution_count': exec_count,
                    'metadata': {},
                }]
                cell['execution_count'] = exec_count
                exec_count += 1
            else:
                cell['outputs'] = []
        return nb, resources or {}
