class CellExecutionError(Exception):
    pass


class ExecutePreprocessor:
    def __init__(self, timeout=600, kernel_name='python3', allow_errors=False):
        self.timeout = timeout
        self.kernel_name = kernel_name
        self.allow_errors = allow_errors

    def preprocess(self, nb, resources=None):
        import ast
        import contextlib
        import io
        import traceback

        def _cell_get(cell, key, default=None):
            if isinstance(cell, dict):
                return cell.get(key, default)
            return getattr(cell, key, default)

        def _cell_set(cell, key, value):
            if isinstance(cell, dict):
                cell[key] = value
            else:
                setattr(cell, key, value)

        exec_count = 1
        cells = nb.get('cells', []) if isinstance(nb, dict) else getattr(nb, 'cells', [])
        exec_globals = {"__name__": "__main__"}

        for cell in cells:
            if _cell_get(cell, 'cell_type', None) != 'code':
                continue

            source = _cell_get(cell, 'source', '') or ''
            if isinstance(source, list):
                source = ''.join(source)
            source = source.strip()

            if source:
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                outputs = []

                try:
                    parsed = ast.parse(source, mode='exec')
                    tail_expr = None
                    body = parsed.body
                    if body and isinstance(body[-1], ast.Expr):
                        tail_expr = ast.Expression(body.pop().value)

                    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                        stderr_buffer
                    ):
                        if body:
                            exec(compile(ast.Module(body=body, type_ignores=[]), '<cell>', 'exec'), exec_globals)
                        if tail_expr is not None:
                            expr_value = eval(compile(tail_expr, '<cell>', 'eval'), exec_globals)
                        else:
                            expr_value = None
                except Exception as e:
                    tb = traceback.format_exc()
                    outputs.append(
                        {
                            'output_type': 'error',
                            'ename': e.__class__.__name__,
                            'evalue': str(e),
                            'traceback': tb.splitlines(),
                        }
                    )
                    _cell_set(cell, 'outputs', outputs)
                    _cell_set(cell, 'execution_count', exec_count)
                    if not self.allow_errors:
                        raise CellExecutionError(str(e)) from e
                else:
                    stdout = stdout_buffer.getvalue()
                    stderr = stderr_buffer.getvalue()
                    if stdout:
                        outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': stdout})
                    if stderr:
                        outputs.append({'output_type': 'stream', 'name': 'stderr', 'text': stderr})
                    if tail_expr is not None and expr_value is not None:
                        outputs.append(
                            {
                                'output_type': 'execute_result',
                                'data': {'text/plain': repr(expr_value)},
                                'execution_count': exec_count,
                                'metadata': {},
                            }
                        )
                    _cell_set(cell, 'outputs', outputs)
                    _cell_set(cell, 'execution_count', exec_count)
                exec_count += 1
            else:
                _cell_set(cell, 'outputs', [])
                _cell_set(cell, 'execution_count', None)
        return nb, resources or {}
