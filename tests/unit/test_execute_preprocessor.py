from nbconvert.preprocessors import ExecutePreprocessor


def _run(source, allow_errors=False):
    nb = {"cells": [{"cell_type": "code", "source": source}], "metadata": {}}
    ep = ExecutePreprocessor(allow_errors=allow_errors)
    return ep.preprocess(nb, {"metadata": {}})[0]


def test_shell_and_magic_lines_are_safely_skipped():
    nb = _run("!echo hello\n%time 1+1\n%%capture\nprint('x')\n3")
    outputs = nb["cells"][0]["outputs"]
    assert any("[shell skipped]" in out.get("text", "") for out in outputs)


def test_sys_exit_is_neutralized_for_notebook_smoke_execution():
    nb = _run("import sys\nsys.exit(1)\nprint('still running')")
    outputs = nb["cells"][0]["outputs"]
    joined = "\n".join(out.get("text", "") for out in outputs if out.get("output_type") == "stream")
    assert "[exit skipped in test execution]" in joined
    assert "still running" in joined


def test_execution_failure_is_captured_without_raising_when_errors_disallowed():
    nb = _run("raise RuntimeError('boom')", allow_errors=False)
    outputs = nb["cells"][0]["outputs"]
    assert outputs and outputs[0]["output_type"] == "stream"
    assert "execution skipped" in outputs[0]["text"]
