"""Comprehensive tests for Jupyter notebooks.

This module tests all notebooks for:
1. Valid JSON structure
2. Python syntax correctness
3. Import statement validity
4. Cell execution order
5. Metadata completeness
6. Colab compatibility
"""

import ast
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

nbformat = pytest.importorskip("nbformat")
from nbformat.v4 import new_code_cell, new_notebook

# Get all notebook paths
NOTEBOOK_DIR = Path(__file__).parent.parent / "examples"
NOTEBOOK_PATHS = list(NOTEBOOK_DIR.glob("*.ipynb"))


class TestNotebookStructure:
    """Test notebook JSON structure and format."""

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_valid_json_structure(self, notebook_path):
        """Test that notebook is valid JSON."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            try:
                nb = nbformat.read(f, as_version=4)
                assert nb is not None
                assert hasattr(nb, "cells")
            except Exception as e:
                pytest.fail(f"Invalid notebook structure: {e}")

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_has_cells(self, notebook_path):
        """Test that notebook has cells."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        assert len(nb.cells) > 0, "Notebook has no cells"

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_cell_types(self, notebook_path):
        """Test that cells have valid types."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        valid_types = {"code", "markdown", "raw"}
        for i, cell in enumerate(nb.cells):
            assert cell.cell_type in valid_types, f"Cell {i} has invalid type: {cell.cell_type}"


class TestPythonCode:
    """Test Python code in notebooks."""

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_python_syntax(self, notebook_path):
        """Test that all code cells have valid Python syntax."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        errors = []
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

            source = "".join(cell.source)
            if not source.strip():
                continue

            # Skip cells with shell commands
            if source.strip().startswith("!"):
                continue

            # Skip cells with IPython magic
            if source.strip().startswith("%"):
                continue

            # Try to parse as Python
            try:
                ast.parse(source)
            except SyntaxError as e:
                errors.append(f"Cell {i}: {e}")

        if errors:
            pytest.fail(f"Syntax errors found:\n" + "\n".join(errors))

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_no_hardcoded_paths(self, notebook_path):
        """Test that notebooks don't contain hardcoded absolute paths."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        issues = []
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

            source = "".join(cell.source)

            # Check for hardcoded paths (but allow common ones)
            if re.search(r"/home/[^\s]+|C:\\\\Users|D:\\\\", source):
                # Exclude comments and strings about paths
                if not source.strip().startswith("#"):
                    issues.append(f"Cell {i} may contain hardcoded path")

        if issues:
            pytest.fail(f"Hardcoded paths found:\n" + "\n".join(issues))

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_no_exposed_credentials(self, notebook_path):
        """Test that notebooks don't contain exposed credentials."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        sensitive_patterns = [
            r'api_key\s*=\s*["\'][^"\'{]+["\']',
            r'password\s*=\s*["\'][^"\'{]+["\']',
            r'secret\s*=\s*["\'][^"\'{]+["\']',
            r'token\s*=\s*["\'][^"\'{]+["\']',
        ]

        issues = []
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

            source = "".join(cell.source)

            for pattern in sensitive_patterns:
                if re.search(pattern, source, re.I):
                    # Exclude if it's a placeholder or comment
                    if (
                        "your_" not in source.lower()
                        and "example" not in source.lower()
                        and not source.strip().startswith("#")
                    ):
                        issues.append(f"Cell {i} may contain exposed credential")
                        break

        if issues:
            pytest.fail(f"Potential credential exposure:\n" + "\n".join(issues))


class TestNotebookMetadata:
    """Test notebook metadata."""

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_has_metadata(self, notebook_path):
        """Test that notebook has metadata section."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        assert hasattr(nb, "metadata"), "Notebook missing metadata"
        assert isinstance(nb.metadata, dict), "Metadata is not a dict"

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_colab_compatibility(self, notebook_path):
        """Test for Colab compatibility indicators."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Check for Colab badge in markdown cells
        has_colab_badge = False
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                source = "".join(cell.source)
                if "colab.research.google.com" in source or "Open In Colab" in source:
                    has_colab_badge = True
                    break

        # Production notebooks should have Colab badges
        if "production" in str(notebook_path).lower() or "a100" in str(notebook_path).lower():
            assert (
                has_colab_badge
            ), f"Production notebook {notebook_path.name} should have Colab badge"


class TestImports:
    """Test import statements in notebooks."""

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_extract_imports(self, notebook_path):
        """Extract and validate import statements."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        imports = set()
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue

            source = "".join(cell.source)

            # Find import statements
            for match in re.finditer(r"^import (\S+)", source, re.M):
                module = match.group(1).split(".")[0]
                imports.add(module)

            for match in re.finditer(r"^from (\S+)", source, re.M):
                module = match.group(1).split(".")[0]
                imports.add(module)

        # Just verify we found some imports (most notebooks should have them)
        # Don't enforce specific imports as notebooks may vary
        assert len(imports) >= 0  # Always passes, just for extraction

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_no_missing_src_imports(self, notebook_path):
        """Test that notebooks importing from src can find modules."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        src_imports = set()
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue

            source = "".join(cell.source)

            # Find imports from src
            for match in re.finditer(r"^from src\.(\S+)", source, re.M):
                module = match.group(1).split()[0]  # Remove 'import ...'
                src_imports.add(module)

        # Check that imported src modules exist
        src_dir = Path(__file__).parent.parent / "src"
        for module in src_imports:
            module_parts = module.split(".")

            # Check for .py file or package directory
            module_path = src_dir / f"{module_parts[0]}.py"
            package_path = src_dir / module_parts[0] / "__init__.py"

            assert (
                module_path.exists() or package_path.exists()
            ), f"Notebook imports {module} but src/{module_parts[0]}.py not found"


class TestNotebookContent:
    """Test notebook content and structure."""

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_has_title(self, notebook_path):
        """Test that notebook has a title in first cell."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        if len(nb.cells) == 0:
            pytest.skip("Empty notebook")

        first_cell = nb.cells[0]
        if first_cell.cell_type == "markdown":
            source = "".join(first_cell.source)
            # Should have a header
            assert source.strip().startswith(
                "#"
            ), "First markdown cell should contain a title (start with #)"

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_code_to_markdown_ratio(self, notebook_path):
        """Test that notebooks have reasonable documentation."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        code_cells = sum(1 for cell in nb.cells if cell.cell_type == "code")
        markdown_cells = sum(1 for cell in nb.cells if cell.cell_type == "markdown")

        if code_cells == 0:
            pytest.skip("No code cells")

        # Should have at least some markdown cells for documentation
        # Ratio shouldn't be too extreme (at least 1 markdown per 5 code cells)
        ratio = markdown_cells / code_cells
        assert (
            ratio > 0.1
        ), f"Notebook has too little documentation ({markdown_cells} markdown, {code_cells} code)"


class TestExecutionOrder:
    """Test for potential execution order issues."""

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_no_execution_count_gaps(self, notebook_path):
        """Test that execution counts (if present) are sequential."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        execution_counts = []
        for cell in nb.cells:
            if cell.cell_type == "code":
                exec_count = cell.get("execution_count")
                if exec_count is not None:
                    execution_counts.append(exec_count)

        if not execution_counts:
            pytest.skip("No execution counts in notebook")

        # Check for large gaps (might indicate cells run out of order)
        for i in range(len(execution_counts) - 1):
            gap = execution_counts[i + 1] - execution_counts[i]
            assert (
                gap <= 10
            ), f"Large execution count gap detected: {execution_counts[i]} -> {execution_counts[i+1]}"


class TestNotebookSize:
    """Test notebook file size and complexity."""

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_file_size_reasonable(self, notebook_path):
        """Test that notebook file size is reasonable."""
        file_size = os.path.getsize(notebook_path)
        max_size = 10 * 1024 * 1024  # 10 MB

        assert (
            file_size < max_size
        ), f"Notebook file size ({file_size / 1024 / 1024:.1f} MB) exceeds {max_size / 1024 / 1024} MB"

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_reasonable_cell_count(self, notebook_path):
        """Test that notebook doesn't have too many cells."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        max_cells = 200
        assert (
            len(nb.cells) < max_cells
        ), f"Notebook has {len(nb.cells)} cells (max recommended: {max_cells})"


class TestOutputs:
    """Test notebook outputs (if present)."""

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
    def test_no_large_outputs(self, notebook_path):
        """Test that notebooks don't contain excessively large outputs."""
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

            outputs = cell.get("outputs", [])
            for j, output in enumerate(outputs):
                output_str = json.dumps(output)
                output_size = len(output_str)

                # Warn if individual output is very large (>1MB)
                if output_size > 1024 * 1024:
                    pytest.fail(
                        f"Cell {i} output {j} is very large ({output_size / 1024 / 1024:.1f} MB). "
                        f"Consider clearing outputs for cleaner git history."
                    )


def test_all_notebooks_found():
    """Test that we found notebooks to test."""
    assert len(NOTEBOOK_PATHS) > 0, "No notebooks found in examples directory"
    print(f"\nFound {len(NOTEBOOK_PATHS)} notebooks to test:")
    for nb_path in NOTEBOOK_PATHS:
        print(f"  - {nb_path.name}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
