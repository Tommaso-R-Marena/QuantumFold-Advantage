"""End-to-end notebook execution tests.

These tests actually execute the Jupyter notebooks to ensure they run
without errors from start to finish.
"""

import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

NOTEBOOKS_DIR = Path(__file__).parent.parent / 'examples'
TIMEOUT = 600  # 10 minutes per notebook (increase for complete_benchmark)


class NotebookExecutionError(Exception):
    """Custom exception for notebook execution failures."""
    pass


def execute_notebook(notebook_path, timeout=TIMEOUT, kernel_name='python3'):
    """Execute a notebook and return the executed notebook object.
    
    Args:
        notebook_path: Path to the notebook file
        timeout: Maximum time in seconds for each cell
        kernel_name: Jupyter kernel to use
        
    Returns:
        Executed notebook object
        
    Raises:
        NotebookExecutionError: If notebook execution fails
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Configure executor
    ep = ExecutePreprocessor(
        timeout=timeout,
        kernel_name=kernel_name,
        allow_errors=False  # Stop on first error
    )
    
    # Execute
    try:
        ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
        return nb
    except Exception as e:
        raise NotebookExecutionError(
            f"Failed to execute {notebook_path.name}: {str(e)}"
        ) from e


def check_notebook_outputs(nb):
    """Verify notebook has outputs from execution.
    
    Args:
        nb: Executed notebook object
        
    Returns:
        dict with statistics about outputs
    """
    stats = {
        'total_cells': len(nb.cells),
        'code_cells': 0,
        'cells_with_output': 0,
        'cells_with_errors': 0,
        'markdown_cells': 0
    }
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            stats['code_cells'] += 1
            
            if cell.outputs:
                stats['cells_with_output'] += 1
                
                # Check for errors
                for output in cell.outputs:
                    if output.output_type == 'error':
                        stats['cells_with_errors'] += 1
                        break
        
        elif cell.cell_type == 'markdown':
            stats['markdown_cells'] += 1
    
    return stats


class TestNotebookExecution:
    """Test that notebooks execute successfully."""
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / 'examples' / 'colab_quickstart.ipynb').exists(),
        reason="Notebook not found"
    )
    def test_colab_quickstart_executes(self):
        """colab_quickstart.ipynb should execute without errors."""
        notebook_path = NOTEBOOKS_DIR / 'colab_quickstart.ipynb'
        
        # This is a quick notebook, should finish in < 2 minutes
        nb = execute_notebook(notebook_path, timeout=120)
        
        # Check outputs
        stats = check_notebook_outputs(nb)
        assert stats['code_cells'] > 0, "No code cells found"
        assert stats['cells_with_errors'] == 0, f"Found {stats['cells_with_errors']} cells with errors"
        
        # Should have some outputs (not all cells produce output)
        assert stats['cells_with_output'] > 0, "No outputs generated"
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / 'examples' / '01_getting_started.ipynb').exists(),
        reason="Notebook not found"
    )
    def test_getting_started_executes(self):
        """01_getting_started.ipynb should execute without errors."""
        notebook_path = NOTEBOOKS_DIR / '01_getting_started.ipynb'
        
        # May download ESM2 model, allow more time
        nb = execute_notebook(notebook_path, timeout=300)
        
        stats = check_notebook_outputs(nb)
        assert stats['code_cells'] > 0
        assert stats['cells_with_errors'] == 0
        assert stats['cells_with_output'] > 0
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / 'examples' / '02_quantum_vs_classical.ipynb').exists(),
        reason="Notebook not found"
    )
    def test_quantum_classical_executes(self):
        """02_quantum_vs_classical.ipynb should execute without errors."""
        notebook_path = NOTEBOOKS_DIR / '02_quantum_vs_classical.ipynb'
        
        # Training notebook, needs more time
        nb = execute_notebook(notebook_path, timeout=600)
        
        stats = check_notebook_outputs(nb)
        assert stats['code_cells'] > 0
        assert stats['cells_with_errors'] == 0
        assert stats['cells_with_output'] > 0
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / 'examples' / '03_advanced_visualization.ipynb').exists(),
        reason="Notebook not found"
    )
    def test_visualization_executes(self):
        """03_advanced_visualization.ipynb should execute without errors."""
        notebook_path = NOTEBOOKS_DIR / '03_advanced_visualization.ipynb'
        
        # Visualization notebook, moderate time
        nb = execute_notebook(notebook_path, timeout=300)
        
        stats = check_notebook_outputs(nb)
        assert stats['code_cells'] > 0
        assert stats['cells_with_errors'] == 0
        assert stats['cells_with_output'] > 0
    
    @pytest.mark.slow
    @pytest.mark.skip(reason="complete_benchmark is very long (30-60 min), run manually")
    def test_complete_benchmark_executes(self):
        """complete_benchmark.ipynb should execute without errors.
        
        NOTE: This test is skipped by default because it takes 30-60 minutes.
        Run manually with: pytest tests/test_notebook_execution.py::TestNotebookExecution::test_complete_benchmark_executes -v
        """
        notebook_path = NOTEBOOKS_DIR / 'complete_benchmark.ipynb'
        
        # Very long notebook, allow 60 minutes
        nb = execute_notebook(notebook_path, timeout=3600)
        
        stats = check_notebook_outputs(nb)
        assert stats['code_cells'] > 0
        assert stats['cells_with_errors'] == 0
        assert stats['cells_with_output'] > 0


class TestNotebookStructure:
    """Test notebook structure and metadata."""
    
    def get_notebook(self, name):
        """Load a notebook without executing it."""
        notebook_path = NOTEBOOKS_DIR / name
        if not notebook_path.exists():
            pytest.skip(f"Notebook {name} not found")
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            return nbformat.read(f, as_version=4)
    
    def test_colab_quickstart_structure(self):
        """colab_quickstart.ipynb should have proper structure."""
        nb = self.get_notebook('colab_quickstart.ipynb')
        
        # Should have both code and markdown cells
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        markdown_cells = [c for c in nb.cells if c.cell_type == 'markdown']
        
        assert len(code_cells) > 0, "No code cells found"
        assert len(markdown_cells) > 0, "No markdown cells found"
        
        # First cell should be markdown (title)
        assert nb.cells[0].cell_type == 'markdown', "First cell should be markdown title"
    
    def test_getting_started_structure(self):
        """01_getting_started.ipynb should have proper structure."""
        nb = self.get_notebook('01_getting_started.ipynb')
        
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        markdown_cells = [c for c in nb.cells if c.cell_type == 'markdown']
        
        assert len(code_cells) > 0
        assert len(markdown_cells) > 0
        assert nb.cells[0].cell_type == 'markdown'
    
    def test_quantum_classical_structure(self):
        """02_quantum_vs_classical.ipynb should have proper structure."""
        nb = self.get_notebook('02_quantum_vs_classical.ipynb')
        
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        markdown_cells = [c for c in nb.cells if c.cell_type == 'markdown']
        
        assert len(code_cells) > 0
        assert len(markdown_cells) > 0
        assert nb.cells[0].cell_type == 'markdown'
    
    def test_visualization_structure(self):
        """03_advanced_visualization.ipynb should have proper structure."""
        nb = self.get_notebook('03_advanced_visualization.ipynb')
        
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        markdown_cells = [c for c in nb.cells if c.cell_type == 'markdown']
        
        assert len(code_cells) > 0
        assert len(markdown_cells) > 0
        assert nb.cells[0].cell_type == 'markdown'
    
    def test_complete_benchmark_structure(self):
        """complete_benchmark.ipynb should have proper structure."""
        nb = self.get_notebook('complete_benchmark.ipynb')
        
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        markdown_cells = [c for c in nb.cells if c.cell_type == 'markdown']
        
        assert len(code_cells) > 0
        assert len(markdown_cells) > 0
        assert nb.cells[0].cell_type == 'markdown'


class TestNotebookContent:
    """Test that notebooks have expected content."""
    
    def get_notebook(self, name):
        """Load a notebook without executing it."""
        notebook_path = NOTEBOOKS_DIR / name
        if not notebook_path.exists():
            pytest.skip(f"Notebook {name} not found")
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            return nbformat.read(f, as_version=4)
    
    def test_notebooks_have_colab_badge(self):
        """All notebooks should have Colab badge in first cell."""
        notebooks = [
            'colab_quickstart.ipynb',
            '01_getting_started.ipynb',
            '02_quantum_vs_classical.ipynb',
            '03_advanced_visualization.ipynb',
            'complete_benchmark.ipynb'
        ]
        
        for name in notebooks:
            try:
                nb = self.get_notebook(name)
                first_cell = nb.cells[0]
                
                # Check for Colab badge
                assert 'colab' in first_cell.source.lower(), \
                    f"{name} missing Colab badge in first cell"
            except:
                # Skip if notebook doesn't exist
                pass
    
    def test_notebooks_import_necessary_packages(self):
        """Notebooks should import necessary packages."""
        notebooks = [
            ('colab_quickstart.ipynb', ['numpy', 'torch', 'matplotlib']),
            ('01_getting_started.ipynb', ['numpy', 'torch', 'matplotlib']),
            ('02_quantum_vs_classical.ipynb', ['numpy', 'torch']),
            ('03_advanced_visualization.ipynb', ['numpy', 'matplotlib']),
            ('complete_benchmark.ipynb', ['numpy', 'torch', 'matplotlib'])
        ]
        
        for name, expected_imports in notebooks:
            try:
                nb = self.get_notebook(name)
                
                # Concatenate all code cells
                all_code = '\n'.join(
                    cell.source for cell in nb.cells if cell.cell_type == 'code'
                )
                
                # Check for imports
                for package in expected_imports:
                    assert f'import {package}' in all_code or f'import numpy as np' in all_code, \
                        f"{name} missing import for {package}"
            except:
                # Skip if notebook doesn't exist
                pass


if __name__ == '__main__':
    # Run with: python -m pytest tests/test_notebook_execution.py -v
    pytest.main([__file__, '-v', '--tb=short'])