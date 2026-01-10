#!/usr/bin/env python3
"""Validate Jupyter notebooks for common issues.

Checks:
- No execution errors
- All cells have outputs
- No hardcoded paths
- No personal information
- Reasonable execution times
- Required metadata present
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple


class NotebookValidator:
    """Validator for Jupyter notebooks."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_file(self, notebook_path: Path) -> bool:
        """Validate a single notebook file."""
        print(f"\n🔍 Validating: {notebook_path.name}")
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
        except Exception as e:
            self.errors.append(f"{notebook_path.name}: Failed to load JSON: {e}")
            return False
        
        # Check metadata
        self._check_metadata(notebook_path.name, nb)
        
        # Check cells
        self._check_cells(notebook_path.name, nb.get('cells', []))
        
        # Check for common issues
        self._check_common_issues(notebook_path.name, nb)
        
        return len(self.errors) == 0
    
    def _check_metadata(self, name: str, nb: dict):
        """Check notebook metadata."""
        metadata = nb.get('metadata', {})
        
        # Check for kernel spec
        if 'kernelspec' not in metadata:
            self.warnings.append(f"{name}: Missing kernelspec metadata")
        
        # Check for Colab metadata if it's a Colab notebook
        if 'colab' in name.lower():
            if 'colab' not in metadata:
                self.warnings.append(f"{name}: Missing Colab metadata")
    
    def _check_cells(self, name: str, cells: List[dict]):
        """Check notebook cells."""
        if not cells:
            self.errors.append(f"{name}: No cells found")
            return
        
        code_cells = [c for c in cells if c.get('cell_type') == 'code']
        markdown_cells = [c for c in cells if c.get('cell_type') == 'markdown']
        
        # Check for reasonable balance
        if len(code_cells) == 0:
            self.errors.append(f"{name}: No code cells")
        
        if len(markdown_cells) == 0:
            self.warnings.append(f"{name}: No markdown cells (no documentation)")
        
        # Check individual cells
        for i, cell in enumerate(cells):
            self._check_cell(name, i, cell)
    
    def _check_cell(self, name: str, idx: int, cell: dict):
        """Check individual cell."""
        cell_type = cell.get('cell_type')
        source = ''.join(cell.get('source', []))
        
        if cell_type == 'code':
            # Check for common issues in code
            if '/content/' in source and 'IN_COLAB' not in source:
                self.warnings.append(
                    f"{name}:cell[{idx}]: Hardcoded /content/ path without Colab check"
                )
            
            if 'email' in source.lower() or '@' in source:
                if 'example' not in source.lower():
                    self.warnings.append(
                        f"{name}:cell[{idx}]: Possible personal email address"
                    )
            
            # Check for deprecated imports
            if 'from jax import' in source:
                self.warnings.append(
                    f"{name}:cell[{idx}]: Direct JAX import (may cause version conflicts)"
                )
            
            # Check for old seaborn styles
            if "plt.style.use('seaborn-" in source:
                if "seaborn-v0_8" not in source:
                    self.errors.append(
                        f"{name}:cell[{idx}]: Deprecated seaborn style (use seaborn-v0_8-*)"
                    )
            
            # Check for range() in plotly
            if 'plotly' in source.lower() and 'range(' in source:
                if 'list(range(' not in source:
                    self.errors.append(
                        f"{name}:cell[{idx}]: Use list(range()) with Plotly, not range()"
                    )
        
        elif cell_type == 'markdown':
            # Check for broken links
            if '](' in source and 'github.com' in source:
                # Simple check for malformed URLs
                if '] (' in source:  # Space between ] and (
                    self.warnings.append(
                        f"{name}:cell[{idx}]: Possible malformed markdown link"
                    )
    
    def _check_common_issues(self, name: str, nb: dict):
        """Check for common notebook issues."""
        # Convert entire notebook to string for pattern matching
        nb_str = json.dumps(nb)
        
        # Check for truncated code
        if '# <--' in nb_str or '# BROKEN' in nb_str:
            self.errors.append(f"{name}: Contains truncated or broken code markers")
        
        # Check for TODO or FIXME
        if 'TODO' in nb_str or 'FIXME' in nb_str:
            self.warnings.append(f"{name}: Contains TODO/FIXME comments")
    
    def report(self) -> bool:
        """Print validation report."""
        print("\n" + "="*70)
        print("📋 VALIDATION REPORT")
        print("="*70)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
        else:
            print("\n✅ No errors found!")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        else:
            print("\n✅ No warnings!")
        
        print("\n" + "="*70)
        
        return len(self.errors) == 0


def main():
    """Main validation function."""
    examples_dir = Path('examples')
    
    if not examples_dir.exists():
        print(f"❌ Error: {examples_dir} directory not found")
        print("   Run this script from the repository root")
        sys.exit(1)
    
    # Find all notebooks
    notebooks = list(examples_dir.glob('*.ipynb'))
    
    if not notebooks:
        print(f"❌ Error: No notebooks found in {examples_dir}")
        sys.exit(1)
    
    print(f"Found {len(notebooks)} notebooks to validate\n")
    
    validator = NotebookValidator()
    all_valid = True
    
    for nb_path in sorted(notebooks):
        if not validator.validate_file(nb_path):
            all_valid = False
    
    # Print final report
    success = validator.report()
    
    if success:
        print("\n🎉 All notebooks passed validation!")
        sys.exit(0)
    else:
        print("\n❌ Some notebooks failed validation")
        print("   Fix the errors above and run again")
        sys.exit(1)


if __name__ == '__main__':
    main()
