#!/usr/bin/env python3
"""
Fix numpy compatibility for Google Colab in all Jupyter notebooks.

This script updates installation cells in notebooks to include automatic
runtime restart after pip install, preventing numpy binary incompatibility errors.

Usage:
    python fix_numpy_colab.py
"""

import json
import os
from pathlib import Path

# Updated installation cell with auto-restart
UPDATED_INSTALL_CELL = [
    "if IN_COLAB:\n",
    "    !git clone https://github.com/Tommaso-R-Marena/QuantumFold-Advantage.git\n",
    "    %cd QuantumFold-Advantage\n",
    "    !pip install -q -e '.[protein-lm]'\n",
    "    !pip install -q py3Dmol nglview biopython imageio\n",
    "    print('✅ Installation complete!')\n",
    "    print('⚠️  Restarting runtime to apply numpy 2.0 upgrade...')\n",
    "    print('    After restart, skip this cell and continue from imports.')\n",
    "    import os\n",
    "    import time\n",
    "    time.sleep(2)\n",
    "    os.kill(os.getpid(), 9)"
]

def fix_notebook(notebook_path):
    """
    Fix a single notebook by updating its installation cell.
    
    Args:
        notebook_path: Path to the notebook file
        
    Returns:
        bool: True if notebook was modified, False otherwise
    """
    print(f"Processing: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    # Find installation cells (look for cells with "git clone" and "pip install")
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Check if this is an installation cell
            if 'git clone' in source and 'pip install' in source:
                # Check if it already has the auto-restart
                if 'os.kill(os.getpid(), 9)' not in source:
                    print(f"  Found installation cell at index {i} - updating...")
                    cell['source'] = UPDATED_INSTALL_CELL
                    modified = True
                else:
                    print(f"  Installation cell at index {i} already has auto-restart")
    
    if modified:
        # Write back the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  ✅ Updated {notebook_path}")
        return True
    else:
        print(f"  ⏭️  No changes needed for {notebook_path}")
        return False

def main():
    """Main function to fix all notebooks in examples directory."""
    examples_dir = Path(__file__).parent
    notebooks = list(examples_dir.glob('*.ipynb'))
    
    print(f"Found {len(notebooks)} notebooks in {examples_dir}\n")
    
    updated_count = 0
    for notebook_path in sorted(notebooks):
        if fix_notebook(notebook_path):
            updated_count += 1
        print()
    
    print(f"\n{'='*60}")
    print(f"Summary: Updated {updated_count}/{len(notebooks)} notebooks")
    print(f"{'='*60}")
    print("\nAll notebooks are now compatible with Colab's numpy 2.0 requirement!")
    print("Users will no longer see binary incompatibility errors.")

if __name__ == '__main__':
    main()
