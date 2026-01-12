#!/usr/bin/env python3
"""Apply NumPy 2.0 compatibility fix to all notebooks.

This script modifies the installation cells in Jupyter notebooks to include
the NumPy <2.0 fix required for autograd/pennylane compatibility.
"""

import json
import sys
from pathlib import Path
import argparse


# The fixed installation code to insert
NUMPY_FIX = """
    # FIX: NumPy 2.0 incompatibility with autograd/pennylane
    print('\\n🔧 Fixing NumPy 2.0 compatibility...')
    !pip uninstall -y numpy jax jaxlib autograd pennylane 2>/dev/null || true
    !pip install --force-reinstall --no-deps 'numpy>=1.23.0,<2.0.0'
    !pip install --no-deps 'autograd>=1.6.2'
    !pip install --no-deps 'pennylane>=0.32.0'
    print('✅ NumPy compatibility fixed')
"""

VERIFICATION = """
    # Verify installation
    print('\\n🔍 Verifying installation...')
    import numpy as np
    import torch
    print(f'✅ NumPy version: {np.__version__}')
    print(f'✅ PyTorch version: {torch.__version__}')
    try:
        import pennylane as qml
        print(f'✅ PennyLane version: {qml.__version__}')
    except:
        print('⚠️  PennyLane not available')
"""


def find_installation_cell(notebook):
    """Find the cell with installation commands.
    
    Args:
        notebook: Loaded notebook dict
        
    Returns:
        Index of installation cell or None
    """
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Look for pip install commands
        if 'pip install' in source and ('numpy' in source or 'torch' in source):
            return i
    
    return None


def has_numpy_fix(cell_source):
    """Check if cell already has NumPy fix.
    
    Args:
        cell_source: Source code of cell
        
    Returns:
        True if fix is present
    """
    source = ''.join(cell_source) if isinstance(cell_source, list) else cell_source
    return 'NumPy 2.0' in source or 'numpy>=1.23.0,<2.0.0' in source


def apply_numpy_fix(notebook_path, dry_run=False, verbose=False):
    """Apply NumPy fix to a notebook.
    
    Args:
        notebook_path: Path to notebook file
        dry_run: If True, don't write changes
        verbose: If True, print detailed info
        
    Returns:
        True if changes were made
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        print(f"❌ Not found: {notebook_path}")
        return False
    
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find installation cell
    install_cell_idx = find_installation_cell(notebook)
    
    if install_cell_idx is None:
        if verbose:
            print(f"⚠️  No installation cell found in {notebook_path.name}")
        return False
    
    cell = notebook['cells'][install_cell_idx]
    
    # Check if fix already present
    if has_numpy_fix(cell['source']):
        if verbose:
            print(f"✅ Already fixed: {notebook_path.name}")
        return False
    
    # Apply fix
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    lines = source.split('\n')
    
    # Find where to insert (after pip upgrade, before other installs)
    insert_idx = None
    for i, line in enumerate(lines):
        if 'pip install --upgrade' in line or 'pip upgrade' in line:
            # Insert after pip upgrade
            insert_idx = i + 1
            break
        elif 'pip install' in line and 'numpy' in line:
            # Insert before numpy install
            insert_idx = i
            break
    
    if insert_idx is None:
        # Insert after git clone / cd commands
        for i, line in enumerate(lines):
            if '%cd' in line or 'cd ' in line:
                insert_idx = i + 1
                break
    
    if insert_idx is None:
        insert_idx = 5  # Safe default
    
    # Insert fix
    fix_lines = NUMPY_FIX.strip().split('\n')
    for j, fix_line in enumerate(fix_lines):
        lines.insert(insert_idx + j, fix_line)
    
    # Add verification at the end (before final message)
    for i in range(len(lines) - 1, -1, -1):
        if 'Installation complete' in lines[i] or 'install' in lines[i].lower():
            verify_lines = VERIFICATION.strip().split('\n')
            for j, verify_line in enumerate(verify_lines):
                lines.insert(i, verify_line)
            break
    
    # Update cell source
    cell['source'] = '\n'.join(lines)
    
    if not dry_run:
        # Write back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"✅ Fixed: {notebook_path.name}")
    else:
        print(f"🔍 Would fix: {notebook_path.name}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Apply NumPy 2.0 fix to notebooks')
    parser.add_argument('notebooks', nargs='*', help='Notebook files to fix (default: all in examples/)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    # Determine notebooks to process
    if args.notebooks:
        notebook_paths = [Path(nb) for nb in args.notebooks]
    else:
        # Find all notebooks in examples/
        examples_dir = Path(__file__).parent.parent / 'examples'
        if not examples_dir.exists():
            print(f"❌ Examples directory not found: {examples_dir}")
            return 1
        
        notebook_paths = sorted(examples_dir.glob('*.ipynb'))
        
        # Skip complete_benchmark.ipynb (already has the fix)
        notebook_paths = [p for p in notebook_paths if 'complete_benchmark' not in p.name]
    
    if not notebook_paths:
        print("❌ No notebooks found")
        return 1
    
    print(f"\n🔧 Processing {len(notebook_paths)} notebook(s)...\n")
    
    fixed_count = 0
    for notebook_path in notebook_paths:
        if apply_numpy_fix(notebook_path, dry_run=args.dry_run, verbose=args.verbose):
            fixed_count += 1
    
    print(f"\n📊 Summary: {fixed_count}/{len(notebook_paths)} notebook(s) {'would be ' if args.dry_run else ''}fixed")
    
    if args.dry_run:
        print("\n💡 Run without --dry-run to apply changes")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())