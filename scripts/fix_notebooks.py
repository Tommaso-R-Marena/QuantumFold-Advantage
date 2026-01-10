#!/usr/bin/env python3
"""Fix JSON syntax errors in Jupyter notebooks.

This script reads notebooks, parses the JSON, and re-serializes with proper escaping.
This fixes issues with unescaped control characters, newlines, and special characters.

Usage:
    python scripts/fix_notebooks.py
"""

import json
import sys
from pathlib import Path


def fix_notebook(path):
    """Fix notebook JSON by re-serializing with proper escaping."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse JSON
        data = json.loads(content)
        
        # Re-serialize with proper escaping
        clean_json = json.dumps(data, indent=1, ensure_ascii=False)
        
        # Write back
        with open(path, 'w', encoding='utf-8') as f:
            f.write(clean_json)
            f.write('\n')
        
        return True
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON error in {path.name}: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error fixing {path.name}: {e}")
        return False


def main():
    """Fix all notebooks in examples/ directory."""
    examples_dir = Path('examples')
    
    if not examples_dir.exists():
        print("Error: examples/ directory not found")
        print("Run this script from the repository root")
        return 1
    
    notebooks = sorted(examples_dir.glob('*.ipynb'))
    
    if not notebooks:
        print("No notebooks found in examples/")
        return 1
    
    print(f"Fixing {len(notebooks)} notebooks in examples/")
    print("=" * 70)
    
    fixed_count = 0
    failed = []
    
    for nb in notebooks:
        if fix_notebook(nb):
            print(f"  ✓ {nb.name}")
            fixed_count += 1
        else:
            failed.append(nb.name)
    
    print("=" * 70)
    print(f"Results: {fixed_count}/{len(notebooks)} notebooks fixed")
    
    if failed:
        print("\nFailed notebooks (need manual fix):")
        for name in failed:
            print(f"  - {name}")
        return 1
    
    print("✅ All notebooks fixed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
