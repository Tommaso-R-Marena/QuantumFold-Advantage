#!/usr/bin/env python3
"""Fix multi-line f-strings in Jupyter notebooks that cause JSON parsing errors.

This script identifies and fixes multi-line f-string statements that span
multiple lines in Jupyter notebook cells, which can cause JSON parsing errors
in Google Colab.

Usage:
    python fix_notebook_json.py <notebook_path>
    python fix_notebook_json.py examples/complete_benchmark.ipynb
"""

import json
import sys
import re
from pathlib import Path


def fix_multiline_fstrings(source_lines):
    """Fix multi-line f-strings by merging them into single lines.
    
    Args:
        source_lines: List of source code lines from a notebook cell
    
    Returns:
        Fixed list of source code lines
    """
    fixed_lines = []
    i = 0
    
    while i < len(source_lines):
        line = source_lines[i]
        
        # Check if this line contains an f-string that might continue
        # Pattern: line ends with a quote (closing an f-string)
        # and the next line starts with f' or f"
        if i < len(source_lines) - 1:
            stripped_line = line.rstrip()
            next_line = source_lines[i + 1].lstrip()
            
            # Check for continuation pattern
            # Line ends with ' or " and next line starts with f' or f"
            if ((stripped_line.endswith("'") or stripped_line.endswith('"')) and
                (next_line.startswith("f'") or next_line.startswith('f"'))):
                
                # Check if they're part of the same statement (same indentation context)
                # Look for leading whitespace
                curr_indent = len(line) - len(line.lstrip())
                next_indent = len(source_lines[i + 1]) - len(source_lines[i + 1].lstrip())
                
                # If next line is indented more than current, it's likely a continuation
                if next_indent > curr_indent:
                    # Merge the f-strings
                    # Remove closing quote from current line
                    merged = stripped_line[:-1]
                    # Remove f' or f" from next line and get the rest
                    continuation = next_line[2:]
                    # Concatenate with a space
                    merged_line = merged + " " + continuation
                    # Preserve original line's leading whitespace
                    leading_space = line[:curr_indent]
                    fixed_lines.append(leading_space + merged_line.lstrip() + "\n")
                    i += 2  # Skip the next line since we merged it
                    continue
        
        fixed_lines.append(line)
        i += 1
    
    return fixed_lines


def fix_notebook(notebook_path):
    """Fix a Jupyter notebook by merging multi-line f-strings.
    
    Args:
        notebook_path: Path to the notebook file
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found: {notebook_path}")
        sys.exit(1)
    
    print(f"Processing: {notebook_path}")
    
    # Read notebook
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse notebook as JSON: {e}")
        print(f"Position: line {e.lineno}, column {e.colno}")
        sys.exit(1)
    
    # Process each cell
    fixed_count = 0
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            original_source = cell['source']
            fixed_source = fix_multiline_fstrings(original_source)
            
            if fixed_source != original_source:
                cell['source'] = fixed_source
                fixed_count += 1
    
    if fixed_count == 0:
        print("No multi-line f-strings found to fix.")
        return
    
    print(f"Fixed {fixed_count} code cell(s)")
    
    # Create backup
    backup_path = notebook_path.with_suffix('.ipynb.backup')
    print(f"Creating backup: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    # Write fixed notebook
    print(f"Writing fixed notebook: {notebook_path}")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ Done!")
    print("\nTo verify the fix:")
    print(f"  python -m json.tool {notebook_path} > /dev/null && echo 'Valid JSON'")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python fix_notebook_json.py <notebook_path>")
        print("Example: python fix_notebook_json.py examples/complete_benchmark.ipynb")
        sys.exit(1)
    
    fix_notebook(sys.argv[1])
