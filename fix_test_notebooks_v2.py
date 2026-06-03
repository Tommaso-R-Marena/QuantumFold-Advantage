import re

path = '.github/workflows/test-notebooks.yml'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Add import json to dry-run-execution
content = content.replace(
    '          import nbformat\n          import glob\n          import os',
    '          import nbformat\n          import glob\n          import os\n          import json'
)

# Fix 2: Use json.dump instead of nbformat.write in execute-lightweight
content = content.replace(
    '          with open(\'test_quickstart.ipynb\', \'w\') as f:\n              nbformat.write(nb, f)',
    '          import json\n          with open(\'test_quickstart.ipynb\', \'w\') as f:\n              json.dump(nb, f)'
)

# Fix 3: More robust syntax-check that preserves indentation when replacing magics
syntax_check_script = """          import nbformat
          import glob
          import ast
          import sys
          import re

          errors = []
          for nb_path in glob.glob('examples/*.ipynb'):
              try:
                  with open(nb_path, 'r', encoding='utf-8') as f:
                      nb = nbformat.read(f, as_version=4)
              except Exception as e:
                  print(f"✗ {nb_path}: Failed to read notebook: {e}")
                  continue

              for i, cell in enumerate(nb.cells):
                  if cell.cell_type == 'code':
                      source = cell.source
                      if isinstance(source, list):
                          source = ''.join(source)

                      lines = source.splitlines()
                      clean_lines = []
                      for line in lines:
                          # Filter out magics and shell commands while preserving indentation
                          if line.strip().startswith('%') or line.strip().startswith('!'):
                              indent = line[:len(line) - len(line.lstrip())]
                              clean_lines.append(indent + 'pass')
                          else:
                              clean_lines.append(line)

                      clean_source = '\\n'.join(clean_lines)
                      if not clean_source.strip():
                          continue

                      try:
                          ast.parse(clean_source)
                      except SyntaxError as e:
                          errors.append(f"{nb_path} cell {i}: {str(e)}")
                          print(f"✗ {nb_path} cell {i}: Syntax error: {e}")

          if errors:
              print(f"\\n{len(errors)} syntax error(s) found")
              sys.exit(1)
          else:
              print("\\n✓ All code cells have valid Python syntax")"""

# Find the syntax-check run block and replace it
# The old block is quite large, let's use a simpler match
start_marker = "Extract and check Python syntax"
# We need to find the run: | and the EOF
# Actually, I'll just rewrite the whole syntax-check job block for safety.

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
