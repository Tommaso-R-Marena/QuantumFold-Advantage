import re

path = '.github/workflows/test-notebooks.yml'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Add import json to dry-run-execution
content = content.replace(
    '          import nbformat\n          import glob\n          import os',
    '          import nbformat\n          import glob\n          import os\n          import json'
)

# Fix 2: Improve syntax-check by handling more magics and f-strings
syntax_check_script = """          import nbformat
          import glob
          import ast
          import sys
          import re

          errors = []
          for nb_path in glob.glob('examples/*.ipynb'):
              with open(nb_path, 'r', encoding='utf-8') as f:
                  nb = nbformat.read(f, as_version=4)

              for i, cell in enumerate(nb.cells):
                  if cell.cell_type == 'code':
                      lines = cell.source.splitlines()
                      clean_lines = []
                      for line in lines:
                          # Filter out magics and shell commands
                          if line.strip().startswith('%') or line.strip().startswith('!'):
                              clean_lines.append('pass')
                          else:
                              clean_lines.append(line)

                      source = '\\n'.join(clean_lines)
                      if not source.strip():
                          continue

                      try:
                          ast.parse(source)
                      except SyntaxError as e:
                          errors.append(f"{nb_path} cell {i}: {str(e)}")
                          print(f"✗ {nb_path} cell {i}: Syntax error: {e}")

          if errors:
              print(f"\\n{len(errors)} syntax error(s) found")
              sys.exit(1)
          else:
              print("\\n✓ All code cells have valid Python syntax")"""

# Use regex to replace the old script block
pattern = r'# Syntax checking.*?\n  syntax-check:.*?Extract and check Python syntax\n        run: \|.*?\n          import nbformat.*?EOF'
# This is a bit risky, let's just use string replace on a unique enough block
old_block = """          import nbformat
          import glob
          import ast
          import sys

          errors = []

          for nb_path in glob.glob('examples/*.ipynb'):
              with open(nb_path, 'r', encoding='utf-8') as f:
                  nb = nbformat.read(f, as_version=4)

              for i, cell in enumerate(nb.cells):
                  if cell.cell_type == 'code':
                      source = ''.join(cell.source)
                      if not source.strip():
                          continue

                      # Skip cells with shell commands
                      if source.strip().startswith('!'):
                          continue

                      # Skip cells with IPython magic
                      if source.strip().startswith('%'):
                          continue

                      try:
                          ast.parse(source)
                      except SyntaxError as e:
                          errors.append(f"{nb_path} cell {i}: {str(e)}")
                          print(f"✗ {nb_path} cell {i}: Syntax error")"""

content = content.replace(old_block, syntax_check_script)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
