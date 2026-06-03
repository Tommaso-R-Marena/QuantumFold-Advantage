import re

path = '.github/workflows/test-notebooks.yml'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# The script was missing indentation for the errors.append block inside the try-except
# And the magics were being replaced with 'pass' but it might still fail if it's
# part of a larger multi-line block.

# Indentation-aware magic replacement script
script = """import nbformat
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
                # Indentation is critical to maintain block integrity for ast.parse
                # Replaces magics and shell commands with 'pass' while preserving indent
                stripped = line.lstrip()
                if stripped.startswith('%') or stripped.startswith('!'):
                    indent = line[:len(line) - len(stripped)]
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

# Inject into YAML using string replacement for the specific block
# (Too lazy to use full yaml parser again, string replacement of the previous script)

# Find the start of the script
start_marker = "import nbformat\n          import glob\n          import ast"
end_marker = "All code cells have valid Python syntax\")"

# Rebuild the whole job block for safety
new_job_content = """  # Syntax checking - ensures code is valid Python
  syntax-check:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install nbformat nbconvert ipython

      - name: Extract and check Python syntax
        run: |
          python - <<'EOF'
""" + "\\n".join(["          " + l for l in script.splitlines()]) + """
          EOF"""

# Use regex to find and replace the job
pattern = r'  # Syntax checking.*?\n  syntax-check:.*?\n          EOF'
content = re.sub(pattern, new_job_content, content, flags=re.DOTALL)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
