import re

path = '.github/workflows/test-notebooks.yml'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the dry-run-execution job to not use nbformat.v4 if it's not working,
# or just ensure it is imported correctly.
# The error was "AttributeError: module 'nbformat' has no attribute 'v4'"
# It should be "from nbformat import v4" or similar if the version is old,
# but usually "nbformat.v4" works in recent versions.
# Let's check how it's used in the workflow.

# Based on logs, it was:
#    limited_nb = nbformat.v4.new_notebook()
#    limited_nb.cells = nb.cells[:min(5, len(nb.cells))]
#    limited_nb.cells.append(
#        nbformat.v4.new_code_cell('print(\"✓ Dry-run completed successfully\")')
#    )

# Let's replace it with a more robust version using json directly since a notebook is just JSON.
# Or use nbformat correctly.

fixed_content = content.replace(
    '    limited_nb = nbformat.v4.new_notebook()',
    '    limited_nb = { \"cells\": [], \"metadata\": nb.metadata, \"nbformat\": nb.nbformat, \"nbformat_minor\": nb.nbformat_minor }'
).replace(
    '    limited_nb.cells = nb.cells[:min(5, len(nb.cells))]',
    '    limited_nb[\"cells\"] = nb.cells[:min(5, len(nb.cells))]'
).replace(
    '    limited_nb.cells.append(',
    '    limited_nb[\"cells\"].append('
).replace(
    '        nbformat.v4.new_code_cell(\'print(\"✓ Dry-run completed successfully\")\')',
    '        { \"cell_type\": \"code\", \"execution_count\": None, \"metadata\": {}, \"outputs\": [], \"source\": [\"print(\\\"✓ Dry-run completed successfully\\\")\"] }'
).replace(
    '        nbformat.write(limited_nb, f)',
    '        json.dump(limited_nb, f)'
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)
