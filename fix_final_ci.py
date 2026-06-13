import re
import os

# 1. Fix .github/workflows/format.yml to use head_ref on PRs
filepath = '.github/workflows/format.yml'
if os.path.exists(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Fix the branch for push action - use main if ref is not head_ref
    content = content.replace('branch: ${{ github.ref }}', 'branch: ${{ github.head_ref || github.ref_name }}')

    # Fix actions versions back to stable
    content = content.replace('uses: actions/checkout@v6', 'uses: actions/checkout@v4')
    content = content.replace('uses: actions/setup-python@v6', 'uses: actions/setup-python@v5')
    content = content.replace('uses: actions/cache@v5', 'uses: actions/cache@v4')

    with open(filepath, 'w') as f:
        f.write(content)

# 2. Fix .github/workflows/test-notebooks.yml to use json.dump since nbformat shim is limited
filepath = '.github/workflows/test-notebooks.yml'
if os.path.exists(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Replace nbformat.write(limited_nb, f) with json.dump(limited_nb, f)
    # The shims in the repo don't have 'write'
    content = content.replace('nbformat.write(limited_nb, f)', 'import json; json.dump(limited_nb, f)')
    content = content.replace('nbformat.write(nb, f)', 'import json; json.dump(nb, f)')

    # Re-fix the attribute error: module 'nbformat' has no attribute 'v4'
    # Actually, looking at nbformat/__init__.py, it DOES NOT expose v4.
    # We should use 'import nbformat.v4' and then 'nbformat.v4.new_notebook()'
    # Wait, the error was AttributeError: module 'nbformat' has no attribute 'v4'
    # even though I added 'import nbformat.v4'.
    # Let's try 'from nbformat import v4' instead.
    content = content.replace('import nbformat.v4', 'from nbformat import v4')
    content = content.replace('nbformat.v4.', 'v4.')

    with open(filepath, 'w') as f:
        f.write(content)

# 3. Fix docker-publish.yml tags prefix to 'sha-' to avoid invalid reference format
filepath = '.github/workflows/docker-publish.yml'
if os.path.exists(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    content = content.replace('type=sha,prefix={{branch}}-', 'type=sha,prefix=sha-')
    # Stable actions
    content = content.replace('uses: actions/checkout@v6', 'uses: actions/checkout@v4')

    with open(filepath, 'w') as f:
        f.write(content)
