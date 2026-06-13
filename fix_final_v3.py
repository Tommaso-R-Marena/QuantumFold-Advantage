import re
import os
import json

# 1. Fix .github/workflows/test-notebooks-execution.yml
filepath = '.github/workflows/test-notebooks-execution.yml'
if os.path.exists(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Fix the missing class names and ensure numpy is installed for conftest
    content = content.replace('pytest tests/test_production_notebook.py::TestConfigurationCells -v',
                              'pip install numpy && pytest tests/test_production_notebook.py::TestNotebookStructure -v')

    # Fix actions versions and add numpy to structure tests
    content = content.replace('pip install pytest nbformat nbconvert torch',
                              'pip install pytest nbformat nbconvert torch numpy')

    with open(filepath, 'w') as f:
        f.write(content)

# 2. Fix examples/02_quantum_vs_classical.ipynb - Join character list
filepath = 'examples/02_quantum_vs_classical.ipynb'
if os.path.exists(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if isinstance(cell.get('source'), list):
            # Join if it's a list of strings OR if it's a list of single characters
            # Some versions of this notebook have been seen with ["i", "m", "p", ...]
            cell['source'] = "".join(cell['source'])

        # Add feature_dim definition to the model initialization cell if missing
        if 'QuantumModel(feature_dim)' in cell.get('source', '') and 'feature_dim =' not in cell.get('source', ''):
             cell['source'] = "feature_dim = 128\n" + cell['source']

    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)

# 3. Fix 01_getting_started.ipynb - ESM model downgrade and repr_layers fix
filepath = 'examples/01_getting_started.ipynb'
if os.path.exists(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        source = "".join(cell.get('source', [])) if isinstance(cell.get('source'), list) else cell.get('source', '')

        # Downgrade ESM model to avoid timeout
        if 'esm2_t33_650M_UR50D' in source:
            source = source.replace('esm2_t33_650M_UR50D', 'esm2_t6_8M_UR50D')
            cell['source'] = source

        # Fix representation layers for 8M model (6 layers)
        if 'repr_layers=[33]' in source:
            source = source.replace('repr_layers=[33]', 'repr_layers=[6]')
            cell['source'] = source

        # Fix emb_dim in ImprovedProteinPredictor
        if 'emb_dim=1280' in source:
            source = source.replace('emb_dim=1280', 'emb_dim=320')
            cell['source'] = source

    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)
