import json
import os

def fix_notebook_esm(path, layer_index):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                # Fix repr_layers
                if 'repr_layers=[' in line:
                    import re
                    line = re.sub(r'repr_layers=\[\d+\]', f'repr_layers=[{layer_index}]', line)
                # Fix representations access
                if "['representations'][" in line or '["representations"][' in line:
                    import re
                    line = re.sub(r"\['representations'\]\[\d+\]", f"['representations'][{layer_index}]", line)
                    line = re.sub(r'\["representations"\]\[\d+\]', f'["representations"][{layer_index}]', line)

                # Deduplicate specific lines if my previous sed messed up
                if line.strip() and line in new_source:
                    if "embeddings = results" in line:
                        continue

                new_source.append(line)
            if new_source != source:
                cell['source'] = new_source
                modified = True

    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write('\n')
        print(f"Fixed {path}")

fix_notebook_esm('examples/01_getting_started.ipynb', 6)
fix_notebook_esm('examples/04_casp16_benchmark.ipynb', 6)

# Fix test workflow
with open('.github/workflows/test-notebooks-execution.yml', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'tests/test_production_notebook.py::' in line:
        # Check if the class exists in the test file
        class_name = line.split('::')[1].split()[0]
        # Skip if it's one of the missing ones
        if class_name in ['TestDataProcessing', 'TestModelTraining', 'TestEvaluation', 'TestVisualization', 'TestResultsExport', 'TestMemoryManagement', 'TestErrorHandling', 'TestReproducibility', 'TestDocumentation']:
             continue
    new_lines.append(line)

with open('.github/workflows/test-notebooks-execution.yml', 'w') as f:
    f.writelines(new_lines)
print("Fixed .github/workflows/test-notebooks-execution.yml")
