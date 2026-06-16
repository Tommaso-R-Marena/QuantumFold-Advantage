import json
import os
import re

def fix_workflow(path):
    with open(path, 'r') as f:
        content = f.read()

    # Standardize artifact action
    content = re.sub(r'actions/upload-artifact@v[356]', 'actions/upload-artifact@v4', content)

    # Fix Docker tag prefix
    content = content.replace('type=sha,prefix={{branch}}-', 'type=sha,prefix=sha-')

    # Fix push branch in format.yml
    if 'format.yml' in path:
        content = content.replace('branch: refs/pull/93/merge', 'branch: ${{ github.head_ref }}')

    # Fix Python HEREDOC blocks with robust replacement
    def repl(m):
        indent = m.group(1)
        code = m.group(2)

        # Determine needed imports based on usage in code
        needed = ["import nbformat", "import nbformat.v4 as nbfv4", "import json"]
        if 'glob.glob' in code: needed.append("import glob")
        if 'sys.exit' in code or 'sys.argv' in code or 'sys.path' in code: needed.append("import sys")
        if 're.' in code: needed.append("import re")
        if 'os.' in code: needed.append("import os")

        # Clean the code of existing messy imports at the top
        lines = code.splitlines()
        while lines and re.match(r'^\s*import (nbformat|json|glob|sys|re|os)', lines[0]):
            lines.pop(0)

        # Reconstruct with proper indentation
        # HEREDOC indent is usually the indent of the line after python - <<'EOF'
        # Let's use 10 spaces as a safe standard for these workflows
        code_indent = "          "
        new_lines = [code_indent + imp for imp in needed]
        new_lines.append("") # empty line after imports
        for line in lines:
            new_lines.append(code_indent + line.strip() if line.strip() else "")

        return "python - <<'EOF'\n" + "\n".join(new_lines) + "\n" + indent + "EOF"

    content = re.sub(r"python - <<'EOF'\n(.*?)\n\s+EOF", repl, content, flags=re.DOTALL)

    with open(path, 'w') as f:
        f.write(content)

def fix_nb_01():
    path = 'examples/01_getting_started.ipynb'
    if not os.path.exists(path): return
    with open(path, 'r') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            # 1. Downgrade model
            src = src.replace("esm2_t36_3B_UR50D", "esm2_t6_8M_UR50D")
            # 2. Fix layer indexing
            src = src.replace("repr_layers=[36]", "repr_layers=[6]")
            src = src.replace("results['representations'][36]", "results['representations'][6]")
            # 3. Fix input_dim for ImprovedProteinPredictor
            src = re.sub(r'input_dim=2560', 'input_dim=320', src)
            cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in src.splitlines()]
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

def fix_nb_02():
    path = 'examples/02_quantum_vs_classical.ipynb'
    if not os.path.exists(path): return
    with open(path, 'r') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if 'class QuantumModel' in src:
                # Completely rewrite the cell to be clean and correct
                new_src = """feature_dim = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QuantumModel(nn.Module):
    def __init__(self, feature_dim, n_qubits=4, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(3, feature_dim)
        if QUANTUM_AVAILABLE:
            self.quantum = QuantumAttentionLayer(
                embed_dim=feature_dim,
                n_qubits=n_qubits,
                n_heads=n_heads
            )
        else:
            self.quantum = nn.MultiheadAttention(
                feature_dim, n_heads, batch_first=True
            )
        self.output = nn.Linear(feature_dim, 3)

    def forward(self, x):
        x = self.input_proj(x)
        if QUANTUM_AVAILABLE:
            x = self.quantum(x)
        else:
            x, _ = self.quantum(x, x, x)
        return self.output(x)

class ClassicalModel(nn.Module):
    def __init__(self, feature_dim, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(3, feature_dim)
        self.attention = nn.MultiheadAttention(
            feature_dim, n_heads, batch_first=True
        )
        self.output = nn.Linear(feature_dim, 3)

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.attention(x, x, x)
        return self.output(x)

print('🏗️  Initializing models...')
quantum_model = QuantumModel(feature_dim).to(device)
classical_model = ClassicalModel(feature_dim).to(device)

q_params = sum(p.numel() for p in quantum_model.parameters())
c_params = sum(p.numel() for p in classical_model.parameters())

print(f'\\n📊 Models initialized on {device}')
print(f'   Quantum parameters:   {q_params:,}')
print(f'   Classical parameters: {c_params:,}')
print(f'   Parameter difference: {abs(q_params - c_params):,}')"""
                cell['source'] = [line + '\n' for line in new_src.splitlines()]
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

for wf in os.listdir('.github/workflows'):
    if wf.endswith('.yml'):
        fix_workflow(os.path.join('.github/workflows', wf))

fix_nb_01()
fix_nb_02()
