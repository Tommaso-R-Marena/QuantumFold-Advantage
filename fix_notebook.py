import json

with open('examples/04_casp16_benchmark.ipynb', 'r') as f:
    content = f.read()

# The error is likely here:
# "source": [
#   "import torch\nfrom src.data.casp16_loader import CASP16DataLoader\nfrom src.benchmarks.casp16_benchmark import CASP16Benchmark\nfrom src.advanced_model import AdvancedProteinFoldingModel\nfrom src.protein_embeddings import ESM2Embedder\n"
# "cell_type": "markdown",

# Let's try to reconstruct it if it's broken
try:
    data = json.loads(content)
    print("Notebook is valid JSON")
except Exception as e:
    print(f"Error: {e}")
    # Manual surgical fix for the specific reported error
    # line 19 column 7 (char 597)
    # Looking at the sed output, it's missing the closing part of the previous cell
    lines = content.splitlines()
    # Find the line with ESM2Embedder
    for i, line in enumerate(lines):
        if "ESM2Embedder" in line and '],' not in lines[i+1]:
             print(f"Found broken cell at line {i+1}")
             lines[i] = lines[i].rstrip() + '"'
             lines.insert(i+1, '      ]')
             lines.insert(i+2, '    },')
             lines.insert(i+3, '    {')
             break

    new_content = '\n'.join(lines)
    try:
        json.loads(new_content)
        with open('examples/04_casp16_benchmark.ipynb', 'w') as f:
            f.write(new_content)
        print("Fixed notebook")
    except Exception as e2:
        print(f"Failed to fix: {e2}")
