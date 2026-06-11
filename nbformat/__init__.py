from .core import NotebookNode, read
import json

def write(nb, fp):
    if hasattr(fp, 'write'):
        json.dump(nb, fp, indent=1)
    else:
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)

import nbformat.v4 as v4
