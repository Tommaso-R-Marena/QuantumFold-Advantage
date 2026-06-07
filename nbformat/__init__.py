from .core import NotebookNode, read
import json

def write(nb, fp):
    if hasattr(fp, 'write'):
        json.dump(nb, fp, indent=2)
    else:
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)

from . import v4
