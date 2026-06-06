import json
import os

def fix_nb_metadata(path):
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except:
            return

    changed = False
    if 'metadata' not in nb or not nb['metadata']:
        nb['metadata'] = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}
        }
        changed = True

    # Fix specific cells known to have syntax errors
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                new_source = []
                for line in source:
                    # Fix Cohen's d
                    if "Cohen's d" in line:
                        line = line.replace("Cohen's d", "Cohen\'s d")
                    # Fix multi-line f-strings with newlines
                    if "print(f'\n" in line:
                        line = line.replace("print(f'\n", "print('\n")
                    new_source.append(line)
                if new_source != source:
                    cell['source'] = new_source
                    changed = True
            elif isinstance(source, str):
                new_source = source
                if "Cohen's d" in new_source:
                    new_source = new_source.replace("Cohen's d", "Cohen\'s d")
                if "print(f'\n" in new_source:
                    new_source = new_source.replace("print(f'\n", "print('\n")
                if new_source != source:
                    cell['source'] = new_source
                    changed = True

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Fixed {path}")

import glob
for nb in glob.glob('examples/*.ipynb'):
    fix_nb_metadata(nb)
