import os
import re

def fix_workflow_imports_and_indent():
    for filename in os.listdir('.github/workflows'):
        if not filename.endswith('.yml'): continue
        path = os.path.join('.github/workflows', filename)
        with open(path, 'r') as f:
            content = f.read()

        # 1. Standardize artifact action
        content = re.sub(r'actions/upload-artifact@v[356]', 'actions/upload-artifact@v4', content)

        # 2. Fix Docker tag prefix
        content = content.replace('type=sha,prefix={{branch}}-', 'type=sha,prefix=sha-')

        # 3. Fix Python blocks in workflows
        # Find all python heredocs
        def fix_block(match):
            indent = match.group(1)
            code = match.group(2)

            # Remove any existing import block at the top if it's messy
            code = re.sub(r'^\s*import (nbformat|json|glob|sys|re|os).*?\n', '', code, flags=re.MULTILINE)

            # Detect what we need
            needed = ["import nbformat", "import nbformat.v4 as nbfv4", "import json"]
            if 'glob.glob' in code: needed.append("import glob")
            if 'sys.exit' in code or 'sys.argv' in code: needed.append("import sys")
            if 're.' in code: needed.append("import re")
            if 'os.' in code: needed.append("import os")

            # Prefix code with needed imports, properly indented
            new_code = "\n".join(needed) + "\n" + code.lstrip()
            # Indent each line of new_code to match the HEREDOC indent
            indented_code = "\n".join(indent + line if line.strip() else line for line in new_code.splitlines())
            return f"python - <<'EOF'\n{indented_code}\n{indent}EOF"

        content = re.sub(r"python - <<'EOF'\n(.*?)\n\s+EOF", fix_block, content, flags=re.DOTALL)

        # 4. Use json.dump instead of nbformat.write in workflows
        content = re.sub(r'nbformat\.write\(([^,]+),\s*f\)', r'json.dump(\1, f, indent=2)', content)

        with open(path, 'w') as f:
            f.write(content)

fix_workflow_imports_and_indent()
