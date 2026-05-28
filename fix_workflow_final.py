import re

with open('.github/workflows/test-notebooks-execution.yml', 'r') as f:
    content = f.read()

# Add numpy to dependencies
content = content.replace('pip install pytest nbformat nbconvert torch', 'pip install pytest nbformat nbconvert torch numpy')
content = content.replace('pip install pytest nbformat nbconvert torch pandas', 'pip install pytest nbformat nbconvert torch numpy pandas')

# Remove specific class calls that don't exist
content = re.sub(r'pytest tests/test_production_notebook\.py::\w+ -v', 'pytest tests/test_production_notebook.py -v', content)

# Remove duplicate consecutive pytest calls on the same file
lines = content.splitlines()
new_lines = []
last_pytest = ""
for line in lines:
    stripped = line.strip()
    if stripped.startswith('pytest tests/test_production_notebook.py -v'):
        if last_pytest == stripped:
            continue
        last_pytest = stripped
    else:
        last_pytest = ""
    new_lines.append(line)

content = '\n'.join(new_lines) + '\n'

with open('.github/workflows/test-notebooks-execution.yml', 'w') as f:
    f.write(content)
