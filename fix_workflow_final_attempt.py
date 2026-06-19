import re

path = '.github/workflows/test-notebooks.yml'

def solve():
    with open(path, 'r') as f:
        lines = f.readlines()

    output = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect start of a step
        if line.strip().startswith('- name:') or line.strip().startswith('- uses:'):
            step_indent = re.match(r'^(\s*)', line).group(1)
            output.append(line)
            i += 1
            # Process lines within the same step
            while i < len(lines) and not (lines[i].strip().startswith('- name:') or lines[i].strip().startswith('- uses:') or re.match(r'^\s*[a-zA-Z0-9_-]+:', lines[i])):
                l = lines[i]
                if l.strip() == '':
                    output.append('\n')
                elif 'run: |' in l:
                     output.append(step_indent + '  run: |\n')
                elif "python - <<'EOF'" in l:
                     python_indent = step_indent + '    '
                     output.append(python_indent + "python - <<'EOF'\n")
                     i += 1
                     while i < len(lines) and lines[i].strip() != 'EOF':
                         # Indent python code
                         output.append(python_indent + '  ' + lines[i].strip() + '\n')
                         i += 1
                     if i < len(lines):
                         output.append(python_indent + '  EOF\n')
                else:
                     output.append(l)
                i += 1
            continue

        output.append(line)
        i += 1

    with open(path, 'w') as f:
        f.writelines(output)

# solve() # Still too complex to get right without manual review of every edge case.

# Let's try a simpler approach: fixed string replacements for the known broken parts.
with open(path, 'r') as f:
    content = f.read()

# Fix the 'run: |' alignment for many steps
content = re.sub(r'(\s+)- name: (.*?)\n\s+run: \|', r'\1- name: \2\n\1  run: |', content)

# Fix the indentation of the python code inside the blocks
# We'll target the blocks specifically.

def fix_eof_block(content, marker):
    # Find the step with 'marker' in name
    # and fix the python block inside it.
    pass

# Actually, the most reliable way is to restore the file and apply surgical changes carefully.
