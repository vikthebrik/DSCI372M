import json

with open('Project_4.ipynb') as f:
    nb = json.load(f)

missing = []
for i, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', []))
    if cell['cell_type'] == 'markdown' and ('YOUR ANSWER HERE' in src or 'TODO' in src or '...' in src):
        missing.append(f"Cell {i} (Markdown): {src[:100]}...")
    elif cell['cell_type'] == 'code' and ('YOUR CODE HERE' in src or 'TODO' in src or '...' in src):
        missing.append(f"Cell {i} (Code): {src[:100]}...")

if missing:
    print("Found missing answers:")
    for m in missing:
        print(m)
else:
    print("No missing answers found!")
