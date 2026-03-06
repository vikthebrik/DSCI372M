import json
with open('Project_4.ipynb') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source_lines = cell.get('source', [])
    source_text = "".join(source_lines)
    if cell['cell_type'] == 'markdown':
        print(f"--- Cell {i} (Markdown) ---")
        if len(source_text) > 200:
            print(source_text[:200] + "...\n")
        else:
            print(source_text + "\n")
