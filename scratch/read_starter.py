import nbformat

nb = nbformat.read("data/starter_cafe.ipynb", as_version=4)
print(f"Total cells in starter_cafe.ipynb: {len(nb.cells)}")
for i, c in enumerate(nb.cells):
    if c.cell_type == "code":
        print(f"=== Code Cell {i} ===")
        print(c.source[:300])
        print("---")
