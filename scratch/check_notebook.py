import nbformat

nb = nbformat.read("cafe_revenue_regression.ipynb", as_version=4)
print(f"Total cells: {len(nb.cells)}")
for i in range(max(0, len(nb.cells) - 8), len(nb.cells)):
    c = nb.cells[i]
    first_line = c.source.split("\n")[0] if c.source else ""
    print(f"Cell {i} [{c.cell_type}]: {first_line}")
