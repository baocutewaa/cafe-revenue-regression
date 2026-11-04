import nbformat

nb = nbformat.read("cafe_revenue_regression.ipynb", as_version=4)
for i in [46, 47, 48]:
    c = nb.cells[i]
    print(f"=== Cell {i} [{c.cell_type}] ===")
    print(c.source[:400])
