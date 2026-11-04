import nbformat

nb = nbformat.read("cafe_revenue_regression.ipynb", as_version=4)
print("--- CELL 66 OUTPUT (Summary Table) ---")
for o in nb.cells[66].outputs:
    if o.output_type == "stream":
        print(o.text)

print("\n--- CELL 68 OUTPUT (CV Benchmark Table) ---")
for o in nb.cells[68].outputs:
    if o.output_type == "stream":
        print(o.text)
