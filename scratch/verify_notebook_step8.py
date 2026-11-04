import nbformat

nb = nbformat.read("cafe_revenue_regression.ipynb", as_version=4)
print(f"Total cells: {len(nb.cells)}")
for idx in range(75, len(nb.cells)):
    c = nb.cells[idx]
    if c.cell_type == "code":
        out_types = [o.output_type for o in c.outputs]
        text_preview = ""
        for o in c.outputs:
            if o.output_type == "stream":
                text_preview = o.text.split("\n")[0]
                break
        print(f"Cell {idx} [CODE]: outputs={len(c.outputs)} ({out_types}) -> '{text_preview}'")
    else:
        title = c.source.split("\n")[0]
        print(f"Cell {idx} [MD]: {title}")
