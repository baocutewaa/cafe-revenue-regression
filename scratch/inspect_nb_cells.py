import json

with open("cafe_revenue_regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in [61, 62, 63, 64, 65, 77, 78, 79, 80]:
    if i < len(nb["cells"]):
        cell = nb["cells"][i]
        src = "".join(cell.get("source", []))
        print(f"=== Cell {i} ({cell['cell_type']}) ===")
        print(src[:300])
        print("...")
