import json

with open("cafe_revenue_regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

for i, cell in enumerate(nb['cells']):
    src = "".join(cell.get("source", []))
    if any(k in src for k in ["DEFAULT_ENSEMBLE_WEIGHTS", "run_step6", "run_step8", "clean_train_outliers", "DeSeasonalizedStoreStats"]):
        print(f"Cell {i} ({cell['cell_type']}):")
        lines = [l.strip() for l in src.split("\n") if any(k in l for k in ["DEFAULT_ENSEMBLE_WEIGHTS", "run_step6", "run_step8", "clean_train_outliers", "DeSeasonalizedStoreStats", "mask_10x", "mask_01x", "weights"])]
        print("  " + "\n  ".join(lines[:5]))
