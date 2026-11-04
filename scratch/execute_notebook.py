import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import time

NOTEBOOK_PATH = "cafe_revenue_regression.ipynb"

print(f"Loading {NOTEBOOK_PATH}...")
nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

print(f"Total cells to execute: {len(nb.cells)}")
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

t0 = time.time()
print("Executing notebook...")
ep.preprocess(nb, {'metadata': {'path': '.'}})
print(f"Execution finished in {time.time() - t0:.2f}s!")

nbformat.write(nb, NOTEBOOK_PATH)
print("Notebook saved with full execution outputs.")
