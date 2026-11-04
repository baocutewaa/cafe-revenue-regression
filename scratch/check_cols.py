import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from src.step3_preprocess import run_step3
from src.step5_scale import get_feature_column_types

train_final, test_final, _, _, _, _ = run_step3("data", verbose=False)
num_cols, cat_cols, ignore_cols = get_feature_column_types(train_final)
print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)
print("Ignore columns:", ignore_cols)
