import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from src.step3_preprocess import run_step3
from src.step5_scale import get_feature_column_types, build_leakage_free_pipeline
from src.step6_model_train import get_candidate_models, WeightedBlendEnsemble

train_final, test_final, stores_clean, weather_clean, holidays_clean, metadata = run_step3("data", verbose=False)
print("train_final shape:", train_final.shape)
print("test_final shape:", test_final.shape)
print("test_final null count in features:")
num_cols, cat_cols, ignore_cols = get_feature_column_types(train_final)
features = num_cols + cat_cols
nulls = test_final[features].isnull().sum()
print(nulls[nulls > 0])
