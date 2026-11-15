import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure src in sys.path
sys.path.insert(0, os.path.abspath('.'))

from src.step1_collect_prepare import load_raw_data, run_step1
from src.step2_explore import generate_eda_plots
from src.step4_split import plot_cv_timeline
from src.step7_evaluate import run_step7

print("=== 1. Generating EDA Plots 1 to 6 ===")
tables = load_raw_data("data")
train_merged, test_merged, _, _ = run_step1("data", verbose=False)
eda_paths = generate_eda_plots(train_merged, tables["weather"], output_dir="plots")
for p in eda_paths:
    print(f"  Generated: {p}")

print("\n=== 2. Generating Cross-Validation Timeline Plot ===")
timeline_path = plot_cv_timeline("plots/time_series_cv_timeline.png")
print(f"  Generated: {timeline_path}")

print("\n=== 3. Generating Evaluation Diagnostic Plots 1 to 6 ===")
eval_results = run_step7("data", plots_dir="plots", verbose=False)
for k, p in eval_results["plots_paths"].items():
    print(f"  Generated: {p}")

print("\nALL 13 PLOTS REGENERATED SUCCESSFULLY IN ENGLISH!")
