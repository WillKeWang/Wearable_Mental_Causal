#!/usr/bin/env python

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

# Load raw Oura data
print("=== RAW OURA DATA ===")
df_oura = pq.read_table("data/raw/oura_all_sleep_summary_stacked_w_temp_max.parquet").to_pandas()
df_oura = df_oura.reset_index()

# Check hr_average in raw data
print(f"HR columns: {[c for c in df_oura.columns if 'hr' in c.lower()]}")
print(f"Raw hr_average stats:")
print(df_oura['hr_average'].describe())
print(f"Sample values: {df_oura['hr_average'].dropna().head(10).values}")

# Load a recent processed file to compare
print("\n=== PROCESSED DATA ===")
processed_files = list(Path("data/preprocessed").glob("*baseline_adj*.csv")) if Path("data/preprocessed").exists() else []
if processed_files:
    df_processed = pd.read_csv(processed_files[0])
    hr_cols = [c for c in df_processed.columns if 'hr_average' in c]
    print(f"Processed HR columns: {hr_cols}")
    if hr_cols:
        print(f"Processed hr_average_mean stats:")
        print(df_processed[hr_cols[0]].describe())
        print(f"Sample values: {df_processed[hr_cols[0]].dropna().head(10).values}")

# Test one participant's baseline calculation
print("\n=== BASELINE CALCULATION TEST ===")
sample_pid = df_oura['pid'].iloc[0]
df_pid = df_oura[df_oura['pid'] == sample_pid].sort_values('date')
print(f"PID: {sample_pid}")
print(f"First 30 days hr_average: {df_pid['hr_average'].head(30).mean():.2f} bpm")
print(f"Overall hr_average: {df_pid['hr_average'].mean():.2f} bpm")
print(f"After baseline adjustment: {(df_pid['hr_average'] - df_pid['hr_average'].head(30).mean()).head(10).values}")