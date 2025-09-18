import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

print("=== INSPECTING DATA STRUCTURE ===")

# Load Oura data
oura_path = "data/raw/oura_all_sleep_summary_stacked_w_temp_max.parquet"
print(f"\n📊 OURA DATA ({oura_path})")
print("-" * 50)

try:
    df_oura = pq.read_table(oura_path).to_pandas()
    print(f"Shape: {df_oura.shape}")
    print(f"Columns ({len(df_oura.columns)}): {list(df_oura.columns[:10])}...")  # First 10 columns
    print(f"All columns: {list(df_oura.columns)}")
    print(f"\nFirst few rows:")
    print(df_oura.head(3))
    
    # Look for participant ID column
    pid_candidates = [col for col in df_oura.columns if 'id' in col.lower() or 'pid' in col.lower() or 'participant' in col.lower()]
    print(f"\nPossible participant ID columns: {pid_candidates}")
    
except Exception as e:
    print(f"Error loading Oura data: {e}")

# Load Survey data  
survey_path = "data/raw/base_monthly.csv"
print(f"\n📋 SURVEY DATA ({survey_path})")
print("-" * 50)

try:
    df_survey = pd.read_csv(survey_path, index_col=0)
    print(f"Shape: {df_survey.shape}")
    print(f"Columns ({len(df_survey.columns)}): {list(df_survey.columns[:10])}...")  # First 10 columns
    print(f"All columns: {list(df_survey.columns)}")
    print(f"\nFirst few rows:")
    print(df_survey.head(3))
    
    # Look for participant ID column
    pid_candidates = [col for col in df_survey.columns if 'id' in col.lower() or 'pid' in col.lower() or 'participant' in col.lower()]
    print(f"\nPossible participant ID columns: {pid_candidates}")
    
except Exception as e:
    print(f"Error loading Survey data: {e}")

print("\n" + "="*60)
print("SUMMARY:")
print("1. What is the participant ID column called in each dataset?")
print("2. Are there any other important columns I should know about?")
print("3. Do the participant IDs match between the two datasets?")