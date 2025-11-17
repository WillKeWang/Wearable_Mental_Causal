# Temporal Causal Estimation - Usage Guide

## Overview

This script (`run_temporal_causal_estimation.py`) performs causal estimation for three temporal causal relationships using Propensity Score Matching (PSM) and Linear Mixed Effects Models (LMM).

## Three Causal Relationships

### 1. Depression (t) → REM Variability (t)
- **Exposure**: `promis_dep_sum_t`
- **Outcome**: `rem_std_t`
- **Adjustment Sets**:
  - Set 1: `['age_binned']`
  - Set 2: `['rmssd_std_tm1']`
  - Set 3: `['promis_anx_sum_tm1', 'promis_dep_sum_tm1']`

### 2. Awake Variability (t) → Depression (t)
- **Exposure**: `awake_std_t`
- **Outcome**: `promis_dep_sum_t`
- **Adjustment Sets**:
  - Set 1: `['promis_anx_sum_tm1', 'promis_dep_sum_tm1']`
  - Set 2: `['age_binned', 'awake_std_tm1', 'onset_latency_std_t', 'onset_latency_std_tm1']`

### 3. Depression (t) → Awake Variability (t)
- **Exposure**: `promis_dep_sum_t`
- **Outcome**: `awake_std_t`
- **Adjustment Sets**:
  - Set 1: `['promis_anx_sum_tm1', 'promis_dep_sum_tm1']`
  - Set 2: `['age_binned', 'awake_std_tm1', 'onset_latency_std_t', 'onset_latency_std_tm1']`

## Data Requirements

Your dataset must be a CSV file from the preprocessed wearable+survey data.

### Required Columns:
- `pid`: Participant ID
- `date`: Survey date (for creating temporal pairs)

### Survey Variables:
- `promis_dep_sum`: Depression score
- `promis_anx_sum`: Anxiety score

### Wearable Variables (ending in _mean or _std):
- `rem_std`: REM sleep variability
- `awake_std`: Awake time variability
- `onset_latency_std`: Sleep onset latency variability
- `rmssd_std`: Heart rate variability (RMSSD)
- Other wearable metrics

### Demographic Variables (optional but recommended):
- `age`: Age (will be binned into categories)
- `sex`: Sex
- `race`: Race
- `ethnicity_hispanic`: Hispanic ethnicity

**Note**: The script automatically creates temporal pairs from consecutive surveys, adding `_t` and `_tm1` suffixes to variables.

## How to Use

### 1. Check the Data Path

The script is already configured with the correct path:

```python
DATA_PATH = "data/preprocessed/full_run/4w_to_0w_before/survey_wearable_28d_before_to_0d_before_baseline_adj_full.csv"
```

This file contains survey and wearable data from 4 weeks before to 0 weeks (survey date) before baseline.

**If you need to use a different data file**, edit line 531 in `run_temporal_causal_estimation.py`.

### 2. Run the Script

```bash
python run_temporal_causal_estimation.py
```

### 3. Optional: Adjust Configuration

You can modify these parameters in the script (around lines 534-539):

```python
# Temporal pairing parameters
MIN_DAYS_GAP = 21  # 3 weeks minimum between surveys
MAX_DAYS_GAP = 35  # 5 weeks maximum between surveys
USE_FIRST_PAIR_ONLY = True  # Use only first valid pair per participant

# Number of bootstrap iterations (default: 100)
N_BOOTSTRAP = 100

# You can also modify the adjustment sets if needed
```

### How Temporal Pairing Works

The script automatically creates temporal pairs from your data:

1. **Loads raw data**: Reads the CSV with variables like `promis_dep_sum`, `rem_std`, etc.
2. **Finds consecutive surveys**: For each participant, identifies pairs of surveys 21-35 days apart
3. **Creates temporal variables**:
   - Variables from the earlier survey (t-1) get suffix `_tm1`
   - Variables from the later survey (t) get suffix `_t`
   - Example: `promis_dep_sum` → `promis_dep_sum_tm1` and `promis_dep_sum_t`
4. **Uses first valid pair**: By default, uses only the first valid pair per participant (faster and cleaner)

## Output

### Console Output
The script will print detailed results for each relationship and adjustment set, including:
- Coefficient estimates with 95% CI
- Cohen's d effect sizes with 95% CI
- Median p-values
- Covariate balance statistics (SMD)

### Visualization Files
Three PNG files will be created:
- `causal_estimation_relationship_1.png` - Depression → REM comparison
- `causal_estimation_relationship_2.png` - Awake → Depression comparison
- `causal_estimation_relationship_3.png` - Depression → Awake comparison

Each visualization shows:
- **Left panel**: Coefficient estimates across different adjustment sets
- **Right panel**: Effect sizes (Cohen's d) across different adjustment sets

## Methods

### Propensity Score Matching (PSM)
- Fast nearest-neighbor matching using sklearn
- Prevents same-participant matching
- Caliper = 0.1 × SD(propensity scores)
- Checks covariate balance using Standardized Mean Difference (SMD < 0.25)

### Linear Mixed Effects Model (LMM)
- Random intercept for participants (pid)
- Adjustment covariates included as fixed effects
- Formula: `outcome ~ treated + covariates + (1|pid)`

### Bootstrap Analysis
- Pair-level resampling (preserves matched structure)
- 100 iterations (default)
- 80% of matched pairs sampled per iteration
- 95% confidence intervals from bootstrap distribution

## Understanding Results

### Coefficient
The average treatment effect (ATE) on the outcome:
- Positive: Higher exposure associated with higher outcome
- Negative: Higher exposure associated with lower outcome
- 95% CI not including 0: Statistically significant effect

### Cohen's d
Standardized effect size:
- Small: |d| ≈ 0.2
- Medium: |d| ≈ 0.5
- Large: |d| ≈ 0.8

### Covariate Balance (SMD)
After matching, check if covariates are balanced:
- ✓ SMD < 0.25: Good balance
- ✗ SMD ≥ 0.25: Poor balance (may indicate confounding)

## Troubleshooting

### "Data file not found" Error
- Ensure the `DATA_PATH` variable points to your actual CSV file
- Use absolute path if relative path doesn't work

### "Expected 4 PROMIS columns" Error
- This error comes from a different script
- Your temporal data doesn't need exactly 4 PROMIS columns
- The cleaning function checks for valid PROMIS ranges (4-20)

### Few Matched Pairs
- If you get very few matched pairs, try:
  - Increasing the caliper: `caliper=0.2` or `caliper=0.3`
  - Reducing the number of covariates in adjustment sets
  - Checking data quality and missingness

### Bootstrap Failures
- If many bootstrap iterations fail:
  - Reduce `sample_frac` (e.g., from 0.8 to 0.6)
  - Increase sample size
  - Check for sufficient matched pairs

## Example Output Interpretation

```
Adjustment Set 1: ['promis_anx_sum_tm1', 'promis_dep_sum_tm1']
  Coefficient: 0.1234 [0.0567, 0.1901]
  Cohen's d:   0.2345 [0.1123, 0.3567]
  p-value:     0.0023
```

**Interpretation:**
- The exposure increases the outcome by 0.1234 units on average
- This is a small-to-medium effect size (d = 0.23)
- The effect is statistically significant (p < 0.05)
- 95% CI doesn't include 0, confirming significance
- Adjusting for anxiety and depression at t-1

## Citation

Based on the causal estimation methodology from:
- Propensity Score Matching for causal inference
- Linear Mixed Effects Models for repeated measures
- Pair-level bootstrapping for robust inference

## Contact

For questions or issues, please check:
1. Data file format and column names
2. Missing values in key variables
3. PROMIS score ranges (should be 4-20)
4. Sufficient sample size for matching
