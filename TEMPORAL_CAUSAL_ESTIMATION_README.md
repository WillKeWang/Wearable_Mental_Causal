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

Your dataset must be a CSV file containing:

### Required Columns:
- `pid`: Participant ID (for random effects grouping)

### Temporal Variables (with _t and _tm1 suffixes):
- **Current time (t)**:
  - `promis_dep_sum_t`
  - `promis_anx_sum_t`
  - `rem_std_t`
  - `awake_std_t`
  - `onset_latency_std_t`

- **Lagged time (t-1)**:
  - `promis_dep_sum_tm1`
  - `promis_anx_sum_tm1`
  - `rmssd_std_tm1`
  - `awake_std_tm1`
  - `onset_latency_std_tm1`

- **Demographic**:
  - `age_binned` (or other demographic variables)

## How to Use

### 1. Update the Data Path

Open `run_temporal_causal_estimation.py` and update line 617:

```python
DATA_PATH = "data/temporal_causal_data.csv"  # Update this path!
```

Replace with your actual data file path.

### 2. Run the Script

```bash
python run_temporal_causal_estimation.py
```

### 3. Optional: Adjust Configuration

You can modify these parameters in the script:

```python
# Number of bootstrap iterations (default: 100)
N_BOOTSTRAP = 100

# You can also modify the adjustment sets if needed
```

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
