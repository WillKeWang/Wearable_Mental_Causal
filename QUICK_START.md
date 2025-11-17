# Quick Start Guide - Temporal Causal Estimation

## Files Created

1. **`run_temporal_causal_estimation.py`** - Main analysis script
2. **`TEMPORAL_CAUSAL_ESTIMATION_README.md`** - Detailed documentation
3. **`QUICK_START.md`** - This file

## Quick Start (2 Steps)

### Step 1: Check Your Data

Your CSV file should have these columns:

**Required:**
- `pid` (participant ID)
- `date` (survey date for creating temporal pairs)
- `promis_dep_sum`, `promis_anx_sum` (survey variables)
- `rem_std`, `awake_std`, `onset_latency_std`, `rmssd_std` (wearable metrics)
- `age`, `sex`, `race`, `ethnicity_hispanic` (demographics, optional)

**Default path** (already configured):
```
data/preprocessed/full_run/4w_to_0w_before/survey_wearable_28d_before_to_0d_before_baseline_adj_full.csv
```

**Note**: The script automatically creates `_t` and `_tm1` versions from consecutive surveys!

### Step 2: Run

```bash
python run_temporal_causal_estimation.py
```

The script will:
1. Load your data
2. Create temporal pairs (consecutive surveys 21-35 days apart)
3. Run causal estimation for all three relationships
4. Generate visualizations

## What You'll Get

### Terminal Output
- Detailed statistics for each relationship
- Covariate balance checks
- Bootstrap progress bars
- Summary tables with coefficients and effect sizes

### Files Created
- `causal_estimation_relationship_1.png` (Depression → REM)
- `causal_estimation_relationship_2.png` (Awake → Depression)
- `causal_estimation_relationship_3.png` (Depression → Awake)

## The Three Relationships

### 1. Depression (t) → REM Variability (t)
```
Exposure: promis_dep_sum_t
Outcome: rem_std_t
Adjustment Sets: 3 different sets tested
```

### 2. Awake Variability (t) → Depression (t)
```
Exposure: awake_std_t
Outcome: promis_dep_sum_t
Adjustment Sets: 2 different sets tested
```

### 3. Depression (t) → Awake Variability (t)
```
Exposure: promis_dep_sum_t
Outcome: awake_std_t
Adjustment Sets: 2 different sets tested
```

## Key Features

✅ **Propensity Score Matching** - Fast sklearn implementation
✅ **Prevents Same-PID Matching** - For valid between-subject effects
✅ **Linear Mixed Effects Models** - Accounts for repeated measures
✅ **Pair-Level Bootstrapping** - Preserves matched structure
✅ **Multiple Adjustment Sets** - Tests sensitivity to confounders
✅ **Automatic Visualizations** - Comparison plots for all sets

## Common Issues

**"File not found"**
→ Check the DATA_PATH variable (line 531)

**"No valid pairs found"**
→ Check if participants have at least 2 surveys 21-35 days apart
→ Adjust MIN_DAYS_GAP and MAX_DAYS_GAP if needed

**"Column not found"**
→ Ensure raw data has `promis_dep_sum`, `promis_anx_sum`, wearable metrics ending in `_std`
→ The script will automatically add `_t` and `_tm1` suffixes

**Few matched pairs**
→ Increase caliper (e.g., 0.2 instead of 0.1)
→ Reduce number of covariates in adjustment sets

**Bootstrap failures**
→ Reduce sample_frac or check data quality
→ Ensure sufficient matched pairs (at least 20-30)

## Next Steps

1. Run the script with your data
2. Check covariate balance (SMD < 0.25)
3. Review coefficient estimates and CIs
4. Compare results across adjustment sets
5. Examine visualizations

## Need Help?

See `TEMPORAL_CAUSAL_ESTIMATION_README.md` for:
- Detailed methodology
- Interpretation guidelines
- Troubleshooting tips
- Parameter tuning options
