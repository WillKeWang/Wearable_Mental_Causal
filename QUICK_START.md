# Quick Start Guide - Temporal Causal Estimation

## Files Created

1. **`run_temporal_causal_estimation.py`** - Main analysis script
2. **`TEMPORAL_CAUSAL_ESTIMATION_README.md`** - Detailed documentation
3. **`QUICK_START.md`** - This file

## Quick Start (3 Steps)

### Step 1: Prepare Your Data

Your CSV file needs these columns:

**Required:**
- `pid` (participant ID)
- `promis_dep_sum_t`, `promis_anx_sum_t` (current time)
- `promis_dep_sum_tm1`, `promis_anx_sum_tm1` (lagged time)
- `rem_std_t`, `awake_std_t` (current time)
- `rmssd_std_tm1`, `awake_std_tm1`, `onset_latency_std_tm1` (lagged time)
- `onset_latency_std_t` (current time)
- `age_binned` (demographic)

### Step 2: Update Data Path

Edit line 617 in `run_temporal_causal_estimation.py`:

```python
DATA_PATH = "path/to/your/data.csv"  # <- Change this!
```

### Step 3: Run

```bash
python run_temporal_causal_estimation.py
```

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
→ Check your DATA_PATH

**"Column not found"**
→ Check your column names match exactly (with _t and _tm1 suffixes)

**Few matched pairs**
→ Increase caliper or reduce covariates

**Bootstrap failures**
→ Reduce sample_frac or check data quality

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
