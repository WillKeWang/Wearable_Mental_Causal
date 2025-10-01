# Wearable Data Preprocessing System

Modular system for preprocessing Oura wearable data with survey responses for longitudinal analysis.

## Installation

```bash
mamba create -n causal_analysis python=3.10 --file requirements.txt
```

Data structure:
```
data/
├── raw/
│   ├── oura_all_sleep_summary_stacked_w_temp_max.parquet
│   └── base_monthly.csv
└── preprocessed/
```

## Quick Start

```python
from data_utils import ProcessingConfig, run_preprocessing

# Standard: 30 days before survey with baseline adjustment
config = ProcessingConfig(
    window_start_offset=-30,  # Days before survey (negative = past)
    window_end_offset=0,      # Survey date
    baseline_enabled=True     # Adjust by participant baseline
)

main_file, baseline_file = run_preprocessing(config)
```

## Time Window Configuration

```python
# Backward: 6 weeks to 2 weeks before survey
config = ProcessingConfig(
    window_start_offset=-42,
    window_end_offset=-14
)

# Forward: 1 week to 5 weeks after survey  
config = ProcessingConfig(
    window_start_offset=7,
    window_end_offset=35
)

# Mixed: 2 weeks before to 1 week after
config = ProcessingConfig(
    window_start_offset=-14,
    window_end_offset=7
)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_start_offset` | -30 | Start of window (days from survey, negative = before) |
| `window_end_offset` | 0 | End of window (days from survey, positive = after) |
| `baseline_enabled` | True | Adjust metrics by subtracting participant's baseline |
| `baseline_days` | 30 | Length of baseline period (first N days of data) |
| `sample_size` | None | Number of participants (None = all) |
| `stat_functions` | ["mean", "std", "rms", "skew"] | Statistics to compute |

## Wearable Features

### Heart Rate & Variability
- `hr_average`: Mean heart rate (bpm)
- `hr_lowest`: Lowest heart rate (bpm)
- `rmssd`: Root mean square of successive differences (HRV measure, ms)

### Respiratory
- `breath_average`: Average breaths per minute
- `breath_v_average`: Average breath variability

### Temperature
- `temperature_deviation`: Deviation from predicted temperature (°C)
- `temperature_trend_deviation`: Trend deviation from baseline (°C)
- `temperature_max`: Maximum daily temperature (°C)

### Sleep Stages (minutes)
- `deep`: Deep sleep duration
- `light`: Light sleep duration
- `rem`: REM sleep duration
- `awake`: Awake time during sleep period
- `total`: Total sleep duration

### Sleep Quality
- `onset_latency`: Time to fall asleep (minutes)
- `efficiency`: Sleep efficiency percentage (0-100)

**All features are aggregated using:** mean, standard deviation (std), RMS, and skewness over the specified time window.

## Output Files

### Naming Convention
`survey_wearable_{window}_{baseline}_{sample}.csv`

Examples:
- `survey_wearable_30d_before_to_0d_before_baseline_adj_full.csv`
- `survey_wearable_7d_after_to_35d_after_baseline_adj_n500.csv`
- `survey_wearable_42d_before_to_14d_before_no_baseline_full.csv`

### Columns
- **IDs**: `pid`, `date`
- **Demographics**: `sex`, `age`, `gender`, `race`, `ethnicity_hispanic`
- **Survey scores**: `promis_dep_sum`, `promis_anx_sum`
- **Temporal**: `after_lockdown`
- **Wearable features**: `{metric}_{stat}` (e.g., `hr_average_mean`, `rmssd_std`)

## Batch Processing

```python
from data_utils import ProcessingConfig, run_preprocessing

configs = {
    "6w_to_2w_before": ProcessingConfig(window_start_offset=-42, window_end_offset=-14),
    "1w_to_5w_after": ProcessingConfig(window_start_offset=7, window_end_offset=35),
    "no_baseline": ProcessingConfig(baseline_enabled=False)
}

for name, config in configs.items():
    main_file, baseline_file = run_preprocessing(config)
    print(f"✅ {name}: {main_file}")
```

## Testing

```python
from example_usage import run_all_examples
run_all_examples()
```

## Common Issues

- **Empty results**: Check time windows contain data for participants
- **Memory issues**: Reduce `sample_size`
- **Missing files**: Ensure data in `data/raw/` directory
- **Baseline adjustment**: Participants need ≥`baseline_days` of data









# Causal Discovery with PC Algorithm

Causal discovery analysis on wearable sensor and psychological survey data using the PC algorithm with bootstrap stability assessment.

## Installation

```bash
pip install pandas numpy causal-learn tqdm
```

## Quick Start

```python
from causal_discovery import run_temporal_pc_analysis

dataset_paths = {
    "dataset_1": "data/survey_wearable_data_1.csv",
    "dataset_2": "data/survey_wearable_data_2.csv"
}

# Recommended: PID-level bootstrapping on baseline surveys
results = run_temporal_pc_analysis(
    dataset_paths,
    n_bootstrap=100,
    sample_frac=0.6,
    alpha=0.05,
    use_pid_bootstrap=True,  # Sample participants, not rows
    use_baseline=True,       # Use earliest survey per participant
    min_frequency=0.1        # Show edges in ≥10% of iterations
)
```

## Data Requirements

CSV files must contain:
- `promis_dep_sum`, `promis_anx_sum`, `pid`
- Sensor features ending in `_mean` or `_std`
- Valid depression/anxiety scores (4-20 range)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_bootstrap` | 100 | Bootstrap iterations |
| `sample_frac` | 0.6 | Fraction of participants sampled per iteration |
| `alpha` | 0.05 | Significance level |
| `use_pid_bootstrap` | True | Sample participants (not rows) - **recommended for longitudinal data** |
| `use_baseline` | True | Use only baseline surveys per participant |
| `min_frequency` | 0.1 | Minimum frequency to display edges |

## Understanding Results

### Edge Report Format

```
1w_to_5w_after:
  ======================================================================
  EDGES TO DEPRESSION/ANXIETY (appeared in ≥10% of iterations):
  ======================================================================
  ★ rem_std → promis_dep_sum: 82/100 (82.0%)
  ★ deep_std → promis_dep_sum: 67/100 (67.0%)
  ★ promis_anx_sum → promis_dep_sum: 91/100 (91.0%)

  ======================================================================
  OTHER EDGES (appeared in ≥10% of iterations):
  ======================================================================
    rem_mean — deep_mean: 45/100 (45.0%)
```

### Stability Levels
- **≥70%**: Strong, robust relationship
- **50-69%**: Moderate, needs validation  
- **<50%**: Weak or uncertain

## Why PID-Level Bootstrapping?

**For longitudinal data with multiple observations per participant:**
- Samples participants, not rows → statistically independent samples
- Uses baseline surveys → cleaner causal interpretation
- Avoids pseudo-replication from repeated measures

**Switch to row-level if you have truly cross-sectional data:**
```python
results = run_temporal_pc_analysis(dataset_paths, use_pid_bootstrap=False)
```

## Testing

```python
from test_causal_discovery import run_all_tests, compare_methods, main

run_all_tests()      # Quick test (10 iterations)
compare_methods()    # Compare PID-level vs row-level
main()              # Full analysis (100 iterations)
```

## Common Issues

- **Low stability (<50%)**: Normal for weak relationships
- **Too many edges**: Increase `min_frequency` (e.g., 0.2)
- **Too few edges**: Decrease `min_frequency` or increase `alpha`
- **No successful iterations**: Data too small or insufficient variance