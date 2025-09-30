# Wearable Data Preprocessing System

A modular, configurable system for preprocessing Oura wearable data with survey responses for longitudinal analysis.

## Features

- **Configurable time windows**: Easily specify any time window relative to survey dates
- **Flexible baseline adjustment**: Enable/disable per-participant baseline adjustment
- **Multiple statistical aggregations**: Mean, standard deviation, RMS, skewness
- **Descriptive file naming**: Output files reflect configuration parameters
- **Comprehensive logging**: Track processing progress and issues
- **Unit testing**: Built-in tests for key functions
- **Memory efficient**: Streaming output for large datasets
- **PROMIS scoring**: Automatic depression and anxiety sum scores

## Installation

1. Ensure you have the required dependencies:
```bash
mamba create -n causal_analysis python=3.10 --file requirements.txt
```

2. Set up your data directory structure:
```
data/
├── raw/
│   ├── oura_all_sleep_summary_stacked_w_temp_max.parquet
│   └── base_monthly.csv
└── preprocessed/
    └── (output files will be saved here)
```

## Quick Start

### Basic Usage

```python
from data_utils import ProcessingConfig, run_preprocessing

# Standard backward-looking analysis (30 days before survey)
config = ProcessingConfig(
    window_start_offset=-30,  # 30 days before survey
    window_end_offset=0,      # up to survey date
    baseline_enabled=True
)

main_file, baseline_file = run_preprocessing(config)
print(f"Results saved to: {main_file}")
```

### Common Configurations

```python
from data_utils import get_example_configs

# Get some predefined configurations, does not necessarily have to follow
configs = get_example_configs()

# Available configurations:
# - backward_30d: Standard 30-day backward analysis
# - forward_30d: 30-day forward-looking analysis  
# - backward_6w_to_2w: 6 weeks to 2 weeks before survey
# - forward_1w_to_5w: 1 week to 5 weeks after survey
# - no_baseline: Analysis without baseline adjustment
# - small_sample: 100-participant sample

# Run any configuration
main_file, baseline_file = run_preprocessing(configs['backward_6w_to_2w'])
```

## Configuration Options

### Time Windows

The `ProcessingConfig` class allows you to specify any time window relative to survey dates:

```python
config = ProcessingConfig(
    window_start_offset=-42,  # Start 6 weeks before survey
    window_end_offset=-14,    # End 2 weeks before survey
    # This analyzes wearable data from 6-2 weeks before each survey
)
```

**Common patterns:**
- Backward analysis: `start < 0, end <= 0`
- Forward analysis: `start >= 0, end > 0`  
- Mixed window: `start < 0, end > 0`

### Baseline Adjustment

```python
# With baseline adjustment (default)
config = ProcessingConfig(
    baseline_enabled=True,
    baseline_days=30,  # Use first 30 days as baseline
    # Wearable metrics will be adjusted by subtracting individual baseline means
)

# Without baseline adjustment
config = ProcessingConfig(
    baseline_enabled=False,
    # Use raw wearable values
)
```

### Sample Size

```python
# Full cohort
config = ProcessingConfig(sample_size=None)

# Specific sample size
config = ProcessingConfig(sample_size=500)
```

### Custom Metrics and Statistics

```python
config = ProcessingConfig(
    metric_cols=["hr_average", "rmssd", "efficiency"],  # Subset of metrics
    stat_functions=["mean", "std", "rms"],              # Subset of statistics
)
```

## Output Files

The system generates descriptively named files based on your configuration:

### Naming Convention

**Main dataset**: `survey_wearable_{window_description}_{baseline}_{sample}.csv`

Examples:
- `survey_wearable_30d_before_to_0d_before_baseline_adj_full.csv`
- `survey_wearable_0d_after_to_30d_after_baseline_adj_n500.csv`
- `survey_wearable_42d_before_to_14d_before_no_baseline_full.csv`

**Baseline dataset**: `baseline_metrics_{baseline_days}d_{sample}.csv`

### File Contents

**Main dataset columns:**
- `pid`: Participant ID
- `date`: Survey completion date
- `sex`, `age`, `gender`, `race`, `ethnicity_hispanic`: Demographics (if available)
- `promis_dep_sum`: PROMIS depression sum score
- `promis_anx_sum`: PROMIS anxiety sum score  
- `after_lockdown`: Binary indicator for post-lockdown surveys
- `{metric}_{stat}`: Statistical aggregations of wearable metrics

**Baseline dataset columns:**
- `pid`: Participant ID
- `{metric}_baseline_mean`: Baseline mean for each metric

## Advanced Usage

### Manual Processing

```python
from data_utils import WearablePreprocessor, ProcessingConfig

# Create configuration
config = ProcessingConfig(
    window_start_offset=-14,
    window_end_offset=7,  # Mixed window: 2 weeks before to 1 week after
    baseline_enabled=True,
    sample_size=100
)

# Manual step-by-step processing
preprocessor = WearablePreprocessor(config)
preprocessor.load_data()

# Get participant list
sample_pids = preprocessor.get_sample_pids()
print(f"Processing {len(sample_pids)} participants")

# Process single participant
baseline_row, survey_rows = preprocessor.process_participant(sample_pids[0])

# Process all participants
main_file, baseline_file = preprocessor.process_all()
```

### Batch Processing

```python
from data_utils import ProcessingConfig, run_preprocessing

# Define multiple configurations
configs = {
    "short_term": ProcessingConfig(window_start_offset=-7, window_end_offset=0),
    "medium_term": ProcessingConfig(window_start_offset=-30, window_end_offset=0),
    "long_term": ProcessingConfig(window_start_offset=-90, window_end_offset=0),
}

# Process all configurations
results = {}
for name, config in configs.items():
    try:
        main_file, baseline_file = run_preprocessing(config)
        results[name] = {"main": main_file, "baseline": baseline_file}
        print(f"✅ {name}: {main_file}")
    except Exception as e:
        print(f"❌ {name}: {e}")
```

## Testing

Run the built-in unit tests:

```python
import unittest
from data_utils import TestWearablePreprocessor

# Run all tests
unittest.main(TestWearablePreprocessor, verbosity=2)
```

## Examples

See `example_usage.py` for comprehensive examples including:

1. Standard backward-looking analysis
2. Forward-looking analysis  
3. Custom time windows
4. Analysis without baseline adjustment
5. Manual step-by-step processing
6. Batch processing multiple configurations

Run examples:
```bash
python example_usage.py
```

## Troubleshooting

### Common Issues

1. **File not found errors**: Ensure data files are in `data/raw/` directory
2. **Memory issues**: Reduce `sample_size` or process in smaller batches
3. **Empty results**: Check that time windows contain data for your participants

### Logging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Data Validation

The system includes automatic validation for:
- Configuration parameters
- Required PROMIS columns (4 depression, 4 anxiety items)
- Data types and date formats
- Time window consistency

## Configuration Reference

### ProcessingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `oura_path` | str | `"data/raw/oura_all_sleep_summary_stacked_w_temp_max.parquet"` | Path to Oura data |
| `survey_path` | str | `"data/raw/base_monthly.csv"` | Path to survey data |
| `output_dir` | str | `"data/preprocessed"` | Output directory |
| `window_start_offset` | int | `-30` | Days before/after survey (start) |
| `window_end_offset` | int | `0` | Days before/after survey (end) |
| `baseline_enabled` | bool | `True` | Enable baseline adjustment |
| `baseline_days` | int | `30` | Baseline period length |
| `sample_size` | int\|None | `None` | Number of participants (None = all) |
| `metric_cols` | List[str] | 15 wearable metrics | Metrics to process |
| `stat_functions` | List[str] | `["mean", "std", "rms", "skew"]` | Statistics to compute |

### Wearable Metrics

Default metrics processed:
- **Heart rate**: `hr_average`, `hr_lowest`
- **Heart rate variability**: `rmssd`
- **Breathing**: `breath_average`, `breath_v_average`
- **Temperature**: `temperature_deviation`, `temperature_trend_deviation`, `temperature_max`
- **Sleep stages**: `deep`, `light`, `rem`, `awake`, `total`
- **Sleep quality**: `onset_latency`, `efficiency`

### Statistical Aggregations

- **mean**: Average value over time window
- **std**: Standard deviation over time window  

## Contributing

When adding new features:

1. Update `ProcessingConfig` with new parameters
2. Add corresponding processing logic in `WearablePreprocessor`
3. Update filename generation in `generate_filename()`
4. Add unit tests in `TestWearablePreprocessor`
5. Update documentation

## License

[Add your license information here]

# Causal Discovery with PC Algorithm

Simple, modular system for causal discovery analysis on wearable sensor and psychological survey data using the PC algorithm with bootstrap stability assessment.

## Features

- PC algorithm implementation with Fisher's Z independence test
- Bootstrap stability analysis for robust edge detection
- Domain knowledge constraints (sensor→outcome directionality, no mean↔std correlations)
- Multi-dataset comparison
- Key edge tracking across datasets

## Installation

```bash
pip install pandas numpy causal-learn tqdm
```

## Quick Start

```python
from causal_discovery import run_pc_analysis

# Define your datasets
dataset_paths = {
    "dataset_1": "data/survey_wearable_data_1.csv",
    "dataset_2": "data/survey_wearable_data_2.csv"
}

# Run analysis
results = run_pc_analysis(
    dataset_paths,
    n_bootstrap=100,      # Bootstrap iterations
    sample_frac=0.6,      # 60% sampling per iteration
    alpha=0.05            # Significance level
)
```

## Main Functions

### Complete Analysis

```python
run_pc_analysis(dataset_paths, n_bootstrap=100, sample_frac=0.6, alpha=0.05)
```
Runs full analysis across multiple datasets and compares results.

### Single Dataset Analysis

```python
from causal_discovery import analyze_single_dataset

results = analyze_single_dataset(
    filepath="data/survey_wearable_data.csv",
    dataset_name="my_dataset",
    n_bootstrap=100
)
```

### Component Functions

```python
from causal_discovery import (
    load_and_prepare_data,      # Load and clean dataset
    prepare_variables,           # Select features and create data matrix
    create_background_knowledge, # Add domain constraints
    bootstrap_pc_analysis        # Run bootstrap analysis
)
```

## Data Requirements

Your CSV files should contain:
- **Required columns**: `promis_dep_sum`, `promis_anx_sum`, `pid`
- **Sensor features**: Variables ending in `_mean` or `_std`
- The system automatically filters invalid depression/anxiety scores (keeps 4-20 range)

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_bootstrap` | 100 | Number of bootstrap iterations |
| `sample_frac` | 0.6 | Fraction of data sampled per iteration |
| `alpha` | 0.05 | Significance level for independence tests |

## Domain Constraints

The system automatically applies two constraint types:

1. **No mean↔std correlations**: Prevents `sensor_mean ↔ sensor_std` edges for the same sensor
2. **Directional enforcement**: Requires `sensor → outcome` direction (sensors can only cause outcomes, not vice versa)

## Output Interpretation

### Stability Assessment

Results show how often each edge appears across bootstrap iterations:

```
rem_std → promis_dep_sum: 82/100 (82.0%) - STRONG
deep_std → promis_dep_sum: 67/100 (67.0%) - MODERATE
promis_anx_sum ↔ promis_dep_sum: 91/100 (91.0%) - STRONG
```

**Stability levels:**
- **STRONG** (≥70%): Robust relationship
- **MODERATE** (50-69%): Potentially real, needs validation
- **WEAK** (30-49%): Uncertain
- **UNSTABLE** (<30%): Likely spurious

### Consistency Across Datasets

When comparing multiple datasets:

```
Consistency Assessment:
  rem_std → depression: CONSISTENT across datasets
    dataset_1: 82% (STRONG)
    dataset_2: 79% (STRONG)
```

**Consistency criteria:**
- **HIGH**: Stability difference < 20%
- **MODERATE**: Stability difference 20-40%
- **LOW**: Stability difference > 40%

## Example: Testing the Code

```python
from test_causal_discovery import run_all_tests, main

# Run quick tests (10 bootstrap iterations)
run_all_tests()

# Run full analysis (100 iterations)
main()
```

## Customizing Key Edges

Track specific relationships of interest:

```python
# Default key edges tracked:
key_edges = [
    ("rem_std", "promis_dep_sum"),
    ("deep_std", "promis_dep_sum"),
    ("promis_anx_sum", "promis_dep_sum")
]

# The bootstrap_pc_analysis function counts how often these appear
```

## How It Works

1. **Data Loading**: Loads CSV, removes missing values, filters valid score ranges
2. **Variable Selection**: Selects depression/anxiety scores + all sensor `_mean` and `_std` features
3. **Background Knowledge**: Creates domain constraints
4. **Bootstrap Loop**: For each iteration:
   - Sample 60% of data randomly
   - Run PC algorithm with constraints
   - Track which edges appear
5. **Stability Analysis**: Calculate frequency of edge appearance across iterations
6. **Comparison**: Compare stability across datasets

## Tips

**Increasing Robustness:**
- Increase `n_bootstrap` (200+ for more stable estimates)
- Decrease `sample_frac` (0.5 for harder test)
- Decrease `alpha` (0.01 for stricter significance)

**Faster Testing:**
- Decrease `n_bootstrap` (10-20 for quick tests)
- Increase `sample_frac` (0.8 for easier convergence)

**Finding More Edges:**
- Increase `alpha` (0.1 for exploratory analysis)
- Note: This may increase false positives

## Common Issues

**"File not found"**: Check file paths match your directory structure

**"No successful iterations"**: Your data may be too small or have insufficient variance

**Low stability**: Normal for weak relationships - focus on edges with >50% stability