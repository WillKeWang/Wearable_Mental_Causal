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
pip install pandas numpy pyarrow scipy tqdm
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

# Get predefined configurations
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
- **rms**: Root mean square over time window
- **skew**: Skewness over time window

## Contributing

When adding new features:

1. Update `ProcessingConfig` with new parameters
2. Add corresponding processing logic in `WearablePreprocessor`
3. Update filename generation in `generate_filename()`
4. Add unit tests in `TestWearablePreprocessor`
5. Update documentation

## License

[Add your license information here]

# Causal Discovery System for Wearable Data

A modular, configurable system for causal discovery analysis of wearable sensor data with psychological survey responses, featuring bootstrap stability analysis and domain knowledge integration.

## Features

- **PC Algorithm Implementation**: Constraint-based causal discovery with statistical independence testing
- **Bootstrap Stability Analysis**: Robust edge stability assessment across multiple data samples
- **Domain Knowledge Integration**: Incorporate expert knowledge as constraints
- **Configurable Parameters**: Easy adjustment of statistical thresholds and analysis settings
- **Discovery/Validation Splits**: Proper separation for unbiased causal inference
- **Graph Visualization**: Automatic generation and saving of causal graphs
- **Comprehensive Logging**: Track analysis progress and edge stability
- **Edge Tracking**: Monitor stability of key relationships of interest
- **Multiple Output Formats**: CSV results, graph images, and analysis summaries

## Installation

### Required Dependencies

```bash
# Core dependencies
pip install pandas numpy matplotlib scikit-learn tqdm

# Causal discovery (main requirement)
pip install causal-learn

# Optional: Enhanced visualization
pip install graphviz pydot
```

### Directory Structure

```
results/
└── causal_discovery/
    ├── standard/
    ├── exploratory/
    ├── strict/
    └── custom/
        ├── *_bootstrap_stability.csv
        ├── *_pc_graph.png
        ├── *_edge_summary.csv
        └── *_data_summary.txt
```

## Quick Start

### Basic Causal Discovery

```python
from causal_discovery_utils import CausalDiscoveryConfig, run_causal_discovery
import pandas as pd

# Load your preprocessed data
df = pd.read_csv("data/preprocessed/survey_wearable_30d_before_to_0d_before_baseline_adj_full.csv")

# Standard configuration
config = CausalDiscoveryConfig(
    alpha=0.05,              # Statistical significance threshold
    n_bootstrap=100,         # Bootstrap iterations for stability
    bootstrap_sample_frac=0.6,  # Fraction of data per bootstrap
    discovery_split=0.5      # Train/validation split
)

# Run complete analysis
results = run_causal_discovery(df, config)

# Results contain discovered edges and stability analysis
print(f"Discovered {len(results['edges'])} causal relationships")
for edge, analysis in results['stability_analysis'].items():
    print(f"{edge}: {analysis['stability']:.1%} stable ({analysis['assessment']})")
```

### Key Edge Monitoring

```python
# Focus on specific relationships of interest
config = CausalDiscoveryConfig(
    key_edges=[
        ("rem_std", "promis_dep_sum"),      # REM sleep variability → Depression
        ("deep_std", "promis_dep_sum"),     # Deep sleep variability → Depression  
        ("hr_average_std", "promis_anx_sum"), # HR variability → Anxiety
        ("promis_anx_sum", "promis_dep_sum")  # Anxiety → Depression
    ]
)

results = run_causal_discovery(df, config)

# Get stability for key relationships
for edge_str, analysis in results['stability_analysis'].items():
    print(f"{edge_str}: {analysis['count']}/{analysis['total']} iterations "
          f"({analysis['stability']:.1%}) - {analysis['assessment']}")
```

## Configuration Options

### CausalDiscoveryConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.05 | Statistical significance for independence tests |
| `n_bootstrap` | int | 100 | Number of bootstrap iterations |
| `bootstrap_sample_frac` | float | 0.6 | Fraction of data per bootstrap sample |
| `discovery_split` | float | 0.5 | Fraction for discovery vs validation |
| `forbid_sensor_correlations` | bool | True | Prevent mean ↔ std edges for same sensor |
| `require_sensor_to_outcome` | bool | True | Enforce sensor → outcome directionality |
| `key_edges` | List[Tuple] | See config | Specific edges to track |
| `stability_thresholds` | Dict | See below | Thresholds for stability assessment |

### Stability Assessment Thresholds

```python
stability_thresholds = {
    "strong": 0.7,     # ≥70% bootstrap frequency
    "moderate": 0.5,   # ≥50% bootstrap frequency  
    "weak": 0.3        # ≥30% bootstrap frequency
}
# <30% = "unstable"
```

### Algorithm Parameters

- **Independence Test**: Fisher's Z-test (default) for continuous variables
- **PC Algorithm**: Stable version with orientation rules
- **Background Knowledge**: Domain constraints as forbidden/required edges

## Analysis Types

### 1. Standard Analysis

Default settings with domain knowledge constraints:

```python
config = CausalDiscoveryConfig()  # Use defaults
results = run_causal_discovery(df, config)
```

**Features:**
- 50/50 discovery/validation split
- Statistical significance α = 0.05
- 100 bootstrap iterations with 60% sampling
- Domain constraints (sensor → outcome, no mean ↔ std)

### 2. Exploratory Analysis

Liberal settings for hypothesis generation:

```python
config = CausalDiscoveryConfig(
    alpha=0.1,                          # More liberal
    forbid_sensor_correlations=False,   # Allow all connections
    require_sensor_to_outcome=False,    # Bidirectional exploration
    n_bootstrap=50
)
```

### 3. Strict Analysis

Conservative settings for robust relationships:

```python
config = CausalDiscoveryConfig(
    alpha=0.01,                         # Very strict
    n_bootstrap=200,                    # More iterations
    bootstrap_sample_frac=0.5,          # Smaller samples
    stability_thresholds={
        "strong": 0.8,                  # Higher thresholds
        "moderate": 0.6,
        "weak": 0.4
    }
)
```

### 4. Custom Edge Analysis

Focus on specific relationships:

```python
config = CausalDiscoveryConfig(
    key_edges=[
        ("sleep_metric", "depression_score"),
        ("hr_variability", "anxiety_score")
    ]
)
```

## Domain Knowledge Constraints

### Sensor Correlation Constraints

Prevents spurious correlations between mean and standard deviation of the same sensor:

```python
# Forbidden: hr_average_mean ↔ hr_average_std
# Forbidden: rem_mean ↔ rem_std
# etc.
```

### Temporal Precedence Constraints

Enforces causal direction based on temporal logic:

```python
# Required: sensor_metric → psychological_outcome
# Required: hr_average_mean → promis_dep_sum
# Required: sleep_efficiency → promis_anx_sum
# etc.
```

## Output Files

### File Naming Convention

Files are named descriptively based on configuration:

```
causal_discovery_alpha{alpha}_bs{n_bootstrap}_{type}.{ext}
```

**Examples:**
- `causal_discovery_alpha0.05_bs100_bootstrap_stability.csv`
- `causal_discovery_alpha0.01_bs200_strict_pc_graph.png`

### Bootstrap Stability Results

**File:** `*_bootstrap_stability.csv`

| Column | Description |
|--------|-------------|
| `from_var` | Source variable |
| `to_var` | Target variable |
| `frequency` | Proportion of bootstrap iterations with edge |
| `count` | Number of iterations with edge |
| `total_iterations` | Total successful bootstrap iterations |

### Edge Summary

**File:** `*_edge_summary.csv`

| Column | Description |
|--------|-------------|
| `from` | Source variable |
| `to` | Target variable |
| `type` | Edge type (directed/undirected) |
| `strength` | Edge strength from final model |

### Analysis Summary

**File:** `*_data_summary.txt`

Human-readable summary including:
- Configuration parameters
- Key edge stability results
- File locations
- Assessment classifications

## Advanced Usage

### Manual Step-by-Step Analysis

```python
from causal_discovery_utils import CausalDiscoveryAnalyzer

analyzer = CausalDiscoveryAnalyzer(config)

# Step 1: Prepare data
df_discovery, df_validation = analyzer.prepare_data(df)

# Step 2: Create data matrix  
X_discovery = analyzer.create_data_matrix(df_discovery)

# Step 3: Add domain knowledge
background_knowledge = analyzer.create_background_knowledge()

# Step 4: Run PC algorithm
causal_graph = analyzer.run_pc_algorithm(X_discovery, background_knowledge)

# Step 5: Bootstrap analysis
bootstrap_results = analyzer.bootstrap_analysis(X_discovery)

# Step 6: Analyze stability
stability_analysis = analyzer.analyze_stability(bootstrap_results)
```

### Batch Comparison

```python
configs = {
    "liberal": CausalDiscoveryConfig(alpha=0.1),
    "standard": CausalDiscoveryConfig(alpha=0.05), 
    "strict": CausalDiscoveryConfig(alpha=0.01)
}

results = {}
for name, config in configs.items():
    results[name] = run_causal_discovery(df, config)
    
# Compare edge counts and stability across configurations
```

### Custom Independence Tests

```python
# Future extension point for other independence tests
config = CausalDiscoveryConfig(
    independence_test="kci",  # Kernel-based conditional independence
    alpha=0.05
)
```

## Interpretation Guidelines

### Stability Assessment

| Stability | Interpretation | Action |
|-----------|----------------|---------|
| **Strong (≥70%)** | Robust relationship likely | Include in final model |
| **Moderate (50-69%)** | Potentially real, needs validation | Test on validation set |
| **Weak (30-49%)** | Uncertain relationship | Requires more data |
| **Unstable (<30%)** | Likely spurious | Exclude from model |

### Edge Types

- **Directed (→)**: Causal relationship with clear direction
- **Undirected (—)**: Association without clear causal direction
- **Absent**: No significant relationship detected

### Clinical Interpretation

**Example findings:**
- `rem_std → promis_dep_sum` (Strong, 80%): REM sleep variability predicts depression severity
- `hr_average_std → promis_anx_sum` (Moderate, 65%): Heart rate variability associated with anxiety
- `promis_anx_sum → promis_dep_sum` (Strong, 85%): Anxiety symptoms predict depression symptoms

## Validation Workflow

1. **Discovery Phase**: Run causal discovery on discovery set
2. **Stability Assessment**: Evaluate bootstrap stability of key edges
3. **Validation Phase**: Test stable relationships on held-out validation set
4. **Clinical Validation**: Confirm relationships align with domain knowledge
5. **Replication**: Test findings on independent datasets

## Troubleshooting

### Common Issues

**Empty Results:**
- Check data has sufficient samples and features
- Verify data contains target variables (`promis_dep_sum`, `promis_anx_sum`)
- Ensure features have meaningful variance

**Low Stability:**
- Increase bootstrap iterations (`n_bootstrap`)
- Adjust sample fraction (`bootstrap_sample_frac`)
- Check for overfitting with too many features

**Algorithm Failures:**
- Reduce alpha for less strict testing
- Check for perfect correlations or constant features
- Ensure data is properly normalized

**Memory Issues:**
- Limit maximum samples (`max_samples`)
- Reduce bootstrap iterations
- Process in smaller batches

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check intermediate results:
```python
# Access discovery data
print(f"Discovery shape: {analyzer.X_discovery.shape}")
print(f"Features: {analyzer.feature_names}")

# Check edge extraction
edges = analyzer.extract_edges(causal_graph.G, analyzer.feature_names)
print(f"Extracted {len(edges)} edges")
```

## Best Practices

### Data Preparation
- Use properly preprocessed wearable data with consistent time windows
- Ensure adequate sample size (>200 observations recommended)
- Check for missing data and outliers
- Consider standardization for mixed-scale variables

### Statistical Considerations
- Choose alpha based on multiple testing burden
- Use bootstrap for stability, not just single-run results  
- Validate key findings on independent data
- Consider effect sizes, not just statistical significance

### Domain Knowledge
- Include biologically plausible constraints
- Prevent obviously spurious relationships
- Allow for bidirectional psychological relationships
- Consider temporal precedence in constraint design

### Interpretation
- Focus on stable, replicable relationships
- Consider clinical significance of effect sizes
- Validate against existing literature
- Use results for hypothesis generation, not definitive causation

## Extensions

The modular design supports extensions:

- **Additional Independence Tests**: KCI, mutual information
- **Score-Based Methods**: GES, GIES for mixed data types
- **Constraint Integration**: Time-based, experimental constraints
- **Validation Methods**: Cross-validation, permutation testing
- **Visualization**: Interactive graph exploration
- **Effect Size Estimation**: Quantify relationship strengths

## References

- Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, prediction, and search.
- Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the presence of latent confounders.
- Zheng, X., et al. (2018). DAGs with NO TEARS: Continuous optimization for structure learning.

## License

[Add your license information here]