#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized wearable data preprocessing utilities for TemPredict dataset.
Based on the original working logic with performance improvements.

Updates:
- Removed baseline subtraction (raw values preserved)
- Added minimum 10-day data requirement per survey window
- Added comprehensive data quality reporting
"""

import time
import gc
import warnings
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Silence common RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom")


@dataclass
class ProcessingConfig:
    """Configuration for wearable data preprocessing."""
    
    # Data paths
    oura_path: str = "data/raw/oura_all_sleep_summary_stacked_w_temp_max.parquet"
    survey_path: str = "data/raw/base_monthly.csv"
    output_dir: str = "data/preprocessed"
    
    # Time window configuration (in days)
    window_start_offset: int = -30  # Days since the survey date
    window_end_offset: int = 0      # Days since the survey date
    
    # Minimum data requirements
    min_days_per_window: int = 10  # Minimum valid days required per survey window
    
    # Processing options
    sample_size: Optional[int] = None  # None for full cohort
    random_seed: int = 42
    
    # Metrics to process
    metric_cols: List[str] = field(default_factory=lambda: [
        "hr_average", "rmssd", "breath_average", "hr_lowest",
        "breath_v_average", "temperature_deviation", 
        "temperature_trend_deviation", "temperature_max",
        "onset_latency", "efficiency", "deep", "light",
        "rem", "awake", "total"
    ])
    
    # Statistical aggregations to compute (simplified to mean and std only)
    stat_functions: List[str] = field(default_factory=lambda: ["mean", "std"])
    
    # Survey patterns for PROMIS scores
    depression_patterns: List[str] = field(default_factory=lambda: ["worthless", "helpless", "depressed", "hopeless"])
    anxiety_patterns: List[str] = field(default_factory=lambda: ["fearful", "anx", "worri", "uneasy"])
    
    # Constants
    lockdown_cutoff: str = "2020-03-22"
    excluded_pid: str = "hnLS_JCWE9qDW__KI4upzrwUsiqVdbEm7J8725U6H15XL9inSZqhb1b9QGsY0K7z744"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.window_start_offset >= self.window_end_offset:
            raise ValueError("window_start_offset must be less than window_end_offset")
        if self.min_days_per_window <= 0:
            raise ValueError("min_days_per_window must be positive")
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError("sample_size must be positive or None")
    
    @property
    def window_direction(self) -> str:
        """Get window direction label."""
        if self.window_start_offset < 0 and self.window_end_offset <= 0:
            return "backward"
        elif self.window_start_offset >= 0 and self.window_end_offset > 0:
            return "forward"
        else:
            return "mixed"
    
    @property
    def window_description(self) -> str:
        """Get human-readable window description."""
        start_desc = f"{abs(self.window_start_offset)}d_{'before' if self.window_start_offset < 0 else 'after'}"
        if self.window_end_offset <= 0:
            end_desc = f"{abs(self.window_end_offset)}d_before"
        else:
            end_desc = f"{abs(self.window_end_offset)}d_after"
        return f"{start_desc}_to_{end_desc}"
    
    def generate_filename(self, file_type: str) -> str:
        """Generate descriptive filename based on configuration."""
        sample_suffix = f"_n{self.sample_size}" if self.sample_size else "_full"
        
        if file_type == "main":
            return f"survey_wearable_{self.window_description}_min{self.min_days_per_window}d{sample_suffix}.csv"
        elif file_type == "report":
            return f"data_quality_report_{self.window_description}{sample_suffix}.txt"
        else:
            raise ValueError(f"Unknown file_type: {file_type}")


@dataclass
class DataQualityStats:
    """Container for data quality statistics."""
    total_surveys: int = 0
    surveys_with_sufficient_data: int = 0
    surveys_excluded_insufficient_data: int = 0
    total_participants: int = 0
    participants_with_data: int = 0
    
    # Per-metric missingness tracking
    metric_missingness: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Per-metric distribution tracking
    metric_distributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Days per window distribution
    days_per_window: List[int] = field(default_factory=list)


class WearablePreprocessor:
    """Main class for preprocessing wearable data with surveys."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize preprocessor with configuration."""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.df_oura = None
        self.df_survey = None
        self.dep_cols = []
        self.anx_cols = []
        
        # Initialize quality statistics
        self.quality_stats = DataQualityStats()
        
        # Raw data collection for quality report
        self._raw_window_data = []
        
        logger.info(f"Initialized WearablePreprocessor with config: {config}")
    
    def load_data(self) -> None:
        """Load Oura and survey data."""
        logger.info("Loading Oura data...")
        
        # Handle both parquet and CSV files
        if self.config.oura_path.endswith('.parquet'):
            self.df_oura = pq.read_table(self.config.oura_path).to_pandas(ignore_metadata=True)
        else:
            self.df_oura = pd.read_csv(self.config.oura_path)
        
        # CRITICAL FIX: Handle MultiIndex with pid and date
        if 'pid' not in self.df_oura.columns:
            if hasattr(self.df_oura.index, 'names') and 'pid' in self.df_oura.index.names:
                self.df_oura = self.df_oura.reset_index()
            elif self.df_oura.index.name == 'pid':
                self.df_oura = self.df_oura.reset_index()
        
        self.df_oura["date"] = pd.to_datetime(self.df_oura["date"], utc=True)
        
        logger.info(f"Loaded {len(self.df_oura)} Oura records")
        
        logger.info("Loading monthly survey data...")
        self.df_survey = (
            pd.read_csv(self.config.survey_path, index_col=0, low_memory=False)
            .rename(columns={"hashed_id": "pid"})
            .pipe(self._convert_dateutc)
        )
        
        # Remove excluded participant
        initial_count = len(self.df_survey)
        self.df_survey = self.df_survey.loc[self.df_survey["pid"] != self.config.excluded_pid]
        logger.info(f"Loaded {len(self.df_survey)} survey records (removed {initial_count - len(self.df_survey)} excluded)")
        
        # Find PROMIS columns
        self.dep_cols = self._match_columns(self.df_survey, self.config.depression_patterns)
        self.anx_cols = self._match_columns(self.df_survey, self.config.anxiety_patterns)
        
        if len(self.dep_cols) != 4 or len(self.anx_cols) != 4:
            raise ValueError(f"Expected 4 PROMIS columns each, found {len(self.dep_cols)} depression, {len(self.anx_cols)} anxiety")
        
        logger.info(f"Found PROMIS columns - Depression: {self.dep_cols}, Anxiety: {self.anx_cols}")
    
    def _convert_dateutc(self, df: pd.DataFrame, src="DateUTC.x", new="date", tz="UTC") -> pd.DataFrame:
        """Convert DateUTC column to proper datetime."""
        dt = pd.to_datetime(df[src], format="%m/%d/%Y %H:%M", errors="coerce")
        return df.assign(**{new: dt.dt.tz_localize(tz)}).drop(columns=[src])
    
    def _match_columns(self, df: pd.DataFrame, patterns: List[str]) -> List[str]:
        """Find columns matching any of the given patterns (case-insensitive)."""
        return sorted(c for c in df.columns if any(p in c.lower() for p in patterns))
    
    def get_sample_pids(self) -> List[str]:
        """Get list of PIDs to process based on sample_size."""
        all_pids = self.df_survey["pid"].unique().tolist()
        
        if self.config.sample_size is None:
            logger.info(f"Processing full cohort: {len(all_pids)} participants")
            return all_pids
        
        rng = np.random.default_rng(self.config.random_seed)
        sample_pids = rng.choice(all_pids, size=min(self.config.sample_size, len(all_pids)), replace=False).tolist()
        logger.info(f"Processing sample: {len(sample_pids)} participants")
        return sample_pids
    
    def extract_time_window(self, df_oura_pid: pd.DataFrame, survey_date: pd.Timestamp) -> pd.DataFrame:
        """Extract wearable data for specified time window around survey date."""
        start_date = survey_date + pd.DateOffset(days=self.config.window_start_offset)
        end_date = survey_date + pd.DateOffset(days=self.config.window_end_offset)
        
        return df_oura_pid[
            (df_oura_pid["date"] >= start_date) & 
            (df_oura_pid["date"] < end_date)
        ][self.config.metric_cols]
    
    def compute_statistics(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """Compute statistical aggregations for window data (mean and std only)."""
        arr = window_data.to_numpy(float, copy=False)
        
        if arr.size == 0:
            result = {}
            for col in self.config.metric_cols:
                for stat in self.config.stat_functions:
                    result[f"{col}_{stat}"] = np.nan
            return result
        
        # Compute statistics efficiently (only mean and std)
        with np.errstate(all="ignore"):
            means = np.nanmean(arr, axis=0)
            stds = np.nanstd(arr, axis=0, ddof=1)
        
        # Format results
        result = {}
        for j, col in enumerate(self.config.metric_cols):
            if "mean" in self.config.stat_functions:
                result[f"{col}_mean"] = means[j]
            if "std" in self.config.stat_functions:
                result[f"{col}_std"] = stds[j]
        
        return result
    
    def process_participant(self, pid: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process a single participant's data without baseline adjustment."""
        df_o = self.df_oura[self.df_oura["pid"] == pid].sort_values("date").copy()
        df_m = self.df_survey[self.df_survey["pid"] == pid].copy()
        
        if df_o.empty or df_m.empty:
            return None, []
        
        # Process each survey response
        survey_rows = []
        for _, survey_row in df_m.iterrows():
            self.quality_stats.total_surveys += 1
            
            # Extract time window data (NO baseline adjustment - raw values)
            window_data = self.extract_time_window(df_o, survey_row["date"])
            
            # Count valid days (rows with at least one non-NaN metric value)
            valid_days = window_data.dropna(how='all').shape[0]
            self.quality_stats.days_per_window.append(valid_days)
            
            # FILTER: Skip surveys with insufficient data
            if valid_days < self.config.min_days_per_window:
                self.quality_stats.surveys_excluded_insufficient_data += 1
                continue
            
            self.quality_stats.surveys_with_sufficient_data += 1
            
            # Collect raw data for quality report
            self._raw_window_data.append(window_data)
            
            # Compute statistics
            stats = self.compute_statistics(window_data)
            
            # Build survey row
            row_data = {
                "pid": pid,
                "date": survey_row["date"],
                "valid_days": valid_days,
                **stats
            }
            
            # Add PROMIS scores
            row_data["promis_dep_sum"] = pd.to_numeric(survey_row[self.dep_cols], errors="coerce").sum()
            row_data["promis_anx_sum"] = pd.to_numeric(survey_row[self.anx_cols], errors="coerce").sum()
            
            # Add demographic info if available
            demo_cols = [c for c in ("sex", "age", "gender", "race", "ethnicity_hispanic") if c in survey_row.index]
            for col in demo_cols:
                row_data[col] = survey_row[col]
            
            # Add lockdown indicator
            lockdown_cutoff = pd.Timestamp(self.config.lockdown_cutoff, tz="UTC")
            row_data["after_lockdown"] = int(survey_row["date"] >= lockdown_cutoff)
            
            survey_rows.append(row_data)
        
        return {"pid": pid}, survey_rows
    
    def _compute_quality_statistics(self) -> None:
        """Compute comprehensive data quality statistics from collected raw data."""
        if not self._raw_window_data:
            logger.warning("No raw data collected for quality statistics")
            return
        
        # Concatenate all window data
        all_data = pd.concat(self._raw_window_data, ignore_index=True)
        
        # Compute per-metric statistics
        for col in self.config.metric_cols:
            if col not in all_data.columns:
                continue
            
            col_data = all_data[col]
            
            # Missingness statistics
            total_obs = len(col_data)
            missing_obs = col_data.isna().sum()
            self.quality_stats.metric_missingness[col] = {
                "total_observations": total_obs,
                "missing_observations": missing_obs,
                "missing_percent": (missing_obs / total_obs * 100) if total_obs > 0 else 0.0
            }
            
            # Distribution statistics (on non-missing data)
            valid_data = col_data.dropna()
            if len(valid_data) > 0:
                self.quality_stats.metric_distributions[col] = {
                    "n": len(valid_data),
                    "mean": float(valid_data.mean()),
                    "std": float(valid_data.std()),
                    "min": float(valid_data.min()),
                    "p25": float(valid_data.quantile(0.25)),
                    "median": float(valid_data.median()),
                    "p75": float(valid_data.quantile(0.75)),
                    "max": float(valid_data.max()),
                    "skewness": float(valid_data.skew()),
                    "kurtosis": float(valid_data.kurtosis())
                }
            else:
                self.quality_stats.metric_distributions[col] = {
                    "n": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
                    "p25": np.nan, "median": np.nan, "p75": np.nan, "max": np.nan,
                    "skewness": np.nan, "kurtosis": np.nan
                }
    
    def _save_quality_report(self) -> str:
        """Generate and save comprehensive data quality report."""
        self._compute_quality_statistics()
        
        report_file = self.output_dir / self.config.generate_filename("report")
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA QUALITY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration summary
            f.write("-" * 80 + "\n")
            f.write("CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Time window: {self.config.window_start_offset} to {self.config.window_end_offset} days relative to survey\n")
            f.write(f"Window description: {self.config.window_description}\n")
            f.write(f"Minimum days required per window: {self.config.min_days_per_window}\n")
            f.write(f"Baseline adjustment: DISABLED (raw values preserved)\n")
            f.write(f"Sample size: {'Full cohort' if self.config.sample_size is None else self.config.sample_size}\n\n")
            
            # Survey-level statistics
            f.write("-" * 80 + "\n")
            f.write("SURVEY-LEVEL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total survey responses processed: {self.quality_stats.total_surveys}\n")
            f.write(f"Surveys with sufficient data (>={self.config.min_days_per_window} days): {self.quality_stats.surveys_with_sufficient_data}\n")
            f.write(f"Surveys excluded (insufficient data): {self.quality_stats.surveys_excluded_insufficient_data}\n")
            if self.quality_stats.total_surveys > 0:
                retention_rate = self.quality_stats.surveys_with_sufficient_data / self.quality_stats.total_surveys * 100
                f.write(f"Retention rate: {retention_rate:.1f}%\n")
            f.write(f"Total participants: {self.quality_stats.total_participants}\n")
            f.write(f"Participants with at least one valid survey: {self.quality_stats.participants_with_data}\n\n")
            
            # Days per window distribution
            f.write("-" * 80 + "\n")
            f.write("DAYS PER WINDOW DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            if self.quality_stats.days_per_window:
                days_arr = np.array(self.quality_stats.days_per_window)
                f.write(f"N windows: {len(days_arr)}\n")
                f.write(f"Mean days: {np.mean(days_arr):.1f}\n")
                f.write(f"Std days: {np.std(days_arr):.1f}\n")
                f.write(f"Min days: {np.min(days_arr)}\n")
                f.write(f"25th percentile: {np.percentile(days_arr, 25):.0f}\n")
                f.write(f"Median days: {np.median(days_arr):.0f}\n")
                f.write(f"75th percentile: {np.percentile(days_arr, 75):.0f}\n")
                f.write(f"Max days: {np.max(days_arr)}\n")
                
                # Histogram of days
                f.write("\nDays distribution histogram:\n")
                bins = [0, 5, 10, 15, 20, 25, 30, np.inf]
                bin_labels = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30+"]
                hist, _ = np.histogram(days_arr, bins=bins)
                for label, count in zip(bin_labels, hist):
                    pct = count / len(days_arr) * 100
                    bar = "#" * int(pct / 2)
                    f.write(f"  {label:>6} days: {count:>6} ({pct:>5.1f}%) {bar}\n")
            f.write("\n")
            
            # Per-metric missingness
            f.write("-" * 80 + "\n")
            f.write("METRIC-LEVEL MISSINGNESS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Metric':<30} {'Total Obs':>12} {'Missing':>12} {'Missing %':>12}\n")
            f.write("-" * 66 + "\n")
            for metric in self.config.metric_cols:
                if metric in self.quality_stats.metric_missingness:
                    stats = self.quality_stats.metric_missingness[metric]
                    f.write(f"{metric:<30} {stats['total_observations']:>12} {stats['missing_observations']:>12} {stats['missing_percent']:>11.2f}%\n")
            f.write("\n")
            
            # Per-metric distributions
            f.write("-" * 80 + "\n")
            f.write("METRIC DISTRIBUTIONS (RAW VALUES)\n")
            f.write("-" * 80 + "\n")
            
            for metric in self.config.metric_cols:
                if metric in self.quality_stats.metric_distributions:
                    stats = self.quality_stats.metric_distributions[metric]
                    f.write(f"\n{metric}:\n")
                    f.write(f"  N (valid):    {stats['n']}\n")
                    f.write(f"  Mean:         {stats['mean']:.4f}\n")
                    f.write(f"  Std:          {stats['std']:.4f}\n")
                    f.write(f"  Min:          {stats['min']:.4f}\n")
                    f.write(f"  25th pctl:    {stats['p25']:.4f}\n")
                    f.write(f"  Median:       {stats['median']:.4f}\n")
                    f.write(f"  75th pctl:    {stats['p75']:.4f}\n")
                    f.write(f"  Max:          {stats['max']:.4f}\n")
                    f.write(f"  Skewness:     {stats['skewness']:.4f}\n")
                    f.write(f"  Kurtosis:     {stats['kurtosis']:.4f}\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Saved data quality report to {report_file}")
        return str(report_file)
    
    def process_all(self) -> Tuple[str, str]:
        """Process all participants and save results."""
        if self.df_oura is None or self.df_survey is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get sample PIDs
        sample_pids = self.get_sample_pids()
        self.quality_stats.total_participants = len(sample_pids)
        
        # Initialize output file
        main_output = self.output_dir / self.config.generate_filename("main")
        main_output.unlink(missing_ok=True)
        
        participants_with_data = 0
        first_row = True
        
        logger.info(f"Processing {len(sample_pids)} participants...")
        logger.info(f"Minimum days required per survey window: {self.config.min_days_per_window}")
        logger.info("Baseline adjustment: DISABLED (using raw values)")
        start_time = time.perf_counter()
        
        for pid in tqdm(sample_pids, desc="Processing participants"):
            try:
                _, survey_rows = self.process_participant(pid)
                
                if survey_rows:
                    participants_with_data += 1
                    
                    # Convert to DataFrame and save
                    df_surveys = pd.DataFrame(survey_rows)
                    
                    # Reorder columns for better readability
                    demo_cols = [c for c in df_surveys.columns if c in ("sex", "age", "gender", "race", "ethnicity_hispanic")]
                    lead_cols = ["pid", "date", "valid_days"] + demo_cols + ["promis_dep_sum", "promis_anx_sum", "after_lockdown"]
                    stat_cols = [c for c in df_surveys.columns if any(s in c for s in self.config.stat_functions)]
                    final_cols = lead_cols + stat_cols
                    
                    df_surveys = df_surveys[[c for c in final_cols if c in df_surveys.columns]]
                    
                    # Stream append to main file
                    df_surveys.to_csv(main_output, mode="w" if first_row else "a", 
                                    header=first_row, index=False)
                    first_row = False
                
            except Exception as e:
                logger.error(f"Error processing PID {pid}: {str(e)}")
                continue
            
            gc.collect()
        
        self.quality_stats.participants_with_data = participants_with_data
        
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Processing complete in {elapsed_time:.1f}s")
        logger.info(f"Main output: {main_output}")
        
        # Generate and save data quality report
        report_file = self._save_quality_report()
        
        # Clear raw data to free memory
        self._raw_window_data = []
        gc.collect()
        
        return str(main_output), report_file


def run_preprocessing(config: ProcessingConfig) -> Tuple[str, str]:
    """Convenience function to run preprocessing with given configuration."""
    preprocessor = WearablePreprocessor(config)
    preprocessor.load_data()
    return preprocessor.process_all()


def get_example_configs() -> Dict[str, ProcessingConfig]:
    """Get example configurations for common use cases."""
    return {
        "backward_4w": ProcessingConfig(
            window_start_offset=-28,  # 4 weeks before
            window_end_offset=0,
            min_days_per_window=10
        ),
        "backward_6w_to_2w": ProcessingConfig(
            window_start_offset=-42,  # 6 weeks
            window_end_offset=-14,    # 2 weeks
            min_days_per_window=10
        ),
        "forward_1w_to_5w": ProcessingConfig(
            window_start_offset=7,    # 1 week
            window_end_offset=35,     # 5 weeks
            min_days_per_window=10
        ),
        "small_sample": ProcessingConfig(
            window_start_offset=-28,
            window_end_offset=0,
            min_days_per_window=10,
            sample_size=100
        )
    }


if __name__ == "__main__":
    # Example usage
    print("Example configurations:")
    configs = get_example_configs()
    
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  Window: {config.window_description}")
        print(f"  Min days: {config.min_days_per_window}")
        print(f"  Output: {config.generate_filename('main')}")