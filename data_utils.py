#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized wearable data preprocessing utilities for TemPredict dataset.
Based on the original working logic with performance improvements.
"""

import time
import gc
import warnings
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

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
    
    # Baseline configuration
    baseline_enabled: bool = True
    baseline_days: int = 30
    
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
    
    # Statistical aggregations to compute (only mean and std)
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
        if self.baseline_days <= 0:
            raise ValueError("baseline_days must be positive")
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
        # Fix: window_end_offset=0 means "up to survey date" which is "before"
        if self.window_end_offset <= 0:
            end_desc = f"{abs(self.window_end_offset)}d_before"
        else:
            end_desc = f"{abs(self.window_end_offset)}d_after"
        return f"{start_desc}_to_{end_desc}"
    
    def generate_filename(self, file_type: str) -> str:
        """Generate descriptive filename based on configuration."""
        baseline_suffix = "_baseline_adj" if self.baseline_enabled else "_no_baseline"
        sample_suffix = f"_n{self.sample_size}" if self.sample_size else "_full"
        
        if file_type == "main":
            return f"survey_wearable_{self.window_description}{baseline_suffix}{sample_suffix}.csv"
        elif file_type == "baseline":
            return f"baseline_metrics_{self.baseline_days}d{sample_suffix}.csv"
        else:
            raise ValueError(f"Unknown file_type: {file_type}")


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
            # Check if pid is in the index (could be MultiIndex)
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
    
    def calculate_baseline_metrics(self, pid: str, df_oura_pid: pd.DataFrame) -> Dict[str, float]:
        """Calculate baseline metrics for a participant."""
        if not self.config.baseline_enabled:
            return {}
        
        base_end = df_oura_pid["date"].iloc[0] + pd.Timedelta(days=self.config.baseline_days - 1)
        baseline_data = df_oura_pid[df_oura_pid["date"] <= base_end][self.config.metric_cols]
        
        if baseline_data.empty:
            logger.warning(f"No baseline data for PID {pid}")
            return {f"{c}_baseline_mean": np.nan for c in self.config.metric_cols}
        
        baseline_means = baseline_data.mean()
        return {f"{c}_baseline_mean": baseline_means[c] for c in self.config.metric_cols}
    
    def apply_baseline_adjustment(self, df_oura_pid: pd.DataFrame, baseline_means: Dict[str, float]) -> pd.DataFrame:
        """Apply baseline adjustment to Oura data."""
        if not self.config.baseline_enabled or not baseline_means:
            return df_oura_pid.copy()
        
        df_adjusted = df_oura_pid.copy()
        for col in self.config.metric_cols:
            baseline_key = f"{col}_baseline_mean"
            if baseline_key in baseline_means and not np.isnan(baseline_means[baseline_key]):
                df_adjusted[col] = df_adjusted[col] - baseline_means[baseline_key]
        
        return df_adjusted
    
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
            # Return NaN for all statistics if no data
            result = {}
            for col in self.config.metric_cols:
                result[f"{col}_mean"] = np.nan
                result[f"{col}_std"] = np.nan
            return result
        
        # Compute statistics (only mean and std)
        with np.errstate(all="ignore"):
            means = np.nanmean(arr, axis=0)
            stds = np.nanstd(arr, axis=0, ddof=1)
        
        # Format results
        result = {}
        for j, col in enumerate(self.config.metric_cols):
            result[f"{col}_mean"] = means[j]
            result[f"{col}_std"] = stds[j]
        
        return result
    
    def process_participant(self, pid: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Process a single participant's data."""
        df_o = self.df_oura[self.df_oura["pid"] == pid].sort_values("date").copy()
        df_m = self.df_survey[self.df_survey["pid"] == pid].copy()
        
        if df_o.empty or df_m.empty:
            return {}, []
        
        # Calculate baseline metrics
        baseline_metrics = self.calculate_baseline_metrics(pid, df_o)
        baseline_row = {"pid": pid, **baseline_metrics}
        
        # Apply baseline adjustment if enabled
        if self.config.baseline_enabled:
            # Only use data after baseline period for analysis
            base_end = df_o["date"].iloc[0] + pd.Timedelta(days=self.config.baseline_days - 1)
            df_o_analysis = df_o[df_o["date"] > base_end].copy()
            df_o_analysis = self.apply_baseline_adjustment(df_o_analysis, baseline_metrics)
        else:
            df_o_analysis = df_o.copy()
        
        if df_o_analysis.empty:
            return baseline_row, []
        
        # Process each survey response
        survey_rows = []
        for _, survey_row in df_m.iterrows():
            # Extract time window data
            window_data = self.extract_time_window(df_o_analysis, survey_row["date"])
            
            # Compute statistics
            stats = self.compute_statistics(window_data)
            
            # Build survey row
            row_data = {
                "pid": pid,
                "date": survey_row["date"],
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
        
        return baseline_row, survey_rows
    
    def process_all(self) -> Tuple[str, str]:
        """Process all participants and save results."""
        if self.df_oura is None or self.df_survey is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get sample PIDs
        sample_pids = self.get_sample_pids()
        
        # Initialize output files
        main_output = self.output_dir / self.config.generate_filename("main")
        baseline_output = self.output_dir / self.config.generate_filename("baseline")
        
        # Clean existing files
        for file_path in [main_output, baseline_output]:
            file_path.unlink(missing_ok=True)
        
        baseline_rows = []
        first_row = True
        
        logger.info(f"Processing {len(sample_pids)} participants...")
        start_time = time.perf_counter()
        
        for pid in tqdm(sample_pids, desc="Processing participants"):
            try:
                baseline_row, survey_rows = self.process_participant(pid)
                
                if baseline_row:
                    baseline_rows.append(baseline_row)
                
                if survey_rows:
                    # Convert to DataFrame and save
                    df_surveys = pd.DataFrame(survey_rows)
                    
                    # Reorder columns for better readability
                    demo_cols = [c for c in df_surveys.columns if c in ("sex", "age", "gender", "race", "ethnicity_hispanic")]
                    lead_cols = ["pid", "date"] + demo_cols + ["promis_dep_sum", "promis_anx_sum", "after_lockdown"]
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
            
            # Cleanup memory
            gc.collect()
        
        # Save baseline metrics
        if baseline_rows:
            pd.DataFrame(baseline_rows).to_csv(baseline_output, index=False)
            logger.info(f"Saved {len(baseline_rows)} baseline records to {baseline_output}")
        
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Processing complete in {elapsed_time:.1f}s")
        logger.info(f"Main output: {main_output}")
        logger.info(f"Baseline output: {baseline_output}")
        
        # Add dataset descriptions
        self._save_descriptions(main_output, baseline_output)
        
        return str(main_output), str(baseline_output)
    
    def _save_descriptions(self, main_file: Path, baseline_file: Path) -> None:
        """Save dataset descriptions."""
        desc_file = self.output_dir / f"{main_file.stem}_description.txt"
        
        with open(desc_file, 'w') as f:
            f.write("Dataset Description\n")
            f.write("==================\n\n")
            f.write(f"Configuration:\n")
            f.write(f"- Time window: {self.config.window_description}\n")
            f.write(f"- Direction: {self.config.window_direction}\n")
            f.write(f"- Baseline adjustment: {'Enabled' if self.config.baseline_enabled else 'Disabled'}\n")
            if self.config.baseline_enabled:
                f.write(f"- Baseline period: {self.config.baseline_days} days\n")
            f.write(f"- Sample size: {'Full cohort' if self.config.sample_size is None else self.config.sample_size}\n")
            f.write(f"- Metrics: {', '.join(self.config.metric_cols)}\n")
            f.write(f"- Statistics: {', '.join(self.config.stat_functions)}\n\n")
            
            f.write(f"Output files:\n")
            f.write(f"- Main dataset: {main_file.name}\n")
            f.write(f"- Baseline metrics: {baseline_file.name}\n\n")
            
            f.write(f"Main dataset columns:\n")
            f.write(f"- pid: Participant ID\n")
            f.write(f"- date: Survey date\n")
            f.write(f"- promis_dep_sum: PROMIS depression sum score\n")
            f.write(f"- promis_anx_sum: PROMIS anxiety sum score\n")
            f.write(f"- after_lockdown: Binary indicator for post-lockdown surveys\n")
            f.write(f"- [metric]_[stat]: Statistical aggregations of wearable metrics\n")
        
        logger.info(f"Saved description to {desc_file}")


def run_preprocessing(config: ProcessingConfig) -> Tuple[str, str]:
    """Convenience function to run preprocessing with given configuration."""
    preprocessor = WearablePreprocessor(config)
    preprocessor.load_data()
    return preprocessor.process_all()


def get_example_configs() -> Dict[str, ProcessingConfig]:
    """Get example configurations for common use cases."""
    return {
        "backward_30d": ProcessingConfig(
            window_start_offset=-30,
            window_end_offset=0,
            baseline_enabled=True
        ),
        "forward_30d": ProcessingConfig(
            window_start_offset=0,
            window_end_offset=30,
            baseline_enabled=True
        ),
        "backward_6w_to_2w": ProcessingConfig(
            window_start_offset=-42,  # 6 weeks
            window_end_offset=-14,    # 2 weeks
            baseline_enabled=True
        ),
        "forward_1w_to_5w": ProcessingConfig(
            window_start_offset=7,    # 1 week
            window_end_offset=35,     # 5 weeks
            baseline_enabled=True
        ),
        "no_baseline": ProcessingConfig(
            window_start_offset=-30,
            window_end_offset=0,
            baseline_enabled=False
        ),
        "small_sample": ProcessingConfig(
            window_start_offset=-30,
            window_end_offset=0,
            baseline_enabled=True,
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
        print(f"  Baseline: {'Yes' if config.baseline_enabled else 'No'}")
        print(f"  Output: {config.generate_filename('main')}")