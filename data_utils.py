"""
Performance-optimized wearable data preprocessing utilities for TemPredict dataset.

This module provides highly optimized preprocessing of Oura wearable data with survey responses,
designed to handle 30K+ participants efficiently.
"""

import time
import gc
import warnings
import logging
import unittest
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from functools import partial
import psutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from numba import jit, prange

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fast skew: SciPy if available, else NumPy fallback
try:
    from scipy.stats import skew
    logger.info("Using SciPy for skewness calculation")
except ImportError:
    logger.warning("SciPy not available, using NumPy fallback for skewness")
    def skew(a, axis=0, nan_policy="omit", bias=False):
        arr = np.asarray(a)
        if nan_policy == "omit":
            arr = np.where(np.isnan(arr), np.nan, arr)
        m = np.nanmean(arr, axis=axis)
        s = np.nanstd(arr, axis=axis, ddof=1)
        return np.nanmean((arr - m)**3, axis=axis) / (s**3)

# Silence common RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*slice")
warnings.filterwarnings("ignore", category=RuntimeWarnings, message="Degrees of freedom")


@jit(nopython=True)
def fast_window_stats(dates, values, survey_date_ts, window_start_days, window_end_days):
    """Numba-optimized window statistics calculation - mean and std only."""
    # Convert survey date to timestamp for comparison
    start_ts = survey_date_ts + window_start_days * 86400_000_000_000  # nanoseconds
    end_ts = survey_date_ts + window_end_days * 86400_000_000_000
    
    # Find indices within window
    mask = (dates >= start_ts) & (dates < end_ts)
    window_values = values[mask]
    
    if len(window_values) == 0:
        return np.nan, np.nan
    
    # Remove NaN values
    valid_values = window_values[~np.isnan(window_values)]
    if len(valid_values) == 0:
        return np.nan, np.nan
    
    # Calculate mean and standard deviation only
    mean_val = np.mean(valid_values)
    
    if len(valid_values) > 1:
        std_val = np.std(valid_values, ddof=1)
    else:
        std_val = 0.0
    
    return mean_val, std_val


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
    chunk_size: int = 500  # Process participants in chunks
    n_jobs: int = -1  # Number of parallel jobs (-1 = all cores)
    
    # Performance options
    optimize_memory: bool = True
    use_multiprocessing: bool = True
    batch_size: int = 1000  # Batch size for I/O operations
    
    # Metrics to process
    metric_cols: List[str] = field(default_factory=lambda: [
        "hr_average", "rmssd", "breath_average", "hr_lowest",
        "breath_v_average", "temperature_deviation", 
        "temperature_trend_deviation", "temperature_max",
        "onset_latency", "efficiency", "deep", "light",
        "rem", "awake", "total"
    ])
    
    # Statistical aggregations to compute
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
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.n_jobs == -1:
            self.n_jobs = mp.cpu_count()
    
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
        end_desc = f"{abs(self.window_end_offset)}d_{'before' if self.window_end_offset < 0 else 'after'}"
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


class PerformanceMonitor:
    """Monitor processing performance and memory usage."""
    
    def __init__(self):
        self.checkpoints = {}
        self.start_time = time.perf_counter()
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    def checkpoint(self, name: str, participants_processed: int = 0):
        """Record a performance checkpoint."""
        current_time = time.perf_counter()
        elapsed = current_time - self.start_time
        memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        rate = participants_processed / elapsed if elapsed > 0 and participants_processed > 0 else 0
        
        self.checkpoints[name] = {
            'elapsed': elapsed,
            'memory_mb': memory,
            'participants': participants_processed,
            'rate_per_min': rate * 60
        }
        
        if rate > 0:
            logger.info(f"{name}: {participants_processed} participants in {elapsed:.1f}s "
                       f"({rate:.1f}/s, {memory:.1f}MB)")
        else:
            logger.info(f"{name}: {elapsed:.1f}s elapsed, {memory:.1f}MB memory")
    
    def estimate_completion(self, current_participants: int, total_participants: int):
        """Estimate completion time based on current progress."""
        if current_participants <= 0:
            return None
            
        elapsed = time.perf_counter() - self.start_time
        rate = current_participants / elapsed
        remaining = total_participants - current_participants
        eta_seconds = remaining / rate if rate > 0 else 0
        
        return {
            'rate_per_second': rate,
            'rate_per_minute': rate * 60,
            'eta_seconds': eta_seconds,
            'eta_minutes': eta_seconds / 60,
            'percent_complete': current_participants / total_participants * 100
        }


class OptimizedWearablePreprocessor:
    """Performance-optimized preprocessor for large-scale wearable data."""
    
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
        self.oura_grouped = None  # Pre-grouped Oura data
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        
        logger.info(f"Initialized OptimizedWearablePreprocessor")
        logger.info(f"Config: {config.chunk_size} chunk size, {config.n_jobs} jobs, "
                   f"multiprocessing={'enabled' if config.use_multiprocessing else 'disabled'}")
    
    def optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage."""
        if not self.config.optimize_memory:
            return df
            
        logger.info("Optimizing data types...")
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in self.config.metric_cols:
                # Keep metric columns as float32 for precision
                if df[col].dtype == np.float64:
                    df[col] = df[col].astype(np.float32)
            else:
                # Downcast other numeric columns
                df[col] = pd.to_numeric(df[col], downcast='integer' if df[col].dtype.kind in 'iu' else 'float')
        
        # Convert PID to category if many repeated values
        if 'pid' in df.columns and df['pid'].nunique() < len(df) * 0.1:
            df['pid'] = df['pid'].astype('category')
        
        new_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(f"Memory optimized: {original_memory:.1f}MB → {new_memory:.1f}MB "
                   f"({(1-new_memory/original_memory)*100:.1f}% reduction)")
        
        return df
    
    def load_data(self) -> None:
        """Load and optimize Oura and survey data."""
        logger.info("Loading data...")
        self.monitor.checkpoint("load_start")
        
        # Load Oura data
        logger.info("Loading Oura parquet data...")
        if self.config.oura_path.endswith('.parquet'):
            # Load only needed columns for memory efficiency
            needed_cols = ['pid', 'date'] + self.config.metric_cols
            self.df_oura = pq.read_table(self.config.oura_path, columns=needed_cols).to_pandas(ignore_metadata=True)
        else:
            self.df_oura = pd.read_csv(self.config.oura_path)
        
        self.df_oura["date"] = pd.to_datetime(self.df_oura["date"], utc=True)
        logger.info(f"Loaded {len(self.df_oura):,} Oura records")
        
        # Optimize Oura data types
        self.df_oura = self.optimize_data_types(self.df_oura)
        self.monitor.checkpoint("oura_loaded")
        
        # Load survey data
        logger.info("Loading monthly survey data...")
        self.df_survey = (
            pd.read_csv(self.config.survey_path, index_col=0)
            .rename(columns={"hashed_id": "pid"})
            .pipe(self._convert_dateutc)
        )
        
        # Remove excluded participant
        initial_count = len(self.df_survey)
        self.df_survey = self.df_survey.loc[self.df_survey["pid"] != self.config.excluded_pid]
        logger.info(f"Loaded {len(self.df_survey):,} survey records "
                   f"(removed {initial_count - len(self.df_survey)} excluded)")
        
        # Optimize survey data types
        self.df_survey = self.optimize_data_types(self.df_survey)
        
        # Find PROMIS columns
        self.dep_cols = self._match_columns(self.df_survey, self.config.depression_patterns)
        self.anx_cols = self._match_columns(self.df_survey, self.config.anxiety_patterns)
        
        if len(self.dep_cols) != 4 or len(self.anx_cols) != 4:
            raise ValueError(f"Expected 4 PROMIS columns each, found {len(self.dep_cols)} depression, {len(self.anx_cols)} anxiety")
        
        logger.info(f"Found PROMIS columns - Depression: {self.dep_cols}, Anxiety: {self.anx_cols}")
        self.monitor.checkpoint("survey_loaded")
        
        # CRITICAL OPTIMIZATION: Pre-group Oura data by participant
        logger.info("Pre-grouping Oura data by participant (this may take a moment)...")
        self.df_oura = self.df_oura.sort_values(['pid', 'date'])  # Sort for efficiency
        self.oura_grouped = dict(list(self.df_oura.groupby('pid')))
        logger.info(f"Pre-grouped {len(self.oura_grouped):,} participants")
        self.monitor.checkpoint("oura_grouped")
        
        # Clear original DataFrame to save memory
        if self.config.optimize_memory:
            del self.df_oura
            gc.collect()
            logger.info("Cleared original Oura DataFrame to save memory")
    
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
            logger.info(f"Processing full cohort: {len(all_pids):,} participants")
            return all_pids
        
        rng = np.random.default_rng(self.config.random_seed)
        sample_pids = rng.choice(all_pids, size=min(self.config.sample_size, len(all_pids)), replace=False).tolist()
        logger.info(f"Processing sample: {len(sample_pids):,} participants")
        return sample_pids
    
    def process_participant_chunk(self, pid_chunk: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Process a chunk of participants efficiently."""
        baseline_rows = []
        survey_rows = []
        
        for pid in pid_chunk:
            try:
                # Use pre-grouped data instead of filtering entire DataFrame
                if pid not in self.oura_grouped:
                    continue
                
                df_oura_pid = self.oura_grouped[pid]
                df_survey_pid = self.df_survey[self.df_survey["pid"] == pid]
                
                if df_oura_pid.empty or df_survey_pid.empty:
                    continue
                
                # Process baseline
                baseline_metrics = self.calculate_baseline_metrics(pid, df_oura_pid)
                if baseline_metrics:
                    baseline_rows.append({"pid": pid, **baseline_metrics})
                
                # Apply baseline adjustment and get analysis data
                df_analysis = self.prepare_analysis_data(df_oura_pid, baseline_metrics)
                if df_analysis.empty:
                    continue
                
                # Convert to numpy arrays for fast processing
                dates_np = df_analysis['date'].values.astype('datetime64[ns]').view('int64')
                
                # Process each survey
                for _, survey_row in df_survey_pid.iterrows():
                    survey_date_ts = pd.Timestamp(survey_row['date']).value
                    
                    # Calculate statistics for each metric efficiently
                    stats = {}
                    for col in self.config.metric_cols:
                        if col in df_analysis.columns:
                            values_np = df_analysis[col].values.astype(np.float64)
                            
                            mean_val, std_val = fast_window_stats(
                                dates_np, values_np, survey_date_ts,
                                self.config.window_start_offset, self.config.window_end_offset
                            )
                            
                            if 'mean' in self.config.stat_functions:
                                stats[f"{col}_mean"] = mean_val
                            if 'std' in self.config.stat_functions:
                                stats[f"{col}_std"] = std_val
                    
                    # Build result row
                    row_data = self.build_survey_row(pid, survey_row, stats)
                    survey_rows.append(row_data)
                    
            except Exception as e:
                logger.error(f"Error processing {pid}: {str(e)}")
                continue
        
        return baseline_rows, survey_rows
    
    def calculate_baseline_metrics(self, pid: str, df_oura_pid: pd.DataFrame) -> Dict[str, float]:
        """Calculate baseline metrics for a participant."""
        if not self.config.baseline_enabled or df_oura_pid.empty:
            return {}
        
        base_end = df_oura_pid["date"].iloc[0] + pd.Timedelta(days=self.config.baseline_days - 1)
        baseline_data = df_oura_pid[df_oura_pid["date"] <= base_end][self.config.metric_cols]
        
        if baseline_data.empty:
            return {f"{c}_baseline_mean": np.nan for c in self.config.metric_cols}
        
        baseline_means = baseline_data.mean()
        return {f"{c}_baseline_mean": baseline_means[c] for c in self.config.metric_cols 
                if c in baseline_means.index}
    
    def prepare_analysis_data(self, df_oura_pid: pd.DataFrame, baseline_metrics: Dict[str, float]) -> pd.DataFrame:
        """Prepare analysis data with optional baseline adjustment."""
        if not self.config.baseline_enabled:
            return df_oura_pid.copy()
        
        # Filter to post-baseline period
        base_end = df_oura_pid["date"].iloc[0] + pd.Timedelta(days=self.config.baseline_days - 1)
        df_analysis = df_oura_pid[df_oura_pid["date"] > base_end].copy()
        
        if df_analysis.empty or not baseline_metrics:
            return df_analysis
        
        # Apply baseline adjustment
        for col in self.config.metric_cols:
            baseline_key = f"{col}_baseline_mean"
            if baseline_key in baseline_metrics and not np.isnan(baseline_metrics[baseline_key]):
                if col in df_analysis.columns:
                    df_analysis[col] = df_analysis[col] - baseline_metrics[baseline_key]
        
        return df_analysis
    
    def build_survey_row(self, pid: str, survey_row: pd.Series, stats: Dict[str, float]) -> Dict[str, Any]:
        """Build survey result row with all required fields."""
        row_data = {
            "pid": pid,
            "date": survey_row["date"],
            **stats
        }
        
        # Add PROMIS scores
        row_data["promis_dep_sum"] = pd.to_numeric(survey_row[self.dep_cols], errors="coerce").sum()
        row_data["promis_anx_sum"] = pd.to_numeric(survey_row[self.anx_cols], errors="coerce").sum()
        
        # Add demographic info if available
        demo_cols = [c for c in ("sex", "age", "gender", "race", "ethnicity_hispanic") 
                    if c in survey_row.index]
        for col in demo_cols:
            row_data[col] = survey_row[col]
        
        # Add lockdown indicator
        lockdown_cutoff = pd.Timestamp(self.config.lockdown_cutoff, tz="UTC")
        row_data["after_lockdown"] = int(survey_row["date"] >= lockdown_cutoff)
        
        return row_data
    
    def process_all(self) -> Tuple[str, str]:
        """Process all participants with optimized chunking and optional multiprocessing."""
        if self.oura_grouped is None or self.df_survey is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get sample PIDs and create chunks
        sample_pids = self.get_sample_pids()
        pid_chunks = [sample_pids[i:i + self.config.chunk_size] 
                     for i in range(0, len(sample_pids), self.config.chunk_size)]
        
        logger.info(f"Processing {len(sample_pids):,} participants in {len(pid_chunks)} chunks")
        
        # Initialize output files
        main_output = self.output_dir / self.config.generate_filename("main")
        baseline_output = self.output_dir / self.config.generate_filename("baseline")
        
        for file_path in [main_output, baseline_output]:
            file_path.unlink(missing_ok=True)
        
        all_baseline_rows = []
        all_survey_rows = []
        participants_processed = 0
        
        self.monitor.checkpoint("processing_start")
        
        # Process chunks
        if self.config.use_multiprocessing and self.config.n_jobs > 1:
            logger.info(f"Using multiprocessing with {self.config.n_jobs} cores")
            
            with mp.Pool(processes=self.config.n_jobs) as pool:
                with tqdm(total=len(pid_chunks), desc="Processing chunks") as pbar:
                    for baseline_rows, survey_rows in pool.imap(self.process_participant_chunk, pid_chunks):
                        all_baseline_rows.extend(baseline_rows)
                        all_survey_rows.extend(survey_rows)
                        participants_processed += len(pid_chunks[pbar.n])
                        pbar.update(1)
                        
                        # Periodic progress update with ETA
                        if pbar.n % 10 == 0:
                            eta = self.monitor.estimate_completion(participants_processed, len(sample_pids))
                            if eta:
                                pbar.set_description(f"Processing chunks (ETA: {eta['eta_minutes']:.1f}m)")
        else:
            logger.info("Using single-threaded processing")
            
            for chunk in tqdm(pid_chunks, desc="Processing chunks"):
                baseline_rows, survey_rows = self.process_participant_chunk(chunk)
                all_baseline_rows.extend(baseline_rows)
                all_survey_rows.extend(survey_rows)
                participants_processed += len(chunk)
                
                # Cleanup memory periodically
                if len(all_survey_rows) >= self.config.batch_size:
                    gc.collect()
        
        self.monitor.checkpoint("processing_complete", participants_processed)
        
        # Save results efficiently
        logger.info("Saving results...")
        
        if all_survey_rows:
            df_surveys = pd.DataFrame(all_survey_rows)
            
            # Reorder columns for readability
            demo_cols = [c for c in df_surveys.columns if c in ("sex", "age", "gender", "race", "ethnicity_hispanic")]
            lead_cols = ["pid", "date"] + demo_cols + ["promis_dep_sum", "promis_anx_sum", "after_lockdown"]
            stat_cols = [c for c in df_surveys.columns if any(s in c for s in self.config.stat_functions)]
            final_cols = lead_cols + stat_cols
            
            df_surveys = df_surveys[[c for c in final_cols if c in df_surveys.columns]]
            df_surveys.to_csv(main_output, index=False)
            logger.info(f"Saved {len(df_surveys):,} survey records to {main_output}")
        
        if all_baseline_rows:
            df_baseline = pd.DataFrame(all_baseline_rows)
            df_baseline.to_csv(baseline_output, index=False)
            logger.info(f"Saved {len(df_baseline):,} baseline records to {baseline_output}")
        
        # Save descriptions
        self._save_descriptions(main_output, baseline_output)
        
        # Final performance summary
        elapsed_time = time.perf_counter() - self.monitor.start_time
        logger.info(f"Processing complete in {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
        logger.info(f"Average rate: {participants_processed/elapsed_time:.1f} participants/second")
        
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
            f.write(f"- Chunk size: {self.config.chunk_size}\n")
            f.write(f"- Parallel jobs: {self.config.n_jobs}\n")
            f.write(f"- Metrics: {', '.join(self.config.metric_cols)}\n")
            f.write(f"- Statistics: {', '.join(self.config.stat_functions)}\n\n")
            
            f.write(f"Output files:\n")
            f.write(f"- Main dataset: {main_file.name}\n")
            f.write(f"- Baseline metrics: {baseline_file.name}\n\n")
            
            f.write(f"Performance summary:\n")
            for name, stats in self.monitor.checkpoints.items():
                f.write(f"- {name}: {stats['elapsed']:.1f}s")
                if stats.get('rate_per_min', 0) > 0:
                    f.write(f" ({stats['rate_per_min']:.1f} participants/min)")
                f.write(f"\n")
        
        logger.info(f"Saved description to {desc_file}")


# Backward compatibility - use optimized version by default
WearablePreprocessor = OptimizedWearablePreprocessor


# Unit Testing
class TestOptimizedWearablePreprocessor(unittest.TestCase):
    """Unit tests for OptimizedWearablePreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig(
            window_start_offset=-7,
            window_end_offset=0,
            baseline_enabled=True,
            baseline_days=14,
            sample_size=5,
            chunk_size=2,
            use_multiprocessing=False  # Disable for testing
        )
    
    def test_config_validation(self):
        """Test configuration validation."""
        with self.assertRaises(ValueError):
            ProcessingConfig(window_start_offset=5, window_end_offset=0)
        
        with self.assertRaises(ValueError):
            ProcessingConfig(baseline_days=-1)
    
    def test_fast_window_stats(self):
        """Test the Numba-optimized statistics function."""
        # Create test data
        dates = np.array([1, 2, 3, 4, 5], dtype='datetime64[ns]').view('int64')
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        survey_date = 3  # Middle date
        
        mean_val, std_val = fast_window_stats(
            dates, values, survey_date, -1, 1
        )
        
        # Should include values at indices 1, 2, 3 (dates 2, 3, 4, values 2, 3, 4)
        expected_mean = 3.0
        self.assertAlmostEqual(mean_val, expected_mean, places=5)
        self.assertFalse(np.isnan(std_val))
    
    def test_performance_monitor(self):
        """Test performance monitoring functionality."""
        monitor = PerformanceMonitor()
        monitor.checkpoint("test", 100)
        
        self.assertIn("test", monitor.checkpoints)
        self.assertGreater(monitor.checkpoints["test"]["elapsed"], 0)
        
        eta = monitor.estimate_completion(50, 100)
        self.assertIsNotNone(eta)
        self.assertEqual(eta["percent_complete"], 50.0)


def run_preprocessing(config: ProcessingConfig) -> Tuple[str, str]:
    """Convenience function to run optimized preprocessing with given configuration."""
    preprocessor = OptimizedWearablePreprocessor(config)
    preprocessor.load_data()
    return preprocessor.process_all()


def get_example_configs() -> Dict[str, ProcessingConfig]:
    """Get example configurations optimized for large-scale processing."""
    return {
        "backward_30d_optimized": ProcessingConfig(
            window_start_offset=-30,
            window_end_offset=0,
            baseline_enabled=True,
            chunk_size=1000,  # Larger chunks for better performance
            n_jobs=-1,
            use_multiprocessing=True
        ),
        "forward_30d_optimized": ProcessingConfig(
            window_start_offset=0,
            window_end_offset=30,
            baseline_enabled=True,
            chunk_size=1000,
            n_jobs=-1,
            use_multiprocessing=True
        ),
        "quick_test": ProcessingConfig(
            window_start_offset=-30,
            window_end_offset=0,
            baseline_enabled=True,
            sample_size=1000,  # Test with 1K participants
            chunk_size=100,
            n_jobs=4
        ),
        "memory_efficient": ProcessingConfig(
            window_start_offset=-30,
            window_end_offset=0,
            baseline_enabled=True,
            chunk_size=500,  # Smaller chunks for limited memory
            optimize_memory=True,
            batch_size=500
        )
    }


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Example usage
    print("\nOptimized configurations for large-scale processing:")
    configs = get_example_configs()
    
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  Window: {config.window_description}")
        print(f"  Chunk size: {config.chunk_size}")
        print(f"  Parallel jobs: {config.n_jobs}")
        print(f"  Output: {config.generate_filename('main')}")