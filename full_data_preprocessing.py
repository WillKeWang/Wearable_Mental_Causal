#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Full preprocessing run on entire dataset with timing analysis.
Runs two time windows: -6 to -2 weeks and 1 to 5 weeks.
"""

import time
import sys
import pandas as pd
import psutil
import os
from pathlib import Path
from datetime import datetime, timedelta
import gc

# Import the data processing module
sys.path.append(str(Path(__file__).parent))

try:
    from data_utils import ProcessingConfig, run_preprocessing
except ImportError as e:
    print(f"Error importing data_utils: {e}")
    sys.exit(1)


def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb / 1024


def print_system_info():
    """Print system information for benchmarking context."""
    print("\nSYSTEM INFORMATION")
    print("=" * 50)
    print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")


def check_data_files():
    """Check if required data files exist and get their info."""
    oura_path = Path("data/raw/oura_all_sleep_summary_stacked_w_temp_max.parquet")
    survey_path = Path("data/raw/base_monthly.csv")
    
    if not oura_path.exists():
        raise FileNotFoundError(f"Oura data file not found: {oura_path}")
    if not survey_path.exists():
        raise FileNotFoundError(f"Survey data file not found: {survey_path}")
    
    # Get file sizes
    oura_size_mb = oura_path.stat().st_size / (1024 * 1024)
    survey_size_mb = survey_path.stat().st_size / (1024 * 1024)
    
    print("\nDATA FILES")
    print("=" * 50)
    print(f"Oura data: {oura_path}")
    print(f"  Size: {oura_size_mb:.1f} MB")
    
    print(f"Survey data: {survey_path}")
    print(f"  Size: {survey_size_mb:.1f} MB")
    
    return str(oura_path), str(survey_path)


def get_data_overview(oura_path, survey_path):
    """Get overview of data size for timing context."""
    print("\nDATA OVERVIEW")
    print("=" * 50)
    
    start_time = time.time()
    print("Loading data for overview...")
    
    try:
        # Load survey data
        survey_df = pd.read_csv(survey_path, index_col=0, low_memory=False)
        if "hashed_id" in survey_df.columns:
            survey_df = survey_df.rename(columns={"hashed_id": "pid"})
        
        # Load Oura data
        if Path(oura_path).suffix == '.parquet':
            import pyarrow.parquet as pq
            oura_df = pq.read_table(oura_path).to_pandas()
        else:
            oura_df = pd.read_csv(oura_path)
        
        # Convert MultiIndex to columns if needed
        if isinstance(oura_df.index, pd.MultiIndex):
            oura_df = oura_df.reset_index()
        
        load_time = time.time() - start_time
        
        # Get overview stats
        survey_participants = survey_df['pid'].nunique()
        survey_records = len(survey_df)
        oura_participants = oura_df['pid'].nunique()
        oura_records = len(oura_df)
        
        # Find overlapping participants
        overlap_participants = len(set(survey_df['pid'].unique()) & set(oura_df['pid'].unique()))
        
        print(f"Data loading time: {format_duration(load_time)}")
        print(f"Survey data: {survey_records:,} records, {survey_participants:,} participants")
        print(f"Oura data: {oura_records:,} records, {oura_participants:,} participants")
        print(f"Overlapping participants: {overlap_participants:,}")
        
        # Memory cleanup
        del survey_df, oura_df
        gc.collect()
        
        return {
            'survey_records': survey_records,
            'survey_participants': survey_participants,
            'oura_records': oura_records,
            'oura_participants': oura_participants,
            'overlap_participants': overlap_participants,
            'load_time': load_time
        }
        
    except Exception as e:
        print(f"Error loading data for overview: {e}")
        return None


def run_single_configuration(config, config_name, data_overview=None):
    """Run preprocessing for a single configuration with detailed timing."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {config_name}")
    print(f"{'='*60}")
    
    # Print configuration details
    print(f"Time window: {config.window_start_offset} to {config.window_end_offset} days")
    print(f"Window description: {config.window_description}")
    print(f"Baseline adjustment: {config.baseline_enabled}")
    print(f"Statistics: {config.stat_functions}")
    print(f"Sample size: {'Full dataset' if config.sample_size is None else config.sample_size}")
    print(f"Output directory: {config.output_dir}")
    
    # Expected output files
    main_filename = config.generate_filename("main")
    baseline_filename = config.generate_filename("baseline")
    print(f"Expected main file: {main_filename}")
    print(f"Expected baseline file: {baseline_filename}")
    
    # Memory before processing
    memory_before = get_memory_usage()
    print(f"Memory usage before: {memory_before:.1f} GB")
    
    # Start timing
    start_time = time.time()
    process_start = time.process_time()
    
    print(f"\nStarting processing at {datetime.now().strftime('%H:%M:%S')}...")
    
    try:
        # Run preprocessing
        main_file, baseline_file = run_preprocessing(config)
        
        # End timing
        end_time = time.time()
        process_end = time.process_time()
        
        wall_time = end_time - start_time
        cpu_time = process_end - process_start
        
        # Memory after processing
        memory_after = get_memory_usage()
        memory_peak = max(memory_before, memory_after)  # Approximate
        
        print(f"Processing completed at {datetime.now().strftime('%H:%M:%S')}")
        
        # Load and analyze results
        try:
            df_main = pd.read_csv(main_file)
            df_baseline = pd.read_csv(baseline_file)
            
            main_records = len(df_main)
            baseline_records = len(df_baseline)
            main_participants = df_main['pid'].nunique()
            baseline_participants = df_baseline['pid'].nunique()
            
            # Get file sizes
            main_size_mb = Path(main_file).stat().st_size / (1024 * 1024)
            baseline_size_mb = Path(baseline_file).stat().st_size / (1024 * 1024)
            
            print(f"\nRESULTS:")
            print(f"  Main dataset: {main_records:,} records, {main_participants:,} participants")
            print(f"  Baseline dataset: {baseline_records:,} records, {baseline_participants:,} participants")
            print(f"  Main file size: {main_size_mb:.1f} MB")
            print(f"  Baseline file size: {baseline_size_mb:.1f} MB")
            
            # Calculate processing rates
            if data_overview:
                input_records = data_overview['oura_records']
                records_per_second = input_records / wall_time if wall_time > 0 else 0
                print(f"  Processing rate: {records_per_second:,.0f} input records/second")
            
        except Exception as e:
            print(f"  Error analyzing results: {e}")
            main_records = baseline_records = 0
            main_participants = baseline_participants = 0
            main_size_mb = baseline_size_mb = 0
        
        print(f"\nTIMING:")
        print(f"  Wall time: {format_duration(wall_time)}")
        print(f"  CPU time: {format_duration(cpu_time)}")
        print(f"  CPU efficiency: {(cpu_time/wall_time)*100:.1f}%" if wall_time > 0 else "N/A")
        
        print(f"\nMEMORY:")
        print(f"  Before: {memory_before:.1f} GB")
        print(f"  After: {memory_after:.1f} GB")
        print(f"  Peak: {memory_peak:.1f} GB")
        print(f"  Delta: {memory_after - memory_before:+.1f} GB")
        
        # Memory cleanup
        if 'df_main' in locals():
            del df_main
        if 'df_baseline' in locals():
            del df_baseline
        gc.collect()
        
        return {
            'config_name': config_name,
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'memory_before': memory_before,
            'memory_after': memory_after,
            'memory_peak': memory_peak,
            'main_records': main_records,
            'baseline_records': baseline_records,
            'main_participants': main_participants,
            'baseline_participants': baseline_participants,
            'main_size_mb': main_size_mb,
            'baseline_size_mb': baseline_size_mb,
            'main_file': main_file,
            'baseline_file': baseline_file,
            'success': True
        }
        
    except Exception as e:
        end_time = time.time()
        wall_time = end_time - start_time
        memory_after = get_memory_usage()
        
        print(f"ERROR: Processing failed after {format_duration(wall_time)}")
        print(f"Error details: {e}")
        
        return {
            'config_name': config_name,
            'wall_time': wall_time,
            'cpu_time': 0,
            'memory_before': memory_before,
            'memory_after': memory_after,
            'memory_peak': memory_after,
            'main_records': 0,
            'baseline_records': 0,
            'main_participants': 0,
            'baseline_participants': 0,
            'main_size_mb': 0,
            'baseline_size_mb': 0,
            'main_file': None,
            'baseline_file': None,
            'success': False,
            'error': str(e)
        }


def main():
    """Main function to run full preprocessing with timing."""
    print("FULL DATA PREPROCESSING RUN WITH TIMING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print system info
    print_system_info()
    
    # Check data files
    try:
        oura_path, survey_path = check_data_files()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    
    # Get data overview
    data_overview = get_data_overview(oura_path, survey_path)
    
    # Create output directory
    output_base = Path("data/processed/full_run")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Define configurations for the two time windows
    configs = [
        {
            'name': '6w_to_2w_before',
            'config': ProcessingConfig(
                oura_path=oura_path,
                survey_path=survey_path,
                output_dir=str(output_base / "6w_to_2w_before"),
                window_start_offset=-42,  # -6 weeks
                window_end_offset=-14,    # -2 weeks
                baseline_enabled=True,
                stat_functions=["mean", "std"],  # Using simplified stats from tests
                sample_size=None  # Full dataset
            )
        },
        {
            'name': '1w_to_5w_after',
            'config': ProcessingConfig(
                oura_path=oura_path,
                survey_path=survey_path,
                output_dir=str(output_base / "1w_to_5w_after"),
                window_start_offset=7,    # +1 week
                window_end_offset=35,     # +5 weeks
                baseline_enabled=True,
                stat_functions=["mean", "std"],  # Using simplified stats from tests
                sample_size=None  # Full dataset
            )
        }
    ]
    
    # Run processing for each configuration
    results = []
    total_start_time = time.time()
    
    for i, config_info in enumerate(configs, 1):
        print(f"\n\n{'='*20} CONFIGURATION {i}/{len(configs)} {'='*20}")
        
        result = run_single_configuration(
            config_info['config'], 
            config_info['name'],
            data_overview
        )
        results.append(result)
        
        # Short break between runs for system recovery
        if i < len(configs):
            print(f"\nWaiting 10 seconds before next configuration...")
            time.sleep(10)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("SUMMARY REPORT")
    print(f"{'='*60}")
    print(f"Total runtime: {format_duration(total_time)}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful_runs = sum(1 for r in results if r['success'])
    print(f"Successful runs: {successful_runs}/{len(results)}")
    
    # Detailed results table
    print(f"\nDETAILED RESULTS:")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Time':<12} {'Records':<12} {'Participants':<12} {'Size (MB)':<10}")
    print("-" * 80)
    
    for result in results:
        if result['success']:
            time_str = format_duration(result['wall_time'])
            records_str = f"{result['main_records']:,}"
            participants_str = f"{result['main_participants']:,}"
            size_str = f"{result['main_size_mb']:.1f}"
            
            print(f"{result['config_name']:<20} {time_str:<12} {records_str:<12} {participants_str:<12} {size_str:<10}")
        else:
            print(f"{result['config_name']:<20} {'FAILED':<12} {'--':<12} {'--':<12} {'--':<10}")
    
    # Performance metrics
    if successful_runs > 0:
        print(f"\nPERFORMANCE METRICS:")
        total_records = sum(r['main_records'] for r in results if r['success'])
        total_processing_time = sum(r['wall_time'] for r in results if r['success'])
        avg_records_per_second = total_records / total_processing_time if total_processing_time > 0 else 0
        
        print(f"  Total records processed: {total_records:,}")
        print(f"  Total processing time: {format_duration(total_processing_time)}")
        print(f"  Average processing rate: {avg_records_per_second:,.0f} records/second")
        
        if data_overview:
            efficiency = (total_records / data_overview['oura_records']) * 100 if data_overview['oura_records'] > 0 else 0
            print(f"  Data utilization: {efficiency:.1f}% of input records")
    
    # Output file locations
    print(f"\nOUTPUT FILES:")
    for result in results:
        if result['success']:
            print(f"  {result['config_name']}:")
            print(f"    Main: {result['main_file']}")
            print(f"    Baseline: {result['baseline_file']}")
    
    # Save timing results
    results_file = output_base / f"timing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)
        print(f"\nTiming results saved to: {results_file}")
    except Exception as e:
        print(f"Error saving timing results: {e}")
    
    print(f"\n🎉 Full preprocessing run completed!")
    return successful_runs == len(configs)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)