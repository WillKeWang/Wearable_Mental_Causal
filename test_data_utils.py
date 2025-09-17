#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the optimized data preprocessing utilities.

This script tests the data_utils.py module with a small sample to verify everything works correctly.
"""

import time
import sys
from pathlib import Path

# Add the current directory to the path so we can import data_utils
sys.path.append(str(Path(__file__).parent))

try:
    from data_utils import ProcessingConfig, run_preprocessing, get_example_configs
    print("✅ Successfully imported data_utils")
except ImportError as e:
    print(f"❌ Failed to import data_utils: {e}")
    sys.exit(1)

def test_configuration():
    """Test configuration creation and validation."""
    print("\n🔧 Testing configuration...")
    
    try:
        # Test basic configuration
        config = ProcessingConfig(
            window_start_offset=-30,
            window_end_offset=0,
            baseline_enabled=True,
            sample_size=100  # Small sample for testing
        )
        
        print(f"✅ Configuration created successfully")
        print(f"   Window: {config.window_description}")
        print(f"   Direction: {config.window_direction}")
        print(f"   Statistics: {config.stat_functions}")
        print(f"   Filename: {config.generate_filename('main')}")
        
        # Verify only mean and std are in stat_functions
        if config.stat_functions == ["mean", "std"]:
            print("✅ Statistics correctly set to mean and std only")
        else:
            print(f"⚠️  Unexpected statistics: {config.stat_functions}")
        
        return config
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return None

def test_with_mock_data():
    """Test processing with mock data if real data is not available."""
    print("\n🧪 Testing with mock data...")
    
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create mock Oura data
    mock_oura_data = []
    start_date = datetime(2020, 1, 1)
    
    for pid in range(10):  # 10 participants
        for day in range(100):  # 100 days each
            date = start_date + timedelta(days=day)
            mock_oura_data.append({
                'pid': f'test_participant_{pid:03d}',
                'date': date,
                'hr_average': 70 + np.random.normal(0, 5),
                'rmssd': 35 + np.random.normal(0, 8),
                'efficiency': 85 + np.random.normal(0, 5),
                'deep': 1.5 + np.random.normal(0, 0.3),
                'light': 3.5 + np.random.normal(0, 0.5),
                'rem': 1.2 + np.random.normal(0, 0.3)
            })
    
    # Create mock survey data
    mock_survey_data = []
    for pid in range(10):
        for month in range(3):  # 3 surveys each
            survey_date = start_date + timedelta(days=30 + month * 30)
            mock_survey_data.append({
                'pid': f'test_participant_{pid:03d}',
                'DateUTC.x': survey_date.strftime("%m/%d/%Y %H:%M"),
                'promis_dep_worthless': np.random.randint(1, 5),
                'promis_dep_helpless': np.random.randint(1, 5),
                'promis_dep_depressed': np.random.randint(1, 5),
                'promis_dep_hopeless': np.random.randint(1, 5),
                'promis_anx_fearful': np.random.randint(1, 5),
                'promis_anx_anxious': np.random.randint(1, 5),
                'promis_anx_worri': np.random.randint(1, 5),
                'promis_anx_uneasy': np.random.randint(1, 5)
            })
    
    # Save mock data
    mock_dir = Path("test_data")
    mock_dir.mkdir(exist_ok=True)
    
    df_oura = pd.DataFrame(mock_oura_data)
    df_survey = pd.DataFrame(mock_survey_data)
    
    oura_file = mock_dir / "mock_oura.csv"
    survey_file = mock_dir / "mock_survey.csv"
    
    df_oura.to_csv(oura_file, index=False)
    df_survey.to_csv(survey_file, index=True)  # Survey file expects index
    
    print(f"✅ Created mock data:")
    print(f"   Oura: {len(df_oura)} records, {df_oura['pid'].nunique()} participants")
    print(f"   Survey: {len(df_survey)} records")
    
    return str(oura_file), str(survey_file)

def run_test_processing():
    """Run a complete test of the processing pipeline."""
    print("\n🚀 Running test processing...")
    
    # Test configuration
    config = test_configuration()
    if not config:
        return False
    
    # Check if real data exists
    real_oura_path = Path("data/raw/oura_all_sleep_summary_stacked_w_temp_max.parquet")
    real_survey_path = Path("data/raw/base_monthly.csv")
    
    if real_oura_path.exists() and real_survey_path.exists():
        print("✅ Found real data files, using them for testing")
        config.oura_path = str(real_oura_path)
        config.survey_path = str(real_survey_path)
        config.sample_size = 50  # Small sample for testing
    else:
        print("⚠️  Real data not found, creating mock data for testing")
        oura_file, survey_file = test_with_mock_data()
        config.oura_path = oura_file
        config.survey_path = survey_file
        config.sample_size = None  # Use all mock data
    
    # Update config for faster testing
    config.chunk_size = 5
    config.n_jobs = 2
    config.use_multiprocessing = False  # Disable for simpler testing
    config.output_dir = "test_output"
    
    print(f"📋 Test configuration:")
    print(f"   Oura path: {config.oura_path}")
    print(f"   Survey path: {config.survey_path}")
    print(f"   Sample size: {config.sample_size}")
    print(f"   Output dir: {config.output_dir}")
    
    try:
        start_time = time.time()
        main_file, baseline_file = run_preprocessing(config)
        elapsed_time = time.time() - start_time
        
        print(f"✅ Processing completed successfully in {elapsed_time:.1f} seconds!")
        print(f"   Main file: {main_file}")
        print(f"   Baseline file: {baseline_file}")
        
        # Verify output files exist and have content
        main_path = Path(main_file)
        baseline_path = Path(baseline_file)
        
        print(f"🔍 Checking output files:")
        print(f"   Main file path: {main_path}")
        print(f"   Main file exists: {main_path.exists()}")
        if main_path.exists():
            print(f"   Main file size: {main_path.stat().st_size} bytes")
        
        print(f"   Baseline file path: {baseline_path}")
        print(f"   Baseline file exists: {baseline_path.exists()}")
        if baseline_path.exists():
            print(f"   Baseline file size: {baseline_path.stat().st_size} bytes")
        
        # Check if output directory exists
        output_dir = Path(config.output_dir)
        print(f"   Output directory: {output_dir}")
        print(f"   Output directory exists: {output_dir.exists()}")
        
        if output_dir.exists():
            files_in_dir = list(output_dir.glob("*"))
            print(f"   Files in output directory: {[f.name for f in files_in_dir]}")
        
        if main_path.exists() and baseline_path.exists():
            import pandas as pd
            try:
                df_main = pd.read_csv(main_path)
                df_baseline = pd.read_csv(baseline_path)
                
                print(f"📊 Output verification:")
                print(f"   Main dataset: {len(df_main)} rows, {len(df_main.columns)} columns")
                print(f"   Baseline dataset: {len(df_baseline)} rows, {len(df_baseline.columns)} columns")
                
                # Check that only mean and std columns are present
                stat_cols = [c for c in df_main.columns if any(s in c for s in ['_mean', '_std', '_rms', '_skew'])]
                mean_cols = [c for c in stat_cols if c.endswith('_mean')]
                std_cols = [c for c in stat_cols if c.endswith('_std')]
                other_stats = [c for c in stat_cols if not (c.endswith('_mean') or c.endswith('_std'))]
                
                print(f"   Statistics columns: {len(mean_cols)} mean, {len(std_cols)} std")
                
                if other_stats:
                    print(f"⚠️  Found unexpected statistic columns: {other_stats}")
                else:
                    print("✅ Only mean and std statistics found (as expected)")
                
                print(f"   Sample columns: {list(df_main.columns[:10])}")
                
                return True
                
            except Exception as e:
                print(f"❌ Error reading output files: {e}")
                return False
        else:
            missing_files = []
            if not main_path.exists():
                missing_files.append("main")
            if not baseline_path.exists():
                missing_files.append("baseline")
            
            print(f"❌ Output files not found: {missing_files}")
            
            # Additional debugging
            print(f"\n🔍 Additional debugging:")
            print(f"   Current working directory: {Path.cwd()}")
            print(f"   Absolute main path: {main_path.absolute()}")
            print(f"   Absolute baseline path: {baseline_path.absolute()}")
            
            return False
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_example_configs():
    """Test example configurations."""
    print("\n📋 Testing example configurations...")
    
    try:
        configs = get_example_configs()
        print(f"✅ Found {len(configs)} example configurations:")
        
        for name, config in configs.items():
            print(f"   {name}:")
            print(f"     Window: {config.window_description}")
            print(f"     Stats: {config.stat_functions}")
            print(f"     Chunk size: {config.chunk_size}")
            
        return True
    except Exception as e:
        print(f"❌ Example configs test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 TESTING OPTIMIZED DATA_UTILS.PY")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Configuration
    if test_configuration():
        tests_passed += 1
    
    # Test 2: Example configurations
    if test_example_configs():
        tests_passed += 1
    
    # Test 3: Full processing pipeline
    if run_test_processing():
        tests_passed += 1
    
    print(f"\n📊 TEST RESULTS")
    print("=" * 30)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The module is working correctly.")
        print("\n💡 Next steps:")
        print("1. Verify the output files look correct")
        print("2. Run with a larger sample size for performance testing")
        print("3. Process your full 30K participant dataset")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)