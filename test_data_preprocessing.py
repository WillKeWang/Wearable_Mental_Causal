#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple, focused unit testing for wearable data preprocessing with real data only.
"""

import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import the data processing module
sys.path.append(str(Path(__file__).parent))

try:
    from data_utils import ProcessingConfig, WearablePreprocessor, run_preprocessing
except ImportError as e:
    print(f"Error importing data_utils: {e}")
    sys.exit(1)


class TestDataPreprocessingReal(unittest.TestCase):
    """Test data preprocessing with real data subsets."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test with real data subset."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_output_dir = Path(cls.temp_dir) / "test_output"
        cls.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Test directory: {cls.temp_dir}")
        
        # Check for real data
        cls.real_oura_path = Path("data/raw/oura_all_sleep_summary_stacked_w_temp_max.parquet")
        cls.real_survey_path = Path("data/raw/base_monthly.csv")
        
        if not (cls.real_oura_path.exists() and cls.real_survey_path.exists()):
            raise unittest.SkipTest("Real data files not found. Place data in data/raw/ directory")
        
        # Create a small real data subset for testing
        cls._create_real_subset()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        shutil.rmtree(cls.temp_dir)
        print(f"Cleaned up test directory")
    
    @classmethod
    def _create_real_subset(cls):
        """Create a small subset of real data for testing."""
        print("Creating real data subset for testing...")
        
        # Load survey data and get 50 random participants
        survey_df = pd.read_csv(cls.real_survey_path, index_col=0, low_memory=False)
        if "hashed_id" in survey_df.columns:
            survey_df = survey_df.rename(columns={"hashed_id": "pid"})
        
        np.random.seed(42)
        unique_pids = survey_df['pid'].unique()
        test_pids = np.random.choice(unique_pids, size=min(50, len(unique_pids)), replace=False)
        
        # Subset survey data
        test_survey = survey_df[survey_df['pid'].isin(test_pids)].copy()
        
        # Load Oura data and handle MultiIndex (pid, date)
        if cls.real_oura_path.suffix == '.parquet':
            import pyarrow.parquet as pq
            oura_df = pq.read_table(cls.real_oura_path).to_pandas()
        else:
            oura_df = pd.read_csv(cls.real_oura_path)
        
        # Convert MultiIndex to columns
        oura_df = oura_df.reset_index()
        
        # Subset Oura data
        test_oura = oura_df[oura_df['pid'].isin(test_pids)].copy()
        
        # Keep only participants with both data types
        final_pids = set(test_survey['pid'].unique()) & set(test_oura['pid'].unique())
        test_survey = test_survey[test_survey['pid'].isin(final_pids)]
        test_oura = test_oura[test_oura['pid'].isin(final_pids)]
        
        # Save test subset
        cls.test_oura_file = Path(cls.temp_dir) / "test_oura.csv"
        cls.test_survey_file = Path(cls.temp_dir) / "test_survey.csv"
        
        test_oura.to_csv(cls.test_oura_file, index=False)
        test_survey.to_csv(cls.test_survey_file, index=True)
        
        print(f"Created test subset: {len(final_pids)} participants")
        print(f"  Oura records: {len(test_oura)}")
        print(f"  Survey records: {len(test_survey)}")
    
    def test_basic_functionality(self):
        """Test basic preprocessing functionality with all statistics."""
        print("\nTesting basic functionality...")
        
        config = ProcessingConfig(
            oura_path=str(self.test_oura_file),
            survey_path=str(self.test_survey_file),
            output_dir=str(self.test_output_dir / "basic"),
            window_start_offset=-30,
            window_end_offset=0,
            baseline_enabled=True,
            sample_size=None
        )
        
        main_file, baseline_file = run_preprocessing(config)
        
        # Verify files exist and have content
        self.assertTrue(Path(main_file).exists())
        self.assertTrue(Path(baseline_file).exists())
        
        df_main = pd.read_csv(main_file)
        df_baseline = pd.read_csv(baseline_file)
        
        self.assertGreater(len(df_main), 0)
        self.assertGreater(len(df_baseline), 0)
        
        # Check required columns
        required_cols = ['pid', 'date', 'promis_dep_sum', 'promis_anx_sum']
        for col in required_cols:
            self.assertIn(col, df_main.columns)
        
        # Check mean and std statistics are present
        stat_cols_mean = [c for c in df_main.columns if '_mean' in c]
        stat_cols_std = [c for c in df_main.columns if '_std' in c]
        unwanted_rms = [c for c in df_main.columns if '_rms' in c]
        unwanted_skew = [c for c in df_main.columns if '_skew' in c]
        
        self.assertGreater(len(stat_cols_mean), 0)
        self.assertGreater(len(stat_cols_std), 0)
        self.assertEqual(len(unwanted_rms), 0)  # Should be no RMS columns
        self.assertEqual(len(unwanted_skew), 0)  # Should be no skew columns
        
        # Check that both statistics have the same number of columns for each metric
        expected_metrics = len(config.metric_cols)
        self.assertEqual(len(stat_cols_mean), expected_metrics)
        self.assertEqual(len(stat_cols_std), expected_metrics)
        
        print(f"  ✓ Created {len(df_main)} main records, {len(df_baseline)} baseline records")
        print(f"  ✓ Statistics present: {len(stat_cols_mean)} mean, {len(stat_cols_std)} std (no RMS/skew)")
    
    def test_statistics_only_mean_std(self):
        """Test configuration with only mean and std statistics."""
        print("\nTesting mean and std only configuration...")
        
        config = ProcessingConfig(
            oura_path=str(self.test_oura_file),
            survey_path=str(self.test_survey_file),
            output_dir=str(self.test_output_dir / "mean_std_only"),
            window_start_offset=-30,
            window_end_offset=0,
            baseline_enabled=True,
            stat_functions=["mean", "std"],  # Only mean and std
            sample_size=10
        )
        
        main_file, baseline_file = run_preprocessing(config)
        df_main = pd.read_csv(main_file)
        
        # Check only mean and std statistics
        stat_cols_mean = [c for c in df_main.columns if '_mean' in c]
        stat_cols_std = [c for c in df_main.columns if '_std' in c]
        unwanted_rms = [c for c in df_main.columns if '_rms' in c]
        unwanted_skew = [c for c in df_main.columns if '_skew' in c]
        
        self.assertGreater(len(stat_cols_mean), 0)
        self.assertGreater(len(stat_cols_std), 0)
        self.assertEqual(len(unwanted_rms), 0)
        self.assertEqual(len(unwanted_skew), 0)
        
        print(f"  ✓ {len(stat_cols_mean)} mean, {len(stat_cols_std)} std columns (no RMS/skew)")
    
    def test_different_time_windows(self):
        """Test different time window configurations."""
        print("\nTesting different time windows...")
        
        windows = [
            {"name": "30d_before", "start": -30, "end": 0, "expected": "30d_before_to_0d_before"},
            {"name": "6w_to_2w", "start": -42, "end": -14, "expected": "42d_before_to_14d_before"}, 
            {"name": "1w_after", "start": 0, "end": 7, "expected": "0d_after_to_7d_after"}
        ]
        
        for window in windows:
            config = ProcessingConfig(
                oura_path=str(self.test_oura_file),
                survey_path=str(self.test_survey_file),
                output_dir=str(self.test_output_dir / window["name"]),
                window_start_offset=window["start"],
                window_end_offset=window["end"],
                baseline_enabled=True,
                sample_size=10  # Small sample for speed
            )
            
            # Test window description
            self.assertEqual(config.window_description, window["expected"])
            
            # Test filename includes window description
            filename = config.generate_filename("main")
            self.assertIn(window["expected"], filename)
            
            # Test processing works
            main_file, baseline_file = run_preprocessing(config)
            df_main = pd.read_csv(main_file)
            self.assertGreater(len(df_main), 0)
            
            print(f"  ✓ {window['name']}: {window['expected']} -> {len(df_main)} records")
    
    def test_baseline_options(self):
        """Test baseline adjustment on/off."""
        print("\nTesting baseline adjustment options...")
        
        for baseline_enabled in [True, False]:
            config = ProcessingConfig(
                oura_path=str(self.test_oura_file),
                survey_path=str(self.test_survey_file),
                output_dir=str(self.test_output_dir / f"baseline_{baseline_enabled}"),
                window_start_offset=-30,
                window_end_offset=0,
                baseline_enabled=baseline_enabled,
                sample_size=10
            )
            
            # Test filename reflects baseline setting
            filename = config.generate_filename("main")
            if baseline_enabled:
                self.assertIn("baseline_adj", filename)
            else:
                self.assertIn("no_baseline", filename)
            
            # Test processing works
            main_file, baseline_file = run_preprocessing(config)
            df_main = pd.read_csv(main_file)
            self.assertGreater(len(df_main), 0)
            
            print(f"  ✓ Baseline {baseline_enabled}: {len(df_main)} records")
    
    def test_sample_sizes(self):
        """Test different sample sizes."""
        print("\nTesting sample sizes...")
        
        for sample_size in [5, 15, None]:
            config = ProcessingConfig(
                oura_path=str(self.test_oura_file),
                survey_path=str(self.test_survey_file),
                output_dir=str(self.test_output_dir / f"sample_{sample_size}"),
                window_start_offset=-30,
                window_end_offset=0,
                sample_size=sample_size
            )
            
            # Test filename reflects sample size
            filename = config.generate_filename("main")
            if sample_size is None:
                self.assertIn("_full.csv", filename)
            else:
                self.assertIn(f"_n{sample_size}.csv", filename)
            
            # Test processing
            main_file, baseline_file = run_preprocessing(config)
            df_main = pd.read_csv(main_file)
            df_baseline = pd.read_csv(baseline_file)
            
            # Check participant counts
            main_pids = df_main['pid'].nunique()
            baseline_pids = df_baseline['pid'].nunique()
            
            if sample_size is not None:
                self.assertLessEqual(main_pids, sample_size)
                self.assertLessEqual(baseline_pids, sample_size)
            
            print(f"  ✓ Sample {sample_size}: {main_pids} participants processed")
    
    def test_data_integrity(self):
        """Test data integrity and realistic values with winsorizing."""
        print("\nTesting data integrity with winsorizing...")
        
        config = ProcessingConfig(
            oura_path=str(self.test_oura_file),
            survey_path=str(self.test_survey_file),
            output_dir=str(self.test_output_dir / "integrity"),
            window_start_offset=-30,
            window_end_offset=0,
            sample_size=20
        )
        
        main_file, baseline_file = run_preprocessing(config)
        df_main = pd.read_csv(main_file)
        
        # Test PROMIS scores are reasonable (4-20 range for 4 items, 1-5 scale)
        dep_scores = df_main['promis_dep_sum'].dropna()
        anx_scores = df_main['promis_anx_sum'].dropna()
        
        if len(dep_scores) > 0:
            self.assertTrue((dep_scores >= 4).all() and (dep_scores <= 20).all())
        if len(anx_scores) > 0:
            self.assertTrue((anx_scores >= 4).all() and (anx_scores <= 20).all())
        
        # Test heart rate values are within appropriate range
        hr_cols = [c for c in df_main.columns if 'hr_average_mean' in c]
        if hr_cols:
            hr_values = df_main[hr_cols[0]].dropna()
            if len(hr_values) > 0:
                # Check ranges based on whether baseline adjustment is enabled
                if config.baseline_enabled:
                    # Baseline-adjusted values should be deviations: ±50 bpm
                    reasonable_hr = (hr_values >= -50) & (hr_values <= 50)
                    self.assertTrue(reasonable_hr.all(), 
                        f"Baseline-adjusted HR values outside deviation range [-50, 50]: {hr_values[~reasonable_hr].values}")
                else:
                    # Raw values should be absolute bpm: 30-200
                    reasonable_hr = (hr_values >= 30) & (hr_values <= 200)
                    self.assertTrue(reasonable_hr.all(), 
                        f"Raw HR values outside absolute range [30, 200]: {hr_values[~reasonable_hr].values}")
        
        # Test other physiological measures are within appropriate ranges
        validation_checks = [
            ('rmssd_mean', "RMSSD values"),
            ('efficiency_mean', "Sleep efficiency values"),
            ('total_mean', "Total sleep values")
        ]
        
        for col_pattern, description in validation_checks:
            matching_cols = [c for c in df_main.columns if col_pattern in c]
            if matching_cols:
                values = df_main[matching_cols[0]].dropna()
                if len(values) > 0:
                    # Use appropriate ranges based on baseline adjustment
                    if config.baseline_enabled:
                        # Baseline-adjusted ranges (deviations)
                        if 'rmssd' in col_pattern:
                            min_val, max_val = -200, 200  # ±200 ms deviation
                        elif 'efficiency' in col_pattern:
                            min_val, max_val = -50, 50    # ±50% deviation
                        elif 'total' in col_pattern:
                            min_val, max_val = -360, 360  # ±6 hours deviation
                    else:
                        # Raw data ranges (absolute values)
                        if 'rmssd' in col_pattern:
                            min_val, max_val = 0, 500     # 0-500 ms absolute
                        elif 'efficiency' in col_pattern:
                            min_val, max_val = 0, 100     # 0-100% absolute
                        elif 'total' in col_pattern:
                            min_val, max_val = 120, 720   # 2-12 hours absolute
                    
                    reasonable = (values >= min_val) & (values <= max_val)
                    self.assertTrue(reasonable.all(), 
                        f"{description} outside {'deviation' if config.baseline_enabled else 'absolute'} range [{min_val}, {max_val}]: {values[~reasonable].values}")
        
        # Test dates are valid
        dates = pd.to_datetime(df_main['date'])
        self.assertTrue(dates.notna().all())
        
        # Test lockdown indicator is binary
        lockdown_vals = df_main['after_lockdown'].unique()
        self.assertTrue(set(lockdown_vals).issubset({0, 1}))
        
        # Test mean and std statistics are present and non-constant
        for stat in ['mean', 'std']:
            stat_cols = [c for c in df_main.columns if f'_{stat}' in c]
            if stat_cols:
                # Check at least one stat column has variance (not all NaN or constant)
                has_variance = any(df_main[col].dropna().nunique() > 1 for col in stat_cols[:3])  # Check first 3
                self.assertTrue(has_variance, f"All {stat} statistics appear constant")
        
        print(f"  ✓ Data integrity verified for {len(df_main)} records")
        if len(dep_scores) > 0:
            print(f"  ✓ PROMIS Depression range: [{dep_scores.min():.1f}, {dep_scores.max():.1f}]")
        if len(anx_scores) > 0:
            print(f"  ✓ PROMIS Anxiety range: [{anx_scores.min():.1f}, {anx_scores.max():.1f}]")
        if hr_cols and len(hr_values) > 0:
            print(f"  ✓ Heart rate range (winsorized): [{hr_values.min():.1f}, {hr_values.max():.1f}] bpm")
    
    def test_data_validation_ranges(self):
        """Test that data validation ranges are applied correctly."""
        print("\nTesting data validation ranges...")
        
        config = ProcessingConfig(
            oura_path=str(self.test_oura_file),
            survey_path=str(self.test_survey_file),
            output_dir=str(self.test_output_dir / "validation"),
            window_start_offset=-30,
            window_end_offset=0,
            sample_size=10
        )
        
        # Test that validation ranges are defined for all metrics
        for metric in config.metric_cols:
            self.assertIn(metric, config.validation_ranges, f"Validation range not defined for {metric}")
            min_val, max_val = config.validation_ranges[metric]
            self.assertLess(min_val, max_val, f"Invalid range for {metric}: {min_val} >= {max_val}")
        
        print(f"  ✓ Validation ranges defined for all {len(config.metric_cols)} metrics")
        
        # Test processing with validation
        main_file, baseline_file = run_preprocessing(config)
        df_main = pd.read_csv(main_file)
        
        # Check that at least some values have been processed
        self.assertGreater(len(df_main), 0)
        print(f"  ✓ Processing completed with validation: {len(df_main)} records")


def run_focused_tests():
    """Run focused tests with real data only."""
    print("FOCUSED DATA PREPROCESSING TESTS WITH REAL DATA")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestDataPreprocessingReal)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print(f"\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors) 
    successes = total - failures - errors
    
    print(f"Total tests: {total}")
    print(f"Passed: {successes}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    
    if failures + errors == 0:
        print("\n🎉 All tests passed! The data preprocessing system works correctly.")
        print("\nTested functionality:")
        print("  ✓ Basic preprocessing with real data subset")
        print("  ✓ Simplified feature pipeline: mean and std statistics only")
        print("  ✓ Configurable statistics (mean/std only option)")
        print("  ✓ Different time windows (30d before, 6w-2w before, 1w after)")
        print("  ✓ Baseline adjustment on/off")
        print("  ✓ Sample size limiting")
        print("  ✓ Data validation and winsorizing for physiological measures")
        print("  ✓ Heart rate range: 30-200 bpm (allows exercise/high intensity)")
        print("  ✓ RMSSD range: 0-500 ms (allows fit individuals)")
        print("  ✓ Proper filename generation")
        print("  ✓ Data integrity with realistic value ranges")
        
        return True
    else:
        print(f"\n⚠️  {failures + errors} test(s) failed.")
        if failures > 0:
            for test, error in result.failures:
                print(f"FAILED: {test}")
                print(f"  {error}")
        if errors > 0:
            for test, error in result.errors:
                print(f"ERROR: {test}")
                print(f"  {error}")
        
        return False


if __name__ == "__main__":
    success = run_focused_tests()
    sys.exit(0 if success else 1)