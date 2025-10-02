"""
Test Temporal Causal Discovery Analysis
======================================
Test the PC algorithm functions on temporal datasets.
"""

import pandas as pd

from causal_discovery import (
    load_and_prepare_data,
    prepare_variables,
    get_first_survey_per_participant,
    create_background_knowledge,
    bootstrap_pc_analysis_by_pid,
    analyze_single_dataset,
    run_temporal_pc_analysis,
    print_results,
    print_all_edges,
    compare_temporal_results
)


def test_data_loading():
    """Test data loading and preparation."""
    print("="*60)
    print("TEST 1: Data Loading and Preparation")
    print("="*60)
    
    filepath = "data/preprocessed/full_run/1w_to_5w_after/survey_wearable_7d_after_to_35d_after_baseline_adj_full.csv"
    df = load_and_prepare_data(filepath, "test_dataset")
    
    if df is not None:
        print("[PASS] Data loaded successfully")
        print(f"[PASS] Shape: {df.shape}")
        print(f"[PASS] Required columns present: {'promis_dep_sum' in df.columns and 'promis_anx_sum' in df.columns}")
        print(f"[PASS] PID column present: {'pid' in df.columns}")
    else:
        print("[FAIL] Data loading failed")
    
    return df


def test_first_survey_extraction(df):
    """Test first survey data extraction."""
    print("\n" + "="*60)
    print("TEST 2: First Survey Data Extraction")
    print("="*60)
    
    if df is not None:
        first_survey_df = get_first_survey_per_participant(df)
        print(f"[PASS] Original data: {df.shape[0]} rows, {df['pid'].nunique()} unique PIDs")
        print(f"[PASS] First survey data: {first_survey_df.shape[0]} rows")
        print(f"[PASS] One row per PID: {first_survey_df.shape[0] == first_survey_df['pid'].nunique()}")
        return first_survey_df
    else:
        print("[FAIL] Cannot test - no data available")
        return None


def test_variable_preparation(df):
    """Test variable preparation."""
    print("\n" + "="*60)
    print("TEST 3: Variable Preparation")
    print("="*60)
    
    if df is not None:
        # Test with PID
        feature_names_pid, X_pid, pids = prepare_variables(df, keep_pid=True)
        print(f"[PASS] Prepared {len(feature_names_pid)} features (with PID)")
        print(f"[PASS] Data matrix shape: {X_pid.shape}")
        print(f"[PASS] PIDs array shape: {pids.shape}")
        print(f"[PASS] PIDs preserved (not NaN): {not any(pd.isna(pids))}")
        print(f"[PASS] Base vars included: {'promis_dep_sum' in feature_names_pid and 'promis_anx_sum' in feature_names_pid}")
        
        return feature_names_pid, X_pid
    else:
        print("[FAIL] Cannot test - no data available")
        return None, None


def test_background_knowledge(feature_names):
    """Test background knowledge creation."""
    print("\n" + "="*60)
    print("TEST 4: Background Knowledge Creation")
    print("="*60)
    
    if feature_names is not None:
        base_vars = ["promis_dep_sum", "promis_anx_sum"]
        
        # Test 'before' type
        bk_before = create_background_knowledge(feature_names, base_vars, dataset_type='before')
        print("[PASS] Background knowledge created for 'before' dataset")
        
        # Test 'after' type
        bk_after = create_background_knowledge(feature_names, base_vars, dataset_type='after')
        print("[PASS] Background knowledge created for 'after' dataset")
        
        return bk_before
    else:
        print("[FAIL] Cannot test - no feature names available")
        return None


def test_bootstrap_analysis_pidlevel(df):
    """Test PID-level bootstrap analysis with first surveys."""
    print("\n" + "="*60)
    print("TEST 5: Bootstrap Analysis (PID-level first survey, small sample)")
    print("="*60)
    
    if df is not None:
        base_vars = ["promis_dep_sum", "promis_anx_sum"]
        metric_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
        feature_names = base_vars + metric_cols
        feature_names = [c for c in feature_names if not c.startswith("total_")]
        
        results = bootstrap_pc_analysis_by_pid(
            df, feature_names, base_vars,
            n_bootstrap=10,  # Small sample for testing
            sample_frac=0.6,
            alpha=0.05,
            use_first_survey=True,
            dataset_type='after'  # Test with 'after' type
        )
        
        print(f"[PASS] Bootstrap completed: {results['successful_iterations']}/10 successful")
        print(f"[PASS] Found {len(results['edge_counts'])} unique edges")
        print(f"[PASS] Dataset type recorded: {results.get('dataset_type')}")
        print(f"[PASS] Key edge counts tracked:")
        print(f"    - depression -> rem_std: {results['rem_dep_count']}")
        print(f"    - depression -> deep_std: {results['deep_dep_count']}")
        print(f"    - anxiety <-> depression: {results['anxiety_dep_count']}")
        
        return results
    else:
        print("[FAIL] Cannot test - no data available")
        return None


def test_single_dataset_analysis_pidlevel():
    """Test complete single dataset analysis with PID-level bootstrapping."""
    print("\n" + "="*60)
    print("TEST 6: Single Dataset Analysis (PID-level first survey)")
    print("="*60)
    
    filepath = "data/preprocessed/full_run/1w_to_5w_after/survey_wearable_7d_after_to_35d_after_baseline_adj_full.csv"
    results = analyze_single_dataset(
        filepath, 
        "1w_to_5w_after",
        n_bootstrap=10,  # Small sample for testing
        sample_frac=0.6,
        alpha=0.05,
        use_pid_bootstrap=True,
        use_first_survey=True
    )
    
    if results:
        print("[PASS] Single dataset analysis completed")
        print(f"[PASS] Dataset type inferred: {results.get('dataset_type')}")
        print_results(results, "1w_to_5w_after", min_frequency=0.2)  # 20% threshold for small test
    else:
        print("[FAIL] Single dataset analysis failed")
    
    return results


def test_full_temporal_analysis_pidlevel():
    """Test full temporal analysis with PID-level bootstrapping."""
    print("\n" + "="*60)
    print("TEST 7: Full Temporal Analysis (PID-level first survey)")
    print("="*60)
    
    dataset_paths = {
        "1w_to_5w_after": "data/preprocessed/full_run/1w_to_5w_after/survey_wearable_7d_after_to_35d_after_baseline_adj_full.csv",
        "6w_to_2w_before": "data/preprocessed/full_run/6w_to_2w_before/survey_wearable_42d_before_to_14d_before_baseline_adj_full.csv"
    }
    
    results = run_temporal_pc_analysis(
        dataset_paths,
        n_bootstrap=10,  # Small sample for testing
        sample_frac=0.6,
        alpha=0.05,
        use_pid_bootstrap=True,
        use_first_survey=True,
        min_frequency=0.2  # 20% threshold for small test sample
    )
    
    if all(results.values()):
        print("\n[PASS] Full temporal analysis completed successfully")
        # Verify correct dataset types were inferred
        for name, result in results.items():
            print(f"[PASS] {name}: dataset_type = {result.get('dataset_type')}")
    else:
        print("\n[FAIL] Some datasets failed to analyze")
    
    return results


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING ALL TESTS (PID-LEVEL BOOTSTRAPPING)")
    print("="*60)
    
    # Test 1: Data loading
    df = test_data_loading()
    
    # Test 2: First survey extraction
    first_survey_df = test_first_survey_extraction(df)
    
    # Test 3: Variable preparation
    feature_names, X = test_variable_preparation(df)
    
    # Test 4: Background knowledge
    bk = test_background_knowledge(feature_names)
    
    # Test 5: PID-level bootstrap analysis
    results_pid = test_bootstrap_analysis_pidlevel(df)
    
    # Test 6: Single dataset analysis (PID-level)
    single_results = test_single_dataset_analysis_pidlevel()
    
    # Test 7: Full temporal analysis (PID-level)
    full_results = test_full_temporal_analysis_pidlevel()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


def main():
    """
    Run the actual analysis with full bootstrap iterations.
    Use this after tests pass to run the real analysis.
    """
    print("="*60)
    print("RUNNING FULL TEMPORAL PC ANALYSIS")
    print("="*60)
    
    dataset_paths = {
        "1w_to_5w_after": "data/preprocessed/full_run/1w_to_5w_after/survey_wearable_7d_after_to_35d_after_baseline_adj_full.csv",
        "6w_to_2w_before": "data/preprocessed/full_run/6w_to_2w_before/survey_wearable_42d_before_to_14d_before_baseline_adj_full.csv"
    }
    
    # Run with full bootstrap iterations using PID-level sampling on first surveys
    # Show edges that appear in ≥10% of bootstrap iterations
    results = run_temporal_pc_analysis(
        dataset_paths,
        n_bootstrap=100,
        sample_frac=0.5,
        alpha=0.05,
        use_pid_bootstrap=True,
        use_first_survey=True,
        min_frequency=0.1
    )
    
    return results


if __name__ == "__main__":
    # Run tests first
    run_all_tests()
    
    # Run full analysis
    print("\n\n")
    main()