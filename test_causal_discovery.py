"""
Test Temporal Causal Discovery Analysis
======================================
Test the PC algorithm functions on temporal datasets.
"""

from causal_discovery import (
    load_and_prepare_data,
    prepare_variables,
    get_baseline_data,
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
        print("✓ Data loaded successfully")
        print(f"✓ Shape: {df.shape}")
        print(f"✓ Required columns present: {'promis_dep_sum' in df.columns and 'promis_anx_sum' in df.columns}")
        print(f"✓ PID column present: {'pid' in df.columns}")
    else:
        print("✗ Data loading failed")
    
    return df


def test_baseline_extraction(df):
    """Test baseline data extraction."""
    print("\n" + "="*60)
    print("TEST 2: Baseline Data Extraction")
    print("="*60)
    
    if df is not None:
        baseline_df = get_baseline_data(df)
        print(f"✓ Original data: {df.shape[0]} rows, {df['pid'].nunique()} unique PIDs")
        print(f"✓ Baseline data: {baseline_df.shape[0]} rows")
        print(f"✓ One row per PID: {baseline_df.shape[0] == baseline_df['pid'].nunique()}")
        return baseline_df
    else:
        print("✗ Cannot test - no data available")
        return None


def test_variable_preparation(df):
    """Test variable preparation."""
    print("\n" + "="*60)
    print("TEST 3: Variable Preparation")
    print("="*60)
    
    if df is not None:
        # Test with PID
        feature_names_pid, X_pid, pids = prepare_variables(df, keep_pid=True)
        print(f"✓ Prepared {len(feature_names_pid)} features (with PID)")
        print(f"✓ Data matrix shape: {X_pid.shape}")
        print(f"✓ PIDs array shape: {pids.shape}")
        print(f"✓ Base vars included: {'promis_dep_sum' in feature_names_pid and 'promis_anx_sum' in feature_names_pid}")
        
        return feature_names_pid, X_pid
    else:
        print("✗ Cannot test - no data available")
        return None, None


def test_background_knowledge(feature_names):
    """Test background knowledge creation."""
    print("\n" + "="*60)
    print("TEST 4: Background Knowledge Creation")
    print("="*60)
    
    if feature_names is not None:
        base_vars = ["promis_dep_sum", "promis_anx_sum"]
        bk = create_background_knowledge(feature_names, base_vars)
        print("✓ Background knowledge created successfully")
        return bk
    else:
        print("✗ Cannot test - no feature names available")
        return None


def test_bootstrap_analysis_pidlevel(df):
    """Test PID-level bootstrap analysis with baseline surveys."""
    print("\n" + "="*60)
    print("TEST 5: Bootstrap Analysis (PID-level baseline, small sample)")
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
            use_baseline=True
        )
        
        print(f"✓ Bootstrap completed: {results['successful_iterations']}/10 successful")
        print(f"✓ Found {len(results['edge_counts'])} unique edges")
        print(f"✓ Key edge counts tracked:")
        print(f"    - rem_std → depression: {results['rem_dep_count']}")
        print(f"    - deep_std → depression: {results['deep_dep_count']}")
        print(f"    - anxiety ↔ depression: {results['anxiety_dep_count']}")
        
        return results
    else:
        print("✗ Cannot test - no data available")
        return None


def test_single_dataset_analysis_pidlevel():
    """Test complete single dataset analysis with PID-level bootstrapping."""
    print("\n" + "="*60)
    print("TEST 6: Single Dataset Analysis (PID-level baseline)")
    print("="*60)
    
    filepath = "data/preprocessed/full_run/1w_to_5w_after/survey_wearable_7d_after_to_35d_after_baseline_adj_full.csv"
    results = analyze_single_dataset(
        filepath, 
        "1w_to_5w_after",
        n_bootstrap=10,  # Small sample for testing
        sample_frac=0.6,
        alpha=0.05,
        use_pid_bootstrap=True,
        use_baseline=True
    )
    
    if results:
        print("✓ Single dataset analysis completed")
        print_results(results, "1w_to_5w_after", min_frequency=0.2)  # 20% threshold for small test
    else:
        print("✗ Single dataset analysis failed")
    
    return results


def test_full_temporal_analysis_pidlevel():
    """Test full temporal analysis with PID-level bootstrapping."""
    print("\n" + "="*60)
    print("TEST 7: Full Temporal Analysis (PID-level baseline)")
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
        use_baseline=True,
        min_frequency=0.2  # 20% threshold for small test sample
    )
    
    if all(results.values()):
        print("\n✓ Full temporal analysis completed successfully")
    else:
        print("\n✗ Some datasets failed to analyze")
    
    return results


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING ALL TESTS (PID-LEVEL BOOTSTRAPPING)")
    print("="*60)
    
    # Test 1: Data loading
    df = test_data_loading()
    
    # Test 2: Baseline extraction
    baseline_df = test_baseline_extraction(df)
    
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
    
    # Run with full bootstrap iterations using PID-level sampling on baseline surveys
    # Show edges that appear in ≥10% of bootstrap iterations
    results = run_temporal_pc_analysis(
        dataset_paths,
        n_bootstrap=100,
        sample_frac=0.6,
        alpha=0.05,
        use_pid_bootstrap=True,
        use_baseline=True,
        min_frequency=0.1
    )
    
    return results


if __name__ == "__main__":
    # Run tests first
    run_all_tests()
    
    # Run full analysis
    print("\n\n")
    main()