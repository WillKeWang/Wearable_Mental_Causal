"""
Test Temporal Causal Discovery Analysis
======================================
Test the PC algorithm functions on temporal datasets.
"""

from causal_discovery import (
    load_and_prepare_data,
    prepare_variables,
    create_background_knowledge,
    bootstrap_pc_analysis,
    analyze_single_dataset,
    run_temporal_pc_analysis,
    print_results,
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
    else:
        print("✗ Data loading failed")
    
    return df


def test_variable_preparation(df):
    """Test variable preparation."""
    print("\n" + "="*60)
    print("TEST 2: Variable Preparation")
    print("="*60)
    
    if df is not None:
        feature_names, X = prepare_variables(df)
        print(f"✓ Prepared {len(feature_names)} features")
        print(f"✓ Data matrix shape: {X.shape}")
        print(f"✓ Base vars included: {'promis_dep_sum' in feature_names and 'promis_anx_sum' in feature_names}")
        return feature_names, X
    else:
        print("✗ Cannot test - no data available")
        return None, None


def test_background_knowledge(feature_names):
    """Test background knowledge creation."""
    print("\n" + "="*60)
    print("TEST 3: Background Knowledge Creation")
    print("="*60)
    
    if feature_names is not None:
        base_vars = ["promis_dep_sum", "promis_anx_sum"]
        bk = create_background_knowledge(feature_names, base_vars)
        print("✓ Background knowledge created successfully")
        return bk
    else:
        print("✗ Cannot test - no feature names available")
        return None


def test_bootstrap_analysis(X, feature_names):
    """Test bootstrap analysis with small sample."""
    print("\n" + "="*60)
    print("TEST 4: Bootstrap Analysis (small sample)")
    print("="*60)
    
    if X is not None and feature_names is not None:
        base_vars = ["promis_dep_sum", "promis_anx_sum"]
        results = bootstrap_pc_analysis(
            X, feature_names, base_vars, 
            n_bootstrap=10,  # Small sample for testing
            sample_frac=0.6,
            alpha=0.05
        )
        
        print(f"✓ Bootstrap completed: {results['successful_iterations']}/10 successful")
        print(f"✓ Found {len(results['edge_counts'])} unique edges")
        print(f"✓ Key edge counts tracked:")
        print(f"    - rem_std → depression: {results['rem_dep_count']}")
        print(f"    - deep_std → depression: {results['deep_dep_count']}")
        print(f"    - anxiety ↔ depression: {results['anxiety_dep_count']}")
        
        return results
    else:
        print("✗ Cannot test - no data or feature names available")
        return None


def test_single_dataset_analysis():
    """Test complete single dataset analysis."""
    print("\n" + "="*60)
    print("TEST 5: Single Dataset Analysis")
    print("="*60)
    
    filepath = "data/preprocessed/full_run/1w_to_5w_after/survey_wearable_7d_after_to_35d_after_baseline_adj_full.csv"
    results = analyze_single_dataset(
        filepath, 
        "1w_to_5w_after",
        n_bootstrap=10,  # Small sample for testing
        sample_frac=0.6,
        alpha=0.05
    )
    
    if results:
        print("✓ Single dataset analysis completed")
        print_results(results, "1w_to_5w_after")
    else:
        print("✗ Single dataset analysis failed")
    
    return results


def test_full_temporal_analysis():
    """Test full temporal analysis with multiple datasets."""
    print("\n" + "="*60)
    print("TEST 6: Full Temporal Analysis")
    print("="*60)
    
    dataset_paths = {
        "1w_to_5w_after": "data/preprocessed/full_run/1w_to_5w_after/survey_wearable_7d_after_to_35d_after_baseline_adj_full.csv",
        "6w_to_2w_before": "data/preprocessed/full_run/6w_to_2w_before/survey_wearable_42d_before_to_14d_before_baseline_adj_full.csv"
    }
    
    results = run_temporal_pc_analysis(
        dataset_paths,
        n_bootstrap=10,  # Small sample for testing
        sample_frac=0.6,
        alpha=0.05
    )
    
    if all(results.values()):
        print("\n✓ Full temporal analysis completed successfully")
    else:
        print("\n✗ Some datasets failed to analyze")
    
    return results


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    # Test 1: Data loading
    df = test_data_loading()
    
    # Test 2: Variable preparation
    feature_names, X = test_variable_preparation(df)
    
    # Test 3: Background knowledge
    bk = test_background_knowledge(feature_names)
    
    # Test 4: Bootstrap analysis
    results = test_bootstrap_analysis(X, feature_names)
    
    # Test 5: Single dataset analysis
    single_results = test_single_dataset_analysis()
    
    # Test 6: Full temporal analysis
    full_results = test_full_temporal_analysis()
    
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
    
    # Run with full bootstrap iterations
    results = run_temporal_pc_analysis(
        dataset_paths,
        n_bootstrap=100,
        sample_frac=0.6,
        alpha=0.05
    )
    
    return results


if __name__ == "__main__":
    # Run tests first
    run_all_tests()
    
    # run full analysis
    main()