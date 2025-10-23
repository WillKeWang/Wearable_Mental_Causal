#############################################################################
# CONFIGURATION - Modify this section for different datasets
#############################################################################

# Dataset paths - Add or modify your datasets here
DATASET_PATHS = {
    "6w_to_2w_before": "data/preprocessed/full_run/6w_to_2w_before/survey_wearable_42d_before_to_14d_before_baseline_adj_full.csv",
    "1w_to_5w_after": "data/preprocessed/full_run/1w_to_5w_after/survey_wearable_7d_after_to_35d_after_baseline_adj_full.csv",
    # "4w_to_0w_before": "data/preprocessed/full_run/4w_to_0w_before/survey_wearable_28d_before_to_0d_before_baseline_adj_full.csv",
    # "0w_to_4w_after": "data/preprocessed/full_run/0w_to_4w_after/survey_wearable_0d_after_to_28d_after_baseline_adj_full.csv",
    # "5w_to_1w_before": "data/preprocessed/full_run/5w_to_1w_before/survey_wearable_35d_before_to_7d_before_baseline_adj_full.csv",

}

# Which datasets to use for visualization (must match keys in DATASET_PATHS)
BEFORE_DATASET = "6w_to_2w_before"
AFTER_DATASET = "1w_to_5w_after"
# BEFORE_DATASET = "4w_to_0w_before"
# AFTER_DATASET = "0w_to_4w_after"
# BEFORE_DATASET = "5w_to_1w_before"

# Output directories
EDGES_DIR = "data/edges"
FIGURES_DIR = "test_figures"

# Analysis parameters
ANALYSIS_PARAMS = {
    'n_bootstrap': 100,
    'sample_frac': 0.5,
    'alpha': 0.05,
    'use_pid_bootstrap': True,
    'use_first_survey': True,
    'min_frequency': 0.1
}

# Visualization parameters
VIZ_PARAMS = {
    'thresholds': [50, 60, 70, 80]
}

#############################################################################
# PIPELINE - No need to modify below this line
#############################################################################

def main():
    """Run the complete causal discovery and visualization pipeline."""
    
    # Step 1: Run causal discovery
    print("="*70)
    print("STEP 1: CAUSAL DISCOVERY ANALYSIS")
    print("="*70)
    print(f"Analyzing {len(DATASET_PATHS)} dataset(s)...")
    for name in DATASET_PATHS.keys():
        print(f"  - {name}")
    print()
    
    from causal_discovery import run_temporal_pc_analysis
    
    results = run_temporal_pc_analysis(
        DATASET_PATHS,
        **ANALYSIS_PARAMS,
        save_edges=True,
        output_dir=EDGES_DIR
    )
    
    print(f"\n✓ Step 1 complete: Edges saved to {EDGES_DIR}/")
    
    # Step 2: Create visualizations
    print("\n" + "="*70)
    print("STEP 2: CREATE VISUALIZATIONS")
    print("="*70)
    print(f"Creating plots for: {BEFORE_DATASET} vs {AFTER_DATASET}")
    print()
    
    from plot_causal_diagram_new import load_edges_for_analysis, create_visualizations
    
    # Load edges
    before_edges_full, after_edges_full = load_edges_for_analysis(
        before_dataset_name=BEFORE_DATASET,
        after_dataset_name=AFTER_DATASET,
        edges_dir=EDGES_DIR
    )
    
    # Create visualizations
    create_visualizations(
        before_edges_full=before_edges_full,
        after_edges_full=after_edges_full,
        before_dataset_name=BEFORE_DATASET,  
        after_dataset_name=AFTER_DATASET,   
        **VIZ_PARAMS,
        output_dir=FIGURES_DIR
    )
    
    print(f"\n✓ Step 2 complete: Plots saved to {FIGURES_DIR}/")
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Datasets analyzed: {list(DATASET_PATHS.keys())}")
    print(f"Edge files: {EDGES_DIR}/")
    print(f"  - {BEFORE_DATASET}_edges.json")
    print(f"  - {AFTER_DATASET}_edges.json")
    print(f"Visualizations: {FIGURES_DIR}/")
    for threshold in VIZ_PARAMS['thresholds']:
        print(f"  - causal_graph_{BEFORE_DATASET}_vs_{AFTER_DATASET}_{threshold}pct.png")
    print("="*70)


if __name__ == "__main__":
    main()