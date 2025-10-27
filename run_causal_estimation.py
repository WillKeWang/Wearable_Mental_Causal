"""
Causal Estimation Analysis - Ready to Run Locally
Execute this script from the directory containing causal_estimation_lmm.py
"""

from causal_estimation import run_causal_estimation

# Dataset paths as specified by user
BEFORE_PATH = 'data/preprocessed/full_run/6w_to_2w_before/survey_wearable_42d_before_to_14d_before_baseline_adj_full.csv'
AFTER_PATH = 'data/preprocessed/full_run/1w_to_5w_after/survey_wearable_7d_after_to_35d_after_baseline_adj_full.csv'

# Analysis configuration
BEFORE_TREATMENT = 'rem_std'
BEFORE_OUTCOME = 'promis_dep_sum'
BEFORE_COVARIATES = ['awake_std']

AFTER_TREATMENT = 'promis_dep_sum'
AFTER_OUTCOME = 'rem_std'
AFTER_COVARIATES = []

N_BOOTSTRAP = 100

print("\n" + "="*70)
print("CAUSAL ESTIMATION: REM-DEPRESSION BIDIRECTIONAL ANALYSIS")
print("="*70)
print("\nConfiguration:")
print(f"  BEFORE case: {BEFORE_PATH}")
print(f"    Causal link: {BEFORE_TREATMENT} → {BEFORE_OUTCOME}")
print(f"    Matching on: {BEFORE_COVARIATES}")
print(f"\n  AFTER case: {AFTER_PATH}")
print(f"    Causal link: {AFTER_TREATMENT} → {AFTER_OUTCOME}")
print(f"    Matching on: {AFTER_COVARIATES if AFTER_COVARIATES else 'None'}")
print(f"\n  Bootstrap iterations: {N_BOOTSTRAP}")
print(f"  Method: Linear Mixed Effects Models")
print(f"  Random effect: Subject (pid)")
print(f"  Baseline surveys: Excluded")
print("="*70)

# Run the analysis
try:
    results = run_causal_estimation(
        before_path=BEFORE_PATH,
        after_path=AFTER_PATH,
        before_treatment=BEFORE_TREATMENT,
        before_outcome=BEFORE_OUTCOME,
        before_covariates=BEFORE_COVARIATES,
        after_treatment=AFTER_TREATMENT,
        after_outcome=AFTER_OUTCOME,
        after_covariates=AFTER_COVARIATES,
        n_bootstrap=N_BOOTSTRAP
    )

    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nResults:")
    print("  - Visualization: causal_estimation_comparison.png")
    
    if results['before'] and results['after']:
        print("\nKey findings:")
        print(f"  BEFORE (REM → Depression):")
        print(f"    Coefficient: {results['before']['coefficient_mean']:7.4f} [{results['before']['coefficient_ci_lower']:6.4f}, {results['before']['coefficient_ci_upper']:6.4f}]")
        print(f"    Cohen's d:   {results['before']['cohens_d_mean']:7.4f} [{results['before']['cohens_d_ci_lower']:6.4f}, {results['before']['cohens_d_ci_upper']:6.4f}]")
        
        print(f"\n  AFTER (Depression → REM):")
        print(f"    Coefficient: {results['after']['coefficient_mean']:7.4f} [{results['after']['coefficient_ci_lower']:6.4f}, {results['after']['coefficient_ci_upper']:6.4f}]")
        print(f"    Cohen's d:   {results['after']['cohens_d_mean']:7.4f} [{results['after']['cohens_d_ci_lower']:6.4f}, {results['after']['cohens_d_ci_upper']:6.4f}]")
        
        # Asymmetry
        coef_diff = abs(results['before']['coefficient_mean']) - abs(results['after']['coefficient_mean'])
        print(f"\n  Effect asymmetry: {coef_diff:7.4f}")
        if abs(coef_diff) > 0.1:
            stronger = "Forward (REM → Depression)" if coef_diff > 0 else "Backward (Depression → REM)"
            print(f"  → {stronger} is stronger")
        else:
            print(f"  → Bidirectional (comparable magnitudes)")

except FileNotFoundError as e:
    print("\n" + "="*70)
    print("❌ ERROR: Dataset files not found")
    print("="*70)
    print(f"\nPlease upload the following files:")
    print(f"  1. {BEFORE_PATH}")
    print(f"  2. {AFTER_PATH}")
    print(f"\nMake sure the directory structure matches the paths above.")
    
except Exception as e:
    print("\n" + "="*70)
    print("❌ ERROR during analysis")
    print("="*70)
    print(f"\nError: {str(e)}")
    import traceback
    traceback.print_exc()