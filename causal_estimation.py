"""
Causal Effects Estimation using Linear Mixed Effects Models
===========================================================
OPTIMIZED VERSION with:
1. Fast PSM using sklearn NearestNeighbors (O(n log n) instead of O(n²))
2. PSM done once before bootstrap (not inside loop)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')


def load_and_clean_data(filepath, dataset_name):
    """Load and clean the dataset."""
    print(f"\n{'='*70}")
    print(f"Loading {dataset_name}")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Clean PROMIS scores
    df_clean = df.dropna(axis=0, how='any')
    df_clean = df_clean[
        (df_clean['promis_dep_sum'] >= 4) & (df_clean['promis_dep_sum'] <= 20) &
        (df_clean['promis_anx_sum'] >= 4) & (df_clean['promis_anx_sum'] <= 20)
    ]
    
    print(f"After cleaning: {df_clean.shape}")
    print(f"Unique participants: {df_clean['pid'].nunique()}")
    
    return df_clean


def remove_first_surveys(df):
    """Remove the first (baseline) survey for each participant."""
    print(f"\n{'='*70}")
    print("Removing First Surveys (Baseline)")
    print(f"{'='*70}")
    
    # Try to find a time column
    time_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'day' in c.lower()]
    
    if time_cols:
        time_col = time_cols[0]
        print(f"Using time column: {time_col}")
        df_sorted = df.sort_values([time_col, 'pid'])
        first_survey_mask = df_sorted.duplicated(subset='pid', keep='first')
        df_no_baseline = df_sorted[first_survey_mask].copy()
    else:
        print("No time column found, using first occurrence per pid")
        first_survey_mask = df.duplicated(subset='pid', keep='first')
        df_no_baseline = df[first_survey_mask].copy()
    
    print(f"Original surveys: {len(df)}")
    print(f"First surveys removed: {len(df) - len(df_no_baseline)}")
    print(f"Remaining surveys: {len(df_no_baseline)}")
    print(f"Participants in final data: {df_no_baseline['pid'].nunique()}")
    
    return df_no_baseline


def create_treatment_groups(df, treatment_var, percentile_threshold=50):
    """Create binary treatment groups based on median split."""
    threshold = df[treatment_var].quantile(percentile_threshold / 100)
    df['treated'] = (df[treatment_var] >= threshold).astype(int)
    
    print(f"\nTreatment variable: {treatment_var}")
    print(f"Threshold (median): {threshold:.4f}")
    print(f"Control group (0): {(df['treated']==0).sum()} observations")
    print(f"Treated group (1): {(df['treated']==1).sum()} observations")
    
    return df


def propensity_score_matching(df, treatment_var, outcome_var, covariates, caliper=0.1):
    """
    FAST propensity score matching using sklearn NearestNeighbors.
    
    Speed: O(n log n) using ball tree instead of O(n²) nested loops.
    This is 100-1000x faster for large datasets.
    """
    print(f"\n{'='*70}")
    print("Fast Propensity Score Matching")
    print(f"{'='*70}")
    print(f"Treatment: {treatment_var}")
    print(f"Outcome: {outcome_var}")
    print(f"Covariates: {covariates}")
    
    if not covariates:
        print("No covariates - skipping matching, using all data")
        return df
    
    start_time = time.time()
    
    # Prepare data
    X = df[covariates].values
    y = df['treated'].values
    
    # Fit propensity score model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X_scaled, y)
    
    # Get propensity scores
    ps_scores = ps_model.predict_proba(X_scaled)[:, 1]
    df['ps_score'] = ps_scores
    
    print(f"\nPropensity scores: mean={ps_scores.mean():.3f}, "
          f"std={ps_scores.std():.3f}, "
          f"range=[{ps_scores.min():.3f}, {ps_scores.max():.3f}]")
    
    # Separate treated and control
    treated_df = df[df['treated'] == 1].copy()
    control_df = df[df['treated'] == 0].copy()
    
    treated_ps = treated_df['ps_score'].values.reshape(-1, 1)
    control_ps = control_df['ps_score'].values.reshape(-1, 1)
    
    print(f"\nTreated group: {len(treated_df)} observations")
    print(f"Control group: {len(control_df)} observations")
    
    # Calculate caliper
    caliper_value = caliper * np.std(ps_scores)
    print(f"Caliper: {caliper_value:.4f}")
    
    # Use NearestNeighbors for fast matching
    print("\n⚡ Performing fast nearest-neighbor matching...")
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='manhattan')
    nn.fit(control_ps)
    
    # Find nearest control for each treated
    distances, indices = nn.kneighbors(treated_ps)
    distances = distances.flatten()
    indices = indices.flatten()
    
    # Apply caliper and create matches
    matched_pairs = []
    used_controls = set()
    
    # Sort by distance to prioritize best matches
    match_order = np.argsort(distances)
    
    for i in match_order:
        distance = distances[i]
        control_idx = indices[i]
        
        # Check caliper and if control already used
        if distance <= caliper_value and control_idx not in used_controls:
            treated_idx = i
            matched_pairs.append((
                treated_df.index[treated_idx],
                control_df.index[control_idx]
            ))
            used_controls.add(control_idx)
    
    print(f"Matched {len(matched_pairs)} pairs")
    
    # Create matched dataset
    matched_indices = []
    for t_idx, c_idx in matched_pairs:
        matched_indices.extend([t_idx, c_idx])
    
    matched_df = df.loc[matched_indices].copy()
    
    elapsed = time.time() - start_time
    print(f"Matched dataset: {len(matched_df)} observations from {matched_df['pid'].nunique()} participants")
    print(f"⏱️  Matching completed in {elapsed:.2f} seconds")
    
    # Check balance
    check_covariate_balance(matched_df, covariates)
    
    return matched_df


def check_covariate_balance(df, covariates):
    """Check covariate balance after matching."""
    if not covariates:
        return
    
    print(f"\n{'='*70}")
    print("Covariate Balance Check")
    print(f"{'='*70}")
    
    treated = df[df['treated'] == 1]
    control = df[df['treated'] == 0]
    
    for covar in covariates:
        mean_t = treated[covar].mean()
        mean_c = control[covar].mean()
        
        var_t = treated[covar].var()
        var_c = control[covar].var()
        pooled_std = np.sqrt((var_t + var_c) / 2)
        
        smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0
        
        status = "✓" if abs(smd) < 0.25 else "✗"
        print(f"{status} {covar:25s}: SMD = {smd:6.3f} (T:{mean_t:7.3f}, C:{mean_c:7.3f})")


def estimate_effect_lmm(df, treatment_var, outcome_var, subject_var='pid'):
    """Estimate treatment effect using Linear Mixed Effects Model."""
    # Prepare data
    df_model = df[['treated', outcome_var, subject_var]].copy()
    df_model = df_model.dropna()
    
    # Fit LMM: outcome ~ treated + (1|subject)
    model = MixedLM.from_formula(
        f'{outcome_var} ~ treated',
        data=df_model,
        groups=df_model[subject_var]
    )
    
    result = model.fit(reml=True, method='powell')
    
    # Extract results
    coef = result.params['treated']
    se = result.bse['treated']
    t_stat = result.tvalues['treated']
    p_value = result.pvalues['treated']
    ci_lower, ci_upper = result.conf_int().loc['treated']
    
    # Calculate effect size (Cohen's d approximation)
    treated_vals = df_model[df_model['treated'] == 1][outcome_var]
    control_vals = df_model[df_model['treated'] == 0][outcome_var]
    pooled_std = np.sqrt((treated_vals.var() + control_vals.var()) / 2)
    cohens_d = coef / pooled_std if pooled_std > 0 else 0
    
    return {
        'coefficient': coef,
        'se': se,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cohens_d': cohens_d,
        'n_obs': len(df_model),
        'n_subjects': df_model[subject_var].nunique()
    }


def bootstrap_analysis(df_matched, treatment_var, outcome_var, 
                       n_bootstrap=100, sample_frac=0.8, subject_var='pid'):
    """
    Bootstrap analysis on ALREADY MATCHED data.
    No matching happens here - only resampling and estimation.
    """
    print(f"\n{'='*70}")
    print(f"Bootstrap Analysis on Matched Data ({n_bootstrap} iterations)")
    print(f"{'='*70}")
    
    bootstrap_results = []
    unique_subjects = df_matched[subject_var].unique()
    n_subjects = len(unique_subjects)
    
    print(f"Starting bootstrap with {n_subjects} participants, {len(df_matched)} observations")
    
    for i in tqdm(range(n_bootstrap), desc="Bootstrapping"):
        # Sample participants with replacement
        n_sample = int(n_subjects * sample_frac)
        sampled_subjects = np.random.choice(unique_subjects, size=n_sample, replace=True)
        
        # Get data for sampled participants
        df_boot = df_matched[df_matched[subject_var].isin(sampled_subjects)].copy()
        
        if len(df_boot) < 20:
            continue
        
        try:
            # Estimate effect using LMM
            result = estimate_effect_lmm(df_boot, treatment_var, outcome_var, subject_var)
            bootstrap_results.append(result)
            
        except Exception as e:
            continue
    
    print(f"\nSuccessful iterations: {len(bootstrap_results)}/{n_bootstrap}")
    
    if len(bootstrap_results) == 0:
        print("No successful bootstrap iterations!")
        return None
    
    # Aggregate results
    coefficients = [r['coefficient'] for r in bootstrap_results]
    ses = [r['se'] for r in bootstrap_results]
    p_values = [r['p_value'] for r in bootstrap_results]
    cohens_ds = [r['cohens_d'] for r in bootstrap_results]
    
    bootstrap_summary = {
        'coefficient_mean': np.mean(coefficients),
        'coefficient_std': np.std(coefficients),
        'coefficient_median': np.median(coefficients),
        'coefficient_ci_lower': np.percentile(coefficients, 2.5),
        'coefficient_ci_upper': np.percentile(coefficients, 97.5),
        'se_mean': np.mean(ses),
        'p_value_median': np.median(p_values),
        'cohens_d_mean': np.mean(cohens_ds),
        'cohens_d_median': np.median(cohens_ds),
        'cohens_d_ci_lower': np.percentile(cohens_ds, 2.5),
        'cohens_d_ci_upper': np.percentile(cohens_ds, 97.5),
        'n_successful': len(bootstrap_results),
        'all_coefficients': coefficients,
        'all_cohens_d': cohens_ds
    }
    
    return bootstrap_summary


def print_bootstrap_summary(summary, label):
    """Print bootstrap summary statistics."""
    print(f"\n{'='*70}")
    print(f"Bootstrap Summary: {label}")
    print(f"{'='*70}")
    print(f"Successful iterations: {summary['n_successful']}")
    print(f"\nCoefficient:")
    print(f"  Mean:        {summary['coefficient_mean']:7.4f}")
    print(f"  Median:      {summary['coefficient_median']:7.4f}")
    print(f"  Std Dev:     {summary['coefficient_std']:7.4f}")
    print(f"  95% CI:      [{summary['coefficient_ci_lower']:6.4f}, {summary['coefficient_ci_upper']:6.4f}]")
    print(f"\nCohen's d:")
    print(f"  Mean:        {summary['cohens_d_mean']:7.4f}")
    print(f"  Median:      {summary['cohens_d_median']:7.4f}")
    print(f"  95% CI:      [{summary['cohens_d_ci_lower']:6.4f}, {summary['cohens_d_ci_upper']:6.4f}]")
    print(f"\nMedian p-value: {summary['p_value_median']:.4f}")


def visualize_bootstrap_results(before_summary, after_summary, output_path):
    """Create visualization comparing forward and backward effects."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Coefficient distributions
    ax = axes[0, 0]
    ax.hist(before_summary['all_coefficients'], bins=30, alpha=0.6, 
            label='Before (REM → Depression)', color='blue', density=True)
    ax.hist(after_summary['all_coefficients'], bins=30, alpha=0.6,
            label='After (Depression → REM)', color='red', density=True)
    ax.axvline(before_summary['coefficient_mean'], color='blue', linestyle='--', linewidth=2)
    ax.axvline(after_summary['coefficient_mean'], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Coefficient', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Coefficient Distributions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Cohen's d distributions
    ax = axes[0, 1]
    ax.hist(before_summary['all_cohens_d'], bins=30, alpha=0.6,
            label='Before (REM → Depression)', color='blue', density=True)
    ax.hist(after_summary['all_cohens_d'], bins=30, alpha=0.6,
            label='After (Depression → REM)', color='red', density=True)
    ax.axvline(before_summary['cohens_d_mean'], color='blue', linestyle='--', linewidth=2)
    ax.axvline(after_summary['cohens_d_mean'], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel("Cohen's d", fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title("Effect Size Distributions", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Comparison bar plot for coefficients
    ax = axes[1, 0]
    conditions = ['Before\n(REM → Dep)', 'After\n(Dep → REM)']
    means = [before_summary['coefficient_mean'], after_summary['coefficient_mean']]
    cis_lower = [before_summary['coefficient_ci_lower'], after_summary['coefficient_ci_lower']]
    cis_upper = [before_summary['coefficient_ci_upper'], after_summary['coefficient_ci_upper']]
    errors_lower = [means[i] - cis_lower[i] for i in range(2)]
    errors_upper = [cis_upper[i] - means[i] for i in range(2)]
    
    colors = ['blue', 'red']
    bars = ax.bar(conditions, means, color=colors, alpha=0.6, edgecolor='black')
    ax.errorbar(conditions, means, yerr=[errors_lower, errors_upper], 
                fmt='none', color='black', capsize=5, linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('Mean Coefficient', fontsize=11)
    ax.set_title('Coefficient Comparison (with 95% CI)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 4: Comparison bar plot for Cohen's d
    ax = axes[1, 1]
    means_d = [before_summary['cohens_d_mean'], after_summary['cohens_d_mean']]
    cis_lower_d = [before_summary['cohens_d_ci_lower'], after_summary['cohens_d_ci_lower']]
    cis_upper_d = [before_summary['cohens_d_ci_upper'], after_summary['cohens_d_ci_upper']]
    errors_lower_d = [means_d[i] - cis_lower_d[i] for i in range(2)]
    errors_upper_d = [cis_upper_d[i] - means_d[i] for i in range(2)]
    
    bars = ax.bar(conditions, means_d, color=colors, alpha=0.6, edgecolor='black')
    ax.errorbar(conditions, means_d, yerr=[errors_lower_d, errors_upper_d],
                fmt='none', color='black', capsize=5, linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel("Mean Cohen's d", fontsize=11)
    ax.set_title("Effect Size Comparison (with 95% CI)", fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    return fig


def run_causal_estimation(before_path, after_path,
                         before_treatment, before_outcome, before_covariates,
                         after_treatment, after_outcome, after_covariates,
                         n_bootstrap=100):
    """
    Run the complete causal estimation analysis.
    
    OPTIMIZED VERSION:
    - Fast PSM using sklearn NearestNeighbors
    - PSM done once per dataset
    - Bootstrap only for effect estimation
    """
    print("\n" + "="*70)
    print("CAUSAL EFFECTS ESTIMATION - OPTIMIZED VERSION")
    print("="*70)
    
    # Load and prepare data
    df_before = load_and_clean_data(before_path, "BEFORE Case (-6 to -2 weeks)")
    df_after = load_and_clean_data(after_path, "AFTER Case (1 to 5 weeks)")
    
    # Remove first surveys
    df_before = remove_first_surveys(df_before)
    df_after = remove_first_surveys(df_after)
    
    # ============================================================
    # BEFORE CASE: rem_std -> depression
    # ============================================================
    print("\n" + "="*70)
    print("BEFORE CASE: REM Variability → Depression")
    print("="*70)
    print(f"Treatment: {before_treatment}")
    print(f"Outcome: {before_outcome}")
    print(f"Covariates: {before_covariates}")
    
    df_before = create_treatment_groups(df_before, before_treatment)
    df_before_matched = propensity_score_matching(
        df_before, before_treatment, before_outcome, before_covariates, caliper=0.1
    )
    
    before_summary = bootstrap_analysis(
        df_before_matched, before_treatment, before_outcome, n_bootstrap=n_bootstrap
    )
    
    if before_summary:
        print_bootstrap_summary(before_summary, "BEFORE Case")
    
    # ============================================================
    # AFTER CASE: depression -> rem_std
    # ============================================================
    print("\n" + "="*70)
    print("AFTER CASE: Depression → REM Variability")
    print("="*70)
    print(f"Treatment: {after_treatment}")
    print(f"Outcome: {after_outcome}")
    print(f"Covariates: {after_covariates}")
    
    df_after = create_treatment_groups(df_after, after_treatment)
    
    if after_covariates:
        df_after_matched = propensity_score_matching(
            df_after, after_treatment, after_outcome, after_covariates, caliper=0.1
        )
    else:
        df_after_matched = df_after
        print("\nNo covariates specified - using full dataset")
    
    after_summary = bootstrap_analysis(
        df_after_matched, after_treatment, after_outcome, n_bootstrap=n_bootstrap
    )
    
    if after_summary:
        print_bootstrap_summary(after_summary, "AFTER Case")
    
    # Visualization
    if before_summary and after_summary:
        output_path = 'causal_estimation_comparison.png'
        visualize_bootstrap_results(before_summary, after_summary, output_path)
    
    # Summary comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    if before_summary and after_summary:
        print(f"\nBEFORE (REM → Depression):")
        print(f"  Coefficient: {before_summary['coefficient_mean']:7.4f} "
              f"[{before_summary['coefficient_ci_lower']:6.4f}, {before_summary['coefficient_ci_upper']:6.4f}]")
        print(f"  Cohen's d:   {before_summary['cohens_d_mean']:7.4f} "
              f"[{before_summary['cohens_d_ci_lower']:6.4f}, {before_summary['cohens_d_ci_upper']:6.4f}]")
        
        print(f"\nAFTER (Depression → REM):")
        print(f"  Coefficient: {after_summary['coefficient_mean']:7.4f} "
              f"[{after_summary['coefficient_ci_lower']:6.4f}, {after_summary['coefficient_ci_upper']:6.4f}]")
        print(f"  Cohen's d:   {after_summary['cohens_d_mean']:7.4f} "
              f"[{after_summary['cohens_d_ci_lower']:6.4f}, {after_summary['cohens_d_ci_upper']:6.4f}]")
        
        coef_diff = abs(before_summary['coefficient_mean']) - abs(after_summary['coefficient_mean'])
        d_diff = abs(before_summary['cohens_d_mean']) - abs(after_summary['cohens_d_mean'])
        
        print(f"\nAsymmetry Analysis:")
        print(f"  Coefficient difference: {coef_diff:7.4f}")
        print(f"  Cohen's d difference:   {d_diff:7.4f}")
        
        if abs(coef_diff) > 0.1:
            stronger = "BEFORE (REM → Depression)" if coef_diff > 0 else "AFTER (Depression → REM)"
            print(f"  → Stronger effect in {stronger} direction")
        else:
            print(f"  → Effects are comparable in magnitude")
    
    return {
        'before': before_summary,
        'after': after_summary
    }


if __name__ == "__main__":
    print("\nPlease provide dataset paths to run the analysis.")
    print("Example:")
    print("results = run_causal_estimation(")
    print("    before_path='path/to/before_dataset.csv',")
    print("    after_path='path/to/after_dataset.csv',")
    print("    before_treatment='rem_std',")
    print("    before_outcome='promis_dep_sum',")
    print("    before_covariates=['awake_std'],")
    print("    after_treatment='promis_dep_sum',")
    print("    after_outcome='rem_std',")
    print("    after_covariates=[],")
    print("    n_bootstrap=100")
    print(")")