"""
Temporal Causal Estimation for Three Causal Relationships
==========================================================
This script runs causal estimation for three specified temporal causal relationships
using Propensity Score Matching (PSM) and Linear Mixed Effects Models (LMM).

The three relationships being estimated:
1. promis_dep_sum_t → rem_std_t
2. awake_std_t → promis_dep_sum_t
3. promis_dep_sum_t → awake_std_t
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
import os
warnings.filterwarnings('ignore')


def load_and_clean_temporal_data(filepath, dataset_name):
    """Load and clean temporal dataset with _t and _tm1 variables."""
    print(f"\n{'='*70}")
    print(f"Loading {dataset_name}")
    print(f"{'='*70}")
    print(f"File: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:20]}...")  # Show first 20 columns

    # Clean PROMIS scores (both _t and _tm1 versions if they exist)
    df_clean = df.copy()

    # Identify PROMIS columns to clean
    promis_cols_to_clean = []
    for col in ['promis_dep_sum_t', 'promis_dep_sum_tm1', 'promis_anx_sum_t', 'promis_anx_sum_tm1']:
        if col in df.columns:
            promis_cols_to_clean.append(col)

    # Apply PROMIS score constraints
    for col in promis_cols_to_clean:
        df_clean = df_clean[
            (df_clean[col] >= 4) & (df_clean[col] <= 20)
        ]

    # Drop rows with any missing values
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(axis=0, how='any')
    dropped_rows = initial_rows - len(df_clean)

    print(f"After cleaning: {df_clean.shape}")
    print(f"  Dropped {dropped_rows} rows due to missing values")
    print(f"Unique participants: {df_clean['pid'].nunique()}")

    return df_clean


def create_treatment_groups(df, treatment_var, percentile_threshold=50):
    """Create binary treatment groups based on median split."""
    threshold = df[treatment_var].quantile(percentile_threshold / 100)
    df['treated'] = (df[treatment_var] >= threshold).astype(int)

    print(f"\nTreatment variable: {treatment_var}")
    print(f"Threshold (median): {threshold:.4f}")
    print(f"Control group (0): {(df['treated']==0).sum()} observations")
    print(f"Treated group (1): {(df['treated']==1).sum()} observations")

    return df


def propensity_score_matching(df, treatment_var, outcome_var, covariates,
                              caliper=0.1, subject_var='pid'):
    """
    Fast propensity score matching using sklearn NearestNeighbors.
    Prevents same-pid matching for between-subject effects.
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

    # Extract PIDs for same-participant check
    treated_pids = treated_df[subject_var].values
    control_pids = control_df[subject_var].values

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

    same_pid_skipped = 0
    for i in match_order:
        distance = distances[i]
        control_idx = indices[i]

        # Prevent same-pid matching
        if control_pids[control_idx] == treated_pids[i]:
            same_pid_skipped += 1
            continue

        # Check caliper and if control already used
        if distance <= caliper_value and control_idx not in used_controls:
            treated_idx = i
            matched_pairs.append((
                treated_df.index[treated_idx],
                control_df.index[control_idx]
            ))
            used_controls.add(control_idx)

    print(f"Matched {len(matched_pairs)} pairs (skipped {same_pid_skipped} same-pid potential matches)")

    # Create matched dataset
    matched_indices = []
    for t_idx, c_idx in matched_pairs:
        matched_indices.extend([t_idx, c_idx])

    matched_df = df.loc[matched_indices].copy()

    # Assign pair_id for pair-level bootstrapping
    pair_id_map = {}
    for k, (t_idx, c_idx) in enumerate(matched_pairs):
        pair_id_map[t_idx] = k
        pair_id_map[c_idx] = k
    matched_df['pair_id'] = matched_df.index.map(pair_id_map)

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
        print(f"{status} {covar:35s}: SMD = {smd:6.3f} (T:{mean_t:7.3f}, C:{mean_c:7.3f})")


def estimate_effect_lmm(df, treatment_var, outcome_var, adjustment_set=None, subject_var='pid'):
    """
    Estimate treatment effect using Linear Mixed Effects Model.
    Includes adjustment covariates to account for residual imbalance.
    """
    # Build the modeling frame with required columns
    cols = ['treated', outcome_var, subject_var]
    if adjustment_set:
        cols += adjustment_set
    df_model = df[cols].copy().dropna()

    # Build formula: include adjustment covariates if provided
    if adjustment_set and len(adjustment_set) > 0:
        fe_terms = " + ".join(adjustment_set)
        formula = f"{outcome_var} ~ treated + {fe_terms}"
    else:
        formula = f"{outcome_var} ~ treated"

    # Fit LMM: outcome ~ treated + covariates + (1|subject)
    model = MixedLM.from_formula(
        formula,
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
                       adjustment_set=None, n_bootstrap=100, sample_frac=0.8,
                       subject_var='pid'):
    """
    Bootstrap analysis on matched data.
    Resamples at the matched-pair level to preserve matching structure.
    """
    print(f"\n{'='*70}")
    print(f"Bootstrap Analysis on Matched Data ({n_bootstrap} iterations)")
    print(f"{'='*70}")

    bootstrap_results = []

    # Resample by matched pairs
    unique_pairs = df_matched['pair_id'].unique()
    n_pairs = len(unique_pairs)

    print(f"Starting bootstrap with {n_pairs} matched pairs, {len(df_matched)} observations")
    if adjustment_set:
        print(f"Including adjustment covariates in LMM: {adjustment_set}")

    for i in tqdm(range(n_bootstrap), desc="Bootstrapping"):
        # Sample matched pairs with replacement
        n_sample_pairs = int(n_pairs * sample_frac)
        sampled_pairs = np.random.choice(unique_pairs, size=n_sample_pairs, replace=True)

        # Keep all observations belonging to the sampled pairs
        df_boot = df_matched[df_matched['pair_id'].isin(sampled_pairs)].copy()

        if len(df_boot) < 20:
            continue

        try:
            # Estimate effect using LMM
            result = estimate_effect_lmm(df_boot, treatment_var, outcome_var,
                                        adjustment_set, subject_var)
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


def run_single_causal_relationship(df, exposure, outcome, adjustment_sets,
                                   relationship_name, n_bootstrap=100):
    """
    Run causal estimation for a single exposure-outcome relationship
    with multiple adjustment sets.

    Returns results for each adjustment set.
    """
    print("\n" + "="*70)
    print(f"CAUSAL RELATIONSHIP: {relationship_name}")
    print("="*70)
    print(f"Exposure: {exposure}")
    print(f"Outcome: {outcome}")
    print(f"Number of adjustment sets: {len(adjustment_sets)}")

    results = {}

    for idx, adj_set in enumerate(adjustment_sets, 1):
        print(f"\n{'='*70}")
        print(f"Adjustment Set {idx}/{len(adjustment_sets)}: {adj_set}")
        print(f"{'='*70}")

        # Create a copy of the dataframe for this analysis
        df_analysis = df.copy()

        # Create treatment groups
        df_analysis = create_treatment_groups(df_analysis, exposure)

        # Propensity score matching
        if adj_set:
            df_matched = propensity_score_matching(
                df_analysis, exposure, outcome, adj_set, caliper=0.1
            )
        else:
            df_matched = df_analysis
            print("\nNo covariates specified - using full dataset")
            # Still need pair_id for bootstrap
            df_matched['pair_id'] = range(len(df_matched))

        # Bootstrap analysis
        summary = bootstrap_analysis(
            df_matched, exposure, outcome,
            adjustment_set=adj_set, n_bootstrap=n_bootstrap
        )

        if summary:
            print_bootstrap_summary(summary, f"{relationship_name} - Adjustment Set {idx}")
            results[f"adj_set_{idx}"] = {
                'adjustment_set': adj_set,
                'summary': summary
            }
        else:
            print(f"⚠️  Warning: No results for adjustment set {idx}")

    return results


def visualize_adjustment_set_comparison(all_results, relationship_name, output_path):
    """Create visualization comparing results across different adjustment sets."""
    n_sets = len(all_results)

    if n_sets == 0:
        print("No results to visualize")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prepare data for plotting
    labels = []
    means = []
    cis_lower = []
    cis_upper = []
    means_d = []
    cis_lower_d = []
    cis_upper_d = []

    for key in sorted(all_results.keys()):
        result = all_results[key]
        adj_set = result['adjustment_set']
        summary = result['summary']

        # Create label
        if adj_set:
            label = f"Set {key.split('_')[-1]}\n{', '.join(adj_set[:2])}"
            if len(adj_set) > 2:
                label += "\n..."
        else:
            label = f"Set {key.split('_')[-1]}\n(no adjustment)"

        labels.append(label)
        means.append(summary['coefficient_mean'])
        cis_lower.append(summary['coefficient_ci_lower'])
        cis_upper.append(summary['coefficient_ci_upper'])
        means_d.append(summary['cohens_d_mean'])
        cis_lower_d.append(summary['cohens_d_ci_lower'])
        cis_upper_d.append(summary['cohens_d_ci_upper'])

    # Plot 1: Coefficient comparison
    ax = axes[0]
    x_pos = np.arange(len(labels))
    errors_lower = [means[i] - cis_lower[i] for i in range(len(means))]
    errors_upper = [cis_upper[i] - means[i] for i in range(len(means))]

    ax.bar(x_pos, means, alpha=0.6, edgecolor='black', color='steelblue')
    ax.errorbar(x_pos, means, yerr=[errors_lower, errors_upper],
                fmt='none', color='black', capsize=5, linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Coefficient', fontsize=11)
    ax.set_title(f'{relationship_name}\nCoefficient Comparison', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Plot 2: Cohen's d comparison
    ax = axes[1]
    errors_lower_d = [means_d[i] - cis_lower_d[i] for i in range(len(means_d))]
    errors_upper_d = [cis_upper_d[i] - means_d[i] for i in range(len(means_d))]

    ax.bar(x_pos, means_d, alpha=0.6, edgecolor='black', color='coral')
    ax.errorbar(x_pos, means_d, yerr=[errors_lower_d, errors_upper_d],
                fmt='none', color='black', capsize=5, linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cohen's d", fontsize=11)
    ax.set_title(f'{relationship_name}\nEffect Size Comparison', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    return fig


def main():
    """Main execution function for all three causal relationships."""

    # ================================================================
    # CONFIGURATION
    # ================================================================

    # Dataset path - REPLACE WITH YOUR ACTUAL DATA PATH
    DATA_PATH = "data/temporal_causal_data.csv"  # Update this path!

    # Number of bootstrap iterations
    N_BOOTSTRAP = 100

    # Define the three causal relationships
    relationships = {
        'relationship_1': {
            'name': 'Depression (t) → REM Variability (t)',
            'exposure': 'promis_dep_sum_t',
            'outcome': 'rem_std_t',
            'adjustment_sets': [
                ['age_binned'],
                ['rmssd_std_tm1'],
                ['promis_anx_sum_tm1', 'promis_dep_sum_tm1']
            ]
        },
        'relationship_2': {
            'name': 'Awake Variability (t) → Depression (t)',
            'exposure': 'awake_std_t',
            'outcome': 'promis_dep_sum_t',
            'adjustment_sets': [
                ['promis_anx_sum_tm1', 'promis_dep_sum_tm1'],
                ['age_binned', 'awake_std_tm1', 'onset_latency_std_t', 'onset_latency_std_tm1']
            ]
        },
        'relationship_3': {
            'name': 'Depression (t) → Awake Variability (t)',
            'exposure': 'promis_dep_sum_t',
            'outcome': 'awake_std_t',
            'adjustment_sets': [
                ['promis_anx_sum_tm1', 'promis_dep_sum_tm1'],
                ['age_binned', 'awake_std_tm1', 'onset_latency_std_t', 'onset_latency_std_tm1']
            ]
        }
    }

    print("\n" + "="*70)
    print("TEMPORAL CAUSAL ESTIMATION - THREE RELATIONSHIPS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data file: {DATA_PATH}")
    print(f"  Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"  Number of relationships: {len(relationships)}")
    print("="*70)

    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ ERROR: Data file not found: {DATA_PATH}")
        print("\nPlease update the DATA_PATH variable to point to your temporal causal data file.")
        print("The file should contain variables with _t and _tm1 suffixes.")
        return

    # Load data
    df = load_and_clean_temporal_data(DATA_PATH, "Temporal Causal Data")

    # Store all results
    all_relationship_results = {}

    # Run analysis for each relationship
    for rel_key, rel_config in relationships.items():
        results = run_single_causal_relationship(
            df=df,
            exposure=rel_config['exposure'],
            outcome=rel_config['outcome'],
            adjustment_sets=rel_config['adjustment_sets'],
            relationship_name=rel_config['name'],
            n_bootstrap=N_BOOTSTRAP
        )

        all_relationship_results[rel_key] = {
            'config': rel_config,
            'results': results
        }

        # Create visualization for this relationship
        if results:
            output_file = f"causal_estimation_{rel_key}.png"
            visualize_adjustment_set_comparison(
                results, rel_config['name'], output_file
            )

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL THREE RELATIONSHIPS")
    print("="*70)

    for rel_key, rel_data in all_relationship_results.items():
        config = rel_data['config']
        results = rel_data['results']

        print(f"\n{config['name']}")
        print(f"  Exposure: {config['exposure']}")
        print(f"  Outcome: {config['outcome']}")
        print(f"  Number of adjustment sets tested: {len(config['adjustment_sets'])}")

        for adj_key, adj_result in results.items():
            adj_set = adj_result['adjustment_set']
            summary = adj_result['summary']

            set_num = adj_key.split('_')[-1]
            print(f"\n  Adjustment Set {set_num}: {adj_set}")
            print(f"    Coefficient: {summary['coefficient_mean']:7.4f} [{summary['coefficient_ci_lower']:6.4f}, {summary['coefficient_ci_upper']:6.4f}]")
            print(f"    Cohen's d:   {summary['cohens_d_mean']:7.4f} [{summary['cohens_d_ci_lower']:6.4f}, {summary['cohens_d_ci_upper']:6.4f}]")
            print(f"    p-value:     {summary['p_value_median']:.4f}")

    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nVisualization files created:")
    for rel_key in relationships.keys():
        print(f"  - causal_estimation_{rel_key}.png")

    return all_relationship_results


if __name__ == "__main__":
    results = main()
