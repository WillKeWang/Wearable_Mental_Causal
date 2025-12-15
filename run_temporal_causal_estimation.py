#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlined Temporal Causal Estimation
=======================================
Runs PSM + LMM causal estimation for adjustment sets from JSON file.
Minimal console output, results saved to CSV.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

# Import temporal pairing function
from temporal_causal_discovery import (
    load_and_encode_data,
    create_adjacent_survey_pairs
)


def load_adjustment_sets(json_path):
    """Load adjustment sets from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    relationships = []
    for item in data:
        if item.get('path_exists') and item.get('adj_sets_final'):
            relationships.append({
                'exposure': item['exposure'],
                'outcome': item['outcome'],
                'adj_sets': item['adj_sets_final']
            })
    return relationships


def load_temporal_data(filepath, min_days_gap=21, max_days_gap=35):
    """Load data and create temporal pairs quietly."""
    df = load_and_encode_data(
        filepath,
        include_demographics=True,
        bin_age_var=True,
        age_bin_size=10,
        race_encoding='grouped'
    )
    
    if df is None:
        raise FileNotFoundError(f"Could not load data from {filepath}")
    
    paired_df, _, _, _, _ = create_adjacent_survey_pairs(
        df,
        include_demographics=True,
        use_binned_age=True,
        min_days_gap=min_days_gap,
        max_days_gap=max_days_gap,
        use_first_pair_only=True
    )
    
    return paired_df


def propensity_score_matching(df, covariates, caliper=0.1, subject_var='pid'):
    """Fast PSM with minimal output."""
    if not covariates:
        df = df.copy()
        df['pair_id'] = range(len(df))
        return df, len(df) // 2
    
    # Prepare data
    X = df[covariates].values
    y = df['treated'].values
    
    # Fit propensity score model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X_scaled, y)
    ps_scores = ps_model.predict_proba(X_scaled)[:, 1]
    df = df.copy()
    df['ps_score'] = ps_scores
    
    # Separate treated and control
    treated_df = df[df['treated'] == 1]
    control_df = df[df['treated'] == 0]
    
    treated_ps = treated_df['ps_score'].values.reshape(-1, 1)
    control_ps = control_df['ps_score'].values.reshape(-1, 1)
    treated_pids = treated_df[subject_var].values
    control_pids = control_df[subject_var].values
    
    caliper_value = caliper * np.std(ps_scores)
    
    # Nearest neighbor matching
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='manhattan')
    nn.fit(control_ps)
    distances, indices = nn.kneighbors(treated_ps)
    distances = distances.flatten()
    indices = indices.flatten()
    
    # Create matches
    matched_pairs = []
    used_controls = set()
    match_order = np.argsort(distances)
    
    for i in match_order:
        distance = distances[i]
        control_idx = indices[i]
        
        if control_pids[control_idx] == treated_pids[i]:
            continue
        
        if distance <= caliper_value and control_idx not in used_controls:
            matched_pairs.append((treated_df.index[i], control_df.index[control_idx]))
            used_controls.add(control_idx)
    
    if not matched_pairs:
        return None, 0
    
    # Create matched dataset
    matched_indices = []
    pair_id_map = {}
    for k, (t_idx, c_idx) in enumerate(matched_pairs):
        matched_indices.extend([t_idx, c_idx])
        pair_id_map[t_idx] = k
        pair_id_map[c_idx] = k
    
    matched_df = df.loc[matched_indices].copy()
    matched_df['pair_id'] = matched_df.index.map(pair_id_map)
    
    return matched_df, len(matched_pairs)


def estimate_effect_lmm(df, outcome_var, adjustment_set=None, subject_var='pid'):
    """Estimate treatment effect using LMM."""
    cols = ['treated', outcome_var, subject_var]
    if adjustment_set:
        cols += [c for c in adjustment_set if c in df.columns]
    
    df_model = df[cols].copy().dropna()
    
    if len(df_model) < 20:
        return None
    
    # Build formula
    if adjustment_set:
        valid_covars = [c for c in adjustment_set if c in df_model.columns]
        if valid_covars:
            fe_terms = " + ".join(valid_covars)
            formula = f"{outcome_var} ~ treated + {fe_terms}"
        else:
            formula = f"{outcome_var} ~ treated"
    else:
        formula = f"{outcome_var} ~ treated"
    
    try:
        model = MixedLM.from_formula(formula, data=df_model, groups=df_model[subject_var])
        result = model.fit(reml=True, method='lbfgs', maxiter=100)
        
        coef = result.params['treated']
        se = result.bse['treated']
        p_value = result.pvalues['treated']
        ci_lower, ci_upper = result.conf_int().loc['treated']
        
        # Cohen's d
        treated_vals = df_model[df_model['treated'] == 1][outcome_var]
        control_vals = df_model[df_model['treated'] == 0][outcome_var]
        pooled_std = np.sqrt((treated_vals.var() + control_vals.var()) / 2)
        cohens_d = coef / pooled_std if pooled_std > 0 else 0
        
        return {
            'coefficient': coef,
            'se': se,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'cohens_d': cohens_d,
            'n_obs': len(df_model),
            'n_subjects': df_model[subject_var].nunique()
        }
    except Exception:
        return None


def bootstrap_estimate(df_matched, outcome_var, adjustment_set=None, 
                       n_bootstrap=100, sample_frac=0.8):
    """Bootstrap analysis on matched data."""
    if 'pair_id' not in df_matched.columns:
        return None
    
    unique_pairs = df_matched['pair_id'].unique()
    n_pairs = len(unique_pairs)
    
    if n_pairs < 10:
        return None
    
    bootstrap_results = []
    
    for _ in range(n_bootstrap):
        n_sample = int(n_pairs * sample_frac)
        sampled_pairs = np.random.choice(unique_pairs, size=n_sample, replace=True)
        df_boot = df_matched[df_matched['pair_id'].isin(sampled_pairs)].copy()
        
        if len(df_boot) < 20:
            continue
        
        result = estimate_effect_lmm(df_boot, outcome_var, adjustment_set)
        if result:
            bootstrap_results.append(result)
    
    if len(bootstrap_results) < 10:
        return None
    
    coefficients = [r['coefficient'] for r in bootstrap_results]
    p_values = [r['p_value'] for r in bootstrap_results]
    cohens_ds = [r['cohens_d'] for r in bootstrap_results]
    
    return {
        'coef_mean': np.mean(coefficients),
        'coef_median': np.median(coefficients),
        'coef_std': np.std(coefficients),
        'coef_ci_lower': np.percentile(coefficients, 2.5),
        'coef_ci_upper': np.percentile(coefficients, 97.5),
        'p_value_median': np.median(p_values),
        'p_value_mean': np.mean(p_values),
        'cohens_d_mean': np.mean(cohens_ds),
        'cohens_d_median': np.median(cohens_ds),
        'cohens_d_ci_lower': np.percentile(cohens_ds, 2.5),
        'cohens_d_ci_upper': np.percentile(cohens_ds, 97.5),
        'n_successful': len(bootstrap_results),
        'n_obs': bootstrap_results[0]['n_obs'],
        'n_subjects': bootstrap_results[0]['n_subjects']
    }


def run_single_adjustment_set(df, exposure, outcome, adj_set, n_bootstrap=100):
    """Run causal estimation for a single adjustment set."""
    # Create treatment groups (median split)
    df_analysis = df.copy()
    threshold = df_analysis[exposure].quantile(0.5)
    df_analysis['treated'] = (df_analysis[exposure] >= threshold).astype(int)
    
    # Check if covariates exist
    valid_covars = [c for c in adj_set if c in df_analysis.columns]
    missing_covars = [c for c in adj_set if c not in df_analysis.columns]
    
    if missing_covars:
        return {
            'status': 'missing_covariates',
            'missing': missing_covars
        }
    
    # Propensity score matching
    df_matched, n_pairs = propensity_score_matching(df_analysis, valid_covars)
    
    if df_matched is None or n_pairs < 10:
        return {'status': 'insufficient_matches', 'n_pairs': n_pairs}
    
    # Bootstrap estimation
    result = bootstrap_estimate(df_matched, outcome, valid_covars, n_bootstrap)
    
    if result is None:
        return {'status': 'bootstrap_failed'}
    
    result['status'] = 'success'
    result['n_pairs'] = n_pairs
    
    return result


def main():
    # Configuration
    DATA_PATH = "data/preprocessed/full_run/4w_to_0w_before_no_baseline_min10/survey_wearable_28d_before_to_0d_before_min10d_full.csv"
    JSON_PATH = "results_bidirectional_analysis.json"
    OUTPUT_PATH = "causal_estimation_results.csv"
    N_BOOTSTRAP = 100
    P_VALUE_THRESHOLD = 0.05
    MAX_ADJ_SET_SIZE = 6  # Only use adjustment sets with at most this many variables
    
    print("=" * 60)
    print("TEMPORAL CAUSAL ESTIMATION")
    print("=" * 60)
    
    # Check files
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found: {DATA_PATH}")
        return
    
    if not os.path.exists(JSON_PATH):
        print(f"ERROR: JSON file not found: {JSON_PATH}")
        return
    
    # Load adjustment sets
    print(f"\nLoading adjustment sets from: {JSON_PATH}")
    relationships = load_adjustment_sets(JSON_PATH)
    
    if not relationships:
        print("No valid relationships found in JSON file.")
        return
    
    # Filter adjustment sets by size
    total_before = sum(len(r['adj_sets']) for r in relationships)
    for rel in relationships:
        rel['adj_sets'] = [s for s in rel['adj_sets'] if len(s) <= MAX_ADJ_SET_SIZE]
    total_after = sum(len(r['adj_sets']) for r in relationships)
    
    print(f"Filtering adjustment sets to max size {MAX_ADJ_SET_SIZE}: {total_before} → {total_after}")
    
    total_sets = total_after
    print(f"Found {len(relationships)} relationship(s) with {total_sets} total adjustment sets")
    
    # Load data
    print(f"\nLoading temporal data from: {DATA_PATH}")
    df = load_temporal_data(DATA_PATH)
    print(f"Loaded {len(df)} temporal pairs from {df['pid'].nunique()} participants")
    
    # Run estimation for each relationship
    all_results = []
    
    for rel in relationships:
        exposure = rel['exposure']
        outcome = rel['outcome']
        adj_sets = rel['adj_sets']
        
        print(f"\nAnalyzing: {exposure} → {outcome}")
        print(f"Adjustment sets: {len(adj_sets)}")
        
        for idx, adj_set in enumerate(tqdm(adj_sets, desc="Processing")):
            result = run_single_adjustment_set(df, exposure, outcome, adj_set, N_BOOTSTRAP)
            
            row = {
                'exposure': exposure,
                'outcome': outcome,
                'adj_set_idx': idx + 1,
                'adj_set_size': len(adj_set),
                'adj_set': '; '.join(adj_set),
                'status': result.get('status', 'unknown')
            }
            
            if result.get('status') == 'success':
                row.update({
                    'coef_mean': result['coef_mean'],
                    'coef_median': result['coef_median'],
                    'coef_std': result['coef_std'],
                    'coef_ci_lower': result['coef_ci_lower'],
                    'coef_ci_upper': result['coef_ci_upper'],
                    'p_value_median': result['p_value_median'],
                    'p_value_mean': result['p_value_mean'],
                    'cohens_d_mean': result['cohens_d_mean'],
                    'cohens_d_median': result['cohens_d_median'],
                    'cohens_d_ci_lower': result['cohens_d_ci_lower'],
                    'cohens_d_ci_upper': result['cohens_d_ci_upper'],
                    'n_pairs': result['n_pairs'],
                    'n_obs': result['n_obs'],
                    'n_subjects': result['n_subjects'],
                    'n_bootstrap_success': result['n_successful'],
                    'significant': result['p_value_median'] < P_VALUE_THRESHOLD
                })
            elif result.get('status') == 'missing_covariates':
                row['notes'] = f"Missing: {result.get('missing', [])}"
            
            all_results.append(row)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save all results
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAll results saved to: {OUTPUT_PATH}")
    
    # Summary statistics
    successful = results_df[results_df['status'] == 'success']
    
    if len(successful) > 0:
        significant = successful[successful['significant'] == True]
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total adjustment sets tested: {len(results_df)}")
        print(f"Successful estimations: {len(successful)}")
        print(f"Significant results (p < {P_VALUE_THRESHOLD}): {len(significant)}")
        
        if len(significant) > 0:
            # Save significant results separately
            sig_output = OUTPUT_PATH.replace('.csv', '_significant.csv')
            significant_sorted = significant.sort_values('p_value_median')
            significant_sorted.to_csv(sig_output, index=False)
            print(f"Significant results saved to: {sig_output}")
            
            # Print top 10 significant results
            print(f"\nTop 10 most significant results:")
            print("-" * 60)
            
            for _, row in significant_sorted.head(10).iterrows():
                print(f"\n  Set {row['adj_set_idx']} (size={row['adj_set_size']}):")
                print(f"    Covariates: {row['adj_set'][:80]}...")
                print(f"    Coefficient: {row['coef_mean']:.4f} [{row['coef_ci_lower']:.4f}, {row['coef_ci_upper']:.4f}]")
                print(f"    Cohen's d:   {row['cohens_d_mean']:.4f} [{row['cohens_d_ci_lower']:.4f}, {row['cohens_d_ci_upper']:.4f}]")
                print(f"    p-value:     {row['p_value_median']:.4f}")
        
        # Effect size summary across all successful
        print(f"\nEffect size summary (all successful):")
        print(f"  Cohen's d range: [{successful['cohens_d_mean'].min():.4f}, {successful['cohens_d_mean'].max():.4f}]")
        print(f"  Cohen's d mean:  {successful['cohens_d_mean'].mean():.4f}")
        print(f"  Cohen's d std:   {successful['cohens_d_mean'].std():.4f}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    
    return results_df


if __name__ == "__main__":
    results = main()