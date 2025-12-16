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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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
    
    try:
        # Prepare data
        X = df[covariates].values
        y = df['treated'].values
        
        # Fit propensity score model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        ps_model = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
        ps_model.fit(X_scaled, y)
        ps_scores = ps_model.predict_proba(X_scaled)[:, 1]
    except Exception:
        return None, 0
    
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


def estimate_effect_lmm(df, outcome_var, adjustment_set=None, subject_var='pid', timeout_sec=30):
    """Estimate treatment effect using LMM with timeout."""
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
        # Use lbfgs (faster) with maxiter cap to prevent infinite loops
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
    N_BOOTSTRAP = 50  # Reduced for speed
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
            # Use timeout to skip slow adjustment sets
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_single_adjustment_set, df, exposure, outcome, adj_set, N_BOOTSTRAP)
                    result = future.result(timeout=120)  # 2 minute timeout per adjustment set
            except FuturesTimeoutError:
                result = {'status': 'timeout'}
            except Exception as e:
                result = {'status': 'error', 'error': str(e)}
            
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
            
            # Effect size summary across SIGNIFICANT results only
            print(f"\nEffect size summary (significant only, n={len(significant)}):")
            print(f"  Cohen's d range: [{significant['cohens_d_mean'].min():.4f}, {significant['cohens_d_mean'].max():.4f}]")
            print(f"  Cohen's d mean:  {significant['cohens_d_mean'].mean():.4f}")
            print(f"  Cohen's d std:   {significant['cohens_d_mean'].std():.4f}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    
    return results_df


def run_subgroup_analysis(data_path, json_path, output_prefix="subgroup", 
                          n_bootstrap=50, p_threshold=0.05, max_adj_size=6,
                          use_top_adj_sets_only=True):
    """
    Run causal estimation stratified by demographic subgroups.
    
    For each stratification variable (sex, age, race), uses only adjustment sets
    that do NOT contain that variable.
    
    Uses sensible demographic groupings:
    - Sex: Male vs Female
    - Age: Young (<40), Middle (40-59), Older (60+)
    - Race: White vs Non-White (due to sample size imbalance)
    """
    print("=" * 60)
    print("SUBGROUP ANALYSIS")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = load_temporal_data(data_path)
    print(f"Loaded {len(df)} temporal pairs from {df['pid'].nunique()} participants")
    
    # Load adjustment sets
    print(f"Loading adjustment sets from: {json_path}")
    relationships = load_adjustment_sets(json_path)
    
    if not relationships:
        print("No valid relationships found.")
        return None
    
    # Define top adjustment sets (based on main analysis results)
    # These are interpretable, effective, and avoid concurrent colliders
    TOP_ADJ_SETS = [
        # Pure lagged sleep architecture
        ['awake_mean_tm1', 'awake_std_tm1', 'rem_mean_tm1', 'rem_std_tm1'],
        # Lagged sleep means
        ['awake_mean_tm1', 'deep_mean_tm1', 'light_mean_tm1', 'rem_mean_tm1'],
        # Sleep + breathing variability
        ['awake_mean_tm1', 'awake_std_tm1', 'breath_v_average_std_tm1', 'rem_mean_tm1'],
        # Sleep + temperature
        ['awake_mean_tm1', 'awake_std_tm1', 'breath_v_average_std_tm1', 'rem_mean_tm1', 'temperature_trend_deviation_std_tm1'],
        # Comprehensive lagged physiology
        ['awake_mean_tm1', 'awake_std_tm1', 'deep_std_tm1', 'light_mean_tm1', 'rem_mean_tm1'],
    ]
    
    # Create sensible demographic groupings
    print("\nCreating demographic subgroups...")
    
    # Age grouping: Young (<40), Middle (40-59), Older (60+)
    if 'age_binned' in df.columns:
        df['age_group'] = pd.cut(
            df['age_binned'], 
            bins=[-1, 35, 55, 100],
            labels=['Young_<40', 'Middle_40-59', 'Older_60+']
        )
        age_counts = df['age_group'].value_counts()
        print(f"  Age groups: {dict(age_counts)}")
    
    # Race grouping: White vs Non-White (due to severe imbalance)
    if 'race_encoded' in df.columns:
        df['race_group'] = df['race_encoded'].apply(
            lambda x: 'White' if x == 0 else 'Non-White'
        )
        race_counts = df['race_group'].value_counts()
        print(f"  Race groups: {dict(race_counts)}")
    
    # Define subgroup configurations
    subgroup_configs = []
    
    # Sex subgroups
    if 'sex_encoded' in df.columns:
        subgroup_configs.append({
            'name': 'sex',
            'column': 'sex_encoded',
            'exclude_vars': ['sex_encoded'],
            'labels': {0: 'Female', 1: 'Male'}
        })
    
    # Age subgroups (using new grouping)
    if 'age_group' in df.columns:
        unique_ages = df['age_group'].dropna().unique()
        subgroup_configs.append({
            'name': 'age',
            'column': 'age_group',
            'exclude_vars': ['age_binned', 'age_group'],
            'labels': {v: str(v) for v in unique_ages}
        })
    
    # Race subgroups (using new grouping)
    if 'race_group' in df.columns:
        subgroup_configs.append({
            'name': 'race',
            'column': 'race_group',
            'exclude_vars': ['race_encoded', 'race_group', 'ethnicity_encoded'],
            'labels': {'White': 'White', 'Non-White': 'Non-White'}
        })
    
    print(f"\nSubgroup stratifications: {[c['name'] for c in subgroup_configs]}")
    
    if use_top_adj_sets_only:
        print(f"Using top {len(TOP_ADJ_SETS)} curated adjustment sets")
    
    all_subgroup_results = []
    
    for rel in relationships:
        exposure = rel['exposure']
        outcome = rel['outcome']
        all_adj_sets = rel['adj_sets']
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {exposure} → {outcome}")
        print(f"{'='*60}")
        
        for config in subgroup_configs:
            strat_name = config['name']
            strat_col = config['column']
            exclude_vars = config['exclude_vars']
            labels = config['labels']
            
            if use_top_adj_sets_only:
                # Use curated top adjustment sets, filtering out stratification vars
                filtered_adj_sets = [
                    s for s in TOP_ADJ_SETS
                    if not any(v in s for v in exclude_vars)
                ]
            else:
                # Filter from all adjustment sets
                filtered_adj_sets = [
                    s for s in all_adj_sets 
                    if len(s) <= max_adj_size and not any(v in s for v in exclude_vars)
                ]
            
            if not filtered_adj_sets:
                print(f"\n  [{strat_name}] No valid adjustment sets after filtering. Skipping.")
                continue
            
            print(f"\n  [{strat_name.upper()}] Using {len(filtered_adj_sets)} adjustment sets (excluded: {exclude_vars})")
            
            # Get unique subgroup values
            subgroup_values = df[strat_col].dropna().unique()
            
            for subgroup_val in subgroup_values:
                subgroup_label = labels.get(subgroup_val, str(subgroup_val))
                df_subgroup = df[df[strat_col] == subgroup_val].copy()
                
                n_subgroup = len(df_subgroup)
                n_participants = df_subgroup['pid'].nunique()
                
                if n_subgroup < 100:  # Increased minimum for more reliable estimates
                    print(f"    {subgroup_label}: n={n_subgroup} (skipped - too small)")
                    continue
                
                print(f"    {subgroup_label}: n={n_subgroup}, {n_participants} participants, {len(filtered_adj_sets)} adj sets")
                
                # Run estimation for each adjustment set
                for idx, adj_set in enumerate(tqdm(filtered_adj_sets, 
                                                    desc=f"      {subgroup_label}", 
                                                    leave=False)):
                    try:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                run_single_adjustment_set, 
                                df_subgroup, exposure, outcome, adj_set, n_bootstrap
                            )
                            result = future.result(timeout=120)
                    except FuturesTimeoutError:
                        result = {'status': 'timeout'}
                    except Exception as e:
                        result = {'status': 'error', 'error': str(e)}
                    
                    row = {
                        'exposure': exposure,
                        'outcome': outcome,
                        'stratification': strat_name,
                        'subgroup': subgroup_label,
                        'subgroup_n': n_subgroup,
                        'subgroup_participants': n_participants,
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
                            'significant': result['p_value_median'] < p_threshold
                        })
                    
                    all_subgroup_results.append(row)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_subgroup_results)
    
    # Save all results
    output_all = f"{output_prefix}_all_results.csv"
    results_df.to_csv(output_all, index=False)
    print(f"\nAll subgroup results saved to: {output_all}")
    
    # Summary by subgroup
    print("\n" + "=" * 60)
    print("SUBGROUP SUMMARY")
    print("=" * 60)
    
    successful = results_df[results_df['status'] == 'success']
    
    if len(successful) > 0:
        significant = successful[successful['significant'] == True]
        
        # Save significant results
        if len(significant) > 0:
            sig_output = f"{output_prefix}_significant.csv"
            significant.to_csv(sig_output, index=False)
            print(f"Significant results saved to: {sig_output}")
        
        # Summary table by subgroup
        print(f"\n{'Stratification':<15} {'Subgroup':<20} {'N':<8} {'Tested':<8} {'Sig':<6} {'Cohen d (sig)':<20}")
        print("-" * 85)
        
        for strat in results_df['stratification'].unique():
            strat_df = results_df[results_df['stratification'] == strat]
            
            for subgroup in sorted(strat_df['subgroup'].unique()):
                sub_df = strat_df[strat_df['subgroup'] == subgroup]
                sub_success = sub_df[sub_df['status'] == 'success']
                sub_sig = sub_success[sub_success['significant'] == True]
                
                n_subgroup = sub_df['subgroup_n'].iloc[0] if len(sub_df) > 0 else 0
                n_tested = len(sub_success)
                n_sig = len(sub_sig)
                
                if n_sig > 0:
                    d_mean = sub_sig['cohens_d_mean'].mean()
                    d_std = sub_sig['cohens_d_mean'].std()
                    if pd.isna(d_std) or n_sig == 1:
                        d_str = f"{d_mean:.3f}"
                    else:
                        d_str = f"{d_mean:.3f} ± {d_std:.3f}"
                else:
                    d_str = "N/A"
                
                print(f"{strat:<15} {subgroup:<20} {n_subgroup:<8} {n_tested:<8} {n_sig:<6} {d_str:<20}")
        
        # Effect comparison across subgroups (for significant results)
        if len(significant) > 0:
            print("\n" + "-" * 60)
            print("Effect size comparison (significant results only):")
            print("-" * 60)
            
            for strat in significant['stratification'].unique():
                strat_sig = significant[significant['stratification'] == strat]
                print(f"\n  {strat.upper()}:")
                
                for subgroup in sorted(strat_sig['subgroup'].unique()):
                    sub_sig = strat_sig[strat_sig['subgroup'] == subgroup]
                    if len(sub_sig) > 0:
                        d_mean = sub_sig['cohens_d_mean'].mean()
                        d_min = sub_sig['cohens_d_mean'].min()
                        d_max = sub_sig['cohens_d_mean'].max()
                        print(f"    {subgroup}: mean={d_mean:.4f}, range=[{d_min:.4f}, {d_max:.4f}], n={len(sub_sig)}")
    
    print("\n" + "=" * 60)
    print("SUBGROUP ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results_df


if __name__ == "__main__":
    import sys
    
    # Check command line args for subgroup analysis
    if len(sys.argv) > 1 and sys.argv[1] == '--subgroup':
        DATA_PATH = "data/preprocessed/full_run/4w_to_0w_before_no_baseline_min10/survey_wearable_28d_before_to_0d_before_min10d_full.csv"
        JSON_PATH = "results_bidirectional_analysis.json"
        
        # Use --all-adj-sets flag to use all adjustment sets instead of curated top ones
        use_top_only = '--all-adj-sets' not in sys.argv
        
        results = run_subgroup_analysis(
            data_path=DATA_PATH,
            json_path=JSON_PATH,
            output_prefix="subgroup_causal",
            n_bootstrap=50,
            p_threshold=0.05,
            max_adj_size=6,
            use_top_adj_sets_only=use_top_only
        )
    else:
        results = main()