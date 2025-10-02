"""
Temporal Causal Discovery using PC Algorithm
============================================
Core functions for temporal causal discovery analysis.
"""

import pandas as pd
import numpy as np
import random
import warnings
from collections import defaultdict
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr
import os

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath, dataset_name):
    """
    Load and prepare data exactly like the original notebook.
    
    Args:
        filepath: Path to dataset file
        dataset_name: Name for display purposes
        
    Returns:
        DataFrame with cleaned data, or None if file not found
    """
    print(f"\nLoading {dataset_name}: {filepath}")
    
    # Try different file extensions
    for ext in ['', '.csv']:
        try_path = filepath + ext if not filepath.endswith(('.csv', '.txt')) else filepath
        if os.path.exists(try_path):
            df = pd.read_csv(try_path)
            break
    else:
        print(f"File not found: {filepath}")
        return None
    
    print(f"Original shape: {df.shape}")
    
    # Same cleaning as original notebook
    cleaned_df = df.dropna(axis=0, how='any')
    cleaned_df = cleaned_df[cleaned_df['promis_dep_sum'] >= 4]
    cleaned_df = cleaned_df[cleaned_df['promis_anx_sum'] >= 4]
    cleaned_df = cleaned_df[cleaned_df['promis_dep_sum'] <= 20]
    cleaned_df = cleaned_df[cleaned_df['promis_anx_sum'] <= 20]
    
    print(f"After cleaning: {cleaned_df.shape}")
    print(f"Unique users: {cleaned_df['pid'].nunique()}")
    
    return cleaned_df


def prepare_variables(df, keep_pid=False):
    """
    Prepare variables and data matrix.
    
    Args:
        df: Cleaned DataFrame
        keep_pid: If True, return pid column alongside data
        
    Returns:
        If keep_pid=False: Tuple of (column_names, data_matrix)
        If keep_pid=True: Tuple of (column_names, data_matrix, pid_array)
    """
    # Same variable selection as notebook
    base_vars = ["promis_dep_sum", "promis_anx_sum"]
    metric_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    cols = base_vars + metric_cols
    cols = [c for c in cols if not c.startswith("total_")]  # Remove total_ variables
    
    print(f"Selected {len(cols)} variables")
    
    # Create data matrix
    df_clean = (df[cols + ['pid'] if keep_pid else cols]
                .apply(pd.to_numeric, errors="coerce")
                .dropna())
    
    X = df_clean[cols].to_numpy()
    
    print(f"Analysis matrix: {X.shape[0]} rows x {X.shape[1]} variables")
    
    if keep_pid:
        pids = df_clean['pid'].to_numpy()
        return cols, X, pids
    else:
        return cols, X


def get_baseline_data(df):
    """
    Extract baseline (earliest) survey for each participant.
    Assumes df has a time-related column or uses first occurrence as baseline.
    
    Args:
        df: DataFrame with cleaned data
        
    Returns:
        DataFrame with only baseline surveys (one per pid)
    """
    # Check if there's a date or time column to determine baseline
    time_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'day' in c.lower()]
    
    if time_cols:
        # Use the first time column found
        time_col = time_cols[0]
        baseline_df = df.sort_values(time_col).groupby('pid').first().reset_index()
        print(f"Baseline determined using '{time_col}' column")
    else:
        # Fall back to first occurrence in dataset
        baseline_df = df.groupby('pid').first().reset_index()
        print("Baseline determined as first occurrence per pid")
    
    print(f"Baseline surveys: {len(baseline_df)} participants")
    
    return baseline_df


def bootstrap_pc_analysis(X, feature_names, base_vars, n_bootstrap=100, sample_frac=0.6, alpha=0.05):
    """
    Run bootstrap analysis with PC algorithm (original row-based sampling).
    
    Args:
        X: Data matrix (n_samples x n_features)
        feature_names: List of variable names
        base_vars: List of outcome variables
        n_bootstrap: Number of bootstrap iterations
        sample_frac: Fraction of data to sample in each iteration
        alpha: Significance level for independence tests
        
    Returns:
        Dictionary with edge counts and key edge results
    """
    edge_counts = defaultdict(int)
    edge_types = {}  # Track whether each edge is directed or undirected
    successful_iterations = 0
    
    # Track key edges
    rem_dep_count = 0
    deep_dep_count = 0
    anxiety_dep_count = 0
    
    for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
        n_sample = int(X.shape[0] * sample_frac)
        sample_indices = random.sample(range(X.shape[0]), n_sample)
        X_bootstrap = X[sample_indices, :]
        
        try:
            # Create background knowledge
            bk = create_background_knowledge(feature_names, base_vars)
            
            # Run PC algorithm (same parameters as notebook)
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    cg = pc(
                        data=X_bootstrap,
                        alpha=alpha,
                        indep_test=fisherz,
                        stable=True,
                        uc_rule=0,
                        background_knowledge=bk,
                        node_names=feature_names,
                        verbose=False
                    )
            
            if cg is None or cg.G is None:
                continue
            
            # Extract edges (same as notebook)
            graph = cg.G
            n_nodes = len(feature_names)
            adj_matrix = graph.graph
            edges_this_iteration = []
            
            for i_node in range(n_nodes):
                for j_node in range(i_node + 1, n_nodes):
                    from_var = feature_names[i_node]
                    to_var = feature_names[j_node]
                    
                    if adj_matrix[i_node, j_node] != 0 or adj_matrix[j_node, i_node] != 0:
                        if adj_matrix[i_node, j_node] != 0 and adj_matrix[j_node, i_node] != 0:
                            # Undirected
                            edge_key = tuple(sorted([from_var, to_var]))
                            edge_types[edge_key] = 'undirected'
                        elif adj_matrix[i_node, j_node] != 0:
                            # Directed i -> j
                            edge_key = (from_var, to_var)
                            edge_types[edge_key] = 'directed'
                        elif adj_matrix[j_node, i_node] != 0:
                            # Directed j -> i
                            edge_key = (to_var, from_var)
                            edge_types[edge_key] = 'directed'
                        
                        edge_counts[edge_key] += 1
                        edges_this_iteration.append(edge_key)
            
            # Count key edges (same as notebook)
            if ('rem_std', 'promis_dep_sum') in edges_this_iteration or \
               tuple(sorted(['rem_std', 'promis_dep_sum'])) in edges_this_iteration:
                rem_dep_count += 1
            if ('deep_std', 'promis_dep_sum') in edges_this_iteration or \
               tuple(sorted(['deep_std', 'promis_dep_sum'])) in edges_this_iteration:
                deep_dep_count += 1
            if ('promis_anx_sum', 'promis_dep_sum') in edges_this_iteration or \
               tuple(sorted(['promis_anx_sum', 'promis_dep_sum'])) in edges_this_iteration:
                anxiety_dep_count += 1
                
            successful_iterations += 1
            
        except:
            continue
    
    return {
        'edge_counts': edge_counts,
        'edge_types': edge_types,
        'successful_iterations': successful_iterations,
        'rem_dep_count': rem_dep_count,
        'deep_dep_count': deep_dep_count,
        'anxiety_dep_count': anxiety_dep_count
    }


def bootstrap_pc_analysis_by_pid(df, feature_names, base_vars, n_bootstrap=100, sample_frac=0.6, alpha=0.05, use_baseline=True):
    """
    Run bootstrap analysis with PC algorithm, sampling at participant (pid) level.
    
    Args:
        df: DataFrame with cleaned data (must include 'pid' column)
        feature_names: List of variable names (excluding 'pid')
        base_vars: List of outcome variables
        n_bootstrap: Number of bootstrap iterations
        sample_frac: Fraction of participants to sample in each iteration
        alpha: Significance level for independence tests
        use_baseline: If True, use only baseline (earliest) survey per pid
        
    Returns:
        Dictionary with edge counts and key edge results
    """
    # Get baseline data if requested
    if use_baseline:
        df_to_use = get_baseline_data(df)
    else:
        df_to_use = df.copy()
    
    # Get unique pids
    unique_pids = df_to_use['pid'].unique()
    n_pids = len(unique_pids)
    print(f"Total participants for sampling: {n_pids}")
    
    edge_counts = defaultdict(int)
    edge_types = {}  # Track whether each edge is directed or undirected
    successful_iterations = 0
    
    # Track key edges
    rem_dep_count = 0
    deep_dep_count = 0
    anxiety_dep_count = 0
    
    for i in tqdm(range(n_bootstrap), desc="Bootstrap (PID-level)"):
        # Sample participants
        n_sample_pids = int(n_pids * sample_frac)
        sampled_pids = random.sample(list(unique_pids), n_sample_pids)
        
        # Get data for sampled participants
        df_bootstrap = df_to_use[df_to_use['pid'].isin(sampled_pids)]
        
        # Prepare data matrix
        X_bootstrap = df_bootstrap[feature_names].to_numpy()
        
        if X_bootstrap.shape[0] < 10:  # Skip if too few samples
            continue
        
        try:
            # Create background knowledge
            bk = create_background_knowledge(feature_names, base_vars)
            
            # Run PC algorithm
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    cg = pc(
                        data=X_bootstrap,
                        alpha=alpha,
                        indep_test=fisherz,
                        stable=True,
                        uc_rule=0,
                        background_knowledge=bk,
                        node_names=feature_names,
                        verbose=False
                    )
            
            if cg is None or cg.G is None:
                continue
            
            # Extract edges
            graph = cg.G
            n_nodes = len(feature_names)
            adj_matrix = graph.graph
            edges_this_iteration = []
            
            for i_node in range(n_nodes):
                for j_node in range(i_node + 1, n_nodes):
                    from_var = feature_names[i_node]
                    to_var = feature_names[j_node]
                    
                    if adj_matrix[i_node, j_node] != 0 or adj_matrix[j_node, i_node] != 0:
                        if adj_matrix[i_node, j_node] != 0 and adj_matrix[j_node, i_node] != 0:
                            # Undirected
                            edge_key = tuple(sorted([from_var, to_var]))
                            edge_types[edge_key] = 'undirected'
                        elif adj_matrix[i_node, j_node] != 0:
                            # Directed i -> j
                            edge_key = (from_var, to_var)
                            edge_types[edge_key] = 'directed'
                        elif adj_matrix[j_node, i_node] != 0:
                            # Directed j -> i
                            edge_key = (to_var, from_var)
                            edge_types[edge_key] = 'directed'
                        
                        edge_counts[edge_key] += 1
                        edges_this_iteration.append(edge_key)
            
            # Count key edges
            if ('rem_std', 'promis_dep_sum') in edges_this_iteration or \
               tuple(sorted(['rem_std', 'promis_dep_sum'])) in edges_this_iteration:
                rem_dep_count += 1
            if ('deep_std', 'promis_dep_sum') in edges_this_iteration or \
               tuple(sorted(['deep_std', 'promis_dep_sum'])) in edges_this_iteration:
                deep_dep_count += 1
            if ('promis_anx_sum', 'promis_dep_sum') in edges_this_iteration or \
               tuple(sorted(['promis_anx_sum', 'promis_dep_sum'])) in edges_this_iteration:
                anxiety_dep_count += 1
                
            successful_iterations += 1
            
        except:
            continue
    
    return {
        'edge_counts': edge_counts,
        'edge_types': edge_types,
        'successful_iterations': successful_iterations,
        'rem_dep_count': rem_dep_count,
        'deep_dep_count': deep_dep_count,
        'anxiety_dep_count': anxiety_dep_count
    }



def create_background_knowledge(feature_names, base_vars, dataset_type=None):
    """
    Create background knowledge constraints for PC algorithm.
    
    Args:
        feature_names: List of all variable names
        base_vars: List of outcome variables (depression, anxiety)
        dataset_type: 'before' or 'after' to enforce temporal direction, None for no temporal constraint
        
    Returns:
        BackgroundKnowledge object with constraints
    """
    bk = BackgroundKnowledge()
    nodes = [GraphNode(name) for name in feature_names]
    name_to_node = {n.get_name(): n for n in nodes}
    
    # Forbid mean -> std connections for same sensor
    sensor_features = [name for name in feature_names if name not in base_vars]
    processed_bases = set()
    
    for feature in sensor_features:
        if feature.endswith('_mean'):
            base_name = feature.replace('_mean', '')
            std_feature = f"{base_name}_std"
            
            if std_feature in sensor_features and base_name not in processed_bases:
                bk.add_forbidden_by_node(name_to_node[feature], name_to_node[std_feature])
                bk.add_forbidden_by_node(name_to_node[std_feature], name_to_node[feature])
                processed_bases.add(base_name)
    
    # Require temporal directions based on dataset type
    outcome_vars = [name for name in feature_names if name in base_vars]
    
    if dataset_type == 'before':
        # BEFORE: wearable comes before survey, so wearable -> survey
        for sensor_feature in sensor_features:
            for outcome_var in outcome_vars:
                bk.add_required_by_node(name_to_node[sensor_feature], name_to_node[outcome_var])
    elif dataset_type == 'after':
        # AFTER: survey comes before wearable, so survey -> wearable
        for sensor_feature in sensor_features:
            for outcome_var in outcome_vars:
                bk.add_required_by_node(name_to_node[outcome_var], name_to_node[sensor_feature])
    else:
        # No temporal constraint specified - default to sensor -> outcome
        for sensor_feature in sensor_features:
            for outcome_var in outcome_vars:
                bk.add_required_by_node(name_to_node[sensor_feature], name_to_node[outcome_var])
    
    return bk


def analyze_single_dataset(filepath, dataset_name, n_bootstrap=100, sample_frac=0.6, alpha=0.05, 
                          use_pid_bootstrap=True, use_baseline=True):
    """
    Analyze a single dataset with PC algorithm.
    
    Args:
        filepath: Path to dataset file
        dataset_name: Name for display purposes
        n_bootstrap: Number of bootstrap iterations
        sample_frac: Fraction of data/participants to sample
        alpha: Significance level
        use_pid_bootstrap: If True, use pid-level bootstrapping; if False, use row-level
        use_baseline: If True (and use_pid_bootstrap=True), use only baseline surveys
        
    Returns:
        Dictionary with analysis results, or None if data loading fails
    """
    # Load and prepare data
    df = load_and_prepare_data(filepath, dataset_name)
    if df is None:
        return None
    
    base_vars = ["promis_dep_sum", "promis_anx_sum"]
    
    if use_pid_bootstrap:
        # Prepare feature names (excluding pid which we'll handle separately)
        metric_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
        feature_names = base_vars + metric_cols
        feature_names = [c for c in feature_names if not c.startswith("total_")]
        
        # Run PID-level bootstrap analysis
        results = bootstrap_pc_analysis_by_pid(df, feature_names, base_vars, 
                                              n_bootstrap, sample_frac, alpha, use_baseline)
    else:
        # Prepare variables (original method)
        feature_names, X = prepare_variables(df, keep_pid=False)
        
        # Run row-level bootstrap analysis
        results = bootstrap_pc_analysis(X, feature_names, base_vars, n_bootstrap, sample_frac, alpha)
    
    return results


def print_results(results, dataset_name, min_frequency=0.1):
    """
    Print results for a single dataset.
    
    Args:
        results: Dictionary with analysis results
        dataset_name: Name of dataset
        min_frequency: Minimum frequency threshold to display edge (default 0.1 = 10%)
    """
    if results:
        total = results['successful_iterations']
        rem_dep = results['rem_dep_count']
        deep_dep = results['deep_dep_count']
        anx_dep = results['anxiety_dep_count']
        
        print(f"\n{dataset_name}:")
        print(f"  rem_std -> promis_dep_sum: {rem_dep}/{total} ({rem_dep/total*100:.1f}%)")
        print(f"  deep_std -> promis_dep_sum: {deep_dep}/{total} ({deep_dep/total*100:.1f}%)")
        print(f"  promis_anx_sum <-> promis_dep_sum: {anx_dep}/{total} ({anx_dep/total*100:.1f}%)")
        
        # Determine dataset type for temporal filtering
        dataset_type = None
        name_lower = dataset_name.lower()
        if 'before' in name_lower:
            dataset_type = 'before'
        elif 'after' in name_lower:
            dataset_type = 'after'
        
        # Print all significant edges with temporal filtering
        print_all_edges(results, min_frequency=min_frequency, dataset_type=dataset_type)


def print_all_edges(results, min_frequency=0.1, dataset_type=None):
    """
    Print all edges that appeared across bootstrap iterations.
    Filters edges based on temporal direction if dataset_type is specified.
    
    Args:
        results: Dictionary with analysis results
        min_frequency: Minimum frequency threshold to display (default 0.1 = 10%)
        dataset_type: 'before' or 'after' to filter edges by temporal direction, None for no filtering
    """
    if not results or 'edge_counts' not in results:
        return
    
    total = results['successful_iterations']
    edge_counts = results['edge_counts']
    edge_types = results.get('edge_types', {})
    outcome_vars = ['promis_dep_sum', 'promis_anx_sum']
    
    # Filter edges by minimum frequency
    significant_edges = {edge: count for edge, count in edge_counts.items() 
                        if count / total >= min_frequency}
    
    if not significant_edges:
        print(f"\nNo edges appeared in >={min_frequency*100:.0f}% of iterations")
        return
    
    # Helper function to check if variable is a wearable metric
    def is_wearable_var(var):
        return var not in outcome_vars and (var.endswith('_mean') or var.endswith('_std'))
    
    # Helper function to check if edge should be printed based on temporal direction
    def should_print_edge(edge, edge_type, dataset_type):
        if dataset_type is None:
            return True
        
        # For undirected edges, always print
        if edge_type == 'undirected':
            return True
        
        # For directed edges
        if edge_type == 'directed' and isinstance(edge, tuple) and len(edge) == 2:
            from_var, to_var = edge
            
            # Check if this is a wearable-survey edge
            from_is_wearable = is_wearable_var(from_var)
            to_is_wearable = is_wearable_var(to_var)
            from_is_survey = from_var in outcome_vars
            to_is_survey = to_var in outcome_vars
            
            # If not a wearable-survey edge, always print
            if not ((from_is_wearable and to_is_survey) or (from_is_survey and to_is_wearable)):
                return True
            
            # Apply temporal filtering for wearable-survey edges
            if dataset_type == 'before':
                # BEFORE: only print wearable -> survey (edges going INTO survey)
                return from_is_wearable and to_is_survey
            elif dataset_type == 'after':
                # AFTER: only print survey -> wearable (edges going INTO wearable)
                return from_is_survey and to_is_wearable
        
        return True
    
    # Sort edges by frequency (descending) and filter by temporal direction
    filtered_edges = [(edge, count) for edge, count in significant_edges.items() 
                     if should_print_edge(edge, edge_types.get(edge, 'directed'), dataset_type)]
    sorted_edges = sorted(filtered_edges, key=lambda x: x[1], reverse=True)
    
    if not sorted_edges:
        direction_msg = ""
        if dataset_type == 'before':
            direction_msg = " (after filtering for wearable -> survey direction)"
        elif dataset_type == 'after':
            direction_msg = " (after filtering for survey -> wearable direction)"
        print(f"\nNo edges to display{direction_msg}")
        return
    
    # Separate outcome-directed edges from others
    outcome_edges = []
    other_edges = []
    
    for edge, count in sorted_edges:
        freq = count / total
        edge_type = edge_types.get(edge, 'directed')
        
        if isinstance(edge, tuple) and len(edge) == 2:
            if edge_type == 'undirected':
                # Undirected edge
                edge_str = f"{edge[0]} -- {edge[1]}"
                is_outcome = any(var in outcome_vars for var in edge)
            else:
                # Directed edge
                from_var, to_var = edge
                edge_str = f"{from_var} -> {to_var}"
                is_outcome = to_var in outcome_vars
        else:
            # Shouldn't happen, but handle gracefully
            edge_str = str(edge)
            is_outcome = False
        
        edge_info = (edge_str, count, freq, is_outcome)
        
        if is_outcome:
            outcome_edges.append(edge_info)
        else:
            other_edges.append(edge_info)
    
    # Print header with temporal context
    direction_note = ""
    if dataset_type == 'before':
        direction_note = " [Showing: Wearable -> Survey edges only]"
    elif dataset_type == 'after':
        direction_note = " [Showing: Survey -> Wearable edges only]"
    
    # Print edges to outcome variables
    if outcome_edges:
        print(f"\n  {'='*70}")
        print(f"  EDGES TO DEPRESSION/ANXIETY (>={min_frequency*100:.0f}%){direction_note}")
        print(f"  {'='*70}")
        for edge_str, count, freq, _ in outcome_edges:
            print(f"  * {edge_str}: {count}/{total} ({freq*100:.1f}%)")
    
    # Print other edges
    if other_edges:
        print(f"\n  {'='*70}")
        print(f"  OTHER EDGES (>={min_frequency*100:.0f}%){direction_note}")
        print(f"  {'='*70}")
        for edge_str, count, freq, _ in other_edges:
            print(f"    {edge_str}: {count}/{total} ({freq*100:.1f}%)")
    
    # Summary statistics
    print(f"\n  Summary:")
    print(f"    Total unique edges found: {len(edge_counts)}")
    print(f"    Edges >={min_frequency*100:.0f}% frequency: {len(significant_edges)}")
    if dataset_type:
        print(f"    Edges displayed after temporal filtering: {len(sorted_edges)}")
    print(f"    Edges to outcomes: {len(outcome_edges)}")
    print(f"    Other edges: {len(other_edges)}")


def compare_temporal_results(results_dict, min_frequency=0.1):
    """
    Compare results across multiple datasets.
    
    Args:
        results_dict: Dictionary mapping dataset names to results
        min_frequency: Minimum frequency threshold to display edges
    """
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    
    # Print individual results
    for name, result in results_dict.items():
        print_results(result, name, min_frequency=min_frequency)
    
    # Consistency check
    valid_results = {k: v for k, v in results_dict.items() if v}
    if len(valid_results) >= 2:
        print(f"\n{'='*60}")
        print("CONSISTENCY ASSESSMENT ACROSS TIME WINDOWS")
        print(f"{'='*60}")
        
        rem_stabilities = [v['rem_dep_count']/v['successful_iterations'] for v in valid_results.values()]
        rem_consistency = max(rem_stabilities) - min(rem_stabilities) < 0.2
        
        print(f"  rem_std -> depression: {'CONSISTENT' if rem_consistency else 'VARIABLE'} across time windows")
        
        for i, (name, stability) in enumerate(zip(valid_results.keys(), rem_stabilities)):
            assessment = "STRONG" if stability >= 0.7 else "MODERATE" if stability >= 0.5 else "WEAK"
            print(f"    {name}: {stability:.1%} ({assessment})")


def run_temporal_pc_analysis(dataset_paths, n_bootstrap=100, sample_frac=0.6, alpha=0.05,
                            use_pid_bootstrap=True, use_baseline=True, min_frequency=0.1):
    """
    Run temporal causal discovery analysis using PC algorithm on multiple datasets.
    
    Args:
        dataset_paths: Dictionary mapping dataset names to file paths
        n_bootstrap: Number of bootstrap iterations
        sample_frac: Fraction of data/participants to sample in each iteration
        alpha: Significance level for independence tests
        use_pid_bootstrap: If True, use pid-level bootstrapping; if False, use row-level
        use_baseline: If True (and use_pid_bootstrap=True), use only baseline surveys
        min_frequency: Minimum frequency threshold to display edges (default 0.1 = 10%)
        
    Returns:
        Dictionary mapping dataset names to analysis results
    """
    print("="*60)
    print("TEMPORAL CAUSAL DISCOVERY ANALYSIS (PC ALGORITHM)")
    if use_pid_bootstrap:
        print(f"Bootstrapping: PID-level (baseline only: {use_baseline})")
    else:
        print("Bootstrapping: Row-level")
    print("="*60)
    
    results = {}
    
    # Analyze each dataset
    for name, path in dataset_paths.items():
        results[name] = analyze_single_dataset(path, name, n_bootstrap, sample_frac, alpha,
                                              use_pid_bootstrap, use_baseline)
    
    # Compare results
    compare_temporal_results(results, min_frequency=min_frequency)
    
    return results


# Example usage
if __name__ == "__main__":
    # Example dataset paths
    dataset_paths = {
        "1w_to_5w_after": "data/preprocessed/full_run/1w_to_5w_after/survey_wearable_7d_after_to_35d_after_baseline_adj_full.csv",
        "6w_to_2w_before": "data/preprocessed/full_run/6w_to_2w_before/survey_wearable_42d_before_to_14d_before_baseline_adj_full.csv"
    }
    
    # Run analysis with PID-level bootstrapping on baseline surveys
    # min_frequency=0.1 means only show edges that appear in >=10% of iterations
    results = run_temporal_pc_analysis(
        dataset_paths,
        n_bootstrap=100,
        sample_frac=0.5,
        alpha=0.05,
        use_pid_bootstrap=True,
        use_baseline=True,
        min_frequency=0.1  # Show edges appearing in >=10% of bootstrap samples
    )