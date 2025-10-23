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
import json

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath, dataset_name):
    """
    Load and prepare data.
    
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
    
    # Cleaning for valid range of PROMIS scores
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
    
    # Create data matrix - FIXED: Don't convert pid to numeric
    if keep_pid:
        # Convert only the feature columns to numeric
        df_features = df[cols].apply(pd.to_numeric, errors="coerce")
        # Keep pid as-is
        df_clean = pd.concat([df_features, df[['pid']]], axis=1).dropna()
        pids = df_clean['pid'].to_numpy()
    else:
        df_clean = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    
    X = df_clean[cols].to_numpy()
    
    print(f"Analysis matrix: {X.shape[0]} rows x {X.shape[1]} variables")
    
    if keep_pid:
        return cols, X, pids
    else:
        return cols, X


def get_first_survey_per_participant(df):
    """
    Extract first (earliest) survey for each participant.
    Assumes df has a time-related column or uses first occurrence.
    
    Args:
        df: DataFrame with cleaned data
        
    Returns:
        DataFrame with only first surveys (one per pid)
    """
    # Check if there's a date or time column to determine first survey
    time_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'day' in c.lower()]
    
    if time_cols:
        # Use the first time column found
        time_col = time_cols[0]
        first_survey_df = df.sort_values(time_col).groupby('pid').first().reset_index()
        print(f"First survey determined using '{time_col}' column")
    else:
        # Fall back to first occurrence in dataset
        first_survey_df = df.groupby('pid').first().reset_index()
        print("First survey determined as first occurrence per pid")
    
    print(f"First surveys: {len(first_survey_df)} participants")
    
    return first_survey_df


def infer_dataset_type(name_or_path):
    """
    Infer dataset type ('before' or 'after') from dataset name or path.
    
    Args:
        name_or_path: Dataset name or file path
        
    Returns:
        'before', 'after', or None if cannot determine
    """
    name_lower = str(name_or_path).lower()
    if 'before' in name_lower:
        return 'before'
    elif 'after' in name_lower:
        return 'after'
    return None


def bootstrap_pc_analysis_by_pid(df, feature_names, base_vars, n_bootstrap=100, sample_frac=0.6, 
                                 alpha=0.05, use_first_survey=True, dataset_type=None, verbose=False):
    """
    Run bootstrap analysis with PC algorithm, sampling at participant (pid) level.
    
    Args:
        df: DataFrame with cleaned data (must include 'pid' column)
        feature_names: List of variable names (excluding 'pid')
        base_vars: List of outcome variables
        n_bootstrap: Number of bootstrap iterations
        sample_frac: Fraction of participants to sample in each iteration
        alpha: Significance level for independence tests
        use_first_survey: If True, use only first (earliest) survey per pid
        dataset_type: 'before' or 'after' to enforce temporal direction
        verbose: If True, print warnings about edge direction corrections
        
    Returns:
        Dictionary with edge counts and key edge results
    """
    # Get first survey data if requested
    if use_first_survey:
        df_to_use = get_first_survey_per_participant(df)
    else:
        df_to_use = df.copy()
    
    # Get unique pids
    unique_pids = df_to_use['pid'].unique()
    n_pids = len(unique_pids)
    print(f"Total participants for sampling: {n_pids}")
    print(f"Dataset type: {dataset_type}")
    
    edge_counts = defaultdict(int)
    edge_types = {}  # Track whether each edge is directed or undirected
    successful_iterations = 0
    
    # Track key edges - direction depends on dataset type
    rem_dep_count = 0
    deep_dep_count = 0
    anxiety_dep_count = 0
    direction_corrections = 0  # Track how many edges we corrected
    
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
            # Create background knowledge with temporal direction
            bk = create_background_knowledge(feature_names, base_vars, dataset_type=dataset_type)
            
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
            
            # Helper function to check if variable is wearable
            def is_wearable(var):
                return var not in base_vars
            
            for i_node in range(n_nodes):
                for j_node in range(i_node + 1, n_nodes):
                    var_i = feature_names[i_node]
                    var_j = feature_names[j_node]
                    
                    # Check if there's any edge between these nodes
                    if adj_matrix[i_node, j_node] != 0 or adj_matrix[j_node, i_node] != 0:
                        # Determine edge type based on adjacency matrix values
                        # In causal-learn PC algorithm:
                        # -1 at position indicates tail (no arrowhead)
                        # 1 at position indicates arrowhead
                        # i -> j: graph[i][j] = -1 (tail at i), graph[j][i] = 1 (head at j)
                        # i <- j: graph[i][j] = 1 (head at i), graph[j][i] = -1 (tail at j)
                        # i -- j: graph[i][j] = -1 (tail at i), graph[j][i] = -1 (tail at j)
                        # i <-> j: graph[i][j] = 1 (head at i), graph[j][i] = 1 (head at j)
                        
                        i_to_j = adj_matrix[i_node, j_node]
                        j_to_i = adj_matrix[j_node, i_node]
                        
                        # Check if this is a survey-wearable edge
                        is_survey_wearable_edge = (is_wearable(var_i) != is_wearable(var_j))
                        
                        # Determine edge structure from matrix values
                        if i_to_j == -1 and j_to_i == 1:
                            # Directed: i -> j
                            edge_key = (var_i, var_j)
                            edge_types[edge_key] = 'directed'
                        elif i_to_j == 1 and j_to_i == -1:
                            # Directed: j -> i
                            edge_key = (var_j, var_i)
                            edge_types[edge_key] = 'directed'
                        elif i_to_j == -1 and j_to_i == -1:
                            # Undirected: i -- j
                            if is_survey_wearable_edge and dataset_type:
                                # For survey-wearable edges, force temporal direction
                                survey_var = var_i if not is_wearable(var_i) else var_j
                                wearable_var = var_j if is_wearable(var_j) else var_i
                                
                                if dataset_type == 'after':
                                    edge_key = (survey_var, wearable_var)
                                else:  # 'before'
                                    edge_key = (wearable_var, survey_var)
                                edge_types[edge_key] = 'directed'
                                direction_corrections += 1
                                if verbose:
                                    print(f"  [Corrected] Forcing {edge_key[0]} -> {edge_key[1]} (was undirected)")
                            else:
                                # Keep as truly undirected
                                edge_key = tuple(sorted([var_i, var_j]))
                                edge_types[edge_key] = 'undirected'
                        elif i_to_j == 1 and j_to_i == 1:
                            # Bidirected: i <-> j
                            if is_survey_wearable_edge and dataset_type:
                                # For survey-wearable edges, force temporal direction
                                survey_var = var_i if not is_wearable(var_i) else var_j
                                wearable_var = var_j if is_wearable(var_j) else var_i
                                
                                if dataset_type == 'after':
                                    edge_key = (survey_var, wearable_var)
                                else:  # 'before'
                                    edge_key = (wearable_var, survey_var)
                                edge_types[edge_key] = 'directed'
                                direction_corrections += 1
                                if verbose:
                                    print(f"  [Corrected] Forcing {edge_key[0]} -> {edge_key[1]} (was bidirected)")
                            else:
                                # Keep as bidirected
                                edge_key = tuple(sorted([var_i, var_j]))
                                edge_types[edge_key] = 'bidirected'
                        else:
                            # Shouldn't happen with standard PC algorithm, but handle gracefully
                            continue
                        
                        edge_counts[edge_key] += 1
                        edges_this_iteration.append(edge_key)
            
            # Count key edges - FIXED: Direction based on dataset type
            # Check for edges in the stored format (now enforced by temporal constraints)
            if dataset_type == 'after':
                # AFTER: survey -> wearable
                if ('promis_dep_sum', 'rem_std') in edges_this_iteration:
                    rem_dep_count += 1
                if ('promis_dep_sum', 'deep_std') in edges_this_iteration:
                    deep_dep_count += 1
            else:
                # BEFORE (or default): wearable -> survey
                if ('rem_std', 'promis_dep_sum') in edges_this_iteration:
                    rem_dep_count += 1
                if ('deep_std', 'promis_dep_sum') in edges_this_iteration:
                    deep_dep_count += 1
            
            # Anxiety-depression can be bidirectional or undirected
            if ('promis_anx_sum', 'promis_dep_sum') in edges_this_iteration or \
               ('promis_dep_sum', 'promis_anx_sum') in edges_this_iteration or \
               tuple(sorted(['promis_anx_sum', 'promis_dep_sum'])) in edges_this_iteration:
                anxiety_dep_count += 1
                
            successful_iterations += 1
            
        except Exception as e:
            continue

    return {
        'edge_counts': edge_counts,
        'edge_types': edge_types,
        'successful_iterations': successful_iterations,
        'rem_dep_count': rem_dep_count,
        'deep_dep_count': deep_dep_count,
        'anxiety_dep_count': anxiety_dep_count,
        'dataset_type': dataset_type,
        'direction_corrections': direction_corrections
    }


def create_background_knowledge(feature_names, base_vars, dataset_type='before'):
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

    if dataset_type == 'after':
        # AFTER: survey comes before wearable, so survey -> wearable ONLY
        for sensor_feature in sensor_features:
            for outcome_var in outcome_vars:
                bk.add_required_by_node(name_to_node[outcome_var], name_to_node[sensor_feature])
                bk.add_forbidden_by_node(name_to_node[sensor_feature], name_to_node[outcome_var])
    elif dataset_type == 'before':
        # BEFORE: wearable comes before survey, so wearable -> survey ONLY
        for sensor_feature in sensor_features:
            for outcome_var in outcome_vars:
                bk.add_required_by_node(name_to_node[sensor_feature], name_to_node[outcome_var])
                bk.add_forbidden_by_node(name_to_node[outcome_var], name_to_node[sensor_feature])
    # else: No temporal constraint if dataset_type is None
    
    return bk


def analyze_single_dataset(filepath, dataset_name, n_bootstrap, sample_frac, alpha,
                          use_pid_bootstrap, use_first_survey):
    """
    Analyze a single dataset.
    
    Args:
        filepath: Path to dataset file
        dataset_name: Name for display and file naming
        n_bootstrap: Number of bootstrap iterations
        sample_frac: Fraction to sample
        alpha: Significance level
        use_pid_bootstrap: Whether to use PID-level bootstrapping
        use_first_survey: Whether to use only first surveys
        
    Returns:
        Analysis results dictionary
    """
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")
    
    # Load data
    df = load_and_prepare_data(filepath, dataset_name)
    if df is None:
        return None
    
    # Infer dataset type
    dataset_type = infer_dataset_type(dataset_name)
    
    # Prepare variables
    base_vars = ["promis_dep_sum", "promis_anx_sum"]
    
    if use_pid_bootstrap:
        # Need to keep pid for bootstrapping
        cols, X, pids = prepare_variables(df, keep_pid=True)
        df_with_features = df[cols + ['pid']].copy()
        
        result = bootstrap_pc_analysis_by_pid(
            df_with_features, cols, base_vars,
            n_bootstrap=n_bootstrap,
            sample_frac=sample_frac,
            alpha=alpha,
            use_first_survey=use_first_survey,
            dataset_type=dataset_type
        )
    else:
        # Row-level bootstrap not implemented in original
        raise NotImplementedError("Row-level bootstrap not supported in this version")
    
    result['dataset_name'] = dataset_name
    result['n_bootstrap'] = n_bootstrap
    
    return result


def save_edges_to_json(results, output_dir='data/edges'):
    """
    Save edge counts to JSON files named by dataset.
    
    Args:
        results: Dictionary mapping dataset names to analysis results
        output_dir: Directory for saving JSON files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name, result in results.items():
        if result is None:
            continue
        
        # Create safe filename from dataset name
        safe_name = dataset_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        output_file = os.path.join(output_dir, f"{safe_name}_edges.json")
        
        # Convert edge tuples to strings for JSON
        # IMPORTANT: edge tuples already have the actual variable names with correct temporal direction
        edge_counts_str = {}
        for edge, count in result['edge_counts'].items():
            # edge is already a tuple like ('rem_std', 'promis_dep_sum') for BEFORE
            # or ('promis_dep_sum', 'rem_std') for AFTER
            edge_key = f"{edge[0]} -> {edge[1]}"
            edge_counts_str[edge_key] = count
        
        # Prepare data to save
        output_data = {
            'dataset_name': dataset_name,
            'dataset_type': result.get('dataset_type'),
            'successful_iterations': result['successful_iterations'],
            'edges': edge_counts_str
        }
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved edges to: {output_file}")


def print_results(result, dataset_name, min_frequency=0.1, outcome_vars=None):
    """
    Print analysis results for a dataset.
    
    Args:
        result: Analysis result dictionary
        dataset_name: Name of dataset
        min_frequency: Minimum frequency threshold to display
        outcome_vars: List of outcome variables
    """
    if result is None:
        print(f"\nNo results for {dataset_name}")
        return
    
    if outcome_vars is None:
        outcome_vars = ['promis_dep_sum', 'promis_anx_sum']
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {dataset_name}")
    print(f"{'='*60}")
    
    dataset_type = result.get('dataset_type')
    edge_counts = result['edge_counts']
    edge_types = result.get('edge_types', {})
    total = result['successful_iterations']
    
    print(f"Successful bootstrap iterations: {total}")
    print(f"Total unique edges found: {len(edge_counts)}")
    
    # Filter edges by frequency
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
                is_outcome = to_var in outcome_vars or from_var in outcome_vars
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
    
    # Print edges involving outcome variables
    if outcome_edges:
        print(f"\n  {'='*70}")
        print(f"  EDGES INVOLVING DEPRESSION/ANXIETY (>={min_frequency*100:.0f}%){direction_note}")
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
    print(f"    Edges involving outcomes: {len(outcome_edges)}")
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
        
        print(f"  rem_std <-> depression: {'CONSISTENT' if rem_consistency else 'VARIABLE'} across time windows")
        
        for i, (name, stability) in enumerate(zip(valid_results.keys(), rem_stabilities)):
            assessment = "STRONG" if stability >= 0.7 else "MODERATE" if stability >= 0.5 else "WEAK"
            print(f"    {name}: {stability:.1%} ({assessment})")


def run_temporal_pc_analysis(dataset_paths, n_bootstrap=100, sample_frac=0.6, alpha=0.05,
                            use_pid_bootstrap=True, use_first_survey=True, min_frequency=0.1,
                            save_edges=True, output_dir='data/edges'):
    """
    Run temporal causal discovery analysis using PC algorithm on multiple datasets.
    
    Args:
        dataset_paths: Dictionary mapping dataset names to file paths
        n_bootstrap: Number of bootstrap iterations
        sample_frac: Fraction of data/participants to sample in each iteration
        alpha: Significance level for independence tests
        use_pid_bootstrap: If True, use pid-level bootstrapping; if False, use row-level
        use_first_survey: If True (and use_pid_bootstrap=True), use only first surveys
        min_frequency: Minimum frequency threshold to display edges (default 0.1 = 10%)
        save_edges: If True, save edge results to JSON files
        output_dir: Directory to save JSON files (default 'data/edges')
        
    Returns:
        Dictionary mapping dataset names to analysis results
    """
    print("="*60)
    print("TEMPORAL CAUSAL DISCOVERY ANALYSIS (PC ALGORITHM)")
    if use_pid_bootstrap:
        print(f"Bootstrapping: PID-level (first survey only: {use_first_survey})")
    else:
        print("Bootstrapping: Row-level")
    print("="*60)
    
    results = {}
    
    # Analyze each dataset
    for name, path in dataset_paths.items():
        results[name] = analyze_single_dataset(path, name, n_bootstrap, sample_frac, alpha,
                                              use_pid_bootstrap, use_first_survey)
    
    # Save edges to JSON files
    if save_edges:
        print(f"\n{'='*60}")
        print("SAVING EDGE RESULTS")
        print(f"{'='*60}")
        
        save_edges_to_json(results, output_dir=output_dir)
    
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
    
    # Run analysis with PID-level bootstrapping on first surveys
    # min_frequency=0.1 means only show edges that appear in >=10% of iterations
    # save_edges=True will save results to JSON files
    results = run_temporal_pc_analysis(
        dataset_paths,
        n_bootstrap=100,
        sample_frac=0.5,
        alpha=0.05,
        use_pid_bootstrap=True,
        use_first_survey=True,
        min_frequency=0.1,
        save_edges=True,
        output_dir='data/edges'
    )