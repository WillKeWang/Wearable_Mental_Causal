"""
Temporal Causal Discovery using PC Algorithm - Version 2
=========================================================
Uses adjacent survey pairs (t-1, t) from the same dataset.
Each pair represents two consecutive months of survey+sensor data.
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

from joblib import Parallel, delayed
import pickle
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')


def encode_sex(sex_value):
    """
    Encode sex as binary numeric (0=female, 1=male).
    
    Args:
        sex_value: 'female', 'male', or other string
        
    Returns:
        0 for female, 1 for male, np.nan for others
    """
    if pd.isna(sex_value):
        return np.nan
    sex_lower = str(sex_value).lower().strip()
    if sex_lower == 'female':
        return 0
    elif sex_lower == 'male':
        return 1
    else:
        return np.nan


def encode_race_grouped(race_value):
    """
    Encode race into grouped categories:
    0 = white
    1 = asian (includes asian, south_asian)
    2 = black (includes black, african)
    3 = other (includes ethnic_other, multiple, middle_eastern, native_hawaiian, native_american)
    
    Args:
        race_value: Race string from dataset
        
    Returns:
        Numeric code 0-3, or np.nan for missing
    """
    if pd.isna(race_value):
        return np.nan
    
    race_lower = str(race_value).lower().strip()
    
    if race_lower == 'white':
        return 0
    elif race_lower in ['asian', 'south_asian']:
        return 1
    elif race_lower in ['black', 'african']:
        return 2
    else:  # ethnic_other, multiple, middle_eastern, native_hawaiian, native_american
        return 3


def encode_race_detailed(race_value):
    """
    Encode race with more detailed categories:
    0 = white
    1 = asian
    2 = black
    3 = south_asian
    4 = multiple
    5 = other (ethnic_other, middle_eastern, native_hawaiian, native_american, african)
    
    Args:
        race_value: Race string from dataset
        
    Returns:
        Numeric code 0-5, or np.nan for missing
    """
    if pd.isna(race_value):
        return np.nan
    
    race_lower = str(race_value).lower().strip()
    
    race_map = {
        'white': 0,
        'asian': 1,
        'black': 2,
        'south_asian': 3,
        'multiple': 4
    }
    
    if race_lower in race_map:
        return race_map[race_lower]
    else:
        return 5  # other


def encode_ethnicity(ethnicity_value):
    """
    Encode hispanic ethnicity as binary:
    0 = not hispanic (False)
    1 = hispanic (True)
    np.nan = skipped or missing
    
    Args:
        ethnicity_value: True, False, 'skipped', or other
        
    Returns:
        0, 1, or np.nan
    """
    if pd.isna(ethnicity_value):
        return np.nan
    
    # Handle boolean values
    if isinstance(ethnicity_value, bool):
        return 1 if ethnicity_value else 0
    
    # Handle string values
    eth_lower = str(ethnicity_value).lower().strip()
    if eth_lower in ['true', '1', 'yes']:
        return 1
    elif eth_lower in ['false', '0', 'no']:
        return 0
    else:  # 'skipped' or other
        return np.nan


def bin_age(age, bin_size=10):
    """
    Bin age into categories (e.g., 20s, 30s, 40s).
    
    Args:
        age: Age value or Series
        bin_size: Size of age bins in years (default 10)
        
    Returns:
        Binned age category (e.g., 20 for ages 20-29, 30 for ages 30-39)
    """
    if pd.isna(age):
        return np.nan
    return int(age // bin_size) * bin_size


def encode_demographics(df, bin_age_var=True, age_bin_size=10, race_encoding='grouped'):
    """
    Encode all demographic variables in the dataframe.
    
    Args:
        df: DataFrame with demographic columns
        bin_age_var: If True, bin age into categories
        age_bin_size: Size of age bins in years (default 10)
        race_encoding: 'grouped' (4 categories) or 'detailed' (6 categories)
        
    Returns:
        DataFrame with encoded demographic variables added
    """
    df_encoded = df.copy()
    
    # Encode sex
    if 'sex' in df.columns:
        df_encoded['sex_encoded'] = df['sex'].apply(encode_sex)
        print(f"Sex encoded: {df_encoded['sex_encoded'].value_counts().to_dict()}")
    
    # Encode and/or bin age
    if 'age' in df.columns:
        if bin_age_var:
            df_encoded['age_binned'] = df['age'].apply(lambda x: bin_age(x, age_bin_size))
            print(f"Age binned (size={age_bin_size}): {sorted(df_encoded['age_binned'].dropna().unique())}")
        else:
            # Keep continuous age as-is (already numeric)
            df_encoded['age_continuous'] = pd.to_numeric(df['age'], errors='coerce')
    
    # Encode race
    if 'race' in df.columns:
        if race_encoding == 'grouped':
            df_encoded['race_encoded'] = df['race'].apply(encode_race_grouped)
            race_map_desc = {0: 'white', 1: 'asian', 2: 'black', 3: 'other'}
        else:  # 'detailed'
            df_encoded['race_encoded'] = df['race'].apply(encode_race_detailed)
            race_map_desc = {0: 'white', 1: 'asian', 2: 'black', 3: 'south_asian', 4: 'multiple', 5: 'other'}
        
        print(f"Race encoded ({race_encoding}): {df_encoded['race_encoded'].value_counts().sort_index().to_dict()}")
        print(f"  Mapping: {race_map_desc}")
    
    # Encode ethnicity
    if 'ethnicity_hispanic' in df.columns:
        df_encoded['ethnicity_encoded'] = df['ethnicity_hispanic'].apply(encode_ethnicity)
        print(f"Ethnicity encoded: {df_encoded['ethnicity_encoded'].value_counts().to_dict()}")
        print(f"  (0=not hispanic, 1=hispanic, NaN=skipped)")
    
    return df_encoded


def load_and_encode_data(filepath, include_demographics=True, 
                        bin_age_var=True, age_bin_size=10, race_encoding='grouped'):
    """
    Load and encode data from CSV file.
    
    Args:
        filepath: Path to dataset file
        include_demographics: If True, encode and include demographic variables
        bin_age_var: If True, bin age into categories (e.g., 20s, 30s)
        age_bin_size: Size of age bins in years (default 10)
        race_encoding: 'grouped' (4 categories) or 'detailed' (6 categories) for race
        
    Returns:
        DataFrame with cleaned and encoded data, or None if file not found
    """
    print(f"\nLoading data: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Encode demographic variables if requested
    if include_demographics:
        print("\nEncoding demographic variables:")
        df = encode_demographics(df, bin_age_var, age_bin_size, race_encoding)
    
    # Basic cleaning for valid range of PROMIS scores
    df = df[df['promis_dep_sum'] >= 4]
    df = df[df['promis_anx_sum'] >= 4]
    df = df[df['promis_dep_sum'] <= 20]
    df = df[df['promis_anx_sum'] <= 20]
    
    print(f"\nAfter PROMIS cleaning: {df.shape}")
    print(f"Unique participants: {df['pid'].nunique()}")
    
    return df


def create_adjacent_survey_pairs(df, include_demographics=True, use_binned_age=True,
                                min_days_gap=21, max_days_gap=35, use_first_pair_only=True):
    """
    Create dataset with adjacent survey pairs (tm1, t) for each participant.
    
    Args:
        df: DataFrame with encoded data, must have 'pid' and 'date' columns
        include_demographics: If True, include demographic variables
        use_binned_age: If True, use binned age instead of continuous age
        min_days_gap: Minimum days between surveys for valid pair (default 21 = 3 weeks)
        max_days_gap: Maximum days between surveys for valid pair (default 35 = 5 weeks)
        use_first_pair_only: If True, use only the first valid pair per participant
        
    Returns:
        Tuple of (paired_df, feature_names, demographic_vars, sensor_vars, survey_vars)
    """
    print("\n" + "="*70)
    print("CREATING ADJACENT SURVEY PAIRS")
    print("="*70)
    print(f"Valid gap between surveys: {min_days_gap}-{max_days_gap} days")
    print(f"Use first valid pair only: {use_first_pair_only}")
    
    # Define variable categories
    survey_vars = ['promis_dep_sum', 'promis_anx_sum']
    
    # Get sensor variables (everything that ends with _mean or _std, excluding total_)
    sensor_vars = [col for col in df.columns 
                   if (col.endswith('_mean') or col.endswith('_std')) 
                   and not col.startswith('total_')]
    
    # Determine demographic variables to use
    demographic_vars = []
    if include_demographics:
        # Age
        if use_binned_age and 'age_binned' in df.columns:
            demographic_vars.append('age_binned')
        elif 'age_continuous' in df.columns:
            demographic_vars.append('age_continuous')
        elif 'age' in df.columns:
            demographic_vars.append('age')
        
        # Sex
        if 'sex_encoded' in df.columns:
            demographic_vars.append('sex_encoded')
        elif 'sex' in df.columns:
            demographic_vars.append('sex')
        
        # Race
        if 'race_encoded' in df.columns:
            demographic_vars.append('race_encoded')
        
        # Ethnicity
        if 'ethnicity_encoded' in df.columns:
            demographic_vars.append('ethnicity_encoded')
    
    print(f"Survey variables: {survey_vars}")
    print(f"Sensor variables: {len(sensor_vars)} variables")
    print(f"Demographic variables: {demographic_vars}")
    
    # Variables to pair across time
    vars_to_pair = survey_vars + sensor_vars
    
    # Convert date to datetime if not already
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by participant and date
    df_sorted = df.sort_values(['pid', 'date']).reset_index(drop=True)
    
    paired_data = []
    skipped_participants = 0
    
    for pid, group in tqdm(df_sorted.groupby('pid'), desc="Processing participants"):
        group = group.sort_values('date').reset_index(drop=True)
        n_surveys = len(group)
        
        if n_surveys < 2:
            skipped_participants += 1
            continue  # Need at least 2 surveys to make a pair
        
        # Get demographic values (same for all surveys from this participant)
        demo_values = {}
        for demo_var in demographic_vars:
            demo_values[demo_var] = group[demo_var].iloc[0]
        
        # Find valid pairs (surveys with gap between min_days_gap and max_days_gap)
        found_valid_pair = False
        for i in range(n_surveys - 1):
            # Calculate days between surveys
            date_tm1 = group['date'].iloc[i]
            date_t = group['date'].iloc[i+1]
            days_gap = (date_t - date_tm1).days
            
            # Check if gap is within valid range
            if min_days_gap <= days_gap <= max_days_gap:
                pair_dict = {'pid': pid}
                
                # Add demographics (time-invariant)
                pair_dict.update(demo_values)
                
                # Add tm1 variables (from survey i)
                for var in vars_to_pair:
                    pair_dict[f'{var}_tm1'] = group[var].iloc[i]
                
                # Add t variables (from survey i+1)
                for var in vars_to_pair:
                    pair_dict[f'{var}_t'] = group[var].iloc[i+1]
                
                paired_data.append(pair_dict)
                found_valid_pair = True
                
                # If we only want first valid pair, break after finding it
                if use_first_pair_only:
                    break
        
        if not found_valid_pair:
            skipped_participants += 1
    
    paired_df = pd.DataFrame(paired_data)
    
    # Remove rows with any NaN values
    paired_df_clean = paired_df.dropna()
    
    print(f"\nPaired observations created: {len(paired_df)}")
    print(f"After removing NaN: {len(paired_df_clean)}")
    print(f"Unique participants: {paired_df_clean['pid'].nunique()}")
    print(f"Participants skipped (no valid pairs): {skipped_participants}")
    
    # Create feature names list (excluding pid)
    feature_names = demographic_vars.copy()
    for var in vars_to_pair:
        feature_names.append(f'{var}_tm1')
    for var in vars_to_pair:
        feature_names.append(f'{var}_t')
    
    print(f"\nTotal features: {len(feature_names)}")
    print(f"  Demographics: {len(demographic_vars)}")
    print(f"  Time-varying (tm1): {len(vars_to_pair)}")
    print(f"  Time-varying (t): {len(vars_to_pair)}")
    
    return paired_df_clean, feature_names, demographic_vars, sensor_vars, survey_vars


def create_background_knowledge(feature_names, demographic_vars, sensor_vars, survey_vars):
    """
    Create background knowledge constraints for PC algorithm.
    
    Constraints:
    1. Temporal order: Forbid *_t -> *_tm1 (future can't cause past)
    2. Demographics: Forbid anything -> demographics (demographics are root causes)
    3. Same sensor mean/std: Forbid any edges between mean and std of the same sensor,
       regardless of time (e.g., rem_mean_tm1 cannot cause rem_std_t)
    
    Args:
        feature_names: List of all variable names
        demographic_vars: List of demographic variable names
        sensor_vars: List of sensor variable base names (without _tm1 or _t suffix)
        survey_vars: List of survey variable base names (without _tm1 or _t suffix)
        
    Returns:
        BackgroundKnowledge object with constraints
    """
    bk = BackgroundKnowledge()
    nodes = [GraphNode(name) for name in feature_names]
    name_to_node = {n.get_name(): n for n in nodes}
    
    print("\n" + "="*70)
    print("CREATING BACKGROUND KNOWLEDGE CONSTRAINTS")
    print("="*70)
    
    # ========================================
    # 1. TEMPORAL ORDER CONSTRAINT
    # ========================================
    # Forbid ALL edges from ANY variable at t to ANY variable at tm1
    print("\n1. Temporal order constraints (forbid ALL t -> tm1):")
    temporal_constraints = 0
    
    # Get all variables with _t suffix
    vars_t = [name for name in feature_names if name.endswith('_t')]
    # Get all variables with _tm1 suffix  
    vars_tm1 = [name for name in feature_names if name.endswith('_tm1')]
    
    # Forbid ALL combinations of t -> tm1
    for var_t in vars_t:
        for var_tm1 in vars_tm1:
            if var_t in name_to_node and var_tm1 in name_to_node:
                bk.add_forbidden_by_node(name_to_node[var_t], name_to_node[var_tm1])
                temporal_constraints += 1
    
    print(f"   Added {temporal_constraints} temporal constraints")
    print(f"   (Forbidding {len(vars_t)} vars at t -> {len(vars_tm1)} vars at tm1)")
    
    # ========================================
    # 2. DEMOGRAPHIC CONSTRAINTS (ROOT CAUSES)
    # ========================================
    print("\n2. Demographic constraints (forbid anything -> demographics):")
    demographic_constraints = 0
    
    for demo_var in demographic_vars:
        if demo_var in name_to_node:
            for other_var in feature_names:
                if other_var != demo_var:
                    # Forbid other_var -> demo_var
                    bk.add_forbidden_by_node(name_to_node[other_var], name_to_node[demo_var])
                    demographic_constraints += 1
    
    print(f"   Added {demographic_constraints} demographic constraints")
    
    # ========================================
    # 3. SAME SENSOR MEAN/STD CONSTRAINTS
    # ========================================
    print("\n3. Same sensor mean/std constraints (forbid all edges between mean and std):")
    sensor_constraints = 0
    
    # Extract base sensor names (without _mean or _std)
    sensor_bases = set()
    for sensor_var in sensor_vars:
        if sensor_var.endswith('_mean'):
            base = sensor_var.replace('_mean', '')
            sensor_bases.add(base)
        elif sensor_var.endswith('_std'):
            base = sensor_var.replace('_std', '')
            sensor_bases.add(base)
    
    print(f"   Found {len(sensor_bases)} sensor base names: {sorted(sensor_bases)}")
    
    # For each sensor base, forbid all edges between its mean and std variants
    ### Updated: forbid edges within the same time window ###
    for base in sensor_bases:
        same_time_pairs = [
            (f'{base}_mean_tm1', f'{base}_std_tm1'),
            (f'{base}_mean_t', f'{base}_std_t')
        ]
        
        # Forbid all edges between mean and std (both directions, all time combinations)
        for mean_var, std_var in same_time_pairs:
            if mean_var in name_to_node and std_var in name_to_node:
                bk.add_forbidden_by_node(name_to_node[mean_var], name_to_node[std_var])
                bk.add_forbidden_by_node(name_to_node[std_var], name_to_node[mean_var])
                sensor_constraints += 2
    
    print(f"   Added {sensor_constraints} same-time sensor mean/std constraints")
    
    print(f"\n   TOTAL CONSTRAINTS: {temporal_constraints + demographic_constraints + sensor_constraints}")
    print("="*70)
    
    return bk


def run_single_iteration(i, paired_df, unique_pids, feature_names, n_pids, sample_frac, 
                         max_samples_per_iteration, alpha, bk, verbose):
        
    # Sample participants
    n_sample_pids = int(n_pids * sample_frac)
    sampled_pids = random.sample(list(unique_pids), n_sample_pids)
    
    # Get data for sampled participants
    df_bootstrap = paired_df[paired_df['pid'].isin(sampled_pids)]
    
    # Further subsample if max_samples_per_iteration is set
    if max_samples_per_iteration is not None and len(df_bootstrap) > max_samples_per_iteration:
        df_bootstrap = df_bootstrap.sample(n=max_samples_per_iteration, random_state=i)
    
    # Prepare data matrix
    X_bootstrap = df_bootstrap[feature_names].to_numpy()
    
    if X_bootstrap.shape[0] < 10:  # Skip if too few samples
        return None
    
    if verbose and i % 10 == 0:
        print(f"\n  Iteration {i}: Running PC on {X_bootstrap.shape[0]} samples...")
    
    try:
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
            return None
        
        # Extract edges
        graph = cg.G
        n_nodes = len(feature_names)
        adj_matrix = graph.graph
        
        iteration_edges = {}
        
        for i_node in range(n_nodes):
            for j_node in range(i_node + 1, n_nodes):
                var_i = feature_names[i_node]
                var_j = feature_names[j_node]
                
                # Check if there's any edge between these nodes
                if adj_matrix[i_node, j_node] != 0 or adj_matrix[j_node, i_node] != 0:
                    i_to_j = adj_matrix[i_node, j_node]
                    j_to_i = adj_matrix[j_node, i_node]
                    
                    # Determine edge structure from matrix values
                    # -1 = tail (no arrowhead), 1 = arrowhead
                    if i_to_j == -1 and j_to_i == 1:
                        # Directed: i -> j
                        edge_key = (var_i, var_j)
                        edge_type = 'directed'
                    elif i_to_j == 1 and j_to_i == -1:
                        # Directed: j -> i
                        edge_key = (var_j, var_i)
                        edge_type = 'directed'
                    elif i_to_j == -1 and j_to_i == -1:
                        # Undirected: i -- j
                        edge_key = tuple(sorted([var_i, var_j]))
                        edge_type = 'undirected'
                    elif i_to_j == 1 and j_to_i == 1:
                        # Bidirected: i <-> j
                        edge_key = tuple(sorted([var_i, var_j]))
                        edge_type = 'bidirected'
                    else:
                        continue
                    
                    iteration_edges[edge_key] = edge_type
        
        return iteration_edges
        
    except Exception as e:
        if verbose:
            print(f"Error in iteration {i}: {e}")
        return None


def bootstrap_pc_analysis(paired_df, feature_names, demographic_vars, sensor_vars, survey_vars,
                          n_bootstrap=100, sample_fracs=[0.5], alpha=0.05, max_samples_per_iteration=None, 
                          verbose=False, save="none", results_dir="causal_discovery_results_temp"):
    """
    Run bootstrap analysis with PC algorithm on paired survey data.
    
    Args:
        paired_df: DataFrame with paired survey data (must include 'pid' column)
        feature_names: List of all feature names (excluding 'pid')
        demographic_vars: List of demographic variable names
        sensor_vars: List of sensor variable base names
        survey_vars: List of survey variable base names
        n_bootstrap: Number of bootstrap iterations
        sample_fracs: Fraction of participants to sample in each iteration
        alpha: Significance level for independence tests
        max_samples_per_iteration: Maximum number of data samples per iteration (for speed)
        verbose: If True, print detailed information
        save: Choose result saving mode ("none", "per_experiment", "combined", "both")
        results_dir: Directory for saving output files
        
    Returns:
        Dictionary mapping each sampling fraction to its analysis results
    """

    # Create directory for saving results
    results_dir = Path(results_dir)
    if save != "none":
        results_dir.mkdir(exist_ok=True)

    # Dictionary to store all experiment results
    all_experiments = {}

    print("\n" + "="*70)
    print("RUNNING BOOTSTRAP PC ANALYSIS")
    print("="*70)
    print(f"Bootstrap iterations: {n_bootstrap}")
    print(f"Sample fraction: {sample_fracs}")
    print(f"Alpha: {alpha}")
    
    # Get unique pids
    unique_pids = paired_df['pid'].unique()
    n_pids = len(unique_pids)
    print(f"Total participants: {n_pids}")
    print(f"Variables: {len(feature_names)}")
    
    # Create background knowledge ONCE (doesn't change across iterations)
    print("\nCreating background knowledge constraints (once for all iterations)...")
    bk = create_background_knowledge(feature_names, demographic_vars, sensor_vars, survey_vars)
    
    print("\nStarting bootstrap iterations...")
    print("(Note: Each iteration may take several minutes with 64 variables)")

    # Run analysis for each sample fraction
    for sample_frac in sample_fracs:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: sample_frac = {sample_frac}")
        print(f"{'='*70}")
        
        # Parallel execution
        results = Parallel(n_jobs=-2, verbose=5)(
            delayed(run_single_iteration)(
                i, paired_df, unique_pids, feature_names, n_pids, sample_frac,
                max_samples_per_iteration, alpha, bk, verbose
            )
            for i in range(n_bootstrap)
        )

        # Aggregate results
        edge_counts = defaultdict(int)
        edge_types = {}  # Track whether each edge is directed or undirected
        successful_iterations = 0

        for result in results:
            if result is not None:
                successful_iterations += 1
                for edge_key, edge_type in result.items():
                    edge_counts[edge_key] += 1
                    edge_types[edge_key] = edge_type

        print(f"Successful iterations: {successful_iterations}/{n_bootstrap}")
        print(f"Unique edges found: {len(edge_counts)}")
                    
        # Save individual results (pickle)
        experiment_results = {
            'sample_frac': sample_frac,
            'n_bootstrap': n_bootstrap,
            'alpha': alpha,
            'edge_counts': dict(edge_counts),
            'edge_types': edge_types,
            'successful_iterations': successful_iterations,
            'demographic_vars': demographic_vars,
            'sensor_vars': sensor_vars,
            'survey_vars': survey_vars,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save per-experiment file if requested
        if save in ("per_experiment", "both"):
            fname = results_dir / f"experiment_frac_{sample_frac:.2f}.pkl"
            with open(fname, "wb") as f:
                pickle.dump(experiment_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved to: {fname}")
        
        # Add to "all_experiments" dictionary
        all_experiments[sample_frac] = experiment_results

    # Save combined results if requested
    if save in ("combined", "both"):
        all_results_filename = results_dir / f"all_experiments_{n_bootstrap}_iterations.pkl"
        with open(all_results_filename, "wb") as f:
            pickle.dump(all_experiments, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nAll results saved to: {all_results_filename}")


    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*70}")
    if save != "none":
        print(f"Results saved in: {results_dir}/")
    else:
        print("Results not saved to disk (in-memory only).")

    return all_experiments


def convert_bootstrap_results_to_df(all_experiments, min_frequency=0):
    """
    Convert the output dictionary from bootstrap_pc_analysis into a tidy DataFrame.

    Parameters
    ----------
    all_experiments : dict
        The dictionary returned by bootstrap_pc_analysis(). 
        Keys are sample fractions, and values are experiment results.
    min_frequency : float, optional
        Minimum edge frequency threshold for inclusion in the DataFrame. Default is 0.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing one row per edge with columns:
        [sample_frac, from_var, to_var, edge_type, edge_str, freq]
    """

    plot_causal_all_sep = []

    for frac, results in all_experiments.items():
        total = results['successful_iterations']
        edge_counts = results['edge_counts']
        edge_types = results.get('edge_types', {})

        for edge, count in edge_counts.items():
            freq = count / total if total > 0 else 0
            from_var, to_var, edge_type, edge_str = None, None, None, None

            if isinstance(edge, tuple) and len(edge) == 2:
                from_var, to_var = edge
                edge_type = edge_types.get(edge, 'directed')
                if edge_type == "directed":
                    edge_str = f"{from_var} -> {to_var}"
                elif edge_type == "undirected":
                    edge_str = f"{from_var} -- {to_var}"
                elif edge_type == "bidirected":
                    edge_str = f"{from_var} <-> {to_var}"
                else:
                    edge_str = f"{from_var} -- {to_var}"  # Default
            else:
                edge_str = str(edge)

            if freq > min_frequency:
                plot_causal_all_sep.append({
                    "sample_frac": frac,
                    "from_var": from_var,
                    "to_var": to_var,
                    "edge_type": edge_type,
                    "edge_str": edge_str,
                    "freq": freq
                })

    df_causal_structure = pd.DataFrame(plot_causal_all_sep)

    df_causal_structure = df_causal_structure.sort_values(
        by=["sample_frac", "freq"], ascending=[True, False]
    ).reset_index(drop=True)

    print(df_causal_structure)

    return df_causal_structure


def print_results(results, min_frequency=0.1):
    """
    Print analysis results with edges grouped by category.
    
    Args:
        results: Dictionary with analysis results
        min_frequency: Minimum frequency threshold to display (default 0.1 = 10%)
    """
    if not results or 'edge_counts' not in results:
        print("No results to display")
        return
    
    total = results['successful_iterations']
    edge_counts = results['edge_counts']
    edge_types = results.get('edge_types', {})
    demographic_vars = results.get('demographic_vars', [])
    survey_vars = results.get('survey_vars', [])
    
    # Filter edges by minimum frequency
    significant_edges = {edge: count for edge, count in edge_counts.items() 
                        if count / total >= min_frequency}
    
    if not significant_edges:
        print(f"\nNo edges appeared in >={min_frequency*100:.0f}% of iterations")
        return
    
    # Sort edges by frequency (descending)
    sorted_edges = sorted(significant_edges.items(), key=lambda x: x[1], reverse=True)
    
    # Categorize edges
    demographic_edges = []
    outcome_edges = []
    other_edges = []
    
    for edge, count in sorted_edges:
        freq = count / total
        edge_type = edge_types.get(edge, 'directed')
        
        if isinstance(edge, tuple) and len(edge) == 2:
            if edge_type == 'undirected':
                edge_str = f"{edge[0]} -- {edge[1]}"
                has_demo = any(var in demographic_vars for var in edge)
                is_outcome = any(any(sv in str(e) for sv in survey_vars) for e in edge)
            elif edge_type == 'bidirected':
                edge_str = f"{edge[0]} <-> {edge[1]}"
                has_demo = any(var in demographic_vars for var in edge)
                is_outcome = any(any(sv in str(e) for sv in survey_vars) for e in edge)
            else:
                # Directed edge
                from_var, to_var = edge
                edge_str = f"{from_var} -> {to_var}"
                has_demo = from_var in demographic_vars
                is_outcome = any(sv in str(to_var) or sv in str(from_var) for sv in survey_vars)
        else:
            edge_str = str(edge)
            has_demo = False
            is_outcome = False
        
        edge_info = (edge_str, count, freq)
        
        if has_demo:
            demographic_edges.append(edge_info)
        elif is_outcome:
            outcome_edges.append(edge_info)
        else:
            other_edges.append(edge_info)
    
    # Print results
    print("\n" + "="*70)
    print(f"RESULTS (edges appearing in >={min_frequency*100:.0f}% of iterations)")
    print("="*70)
    
    if demographic_edges:
        print(f"\nDEMOGRAPHIC INFLUENCES:")
        print("-" * 70)
        for edge_str, count, freq in demographic_edges:
            print(f"  {edge_str}: {count}/{total} ({freq*100:.1f}%)")
    
    if outcome_edges:
        print(f"\nEDGES INVOLVING DEPRESSION/ANXIETY:")
        print("-" * 70)
        for edge_str, count, freq in outcome_edges:
            print(f"  {edge_str}: {count}/{total} ({freq*100:.1f}%)")
    
    if other_edges:
        print(f"\nOTHER EDGES (Sensor-Sensor):")
        print("-" * 70)
        for edge_str, count, freq in other_edges:
            print(f"  {edge_str}: {count}/{total} ({freq*100:.1f}%)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY:")
    print(f"  Total unique edges: {len(edge_counts)}")
    print(f"  Edges >={min_frequency*100:.0f}% frequency: {len(significant_edges)}")
    print(f"  Demographic edges: {len(demographic_edges)}")
    print(f"  Outcome edges: {len(outcome_edges)}")
    print(f"  Other edges: {len(other_edges)}")
    print("="*70)


def check_specific_edges(results, var1, var2, min_frequency=0.0):
    """
    Check for all edges between two variables (across all time points and directions).
    
    Args:
        results: Dictionary with analysis results
        var1: First variable base name (e.g., 'rem_std', 'promis_dep_sum')
        var2: Second variable base name
        min_frequency: Minimum frequency threshold (default 0.0 to show all)
        
    Returns:
        List of edges found
    """
    if not results or 'edge_counts' not in results:
        print("No results to check")
        return []
    
    total = results['successful_iterations']
    edge_counts = results['edge_counts']
    edge_types = results.get('edge_types', {})
    
    print(f"\n{'='*70}")
    print(f"CHECKING EDGES BETWEEN: {var1} and {var2}")
    print(f"{'='*70}")
    
    # Generate all possible combinations
    time_suffixes = ['_tm1', '_t']
    var1_versions = [f"{var1}{suffix}" for suffix in time_suffixes]
    var2_versions = [f"{var2}{suffix}" for suffix in time_suffixes]
    
    found_edges = []
    
    # Check directed edges (both directions)
    for v1 in var1_versions:
        for v2 in var2_versions:
            # Check v1 -> v2
            edge_key = (v1, v2)
            if edge_key in edge_counts:
                count = edge_counts[edge_key]
                freq = count / total
                if freq >= min_frequency:
                    edge_type = edge_types.get(edge_key, 'directed')
                    found_edges.append((f"{v1} -> {v2}", count, freq, edge_type))
            
            # Check v2 -> v1
            edge_key = (v2, v1)
            if edge_key in edge_counts:
                count = edge_counts[edge_key]
                freq = count / total
                if freq >= min_frequency:
                    edge_type = edge_types.get(edge_key, 'directed')
                    found_edges.append((f"{v2} -> {v1}", count, freq, edge_type))
            
            # Check undirected (sorted tuple)
            edge_key = tuple(sorted([v1, v2]))
            if edge_key in edge_counts:
                count = edge_counts[edge_key]
                freq = count / total
                edge_type = edge_types.get(edge_key, 'undirected')
                if freq >= min_frequency and edge_type in ['undirected', 'bidirected']:
                    symbol = '--' if edge_type == 'undirected' else '<->'
                    found_edges.append((f"{v1} {symbol} {v2}", count, freq, edge_type))
    
    if found_edges:
        # Sort by frequency
        found_edges.sort(key=lambda x: x[1], reverse=True)
        print(f"\nFound {len(found_edges)} edges:")
        for edge_str, count, freq, edge_type in found_edges:
            print(f"  {edge_str}: {count}/{total} ({freq*100:.1f}%) [{edge_type}]")
    else:
        print(f"\nNo edges found between {var1} and {var2}")
    
    print("="*70)
    
    return found_edges


def run_analysis(filepath, n_bootstrap=100, sample_frac=[0.6], alpha=0.05,
                min_frequency=0.1, max_samples_per_iteration=None,
                min_days_gap=21, max_days_gap=35, use_first_pair_only=True,
                include_demographics=True, bin_age_var=True, age_bin_size=10, 
                race_encoding='grouped', save="none", results_dir = "causal_discovery_results_temp"):
    """
    Run complete causal discovery analysis pipeline.
    
    Args:
        filepath: Path to dataset file
        n_bootstrap: Number of bootstrap iterations
        sample_frac: Fraction of participants to sample per iteration
        alpha: Significance level for independence tests
        min_frequency: Minimum frequency threshold to display edges
        max_samples_per_iteration: Maximum number of samples per iteration (for speed)
        min_days_gap: Minimum days between surveys for valid pair (default 21 = 3 weeks)
        max_days_gap: Maximum days between surveys for valid pair (default 35 = 5 weeks)
        use_first_pair_only: If True, use only first valid pair per participant (faster)
        include_demographics: If True, include demographic variables
        bin_age_var: If True, bin age into categories
        age_bin_size: Size of age bins in years
        race_encoding: 'grouped' or 'detailed' for race encoding
        
    Returns:
        Dictionary with analysis results
    """
    # Load and encode data
    df = load_and_encode_data(filepath, include_demographics, 
                             bin_age_var, age_bin_size, race_encoding)
    if df is None:
        return None
    
    # Create adjacent survey pairs
    paired_df, feature_names, demographic_vars, sensor_vars, survey_vars = \
        create_adjacent_survey_pairs(df, include_demographics, bin_age_var,
                                    min_days_gap, max_days_gap, use_first_pair_only)
    
    # Run bootstrap analysis
    results = bootstrap_pc_analysis(paired_df, feature_names, demographic_vars, 
                                   sensor_vars, survey_vars, n_bootstrap, 
                                   sample_frac, alpha, max_samples_per_iteration,
                                   save, results_dir)
    
    # Print results
    # print_results(results, min_frequency)
    
    return results


# Example usage
if __name__ == "__main__":
    filepath = "data/preprocessed/full_run/4w_to_0w_before/survey_wearable_28d_before_to_0d_before_baseline_adj_full.csv"
    
    # For quick testing (faster but less stable):
    # results = run_analysis(
    #     filepath=filepath,
    #     n_bootstrap=10,  # Fewer iterations for testing
    #     sample_frac=0.3,  # Smaller sample fraction
    #     max_samples_per_iteration=1000,  # Limit samples per iteration
    #     alpha=0.05,
    #     min_frequency=0.3,  # Higher threshold since fewer iterations
    #     min_days_gap=21,  # 3 weeks
    #     max_days_gap=35,  # 5 weeks
    #     use_first_pair_only=True,  # Only first valid pair per participant
    #     include_demographics=True,
    #     bin_age_var=True,
    #     age_bin_size=10,
    #     race_encoding='grouped'
    # )
    
    # For full analysis with optimized speed:
    results = run_analysis(
        filepath=filepath,
        n_bootstrap=100, # Using backend LokyBackend with 23 concurrent workers, it takes ~2 min per sample_frac for n_bootstrap = 100
        sample_frac=[0.5, 0.6, 0.7, 0.8, 0.9], # Can be more fine-grained
        max_samples_per_iteration=None,
        alpha=0.05,
        min_frequency=0.1,
        min_days_gap=14,  # 3 weeks minimum gap
        max_days_gap=42,  # 5 weeks maximum gap
        use_first_pair_only=True,  # Use only first valid pair per participant (much faster!)
        include_demographics=True,
        bin_age_var=True,
        age_bin_size=10,
        race_encoding='grouped',
        save='both',
        results_dir = "causal_discovery_results_temp"
    )
