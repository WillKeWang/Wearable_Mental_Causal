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


def load_and_prepare_data(filepath, dataset_name, include_demographics=True, 
                         bin_age_var=True, age_bin_size=10, race_encoding='grouped'):
    """
    Load and prepare data.
    
    Args:
        filepath: Path to dataset file
        dataset_name: Name for display purposes
        include_demographics: If True, encode and include demographic variables
        bin_age_var: If True, bin age into categories (e.g., 20s, 30s)
        age_bin_size: Size of age bins in years (default 10)
        race_encoding: 'grouped' (4 categories) or 'detailed' (6 categories) for race
        
    Returns:
        DataFrame with cleaned and encoded data, or None if file not found
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
    
    # Encode demographic variables if requested
    if include_demographics:
        print("\nEncoding demographic variables:")
        available_demos = []
        missing_demos = []
        
        # Check which demographic variables are available
        for demo_var in ['age', 'sex', 'race', 'ethnicity_hispanic']:
            if demo_var in df.columns:
                available_demos.append(demo_var)
            else:
                missing_demos.append(demo_var)
        
        if missing_demos:
            print(f"  Missing demographic variables: {missing_demos}")
        
        if not available_demos:
            print("  No demographic variables found. Continuing without demographics...")
            include_demographics = False
        else:
            # Encode all available demographic variables
            df = encode_demographics(df, bin_age_var, age_bin_size, race_encoding)
    
    # Cleaning for valid range of PROMIS scores
    cleaned_df = df.dropna(axis=0, how='any')
    cleaned_df = cleaned_df[cleaned_df['promis_dep_sum'] >= 4]
    cleaned_df = cleaned_df[cleaned_df['promis_anx_sum'] >= 4]
    cleaned_df = cleaned_df[cleaned_df['promis_dep_sum'] <= 20]
    cleaned_df = cleaned_df[cleaned_df['promis_anx_sum'] <= 20]
    
    print(f"\nAfter cleaning: {cleaned_df.shape}")
    print(f"Unique users: {cleaned_df['pid'].nunique()}")
    
    return cleaned_df


def prepare_variables(df, keep_pid=False, include_demographics=True, use_binned_age=True):
    """
    Prepare variables and data matrix.
    
    Args:
        df: Cleaned DataFrame (with encoded demographics if applicable)
        keep_pid: If True, return pid column alongside data
        include_demographics: If True, include demographic variables
        use_binned_age: If True, use binned age instead of continuous age
        
    Returns:
        If keep_pid=False: Tuple of (column_names, data_matrix)
        If keep_pid=True: Tuple of (column_names, data_matrix, pid_array)
    """
    # Same variable selection as notebook
    base_vars = ["promis_dep_sum", "promis_anx_sum"]
    metric_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    cols = base_vars + metric_cols
    cols = [c for c in cols if not c.startswith("total_")]  # Remove total_ variables
    
    # Add demographic variables if requested
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
    
    # Add demographics at the beginning (they causally precede everything)
    cols = demographic_vars + cols
    
    print(f"\nSelected {len(cols)} variables:")
    if demographic_vars:
        print(f"  Demographic variables ({len(demographic_vars)}): {demographic_vars}")
    print(f"  Outcome variables (2): {base_vars}")
    print(f"  Wearable metrics ({len(metric_cols)}): {len([c for c in metric_cols if c not in cols[:len(demographic_vars) + 2]])} after filtering")
    
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


def bootstrap_pc_analysis_by_pid(df, feature_names, base_vars, demographic_vars=None,
                                 n_bootstrap=100, sample_frac=0.6, alpha=0.05, 
                                 use_first_survey=True, dataset_type=None, verbose=False):
    """
    Run bootstrap analysis with PC algorithm, sampling at participant (pid) level.
    
    Args:
        df: DataFrame with cleaned data (must include 'pid' column)
        feature_names: List of variable names (excluding 'pid')
        base_vars: List of outcome variables
        demographic_vars: List of demographic variables (age, sex) that can only be causes
        n_bootstrap: Number of bootstrap iterations
        sample_frac: Fraction of participants to sample in each iteration
        alpha: Significance level for independence tests
        use_first_survey: If True, use only first (earliest) survey per pid
        dataset_type: 'before' or 'after' to enforce temporal direction
        verbose: If True, print warnings about edge direction corrections
        
    Returns:
        Dictionary with edge counts and key edge results
    """
    if demographic_vars is None:
        demographic_vars = []
    
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
    if demographic_vars:
        print(f"Demographic variables (root causes only): {demographic_vars}")
    
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
            # Create background knowledge with temporal direction and demographic constraints
            bk = create_background_knowledge(feature_names, base_vars, demographic_vars, 
                                            dataset_type=dataset_type)
            
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
                return var not in base_vars and var not in demographic_vars
            
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
    
    # TODO: Yishu to add another function to report the entire edge list with frequencies

    return {
        'edge_counts': edge_counts,
        'edge_types': edge_types,
        'successful_iterations': successful_iterations,
        'rem_dep_count': rem_dep_count,
        'deep_dep_count': deep_dep_count,
        'anxiety_dep_count': anxiety_dep_count,
        'dataset_type': dataset_type,
        'direction_corrections': direction_corrections,
        'demographic_vars': demographic_vars
    }


def create_background_knowledge(feature_names, base_vars, demographic_vars=None, 
                               dataset_type='before'):
    """
    Create background knowledge constraints for PC algorithm.
    
    Args:
        feature_names: List of all variable names
        base_vars: List of outcome variables (depression, anxiety)
        demographic_vars: List of demographic variables (age, sex) that can only be root causes
        dataset_type: 'before' or 'after' to enforce temporal direction, None for no temporal constraint
        
    Returns:
        BackgroundKnowledge object with constraints
    """
    if demographic_vars is None:
        demographic_vars = []
    
    bk = BackgroundKnowledge()
    nodes = [GraphNode(name) for name in feature_names]
    name_to_node = {n.get_name(): n for n in nodes}
    
    # ========================================
    # DEMOGRAPHIC CONSTRAINTS (ROOT CAUSES)
    # ========================================
    # Demographic variables can ONLY be causes, never outcomes
    # This means: allow demo -> anything, forbid anything -> demo
    for demo_var in demographic_vars:
        if demo_var in name_to_node:
            for other_var in feature_names:
                if other_var != demo_var:
                    # Forbid: other_var -> demo_var
                    # This ensures demographics are never affected by anything
                    bk.add_forbidden_by_node(name_to_node[other_var], name_to_node[demo_var])
    
    print(f"Added demographic constraints: {len(demographic_vars)} variables as root causes only")
    
    # ========================================
    # SENSOR FEATURE CONSTRAINTS
    # ========================================
    # Forbid mean -> std connections for same sensor
    sensor_features = [name for name in feature_names 
                      if name not in base_vars and name not in demographic_vars]
    processed_bases = set()
    
    for feature in sensor_features:
        if feature.endswith('_mean'):
            base_name = feature.replace('_mean', '')
            std_feature = f"{base_name}_std"
            
            if std_feature in sensor_features and base_name not in processed_bases:
                bk.add_forbidden_by_node(name_to_node[feature], name_to_node[std_feature])
                bk.add_forbidden_by_node(name_to_node[std_feature], name_to_node[feature])
                processed_bases.add(base_name)
    
    # ========================================
    # TEMPORAL DIRECTION CONSTRAINTS
    # ========================================
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


def analyze_single_dataset(filepath, dataset_name, n_bootstrap=100, sample_frac=0.5, alpha=0.05, 
                          use_pid_bootstrap=True, use_first_survey=True, 
                          include_demographics=True, bin_age_var=True, age_bin_size=10,
                          race_encoding='grouped'):
    """
    Analyze a single dataset with PC algorithm.
    
    Args:
        filepath: Path to dataset file
        dataset_name: Name for display purposes
        n_bootstrap: Number of bootstrap iterations
        sample_frac: Fraction of data/participants to sample
        alpha: Significance level
        use_pid_bootstrap: If True, use pid-level bootstrapping
        use_first_survey: If True (and use_pid_bootstrap=True), use only first surveys
        include_demographics: If True, include demographic variables as covariates
        bin_age_var: If True, use binned age categories
        age_bin_size: Size of age bins in years (default 10)
        race_encoding: 'grouped' (4 categories) or 'detailed' (6 categories)
        
    Returns:
        Dictionary with analysis results, or None if data loading fails
    """
    # Load and prepare data
    df = load_and_prepare_data(filepath, dataset_name, include_demographics, 
                               bin_age_var, age_bin_size, race_encoding)
    if df is None:
        return None
    
    base_vars = ["promis_dep_sum", "promis_anx_sum"]
    
    # Infer dataset type from name or path
    dataset_type = infer_dataset_type(dataset_name)
    
    # Identify demographic variables in the dataset (encoded versions)
    demographic_vars = []
    if include_demographics:
        # Age
        if bin_age_var and 'age_binned' in df.columns:
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
    
    if use_pid_bootstrap:
        # Prepare feature names (excluding pid which we'll handle separately)
        metric_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
        feature_names = demographic_vars + base_vars + metric_cols
        feature_names = [c for c in feature_names if not c.startswith("total_")]
        
        # Run PID-level bootstrap analysis
        results = bootstrap_pc_analysis_by_pid(df, feature_names, base_vars, demographic_vars,
                                              n_bootstrap, sample_frac, alpha, 
                                              use_first_survey, dataset_type)
    else:
        # Prepare variables (original method)
        feature_names, X = prepare_variables(df, include_demographics, bin_age_var, keep_pid=False)
        
        # Note: Original row-level bootstrap doesn't use dataset_type
        # Would need to be added if needed
        results = bootstrap_pc_analysis(X, feature_names, base_vars, n_bootstrap, sample_frac, alpha)
        results['dataset_type'] = dataset_type
        results['demographic_vars'] = demographic_vars
    
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
        dataset_type = results.get('dataset_type')
        corrections = results.get('direction_corrections', 0)
        demographic_vars = results.get('demographic_vars', [])
        
        # FIXED: Print correct direction based on dataset type
        print(f"\n{dataset_name}:")
        if demographic_vars:
            print(f"  [Demographics included as root causes: {', '.join(demographic_vars)}]")
        if dataset_type == 'after':
            print(f"  promis_dep_sum -> rem_std: {rem_dep}/{total} ({rem_dep/total*100:.1f}%)")
            print(f"  promis_dep_sum -> deep_std: {deep_dep}/{total} ({deep_dep/total*100:.1f}%)")
        else:  # 'before' or default
            print(f"  rem_std -> promis_dep_sum: {rem_dep}/{total} ({rem_dep/total*100:.1f}%)")
            print(f"  deep_std -> promis_dep_sum: {deep_dep}/{total} ({deep_dep/total*100:.1f}%)")
        print(f"  promis_anx_sum <-> promis_dep_sum: {anx_dep}/{total} ({anx_dep/total*100:.1f}%)")
        
        if corrections > 0:
            print(f"  [Note: {corrections} survey-wearable edges were corrected to enforce temporal direction]")
        
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
    demographic_vars = results.get('demographic_vars', [])
    
    # Filter edges by minimum frequency
    significant_edges = {edge: count for edge, count in edge_counts.items() 
                        if count / total >= min_frequency}
    
    if not significant_edges:
        print(f"\nNo edges appeared in >={min_frequency*100:.0f}% of iterations")
        return
    
    # Helper function to check if variable is a wearable metric
    def is_wearable_var(var):
        return var not in outcome_vars and var not in demographic_vars and \
               (var.endswith('_mean') or var.endswith('_std'))
    
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
    
    # Separate edges into categories
    demographic_edges = []  # Edges from demographics to other variables
    outcome_edges = []      # Edges involving outcome variables
    other_edges = []        # Other edges
    
    for edge, count in sorted_edges:
        freq = count / total
        edge_type = edge_types.get(edge, 'directed')
        
        if isinstance(edge, tuple) and len(edge) == 2:
            if edge_type == 'undirected':
                # Undirected edge
                edge_str = f"{edge[0]} -- {edge[1]}"
                has_demo = any(var in demographic_vars for var in edge)
                is_outcome = any(var in outcome_vars for var in edge)
            else:
                # Directed edge
                from_var, to_var = edge
                edge_str = f"{from_var} -> {to_var}"
                has_demo = from_var in demographic_vars
                is_outcome = to_var in outcome_vars or from_var in outcome_vars
        else:
            # Shouldn't happen, but handle gracefully
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
    
    # Print header with temporal context
    direction_note = ""
    if dataset_type == 'before':
        direction_note = " [Showing: Wearable -> Survey edges only]"
    elif dataset_type == 'after':
        direction_note = " [Showing: Survey -> Wearable edges only]"
    
    # Print demographic edges (if any)
    if demographic_edges:
        print(f"\n  {'='*70}")
        print(f"  DEMOGRAPHIC INFLUENCES (>={min_frequency*100:.0f}%)")
        print(f"  {'='*70}")
        for edge_str, count, freq in demographic_edges:
            print(f"  * {edge_str}: {count}/{total} ({freq*100:.1f}%)")
    
    # Print edges involving outcome variables
    if outcome_edges:
        print(f"\n  {'='*70}")
        print(f"  EDGES INVOLVING DEPRESSION/ANXIETY (>={min_frequency*100:.0f}%){direction_note}")
        print(f"  {'='*70}")
        for edge_str, count, freq in outcome_edges:
            print(f"  * {edge_str}: {count}/{total} ({freq*100:.1f}%)")
    
    # Print other edges
    if other_edges:
        print(f"\n  {'='*70}")
        print(f"  OTHER EDGES (>={min_frequency*100:.0f}%){direction_note}")
        print(f"  {'='*70}")
        for edge_str, count, freq in other_edges:
            print(f"    {edge_str}: {count}/{total} ({freq*100:.1f}%)")
    
    # Summary statistics
    print(f"\n  Summary:")
    print(f"    Total unique edges found: {len(edge_counts)}")
    print(f"    Edges >={min_frequency*100:.0f}% frequency: {len(significant_edges)}")
    if dataset_type:
        print(f"    Edges displayed after temporal filtering: {len(sorted_edges)}")
    if demographic_vars:
        print(f"    Edges from demographics: {len(demographic_edges)}")
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
                            include_demographics=True, bin_age_var=True, age_bin_size=10,
                            race_encoding='grouped'):
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
        include_demographics: If True, include demographic variables as root cause covariates
        bin_age_var: If True, bin age into categories (e.g., 20s, 30s)
        age_bin_size: Size of age bins in years (default 10)
        race_encoding: 'grouped' (4 categories) or 'detailed' (6 categories) for race variable
        
    Returns:
        Dictionary mapping dataset names to analysis results
    """
    print("="*60)
    print("TEMPORAL CAUSAL DISCOVERY ANALYSIS (PC ALGORITHM)")
    if use_pid_bootstrap:
        print(f"Bootstrapping: PID-level (first survey only: {use_first_survey})")
    else:
        print("Bootstrapping: Row-level")
    if include_demographics:
        age_type = f"binned ({age_bin_size}-year bins)" if bin_age_var else "continuous"
        print(f"Demographics included as root causes:")
        print(f"  - Age: {age_type}")
        print(f"  - Sex: binary (0=female, 1=male)")
        print(f"  - Race: {race_encoding} encoding")
        print(f"  - Ethnicity: binary (0=not hispanic, 1=hispanic)")
    print("="*60)
    
    results = {}
    
    # Analyze each dataset
    for name, path in dataset_paths.items():
        results[name] = analyze_single_dataset(path, name, n_bootstrap, sample_frac, alpha,
                                              use_pid_bootstrap, use_first_survey,
                                              include_demographics, bin_age_var, 
                                              age_bin_size, race_encoding)
    
    # Compare results
    compare_temporal_results(results, min_frequency=min_frequency)
    
    return results

# TODO: Add function to count and report the entire edge list with frequencies 

# Example usage
if __name__ == "__main__":
    # Example dataset paths
    dataset_paths = {
        "1w_to_5w_after": "data/preprocessed/full_run/1w_to_5w_after/survey_wearable_7d_after_to_35d_after_baseline_adj_full.csv",
        "6w_to_2w_before": "data/preprocessed/full_run/6w_to_2w_before/survey_wearable_42d_before_to_14d_before_baseline_adj_full.csv"
    }
    
    # Run analysis with PID-level bootstrapping on first surveys
    # Now includes all demographic variables (age, sex, race, ethnicity) as root cause covariates
    # Demographics can only be causes in the causal graph, never outcomes
    # min_frequency=0.1 means only show edges that appear in >=10% of iterations
    results = run_temporal_pc_analysis(
        dataset_paths,
        n_bootstrap=100,
        sample_frac=0.5,
        alpha=0.05,
        use_pid_bootstrap=True,
        use_first_survey=True,
        min_frequency=0.1,
        include_demographics=True,    # Include demographic variables
        bin_age_var=True,              # Use binned age (20s, 30s, etc.)
        age_bin_size=10,               # 10-year bins
        race_encoding='grouped'        # Use grouped race encoding (4 categories)
                                       # Options: 'grouped' or 'detailed'
    )