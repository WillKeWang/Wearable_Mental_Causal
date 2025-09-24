"""
Simple Temporal Causal Discovery Analysis
========================================
Direct adaptation of your notebook for the two new temporal datasets.
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
    """Load and prepare data exactly like the original notebook."""
    print(f"\nLoading {dataset_name}: {filepath}")
    
    # Try different file extensions
    for ext in ['', '.csv', '.txt']:
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

def create_background_knowledge(feature_names, base_vars):
    """Create background knowledge - same as original notebook."""
    bk = BackgroundKnowledge()
    nodes = [GraphNode(name) for name in feature_names]
    name_to_node = {n.get_name(): n for n in nodes}
    
    # Forbid mean ↔ std connections for same sensor
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
    
    # Require sensor → outcome directions
    outcome_vars = [name for name in feature_names if name in base_vars]
    for sensor_feature in sensor_features:
        for outcome_var in outcome_vars:
            bk.add_required_by_node(name_to_node[sensor_feature], name_to_node[outcome_var])
    
    return bk

def bootstrap_analysis(X, feature_names, base_vars, n_bootstrap=100):
    """Bootstrap analysis - same as original notebook."""
    edge_counts = defaultdict(int)
    successful_iterations = 0
    
    # Track key edges
    rem_dep_count = 0
    deep_dep_count = 0
    anxiety_dep_count = 0
    
    for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
        n_sample = int(X.shape[0] * 0.6)  # Same 60% sampling
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
                        alpha=0.05,
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
                        elif adj_matrix[i_node, j_node] != 0:
                            # Directed i -> j
                            edge_key = (from_var, to_var)
                        elif adj_matrix[j_node, i_node] != 0:
                            # Directed j -> i
                            edge_key = (to_var, from_var)
                        
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
        'successful_iterations': successful_iterations,
        'rem_dep_count': rem_dep_count,
        'deep_dep_count': deep_dep_count,
        'anxiety_dep_count': anxiety_dep_count
    }

def analyze_dataset(filepath, dataset_name):
    """Analyze a single dataset."""
    # Load and prepare data
    df = load_and_prepare_data(filepath, dataset_name)
    if df is None:
        return None
    
    # Same variable selection as notebook
    base_vars = ["promis_dep_sum", "promis_anx_sum"]
    metric_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    cols = base_vars + metric_cols
    cols = [c for c in cols if not c.startswith("total_")]  # Remove total_ variables
    
    print(f"Selected {len(cols)} variables")
    
    # Create data matrix
    X = (df[cols]
         .apply(pd.to_numeric, errors="coerce")
         .dropna()
         .to_numpy())
    
    print(f"Analysis matrix: {X.shape[0]} rows × {X.shape[1]} variables")
    
    # Run bootstrap analysis
    results = bootstrap_analysis(X, cols, base_vars, n_bootstrap=100)
    
    return results

def main():
    """Main analysis function."""
    print("="*60)
    print("TEMPORAL CAUSAL DISCOVERY ANALYSIS")
    print("="*60)
    
    # Dataset paths
    datasets = {
        "1w_to_5w_after": "data/preprocessed/full_run/1w_to_5w_after/survey_wearable_7d_after_to_35d_after_baseline_adj_full.csv",
        "6w_to_2w_before": "data/preprocessed/full_run/6w_to_2w_before/survey_wearable_42d_before_to_14d_before_baseline_adj_full.csv"
    }
    
    results = {}
    
    # Analyze each dataset
    for name, path in datasets.items():
        results[name] = analyze_dataset(path, name)
    
    # Compare results
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    
    for name, result in results.items():
        if result:
            total = result['successful_iterations']
            rem_dep = result['rem_dep_count']
            deep_dep = result['deep_dep_count']
            anx_dep = result['anxiety_dep_count']
            
            print(f"\n{name}:")
            print(f"  rem_std → promis_dep_sum: {rem_dep}/{total} ({rem_dep/total*100:.1f}%)")
            print(f"  deep_std → promis_dep_sum: {deep_dep}/{total} ({deep_dep/total*100:.1f}%)")
            print(f"  promis_anx_sum ↔ promis_dep_sum: {anx_dep}/{total} ({anx_dep/total*100:.1f}%)")
    
    # Consistency check
    if len([r for r in results.values() if r]) >= 2:
        print(f"\nConsistency Assessment:")
        valid_results = {k: v for k, v in results.items() if v}
        
        rem_stabilities = [v['rem_dep_count']/v['successful_iterations'] for v in valid_results.values()]
        rem_consistency = max(rem_stabilities) - min(rem_stabilities) < 0.2
        
        print(f"  rem_std → depression: {'CONSISTENT' if rem_consistency else 'VARIABLE'} across time windows")
        
        for i, (name, stability) in enumerate(zip(valid_results.keys(), rem_stabilities)):
            assessment = "STRONG" if stability >= 0.7 else "MODERATE" if stability >= 0.5 else "WEAK"
            print(f"    {name}: {stability:.1%} ({assessment})")

if __name__ == "__main__":
    main()