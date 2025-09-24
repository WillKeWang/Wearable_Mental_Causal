"""
Modular Causal Discovery Functions
=================================
Clean, reusable functions for temporal causal discovery analysis.
"""

import pandas as pd
import numpy as np
import random
import warnings
from collections import defaultdict
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

warnings.filterwarnings('ignore')


class DataProcessor:
    """Handle data loading and preparation."""
    
    @staticmethod
    def load_dataset(filepath: str, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load dataset with automatic format detection."""
        print(f"\nLoading {dataset_name}: {filepath}")
        
        # Try different file extensions
        for ext in ['', '.csv', '.txt']:
            try_path = filepath + ext if not filepath.endswith(('.csv', '.txt')) else filepath
            if os.path.exists(try_path):
                try:
                    df = pd.read_csv(try_path)
                    break
                except:
                    # Try tab-separated
                    df = pd.read_csv(try_path, sep='\t')
                    break
        else:
            print(f"ERROR: File not found: {filepath}")
            return None
        
        print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        if 'pid' in df.columns:
            print(f"Unique participants: {df['pid'].nunique()}")
        
        return df
    
    @staticmethod
    def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard cleaning filters (same as original notebook)."""
        print(f"Original shape: {df.shape}")
        
        # Drop missing values
        cleaned_df = df.dropna(axis=0, how='any')
        print(f"After dropping NAs: {cleaned_df.shape}")
        
        # Filter depression and anxiety scores (same ranges as notebook)
        if 'promis_dep_sum' in cleaned_df.columns:
            cleaned_df = cleaned_df[
                (cleaned_df['promis_dep_sum'] >= 4) & 
                (cleaned_df['promis_dep_sum'] <= 20)
            ]
            print(f"After depression filtering: {cleaned_df.shape}")
        
        if 'promis_anx_sum' in cleaned_df.columns:
            cleaned_df = cleaned_df[
                (cleaned_df['promis_anx_sum'] >= 4) & 
                (cleaned_df['promis_anx_sum'] <= 20)
            ]
            print(f"After anxiety filtering: {cleaned_df.shape}")
        
        if 'pid' in cleaned_df.columns:
            print(f"Final participants: {cleaned_df['pid'].nunique()}")
        
        return cleaned_df
    
    @staticmethod
    def prepare_variables(df: pd.DataFrame, 
                         base_vars: List[str] = None,
                         exclude_patterns: List[str] = None) -> Tuple[List[str], np.ndarray]:
        """Prepare variable list and data matrix."""
        if base_vars is None:
            base_vars = ["promis_dep_sum", "promis_anx_sum"]
        if exclude_patterns is None:
            exclude_patterns = ["total_"]
        
        # Get sensor metrics
        metric_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
        
        # Combine variables
        cols = base_vars + metric_cols
        
        # Remove excluded patterns
        for pattern in exclude_patterns:
            cols = [c for c in cols if not c.startswith(pattern)]
        
        print(f"Selected {len(cols)} variables")
        print(f"  Base variables: {base_vars}")
        print(f"  Sensor metrics: {len(metric_cols)}")
        
        # Create data matrix
        X = (df[cols]
             .apply(pd.to_numeric, errors="coerce")
             .dropna()
             .to_numpy())
        
        print(f"Analysis matrix: {X.shape[0]} rows × {X.shape[1]} variables")
        
        return cols, X


class BackgroundKnowledgeBuilder:
    """Create domain-specific background knowledge constraints."""
    
    @staticmethod
    def create_constraints(feature_names: List[str], 
                          base_vars: List[str],
                          forbid_sensor_correlations: bool = True,
                          require_sensor_to_outcome: bool = True) -> BackgroundKnowledge:
        """Create background knowledge constraints."""
        bk = BackgroundKnowledge()
        nodes = [GraphNode(name) for name in feature_names]
        name_to_node = {n.get_name(): n for n in nodes}
        
        constraint_count = 0
        
        if forbid_sensor_correlations:
            constraint_count += BackgroundKnowledgeBuilder._add_sensor_constraints(
                bk, feature_names, base_vars, name_to_node
            )
        
        if require_sensor_to_outcome:
            constraint_count += BackgroundKnowledgeBuilder._add_direction_constraints(
                bk, feature_names, base_vars, name_to_node
            )
        
        print(f"Created {constraint_count} background knowledge constraints")
        return bk
    
    @staticmethod
    def _add_sensor_constraints(bk: BackgroundKnowledge, 
                               feature_names: List[str],
                               base_vars: List[str],
                               name_to_node: Dict) -> int:
        """Forbid mean ↔ std connections for same sensor."""
        sensor_features = [name for name in feature_names if name not in base_vars]
        processed_bases = set()
        count = 0
        
        for feature in sensor_features:
            if feature.endswith('_mean'):
                base_name = feature.replace('_mean', '')
                std_feature = f"{base_name}_std"
                
                if std_feature in sensor_features and base_name not in processed_bases:
                    bk.add_forbidden_by_node(name_to_node[feature], name_to_node[std_feature])
                    bk.add_forbidden_by_node(name_to_node[std_feature], name_to_node[feature])
                    processed_bases.add(base_name)
                    count += 2
        
        return count
    
    @staticmethod
    def _add_direction_constraints(bk: BackgroundKnowledge,
                                  feature_names: List[str],
                                  base_vars: List[str],
                                  name_to_node: Dict) -> int:
        """Require sensor → outcome directions."""
        sensor_features = [name for name in feature_names if name not in base_vars]
        outcome_vars = [name for name in feature_names if name in base_vars]
        count = 0
        
        for sensor_feature in sensor_features:
            for outcome_var in outcome_vars:
                bk.add_required_by_node(name_to_node[sensor_feature], name_to_node[outcome_var])
                count += 1
        
        return count


class BootstrapAnalyzer:
    """Handle bootstrap causal discovery analysis."""
    
    @staticmethod
    def run_bootstrap_analysis(X: np.ndarray,
                              feature_names: List[str],
                              base_vars: List[str],
                              n_bootstrap: int = 100,
                              sample_frac: float = 0.6,
                              alpha: float = 0.05,
                              key_edges: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Run bootstrap analysis with PC algorithm."""
        
        if key_edges is None:
            key_edges = [
                ("rem_std", "promis_dep_sum"),
                ("deep_std", "promis_dep_sum"),
                ("promis_anx_sum", "promis_dep_sum")
            ]
        
        print(f"Starting bootstrap analysis: {n_bootstrap} iterations, {sample_frac*100}% sampling")
        
        edge_counts = defaultdict(int)
        key_edge_counts = {str(edge): 0 for edge in key_edges}
        successful_iterations = 0
        
        for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
            try:
                # Bootstrap sample
                n_sample = int(X.shape[0] * sample_frac)
                sample_indices = random.sample(range(X.shape[0]), n_sample)
                X_bootstrap = X[sample_indices, :]
                
                # Create background knowledge
                bk = BackgroundKnowledgeBuilder.create_constraints(feature_names, base_vars)
                
                # Run PC algorithm silently
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
                edges_this_iteration = BootstrapAnalyzer._extract_edges(
                    cg.G, feature_names, edge_counts
                )
                
                # Count key edges
                BootstrapAnalyzer._count_key_edges(
                    edges_this_iteration, key_edges, key_edge_counts
                )
                
                successful_iterations += 1
                
            except Exception:
                continue
        
        print(f"Bootstrap completed: {successful_iterations}/{n_bootstrap} successful iterations")
        
        return {
            'edge_counts': dict(edge_counts),
            'key_edge_counts': key_edge_counts,
            'successful_iterations': successful_iterations,
            'total_iterations': n_bootstrap,
            'key_edges': key_edges
        }
    
    @staticmethod
    def _extract_edges(graph, feature_names: List[str], edge_counts: Dict) -> List[Tuple]:
        """Extract edges from causal graph."""
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
        
        return edges_this_iteration
    
    @staticmethod
    def _count_key_edges(edges_this_iteration: List[Tuple],
                        key_edges: List[Tuple[str, str]],
                        key_edge_counts: Dict[str, int]) -> None:
        """Count occurrences of key edges."""
        for edge in key_edges:
            edge_str = str(edge)
            if (edge in edges_this_iteration or 
                tuple(sorted(edge)) in edges_this_iteration or
                (edge[1], edge[0]) in edges_this_iteration):
                key_edge_counts[edge_str] += 1


class ResultsAnalyzer:
    """Analyze and compare results across datasets."""
    
    @staticmethod
    def analyze_stability(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate stability metrics for key edges."""
        key_edge_counts = results['key_edge_counts']
        successful_iterations = results['successful_iterations']
        
        stability_analysis = {}
        
        for edge_str, count in key_edge_counts.items():
            stability = count / successful_iterations if successful_iterations > 0 else 0
            
            if stability >= 0.7:
                assessment = "STRONG"
            elif stability >= 0.5:
                assessment = "MODERATE"
            elif stability >= 0.3:
                assessment = "WEAK"
            else:
                assessment = "UNSTABLE"
            
            stability_analysis[edge_str] = {
                'count': count,
                'total': successful_iterations,
                'stability': stability,
                'assessment': assessment
            }
        
        return stability_analysis
    
    @staticmethod
    def compare_temporal_results(all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare results across temporal windows."""
        print(f"\n{'='*60}")
        print("RESULTS COMPARISON")
        print(f"{'='*60}")
        
        comparison = {
            'dataset_results': {},
            'consistency_analysis': {}
        }
        
        # Analyze each dataset
        for dataset_name, results in all_results.items():
            if results:
                stability_analysis = ResultsAnalyzer.analyze_stability(results)
                comparison['dataset_results'][dataset_name] = stability_analysis
                
                total = results['successful_iterations']
                print(f"\n{dataset_name}:")
                
                for edge_str, analysis in stability_analysis.items():
                    count = analysis['count']
                    stability = analysis['stability']
                    assessment = analysis['assessment']
                    print(f"  {edge_str}: {count}/{total} ({stability*100:.1f}%) - {assessment}")
        
        # Consistency assessment
        if len(comparison['dataset_results']) >= 2:
            comparison['consistency_analysis'] = ResultsAnalyzer._assess_consistency(
                comparison['dataset_results']
            )
        
        return comparison
    
    @staticmethod
    def _assess_consistency(dataset_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess consistency of key relationships across datasets."""
        print(f"\nConsistency Assessment:")
        
        consistency_analysis = {}
        
        # Get all unique edges
        all_edges = set()
        for results in dataset_results.values():
            all_edges.update(results.keys())
        
        for edge_str in all_edges:
            stabilities = []
            assessments = []
            
            for dataset_name, results in dataset_results.items():
                if edge_str in results:
                    stabilities.append(results[edge_str]['stability'])
                    assessments.append(results[edge_str]['assessment'])
            
            if len(stabilities) >= 2:
                stability_diff = max(stabilities) - min(stabilities)
                consistency = "HIGH" if stability_diff < 0.2 else "MODERATE" if stability_diff < 0.4 else "LOW"
                
                consistency_analysis[edge_str] = {
                    'stabilities': stabilities,
                    'difference': stability_diff,
                    'consistency': consistency,
                    'assessments': assessments
                }
                
                # Clean edge name for display
                edge_display = edge_str.replace("('", "").replace("')", "").replace("', '", " → ")
                print(f"  {edge_display}: {consistency} consistency")
                
                for i, (dataset_name, stability) in enumerate(zip(dataset_results.keys(), stabilities)):
                    print(f"    {dataset_name}: {stability:.1%} ({assessments[i]})")
        
        return consistency_analysis
    
    @staticmethod
    def save_results(comparison_results: Dict[str, Any], 
                    output_dir: str = "results/temporal_analysis") -> None:
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_file = output_path / "temporal_analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Temporal Causal Discovery Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Dataset Results:\n")
            for dataset_name, results in comparison_results['dataset_results'].items():
                f.write(f"\n{dataset_name}:\n")
                for edge_str, analysis in results.items():
                    f.write(f"  {edge_str}: {analysis['stability']:.1%} ({analysis['assessment']})\n")
            
            if 'consistency_analysis' in comparison_results:
                f.write("\nConsistency Analysis:\n")
                for edge_str, consistency_info in comparison_results['consistency_analysis'].items():
                    f.write(f"  {edge_str}: {consistency_info['consistency']} consistency\n")
        
        print(f"\nResults saved to {summary_file}")


class TemporalCausalAnalyzer:
    """Main orchestrator for temporal causal discovery analysis."""
    
    def __init__(self, 
                 base_vars: List[str] = None,
                 exclude_patterns: List[str] = None,
                 key_edges: List[Tuple[str, str]] = None,
                 n_bootstrap: int = 100,
                 sample_frac: float = 0.6,
                 alpha: float = 0.05):
        
        self.base_vars = base_vars or ["promis_dep_sum", "promis_anx_sum"]
        self.exclude_patterns = exclude_patterns or ["total_"]
        self.key_edges = key_edges or [
            ("rem_std", "promis_dep_sum"),
            ("deep_std", "promis_dep_sum"),
            ("promis_anx_sum", "promis_dep_sum")
        ]
        self.n_bootstrap = n_bootstrap
        self.sample_frac = sample_frac
        self.alpha = alpha
        
        self.results = {}
    
    def analyze_dataset(self, filepath: str, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Analyze a single dataset."""
        # Load and clean data
        df = DataProcessor.load_dataset(filepath, dataset_name)
        if df is None:
            return None
        
        cleaned_df = DataProcessor.clean_dataset(df)
        
        # Prepare variables and data matrix
        feature_names, X = DataProcessor.prepare_variables(
            cleaned_df, self.base_vars, self.exclude_patterns
        )
        
        # Run bootstrap analysis
        results = BootstrapAnalyzer.run_bootstrap_analysis(
            X, feature_names, self.base_vars,
            n_bootstrap=self.n_bootstrap,
            sample_frac=self.sample_frac,
            alpha=self.alpha,
            key_edges=self.key_edges
        )
        
        return results
    
    def analyze_multiple_datasets(self, dataset_paths: Dict[str, str]) -> Dict[str, Any]:
        """Analyze multiple datasets and compare results."""
        print("=" * 70)
        print("TEMPORAL CAUSAL DISCOVERY ANALYSIS")
        print("=" * 70)
        
        # Analyze each dataset
        for dataset_name, filepath in dataset_paths.items():
            self.results[dataset_name] = self.analyze_dataset(filepath, dataset_name)
        
        # Compare results
        comparison_results = ResultsAnalyzer.compare_temporal_results(self.results)
        
        # Save results
        ResultsAnalyzer.save_results(comparison_results)
        
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*70}")
        
        return comparison_results


def run_temporal_causal_analysis(dataset_paths: Dict[str, str],
                                **kwargs) -> Dict[str, Any]:
    """Convenience function to run complete temporal analysis."""
    analyzer = TemporalCausalAnalyzer(**kwargs)
    return analyzer.analyze_multiple_datasets(dataset_paths)


# Example usage
if __name__ == "__main__":
    # Example dataset paths
    dataset_paths = {
        "1w_to_5w_after": "data/preprocessed/full_run/1w_to_5w_after/survey_wearable_7d_after_to_35d_after_baseline_adj_full.csv",
        "6w_to_2w_before": "data/preprocessed/full_run/6w_to_2w_before/survey_wearable_42d_before_to_14d_before_baseline_adj_full.csv"
    }
    
    # Run analysis
    results = run_temporal_causal_analysis(
        dataset_paths,
        n_bootstrap=100,  # Same as your successful analysis
        sample_frac=0.6,  # Same as your successful analysis
        alpha=0.05
    )