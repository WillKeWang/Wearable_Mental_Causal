"""
Modular causal discovery utilities for wearable data analysis.

This module provides configurable causal discovery analysis including:
- Data splitting for discovery/validation
- Background knowledge constraints
- PC algorithm with domain knowledge
- Bootstrap stability analysis
- Graph visualization and export
"""

import os
import io
import random
import warnings
import logging
import unittest
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set
from pathlib import Path
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Causal discovery imports
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import fisherz
from causallearn.search.ScoreBased.GES import ges
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.Node import Node
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class CausalDiscoveryConfig:
    """Configuration for causal discovery analysis."""
    
    # Data configuration
    outcome_vars: List[str] = field(default_factory=lambda: ["promis_dep_sum", "promis_anx_sum"])
    metric_patterns: List[str] = field(default_factory=lambda: ["_mean", "_std"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["total_"])
    
    # Data splitting
    discovery_split: float = 0.5  # Fraction for discovery set
    validation_split: float = 0.5  # Fraction for validation set
    split_seed: int = 42
    max_samples: Optional[int] = None  # Limit total samples (None = all)
    
    # PC algorithm parameters
    alpha: float = 0.05
    independence_test: str = "fisherz"  # fisherz, kci, etc.
    stable: bool = True
    uc_rule: int = 0
    verbose: bool = False
    
    # Background knowledge
    forbid_sensor_correlations: bool = True  # Forbid mean ↔ std of same sensor
    require_sensor_to_outcome: bool = True   # Require sensor → outcome direction
    
    # Bootstrap analysis
    n_bootstrap: int = 100
    bootstrap_sample_frac: float = 0.6
    stability_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "strong": 0.7,
        "moderate": 0.5,
        "weak": 0.3
    })
    
    # Key edges to track
    key_edges: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("rem_std", "promis_dep_sum"),
        ("deep_std", "promis_dep_sum"),
        ("promis_anx_sum", "promis_dep_sum")
    ])
    
    # Output configuration
    output_dir: str = "results/causal_discovery"
    save_graphs: bool = True
    graph_format: str = "png"
    save_bootstrap_results: bool = True
    
    # Visualization
    figure_size: Tuple[int, int] = (12, 10)
    show_graphs: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.discovery_split < 1:
            raise ValueError("discovery_split must be between 0 and 1")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if not 0 < self.bootstrap_sample_frac <= 1:
            raise ValueError("bootstrap_sample_frac must be between 0 and 1")
        if self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")
    
    def generate_filename(self, file_type: str, additional_info: str = "") -> str:
        """Generate descriptive filename based on configuration."""
        base_name = f"causal_discovery_alpha{self.alpha}_bs{self.n_bootstrap}"
        if additional_info:
            base_name += f"_{additional_info}"
        
        if file_type == "bootstrap_results":
            return f"{base_name}_bootstrap_stability.csv"
        elif file_type == "pc_graph":
            return f"{base_name}_pc_graph.{self.graph_format}"
        elif file_type == "edge_summary":
            return f"{base_name}_edge_summary.csv"
        elif file_type == "data_summary":
            return f"{base_name}_data_summary.txt"
        else:
            raise ValueError(f"Unknown file_type: {file_type}")


class CausalDiscoveryAnalyzer:
    """Main class for causal discovery analysis."""
    
    def __init__(self, config: CausalDiscoveryConfig):
        """Initialize analyzer with configuration."""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.df_discovery = None
        self.df_validation = None
        self.feature_names = []
        self.X_discovery = None
        self.X_validation = None
        
        logger.info(f"Initialized CausalDiscoveryAnalyzer with config: {config}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into discovery and validation sets."""
        logger.info("Preparing data for causal discovery...")
        
        # Get feature columns
        metric_cols = []
        for pattern in self.config.metric_patterns:
            metric_cols.extend([c for c in df.columns if c.endswith(pattern)])
        
        # Remove excluded patterns
        for exclude_pattern in self.config.exclude_patterns:
            metric_cols = [c for c in metric_cols if not c.startswith(exclude_pattern)]
        
        self.feature_names = self.config.outcome_vars + list(set(metric_cols))
        logger.info(f"Selected {len(self.feature_names)} features: {len(self.config.outcome_vars)} outcomes + {len(set(metric_cols))} metrics")
        
        # Limit samples if specified
        if self.config.max_samples and len(df) > self.config.max_samples:
            df = df.sample(n=self.config.max_samples, random_state=self.config.split_seed)
            logger.info(f"Limited to {self.config.max_samples} samples")
        
        # Split subjects (not rows) for discovery/validation
        unique_pids = df['pid'].unique()
        np.random.seed(self.config.split_seed)
        shuffled_pids = np.random.permutation(unique_pids)
        
        n_subjects = len(unique_pids)
        discovery_split_point = int(n_subjects * self.config.discovery_split)
        
        discovery_pids = shuffled_pids[:discovery_split_point]
        validation_pids = shuffled_pids[discovery_split_point:]
        
        self.df_discovery = df[df['pid'].isin(discovery_pids)].copy()
        self.df_validation = df[df['pid'].isin(validation_pids)].copy()
        
        logger.info(f"Data split: {len(discovery_pids)} discovery subjects ({len(self.df_discovery)} rows), "
                   f"{len(validation_pids)} validation subjects ({len(self.df_validation)} rows)")
        
        return self.df_discovery, self.df_validation
    
    def create_data_matrix(self, df: pd.DataFrame, standardize: bool = False) -> np.ndarray:
        """Create clean numerical data matrix."""
        X = (df[self.feature_names]
             .apply(pd.to_numeric, errors="coerce")
             .dropna()
             .to_numpy())
        
        if standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            logger.info("Applied standardization to data matrix")
        
        logger.info(f"Created data matrix: {X.shape[0]} rows × {X.shape[1]} variables")
        return X
    
    def create_background_knowledge(self) -> BackgroundKnowledge:
        """Create background knowledge with domain-driven constraints."""
        logger.info("Creating background knowledge constraints...")
        
        bk = BackgroundKnowledge()
        nodes = [GraphNode(name) for name in self.feature_names]
        name_to_node = {n.get_name(): n for n in nodes}
        
        constraint_count = 0
        
        # Constraint 1: Forbid mean ↔ std connections for same sensor
        if self.config.forbid_sensor_correlations:
            sensor_features = [name for name in self.feature_names if name not in self.config.outcome_vars]
            processed_bases = set()
            
            for feature in sensor_features:
                if feature.endswith('_mean'):
                    base_name = feature.replace('_mean', '')
                    std_feature = f"{base_name}_std"
                    
                    if std_feature in sensor_features and base_name not in processed_bases:
                        bk.add_forbidden_by_node(name_to_node[feature], name_to_node[std_feature])
                        bk.add_forbidden_by_node(name_to_node[std_feature], name_to_node[feature])
                        processed_bases.add(base_name)
                        constraint_count += 2
                        logger.debug(f"Forbidden: {feature} ↔ {std_feature}")
            
            logger.info(f"Added {len(processed_bases)} forbidden sensor correlation pairs")
        
        # Constraint 2: Require sensor → outcome directions
        if self.config.require_sensor_to_outcome:
            sensor_features = [name for name in self.feature_names if name not in self.config.outcome_vars]
            
            for sensor_feature in sensor_features:
                for outcome_var in self.config.outcome_vars:
                    if outcome_var in self.feature_names:
                        bk.add_required_by_node(name_to_node[sensor_feature], name_to_node[outcome_var])
                        constraint_count += 1
                        logger.debug(f"Required: {sensor_feature} → {outcome_var}")
            
            logger.info(f"Added {len(sensor_features) * len([o for o in self.config.outcome_vars if o in self.feature_names])} required directions")
        
        logger.info(f"Total background knowledge constraints: {constraint_count}")
        return bk
    
    def run_pc_algorithm(self, X: np.ndarray, background_knowledge: Optional[BackgroundKnowledge] = None):
        """Run PC algorithm with optional background knowledge."""
        logger.info("Running PC algorithm...")
        
        # Get independence test function
        if self.config.independence_test == "fisherz":
            indep_test = fisherz
        else:
            raise ValueError(f"Independence test {self.config.independence_test} not implemented")
        
        try:
            cg = pc(
                data=X,
                alpha=self.config.alpha,
                indep_test=indep_test,
                stable=self.config.stable,
                uc_rule=self.config.uc_rule,
                background_knowledge=background_knowledge,
                node_names=self.feature_names,
                verbose=self.config.verbose
            )
            
            if cg is None or cg.G is None:
                logger.error("PC algorithm returned None result")
                return None
            
            logger.info("PC algorithm completed successfully")
            return cg
            
        except Exception as e:
            logger.error(f"PC algorithm failed: {str(e)}")
            return None
    
    def extract_edges(self, graph, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Extract edges from causal graph."""
        edges = []
        n_nodes = len(feature_names)
        adj_matrix = graph.graph
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                from_var = feature_names[i]
                to_var = feature_names[j]
                
                if adj_matrix[i, j] != 0 or adj_matrix[j, i] != 0:
                    if adj_matrix[i, j] != 0 and adj_matrix[j, i] != 0:
                        # Undirected edge
                        edges.append({
                            "from": from_var,
                            "to": to_var,
                            "type": "undirected",
                            "strength": abs(adj_matrix[i, j])
                        })
                    elif adj_matrix[i, j] != 0:
                        # Directed edge i -> j
                        edges.append({
                            "from": from_var,
                            "to": to_var,
                            "type": "directed",
                            "strength": abs(adj_matrix[i, j])
                        })
                    elif adj_matrix[j, i] != 0:
                        # Directed edge j -> i
                        edges.append({
                            "from": to_var,
                            "to": from_var,
                            "type": "directed",
                            "strength": abs(adj_matrix[j, i])
                        })
        
        return edges
    
    def save_graph(self, graph, filename: str, title: str = "Causal Graph") -> None:
        """Save causal graph as image."""
        if not self.config.save_graphs:
            return
        
        try:
            pyd = GraphUtils.to_pydot(graph, labels=self.feature_names)
            
            # Save to file
            filepath = self.output_dir / filename
            if self.config.graph_format.lower() == "png":
                pyd.write_png(str(filepath))
            elif self.config.graph_format.lower() == "svg":
                pyd.write_svg(str(filepath))
            else:
                pyd.write_png(str(filepath.with_suffix('.png')))
            
            logger.info(f"Saved graph to {filepath}")
            
            # Display if requested
            if self.config.show_graphs:
                img_data = pyd.create_png()
                img = mpimg.imread(io.BytesIO(img_data), format="png")
                
                plt.figure(figsize=self.config.figure_size)
                plt.title(title, fontsize=16, fontweight='bold')
                plt.axis("off")
                plt.imshow(img)
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            logger.error(f"Failed to save graph: {str(e)}")
    
    def bootstrap_analysis(self, X: np.ndarray) -> Dict[str, Any]:
        """Run bootstrap analysis for edge stability."""
        logger.info(f"Starting bootstrap analysis with {self.config.n_bootstrap} iterations...")
        
        edge_counts = defaultdict(int)
        edge_directions = defaultdict(list)
        key_edge_counts = {str(edge): 0 for edge in self.config.key_edges}
        successful_iterations = 0
        
        for i in tqdm(range(self.config.n_bootstrap), desc="Bootstrap iterations"):
            try:
                # Sample data
                n_sample = int(X.shape[0] * self.config.bootstrap_sample_frac)
                sample_indices = random.sample(range(X.shape[0]), n_sample)
                X_bootstrap = X[sample_indices, :]
                
                # Create background knowledge for this iteration
                bk_bootstrap = self.create_background_knowledge()
                
                # Run PC algorithm silently
                with open(os.devnull, 'w') as devnull:
                    with redirect_stdout(devnull), redirect_stderr(devnull):
                        cg_bootstrap = self.run_pc_algorithm(X_bootstrap, bk_bootstrap)
                
                if cg_bootstrap is None or cg_bootstrap.G is None:
                    continue
                
                # Extract edges
                edges = self.extract_edges(cg_bootstrap.G, self.feature_names)
                edges_this_iteration = set()
                
                for edge in edges:
                    if edge["type"] == "undirected":
                        edge_key = tuple(sorted([edge["from"], edge["to"]]))
                    else:
                        edge_key = (edge["from"], edge["to"])
                    
                    edge_counts[edge_key] += 1
                    edge_directions[edge_key].append(edge["type"])
                    edges_this_iteration.add(edge_key)
                
                # Count key edges
                for key_edge in self.config.key_edges:
                    key_edge_str = str(key_edge)
                    if (key_edge in edges_this_iteration or 
                        tuple(sorted(key_edge)) in edges_this_iteration or
                        (key_edge[1], key_edge[0]) in edges_this_iteration):
                        key_edge_counts[key_edge_str] += 1
                
                successful_iterations += 1
                
            except Exception as e:
                logger.debug(f"Bootstrap iteration {i} failed: {str(e)}")
                continue
        
        logger.info(f"Bootstrap analysis completed: {successful_iterations}/{self.config.n_bootstrap} successful iterations")
        
        return {
            "edge_counts": dict(edge_counts),
            "edge_directions": dict(edge_directions),
            "key_edge_counts": key_edge_counts,
            "successful_iterations": successful_iterations,
            "total_iterations": self.config.n_bootstrap
        }
    
    def analyze_stability(self, bootstrap_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze edge stability from bootstrap results."""
        successful_iterations = bootstrap_results["successful_iterations"]
        key_edge_counts = bootstrap_results["key_edge_counts"]
        
        stability_analysis = {}
        
        for edge_str, count in key_edge_counts.items():
            stability = count / successful_iterations if successful_iterations > 0 else 0
            
            if stability >= self.config.stability_thresholds["strong"]:
                assessment = "STRONG"
            elif stability >= self.config.stability_thresholds["moderate"]:
                assessment = "MODERATE"
            elif stability >= self.config.stability_thresholds["weak"]:
                assessment = "WEAK"
            else:
                assessment = "UNSTABLE"
            
            stability_analysis[edge_str] = {
                "count": count,
                "total": successful_iterations,
                "stability": stability,
                "assessment": assessment
            }
        
        return stability_analysis
    
    def save_results(self, bootstrap_results: Dict[str, Any], stability_analysis: Dict[str, Any], 
                    edges: List[Dict[str, Any]]) -> None:
        """Save analysis results to files."""
        
        # Save bootstrap edge stability
        if self.config.save_bootstrap_results:
            edge_counts = bootstrap_results["edge_counts"]
            successful_iterations = bootstrap_results["successful_iterations"]
            
            bootstrap_df = pd.DataFrame([
                {
                    'from_var': edge[0] if isinstance(edge, tuple) and len(edge) >= 2 else str(edge),
                    'to_var': edge[1] if isinstance(edge, tuple) and len(edge) >= 2 else "",
                    'frequency': count / successful_iterations,
                    'count': count,
                    'total_iterations': successful_iterations
                }
                for edge, count in edge_counts.items() 
                if isinstance(edge, tuple) and len(edge) == 2
            ]).sort_values('frequency', ascending=False)
            
            bootstrap_file = self.output_dir / self.config.generate_filename("bootstrap_results")
            bootstrap_df.to_csv(bootstrap_file, index=False)
            logger.info(f"Saved bootstrap results to {bootstrap_file}")
        
        # Save edge summary
        if edges:
            edges_df = pd.DataFrame(edges)
            edge_file = self.output_dir / self.config.generate_filename("edge_summary")
            edges_df.to_csv(edge_file, index=False)
            logger.info(f"Saved edge summary to {edge_file}")
        
        # Save data summary
        summary_file = self.output_dir / self.config.generate_filename("data_summary")
        with open(summary_file, 'w') as f:
            f.write("Causal Discovery Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Configuration:\n")
            f.write(f"- Algorithm: PC with alpha={self.config.alpha}\n")
            f.write(f"- Bootstrap iterations: {self.config.n_bootstrap}\n")
            f.write(f"- Sample fraction: {self.config.bootstrap_sample_frac}\n")
            f.write(f"- Discovery split: {self.config.discovery_split}\n")
            f.write(f"- Features: {len(self.feature_names)}\n")
            f.write(f"- Background knowledge: {self.config.forbid_sensor_correlations and self.config.require_sensor_to_outcome}\n\n")
            
            f.write("Key Edge Stability:\n")
            for edge_str, analysis in stability_analysis.items():
                f.write(f"- {edge_str}: {analysis['count']}/{analysis['total']} "
                       f"({analysis['stability']:.1%}) - {analysis['assessment']}\n")
            
            f.write(f"\nFiles generated:\n")
            f.write(f"- Bootstrap results: {self.config.generate_filename('bootstrap_results')}\n")
            f.write(f"- Edge summary: {self.config.generate_filename('edge_summary')}\n")
            f.write(f"- PC graph: {self.config.generate_filename('pc_graph')}\n")
        
        logger.info(f"Saved analysis summary to {summary_file}")
    
    def run_full_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run complete causal discovery analysis."""
        logger.info("Starting full causal discovery analysis...")
        
        # Prepare data
        self.prepare_data(df)
        self.X_discovery = self.create_data_matrix(self.df_discovery)
        
        # Create background knowledge
        background_knowledge = self.create_background_knowledge()
        
        # Run PC algorithm on discovery set
        logger.info("Running PC algorithm on discovery set...")
        cg = self.run_pc_algorithm(self.X_discovery, background_knowledge)
        
        if cg is None:
            logger.error("PC algorithm failed, cannot continue analysis")
            return {}
        
        # Extract edges
        edges = self.extract_edges(cg.G, self.feature_names)
        logger.info(f"Discovered {len(edges)} edges")
        
        # Save main graph
        graph_filename = self.config.generate_filename("pc_graph")
        self.save_graph(cg.G, graph_filename, "PC Algorithm - Discovery Set")
        
        # Run bootstrap analysis
        bootstrap_results = self.bootstrap_analysis(self.X_discovery)
        
        # Analyze stability
        stability_analysis = self.analyze_stability(bootstrap_results)
        
        # Save all results
        self.save_results(bootstrap_results, stability_analysis, edges)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("CAUSAL DISCOVERY RESULTS")
        logger.info("=" * 60)
        
        for edge_str, analysis in stability_analysis.items():
            logger.info(f"{edge_str}: {analysis['count']}/{analysis['total']} "
                       f"({analysis['stability']:.1%}) - {analysis['assessment']}")
        
        return {
            "causal_graph": cg,
            "edges": edges,
            "bootstrap_results": bootstrap_results,
            "stability_analysis": stability_analysis,
            "feature_names": self.feature_names
        }


# Unit Testing
class TestCausalDiscoveryAnalyzer(unittest.TestCase):
    """Unit tests for CausalDiscoveryAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CausalDiscoveryConfig(
            n_bootstrap=5,  # Small number for testing
            max_samples=100,
            show_graphs=False,
            save_graphs=False
        )
    
    def test_config_validation(self):
        """Test configuration validation."""
        with self.assertRaises(ValueError):
            CausalDiscoveryConfig(discovery_split=0)
        
        with self.assertRaises(ValueError):
            CausalDiscoveryConfig(alpha=0)
        
        with self.assertRaises(ValueError):
            CausalDiscoveryConfig(n_bootstrap=0)
    
    def test_filename_generation(self):
        """Test filename generation."""
        filename = self.config.generate_filename("bootstrap_results")
        self.assertIn("alpha0.05", filename)
        self.assertIn("bs5", filename)
        self.assertIn("bootstrap_stability.csv", filename)
    
    def test_data_preparation(self):
        """Test data preparation with mock data."""
        # Create mock data
        np.random.seed(42)
        mock_data = pd.DataFrame({
            'pid': np.repeat(range(10), 5),
            'promis_dep_sum': np.random.normal(10, 2, 50),
            'promis_anx_sum': np.random.normal(8, 2, 50),
            'hr_average_mean': np.random.normal(70, 10, 50),
            'hr_average_std': np.random.normal(5, 2, 50),
            'total_sleep_mean': np.random.normal(8, 1, 50)  # Should be excluded
        })
        
        analyzer = CausalDiscoveryAnalyzer(self.config)
        df_discovery, df_validation = analyzer.prepare_data(mock_data)
        
        # Check splits
        self.assertGreater(len(df_discovery), 0)
        self.assertGreater(len(df_validation), 0)
        
        # Check feature selection
        self.assertIn('promis_dep_sum', analyzer.feature_names)
        self.assertIn('hr_average_mean', analyzer.feature_names)
        self.assertNotIn('total_sleep_mean', analyzer.feature_names)  # Should be excluded
    
    def test_background_knowledge_creation(self):
        """Test background knowledge creation."""
        analyzer = CausalDiscoveryAnalyzer(self.config)
        analyzer.feature_names = ['promis_dep_sum', 'hr_average_mean', 'hr_average_std']
        
        bk = analyzer.create_background_knowledge()
        self.assertIsNotNone(bk)
    
    def test_edge_extraction(self):
        """Test edge extraction from mock graph."""
        # This would require creating a mock causal graph object
        # For now, just test that the method exists and can be called
        analyzer = CausalDiscoveryAnalyzer(self.config)
        analyzer.feature_names = ['var1', 'var2']
        
        # Test with mock adjacency matrix
        class MockGraph:
            def __init__(self):
                self.graph = np.array([[0, 1], [0, 0]])  # var1 -> var2
        
        mock_graph = MockGraph()
        edges = analyzer.extract_edges(mock_graph, ['var1', 'var2'])
        
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['from'], 'var1')
        self.assertEqual(edges[0]['to'], 'var2')
        self.assertEqual(edges[0]['type'], 'directed')


def run_causal_discovery(df: pd.DataFrame, config: CausalDiscoveryConfig) -> Dict[str, Any]:
    """Convenience function to run causal discovery with given configuration."""
    analyzer = CausalDiscoveryAnalyzer(config)
    return analyzer.run_full_analysis(df)


# Example configurations
def get_example_configs() -> Dict[str, CausalDiscoveryConfig]:
    """Get example configurations for common use cases."""
    return {
        "standard": CausalDiscoveryConfig(
            alpha=0.05,
            n_bootstrap=100,
            bootstrap_sample_frac=0.6
        ),
        "strict": CausalDiscoveryConfig(
            alpha=0.01,  # More strict independence test
            n_bootstrap=200,
            bootstrap_sample_frac=0.5
        ),
        "exploratory": CausalDiscoveryConfig(
            alpha=0.1,  # More liberal
            n_bootstrap=50,
            forbid_sensor_correlations=False,  # Allow all connections
            require_sensor_to_outcome=False
        ),
        "quick_test": CausalDiscoveryConfig(
            alpha=0.05,
            n_bootstrap=20,
            max_samples=1000,
            bootstrap_sample_frac=0.8
        ),
        "detailed": CausalDiscoveryConfig(
            alpha=0.05,
            n_bootstrap=500,  # More bootstrap iterations
            bootstrap_sample_frac=0.5,
            stability_thresholds={
                "strong": 0.8,    # Higher thresholds
                "moderate": 0.6,
                "weak": 0.4
            }
        )
    }


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Example usage
    print("\nExample configurations:")
    configs = get_example_configs()
    
    for name, config in configs.items():
        print(f"\n{name.upper()}:")
        print(f"  Alpha: {config.alpha}")
        print(f"  Bootstrap iterations: {config.n_bootstrap}")
        print(f"  Sample fraction: {config.bootstrap_sample_frac}")
        print(f"  Background knowledge: {config.forbid_sensor_correlations and config.require_sensor_to_outcome}")
        print(f"  Output files: {config.generate_filename('bootstrap_results')}")