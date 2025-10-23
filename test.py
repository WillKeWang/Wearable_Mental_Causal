# Test 1: Can import functions
from plot_causal_diagram_new import load_edges_from_json, load_edges_for_analysis, create_visualizations
print("✓ All functions imported successfully")

# Test 2: Can load edges (assuming JSON files exist)
before, after = load_edges_for_analysis('6w_to_2w_before', '1w_to_5w_after', 'data/edges')
print(f"✓ Loaded {len(before)} before edges and {len(after)} after edges")

# Test 3: Can create visualizations
create_visualizations(before, after, thresholds=[50], output_dir='test_figures')
print("✓ Created test visualization")