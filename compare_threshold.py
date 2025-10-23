#!/usr/bin/env python3
"""
Compare edges between two thresholds to understand why plots might look the same.
"""

import json
import sys

def load_edges_from_json(json_file):
    """Load edges from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    edges_dict = {}
    successful_iterations = data['successful_iterations']
    
    for edge_str, count in data['edges'].items():
        parts = edge_str.split(' -> ')
        edge = (parts[0].strip(), parts[1].strip())
        percentage = int(round((count / successful_iterations) * 100))
        edges_dict[edge] = percentage
    
    return edges_dict

def filter_edges(edges, threshold):
    """Filter edges by threshold."""
    return {k: v for k, v in edges.items() if v >= threshold}

def find_ancestors(target_vars, edges):
    """Find all ancestors (nodes with paths to target) in directed graph."""
    ancestors = set(target_vars)
    changed = True
    
    while changed:
        changed = False
        for (source, target) in edges.keys():
            if target in ancestors and source not in ancestors:
                ancestors.add(source)
                changed = True
    
    return ancestors

def find_descendants(source_vars, edges):
    """Find all descendants (nodes reachable from source) in directed graph."""
    descendants = set(source_vars)
    changed = True
    
    while changed:
        changed = False
        for (source, target) in edges.keys():
            if source in descendants and target not in descendants:
                descendants.add(target)
                changed = True
    
    return descendants

def filter_relevant_edges(edges, target_vars, threshold, direction='before'):
    """Filter edges by threshold and relevance."""
    filtered_edges = {k: v for k, v in edges.items() if v >= threshold}
    
    if direction == 'before':
        relevant_nodes = find_ancestors(target_vars, filtered_edges)
    else:
        relevant_nodes = find_descendants(target_vars, filtered_edges)
    
    final_edges = {
        (source, target): pct 
        for (source, target), pct in filtered_edges.items()
        if source in relevant_nodes and target in relevant_nodes
    }
    
    return final_edges

def compare_thresholds(json_file, threshold1, threshold2, direction='after'):
    """Compare edges at two different thresholds."""
    print(f"\nAnalyzing: {json_file}")
    print(f"Direction: {direction}")
    print("="*70)
    
    # Load edges
    all_edges = load_edges_from_json(json_file)
    
    # Set target variables
    target_vars = {'promis_dep_sum', 'promis_anx_sum'}
    
    # Filter at both thresholds
    edges_t1 = filter_relevant_edges(all_edges, target_vars, threshold1, direction)
    edges_t2 = filter_relevant_edges(all_edges, target_vars, threshold2, direction)
    
    print(f"\nThreshold {threshold1}%: {len(edges_t1)} edges")
    print(f"Threshold {threshold2}%: {len(edges_t2)} edges")
    
    # Find differences
    only_in_t1 = set(edges_t1.keys()) - set(edges_t2.keys())
    only_in_t2 = set(edges_t2.keys()) - set(edges_t1.keys())
    in_both = set(edges_t1.keys()) & set(edges_t2.keys())
    
    print(f"\nIn both: {len(in_both)} edges")
    print(f"Only in {threshold1}%: {len(only_in_t1)} edges")
    print(f"Only in {threshold2}%: {len(only_in_t2)} edges")
    
    if len(edges_t1) == len(edges_t2) and len(only_in_t1) == 0:
        print("\n⚠️  WARNING: Both thresholds have IDENTICAL edges!")
        print("   → This is why the plots look the same")
        
        # Show why
        print("\n   Checking raw edge percentages in range...")
        edges_in_range = {k: v for k, v in all_edges.items() 
                         if threshold1 <= v < threshold2}
        
        if not edges_in_range:
            print(f"   → NO edges have percentages between {threshold1}% and {threshold2}%")
            print(f"   → All edges are either <{threshold1}% or ≥{threshold2}%")
        else:
            print(f"   → Found {len(edges_in_range)} edges with {threshold1}% ≤ pct < {threshold2}%:")
            for edge, pct in sorted(edges_in_range.items(), key=lambda x: x[1], reverse=True):
                print(f"      {pct}%: {edge[0]} -> {edge[1]}")
            
            print("\n   → But these edges are NOT connected to Depression/Anxiety!")
            print("   → So they get filtered out by relevance check")
    else:
        print(f"\n✓  Thresholds have different edges")
        
        if only_in_t1:
            print(f"\nEdges only in {threshold1}% (will disappear at {threshold2}%):")
            for edge in sorted(only_in_t1):
                pct = edges_t1[edge]
                print(f"  {pct}%: {edge[0]} -> {edge[1]}")
    
    return edges_t1, edges_t2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_thresholds.py <json_file> [threshold1] [threshold2] [direction]")
        print("\nExample:")
        print("  python compare_thresholds.py data/edges/0w_to_4w_after_edges.json 70 80 after")
        print("\nDefaults: threshold1=70, threshold2=80, direction=after")
        sys.exit(1)
    
    json_file = sys.argv[1]
    threshold1 = int(sys.argv[2]) if len(sys.argv) > 2 else 70
    threshold2 = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    direction = sys.argv[4] if len(sys.argv) > 4 else 'after'
    
    try:
        edges_t1, edges_t2 = compare_thresholds(json_file, threshold1, threshold2, direction)
        
        print("\n" + "="*70)
        print("CONCLUSION:")
        print("="*70)
        
        if len(edges_t1) == len(edges_t2):
            print("The plots look the same because:")
            print("  1. No edges have percentages between 70-79%")
            print("  2. OR all edges in that range are not connected to Depression/Anxiety")
            print("  3. So after filtering for relevance, both thresholds have identical edges")
            print("\nThis is NOT a bug - it's just how your data looks!")
        else:
            print(f"The plots should look different:")
            print(f"  - {threshold1}% has {len(edges_t1)} edges")
            print(f"  - {threshold2}% has {len(edges_t2)} edges")
            print(f"  - Difference: {abs(len(edges_t1) - len(edges_t2))} edges")
        
    except FileNotFoundError:
        print(f"Error: File not found: {json_file}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()