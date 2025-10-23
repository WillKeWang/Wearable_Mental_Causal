#!/usr/bin/env python3
"""
Diagnostic script to check edge percentages and understand threshold behavior.
"""

import json
import sys

def analyze_edges(json_file):
    """Analyze edge percentages in a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    edges_dict = {}
    successful_iterations = data['successful_iterations']
    
    print(f"\nFile: {json_file}")
    print(f"Successful iterations: {successful_iterations}")
    print("="*70)
    
    # Calculate percentages
    for edge_str, count in data['edges'].items():
        percentage = int(round((count / successful_iterations) * 100))
        edges_dict[edge_str] = percentage
    
    # Sort by percentage
    sorted_edges = sorted(edges_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Count edges at different thresholds
    threshold_counts = {
        50: sum(1 for _, pct in sorted_edges if pct >= 50),
        60: sum(1 for _, pct in sorted_edges if pct >= 60),
        70: sum(1 for _, pct in sorted_edges if pct >= 70),
        80: sum(1 for _, pct in sorted_edges if pct >= 80),
        90: sum(1 for _, pct in sorted_edges if pct >= 90)
    }
    
    print("\nEdge counts by threshold:")
    for threshold, count in threshold_counts.items():
        print(f"  ≥{threshold}%: {count} edges")
    
    # Show distribution
    print("\nPercentage distribution:")
    bins = {
        "100%": sum(1 for _, pct in sorted_edges if pct == 100),
        "90-99%": sum(1 for _, pct in sorted_edges if 90 <= pct < 100),
        "80-89%": sum(1 for _, pct in sorted_edges if 80 <= pct < 90),
        "70-79%": sum(1 for _, pct in sorted_edges if 70 <= pct < 80),
        "60-69%": sum(1 for _, pct in sorted_edges if 60 <= pct < 70),
        "50-59%": sum(1 for _, pct in sorted_edges if 50 <= pct < 60),
    }
    for bin_range, count in bins.items():
        print(f"  {bin_range}: {count} edges")
    
    # Show edges in 70-80% range (the problem area)
    print("\nEdges in 70-79% range (should appear at 70% but not 80%):")
    edges_70_80 = [(edge, pct) for edge, pct in sorted_edges if 70 <= pct < 80]
    if edges_70_80:
        for edge, pct in edges_70_80:
            print(f"  {pct}%: {edge}")
    else:
        print("  ⚠️  NO EDGES IN THIS RANGE - This is why 70% and 80% look the same!")
    
    # Show top edges
    print("\nTop 20 edges by percentage:")
    for edge, pct in sorted_edges[:20]:
        print(f"  {pct}%: {edge}")
    
    return threshold_counts, edges_70_80


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_edges.py <json_file>")
        print("\nExample:")
        print("  python diagnose_edges.py data/edges/0w_to_4w_after_edges.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        counts, edges_70_80 = analyze_edges(json_file)
        
        print("\n" + "="*70)
        print("DIAGNOSIS:")
        print("="*70)
        
        if counts[70] == counts[80]:
            print("⚠️  70% and 80% thresholds have SAME number of edges!")
            if not edges_70_80:
                print("   → Reason: NO edges have percentages between 70-79%")
                print("   → All edges are either <70% or ≥80%")
                print("   → This is why the plots look identical!")
            else:
                print("   → But there ARE edges in 70-79% range...")
                print("   → Something might be wrong with filtering logic")
        else:
            print("✓  70% and 80% thresholds have different numbers of edges")
            print(f"   → 70%: {counts[70]} edges")
            print(f"   → 80%: {counts[80]} edges")
            print(f"   → Difference: {counts[70] - counts[80]} edges")
        
    except FileNotFoundError:
        print(f"Error: File not found: {json_file}")
        print("\nMake sure you've run causal discovery first!")
    except Exception as e:
        print(f"Error: {e}")