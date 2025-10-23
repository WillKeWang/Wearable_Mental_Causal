import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np
import json
import os


# ============================================================================
# JSON LOADING FUNCTIONS
# ============================================================================

def load_edges_from_json(json_file):
    """Load edges from JSON file created by causal_discovery.py."""
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


def load_edges_for_analysis(before_dataset_name, after_dataset_name, edges_dir='data/edges'):
    """Load edges for a pair of before/after datasets."""
    before_file = os.path.join(edges_dir, f"{before_dataset_name}_edges.json")
    after_file = os.path.join(edges_dir, f"{after_dataset_name}_edges.json")
    
    if not os.path.exists(before_file):
        raise FileNotFoundError(f"Before edges file not found: {before_file}")
    if not os.path.exists(after_file):
        raise FileNotFoundError(f"After edges file not found: {after_file}")
    
    print(f"Loading BEFORE edges from: {before_file}")
    before_edges = load_edges_from_json(before_file)
    
    print(f"Loading AFTER edges from: {after_file}")
    after_edges = load_edges_from_json(after_file)
    
    print(f"✓ Loaded {len(before_edges)} edges from BEFORE period")
    print(f"✓ Loaded {len(after_edges)} edges from AFTER period")
    
    return before_edges, after_edges


# ============================================================================
# GRAPH FILTERING FUNCTIONS
# ============================================================================

def find_ancestors(target_nodes, edges):
    """Find all ancestors (nodes with paths to target) in directed graph."""
    ancestors = set(target_nodes)
    changed = True
    
    while changed:
        changed = False
        for (source, target) in edges.keys():
            if target in ancestors and source not in ancestors:
                ancestors.add(source)
                changed = True
    
    return ancestors


def find_descendants(source_nodes, edges):
    """Find all descendants (nodes reachable from source) in directed graph."""
    descendants = set(source_nodes)
    changed = True
    
    while changed:
        changed = False
        for (source, target) in edges.keys():
            if source in descendants and target not in descendants:
                descendants.add(target)
                changed = True
    
    return descendants


def filter_edges_by_threshold_then_relevance(edges, target_vars, threshold, direction='before'):
    """Filter by threshold, THEN trace ancestors/descendants."""
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


# ============================================================================
# VISUALIZATION HELPER FUNCTIONS
# ============================================================================

def format_label(var):
    """Format variable label, breaking long names into two lines."""
    if var == 'promis_dep_sum':
        return 'Depression'
    elif var == 'promis_anx_sum':
        return 'Anxiety'
    
    label = var.replace('_', ' ')
    
    if len(label) > 20:
        words = label.split()
        mid = len(words) // 2
        line1 = ' '.join(words[:mid])
        line2 = ' '.join(words[mid:])
        return f"{line1}\n{line2}"
    
    return label


def get_ellipse_size(var):
    """Get ellipse size based on variable name length."""
    label = format_label(var)
    
    if '\n' in label:
        max_line_len = max(len(line) for line in label.split('\n'))
        width = max(3.2, max_line_len * 0.18 + 1.2)
        height = 1.0
    else:
        width = max(3.2, len(label) * 0.14 + 0.7)
        height = 0.7
    
    return width, height


def get_ellipse_intersection(center, width, height, target_point):
    """Calculate the point where a line from center to target intersects the ellipse boundary."""
    cx, cy = center
    tx, ty = target_point
    
    dx = tx - cx
    dy = ty - cy
    angle = np.arctan2(dy, dx)
    
    a = width / 2
    b = height / 2
    
    x = cx + a * np.cos(angle)
    y = cy + b * np.sin(angle)
    
    return (x, y)


def create_hierarchical_layout(edges):
    """Create a hierarchical layout with DYNAMIC positioning based on present nodes."""
    survey_vars = {'promis_dep_sum', 'promis_anx_sum'}
    
    all_nodes = set()
    for (source, target) in edges.keys():
        all_nodes.add(source)
        all_nodes.add(target)
    
    wearable_vars = all_nodes - survey_vars
    num_wearables = len(wearable_vars)
    
    # Adjust spacing based on number of variables
    if num_wearables > 15:
        base_col_spacing = 4.5
        base_row_spacing = 3.5
        x_scatter_range = 0.6
        y_scatter_range = 1.5
    elif num_wearables > 10:
        base_col_spacing = 3.8
        base_row_spacing = 3.0
        x_scatter_range = 0.5
        y_scatter_range = 1.2
    else:
        base_col_spacing = 3.0
        base_row_spacing = 2.5
        x_scatter_range = 0.3
        y_scatter_range = 1.0
    
    # Group wearables by category
    sleep_vars = sorted([v for v in wearable_vars if any(x in v for x in ['deep', 'rem', 'light', 'awake', 'efficiency', 'onset'])])
    hr_vars = sorted([v for v in wearable_vars if 'hr' in v or 'rmssd' in v])
    breath_vars = sorted([v for v in wearable_vars if 'breath' in v])
    temp_vars = sorted([v for v in wearable_vars if 'temperature' in v or 'temp' in v])
    
    # Filter out empty categories
    categories = []
    if sleep_vars:
        categories.append(('sleep', sleep_vars))
    if hr_vars:
        categories.append(('hr', hr_vars))
    if breath_vars:
        categories.append(('breath', breath_vars))
    if temp_vars:
        categories.append(('temp', temp_vars))
    
    positions = {}
    
    # Position survey variables at top center
    survey_present = []
    if 'promis_dep_sum' in all_nodes:
        survey_present.append('promis_dep_sum')
    if 'promis_anx_sum' in all_nodes:
        survey_present.append('promis_anx_sum')
    
    # Adjust canvas size based on number of nodes
    # if num_wearables > 15:
    #     canvas_width = 36
    #     x_center = 18
    # elif num_wearables > 10:
    #     canvas_width = 30
    #     x_center = 15
    # else:
    #     canvas_width = 24
    #     x_center = 12
    
    # fix canvas width and center
    canvas_width = 30
    x_center = 15

    # Center survey variables
    if len(survey_present) == 2:
        positions['promis_dep_sum'] = (x_center, 17)
        positions['promis_anx_sum'] = (x_center, 15)
    elif len(survey_present) == 1:
        positions[survey_present[0]] = (x_center, 16)
    
    # If no wearable variables, return early
    if not categories:
        return positions, canvas_width
    
    # Calculate dynamic positioning for wearable categories
    num_categories = len(categories)
    
    # Distribute categories evenly across horizontal space with margins
    margin = canvas_width * 0.1
    usable_width = canvas_width - 2 * margin
    
    if num_categories == 1:
        x_starts = [x_center]
    elif num_categories == 2:
        x_starts = [margin + usable_width * 0.25, margin + usable_width * 0.75]
    elif num_categories == 3:
        x_starts = [margin + usable_width * 0.17, margin + usable_width * 0.5, margin + usable_width * 0.83]
    else:  # 4 categories
        x_starts = [margin + usable_width * 0.125, margin + usable_width * 0.375, 
                    margin + usable_width * 0.625, margin + usable_width * 0.875]
    
    y_start = 11
    
    # Use a hash-based approach for consistent but scattered positioning
    def get_scatter(var_name, i):
        """Generate consistent scatter based on variable name and index."""
        hash_val = hash(var_name + str(i))
        y_scatter = (hash_val % 100) / 50.0 - 1.0  # Range: -1.0 to 1.0
        x_scatter = ((hash_val // 100) % 60) / 100.0 - 0.3  # Range: -0.3 to 0.3
        return x_scatter * x_scatter_range / 0.3, y_scatter * y_scatter_range / 1.0
    
    # Position each category's variables
    for cat_idx, (cat_name, cat_vars) in enumerate(categories):
        x_base = x_starts[cat_idx]
        num_vars = len(cat_vars)
        
        # Special case: very few variables (2-3) - arrange horizontally
        if num_wearables <= 3:
            for i, var in enumerate(cat_vars):
                x = x_base + (i - (num_vars - 1) / 2) * base_col_spacing * 1.5
                y = y_start
                
                # Add scatter
                x_scatter, y_scatter = get_scatter(var, i)
                # Reduce vertical scatter for horizontal arrangement
                positions[var] = (x + x_scatter, y + y_scatter * 0.3)
        else:
            # Calculate grid dimensions for this category
            if num_vars <= 2:
                cols = 1
                col_spacing = 0
            elif num_vars <= 6:
                cols = 2
                col_spacing = base_col_spacing
            else:
                cols = 3
                col_spacing = base_col_spacing
            
            rows = (num_vars + cols - 1) // cols
            
            # Center the grid horizontally
            grid_width = (cols - 1) * col_spacing
            x_offset_start = -grid_width / 2
            
            # Adjust vertical spacing
            row_spacing = base_row_spacing
            
            for i, var in enumerate(cat_vars):
                col = i % cols
                row = i // cols
                
                x = x_base + x_offset_start + col * col_spacing
                y = y_start - row * row_spacing
                
                # Add significant scatter to avoid linear arrangements
                x_scatter, y_scatter = get_scatter(var, i)
                
                # Extra separation for breath variables - REDUCED to avoid overflow
                if 'breath' in var and 'breath_average' in var:
                    if 'std' in var:
                        x_scatter += 0.3
                    elif 'mean' in var:
                        x_scatter -= 0.3
                
                positions[var] = (x + x_scatter, y + y_scatter)
    
    return positions, canvas_width


def create_clean_hierarchical_plot(edges, title, ax):
    """Create hierarchical network visualization with arrows to circumference."""
    survey_vars = {'promis_dep_sum', 'promis_anx_sum'}
    
    # Get number of nodes for this graph
    all_nodes = set()
    for (source, target) in edges.keys():
        all_nodes.add(source)
        all_nodes.add(target)
    num_nodes = len(all_nodes)
    
    pos, canvas_width = create_hierarchical_layout(edges)
    
    # Store ellipse sizes for intersection calculation
    ellipse_sizes = {}
    for node in pos.keys():
        ellipse_sizes[node] = get_ellipse_size(node)
    
    # Draw nodes FIRST (behind arrows)
    for node, node_pos in pos.items():
        is_survey = node in survey_vars
        width, height = get_ellipse_size(node)
        label = format_label(node)
        
        if is_survey:
            ellipse = Ellipse(node_pos, width, height,
                            facecolor='white',
                            edgecolor='black',
                            linewidth=2.5,
                            zorder=2)
            ax.add_patch(ellipse)
            fontsize = 9
            fontweight = 'bold'
        else:
            ellipse = Ellipse(node_pos, width, height,
                            facecolor='white',
                            edgecolor='#2C3E50',
                            linewidth=1,
                            zorder=2)
            ax.add_patch(ellipse)
            fontsize = 6.5
            fontweight = 'normal'
        
        ax.text(node_pos[0], node_pos[1], label,
               ha='center', va='center',
               fontsize=fontsize, fontweight=fontweight,
               zorder=4)
    
    # Draw edges with circumference connections
    for (source, target), percentage in edges.items():
        if source not in pos or target not in pos:
            continue
        
        source_center = pos[source]
        target_center = pos[target]
        source_width, source_height = ellipse_sizes[source]
        target_width, target_height = ellipse_sizes[target]
        
        # Calculate intersection points on circumferences
        start_pos = get_ellipse_intersection(source_center, source_width, source_height, target_center)
        end_pos = get_ellipse_intersection(target_center, target_width, target_height, source_center)
        
        # Map to opacity
        if percentage >= 80:
            alpha = 0.3 + ((percentage - 80) / 20) * 0.7
        elif percentage >= 60:
            alpha = 0.25 + ((percentage - 60) / 20) * 0.5
        else:
            alpha = 0.2 + ((percentage - 50) / 10) * 0.3
        
        arrow = FancyArrowPatch(
            start_pos, end_pos,
            arrowstyle='-|>',
            color='black',
            linewidth=0.9,
            alpha=alpha,
            connectionstyle="arc3,rad=0.0",
            mutation_scale=20,
            zorder=3,
            shrinkA=0,
            shrinkB=0
        )
        ax.add_patch(arrow)
    
    # Dynamic axis limits with padding
    padding = 3
    ax.set_xlim(-padding, canvas_width + padding)
    ax.set_ylim(-padding, 20)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    return num_nodes


def count_nodes(edges):
    """Count unique nodes in edge dictionary."""
    nodes = set()
    for (source, target) in edges.keys():
        nodes.add(source)
        nodes.add(target)
    return len(nodes)


def get_figure_size(num_nodes):
    """Determine appropriate figure size based on number of nodes."""
    if num_nodes > 15:
        return (28, 24)
    elif num_nodes > 10:
        return (24, 22)
    else:
        return (20, 22)


# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def create_visualizations(before_edges_full, after_edges_full, 
                         before_dataset_name=None, after_dataset_name=None,
                         thresholds=[50, 60, 80], output_dir='.',
                         save_individual=True):
    """
    Create causal diagram visualizations at different thresholds.
    
    Args:
        before_edges_full: Dictionary of edges for BEFORE period
        after_edges_full: Dictionary of edges for AFTER period
        before_dataset_name: Name of before dataset (for filename)
        after_dataset_name: Name of after dataset (for filename)
        thresholds: List of percentage thresholds to visualize
        output_dir: Directory to save output figures
        save_individual: If True, also save individual plots for BEFORE and AFTER
    """
    os.makedirs(output_dir, exist_ok=True)
    
    target_vars = {'promis_dep_sum', 'promis_anx_sum'}
    
    # Create filtered edge sets for all thresholds
    for threshold in thresholds:
        before_edges = filter_edges_by_threshold_then_relevance(before_edges_full, target_vars, threshold, 'before')
        after_edges = filter_edges_by_threshold_then_relevance(after_edges_full, target_vars, threshold, 'after')
        
        # Count nodes and determine figure size
        n_before = count_nodes(before_edges)
        n_after = count_nodes(after_edges)
        max_nodes = max(n_before, n_after)
        figsize = get_figure_size(max_nodes)
        
        # ====================================================================
        # 1. SAVE COMBINED PLOT (both BEFORE and AFTER)
        # ====================================================================
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(f'Temporal Causal Discovery: Relevant Pathways (Edges ≥{threshold}%)',
                    fontsize=16, fontweight='bold', y=0.985)
        
        create_clean_hierarchical_plot(before_edges,
                                      'BEFORE (6-2 weeks before) | Wearable → Survey',
                                      axes[0])
        create_clean_hierarchical_plot(after_edges,
                                      'AFTER (1-5 weeks after) | Survey → Wearable',
                                      axes[1])
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Create filename with dataset names
        if before_dataset_name and after_dataset_name:
            output_file = os.path.join(output_dir, 
                f'causal_graph_{before_dataset_name}_vs_{after_dataset_name}_{threshold}pct.png')
        else:
            output_file = os.path.join(output_dir, f'causal_graph_{threshold}pct.png')
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n{threshold}% threshold:")
        print(f"  BEFORE: {len(before_edges)} edges, {n_before} nodes")
        print(f"  AFTER: {len(after_edges)} edges, {n_after} nodes")
        print(f"  Combined plot saved to: {output_file}")
        
        # ====================================================================
        # 2. SAVE INDIVIDUAL PLOTS (optional)
        # ====================================================================
        if save_individual:
            # Determine individual figure size (half of combined)
            individual_figsize = (figsize[0], figsize[1] / 2.2)
            
            # --- BEFORE plot ---
            fig_before, ax_before = plt.subplots(1, 1, figsize=individual_figsize)
            fig_before.suptitle(f'Causal Discovery: BEFORE Period (Edges ≥{threshold}%)',
                               fontsize=14, fontweight='bold', y=0.98)
            
            create_clean_hierarchical_plot(before_edges,
                                          'BEFORE (6-2 weeks before) | Wearable → Survey',
                                          ax_before)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if before_dataset_name:
                before_file = os.path.join(output_dir, 
                    f'causal_graph_{before_dataset_name}_{threshold}pct.png')
            else:
                before_file = os.path.join(output_dir, f'causal_graph_before_{threshold}pct.png')
            
            plt.savefig(before_file, dpi=300, bbox_inches='tight')
            plt.close(fig_before)
            print(f"  BEFORE individual plot: {before_file}")
            
            # --- AFTER plot ---
            fig_after, ax_after = plt.subplots(1, 1, figsize=individual_figsize)
            fig_after.suptitle(f'Causal Discovery: AFTER Period (Edges ≥{threshold}%)',
                              fontsize=14, fontweight='bold', y=0.98)
            
            create_clean_hierarchical_plot(after_edges,
                                          'AFTER (1-5 weeks after) | Survey → Wearable',
                                          ax_after)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if after_dataset_name:
                after_file = os.path.join(output_dir, 
                    f'causal_graph_{after_dataset_name}_{threshold}pct.png')
            else:
                after_file = os.path.join(output_dir, f'causal_graph_after_{threshold}pct.png')
            
            plt.savefig(after_file, dpi=300, bbox_inches='tight')
            plt.close(fig_after)
            print(f"  AFTER individual plot: {after_file}")


# ============================================================================
# STANDALONE EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    # This code only runs when the file is executed directly
    # Not when imported by run_causal_analysis_main.py
    
    print("="*70)
    print("STANDALONE TEST MODE")
    print("="*70)
    print("Loading edges from data/edges/ directory...")
    
    try:
        before_edges_full, after_edges_full = load_edges_for_analysis(
            before_dataset_name='6w_to_2w_before',
            after_dataset_name='1w_to_5w_after',
            edges_dir='data/edges'
        )
        
        print("\nCreating visualizations...")
        create_visualizations(
            before_edges_full=before_edges_full,
            after_edges_full=after_edges_full,
            before_dataset_name='6w_to_2w_before',
            after_dataset_name='1w_to_5w_after',
            thresholds=[50, 60, 70, 80],
            output_dir='test_figures'
        )
        
        print("\n" + "="*70)
        print("TEST COMPLETE!")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure:")
        print("  1. Causal discovery has been run first")
        print("  2. JSON files exist in data/edges/")
        print("  3. Or modify the paths in this __main__ block")



# # MODIFIED FOR POSITION FIXES - 2
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.patches import Ellipse, FancyArrowPatch
# import numpy as np
# import json
# import os


# # ============================================================================
# # JSON LOADING FUNCTIONS
# # ============================================================================

# def load_edges_from_json(json_file):
#     """Load edges from JSON file created by causal_discovery.py."""
#     with open(json_file, 'r') as f:
#         data = json.load(f)
    
#     edges_dict = {}
#     successful_iterations = data['successful_iterations']
    
#     for edge_str, count in data['edges'].items():
#         parts = edge_str.split(' -> ')
#         edge = (parts[0].strip(), parts[1].strip())
#         percentage = int(round((count / successful_iterations) * 100))
#         edges_dict[edge] = percentage
    
#     return edges_dict


# def load_edges_for_analysis(before_dataset_name, after_dataset_name, edges_dir='data/edges'):
#     """Load edges for a pair of before/after datasets."""
#     before_file = os.path.join(edges_dir, f"{before_dataset_name}_edges.json")
#     after_file = os.path.join(edges_dir, f"{after_dataset_name}_edges.json")
    
#     if not os.path.exists(before_file):
#         raise FileNotFoundError(f"Before edges file not found: {before_file}")
#     if not os.path.exists(after_file):
#         raise FileNotFoundError(f"After edges file not found: {after_file}")
    
#     print(f"Loading BEFORE edges from: {before_file}")
#     before_edges = load_edges_from_json(before_file)
    
#     print(f"Loading AFTER edges from: {after_file}")
#     after_edges = load_edges_from_json(after_file)
    
#     print(f"✓ Loaded {len(before_edges)} edges from BEFORE period")
#     print(f"✓ Loaded {len(after_edges)} edges from AFTER period")
    
#     return before_edges, after_edges


# # ============================================================================
# # GRAPH FILTERING FUNCTIONS
# # ============================================================================

# def find_ancestors(target_nodes, edges):
#     """Find all ancestors (nodes with paths to target) in directed graph."""
#     ancestors = set(target_nodes)
#     changed = True
    
#     while changed:
#         changed = False
#         for (source, target) in edges.keys():
#             if target in ancestors and source not in ancestors:
#                 ancestors.add(source)
#                 changed = True
    
#     return ancestors


# def find_descendants(source_nodes, edges):
#     """Find all descendants (nodes reachable from source) in directed graph."""
#     descendants = set(source_nodes)
#     changed = True
    
#     while changed:
#         changed = False
#         for (source, target) in edges.keys():
#             if source in descendants and target not in descendants:
#                 descendants.add(target)
#                 changed = True
    
#     return descendants


# def filter_edges_by_threshold_then_relevance(edges, target_vars, threshold, direction='before'):
#     """Filter by threshold, THEN trace ancestors/descendants."""
#     filtered_edges = {k: v for k, v in edges.items() if v >= threshold}
    
#     if direction == 'before':
#         relevant_nodes = find_ancestors(target_vars, filtered_edges)
#     else:
#         relevant_nodes = find_descendants(target_vars, filtered_edges)
    
#     final_edges = {
#         (source, target): pct 
#         for (source, target), pct in filtered_edges.items()
#         if source in relevant_nodes and target in relevant_nodes
#     }
    
#     return final_edges


# # ============================================================================
# # VISUALIZATION HELPER FUNCTIONS
# # ============================================================================

# def format_label(var):
#     """Format variable label, breaking long names into two lines."""
#     if var == 'promis_dep_sum':
#         return 'Depression'
#     elif var == 'promis_anx_sum':
#         return 'Anxiety'
    
#     label = var.replace('_', ' ')
    
#     if len(label) > 20:
#         words = label.split()
#         mid = len(words) // 2
#         line1 = ' '.join(words[:mid])
#         line2 = ' '.join(words[mid:])
#         return f"{line1}\n{line2}"
    
#     return label


# def get_ellipse_size(var):
#     """Get ellipse size based on variable name length."""
#     label = format_label(var)
    
#     if '\n' in label:
#         max_line_len = max(len(line) for line in label.split('\n'))
#         width = max(3.2, max_line_len * 0.18 + 1.2)
#         height = 1.0
#     else:
#         width = max(3.2, len(label) * 0.14 + 0.7)
#         height = 0.7
    
#     return width, height


# def get_ellipse_intersection(center, width, height, target_point):
#     """Calculate the point where a line from center to target intersects the ellipse boundary."""
#     cx, cy = center
#     tx, ty = target_point
    
#     dx = tx - cx
#     dy = ty - cy
#     angle = np.arctan2(dy, dx)
    
#     a = width / 2
#     b = height / 2
    
#     x = cx + a * np.cos(angle)
#     y = cy + b * np.sin(angle)
    
#     return (x, y)


# def create_hierarchical_layout(edges):
#     """Create a hierarchical layout with DYNAMIC positioning based on present nodes."""
#     survey_vars = {'promis_dep_sum', 'promis_anx_sum'}
    
#     all_nodes = set()
#     for (source, target) in edges.keys():
#         all_nodes.add(source)
#         all_nodes.add(target)
    
#     wearable_vars = all_nodes - survey_vars
#     num_wearables = len(wearable_vars)
    
#     # Fixed spacing for consistent layout
#     base_col_spacing = 3.5
#     base_row_spacing = 2.8
    
#     # Group wearables by category
#     sleep_vars = sorted([v for v in wearable_vars if any(x in v for x in ['deep', 'rem', 'light', 'awake', 'efficiency', 'onset'])])
#     hr_vars = sorted([v for v in wearable_vars if 'hr' in v or 'rmssd' in v])
#     breath_vars = sorted([v for v in wearable_vars if 'breath' in v])
#     temp_vars = sorted([v for v in wearable_vars if 'temperature' in v or 'temp' in v])
    
#     # Filter out empty categories
#     categories = []
#     if sleep_vars:
#         categories.append(('sleep', sleep_vars))
#     if hr_vars:
#         categories.append(('hr', hr_vars))
#     if breath_vars:
#         categories.append(('breath', breath_vars))
#     if temp_vars:
#         categories.append(('temp', temp_vars))
    
#     positions = {}
    
#     # Position survey variables at top center
#     survey_present = []
#     if 'promis_dep_sum' in all_nodes:
#         survey_present.append('promis_dep_sum')
#     if 'promis_anx_sum' in all_nodes:
#         survey_present.append('promis_anx_sum')
    
#     # Adjust canvas size based on number of nodes
#     # FIXED CANVAS SIZE - always 30 units wide
#     canvas_width = 30
#     x_center = 15
    
#     # FIXED POSITIONS for Depression and Anxiety - always at center
#     if len(survey_present) == 2:
#         positions['promis_dep_sum'] = (15, 18)   # Fixed at (15, 18)
#         positions['promis_anx_sum'] = (15, 16)   # Fixed at (15, 16)
#     elif len(survey_present) == 1:
#         positions[survey_present[0]] = (15, 17)  # Fixed at (15, 17)
    
#     # If no wearable variables, return early
#     if not categories:
#         return positions, canvas_width
    
#     # Calculate dynamic positioning for wearable categories
#     num_categories = len(categories)
    
#     # Distribute categories evenly across horizontal space with margins
#     margin = canvas_width * 0.1
#     usable_width = canvas_width - 2 * margin
    
#     if num_categories == 1:
#         x_starts = [x_center]
#     elif num_categories == 2:
#         x_starts = [margin + usable_width * 0.25, margin + usable_width * 0.75]
#     elif num_categories == 3:
#         x_starts = [margin + usable_width * 0.17, margin + usable_width * 0.5, margin + usable_width * 0.83]
#     else:  # 4 categories
#         x_starts = [margin + usable_width * 0.125, margin + usable_width * 0.375, 
#                     margin + usable_width * 0.625, margin + usable_width * 0.875]
    
#     y_start = 11
    
#     # FIXED POSITIONS: Use deterministic offset based on variable name
#     # This ensures positions stay constant across different thresholds
#     def get_fixed_offset(var_name, i):
#         """
#         Generate FIXED, deterministic offset based on variable name.
#         Uses alphabetical position and specific patterns for consistency.
#         """
#         # Create a deterministic seed from variable name
#         # Use sum of character codes for consistency
#         name_sum = sum(ord(c) for c in var_name)
        
#         # Map to small offset ranges to avoid canvas overflow
#         # Reduced from previous scatter ranges for safety
#         x_offset = ((name_sum % 31) / 100.0) - 0.15  # Range: -0.15 to 0.16
#         y_offset = ((name_sum % 41) / 50.0) - 0.40   # Range: -0.40 to 0.42
        
#         return x_offset, y_offset
    
#     # Position each category's variables
#     for cat_idx, (cat_name, cat_vars) in enumerate(categories):
#         x_base = x_starts[cat_idx]
#         num_vars = len(cat_vars)
        
#         # Special case: very few variables (2-3) - arrange horizontally
#         if num_wearables <= 3:
#             for i, var in enumerate(cat_vars):
#                 x = x_base + (i - (num_vars - 1) / 2) * base_col_spacing * 1.5
#                 y = y_start
                
#                 # Add fixed offset (much smaller for horizontal)
#                 x_offset, y_offset = get_fixed_offset(var, i)
#                 # Reduce vertical offset for horizontal arrangement
#                 positions[var] = (x + x_offset, y + y_offset * 0.3)
#         else:
#             # Calculate grid dimensions for this category
#             if num_vars <= 2:
#                 cols = 1
#                 col_spacing = 0
#             elif num_vars <= 6:
#                 cols = 2
#                 col_spacing = base_col_spacing
#             else:
#                 cols = 3
#                 col_spacing = base_col_spacing
            
#             rows = (num_vars + cols - 1) // cols
            
#             # Center the grid horizontally
#             grid_width = (cols - 1) * col_spacing
#             x_offset_start = -grid_width / 2
            
#             # Adjust vertical spacing
#             row_spacing = base_row_spacing
            
#             for i, var in enumerate(cat_vars):
#                 col = i % cols
#                 row = i // cols
                
#                 x = x_base + x_offset_start + col * col_spacing
#                 y = y_start - row * row_spacing
                
#                 # Add FIXED offset to avoid perfect grid alignment
#                 x_offset, y_offset = get_fixed_offset(var, i)
                
#                 # NO special separation for breath variables
#                 # Keep everything deterministic and bounded
                
#                 positions[var] = (x + x_offset, y + y_offset)
    
#     return positions, canvas_width


# def create_clean_hierarchical_plot(edges, title, ax):
#     """Create hierarchical network visualization with arrows to circumference."""
#     survey_vars = {'promis_dep_sum', 'promis_anx_sum'}
    
#     # Get number of nodes for this graph
#     all_nodes = set()
#     for (source, target) in edges.keys():
#         all_nodes.add(source)
#         all_nodes.add(target)
#     num_nodes = len(all_nodes)
    
#     pos, canvas_width = create_hierarchical_layout(edges)
    
#     # Store ellipse sizes for intersection calculation
#     ellipse_sizes = {}
#     for node in pos.keys():
#         ellipse_sizes[node] = get_ellipse_size(node)
    
#     # Draw nodes FIRST (behind arrows)
#     for node, node_pos in pos.items():
#         is_survey = node in survey_vars
#         width, height = get_ellipse_size(node)
#         label = format_label(node)
        
#         if is_survey:
#             ellipse = Ellipse(node_pos, width, height,
#                             facecolor='white',
#                             edgecolor='black',
#                             linewidth=2.5,
#                             zorder=2)
#             ax.add_patch(ellipse)
#             fontsize = 9
#             fontweight = 'bold'
#         else:
#             ellipse = Ellipse(node_pos, width, height,
#                             facecolor='white',
#                             edgecolor='#2C3E50',
#                             linewidth=1,
#                             zorder=2)
#             ax.add_patch(ellipse)
#             fontsize = 6.5
#             fontweight = 'normal'
        
#         ax.text(node_pos[0], node_pos[1], label,
#                ha='center', va='center',
#                fontsize=fontsize, fontweight=fontweight,
#                zorder=4)
    
#     # Draw edges with circumference connections
#     for (source, target), percentage in edges.items():
#         if source not in pos or target not in pos:
#             continue
        
#         source_center = pos[source]
#         target_center = pos[target]
#         source_width, source_height = ellipse_sizes[source]
#         target_width, target_height = ellipse_sizes[target]
        
#         # Calculate intersection points on circumferences
#         start_pos = get_ellipse_intersection(source_center, source_width, source_height, target_center)
#         end_pos = get_ellipse_intersection(target_center, target_width, target_height, source_center)
        
#         # Map to opacity
#         if percentage >= 80:
#             alpha = 0.3 + ((percentage - 80) / 20) * 0.7
#         elif percentage >= 60:
#             alpha = 0.25 + ((percentage - 60) / 20) * 0.5
#         else:
#             alpha = 0.2 + ((percentage - 50) / 10) * 0.3
        
#         arrow = FancyArrowPatch(
#             start_pos, end_pos,
#             arrowstyle='-|>',
#             color='black',
#             linewidth=0.9,
#             alpha=alpha,
#             connectionstyle="arc3,rad=0.0",
#             mutation_scale=20,
#             zorder=3,
#             shrinkA=0,
#             shrinkB=0
#         )
#         ax.add_patch(arrow)
    
#     # Dynamic axis limits with MORE padding for better visibility
#     padding = 4  # Increased from 3 to 4
#     ax.set_xlim(-padding, canvas_width + padding)
#     ax.set_ylim(-2, 22)  # Increased vertical space
#     ax.set_aspect('equal')
#     ax.axis('off')
#     ax.set_title(title, fontsize=13, fontweight='bold', pad=20)  # Increased from 12 to 13
    
#     return num_nodes


# def count_nodes(edges):
#     """Count unique nodes in edge dictionary."""
#     nodes = set()
#     for (source, target) in edges.keys():
#         nodes.add(source)
#         nodes.add(target)
#     return len(nodes)


# def get_figure_size(num_nodes):
#     """Determine appropriate figure size based on number of nodes."""
#     if num_nodes > 15:
#         return (28, 24)
#     elif num_nodes > 10:
#         return (24, 22)
#     else:
#         return (20, 22)


# # ============================================================================
# # MAIN VISUALIZATION FUNCTION
# # ============================================================================

# def create_visualizations(before_edges_full, after_edges_full, 
#                          before_dataset_name=None, after_dataset_name=None,
#                          thresholds=[50, 60, 70, 80], output_dir='.',
#                          save_individual=True):
#     """
#     Create causal diagram visualizations at different thresholds.
    
#     Args:
#         before_edges_full: Dictionary of edges for BEFORE period
#         after_edges_full: Dictionary of edges for AFTER period
#         before_dataset_name: Name of before dataset (for filename)
#         after_dataset_name: Name of after dataset (for filename)
#         thresholds: List of percentage thresholds to visualize
#         output_dir: Directory to save output figures
#         save_individual: If True, also save individual plots for BEFORE and AFTER
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     target_vars = {'promis_dep_sum', 'promis_anx_sum'}
    
#     # Create filtered edge sets for all thresholds
#     for threshold in thresholds:
#         before_edges = filter_edges_by_threshold_then_relevance(before_edges_full, target_vars, threshold, 'before')
#         after_edges = filter_edges_by_threshold_then_relevance(after_edges_full, target_vars, threshold, 'after')
        
#         # Count nodes and determine figure size
#         n_before = count_nodes(before_edges)
#         n_after = count_nodes(after_edges)
#         max_nodes = max(n_before, n_after)
#         figsize = get_figure_size(max_nodes)
        
#         # ====================================================================
#         # 1. SAVE COMBINED PLOT (both BEFORE and AFTER)
#         # ====================================================================
#         fig, axes = plt.subplots(2, 1, figsize=figsize)
#         fig.suptitle(f'Temporal Causal Discovery: Relevant Pathways (Edges ≥{threshold}%)',
#                     fontsize=18, fontweight='bold', y=0.985)  # Increased from 16 to 18
        
#         create_clean_hierarchical_plot(before_edges,
#                                       'BEFORE (6-2 weeks before) | Wearable → Survey',
#                                       axes[0])
#         create_clean_hierarchical_plot(after_edges,
#                                       'AFTER (1-5 weeks after) | Survey → Wearable',
#                                       axes[1])
        
#         plt.tight_layout(rect=[0, 0, 1, 0.98])
        
#         # Create filename with dataset names
#         if before_dataset_name and after_dataset_name:
#             output_file = os.path.join(output_dir, 
#                 f'causal_graph_{before_dataset_name}_vs_{after_dataset_name}_{threshold}pct.png')
#         else:
#             output_file = os.path.join(output_dir, f'causal_graph_{threshold}pct.png')
        
#         plt.savefig(output_file, dpi=300, bbox_inches='tight')
#         plt.close(fig)
        
#         print(f"\n{threshold}% threshold:")
#         print(f"  BEFORE: {len(before_edges)} edges, {n_before} nodes")
#         print(f"  AFTER: {len(after_edges)} edges, {n_after} nodes")
#         print(f"  Combined plot saved to: {output_file}")
        
#         # ====================================================================
#         # 2. SAVE INDIVIDUAL PLOTS (optional)
#         # ====================================================================
#         if save_individual:
#             # CRITICAL: Clear any matplotlib state from previous iterations
#             plt.clf()
#             plt.close('all')
            
#             # Determine individual figure size (half of combined)
#             individual_figsize = (figsize[0], figsize[1] / 2.2)
            
#             # --- BEFORE plot ---
#             fig_before, ax_before = plt.subplots(1, 1, figsize=individual_figsize)
#             fig_before.suptitle(f'Causal Discovery: BEFORE Period (Edges ≥{threshold}%)',
#                                fontsize=16, fontweight='bold', y=0.98)  # Increased from 14 to 16
            
#             create_clean_hierarchical_plot(before_edges,
#                                           'BEFORE (6-2 weeks before) | Wearable → Survey',
#                                           ax_before)
            
#             plt.tight_layout(rect=[0, 0, 1, 0.96])
            
#             if before_dataset_name:
#                 before_file = os.path.join(output_dir, 
#                     f'causal_graph_{before_dataset_name}_{threshold}pct.png')
#             else:
#                 before_file = os.path.join(output_dir, f'causal_graph_before_{threshold}pct.png')
            
#             plt.savefig(before_file, dpi=300, bbox_inches='tight')
#             plt.close(fig_before)
#             print(f"  BEFORE individual plot: {before_file}")
            
#             # --- AFTER plot ---
#             print(f"\n  DEBUG: About to create AFTER plot for threshold {threshold}%")
#             print(f"  DEBUG: after_edges contains {len(after_edges)} edges")
#             if len(after_edges) > 0:
#                 first_5_edges = list(after_edges.keys())[:5]
#                 print(f"  DEBUG: First 5 edges: {first_5_edges}")
            
#             fig_after, ax_after = plt.subplots(1, 1, figsize=individual_figsize)
#             fig_after.suptitle(f'Causal Discovery: AFTER Period (Edges ≥{threshold}%)',
#                               fontsize=16, fontweight='bold', y=0.98)  # Increased from 14 to 16
            
#             create_clean_hierarchical_plot(after_edges,
#                                           'AFTER (1-5 weeks after) | Survey → Wearable',
#                                           ax_after)
            
#             plt.tight_layout(rect=[0, 0, 1, 0.96])
            
#             if after_dataset_name:
#                 after_file = os.path.join(output_dir, 
#                     f'causal_graph_{after_dataset_name}_{threshold}pct.png')
#             else:
#                 after_file = os.path.join(output_dir, f'causal_graph_after_{threshold}pct.png')
            
#             plt.savefig(after_file, dpi=300, bbox_inches='tight')
#             plt.close(fig_after)
#             print(f"  AFTER individual plot: {after_file}")


# # ============================================================================
# # STANDALONE EXECUTION (for testing)
# # ============================================================================

# if __name__ == "__main__":
#     # This code only runs when the file is executed directly
#     # Not when imported by run_causal_analysis_main.py
    
#     print("="*70)
#     print("STANDALONE TEST MODE")
#     print("="*70)
#     print("Loading edges from data/edges/ directory...")
    
#     try:
#         before_edges_full, after_edges_full = load_edges_for_analysis(
#             before_dataset_name='6w_to_2w_before',
#             after_dataset_name='1w_to_5w_after',
#             edges_dir='data/edges'
#         )
        
#         print("\nCreating visualizations...")
#         create_visualizations(
#             before_edges_full=before_edges_full,
#             after_edges_full=after_edges_full,
#             # before_dataset_name='6w_to_2w_before',
#             # before_dataset_name='5w_to_1w_before',
#             before_dataset_name='4w_to_0w_before',
#             # after_dataset_name='1w_to_5w_after',
#             after_dataset_name='0w_to_4w_after',
#             thresholds=[50, 60, 70, 80],
#             output_dir='test_figures'
#         )
        
#         print("\n" + "="*70)
#         print("TEST COMPLETE!")
#         print("="*70)
        
#     except FileNotFoundError as e:
#         print(f"\n✗ Error: {e}")
#         print("\nPlease ensure:")
#         print("  1. Causal discovery has been run first")
#         print("  2. JSON files exist in data/edges/")
#         print("  3. Or modify the paths in this __main__ block")