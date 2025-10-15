import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np

# All edges ≥50% for BEFORE period (wearable → survey)
before_edges_full = {
    ('rem_std', 'promis_dep_sum'): 93,
    ('awake_std', 'promis_dep_sum'): 83,
    ('promis_dep_sum', 'promis_anx_sum'): 67,
    ('deep_std', 'promis_dep_sum'): 60,
    ('rmssd_std', 'hr_average_std'): 100,
    ('hr_lowest_mean', 'hr_average_std'): 100,
    ('hr_lowest_std', 'hr_average_std'): 100,
    ('hr_lowest_mean', 'breath_average_mean'): 100,
    ('temperature_deviation_mean', 'breath_average_mean'): 100,
    ('temperature_trend_deviation_std', 'breath_average_mean'): 100,
    ('temperature_trend_deviation_mean', 'temperature_deviation_std'): 100,
    ('temperature_trend_deviation_std', 'temperature_deviation_std'): 100,
    ('deep_mean', 'efficiency_mean'): 100,
    ('rem_mean', 'efficiency_mean'): 100,
    ('breath_v_average_mean', 'hr_average_mean'): 99,
    ('rem_mean', 'breath_average_mean'): 99,
    ('awake_mean', 'efficiency_mean'): 99,
    ('awake_std', 'efficiency_std'): 99,
    ('temperature_deviation_std', 'hr_average_std'): 98,
    ('awake_std', 'hr_average_std'): 98,
    ('hr_lowest_mean', 'rmssd_mean'): 97,
    ('temperature_trend_deviation_mean', 'breath_average_mean'): 97,
    ('deep_std', 'breath_average_std'): 96,
    ('awake_std', 'breath_average_std'): 96,
    ('rem_mean', 'breath_v_average_mean'): 96,
    ('breath_v_average_std', 'hr_average_std'): 94,
    ('deep_mean', 'breath_v_average_mean'): 94,
    ('temperature_max_std', 'efficiency_std'): 92,
    ('light_std', 'deep_std'): 92,
    ('onset_latency_std', 'hr_average_std'): 91,
    ('temperature_max_std', 'deep_std'): 91,
    ('rem_mean', 'rmssd_mean'): 90,
    ('awake_std', 'light_std'): 90,
    ('breath_v_average_std', 'breath_average_std'): 87,
    ('temperature_max_std', 'hr_lowest_std'): 86,
    ('temperature_max_std', 'light_std'): 85,
    ('temperature_max_mean', 'hr_lowest_std'): 84,
    ('breath_v_average_std', 'deep_std'): 84,
    ('hr_lowest_mean', 'hr_average_mean'): 82,
    ('rem_mean', 'temperature_max_mean'): 81,
    ('rmssd_mean', 'hr_average_mean'): 78,
    ('breath_average_std', 'hr_average_std'): 77,
    ('deep_mean', 'rmssd_mean'): 71,
    ('temperature_max_mean', 'efficiency_mean'): 70,
    ('rem_std', 'breath_average_std'): 69,
    ('rem_mean', 'light_mean'): 69,
    ('hr_lowest_std', 'efficiency_std'): 69,
    ('breath_average_mean', 'hr_average_mean'): 68,
    ('awake_mean', 'light_mean'): 68,
    ('light_mean', 'deep_mean'): 68,
    ('light_std', 'efficiency_std'): 62,
    ('light_mean', 'temperature_max_mean'): 61,
    ('temperature_deviation_std', 'breath_average_std'): 60,
    ('breath_v_average_std', 'rem_std'): 57,
    ('efficiency_std', 'deep_std'): 57,
    ('light_std', 'temperature_deviation_std'): 56,
    ('rem_std', 'awake_std'): 55,
    ('deep_std', 'rmssd_std'): 54,
    ('temperature_trend_deviation_std', 'rmssd_std'): 54,
    ('light_std', 'deep_mean'): 53,
    ('awake_std', 'temperature_trend_deviation_std'): 52,
    ('light_std', 'rem_std'): 52,
}

# All edges ≥50% for AFTER period (survey → wearable)
after_edges_full = {
    ('promis_anx_sum', 'promis_dep_sum'): 100,
    ('promis_dep_sum', 'deep_std'): 96,
    ('promis_dep_sum', 'rem_std'): 87,
    ('promis_anx_sum', 'temperature_trend_deviation_std'): 58,
    ('rmssd_std', 'hr_average_std'): 100,
    ('temperature_trend_deviation_mean', 'breath_average_mean'): 100,
    ('rem_mean', 'breath_average_mean'): 100,
    ('deep_mean', 'efficiency_mean'): 100,
    ('rem_mean', 'efficiency_mean'): 100,
    ('awake_mean', 'efficiency_mean'): 100,
    ('breath_v_average_std', 'breath_average_std'): 99,
    ('temperature_trend_deviation_mean', 'temperature_deviation_mean'): 99,
    ('temperature_trend_deviation_std', 'temperature_deviation_mean'): 99,
    ('temperature_trend_deviation_mean', 'temperature_deviation_std'): 98,
    ('temperature_trend_deviation_std', 'temperature_deviation_std'): 98,
    ('deep_std', 'rmssd_std'): 97,
    ('awake_std', 'breath_average_mean'): 97,
    ('awake_std', 'hr_average_std'): 96,
    ('awake_std', 'efficiency_std'): 95,
    ('awake_mean', 'light_mean'): 93,
    ('temperature_max_mean', 'efficiency_mean'): 92,
    ('hr_lowest_mean', 'rmssd_mean'): 90,
    ('deep_std', 'breath_average_std'): 90,
    ('hr_lowest_std', 'hr_average_std'): 88,
    ('deep_std', 'breath_average_mean'): 88,
    ('breath_v_average_std', 'hr_average_std'): 87,
    ('rem_mean', 'breath_average_std'): 87,
    ('hr_lowest_std', 'efficiency_std'): 87,
    ('hr_lowest_mean', 'breath_average_mean'): 87,
    ('deep_mean', 'breath_v_average_mean'): 86,
    ('rem_mean', 'breath_v_average_mean'): 86,
    ('temperature_deviation_std', 'breath_average_std'): 86,
    ('breath_v_average_std', 'deep_std'): 84,
    ('onset_latency_std', 'hr_average_std'): 83,
    ('hr_lowest_mean', 'hr_average_std'): 82,
    ('rem_std', 'breath_v_average_std'): 81,
    ('temperature_deviation_std', 'hr_average_std'): 80,
    ('temperature_max_mean', 'hr_lowest_std'): 79,
    ('light_std', 'deep_std'): 78,
    ('awake_std', 'light_std'): 77,
    ('awake_std', 'onset_latency_std'): 76,
    ('deep_mean', 'rmssd_mean'): 74,
    ('hr_lowest_mean', 'hr_average_mean'): 74,
    ('deep_std', 'hr_lowest_std'): 74,
    ('rem_mean', 'rmssd_mean'): 73,
    ('rmssd_mean', 'hr_average_mean'): 73,
    ('light_std', 'breath_average_std'): 72,
    ('rem_mean', 'light_mean'): 72,
    ('rem_std', 'light_std'): 72,
    ('breath_v_average_mean', 'rmssd_std'): 71,
    ('awake_std', 'breath_average_std'): 70,
    ('temperature_deviation_mean', 'hr_average_mean'): 68,
    ('light_std', 'temperature_deviation_std'): 66,
    ('breath_v_average_mean', 'breath_average_std'): 64,
    ('rem_std', 'awake_std'): 63,
    ('temperature_deviation_mean', 'breath_average_mean'): 62,
    ('light_mean', 'efficiency_mean'): 61,
    ('deep_mean', 'light_mean'): 60,
    ('onset_latency_std', 'rmssd_std'): 59,
    ('deep_std', 'temperature_max_std'): 59,
    ('awake_std', 'temperature_deviation_std'): 58,
    ('breath_average_std', 'hr_average_std'): 57,
    ('efficiency_std', 'deep_std'): 57,
    ('temperature_max_mean', 'light_mean'): 56,
    ('temperature_max_std', 'efficiency_std'): 56,
    ('light_mean', 'breath_average_mean'): 56,
    ('temperature_trend_deviation_std', 'rmssd_std'): 53,
    ('light_std', 'efficiency_std'): 53,
    ('awake_std', 'temperature_max_std'): 52,
    ('breath_average_mean', 'hr_average_mean'): 51,
    ('temperature_max_std', 'light_std'): 50,
    ('light_std', 'temperature_max_std'): 50,
}

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
    """
    CORRECT ORDER: First filter by threshold, THEN trace ancestors/descendants.
    """
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
    """
    Calculate the point where a line from center to target intersects the ellipse boundary.
    """
    cx, cy = center
    tx, ty = target_point
    
    # Direction vector
    dx = tx - cx
    dy = ty - cy
    
    # Angle to target
    angle = np.arctan2(dy, dx)
    
    # Semi-axes
    a = width / 2
    b = height / 2
    
    # Point on ellipse at this angle
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
    if num_wearables > 15:
        canvas_width = 36
        x_center = 18
    elif num_wearables > 10:
        canvas_width = 30
        x_center = 15
    else:
        canvas_width = 24
        x_center = 12
    
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
    usable_width = canvas_width - 2 * margin, canvas_width
    
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
                
                # Extra separation for breath variables
                if 'breath' in var and 'breath_average' in var:
                    if 'std' in var:
                        x_scatter += 0.5
                    elif 'mean' in var:
                        x_scatter -= 0.5
                
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
            shrinkA=0,  # No shrink needed - already at circumference
            shrinkB=0
        )
        ax.add_patch(arrow)
    
    # Dynamic axis limits with padding
    padding = 2
    ax.set_xlim(-padding, canvas_width + padding)
    ax.set_ylim(-padding, 20)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    return num_nodes

# Target variables
target_vars = {'promis_dep_sum', 'promis_anx_sum'}

# Create filtered edge sets
before_edges_50 = filter_edges_by_threshold_then_relevance(before_edges_full, target_vars, 50, 'before')
after_edges_50 = filter_edges_by_threshold_then_relevance(after_edges_full, target_vars, 50, 'after')

before_edges_60 = filter_edges_by_threshold_then_relevance(before_edges_full, target_vars, 60, 'before')
after_edges_60 = filter_edges_by_threshold_then_relevance(after_edges_full, target_vars, 60, 'after')

before_edges_80 = filter_edges_by_threshold_then_relevance(before_edges_full, target_vars, 80, 'before')
after_edges_80 = filter_edges_by_threshold_then_relevance(after_edges_full, target_vars, 80, 'after')

# Helper function to count nodes in edges
def count_nodes(edges):
    """Count unique nodes in edge dictionary."""
    nodes = set()
    for (source, target) in edges.keys():
        nodes.add(source)
        nodes.add(target)
    return len(nodes)

# Helper function to determine figure size
def get_figure_size(num_nodes):
    """Determine appropriate figure size based on number of nodes."""
    if num_nodes > 15:
        return (28, 24)
    elif num_nodes > 10:
        return (24, 22)
    else:
        return (20, 22)

# Create figures with DYNAMIC sizing based on node count
# 50% threshold
n_before_50 = count_nodes(before_edges_50)
n_after_50 = count_nodes(after_edges_50)
max_nodes_50 = max(n_before_50, n_after_50)
figsize_50 = get_figure_size(max_nodes_50)

fig1, axes1 = plt.subplots(2, 1, figsize=figsize_50)
fig1.suptitle('Temporal Causal Discovery: Relevant Pathways (Edges ≥50%)',
             fontsize=16, fontweight='bold', y=0.985)

create_clean_hierarchical_plot(before_edges_50,
                               'BEFORE (6-2 weeks before) | Wearable → Survey',
                               axes1[0])
create_clean_hierarchical_plot(after_edges_50,
                               'AFTER (1-5 weeks after) | Survey → Wearable',
                               axes1[1])

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('causal_graph_50pct.png', dpi=300, bbox_inches='tight')

# 60% threshold
n_before_60 = count_nodes(before_edges_60)
n_after_60 = count_nodes(after_edges_60)
max_nodes_60 = max(n_before_60, n_after_60)
figsize_60 = get_figure_size(max_nodes_60)

fig2, axes2 = plt.subplots(2, 1, figsize=figsize_60)
fig2.suptitle('Temporal Causal Discovery: Relevant Pathways (Edges ≥60%)',
             fontsize=16, fontweight='bold', y=0.985)

create_clean_hierarchical_plot(before_edges_60,
                               'BEFORE (6-2 weeks before) | Wearable → Survey',
                               axes2[0])
create_clean_hierarchical_plot(after_edges_60,
                               'AFTER (1-5 weeks after) | Survey → Wearable',
                               axes2[1])

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('causal_graph_60pct.png', dpi=300, bbox_inches='tight')

# 80% threshold
n_before_80 = count_nodes(before_edges_80)
n_after_80 = count_nodes(after_edges_80)
max_nodes_80 = max(n_before_80, n_after_80)
figsize_80 = get_figure_size(max_nodes_80)

fig3, axes3 = plt.subplots(2, 1, figsize=figsize_80)
fig3.suptitle('Temporal Causal Discovery: Relevant Pathways (Edges ≥80%)',
             fontsize=16, fontweight='bold', y=0.985)

create_clean_hierarchical_plot(before_edges_80,
                               'BEFORE (6-2 weeks before) | Wearable → Survey',
                               axes3[0])
create_clean_hierarchical_plot(after_edges_80,
                               'AFTER (1-5 weeks after) | Survey → Wearable',
                               axes3[1])

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('causal_graph_80pct.png', dpi=300, bbox_inches='tight')

plt.show()

# Print statistics
print(f"\n50% threshold:")
print(f"  BEFORE: {len(before_edges_50)} edges, {n_before_50} nodes")
print(f"  AFTER: {len(after_edges_50)} edges, {n_after_50} nodes")
print(f"\n60% threshold:")
print(f"  BEFORE: {len(before_edges_60)} edges, {n_before_60} nodes")
print(f"  AFTER: {len(after_edges_60)} edges, {n_after_60} nodes")
print(f"\n80% threshold:")
print(f"  BEFORE: {len(before_edges_80)} edges, {n_before_80} nodes")
print(f"  AFTER: {len(after_edges_80)} edges, {n_after_80} nodes")