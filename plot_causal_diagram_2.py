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

def filter_edges_by_relevance(edges, target_vars, direction='before'):
    """Filter edges to include only those relevant to target variables."""
    if direction == 'before':
        # Find all ancestors of target variables
        relevant_nodes = find_ancestors(target_vars, edges)
    else:  # 'after'
        # Find all descendants of target variables
        relevant_nodes = find_descendants(target_vars, edges)
    
    # Filter edges to only include those between relevant nodes
    filtered_edges = {
        (source, target): pct 
        for (source, target), pct in edges.items()
        if source in relevant_nodes and target in relevant_nodes
    }
    
    return filtered_edges

def filter_edges_by_threshold(edges, threshold):
    """Filter edges by percentage threshold."""
    return {k: v for k, v in edges.items() if v >= threshold}

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
        width = max(2.8, max_line_len * 0.15 + 1.0)
        height = 0.9
    else:
        width = max(2.8, len(label) * 0.12 + 0.5)
        height = 0.6
    
    return width, height

def create_hierarchical_layout(edges):
    """Create a hierarchical layout with survey at top, grouped wearables below."""
    survey_vars = {'promis_dep_sum', 'promis_anx_sum'}
    
    all_nodes = set()
    for (source, target) in edges.keys():
        all_nodes.add(source)
        all_nodes.add(target)
    
    wearable_vars = all_nodes - survey_vars
    
    # Group wearables by category
    sleep_vars = [v for v in wearable_vars if any(x in v for x in ['deep', 'rem', 'light', 'awake', 'efficiency', 'onset'])]
    hr_vars = [v for v in wearable_vars if 'hr' in v or 'rmssd' in v]
    breath_vars = [v for v in wearable_vars if 'breath' in v]
    temp_vars = [v for v in wearable_vars if 'temperature' in v or 'temp' in v]
    
    positions = {}
    
    # Position survey variables at top center
    if 'promis_dep_sum' in all_nodes:
        positions['promis_dep_sum'] = (10, 18)
    if 'promis_anx_sum' in all_nodes:
        positions['promis_anx_sum'] = (10, 16)
    
    y_base = 12
    x_positions = {
        'sleep': (2, y_base),
        'hr': (9, y_base - 4),
        'breath': (17, y_base - 4),
        'temp': (25, y_base)
    }
    
    for i, var in enumerate(sorted(sleep_vars)):
        x_offset = (i % 3) * 4.5
        y_offset = (i // 3) * -2.5
        positions[var] = (x_positions['sleep'][0] + x_offset, x_positions['sleep'][1] + y_offset)
    
    for i, var in enumerate(sorted(hr_vars)):
        x_offset = (i % 2) * 4.5
        y_offset = (i // 2) * -2.5
        positions[var] = (x_positions['hr'][0] + x_offset, x_positions['hr'][1] + y_offset)
    
    for i, var in enumerate(sorted(breath_vars)):
        x_offset = (i % 2) * 4.5
        y_offset = (i // 2) * -2.5
        positions[var] = (x_positions['breath'][0] + x_offset, x_positions['breath'][1] + y_offset)
    
    for i, var in enumerate(sorted(temp_vars)):
        x_offset = (i % 2) * 4.5
        y_offset = (i // 2) * -2.5
        positions[var] = (x_positions['temp'][0] + x_offset, x_positions['temp'][1] + y_offset)
    
    return positions

def create_clean_hierarchical_plot(edges, title, ax):
    """Create hierarchical network visualization."""
    survey_vars = {'promis_dep_sum', 'promis_anx_sum'}
    
    pos = create_hierarchical_layout(edges)
    
    # Draw edges
    for (source, target), percentage in edges.items():
        if source not in pos or target not in pos:
            continue
        
        start_pos = pos[source]
        end_pos = pos[target]
        
        # Map 50%-100% to 0.2-1.0 opacity
        alpha = 0.2 + ((percentage - 50) / 50) * 0.8
        
        arrow = FancyArrowPatch(
            start_pos, end_pos,
            arrowstyle='-|>',
            color='black',
            linewidth=0.8,
            alpha=alpha,
            connectionstyle="arc3,rad=0.0",
            mutation_scale=20,
            zorder=1,
            shrinkA=10,
            shrinkB=10
        )
        ax.add_patch(arrow)
    
    # Draw nodes
    for node, node_pos in pos.items():
        is_survey = node in survey_vars
        width, height = get_ellipse_size(node)
        label = format_label(node)
        
        if is_survey:
            ellipse = Ellipse(node_pos, width, height,
                            facecolor='white',
                            edgecolor='black',
                            linewidth=2.5,
                            zorder=3)
            ax.add_patch(ellipse)
            fontsize = 9
            fontweight = 'bold'
        else:
            ellipse = Ellipse(node_pos, width, height,
                            facecolor='white',
                            edgecolor='#2C3E50',
                            linewidth=1,
                            zorder=3)
            ax.add_patch(ellipse)
            fontsize = 6.5
            fontweight = 'normal'
        
        ax.text(node_pos[0], node_pos[1], label,
               ha='center', va='center',
               fontsize=fontsize, fontweight=fontweight,
               zorder=4)
    
    ax.set_xlim(-2, 32)
    ax.set_ylim(-2, 20)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

# Target variables
target_vars = {'promis_dep_sum', 'promis_anx_sum'}

# Filter edges for relevance and threshold
before_edges_50 = filter_edges_by_relevance(before_edges_full, target_vars, 'before')
after_edges_50 = filter_edges_by_relevance(after_edges_full, target_vars, 'after')

before_edges_80 = filter_edges_by_threshold(before_edges_50, 80)
after_edges_80 = filter_edges_by_threshold(after_edges_50, 80)

# Create figures
fig1, axes1 = plt.subplots(1, 2, figsize=(26, 14))
fig1.suptitle('Temporal Causal Discovery: Relevant Pathways (Edges ≥50%)',
             fontsize=15, fontweight='bold', y=0.97)

create_clean_hierarchical_plot(before_edges_50,
                               'BEFORE (6-2 weeks before)\nWearable → Survey',
                               axes1[0])
create_clean_hierarchical_plot(after_edges_50,
                               'AFTER (1-5 weeks after)\nSurvey → Wearable',
                               axes1[1])

legend_elements = [
    mpatches.FancyBboxPatch((0, 0), 1, 0.4, boxstyle="round,pad=0.1", 
                            facecolor='white', edgecolor='black', linewidth=2.5, 
                            label='Survey Variables (bold border)'),
    mpatches.FancyBboxPatch((0, 0), 1, 0.4, boxstyle="round,pad=0.1",
                            facecolor='white', edgecolor='#2C3E50', linewidth=1, 
                            label='Wearable Variables'),
    mpatches.Patch(facecolor='black', alpha=0.95, label='90-100% frequency'),
    mpatches.Patch(facecolor='black', alpha=0.65, label='70-90% frequency'),
    mpatches.Patch(facecolor='black', alpha=0.35, label='50-70% frequency')
]
fig1.legend(handles=legend_elements, loc='lower center',
          ncol=5, fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))

plt.tight_layout()
plt.savefig('causal_graph_50pct.png', dpi=300, bbox_inches='tight')

# Create 80% threshold figure
fig2, axes2 = plt.subplots(1, 2, figsize=(26, 14))
fig2.suptitle('Temporal Causal Discovery: Relevant Pathways (Edges ≥80%)',
             fontsize=15, fontweight='bold', y=0.97)

create_clean_hierarchical_plot(before_edges_80,
                               'BEFORE (6-2 weeks before)\nWearable → Survey',
                               axes2[0])
create_clean_hierarchical_plot(after_edges_80,
                               'AFTER (1-5 weeks after)\nSurvey → Wearable',
                               axes2[1])

legend_elements_80 = [
    mpatches.FancyBboxPatch((0, 0), 1, 0.4, boxstyle="round,pad=0.1", 
                            facecolor='white', edgecolor='black', linewidth=2.5, 
                            label='Survey Variables (bold border)'),
    mpatches.FancyBboxPatch((0, 0), 1, 0.4, boxstyle="round,pad=0.1",
                            facecolor='white', edgecolor='#2C3E50', linewidth=1, 
                            label='Wearable Variables'),
    mpatches.Patch(facecolor='black', alpha=0.95, label='90-100% frequency'),
    mpatches.Patch(facecolor='black', alpha=0.75, label='80-90% frequency')
]
fig2.legend(handles=legend_elements_80, loc='lower center',
          ncol=4, fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))

plt.tight_layout()
plt.savefig('causal_graph_80pct.png', dpi=300, bbox_inches='tight')

plt.show()

# Print statistics
print(f"\n50% threshold:")
print(f"  BEFORE: {len(before_edges_50)} edges")
print(f"  AFTER: {len(after_edges_50)} edges")
print(f"\n80% threshold:")
print(f"  BEFORE: {len(before_edges_80)} edges")
print(f"  AFTER: {len(after_edges_80)} edges")



claude_comment = "I think this is better now. But do the following: remove the legends, and stack the subplots vertically instead of horizontally. Need your help with this though, because for the 80pct version, some arrows were covered and difficult to see. And more importantly, some variables are included that I don't know how they are causing depression. Like deep_std, it seems to be causing depression only when the threshold is 50%, but its parents are still showing up in the whole graph. You may need to filter for the threshold early and then do the ancester tracing. Also, add another threshold for 60%."