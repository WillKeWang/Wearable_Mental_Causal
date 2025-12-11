import argparse
import json
import pandas as pd
from collections import defaultdict, deque
from itertools import product
from typing import List, Tuple, Set, FrozenSet, Dict, Any
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# R and dagitty
import os
os.environ["RPY2_CFFI_MODE"] = "ABI"
import rpy2.robjects as ro

def ensure_dagitty_loaded():
    try:
        ro.r('suppressPackageStartupMessages(suppressWarnings(library(dagitty)))')
    except Exception as e:
        raise RuntimeError(
            "Failed to load 'dagitty' R package. "
            "Install in R: install.packages('dagitty')"
        ) from e


# ---------- 1) CSV -> edge list ----------

def extract_edge_list_from_csv(
    csv_path,
    sample_frac,
    existence_threshold = 50,
    direction_count_threshold = 10,
    orientation_threshold = 0.5,
):
    """
    Load causal discovery edge results from CSV and return a cleaned edge list.

    Args:
        csv_path: path to the causal discovery results CSV file.
        sample_frac: fraction value to filter (e.g., 0.8).
        existence_threshold: minimum total frequency for an edge to be considered.
        direction_count_threshold: minimum directional count required to call an edge directed.
        orientation_threshold: ratio threshold (e.g., 0.85) to determine direction robustness.

    Returns:
        A list of (a, b, edge_type) tuples such as ('X', 'Y', '->') or ('X', 'Y', '--').
    """
    # ---------- Step 1: Load and filter ----------
    df = pd.read_csv(csv_path)
    df_filtered = df.loc[df["sample_frac"] == sample_frac].reset_index(drop=True)

    # ---------- Step 2: Aggregate frequencies for each unordered pair ----------
    # Structure:
    # edge_stat[(left, right)] = { "Nab": X, "Nba": Y, "Nun": Z }
    edge_stat = {}

    for _, row in df_filtered.iterrows():
        a = row["from_var"]
        b = row["to_var"]
        typ = row["edge_type"]         # "directed", "undirected", "bidirected"
        freq = int(round(row["freq"] * 100))  # convert back to percentage count

        # Sort to get the unordered pair key
        key = tuple(sorted([a, b]))
        if key not in edge_stat:
            edge_stat[key] = {"Nab": 0, "Nba": 0, "Nun": 0}

        # Directed case: A → B or B → A
        if typ == "directed":
            # If (a,b) matches sorted key order, then it's A→B
            if (a, b) == key:
                edge_stat[key]["Nab"] += freq
            else:
                edge_stat[key]["Nba"] += freq

        # Undirected case: always added to Nun
        elif typ == "undirected":
            edge_stat[key]["Nun"] += freq

        # Optional: Treat bidirected same as undirected (or keep separate if needed)
        elif typ == "bidirected":
            edge_stat[key]["Nun"] += freq

    # ---------- Step 3: Build final edge list ----------
    edge_list = []

    for (left, right), stat in edge_stat.items():
        Nab = stat["Nab"]  # left → right
        Nba = stat["Nba"]  # right → left
        Nun = stat["Nun"]
        total = Nab + Nba + Nun

        # Filter out weak edges
        if total < existence_threshold:
            continue

        # Decide direction based on directional dominance
        if Nab > Nba:
            if Nab >= direction_count_threshold and Nab / (Nab + Nba) >= orientation_threshold:
                edge_list.append((left, right, "->"))
            else:
                edge_list.append((left, right, "--"))

        elif Nba > Nab:
            if Nba >= direction_count_threshold and Nba / (Nab + Nba) >= orientation_threshold:
                edge_list.append((right, left, "->"))
            else:
                edge_list.append((right, left, "--"))

        # If fully tied or no strong evidence
        else:
            edge_list.append((left, right, "--"))

    return edge_list

# ---------- 2) Graph parsing from edge list (optional) ----------
def parse_dagitty_graph(edge_list):
    """
    Given an edge_list already in the form of (src, dst, type),
    return a set of unique node names and the edge list itself.

    Example:
        input = [
            ('age_binned', 'promis_dep_sum_tm1', '->'),
            ('age_binned', 'promis_anx_sum_tm1', '->'),
            ('promis_anx_sum_tm1', 'promis_dep_sum_tm1', '--')
        ]

        output:
            nodes = {'age_binned', 'promis_dep_sum_tm1', 'promis_anx_sum_tm1'}
            edges = same as input
    """
    nodes = set()
    edges = []

    for u, v, t in edge_list:
        nodes.add(u)
        nodes.add(v)
        edges.append((u, v, t))

    return nodes, edges

# ---------- 3) Ancestor computation (ignoring undirected edges) ----------
def ancestors_directed_only(edges, query_nodes, include_self=True): 
    """
    Find all ancestor nodes of given query nodes using only directed edges.

    Args:
        edges: list of (src, dst, type) tuples, where type is '->' or '--'.
        query_nodes: list or set of nodes for which to find ancestors.
        include_self: if True, include the query nodes themselves
                      (consistent with DAGitty's behavior).

    Returns:
        A set of all ancestor nodes that have a directed path
        leading to any of the query nodes.
    """
    parents = defaultdict(set)
    for src, dst, typ in edges:
        if typ == '->':
            parents[dst].add(src)

    anc = set()
    # DAGitty's ancestors() by default includes the node itself.
    if include_self:
        anc.update(query_nodes)

    stack = list(query_nodes)
    seen = set(query_nodes)

    # Depth-first traversal following parent (reverse) links
    while stack:
        cur = stack.pop()
        for p in parents.get(cur, set()):
            if p not in seen:
                seen.add(p)
                anc.add(p)
                stack.append(p)
    return anc

# ---------- 4) Edge filtering ----------
def filter_edges_by_nodes(edges, keep_nodes):
    """
    Filter the given edge list to include only the edges whose
    source and destination nodes are both within a specified set.

    Args:
        edges: list of (src, dst, type) tuples representing edges in the graph.
        keep_nodes: set of node names to keep in the filtered graph.

    Returns:
        A filtered list of edges where both endpoints belong to keep_nodes.
    """
    return [(u, v, t) for (u, v, t) in edges if u in keep_nodes and v in keep_nodes]

# ---------- 5) Cycle and name checking ----------
def creates_cycle(adj, u, v):
    """
    Check whether adding a directed edge u -> v would create a cycle
    in the current directed adjacency structure.

    Args:
        adj: dict[str, set[str]], representing the current adjacency list
             of directed edges (e.g., adj[u] = {v1, v2, ...}).
        u: source node of the new edge.
        v: destination node of the new edge.

    Returns:
        True if adding u -> v would introduce a cycle (i.e., if a path
        already exists from v back to u), otherwise False.
    """
    if u == v:
        return True
    visited = set()
    dq = deque([v])
    while dq:
        x = dq.popleft()
        if x == u:
            return True
        for w in adj.get(x, ()):
            if w not in visited:
                visited.add(w)
                dq.append(w)
    return False


def is_temporal(name):
    return name.endswith('_t') or name.endswith('_tm1')

# ---------- 6) Orientation rules for undirected edges ----------
def orient_pdag_to_dag_all(edges):
    """
    Generate all possible DAG completions from a given PDAG
    by orienting each undirected edge ('--') in both possible directions.

    Each resulting DAG must satisfy two constraints:
        1. No directed cycles are introduced.
        2. The destination node (outcome) must be temporally valid,
           i.e., its name must end with '_t' or '_tm1'.

    Args:
        edges: list of (src, dst, type) tuples representing the PDAG.
               'type' can be '->' (directed) or '--' (undirected).

    Returns:
        A list of DAGs, where each DAG is represented as a list of (src, dst)
        directed edges that form a valid acyclic graph.

    Notes:
        - For each undirected edge (u, v), both orientations (u→v and v→u)
          are tested independently.
        - Only combinations that pass all constraints (no cycles, valid
          temporal naming) are included in the final result.
    """
    directed = [(u, v) for (u, v, t) in edges if t == '->']
    undirected = [(u, v) for (u, v, t) in edges if t == '--']

    all_dags = []
    n_undirected = len(undirected)
    total_combinations = 2 ** n_undirected

    for bits in tqdm(product([0, 1], repeat=n_undirected), total=total_combinations, desc=f"Orienting {n_undirected} undirected edges"):
        adj = defaultdict(set)
        # Add existing directed edges first
        for u, v in directed:
            adj[u].add(v)
        dir_edges = directed.copy()

        ok = True
        for (u, v), bit in zip(undirected, bits):
            # bit=0 → u→v, bit=1 → v→u
            src, dst = (u, v) if bit == 0 else (v, u)

            # Temporal constraint: only variables ending with '_t' or '_tm1'
            # can appear as outcome (destination) nodes.
            if not is_temporal(dst):
                ok = False
                break

            # Cycle check
            if creates_cycle(adj, src, dst):
                ok = False
                break

            # Add the directed edge
            adj[src].add(dst)
            dir_edges.append((src, dst))

        if ok:
            all_dags.append(dir_edges)

    print(f"[INFO] Generated {len(all_dags)} valid DAGs (out of {total_combinations} possible combinations).")

    return all_dags

# ---------- 7) Build DAGitty-compatible graph string ----------
def build_dagitty_dag_string(nodes, directed_edges):
    """
    Construct a DAGitty-compatible DAG string representation from
    a given set of nodes and directed edges.

    Args:
        nodes: set of node names (strings) to include in the DAG.
        directed_edges: list of (src, dst) tuples representing directed edges.

    Returns:
        A formatted string defining the DAG in DAGitty syntax, e.g.:

            dag {
                A
                B
                C
                A -> B ;
                B -> C ;
            }

    Notes:
        - Each node is printed on a new line (sorted alphabetically).
        - Each directed edge is represented as 'A -> B ;' on a single line.
        - The output can be directly parsed by the R `dagitty` package.
    """
    node_lines = "\n".join(sorted(nodes))
    edge_lines = " ".join(f"{u} -> {v} ;" for (u, v) in directed_edges)
    return f"dag {{\n{node_lines}\n{edge_lines}\n}}"

# ---------- 8) Compute adjustment sets for a single DAG ----------
def adjustment_sets_for_dag(nodes, directed_edges, exposure, outcome):
    """
    Compute minimal adjustment sets for a single DAG using the R 'dagitty' package.

    Args:
        nodes: set of node names (strings).
        directed_edges: list of (src, dst) tuples representing directed edges.
        exposure: name of the exposure (treatment) variable.
        outcome: name of the outcome variable.

    Returns:
        dag_str: DAGitty-compatible DAG string.
        py_sets: list of adjustment sets (each set represented as a list of variable names).
                 Example: [] represents an empty adjustment set {}.
        min_size: size (int) of the smallest minimal adjustment set,
                  or None if no adjustment set exists.

    Notes:
        - The function constructs the DAG in DAGitty syntax, passes it to R via `rpy2`,
          and retrieves the minimal adjustment sets using `adjustmentSets()`.
        - DAGitty may return multiple valid minimal adjustment sets; all are included.
    """
    dag_str = build_dagitty_dag_string(nodes, directed_edges)
    g = ro.r['dagitty'](dag_str)
    adj = ro.r['adjustmentSets'](g, exposure=exposure, outcome=outcome, type="minimal")

    # Each element in 'adj' is an R character vector (possibly empty)
    py_sets = [list(s) for s in adj]  # [] == {} → empty adjustment set
    min_size = min((len(s) for s in py_sets), default=None)
    return dag_str, py_sets, min_size

# ---------- 9) Evaluate all possible DAG completions ----------
def evaluate_all_dags(dag_edge_sets, nodes, exposure, outcome):
    """
    Compute adjustment sets across all DAG completions derived from a PDAG.

    Args:
        dag_edge_sets: list of DAGs, where each DAG is represented as a list of (src, dst) edges.
        nodes: set of all node names (strings).
        exposure: name of the exposure (treatment) variable.
        outcome: name of the outcome variable.

    Returns:
        A list of dictionaries, one per DAG, each containing:
            - "idx": DAG index number.
            - "edges": list of directed edges used in this DAG.
            - "dag_str": the DAGitty string representation.
            - "adj_sets": list of minimal adjustment sets (possibly empty).
            - "min_size": size of the smallest minimal adjustment set.
            - "has_sets": boolean indicating whether any adjustment set exists.

    Notes:
        - This function iterates through all possible DAG completions (e.g., generated by
          `orient_pdag_to_dag_all()`), computes adjustment sets for each, and aggregates results.
        - It is useful for exploring the impact of undirected edge orientations on causal adjustment.
    """
    results = []
    for i, dir_edges in enumerate(tqdm(dag_edge_sets, desc="Evaluating DAGs")):
        dag_str, sets_i, min_size = adjustment_sets_for_dag(nodes, dir_edges, exposure, outcome)
        results.append({
            "idx": i,
            "edges": dir_edges,
            "dag_str": dag_str,
            "adj_sets": sets_i,     # [[], ['a','b'], ...]
            "min_size": min_size,   # None (no set) or int
            "has_sets": len(sets_i) > 0
        })
    return results

# ---------- 10) Common adjustment sets with fallback ----------
def common_adj_sets_from_results(results, sub_edges):
    """
    Given a list of results (each containing {'adj_sets': [[...], [...], ...]}),
    return the adjustment sets that appear in all runs.
    The return value is a list of sorted lists representing common adjustment sets.
    """
    # Convert each run’s adj_sets into a set of frozensets for intersection
    per_run_sets = []
    for r in results:
        sets_i = r.get("adj_sets", [])
        set_of_fsets = {frozenset(s) for s in sets_i}
        per_run_sets.append(set_of_fsets)

    if not per_run_sets:
        return []

    # If any run has an empty set of adj_sets, the overall intersection is empty
    if any(len(s) == 0 for s in per_run_sets):
        inter = set()
    else:
        inter = set.intersection(*per_run_sets)

    # ---------- Case 1: Common adjustment sets exist ----------
    if inter:
        print(f"[INFO] Common adjustment sets found across {len(results)} DAGs.")
        out = [sorted(list(s)) for s in inter]
        out.sort(key=lambda xs: (len(xs), tuple(xs)))
        return out
    
    # ---------- Case 2: No intersection → fallback strategy ----------
    print("[INFO] No common adjustment sets found. Using fallback strategy based on directional matches.")

    if not sub_edges:
        print("[WARN] No sub_edges provided; cannot apply fallback.")
        return []
    
    must_have_pairs = [(a, b) for (a, b, t) in sub_edges if t == "--"]
    if not must_have_pairs:
        print("[WARN] No undirected edges ('--') found; fallback skipped.")
        return []
    
    best_r = None
    best_rank = None  # Ranking tuple: (hits, has_sets_bool, -min_size_or_big)

    for r in results:
        edges_set = set(r.get("edges", []))  # Set of directed edges {(u,v), ...}
        # Direction is enforced: (b,a) is not counted
        hits = sum(1 for p in must_have_pairs if p in edges_set)

        has_sets = bool(r.get("has_sets", False))
        ms = r.get("min_size", None)
        min_size_or_big = ms if ms is not None else 10**9

        # Higher hits → True has_sets → smaller min_size are preferred
        rank = (hits, has_sets, -min_size_or_big)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_r = r

    # If there is no DAG with matching direction pairs, return an empty list
    if best_r is None or best_rank[0] == 0:
        print("[WARN] No DAG matched any undirected edge orientations. Returning empty result.")
        return []
    
    print(f"[INFO] Fallback DAG selected (idx={best_r['idx']}, matched {best_rank[0]} undirected edges).")

    # Return the adj_sets of the best-matching DAG (sorted for consistency)
    chosen = [sorted(s) for s in best_r.get("adj_sets", [])]
    chosen.sort(key=lambda xs: (len(xs), tuple(xs)))
    return chosen

def find_directed_path(directed_edges, exposure, outcome, verbose=False):
    """
    Find one directed path exposure → ... → outcome.
    Returns the path as a list of nodes, e.g. ['A','B','C','D'].
    Returns None if no path exists.

    Supports edges in (src, dst) or (src, dst, type) form.
    """

    # Build adjacency for directed edges only
    adj = defaultdict(set)
    for e in directed_edges:
        if len(e) == 3:
            src, dst, typ = e
            if typ == '->':
                adj[src].add(dst)
        else:
            src, dst = e
            adj[src].add(dst)

    # BFS queue holds (node, path_so_far)
    dq = deque([(exposure, [exposure])])
    visited = set([exposure])

    while dq:
        cur, path = dq.popleft()

        if cur == outcome:
            if verbose:
                print(f"[INFO] Directed path found: {' -> '.join(path)}")
            return path

        for nxt in adj.get(cur, ()):
            if nxt not in visited:
                visited.add(nxt)
                dq.append((nxt, path + [nxt]))

    if verbose:
        print(f"[INFO] No directed path exists from {exposure} to {outcome}")
    return None

############################ ---------- main ---------- ############################
def main():
    """
    parser = argparse.ArgumentParser(description="Temporal causal adjustment set finder using DAGitty")
    parser.add_argument("--csv_path", required=True, help="Path to causal discovery CSV")
    parser.add_argument("--sample_frac", type=float, required=True, help="Sample fraction to filter (e.g., 0.8)")
    parser.add_argument("--exposure", required=True, help="Exposure variable name")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument("--existence_threshold", type=int, default=50)
    parser.add_argument("--direction_count_threshold", type=int, default=10)
    parser.add_argument("--orientation_threshold", type=float, default=0.85)
    parser.add_argument("--save_results", default=None, help="Optional path to save JSON results")
    args = parser.parse_args()
    """

    class Args:
        pass

    args = Args()
    args.csv_path = "causal_discovery_results/all_edges_sample_frac_with_vars.csv"
    args.sample_frac = 0.8
    args.exposure = "promis_dep_sum_t"
    args.outcome = "awake_std_t"
    args.existence_threshold = 50
    args.direction_count_threshold = 10
    args.orientation_threshold = 0.50
    args.save_results = None

    ensure_dagitty_loaded()

    edge_list = extract_edge_list_from_csv(
        csv_path=args.csv_path,
        sample_frac=args.sample_frac,
        existence_threshold=args.existence_threshold,
        direction_count_threshold=args.direction_count_threshold,
        orientation_threshold=args.orientation_threshold,
    )

    anc_nodes = ancestors_directed_only(edge_list, [args.exposure, args.outcome], include_self=True)
    sub_edges = filter_edges_by_nodes(edge_list, anc_nodes)

    path = find_directed_path(edge_list, args.exposure, args.outcome, verbose=True)

    if path is not None:
        dag_edge_sets = orient_pdag_to_dag_all(sub_edges)
        results = evaluate_all_dags(dag_edge_sets, anc_nodes, args.exposure, args.outcome)
        adj_sets_final = common_adj_sets_from_results(results, sub_edges)
    else:
        print("No path from treatment to outcome. Adjustment set cannot be identified.")

    print("\n=== FINAL ADJUSTMENT SETS ===")
    if adj_sets_final:
        for s in adj_sets_final:
            print(s)
    else:
        print("No common or fallback adjustment set found.")

    if args.save_results:
        out = {
            "adj_sets_final": adj_sets_final,
            "results_summary": [
                {
                    "idx": r["idx"],
                    "edges": r["edges"],
                    "min_size": r["min_size"],
                    "has_sets": r["has_sets"]
                }
                for r in results
            ]
        }
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nSaved results to {args.save_results}")

if __name__ == "__main__":

    main()
