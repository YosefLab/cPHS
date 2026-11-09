#!/usr/bin/env python3
"""
NNI perturbation analysis v2 — edge-normalized.

Perturbation levels defined as fraction of internal edges (not fraction of n).
Saves tree newick strings for reproducibility.
Computes cPHS, parsimony, likelihood, RF, triplets.

Usage:
    python nni_analysis_v2.py --tumor 3435_NT_T3 --reps 20
    python nni_analysis_v2.py --tumor 3435_NT_T3 --reps 20 --start-rep 10
    python nni_analysis_v2.py --tumor ALL --reps 20
"""

import os, sys, argparse, pickle, warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import binom

warnings.filterwarnings('ignore')

import cassiopeia as cass
from cassiopeia.tools.branch_length_estimator.IIDExponentialMLE import IIDExponentialMLE

# ── Paths ──
TREES_DIR = '/home/projects/nyosef/yaaraneu/KPTracer-Data/trees'
MERGED_CSV = '/home/projects/nyosef/yaaraneu/KP_results_20260311_022408/merged_all_tumors.csv'
OUTPUT_DIR = '/home/projects/nyosef/yaaraneu/nni_analysis_v2'

STOCHASTIC_MISSING_PROB = 0.10

# Perturbation levels as fraction of internal edges
EDGE_FRACTIONS = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00, 1.50, 2.00]

# Tumors to skip (cPHS at numerical floor)
SKIP_TUMORS = {'3434_NT_T2', '3513_NT_T2', '3519_Lkb1_All', '3726_NT_T2', '3730_NT_T1'}

EXCLUDED = {'3430_NT_T1', '3434_NT_T1', '3777_Apc_T5'}
Q_THRESHOLD = 0.35
K_FLAG_CUTOFF = 15


# ══════════════════════════════════════════════════════════════════════
# Data loading (matching kp_phs_all.py exactly)
# ══════════════════════════════════════════════════════════════════════

def load_tumor_data(tumor):
    cm_path = os.path.join(TREES_DIR, f'{tumor}_character_matrix.txt')
    priors_path = os.path.join(TREES_DIR, f'{tumor}_priors.pkl')
    cm_raw = pd.read_csv(cm_path, sep='\t', index_col=0)
    cm = cm_raw.apply(pd.to_numeric, errors='coerce').fillna(-1).astype(int)
    with open(priors_path, 'rb') as f:
        priors_raw = pickle.load(f)
    priors = {si: {int(k): v for k, v in sp.items()}
              for si, sp in priors_raw.items()}
    return cm, priors


def load_precomputed_tree(tumor, cm, priors):
    nwk_path = os.path.join(TREES_DIR, f'{tumor}_tree.nwk')
    with open(nwk_path) as f:
        nwk = f.read().strip()
    tree = cass.data.CassiopeiaTree(
        character_matrix=cm, tree=nwk, priors=priors)
    return postprocess(tree)


def postprocess(tree):
    tree.collapse_unifurcations()
    tree.reconstruct_ancestral_characters()
    tree.set_character_states(tree.root, [0] * tree.n_character)
    tree.scale_to_unit_length()
    IIDExponentialMLE().estimate_branch_lengths(tree)
    return tree


def reconstruct(cm, priors, method):
    tree = cass.data.CassiopeiaTree(character_matrix=cm.copy(), priors=priors)
    if method == 'cassiopeia':
        solver = cass.solver.VanillaGreedySolver()
    elif method == 'smj':
        solver = cass.solver.SharedMutationJoiningSolver()
    else:
        raise ValueError(f"Unknown method: {method}")
    solver.solve(tree)
    return postprocess(tree)


# ══════════════════════════════════════════════════════════════════════
# Tree structure analysis
# ══════════════════════════════════════════════════════════════════════

def count_internal_edges(tree):
    """Count internal edges (valid NNI targets)."""
    G = tree.get_tree_topology()
    n_edges = 0
    for u, v in G.edges():
        ch_u = list(G.successors(u))
        ch_v = list(G.successors(v))
        if len(ch_u) >= 2 and len(ch_v) >= 1 and v in ch_u:
            n_edges += 1
    return n_edges


# ══════════════════════════════════════════════════════════════════════
# NNI perturbation (from kp_phs_all.py)
# ══════════════════════════════════════════════════════════════════════

def perturb_topology_nni(tree, n_moves, seed=42):
    rng = np.random.RandomState(seed)
    perturbed = deepcopy(tree)
    G = perturbed.get_tree_topology()
    for _ in range(n_moves):
        internal_edges = []
        for u, v in G.edges():
            ch_u = list(G.successors(u))
            ch_v = list(G.successors(v))
            if len(ch_u) >= 2 and len(ch_v) >= 1 and v in ch_u:
                internal_edges.append((u, v))
        if not internal_edges:
            break
        parent, child = internal_edges[rng.randint(len(internal_edges))]
        siblings = [s for s in G.successors(parent) if s != child]
        grandchildren = list(G.successors(child))
        if not siblings or not grandchildren:
            continue
        sib = siblings[rng.randint(len(siblings))]
        gc = grandchildren[rng.randint(len(grandchildren))]
        G.remove_edge(parent, sib)
        G.remove_edge(child, gc)
        G.add_edge(parent, gc)
        G.add_edge(child, sib)

    def _to_newick(G, node):
        children = list(G.successors(node))
        if not children:
            return str(node)
        return "(" + ",".join(_to_newick(G, c) for c in children) + ")"

    new_nwk = _to_newick(G, perturbed.root) + ";"
    new_tree = cass.data.CassiopeiaTree(
        character_matrix=tree.character_matrix.copy(),
        tree=new_nwk, priors=tree.priors)
    return postprocess(new_tree), new_nwk


def tree_to_newick(tree):
    """Extract newick from a CassiopeiaTree."""
    G = tree.get_tree_topology()
    def _to_newick(G, node):
        children = list(G.successors(node))
        if not children:
            return str(node)
        return "(" + ",".join(_to_newick(G, c) for c in children) + ")"
    return _to_newick(G, tree.root) + ";"


# ══════════════════════════════════════════════════════════════════════
# Metric computation
# ══════════════════════════════════════════════════════════════════════

def get_lam_and_q(tree):
    cm = tree.character_matrix.to_numpy()
    proportion_mutated = np.sum(cm > 0) / np.sum(cm >= 0)
    lam = -np.log(1.0 - proportion_mutated)
    q_per_site = []
    for site_idx in tree.priors:
        vals = np.array(list(tree.priors[site_idx].values()))
        vals = vals / vals.sum()
        q_per_site.append(np.sum(vals ** 2))
    q = np.mean(q_per_site)
    return lam, q


def calc_phs_and_lca_times(tree):
    cm = tree.character_matrix
    n, k = cm.shape
    res       = pd.DataFrame(np.zeros((n, n)) - 1, index=cm.index, columns=cm.index)
    lca_times = pd.DataFrame(np.zeros((n, n)) - 1, index=cm.index, columns=cm.index)
    for lca in tree.internal_nodes:
        lca_states = tree.get_character_states(lca)
        lca_height = tree.get_time(lca)
        children = tree.children(lca)
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                for leaf_1 in tree.leaves_in_subtree(children[i]):
                    s1 = tree.get_character_states(leaf_1)
                    for leaf_2 in tree.leaves_in_subtree(children[j]):
                        s2 = tree.get_character_states(leaf_2)
                        phs = sum(1 for c in range(k)
                                  if lca_states[c] == 0 and s1[c] > 0 and s1[c] == s2[c])
                        res.loc[leaf_1, leaf_2]       = phs
                        res.loc[leaf_2, leaf_1]       = phs
                        lca_times.loc[leaf_1, leaf_2] = lca_height
                        lca_times.loc[leaf_2, leaf_1] = lca_height
    for leaf in tree.leaves:
        res.loc[leaf, leaf] = 0
        lca_times.loc[leaf, leaf] = 0
    return res, lca_times


def calc_cPHS(tree, lam, q):
    phs, lca_times = calc_phs_and_lca_times(tree)
    n, k = tree.character_matrix.shape
    alpha = np.exp(-lam * lca_times)
    beta  = 1 - np.exp(-lam * (1 - lca_times))
    prob  = alpha * beta**2 * q
    prob[lca_times.to_numpy() == 1] = 1
    cdf = binom.cdf(phs - 1, k, prob)
    pvalues_matrix = 1 - cdf
    off_diag = ~np.eye(n, dtype=bool)
    pvalues = np.array(pvalues_matrix[off_diag])
    pvalues[pvalues == 0] = np.finfo(float).eps
    N = len(pvalues)
    pvalues_sorted = np.sort(pvalues)
    adjusted = pvalues_sorted * N / np.arange(1, N + 1)
    return float(min(adjusted))


def safe_likelihood(tree):
    tree.parameters['stochastic_missing_probability'] = STOCHASTIC_MISSING_PROB
    tree.parameters.pop('heritable_missing_rate', None)
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            lik = cass.tools.calculate_likelihood_continuous(tree)
        if np.isinf(lik):
            return np.nan
        return lik
    except Exception:
        try:
            tree.parameters['stochastic_missing_probability'] = 0.0
            with np.errstate(divide='ignore', invalid='ignore'):
                lik = cass.tools.calculate_likelihood_continuous(tree)
            if np.isinf(lik):
                return np.nan
            return lik
        except Exception:
            return np.nan


def compute_metrics(tree, base_tree=None):
    lam, q = get_lam_and_q(tree)

    try:
        cphs = calc_cPHS(tree, lam, q)
    except Exception as e:
        print(f"    cPHS error: {e}")
        cphs = np.nan

    try:
        parsimony = cass.tools.tree_metrics.calculate_parsimony(tree)
    except Exception:
        parsimony = np.nan

    likelihood = safe_likelihood(tree)

    result = {
        'cPHS': cphs,
        'parsimony': parsimony,
        'likelihood': likelihood,
    }

    if base_tree is not None:
        try:
            rf, rf_max = cass.critique.compare.robinson_foulds(base_tree, tree)
            result['rf_to_base'] = rf / rf_max if rf_max > 0 else 0
        except Exception:
            result['rf_to_base'] = np.nan

        try:
            trip = cass.critique.compare.triplets_correct(base_tree, tree)
            result['triplets_to_base'] = np.mean(list(trip[0].values()))
        except Exception:
            result['triplets_to_base'] = np.nan

    return result


# ══════════════════════════════════════════════════════════════════════
# Main analysis
# ══════════════════════════════════════════════════════════════════════

def get_reliable_tumors():
    df = pd.read_csv(MERGED_CSV)
    df = df[~df.tumor.isin(EXCLUDED)]
    df = df[df.n_sites >= 7]
    reliable = []
    for tumor in sorted(df.tumor.unique()):
        info = df[df.tumor == tumor].iloc[0]
        q = info.q
        k = int(info.n_sites)
        if q < Q_THRESHOLD and k > K_FLAG_CUTOFF and tumor not in SKIP_TUMORS:
            reliable.append(tumor)
    return reliable


def analyze_tumor(tumor, n_reps, start_rep, output_path):
    print(f"\n{'='*60}")
    print(f"Tumor: {tumor}")
    print(f"{'='*60}")

    # Load data
    cm, priors = load_tumor_data(tumor)
    n_cells = len(cm)
    print(f"  Cells: {n_cells}, Sites: {cm.shape[1]}")

    # Load/reconstruct base trees and count edges
    base_trees = {}
    edge_counts = {}

    # Cassiopeia-Hybrid (precomputed)
    try:
        tree_cass = load_precomputed_tree(tumor, cm, priors)
        n_edges_cass = count_internal_edges(tree_cass)
        base_trees['cassiopeia'] = tree_cass
        edge_counts['cassiopeia'] = n_edges_cass
        print(f"  Cassiopeia: {tree_cass.n_cell} leaves, {n_edges_cass} internal edges")
    except Exception as e:
        print(f"  WARNING: Could not load Cassiopeia tree: {e}")

    # SMJ
    try:
        tree_smj = reconstruct(cm, priors, 'smj')
        n_edges_smj = count_internal_edges(tree_smj)
        base_trees['smj'] = tree_smj
        edge_counts['smj'] = n_edges_smj
        print(f"  SMJ: {tree_smj.n_cell} leaves, {n_edges_smj} internal edges")
    except Exception as e:
        print(f"  WARNING: Could not reconstruct SMJ tree: {e}")

    if not base_trees:
        print(f"  ERROR: No base trees, skipping")
        return

    # Check for existing results (resume)
    existing_keys = set()
    if os.path.exists(output_path):
        try:
            old = pd.read_csv(output_path)
            for _, row in old.iterrows():
                existing_keys.add((row.method, row.edge_frac, row.rep))
            print(f"  Resuming: {len(existing_keys)} entries already done")
        except Exception:
            pass

    # Run perturbation analysis
    rows = []
    end_rep = start_rep + n_reps

    for method_name, base_tree in base_trees.items():
        n_edges = edge_counts[method_name]
        print(f"\n  Method: {method_name} ({n_edges} internal edges)")

        # Get base tree newick
        base_nwk = tree_to_newick(base_tree)

        for rep in range(start_rep, end_rep):
            for edge_frac in EDGE_FRACTIONS:
                key = (method_name, edge_frac, rep)
                if key in existing_keys:
                    continue

                n_moves = max(0, int(edge_frac * n_edges))

                if edge_frac == 0.0:
                    # Base tree
                    metrics = compute_metrics(base_tree)
                    metrics['rf_to_base'] = 0.0
                    metrics['triplets_to_base'] = 1.0
                    nwk = base_nwk
                else:
                    seed = int(edge_frac * 10000) + rep * 137 + hash(tumor) % 10000
                    try:
                        pt, nwk = perturb_topology_nni(base_tree, n_moves, seed=seed)
                        metrics = compute_metrics(pt, base_tree=base_tree)
                    except Exception as e:
                        print(f"    {edge_frac:.0%} rep {rep} FAILED: {e}")
                        metrics = {k: np.nan for k in ['cPHS', 'parsimony', 'likelihood',
                                                         'rf_to_base', 'triplets_to_base']}
                        nwk = ''

                rows.append({
                    'tumor': tumor, 'method': method_name,
                    'edge_frac': edge_frac, 'n_moves': n_moves,
                    'n_internal_edges': n_edges,
                    'rep': rep,
                    'newick': nwk,
                    **metrics
                })

            print(f"    Rep {rep}: done ({len(EDGE_FRACTIONS)} levels)")

    # Save
    if rows:
        new_df = pd.DataFrame(rows)
        if os.path.exists(output_path):
            old_df = pd.read_csv(output_path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=['method', 'edge_frac', 'rep'], keep='last')
            combined.to_csv(output_path, index=False)
        else:
            new_df.to_csv(output_path, index=False)
        print(f"\n  Saved {len(rows)} new rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='NNI perturbation analysis v2 (edge-normalized)')
    parser.add_argument('--tumor', type=str, required=True,
                        help='Tumor name or "ALL"')
    parser.add_argument('--reps', type=int, default=20,
                        help='Number of replicates to run')
    parser.add_argument('--start-rep', type=int, default=0,
                        help='Starting replicate index (for parallel jobs)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.tumor == 'ALL':
        tumors = get_reliable_tumors()
    else:
        tumors = [args.tumor]

    print(f"NNI Analysis v2 (edge-normalized): {len(tumors)} tumors")
    print(f"Reps: {args.start_rep} to {args.start_rep + args.reps - 1}")
    print(f"Edge fraction levels: {EDGE_FRACTIONS}")
    print(f"Output: {OUTPUT_DIR}")

    for tumor in tumors:
        output_path = os.path.join(OUTPUT_DIR, f'{tumor}_nni_v2.csv')
        analyze_tumor(tumor, args.reps, args.start_rep, output_path)

    print(f"\nDone. Results in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
