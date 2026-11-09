#!/usr/bin/env python3
"""
Patch NNI results with likelihood scores.

Re-creates trees using the same deterministic seeds as nni_analysis.py,
computes only likelihood (skips cPHS which is slow), and updates the CSV.

Usage:
    python nni_patch_likelihood.py --tumor 3467_Lkb1_T1
    python nni_patch_likelihood.py --tumor ALL
"""

import os, sys, argparse, pickle, warnings
import numpy as np
import pandas as pd
from copy import deepcopy

warnings.filterwarnings('ignore')

import cassiopeia as cass
from cassiopeia.tools.branch_length_estimator.IIDExponentialMLE import IIDExponentialMLE

# ── Paths ──
TREES_DIR = '/home/projects/nyosef/yaaraneu/KPTracer-Data/trees'
NNI_DIR = '/home/projects/nyosef/yaaraneu/nni_analysis'

STOCHASTIC_MISSING_PROB = 0.10


# ══════════════════════════════════════════════════════════════════════
# Data loading and tree operations (same as nni_analysis.py)
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


def reconstruct_smj(cm, priors):
    tree = cass.data.CassiopeiaTree(character_matrix=cm.copy(), priors=priors)
    solver = cass.solver.SharedMutationJoiningSolver()
    solver.solve(tree)
    return postprocess(tree)


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
    return postprocess(new_tree)


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
        # Fallback: try with 0.0
        try:
            tree.parameters['stochastic_missing_probability'] = 0.0
            with np.errstate(divide='ignore', invalid='ignore'):
                lik = cass.tools.calculate_likelihood_continuous(tree)
            if np.isinf(lik):
                return np.nan
            return lik
        except Exception:
            return np.nan


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def patch_tumor(tumor):
    csv_path = os.path.join(NNI_DIR, f'{tumor}_nni.csv')
    if not os.path.exists(csv_path):
        print(f"  {tumor}: no CSV found, skipping")
        return

    df = pd.read_csv(csv_path)
    n_rows = len(df)
    print(f"\n{'='*60}")
    print(f"Tumor: {tumor} ({n_rows} rows)")

    # Check if already patched
    if df.likelihood.notna().sum() > 0:
        print(f"  Already has {df.likelihood.notna().sum()} likelihood values, skipping")
        return

    # Load data
    cm, priors = load_tumor_data(tumor)
    n_cells = len(cm)
    print(f"  Cells: {n_cells}, Sites: {cm.shape[1]}")

    # Build base trees
    base_trees = {}
    try:
        base_trees['cassiopeia'] = load_precomputed_tree(tumor, cm, priors)
        print(f"  Loaded Cassiopeia tree")
    except Exception as e:
        print(f"  Cassiopeia load failed: {e}")

    try:
        base_trees['smj'] = reconstruct_smj(cm, priors)
        print(f"  Reconstructed SMJ tree")
    except Exception as e:
        print(f"  SMJ reconstruct failed: {e}")

    # Compute likelihood for each row
    count = 0
    for idx, row in df.iterrows():
        method = row.method
        pct = row.pct
        rep = row.rep
        frac = row.frac

        if method not in base_trees:
            continue

        base_tree = base_trees[method]

        if pct == 0:
            # Base tree
            lik = safe_likelihood(base_tree)
        else:
            # Re-create perturbed tree with same seed
            n_moves = int(row.n_moves)
            seed = int(frac * 10000) + rep * 137 + hash(tumor) % 10000
            try:
                pt = perturb_topology_nni(base_tree, n_moves, seed=seed)
                lik = safe_likelihood(pt)
            except Exception:
                lik = np.nan

        df.at[idx, 'likelihood'] = lik
        count += 1

        if count % 50 == 0:
            print(f"  {count}/{n_rows} done")

    # Save
    df.to_csv(csv_path, index=False)
    n_valid = df.likelihood.notna().sum()
    print(f"  Done: {n_valid}/{n_rows} likelihood values computed")


def main():
    parser = argparse.ArgumentParser(description='Patch NNI results with likelihood')
    parser.add_argument('--tumor', type=str, required=True,
                        help='Tumor name or "ALL"')
    args = parser.parse_args()

    if args.tumor == 'ALL':
        tumors = sorted([f.replace('_nni.csv', '')
                        for f in os.listdir(NNI_DIR) if f.endswith('_nni.csv')])
    else:
        tumors = [args.tumor]

    print(f"Patching likelihood for {len(tumors)} tumors")

    for tumor in tumors:
        patch_tumor(tumor)

    print("\nDone.")


if __name__ == '__main__':
    main()
