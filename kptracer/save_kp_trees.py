#!/usr/bin/env python3
"""
Rebuild and save the KPTracer reconstructions used in the paper, reusing the
EXACT functions from kptracer/kp_phs_all.py (everything after the download cell and before Cell 7).

Saves Newick with branch lengths:
    OUT_DIR/{tumor}/{method}.nwk     method in maxcut, smj, spectral, spectral_greedy
    OUT_DIR/{tumor}/random_{r}.nwk   r = 0..4 (same seeds as the paper)
    OUT_DIR/{tumor}/parsimony.csv    for verification against merged_all_tumors.csv

MaxCut is stochastic and was unseeded in the original run. Here it is seeded
(np.random + random) and seeds 0..MAX_TRIES-1 are tried until the parsimony equals
the frozen value; the seed used is recorded. cPHS of the saved MaxCut tree is also
recorded so it can be compared with the frozen value.

Usage:
    python save_kp_trees.py --tumor 3467_Lkb1_T1
"""
import argparse, os, sys, random
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

PIPELINE = os.path.join(REPO, 'kptracer', 'kp_phs_all.py')
FROZEN = os.path.join(REPO, 'results', 'merged_all_tumors.csv')
OUT_DIR = os.path.join(REPO, 'output', 'trees')
METHODS = ["maxcut", "smj", "spectral", "spectral_greedy"]   # all stochastic, seed-matched
N_RANDOM = 5
MAX_TRIES = 20

# ── load kp_phs_all.py from "# In[2]:" (after the download cell) up to Cell 7 (before the main loop) ──
src = open(PIPELINE).read()
a, b = src.find("# In[2]:"), src.find("# %% Cell 7")
if a < 0 or b < 0:
    sys.exit("Cell markers not found in kp_phs_all.py")
ns = {"__name__": "kp_phs_all_defs"}
exec(compile(src[a:b], PIPELINE, "exec"), ns)
cass = ns["cass"]


def to_newick_with_lengths(tree):
    def rec(node):
        kids = tree.children(node)
        if not kids:
            return str(node)
        return "(" + ",".join(f"{rec(c)}:{tree.get_branch_length(node, c):.10g}" for c in kids) + ")"
    return rec(tree.root) + ";"


def save(tree, path):
    with open(path, "w") as f:
        f.write(to_newick_with_lengths(tree) + "\n")


def solve_matching(cm, priors, method, target):
    """Seed np.random + random with 0,1,2,... until parsimony == target.
    All four solvers are stochastic in Cassiopeia 2.1.0 and were unseeded in the
    original run. Returns (tree, seed, matched); falls back to seed 0 if no match."""
    first = None
    for s in range(MAX_TRIES):
        np.random.seed(s); random.seed(s)
        t = ns["reconstruct_with_solver"](cm, priors, method)
        if t is None:
            return None, s, False
        if first is None:
            first = t
        if target is None or cass.tools.calculate_parsimony(t) == target:
            return t, s, target is not None
    return first, 0, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tumor", required=True)
    tumor = ap.parse_args().tumor
    out = os.path.join(OUT_DIR, tumor)
    os.makedirs(out, exist_ok=True)

    cm, priors = ns["load_cm_and_priors"](tumor)
    fz = pd.read_csv(FROZEN)
    fz = fz[fz.tumor == tumor]
    rows = []  # tumor, method, rep, parsimony, seed, matched, cPHS

    tree_cass, _, _ = ns["load_precomputed"](tumor, "cassiopeia")
    lam, q = ns["get_lam_and_q"](tree_cass)        # as in process_tumor
    for m in METHODS:
        tgt = fz[fz.method == m].parsimony
        target = int(tgt.iloc[0]) if len(tgt) else None
        t, s, ok = solve_matching(cm, priors, m, target)
        if t is None:
            rows.append((tumor, m, "", "", s, False, "")); continue
        save(t, f"{out}/{m}.nwk")
        cphs = ns["calc_cPHS"](t, lam, q) if m == "maxcut" or not ok else ""
        rows.append((tumor, m, "", cass.tools.calculate_parsimony(t), s, ok, cphs))
        print(f"  {m}: seed {s}, matched={ok}")

    for r, t in enumerate(ns["make_random_trees"](cm, priors, N_RANDOM)):
        save(t, f"{out}/random_{r}.nwk")
        rows.append((tumor, "random", r, cass.tools.calculate_parsimony(t), "", "", ""))

    pd.DataFrame(rows, columns=["tumor", "method", "rep", "parsimony",
                                "seed", "matched", "cPHS"]
                 ).to_csv(f"{out}/parsimony.csv", index=False)
    print(f"  {tumor}: done")


if __name__ == "__main__":
    main()
