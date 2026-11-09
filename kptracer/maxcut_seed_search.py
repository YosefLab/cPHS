#!/usr/bin/env python3
"""
Extend the MaxCut seed search for one tumor, for trees where save_kp_trees.py found no match
within MAX_TRIES. Tries seeds START..END-1; on a match, replaces <trees>/<tumor>/maxcut.nwk and
updates its row in parsimony.csv (seed, matched, cPHS).

    python kptracer/maxcut_seed_search.py --tumor 3435_NT_T3 [--start 20 --end 200 --trees trees]
"""
import argparse, os, random
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKT = os.path.join(REPO, 'kptracer', 'save_kp_trees.py')
g = {"__name__": "maxcut_seed_search", "__file__": SKT}
exec(open(SKT).read().split("def main")[0], g)       # pipeline functions, exactly as in save_kp_trees.py
ns, cass, save = g["ns"], g["cass"], g["save"]

ap = argparse.ArgumentParser()
ap.add_argument("--tumor", required=True)
ap.add_argument("--start", type=int, default=20)
ap.add_argument("--end", type=int, default=200)
ap.add_argument("--trees", default=os.path.join(REPO, "trees"))
a = ap.parse_args(); t = a.tumor

fz = pd.read_csv(os.path.join(REPO, "results", "merged_all_tumors.csv"))
target = int(fz[(fz.tumor == t) & (fz.method == "maxcut")].parsimony.iloc[0])
cm, priors = ns["load_cm_and_priors"](t)
for s in range(a.start, a.end):
    np.random.seed(s); random.seed(s)
    tree = ns["reconstruct_with_solver"](cm, priors, "maxcut")
    if tree is None: break
    if cass.tools.calculate_parsimony(tree) == target:
        save(tree, os.path.join(a.trees, t, "maxcut.nwk"))
        tree_cass, _, _ = ns["load_precomputed"](t, "cassiopeia")
        lam, q = ns["get_lam_and_q"](tree_cass)
        p = os.path.join(a.trees, t, "parsimony.csv"); d = pd.read_csv(p); i = d.index[d.method == "maxcut"][0]
        d.loc[i, ["parsimony", "seed", "matched", "cPHS"]] = [target, s, True, ns["calc_cPHS"](tree, lam, q)]
        d.to_csv(p, index=False)
        print(f"{t}: matched at seed {s}"); break
else:
    print(f"{t}: no match in seeds {a.start}-{a.end - 1}")
