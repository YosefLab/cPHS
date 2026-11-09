"""
Minimal example: score the deposited Cassiopeia tree of one KPTracer tumor with cPHS.

    python examples/score_one_tree.py --data ~/KPTracer-Data --tumor 3467_Lkb1_T1

Data: Yang et al. (2022), Zenodo DOI 10.5281/zenodo.5847462 (KPTracer-Data.tar.gz).
"""
import argparse, os, pickle, sys
import pandas as pd
import cassiopeia as cass
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cphs import calc_cPHS, get_lam_and_q, prepare_tree

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to the extracted KPTracer-Data folder")
ap.add_argument("--tumor", required=True)
a = ap.parse_args()

d = os.path.join(os.path.expanduser(a.data), "trees")
cm = pd.read_csv(f"{d}/{a.tumor}_character_matrix.txt", sep="\t", index_col=0)
cm = cm.apply(pd.to_numeric, errors="coerce").fillna(-1).astype(int)
with open(f"{d}/{a.tumor}_priors.pkl", "rb") as f:
    priors = {i: {int(s): p for s, p in v.items()} for i, v in pickle.load(f).items()}
with open(f"{d}/{a.tumor}_tree.nwk") as f:
    nwk = f.read().strip()

tree = prepare_tree(cass.data.CassiopeiaTree(character_matrix=cm, tree=nwk, priors=priors))
lam, q = get_lam_and_q(tree)
print(f"{a.tumor}: n={cm.shape[0]}, k={cm.shape[1]}, lambda={lam:.3f}, q={q:.4f}, "
      f"cPHS={calc_cPHS(tree, lam, q):.3e}")
