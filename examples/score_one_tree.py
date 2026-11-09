"""
Score one KPTracer tree with cPHS and parsimony, and compare with the value in the paper.

    python examples/score_one_tree.py --data ~/KPTracer-Data --tumor 3435_NT_T3
        scores the Cassiopeia tree deposited by Yang et al. (2022)
    python examples/score_one_tree.py --data ~/KPTracer-Data --tumor 3467_Lkb1_T1 --tree trees/3467_Lkb1_T1/smj.nwk
        scores a tree deposited in this repository

Data: Yang et al. (2022), Zenodo DOI 10.5281/zenodo.5847462 (KPTracer-Data.tar.gz).
"""
import argparse, os, pickle, sys
import pandas as pd
import cassiopeia as cass
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from cphs import calc_cPHS, get_lam_and_q, prepare_tree

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to the extracted KPTracer-Data folder")
ap.add_argument("--tumor", required=True)
ap.add_argument("--tree", help="Newick file to score (default: the deposited Cassiopeia tree)")
a = ap.parse_args()

d = os.path.join(os.path.expanduser(a.data), "trees")
cm = pd.read_csv(f"{d}/{a.tumor}_character_matrix.txt", sep="\t", index_col=0)
cm = cm.apply(pd.to_numeric, errors="coerce").fillna(-1).astype(int)
with open(f"{d}/{a.tumor}_priors.pkl", "rb") as f:
    priors = {i: {int(s): p for s, p in v.items()} for i, v in pickle.load(f).items()}
with open(f"{d}/{a.tumor}_tree.nwk") as f:
    nwk_cass = f.read().strip()

# lambda and q as in the paper: from the Cassiopeia tree of the tumor
tree_cass = prepare_tree(cass.data.CassiopeiaTree(character_matrix=cm, tree=nwk_cass, priors=priors))
lam, q = get_lam_and_q(tree_cass)

if a.tree:
    with open(a.tree) as f:
        tree = prepare_tree(cass.data.CassiopeiaTree(character_matrix=cm, tree=f.read().strip(), priors=priors))
    name = os.path.basename(a.tree).replace(".nwk", "")
else:
    tree, name = tree_cass, "cassiopeia"

cphs, pars = calc_cPHS(tree, lam, q), cass.tools.calculate_parsimony(tree)
print(f"{a.tumor} {name}: n={cm.shape[0]}, k={cm.shape[1]}, lambda={lam:.3f}, q={q:.4f}, "
      f"parsimony={pars}, cPHS={cphs:.3e}")

# value in the paper (random_r is the r-th random tree)
fz = pd.read_csv(os.path.join(REPO, "results", "merged_all_tumors.csv"))
fz = fz[fz.tumor == a.tumor]
if name.startswith("random_"):
    row = fz[fz.method == "random"].iloc[int(name.split("_")[1])]
else:
    row = fz[fz.method == name].iloc[0] if (fz.method == name).any() else None
if row is not None:
    print(f"  paper: parsimony={row.parsimony:.0f}, cPHS={row.cPHS:.3e}")
