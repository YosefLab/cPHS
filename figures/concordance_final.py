#!/usr/bin/env python
"""
Best-method-per-tumor concordance on the 42 reliable tumors (q<0.35 & k>15).
Orientations verified from random-tree rows:
  cPHS:       higher = better  -> best = argmax
  parsimony:  lower  = better  -> best = argmin
  likelihood: higher = better  -> best = argmax
Run:
  /home/projects/nyosef/yaaraneu/.conda/envs/cass/bin/python concordance_final.py
"""
import os
import numpy as np
import pandas as pd

MERGED = os.path.expanduser("~/KP_results_20260311_022408/merged_all_tumors.csv")
METHODS = ["cassiopeia", "maxcut", "nj", "smj", "spectral", "spectral_greedy"]

df = pd.read_csv(MERGED)
df = df[df["method"].str.lower().isin(METHODS)].copy()
df = df[(df["q"] < 0.35) & (df["n_sites"] > 15)]
tumors = sorted(df["tumor"].unique())
print(f"Reliable tumors: {len(tumors)}")

def best(sub, col, higher_better):
    s = sub.dropna(subset=[col])
    if s.empty:
        return None
    i = s[col].idxmax() if higher_better else s[col].idxmin()
    return s.loc[i, "method"].lower()

rows = []
for t in tumors:
    sub = df[df["tumor"] == t]
    bc = best(sub, "cPHS", True)
    bp = best(sub, "parsimony", False)
    bl = best(sub, "likelihood", True)
    if None in (bc, bp, bl):
        continue
    rows.append((t, bc, bp, bl))

R = pd.DataFrame(rows, columns=["tumor", "cphs", "pars", "lik"])
n = len(R)
print(f"Tumors with all three best-method calls: {n}\n")

def agree(a, b):
    m = (R[a] == R[b]); return m.sum(), 100*m.mean()
for a, b, label in [("cphs","pars","cPHS vs parsimony"),
                    ("cphs","lik","cPHS vs likelihood"),
                    ("pars","lik","parsimony vs likelihood")]:
    k, pct = agree(a, b)
    print(f"{label:26s}: {k}/{n} = {pct:.0f}%")
all3 = ((R["cphs"]==R["pars"]) & (R["pars"]==R["lik"])).sum()
print(f"\nAll three agree: {all3}/{n} = {100*all3/n:.0f}%")
print("All-three winners:",
      R[(R["cphs"]==R["pars"]) & (R["pars"]==R["lik"])]["cphs"].value_counts().to_dict())
disc = R[R["cphs"] != R["lik"]]
print("When cPHS!=lik, lik picks:", disc["lik"].value_counts().to_dict())
print("\nBest-cPHS method counts:", R["cphs"].value_counts().to_dict())
