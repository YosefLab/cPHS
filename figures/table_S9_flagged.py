#!/usr/bin/env python3
"""
Table S9 (tab:perturbed_flagged): cPHS of NNI-perturbed Cassiopeia Greedy trees
for the 6 flagged tumors (k <= 15).

The published table was typed from values read off merged_all_tumors.csv
(April 2026 analysis session); this script recomputes them from that file:
  Best recon  = highest cPHS among the six reconstructions of the tumor
  x% NNI      = cPHS of the perturbed_{x}pct row (Cassiopeia tree + NNI, from kp_phs_all.py)

    python figures/table_S9_flagged.py results/merged_all_tumors.csv
"""
import sys
import pandas as pd

FROZEN = sys.argv[1] if len(sys.argv) > 1 else "results/merged_all_tumors.csv"
TUMORS = ["3430_NT_T2", "3454_Lkb1_T3", "3508_Apc_T2_Fam", "3433_NT_T2", "3510_Apc_T2", "3732_Lkb1_T2"]
LEVELS = ["perturbed_5pct", "perturbed_10pct", "perturbed_25pct", "perturbed_50pct"]
RECON = ["cassiopeia", "nj", "maxcut", "smj", "spectral", "spectral_greedy"]

df = pd.read_csv(FROZEN)
print("Tumor & $k$ & Best recon & 5\\% NNI & 10\\% NNI & 25\\% NNI & 50\\% NNI \\\\")
for t in TUMORS:
    sub = df[df.tumor == t]
    if sub.empty:
        print(f"{t}: not in file"); continue
    k = int(sub.n_sites.iloc[0])
    best = sub[sub.method.isin(RECON)].cPHS.max()
    vals = [sub[sub.method == lv].cPHS.iloc[0] if (sub.method == lv).any() else float("nan") for lv in LEVELS]
    print(f"{t.replace('_', chr(92) + '_')} & {k} & {best:.1e} & " + " & ".join(f"{v:.1e}" for v in vals) + " \\\\")
