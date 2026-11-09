#!/usr/bin/env python3
"""
Table S10 (tab:degradation): effect of NNI perturbation on the Cassiopeia Greedy tree,
low-q tumors only. Reconstructed verbatim from the code that generated
appendix_table_degradation.tex (April 2026 analysis session).

Input : merged_all_tumors.csv (frozen results; perturbed_*pct rows come from kp_phs_all.py)
Output: appendix_table_degradation.tex + the table printed to stdout

Notes on definitions, kept exactly as used for the paper table:
  - tumor set: all tumors in the frozen CSV except 3430_NT_T1, 3434_NT_T1, 3777_Apc_T5,
    with n_sites >= 7 and q < 0.35  (48 tumors)
  - "cPHS pass" uses a uniform threshold t = 0.05 (ALPHA below)
"""
import sys
import numpy as np
import pandas as pd

FROZEN = sys.argv[1] if len(sys.argv) > 1 else "results/merged_all_tumors.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "appendix_table_degradation.tex"

df = pd.read_csv(FROZEN)
excluded = {'3430_NT_T1', '3434_NT_T1', '3777_Apc_T5'}
df = df[~df.tumor.isin(excluded)]
df = df[df.n_sites >= 7]
tumors = sorted(df.tumor.unique())
eps = 1e-16
Q_THRESH = 0.35
ALPHA = 0.05

lowq_tumors = [t for t in tumors if df[df.tumor == t].iloc[0].q < Q_THRESH]

levels = [
    ('cassiopeia', 'Cassiopeia (intact)'),
    ('perturbed_5pct', 'Cass + 5\\% NNI'),
    ('perturbed_10pct', 'Cass + 10\\% NNI'),
    ('perturbed_25pct', 'Cass + 25\\% NNI'),
    ('perturbed_50pct', 'Cass + 50\\% NNI'),
]

lines = [r"\begin{table}[ht]", r"\centering",
         r"\caption{Effect of NNI perturbation on reconstruction quality, restricted to the 48 low-$q$ tumors ($q < 0.35$) where cPHS is fully reliable. Starting from the Cassiopeia Greedy tree, increasing fractions of NNI moves are applied. All metrics degrade monotonically with perturbation level. cPHS median and mean $\log_{10}$ show the typical cPHS value; Pars vs rand and Lik vs rand show the mean percent improvement over random trees.}",
         r"\label{tab:degradation}", r"\small", r"\begin{tabular}{l r r r r r r}", r"\toprule",
         r"Level & $N$ & cPHS pass & cPHS median & Mean $\log_{10}$ & Pars vs rand & Lik vs rand \\",
         r"\midrule"]

print(f"low-q tumors: {len(lowq_tumors)}")
for method_key, label in levels:
    cphs_vals, pars_pcts, lik_pcts = [], [], []
    for t in lowq_tumors:
        sub = df[df.tumor == t]
        rand_pars = sub[sub.category == 'random']['parsimony'].dropna().mean()
        rand_lik = sub[sub.category == 'random']['likelihood'].dropna().mean()
        msub = sub[sub.method == method_key]
        if len(msub) == 0:
            continue
        r = msub.iloc[0]
        if pd.notna(r.cPHS): cphs_vals.append(r.cPHS)
        if pd.notna(r.parsimony) and rand_pars > 0:
            pars_pcts.append((rand_pars - r.parsimony) / rand_pars * 100)
        if pd.notna(r.likelihood) and rand_lik != 0:
            lik_pcts.append((r.likelihood - rand_lik) / abs(rand_lik) * 100)
    n = len(cphs_vals)
    pass_rate = np.mean([c >= ALPHA for c in cphs_vals]) * 100
    median_cphs = np.median(cphs_vals)
    mean_log = np.mean(np.log10(np.clip(cphs_vals, eps, None)))
    pvr, lvr = np.mean(pars_pcts), np.mean(lik_pcts)
    row = f"{label} & {n} & {pass_rate:.1f}\\% & {median_cphs:.2e} & ${mean_log:.2f}$ & {pvr:.1f}\\% & {lvr:.0f}\\% \\\\"
    lines.append(row); print(row)

cphs_r, pars_r, lik_r = [], [], []
for t in lowq_tumors:
    sub = df[df.tumor == t]
    rand = sub[sub.category == 'random']
    rand_pars = rand['parsimony'].dropna().mean()
    rand_lik = rand['likelihood'].dropna().mean()
    for _, r in rand.iterrows():
        if pd.notna(r.cPHS): cphs_r.append(r.cPHS)
        if pd.notna(r.parsimony) and rand_pars > 0:
            pars_r.append((rand_pars - r.parsimony) / rand_pars * 100)
        if pd.notna(r.likelihood) and rand_lik != 0:
            lik_r.append((r.likelihood - rand_lik) / abs(rand_lik) * 100)
row = (f"Random & {len(cphs_r)} & {np.mean([c >= ALPHA for c in cphs_r])*100:.1f}\\% & "
       f"{np.median(cphs_r):.2e} & ${np.mean(np.log10(np.clip(cphs_r, eps, None))):.2f}$ & "
       f"{np.mean(pars_r):.1f}\\% & {np.mean(lik_r):.0f}\\% \\\\")
lines += [r"\midrule", row, r"\bottomrule", r"\end{tabular}", r"\end{table}"]
print(row)

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {OUT}")
