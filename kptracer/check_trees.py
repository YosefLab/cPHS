#!/usr/bin/env python3
"""
Compare the regenerated trees (output/trees/*/parsimony.csv, from save_kp_trees.py) with the
trees scored in the paper (results/merged_all_tumors.csv). Writes unmatched.csv: every tree whose
parsimony differs, with paper vs regenerated cPHS and whether the accept/reject decision changes
(t = 1e-3 for k <= 24, 1e-4 for k >= 25).

    python kptracer/check_trees.py [trees_dir]
"""
import glob, os, sys
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'output', 'trees')
fz = pd.read_csv(os.path.join(REPO, 'results', 'merged_all_tumors.csv'))
thr = lambda k: 1e-3 if k <= 24 else 1e-4

rows, n_trees, n_tumors = [], 0, 0
for f in sorted(glob.glob(os.path.join(TREES, '*', 'parsimony.csv'))):
    d = pd.read_csv(f); t = d.tumor.iloc[0]; z = fz[fz.tumor == t]; n_tumors += 1
    k = int(z.n_sites.iloc[0])
    for m in ['maxcut', 'smj', 'spectral', 'spectral_greedy']:
        a, b = d[d.method == m], z[z.method == m]
        if a.empty or b.empty: continue
        a, b = a.iloc[0], b.iloc[0]; n_trees += 1
        if float(a.parsimony) != float(b.parsimony):
            c_new, c_old = float(a.cPHS), float(b.cPHS)
            rows.append(dict(tumor=t, method=m, k=k, threshold=thr(k),
                             parsimony_paper=int(b.parsimony), parsimony_deposited=int(a.parsimony),
                             cPHS_paper=c_old, cPHS_deposited=c_new,
                             pass_paper=c_old >= thr(k), pass_deposited=c_new >= thr(k),
                             same_decision=(c_old >= thr(k)) == (c_new >= thr(k))))
    rn = list(d[d.method == 'random'].parsimony.astype(float)); ro = list(z[z.method == 'random'].parsimony.astype(float))
    n_trees += len(rn)
    if rn != ro: print(f'WARNING random trees differ for {t}: {rn} vs {ro}')

u = pd.DataFrame(rows)
u.to_csv(os.path.join(TREES, 'unmatched.csv'), index=False)
print(f'{n_tumors} tumors, {n_trees} trees checked; {len(u)} reconstructions differ in parsimony')
if len(u):
    print(u.groupby('method').size().to_string())
    print('decision changes:', int((~u.same_decision).sum()))
    print(u[~u.same_decision].to_string(index=False) if (~u.same_decision).any() else '  none')
