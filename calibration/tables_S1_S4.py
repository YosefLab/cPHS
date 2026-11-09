#!/usr/bin/env python3
"""
Recompute Tables S1-S4 (Appendix M threshold calibration) from the simulation results.

For each (n, k, m, rho, epsilon) row, every reconstructed tree in the results file is
labelled accurate if rf_norm <= epsilon, and predicted accurate if cPHS >= t.
BA(t) = (sensitivity + specificity) / 2. "Med. cPHS" = median cPHS over all reconstructions.

Results files (from calibration/threshold_simulation*.py), searched in this order:
  threshold_n{n}_k{k}.csv                (m = 50, rho = 0.5)
  threshold_n{n}_k{k}_m{m}.csv           (rho = 0.5)
  threshold_n{n}_k{k}_m{m}_rho{r}.csv
Every matching file is reported, so the one behind each published row can be identified.

    python calibration/tables_S1_S4.py ~/threshold_calibration/results
"""
import os, sys
import numpy as np
import pandas as pd

RES = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "results")
T = [0.05, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]
RCODE = {0.5: "5", 0.7: "7", 0.9: "9", 0.99: "99"}


def candidates(n, k, m, rho):
    r = RCODE[rho]
    c = []
    if m == 50 and rho == 0.5:
        c.append(f"threshold_n{n}_k{k}.csv")
    if rho == 0.5:
        c.append(f"threshold_n{n}_k{k}_m{m}.csv")
    c.append(f"threshold_n{n}_k{k}_m{m}_rho{r}.csv")
    if m == 50:
        c.append(f"threshold_n{n}_k{k}_rho{r}.csv")
    return [f for f in c if os.path.exists(os.path.join(RES, f))]


def ba_row(df, eps, logbase=None):
    acc = df.rf_norm.values <= eps
    out = []
    for t in T:
        if logbase is None:
            pred = df.cPHS.values >= t
        else:
            pred = df.log_cPHS.values >= (np.log(t) if logbase == "ln" else np.log10(t))
        sens = (pred & acc).sum() / max(acc.sum(), 1)
        spec = (~pred & ~acc).sum() / max((~acc).sum(), 1)
        out.append(0.5 * (sens + spec))
    if logbase is None:
        med = np.median(df.cPHS.values)
    else:
        lm = np.median(df.log_cPHS.values)
        med = np.exp(lm) if logbase == "ln" else 10 ** lm
    return out, med, int(acc.sum()), int((~acc).sum())


def show(label, n, k, m, rho, eps):
    files = candidates(n, k, m, rho)
    if not files:
        print(f"{label:<28} no results file found"); return
    for f in files:
        df = pd.read_csv(os.path.join(RES, f))
        modes = ["ln", "log10"] if "log_cPHS" in df.columns else [None]
        for mode in modes:
            ba, med, na, ni = ba_row(df, eps, mode)
            tag = f + (f" [{mode}]" if mode else "")
            print(f"{label:<28} " + " ".join(f"{b:.2f}" for b in ba) +
                  f"  med={med:.1e}  acc/inacc={na}/{ni}  <- {tag}")


print("columns: BA(0.05) BA(0.01) BA(1e-3) BA(1e-4) BA(1e-5) BA(1e-6)\n")

print("=== Table S1 (rho = 0.5, m = 50) ===")
s1 = [(100,10,2/3),(200,10,2/3),(300,10,2/3),(500,10,2/3),(1000,10,2/3),
      (100,15,.50),(200,15,.50),(300,15,.50),(500,15,2/3),
      (100,20,.50),(200,20,.50),(300,20,.50),(500,20,.50),(1000,20,.50),
      (100,25,.50),(200,25,.50),(300,25,.50),(500,25,.50),
      (100,30,.50),(200,30,.50),(300,30,.50),(500,30,.50),(1000,30,.50)]
for n, k, e in s1:
    show(f"n={n} k={k} eps={e}", n, k, 50, 0.5, e)

for n, name in [(100, "S2"), (500, "S3")]:
    print(f"\n=== Table {name} (n = {n}, eps = 0.50, m = 50) ===")
    for k in [15, 20, 25, 30]:
        for rho in [0.5, 0.7, 0.9, 0.99]:
            show(f"k={k} rho={rho}", n, k, 50, rho, 0.50)

print("\n=== Table S4 (n = 100, eps = 0.50) ===")
for k in [25, 30]:
    for m in [50, 10, 3]:
        for rho in [0.5, 0.9]:
            show(f"k={k} m={m} rho={rho}", 100, k, m, rho, 0.50)
