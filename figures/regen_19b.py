#!/usr/bin/env python3
"""
Regenerate 19b reliability figure, panels A and B only, on the 63 analyzed tumors.
  A: per-tumor random-tree pass fraction vs q (calibrated k-dependent thresholds)
  B: discrimination power (best reconstruction cPHS - random mean, in log10) vs q
Reads merged_all_tumors.csv and the 63-tumor summary spreadsheet.
"""
import os
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLOT_DIR = os.path.join(REPO, 'output', 'figures')
MERGED   = os.path.join(REPO, 'results', 'merged_all_tumors.csv')
Q_THRESHOLD = 0.35
RECON = ['cassiopeia','nj','maxcut','smj','spectral','spectral_greedy']

def thr(k):
    # calibrated k-dependent threshold; None means flagged (k<=15)
    if k <= 15: return 0.05
    if k <= 24: return 1e-3
    return 1e-4

plt.rcParams.update({'font.size':16,'axes.titlesize':19,'axes.labelsize':16,
                     'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':13})

df = pd.read_csv(MERGED)

# --- define the 63 analyzed tumors: not Yang-rejected AND all 6 reconstructions present ---
# all-6 present = the tumor has a non-NaN cPHS for every reconstruction method
def has_all6(t):
    s = df[(df.tumor==t)&(df.method.isin(RECON))].dropna(subset=['cPHS'])
    return s.method.nunique()==6
# Yang-reject set is NOT in merged (merged has 74 = not-rejected already), so all-6 is the filter here.
tumors = [t for t in df.tumor.unique() if has_all6(t)]
print(f"tumors with all 6 reconstructions in merged: {len(tumors)}")

rows=[]
for t in tumors:
    sub = df[df.tumor==t]
    k = sub.n_sites.iloc[0]; q = sub.q.iloc[0]
    t_thr = thr(k)
    rec = sub[sub.method.isin(RECON)].dropna(subset=['cPHS'])
    ran = sub[sub.method=='random'].dropna(subset=['cPHS'])
    if rec.empty or ran.empty: continue
    # panel A: fraction of random trees passing
    rand_pass = (ran.cPHS >= t_thr).mean()
    # panel B: discrimination = log10(best recon cPHS) - log10(mean random cPHS)
    best_rec = rec.cPHS.max()
    mean_ran = ran.cPHS.mean()
    eps=1e-300
    sep = np.log10(best_rec+eps) - np.log10(mean_ran+eps)
    rows.append(dict(tumor=t,q=q,k=k,rand_pass=rand_pass,sep=sep))
r=pd.DataFrame(rows)
print(f"plotted tumors: {len(r)}")
print(f"  random passes>0: {(r.rand_pass>0).sum()} | all at q>=0.35? {bool((r[r.rand_pass>0].q>=Q_THRESHOLD).all())}")

fig, axes = plt.subplots(1,2,figsize=(18,7))

# Panel A
ax=axes[0]
lowq = r[r.q<Q_THRESHOLD]; hiq = r[r.q>=Q_THRESHOLD]
ax.scatter(lowq.q, lowq.rand_pass*100, c='#228B22', s=80, edgecolors='black', linewidth=0.4, label=f'low $q$ ({len(lowq)})', zorder=3)
ax.scatter(hiq.q,  hiq.rand_pass*100,  c='#FFA500', s=80, edgecolors='black', linewidth=0.4, label=f'high $q$ ({len(hiq)})', zorder=3)
ax.axvline(Q_THRESHOLD, color='red', ls='--', alpha=0.6)
ax.axvspan(Q_THRESHOLD,1.0,alpha=0.07,color='red')
ax.set_xlabel('Collision probability $q$'); ax.set_ylabel('% random trees passing cPHS')
ax.set_title('A. Random-tree pass rate', fontweight='bold')
ax.set_xlim(0,1.0); ax.legend()

# Panel B
ax=axes[1]
ax.scatter(lowq.q, lowq.sep, c='#228B22', s=80, edgecolors='black', linewidth=0.4, zorder=3)
ax.scatter(hiq.q,  hiq.sep,  c='#FFA500', s=80, edgecolors='black', linewidth=0.4, zorder=3)
ax.axvline(Q_THRESHOLD, color='red', ls='--', alpha=0.6)
ax.axvspan(Q_THRESHOLD,1.0,alpha=0.07,color='red')
ax.set_xlabel('Collision probability $q$'); ax.set_ylabel(r'Discrimination power ($\Delta\log_{10}$ cPHS)')
ax.set_title('B. Reconstruction vs random separation', fontweight='bold')
ax.set_xlim(0,1.0)

plt.tight_layout()
out=os.path.join(PLOT_DIR,'19b_cphs_reliability_filtered.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print('Saved', out)
