#!/usr/bin/env python3
"""19c: cPHS reliability vs k (panels A,B) + 2D k-vs-q map (panel C), 63 tumors."""
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
T63      = os.path.join(REPO, 'results', 'tumors63.txt')
Q_THRESHOLD = 0.35
K_FLAG = 15
RECON = ['cassiopeia','nj','maxcut','smj','spectral','spectral_greedy']
eps = 1e-300

def thr(k):  # chosen convention: 1e-3 for k<=24 (incl flagged), 1e-4 for k>=25
    return 1e-3 if k <= 24 else 1e-4

GENO_COLORS = {'Apc':'#1f77b4','Lkb1':'#ff7f0e','NT':'#2ca02c'}

plt.rcParams.update({'font.size':16,'axes.titlesize':18,'axes.labelsize':15,
                     'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':12})

df = pd.read_csv(MERGED)
t63 = set(open(T63).read().split())

rows=[]
for t in [x for x in df.tumor.unique() if x in t63]:
    sub = df[df.tumor==t]
    k=int(sub.n_sites.iloc[0]); q=sub.q.iloc[0]; geno=sub.genotype.iloc[0]
    ran = sub[sub.method=='random']['cPHS'].dropna()
    rec = sub[sub.method.isin(RECON)]['cPHS'].dropna()
    if len(ran)==0 or len(rec)==0: continue
    npass = int((ran>=thr(k)).sum())
    frac  = npass/len(ran)
    best  = rec.max(); rmean = ran.mean()
    sep = np.log10(max(best,eps)) - np.log10(max(rmean,eps)) if best>0 and rmean>0 else 0
    rows.append(dict(tumor=t,k=k,q=q,geno=geno,npass=npass,frac=frac,sep=sep))
r=pd.DataFrame(rows)
print(f"plotted {len(r)} tumors; random-pass tumors: {(r.npass>0).sum()}, total passes {int(r.npass.sum())}")

fig, axes = plt.subplots(1,3,figsize=(24,7))

def scatter_by_pass(ax, x, ycol):
    for _,row in r.iterrows():
        c = GENO_COLORS.get(row.geno,'#808080')
        m = 'X' if row.npass>0 else 'o'
        s = 95 if row.npass>0 else 55
        ax.scatter(row[x], row[ycol], c=c, s=s, marker=m,
                   edgecolors='black', linewidth=0.5, zorder=3)

# Panel A: random-pass % vs k
ax=axes[0]
rng=np.random.RandomState(42)
for _,row in r.iterrows():
    c=GENO_COLORS.get(row.geno,'#808080'); m='X' if row.npass>0 else 'o'; s=95 if row.npass>0 else 55
    ax.scatter(row.k, row.frac*100 + rng.uniform(-3,3), c=c, s=s, marker=m,
               edgecolors='black', linewidth=0.5, zorder=3)
ax.axvline(K_FLAG, color='purple', ls='--', alpha=0.6)
ax.set_xlabel('Number of character sites $k$'); ax.set_ylabel('% random trees passing cPHS (jittered)')
ax.set_title('A. Random-tree pass rate vs $k$', fontweight='bold'); ax.set_ylim(-10,110)
for g,c in GENO_COLORS.items(): ax.scatter([],[],c=c,s=45,edgecolors='black',linewidth=0.4,label=g)
ax.scatter([],[],c='gray',s=70,marker='X',edgecolors='black',linewidth=0.4,label='Random passes')
ax.legend(loc='upper right')

# Panel B: discrimination vs k
ax=axes[1]
scatter_by_pass(ax,'k','sep')
ax.axvline(K_FLAG, color='purple', ls='--', alpha=0.6)
ax.axhline(0, color='gray', ls='-', alpha=0.3)
ax.set_xlabel('Number of character sites $k$')
ax.set_ylabel(r'$\Delta\log_{10}$(cPHS): best recon $-$ random mean')
ax.set_title('B. Discrimination power vs $k$', fontweight='bold')

# Panel C: 2D k vs q, colored by pass
ax=axes[2]
safe=r[r.npass==0]; dang=r[r.npass>0]
ax.scatter(safe.q, safe.k, c='#228B22', s=60, edgecolors='black', linewidth=0.4, zorder=3,
           label=f'No random passes ({len(safe)})')
ax.scatter(dang.q, dang.k, c='#DC143C', s=95, marker='X', edgecolors='black', linewidth=0.5, zorder=4,
           label=f'Random passes ({len(dang)})')
ax.axvline(Q_THRESHOLD, color='red', ls='--', alpha=0.5)
ax.axhline(K_FLAG, color='purple', ls='--', alpha=0.5)
ax.axvspan(Q_THRESHOLD,1.0,alpha=0.06,color='red')
ax.axhspan(0,K_FLAG,alpha=0.05,color='purple')
ax.set_xlim(0,1.0); ax.set_ylim(0, r.k.max()+5)
ax.set_xlabel('Collision probability $q$'); ax.set_ylabel('Number of character sites $k$')
ax.set_title('C. Reliability map ($k$ vs $q$)', fontweight='bold'); ax.legend(loc='upper right')

fig.suptitle(f'cPHS reliability vs recording capacity — {len(r)} analyzed tumors', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
out=os.path.join(PLOT_DIR,'19c_reliability_by_k.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)
