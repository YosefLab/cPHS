#!/usr/bin/env python3
"""10b: separation vs predictors (q, lambda*q, lambda, k, k/n, n), 63 tumors."""
import os, numpy as np, pandas as pd
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLOT_DIR = os.path.join(REPO, 'output', 'figures')
os.makedirs(PLOT_DIR, exist_ok=True)
df=pd.read_csv(os.path.join(REPO, 'results', 'merged_all_tumors.csv'))
t63=set(open(os.path.join(REPO, 'results', 'tumors63.txt')).read().split())
RECON=['cassiopeia','nj','maxcut','smj','spectral','spectral_greedy']; eps=1e-300
GENO_COLORS={'Apc':'#1f77b4','Lkb1':'#ff7f0e','NT':'#2ca02c'}

plt.rcParams.update({'font.size':15,'axes.titlesize':17,'axes.labelsize':14,
                     'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':11})

rows=[]
for t in [x for x in df.tumor.unique() if x in t63]:
    sub=df[df.tumor==t]
    k=int(sub.n_sites.iloc[0]); n=int(sub.n_cells.iloc[0]); q=sub.q.iloc[0]; lam=sub.lam.iloc[0]; geno=sub.genotype.iloc[0]
    rec=sub[sub.method.isin(RECON)]['cPHS'].dropna(); ran=sub[sub.method=='random']['cPHS'].dropna()
    if len(rec)==0 or len(ran)==0: continue
    sep=np.log10(max(rec.max(),eps))-np.log10(max(ran.mean(),eps))
    rows.append(dict(k=k,n=n,q=q,lam=lam,kn=k/n,lq=lam*q,sep=sep,geno=geno))
r=pd.DataFrame(rows)
print('plotted', len(r), 'tumors')

panels=[('q',r'Collision probability $q$'),
        ('lq',r'$\lambda \times q$'),
        ('lam',r'Mutation rate $\lambda$'),
        ('k',r'Character sites $k$'),
        ('kn',r'$k/n$'),
        ('n',r'Number of cells $n$')]
letters='ABCDEF'

fig,axes=plt.subplots(2,3,figsize=(22,13))
for i,(col,xlabel) in enumerate(panels):
    ax=axes[i//3, i%3]
    for _,row in r.iterrows():
        ax.scatter(row[col], row.sep, c=GENO_COLORS.get(row.geno,'#808080'),
                   s=70, edgecolors='black', linewidth=0.4, zorder=3)
    rho,p=spearmanr(r[col], r.sep)
    ax.set_xlabel(xlabel); ax.set_ylabel(r'Separation ($\Delta\log_{10}$ cPHS)')
    ax.set_title(f'{letters[i]}. vs {xlabel} ($\\rho={rho:+.2f}$)', fontweight='bold')
    if col=='n': ax.set_xscale('log')
for g,c in GENO_COLORS.items(): axes[0,0].scatter([],[],c=c,s=50,edgecolors='black',linewidth=0.4,label=g)
axes[0,0].legend(loc='upper right')

fig.suptitle(f'Predictors of cPHS discriminatory power — {len(r)} analyzed tumors', fontsize=19, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
out=os.path.join(PLOT_DIR,'FigS19_predictors.png')
fig.savefig(out, dpi=150, bbox_inches='tight'); print('saved', out)
