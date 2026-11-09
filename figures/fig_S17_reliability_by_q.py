#!/usr/bin/env python3
"""
Fig S17: cPHS reliability vs collision probability q (63 analyzed tumors).
The plotting block below is copied verbatim from the "PLOT 19b" cell of
fig19a_res4.py, which produced the published figure; only the setup lines
and the two file paths differ.
    python figures/fig_S17_reliability_by_q.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(REPO, 'output', 'figures')
os.makedirs(PLOT_DIR, exist_ok=True)
eps = 1e-17          # as in fig19a_res4.py
Q_THRESHOLD = 0.35   # as in fig19a_res4.py
df = pd.read_csv(os.path.join(REPO, 'results', 'merged_all_tumors.csv'))

GENO_COLORS = {}
for g in df.genotype.unique():
    if g == 'NT':     GENO_COLORS[g] = '#4682B4'
    elif g == 'Lkb1': GENO_COLORS[g] = '#FF6347'
    elif g == 'Apc':  GENO_COLORS[g] = '#228B22'
    else:             GENO_COLORS[g] = '#808080'


# ── PLOT 19b: cPHS reliability — panels A & B only, 63 analyzed tumors ─────
# Restrict to the 63 analyzed tumors (read explicit list), and use the
# k-dependent calibrated threshold for the random-pass test.
with open(os.path.join(REPO, "results", "tumors63.txt")) as fh:
    TUMORS_63 = set(line.strip() for line in fh if line.strip())
tumors_19b = [t for t in df.tumor.unique() if t in TUMORS_63]
print(f"19b: plotting {len(tumors_19b)} tumors (intersected with the 63)")

tumor_q_stats = []
for tumor in tumors_19b:
    sub = df[df.tumor == tumor]
    info = sub.iloc[0]
    q_val = info.q
    n_cells = int(info.n_cells)
    n_sites = int(info.n_sites) if pd.notna(info.get('n_sites')) else 0
    geno = info.genotype

    rand_sub = sub[sub.category == 'random']['cPHS'].dropna()
    n_rand = len(rand_sub)
    t_thr = 1e-3 if n_sites <= 24 else 1e-4              # 1e-3 for k<=24 (incl. flagged), 1e-4 for k>=25
    n_rand_pass = int((rand_sub >= t_thr).sum())
    frac_rand_pass = n_rand_pass / n_rand if n_rand > 0 else 0

    recon = sub[sub.category.isin(['precomputed', 'solver'])]
    best_cphs = recon['cPHS'].max() if len(recon) else np.nan

    rand_mean = rand_sub.mean() if n_rand > 0 else np.nan
    if pd.notna(best_cphs) and best_cphs > 0 and pd.notna(rand_mean) and rand_mean > 0:
        sep = np.log10(max(best_cphs, eps)) - np.log10(max(rand_mean, eps))
    else:
        sep = 0

    tumor_q_stats.append({
        'tumor': tumor, 'q': q_val, 'n_cells': n_cells, 'n_sites': n_sites,
        'genotype': geno, 'frac_rand_pass': frac_rand_pass,
        'n_rand_pass': n_rand_pass, 'n_rand': n_rand,
        'best_cphs': best_cphs, 'rand_mean': rand_mean, 'separation': sep
    })

tqs = pd.DataFrame(tumor_q_stats)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# ── Panel A: q vs % random passing ──
ax = axes[0]
rng_jitter = np.random.RandomState(42)
for _, row in tqs.iterrows():
    c = GENO_COLORS.get(row.genotype, '#808080')
    jit = rng_jitter.uniform(-3, 3)
    y = row.frac_rand_pass * 100 + jit
    marker = 'X' if row.n_rand_pass > 0 else 'o'
    size = 90 if row.n_rand_pass > 0 else 55
    ax.scatter(row.q, y, c=c, s=size, marker=marker,
               edgecolors='black', linewidth=0.5, zorder=3)
ax.axvline(Q_THRESHOLD, color='red', ls='--', alpha=0.6)
ax.set_xlabel('Collision probability $q$')
ax.set_ylabel('% random trees passing cPHS (jittered)')
ax.set_title('A. Random-tree pass rate', fontweight='bold')
ax.set_ylim(-10, 110)
for g, c in GENO_COLORS.items():
    ax.scatter([], [], c=c, s=45, edgecolors='black', linewidth=0.4, label=g)
ax.scatter([], [], c='gray', s=70, marker='X', edgecolors='black', linewidth=0.4, label='Random passes')
ax.legend(fontsize=11, loc='center left')

# ── Panel B: q vs separation ──
ax = axes[1]
for _, row in tqs.iterrows():
    c = GENO_COLORS.get(row.genotype, '#808080')
    marker = 'X' if row.n_rand_pass > 0 else 'o'
    size = 90 if row.n_rand_pass > 0 else 55
    ax.scatter(row.q, row.separation, c=c, s=size, marker=marker,
               edgecolors='black', linewidth=0.5, zorder=3)
ax.axvline(Q_THRESHOLD, color='red', ls='--', alpha=0.6)
ax.axhline(0, color='gray', ls='-', alpha=0.3)
ax.set_xlabel('Collision probability $q$')
ax.set_ylabel(r'$\Delta\log_{10}$(cPHS): best recon $-$ random mean')
ax.set_title('B. Discrimination power', fontweight='bold')

n_danger = int((tqs.n_rand_pass > 0).sum())
fig.suptitle(f'cPHS reliability across the {len(tqs)} analyzed tumors',
             fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(PLOT_DIR, "FigS17_reliability_by_q.png"), dpi=150, bbox_inches='tight')
print(f"19b saved: {len(tqs)} tumors, random passes>0 in {n_danger} tumors")
