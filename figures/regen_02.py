#!/usr/bin/env python3
"""
Regenerate Fig6_method_ranking.png with:
  - Larger font sizes
  - No NNI/perturbed columns (dedicated NNI figure exists)
"""

import os, re
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ──
PLOT_DIR = os.path.join(REPO, 'output', 'figures')
os.makedirs(PLOT_DIR, exist_ok=True)
MIN_SITES = 7
K_FLAG_CUTOFF = 15
Q_CPHS_THRESHOLD = 0.35
ALPHA_LENIENT = 1e-3
ALPHA_STRICT = 1e-4
eps = 1e-17

# ── Larger fonts ──
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
})

def get_threshold(k):
    # chosen convention: flagged k<=15 scored at 1e-3 (same as k=16-24)
    if k <= 24:
        return ALPHA_LENIENT
    else:
        return ALPHA_STRICT

def short_label(m):
    if m == 'cassiopeia': return 'Cass'
    if m == 'spectral_greedy': return 'Spec-G'
    if m == 'random': return 'Random'
    return m.upper() if len(m) <= 3 else m.capitalize()

METHOD_COLORS = {
    'cassiopeia': '#4682B4', 'nj': '#E74C3C',
    'maxcut': '#228B22', 'spectral': '#9370DB',
    'spectral_greedy': '#FF8C00', 'smj': '#8B4513',
    'random': '#A0A0A0',
}

# ── Load and process data ──
df = pd.read_csv(os.path.join(REPO, 'results', 'merged_all_tumors.csv'))
for col in ['cPHS', 'parsimony', 'likelihood', 'q', 'n_cells', 'n_sites']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'genotype' not in df.columns or df['genotype'].isna().all():
    df['genotype'] = df['tumor'].apply(
        lambda t: t.split("_")[1] if len(t.split("_")) >= 2 else "?")

# Exclude tumors with < MIN_SITES
excluded_tumors = set()
for t in df.tumor.unique():
    sub = df[df.tumor == t]
    n_sites = sub.iloc[0].get('n_sites', 999)
    if pd.notna(n_sites) and int(n_sites) < MIN_SITES:
        excluded_tumors.add(t)

# Restrict to the 63 analyzed tumors (explicit list), not the site-filter set
with open(os.path.join(REPO, 'results', 'tumors63.txt')) as _fh:
    TUMORS_63 = set(l.strip() for l in _fh if l.strip())
df = df[df.tumor.isin(TUMORS_63)].copy()
tumors = sorted(df.tumor.unique())
n_tumors = len(tumors)

# Likelihood availability
recon_mask = df.category.isin(['precomputed', 'solver'])
lik_frac = df.loc[recon_mask, 'likelihood'].notna().mean() if 'likelihood' in df.columns else 0
HAS_LIK = lik_frac > 0.5

# Per-tumor normalization
df['pars_norm'] = np.nan
df['lik_norm'] = np.nan
for tumor in tumors:
    mask = df.tumor == tumor
    sub = df.loc[mask]
    pv = sub['parsimony'].dropna()
    if len(pv) > 0:
        pmax = pv.max()
        if pmax > 0:
            df.loc[mask, 'pars_norm'] = 1.0 - sub['parsimony'] / pmax
    if HAS_LIK:
        lv = sub['likelihood'].dropna()
        if len(lv) > 1:
            lmin, lmax = lv.min(), lv.max()
            if lmax > lmin:
                df.loc[mask, 'lik_norm'] = (sub['likelihood'] - lmin) / (lmax - lmin)
            else:
                df.loc[mask, 'lik_norm'] = 0.5

# Method lists — NO perturbed
solver_methods = sorted(df[df.category == 'solver']['method'].unique())
precomputed_methods = sorted(df[df.category == 'precomputed']['method'].unique())
ALL_RECON_METHODS = precomputed_methods + solver_methods

# Plot order: reconstruction methods + random (no perturbed)
PLOT2_ORDER = ALL_RECON_METHODS + ['random']
PLOT2_LABELS = [short_label(m) for m in PLOT2_ORDER]
PLOT2_COLORS = [METHOD_COLORS.get(m, '#C0C0C0') for m in PLOT2_ORDER]

# Reliable low-q tumors
reliable_lowq_set = set()
for t in tumors:
    info = df[df.tumor == t].iloc[0]
    if info.q < Q_CPHS_THRESHOLD and int(info.n_sites) > K_FLAG_CUTOFF:
        reliable_lowq_set.add(t)
n_reliable_lowq = len(reliable_lowq_set)

def _get_vals(method, metric):
    if method == 'random':
        return df[df.category == 'random'][metric].dropna()
    else:
        return df[df.method == method][metric].dropna()

def _pass_rate_kdep(method, tumor_set):
    passes = 0; total = 0
    for t in tumor_set:
        sub = df[df.tumor == t]
        k = int(sub.iloc[0].n_sites)
        thresh = get_threshold(k)
        if thresh is None: continue
        if method == 'random':
            msub = sub[sub.category == 'random']
        else:
            msub = sub[sub.method == method]
        for _, r in msub.iterrows():
            if pd.notna(r.cPHS):
                total += 1
                if r.cPHS >= thresh: passes += 1
    return (passes / total * 100) if total > 0 else 0

print(f"Loaded {n_tumors} tumors, {len(ALL_RECON_METHODS)} methods, HAS_LIK={HAS_LIK}")
print(f"Reliable low-q: {n_reliable_lowq}")

# ── PLOT ──
n_cols = 2
n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 7 * n_rows))
x = np.arange(len(PLOT2_ORDER))

# ── Row 1, Left: cPHS pass rate — all tumors ──
ax = axes[0, 0]
pass_rates = [(_pass_rate_kdep(m, set(tumors))
               if len(_get_vals(m, 'cPHS')) else 0) for m in PLOT2_ORDER]
bars = ax.bar(x, pass_rates, color=PLOT2_COLORS, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(PLOT2_LABELS, rotation=45)
ax.set_ylabel("% tumors passing cPHS (k-dep threshold)")
ax.set_ylim(0, 110)
ax.set_title(f"cPHS pass rate — all {n_tumors} tumors\n(k-dependent threshold; higher = better)",
             fontweight='bold')
for bar, rate in zip(bars, pass_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{rate:.0f}%', ha='center', va='bottom', fontsize=12)

# ── Row 1, Right: cPHS pass rate — reliable low-q only ──
ax = axes[0, 1]
pass_rates_lq = [_pass_rate_kdep(m, reliable_lowq_set) for m in PLOT2_ORDER]
bars = ax.bar(x, pass_rates_lq, color=PLOT2_COLORS, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(PLOT2_LABELS, rotation=45)
ax.set_ylabel("% tumors passing cPHS (k-dep threshold)")
ax.set_ylim(0, 110)
ax.set_title(f"cPHS pass rate — reliable low-q (n={n_reliable_lowq})\n"
             f"(k-dependent threshold; higher = better)", fontweight='bold')
for bar, rate in zip(bars, pass_rates_lq):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{rate:.0f}%', ha='center', va='bottom', fontsize=12)

# ── Row 2, Left: Parsimony normalized ──
ax = axes[1, 0]
pn_m, pn_s = [], []
for m in PLOT2_ORDER:
    vals = _get_vals(m, 'pars_norm')
    pn_m.append(vals.mean() if len(vals) else 0)
    pn_s.append(vals.std() if len(vals) else 0)
ax.bar(x, pn_m, yerr=pn_s, capsize=3, color=PLOT2_COLORS, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(PLOT2_LABELS, rotation=45)
ax.set_ylabel(r"1 $-$ parsimony/max  ($\uparrow$ = better)")
ax.set_title(f"Parsimony normalized — all {n_tumors} tumors\n(higher = better, 0–1 per tumor)",
             fontweight='bold')
ax.set_ylim(0, 1.15)

# ── Row 2, Right: Likelihood normalized ──
if HAS_LIK:
    ax = axes[1, 1]
    ln_m, ln_s = [], []
    for m in PLOT2_ORDER:
        vals = _get_vals(m, 'lik_norm')
        ln_m.append(vals.mean() if len(vals) else 0)
        ln_s.append(vals.std() if len(vals) else 0)
    ax.bar(x, ln_m, yerr=ln_s, capsize=3, color=PLOT2_COLORS, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(PLOT2_LABELS, rotation=45)
    ax.set_ylabel(r"(lik $-$ min) / (max $-$ min)  ($\uparrow$ = better)")
    ax.set_title(f"Likelihood normalized — all {n_tumors} tumors\n(higher = better, 0–1 per tumor)",
                 fontweight='bold')
    ax.set_ylim(0, 1.15)
else:
    axes[1, 1].axis('off')

fig.suptitle(f"Method ranking ({n_tumors} analyzed tumors)",
             fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = os.path.join(PLOT_DIR, 'Fig6_method_ranking.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
