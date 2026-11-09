#!/usr/bin/env python3
"""
NNI v2 perturbation analysis plots (edge-normalized).

Reads from ~/nni_analysis_v2/*.csv
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 15,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12})

# ── Paths ──
NNI_DIR = '/home/projects/nyosef/yaaraneu/nni_analysis_v2'
MERGED_CSV = '/home/projects/nyosef/yaaraneu/KP_results_20260311_022408/merged_all_tumors.csv'
OUTPUT_DIR = '/home/projects/nyosef/yaaraneu/nni_v2_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXCLUDED = {'3430_NT_T1', '3434_NT_T1', '3777_Apc_T5'}
Q_THRESHOLD = 0.35
K_FLAG_CUTOFF = 15

def get_threshold(k):
    if k <= K_FLAG_CUTOFF: return 0.05
    elif k <= 24: return 1e-3
    else: return 1e-4

# ══════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════

print("Loading NNI v2 results...")
all_dfs = []
for f in sorted(glob.glob(f'{NNI_DIR}/*.csv')):
    all_dfs.append(pd.read_csv(f))

nni = pd.concat(all_dfs, ignore_index=True)
print(f"  {len(nni)} rows, {nni.tumor.nunique()} tumors")

# Load tumor metadata
merged = pd.read_csv(MERGED_CSV)
merged = merged[~merged.tumor.isin(EXCLUDED)]

tumor_info = {}
for tumor in nni.tumor.unique():
    info = merged[merged.tumor == tumor]
    if len(info) == 0: continue
    info = info.iloc[0]
    n = int(info.n_cells)
    k = int(info.n_sites)
    q = info.q
    t = get_threshold(k)
    tumor_info[tumor] = {'n': n, 'k': k, 'q': q, 'threshold': t}

# Determine pass/fail at base
for tumor in tumor_info:
    t = tumor_info[tumor]['threshold']
    for method in ['cassiopeia', 'smj']:
        sub = nni[(nni.tumor == tumor) & (nni.method == method) & (nni.edge_frac == 0)]
        if len(sub):
            tumor_info[tumor][f'{method}_passes'] = sub.cPHS.iloc[0] >= t
            tumor_info[tumor][f'{method}_cphs0'] = sub.cPHS.iloc[0]

tumors = sorted(tumor_info.keys(), key=lambda t: tumor_info[t]['n'])
edge_fracs = sorted(nni.edge_frac.unique())
print(f"  {len(tumors)} tumors")
print(f"  Edge fraction levels: {edge_fracs}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 1a: Pass count bar chart (majority of reps)
# ══════════════════════════════════════════════════════════════════════

print("\nPlot 1a: Pass count bar chart...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for idx, method in enumerate(['cassiopeia', 'smj']):
    ax = axes[idx]
    pass_counts = []
    total_counts = []
    for ef in edge_fracs:
        count = 0; total = 0
        for tumor in tumors:
            t = tumor_info[tumor]['threshold']
            sub = nni[(nni.tumor == tumor) & (nni.method == method) & (nni.edge_frac == ef)]
            if len(sub) == 0: continue
            if (sub.cPHS >= t).mean() > 0.5:
                count += 1
            total += 1
        pass_counts.append(count)
        total_counts.append(total)

    ax.bar(range(len(edge_fracs)), pass_counts, color='#228B22', alpha=0.7,
           edgecolor='black', linewidth=0.5)
    ax.bar(range(len(edge_fracs)),
           [total_counts[i] - pass_counts[i] for i in range(len(edge_fracs))],
           bottom=pass_counts, color='#DC143C', alpha=0.4,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(edge_fracs)))
    ax.set_xticklabels([f'{ef:.0%}' for ef in edge_fracs], rotation=45)
    ax.set_xlabel('NNI perturbation (% of internal edges)')
    ax.set_ylabel('Number of tumors')
    ax.set_title(f'{method.capitalize()}\n({total_counts[0]} tumors)')
    for i, (p, tc) in enumerate(zip(pass_counts, total_counts)):
        ax.text(i, p + 0.3, str(p), ha='center', fontsize=10, fontweight='bold')

fig.suptitle('Tumors passing cPHS at each NNI level (edge-normalized)\n'
             '(tumor passes if >50% of 20 reps pass)', fontsize=16, fontweight='bold')
legend_elements = [
    plt.Rectangle((0,0), 1, 1, fc='#228B22', alpha=0.7, ec='black', label='Pass'),
    plt.Rectangle((0,0), 1, 1, fc='#DC143C', alpha=0.4, ec='black', label='Fail'),
]
fig.legend(handles=legend_elements, loc='upper right', fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '01a_pass_count.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 01a")

# ══════════════════════════════════════════════════════════════════════
# PLOT 1b: Mean fraction passing
# ══════════════════════════════════════════════════════════════════════

print("Plot 1b: Mean fraction passing...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for idx, method in enumerate(['cassiopeia', 'smj']):
    ax = axes[idx]
    mean_frac = []
    std_frac = []
    for ef in edge_fracs:
        fracs = []
        for tumor in tumors:
            t = tumor_info[tumor]['threshold']
            sub = nni[(nni.tumor == tumor) & (nni.method == method) & (nni.edge_frac == ef)]
            if len(sub) == 0: continue
            fracs.append((sub.cPHS >= t).mean())
        mean_frac.append(np.mean(fracs) if fracs else np.nan)
        std_frac.append(np.std(fracs) if fracs else np.nan)
    mean_frac = np.array(mean_frac)
    std_frac = np.array(std_frac)
    ax.bar(range(len(edge_fracs)), mean_frac, color='#4682B4', alpha=0.7,
           edgecolor='black', linewidth=0.5)
    ax.errorbar(range(len(edge_fracs)), mean_frac, yerr=std_frac,
                fmt='none', color='black', capsize=3)
    ax.set_xticks(range(len(edge_fracs)))
    ax.set_xticklabels([f'{ef:.0%}' for ef in edge_fracs], rotation=45)
    ax.set_xlabel('NNI perturbation (% of internal edges)')
    ax.set_ylabel('Mean fraction of reps passing')
    ax.set_title(f'{method.capitalize()}')
    ax.set_ylim(0, 1.05)
    for i, m in enumerate(mean_frac):
        if not np.isnan(m):
            ax.text(i, m + 0.03, f'{m:.2f}', ha='center', fontsize=9)

fig.suptitle('Mean fraction of reps passing cPHS (edge-normalized)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '01b_mean_frac.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 01b")

# ══════════════════════════════════════════════════════════════════════
# PLOT 1c: Total pass percentage (pooled)
# ══════════════════════════════════════════════════════════════════════

print("Plot 1c: Total pass pct pooled...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for idx, method in enumerate(['cassiopeia', 'smj']):
    ax = axes[idx]
    pct_passing = []
    total_trees = []
    for ef in edge_fracs:
        n_pass = 0; n_total = 0
        for tumor in tumors:
            t = tumor_info[tumor]['threshold']
            sub = nni[(nni.tumor == tumor) & (nni.method == method) & (nni.edge_frac == ef)]
            n_pass += (sub.cPHS >= t).sum()
            n_total += len(sub)
        pct_passing.append(n_pass / n_total * 100 if n_total > 0 else 0)
        total_trees.append(n_total)
    ax.bar(range(len(edge_fracs)), pct_passing, color='#4682B4', alpha=0.7,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(edge_fracs)))
    ax.set_xticklabels([f'{ef:.0%}' for ef in edge_fracs], rotation=45)
    ax.set_xlabel('NNI perturbation (% of internal edges)')
    ax.set_ylabel('% of trees passing cPHS')
    ax.set_title(f'{method.capitalize()}\n({total_trees[0]} trees)')
    ax.set_ylim(0, 105)
    for i, p in enumerate(pct_passing):
        ax.text(i, p + 1.5, f'{p:.0f}%', ha='center', fontsize=9, fontweight='bold')

fig.suptitle('Total % of trees passing cPHS (pooled across tumors × reps, edge-normalized)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '01c_total_pct.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 01c")

# ══════════════════════════════════════════════════════════════════════
# PLOT 2: Mean distance metrics
# ══════════════════════════════════════════════════════════════════════

print("Plot 2: Mean distances...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax_idx, (metric, ylabel) in enumerate([
    ('rf_to_base', 'RF distance to original'),
    ('triplets_to_base', 'Triplets correct vs original'),
    ('likelihood', 'Likelihood'),
]):
    ax = axes[ax_idx]
    for method, color, ls in [('cassiopeia', '#1f77b4', '-'), ('smj', '#ff7f0e', '--')]:
        means = []; stds = []
        for ef in edge_fracs:
            vals = nni[(nni.method == method) & (nni.edge_frac == ef)][metric].dropna()
            means.append(vals.mean()); stds.append(vals.std())
        means = np.array(means); stds = np.array(stds)
        ax.plot(edge_fracs, means, color=color, ls=ls, marker='o', ms=5, lw=2, label=method)
        ax.fill_between(edge_fracs, means - stds, means + stds, alpha=0.15, color=color)
    ax.set_xlabel('NNI perturbation (% of internal edges)')
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '02_mean_distances.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 02")

# ══════════════════════════════════════════════════════════════════════
# PLOT 3: Normalized parsimony
# ══════════════════════════════════════════════════════════════════════

print("Plot 3: Normalized parsimony...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for method, color, ls in [('cassiopeia', '#1f77b4', '-'), ('smj', '#ff7f0e', '--')]:
    norm_means = []; norm_stds = []
    for ef in edge_fracs:
        all_norms = []
        for tumor in tumors:
            base = nni[(nni.tumor == tumor) & (nni.method == method) & (nni.edge_frac == 0)]['parsimony'].dropna()
            pert = nni[(nni.tumor == tumor) & (nni.method == method) & (nni.edge_frac == ef)]['parsimony'].dropna()
            if len(base) == 0 or len(pert) == 0: continue
            base_val = base.iloc[0]
            if base_val == 0: continue
            all_norms.extend((pert.values - base_val) / base_val * 100)
        norm_means.append(np.mean(all_norms) if all_norms else np.nan)
        norm_stds.append(np.std(all_norms) if all_norms else np.nan)
    norm_means = np.array(norm_means); norm_stds = np.array(norm_stds)
    ax.plot(edge_fracs, norm_means, color=color, ls=ls, marker='o', ms=5, lw=2, label=method)
    ax.fill_between(edge_fracs, norm_means - norm_stds, norm_means + norm_stds, alpha=0.15, color=color)

ax.set_xlabel('NNI perturbation (% of internal edges)')
ax.set_ylabel('Parsimony increase (%)')
ax.set_title('Parsimony degradation under NNI (edge-normalized)')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
ax.axhline(0, color='gray', ls=':', alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '03_norm_parsimony.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 03")

# ══════════════════════════════════════════════════════════════════════
# PLOT 4: cPHS crossing — fraction of reps passing per tumor
# ══════════════════════════════════════════════════════════════════════

print("Plot 4: cPHS crossing...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for idx, method in enumerate(['cassiopeia', 'smj']):
    ax = axes[idx]
    passing = [t for t in tumors if tumor_info[t].get(f'{method}_passes', False)]

    for i, tumor in enumerate(passing):
        t = tumor_info[tumor]['threshold']
        n = tumor_info[tumor]['n']
        frac_pass = []
        for ef in edge_fracs:
            sub = nni[(nni.tumor == tumor) & (nni.method == method) & (nni.edge_frac == ef)]
            frac_pass.append((sub.cPHS >= t).mean() if len(sub) else np.nan)
        color = plt.cm.viridis(i / max(len(passing) - 1, 1))
        ax.plot(edge_fracs, frac_pass, color=color, marker='o', ms=4, lw=1.5, alpha=0.7,
                label=f'{tumor} (n={n})')

    ax.set_xlabel('NNI perturbation (% of internal edges)')
    ax.set_ylabel('Fraction of reps passing cPHS')
    ax.set_title(f'{method.capitalize()}\n({len(passing)} originally passing)\n'
                 f'Each line = 1 tumor, y = fraction of 20 reps passing')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color='red', ls='--', alpha=0.4, label='50% threshold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc='lower left')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '04_cphs_crossing.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 04")

# ══════════════════════════════════════════════════════════════════════
# PLOT 5: Per-tumor cPHS overlay (20 reps)
# ══════════════════════════════════════════════════════════════════════

print("Plot 5: Per-tumor cPHS overlay...")

# Pick representative tumors
passing_any = [t for t in tumors if tumor_info[t].get('cassiopeia_passes', False)
               or tumor_info[t].get('smj_passes', False)]
failing_all = [t for t in tumors if not tumor_info[t].get('cassiopeia_passes', False)
               and not tumor_info[t].get('smj_passes', False)]

sample = []
if len(passing_any) >= 1: sample.append(passing_any[0])
if len(passing_any) >= 3: sample.append(passing_any[len(passing_any)//3])
if len(passing_any) >= 2: sample.append(passing_any[2*len(passing_any)//3])
if len(passing_any) >= 1: sample.append(passing_any[-1])
if len(failing_all) >= 1: sample.append(failing_all[0])
if len(failing_all) >= 2: sample.append(failing_all[len(failing_all)//2])
seen = set()
sample = [t for t in sample if t not in seen and not seen.add(t)][:6]
print(f"  Samples: {[(t, tumor_info[t]['n']) for t in sample]}")

fig, axes = plt.subplots(len(sample), 2, figsize=(16, 4 * len(sample)), squeeze=False)

for row, tumor in enumerate(sample):
    info = tumor_info[tumor]
    t = info['threshold']
    for col, method in enumerate(['cassiopeia', 'smj']):
        ax = axes[row, col]
        sub = nni[(nni.tumor == tumor) & (nni.method == method)]
        if len(sub) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue
        for rep in sub.rep.unique():
            rd = sub[sub.rep == rep].sort_values('edge_frac')
            ax.plot(rd.edge_frac, np.log10(rd.cPHS.clip(1e-20)),
                    color='steelblue', alpha=0.2, lw=0.8)
        mean_data = sub.groupby('edge_frac').cPHS.mean()
        ax.plot(mean_data.index, np.log10(mean_data.clip(1e-20)),
                color='darkblue', lw=2.5, marker='o', ms=4, label='mean')
        ax.axhline(np.log10(t), color='red', ls='--', lw=1.5, alpha=0.7,
                    label=f'threshold ({t:.0e})')
        passes = info.get(f'{method}_passes', False)
        ax.set_title(f'{tumor} — {method}\nn={info["n"]}, q={info["q"]:.3f}, '
                     f'{"PASS" if passes else "FAIL"}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Edge fraction')
        ax.set_ylabel('log₁₀(cPHS)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '05_per_tumor_cphs.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 05")

# ══════════════════════════════════════════════════════════════════════
# PLOT 6: Per-tumor parsimony & RF overlay (20 reps)
# ══════════════════════════════════════════════════════════════════════

print("Plot 6: Per-tumor parsimony & RF overlay...")
fig, axes = plt.subplots(len(sample), 4, figsize=(24, 4 * len(sample)), squeeze=False)

for row, tumor in enumerate(sample):
    for mi, method in enumerate(['cassiopeia', 'smj']):
        sub = nni[(nni.tumor == tumor) & (nni.method == method)]
        if len(sub) == 0: continue
        for metric_idx, (metric, ylabel, c) in enumerate([
            ('parsimony', 'Parsimony', 'steelblue'),
            ('rf_to_base', 'RF to original', 'coral'),
        ]):
            ax = axes[row, mi * 2 + metric_idx]
            for rep in sub.rep.unique():
                rd = sub[sub.rep == rep].sort_values('edge_frac')
                ax.plot(rd.edge_frac, rd[metric], color=c, alpha=0.2, lw=0.8)
            mean_data = sub.groupby('edge_frac')[metric].mean()
            dark_c = 'darkblue' if c == 'steelblue' else 'darkred'
            ax.plot(mean_data.index, mean_data.values, color=dark_c, lw=2.5, marker='o', ms=4)
            ax.set_title(f'{tumor} — {method} — {ylabel}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Edge fraction')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '06_per_tumor_pars_rf.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 06")

# ══════════════════════════════════════════════════════════════════════
# PLOT 7: All tumors overlaid (faint lines + bold mean)
# ══════════════════════════════════════════════════════════════════════

print("Plot 7: All tumors overlaid...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics_config = [
    ('parsimony', 'Parsimony (% increase from base)', axes[0, 0], True),
    ('rf_to_base', 'RF distance to original', axes[0, 1], False),
    ('triplets_to_base', 'Triplets correct', axes[1, 0], False),
    ('likelihood', 'Likelihood', axes[1, 1], False),
]

for metric, ylabel, ax, normalize in metrics_config:
    for method, color in [('cassiopeia', '#1f77b4'), ('smj', '#ff7f0e')]:
        # Individual tumor lines
        for tumor in tumors:
            sub = nni[(nni.tumor == tumor) & (nni.method == method)]
            if len(sub) == 0: continue
            base_val = sub[sub.edge_frac == 0][metric].mean()
            if np.isnan(base_val): continue
            if normalize and base_val == 0: continue
            mean_per_ef = sub.groupby('edge_frac')[metric].mean()
            if normalize:
                norm = (mean_per_ef - base_val) / abs(base_val) * 100
            else:
                norm = mean_per_ef
            ax.plot(norm.index, norm.values, color=color, alpha=0.15, lw=0.8)

        # Grand mean
        all_norms = []
        for tumor in tumors:
            sub = nni[(nni.tumor == tumor) & (nni.method == method)]
            if len(sub) == 0: continue
            base_val = sub[sub.edge_frac == 0][metric].mean()
            if np.isnan(base_val): continue
            if normalize and base_val == 0: continue
            mean_per_ef = sub.groupby('edge_frac')[metric].mean()
            if normalize:
                norm = (mean_per_ef - base_val) / abs(base_val) * 100
            else:
                norm = mean_per_ef
            all_norms.append(norm)
        if all_norms:
            combined = pd.concat(all_norms, axis=1)
            grand_mean = combined.mean(axis=1)
            ls = '-' if method == 'cassiopeia' else '--'
            ax.plot(grand_mean.index, grand_mean.values, color=color, lw=3, ls=ls,
                    marker='o', ms=5, label=method)

    ax.set_xlabel('NNI perturbation (% of internal edges)')
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '07_all_tumors_overlaid.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 07")

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"Summary")
print(f"{'='*60}")
print(f"Tumors: {len(tumors)}")
print(f"Edge fractions: {edge_fracs}")

for method in ['cassiopeia', 'smj']:
    passing = [t for t in tumors if tumor_info[t].get(f'{method}_passes', False)]
    print(f"\n{method}: {len(passing)}/{len(tumors)} originally pass")
    for ef in edge_fracs:
        still_pass = 0
        for tumor in passing:
            t = tumor_info[tumor]['threshold']
            sub = nni[(nni.tumor == tumor) & (nni.method == method) & (nni.edge_frac == ef)]
            if len(sub) and (sub.cPHS >= t).mean() > 0.5:
                still_pass += 1
        print(f"  After {ef:.0%} edges NNI: {still_pass}/{len(passing)} still pass")

print(f"\nPlots saved to {OUTPUT_DIR}")
