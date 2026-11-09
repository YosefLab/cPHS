#!/usr/bin/env python3
"""
Plot simulation pass rate as function of (n, q),
with KP tumors overlaid as pass/fail per reconstruction method.

One plot per method × k value.
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

# Global font size increase
plt.rcParams.update({'font.size': 20, 'axes.titlesize': 22, 'axes.labelsize': 20,
                     'xtick.labelsize': 17, 'ytick.labelsize': 17})

# ── Paths ──
RESULTS_DIR = '/home/projects/nyosef/yaaraneu/threshold_calibration/results'
MERGED_CSV = '/home/projects/nyosef/yaaraneu/KP_results_20260311_022408/merged_all_tumors.csv'
OUTPUT_DIR = '/home/projects/nyosef/yaaraneu/sim_vs_kp_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXCLUDED = {'3430_NT_T1', '3434_NT_T1', '3777_Apc_T5'}
Q_THRESHOLD = 0.35
K_FLAG_CUTOFF = 15

def get_threshold(k):
    if k <= K_FLAG_CUTOFF: return 0.05
    elif k <= 24: return 1e-3
    else: return 1e-4

# ══════════════════════════════════════════════════════════════════════
# Load simulation data
# ══════════════════════════════════════════════════════════════════════

print("Loading simulation results...")
sim_rows = []

for f in sorted(glob.glob(f'{RESULTS_DIR}/threshold_n*_m*_rho9.csv')):
    try:
        df = pd.read_csv(f).drop_duplicates()
        n = int(df.n.iloc[0]); k = int(df.k.iloc[0]); m = int(df.m.iloc[0])
        if k == 25: continue
        reps = df.rep.nunique()
        q_approx = 1.0 / m
        thresh = 1e-4 if k >= 25 else 1e-3
        pass_rate = (df.cPHS >= thresh).sum() / len(df)
        sim_rows.append({'n': n, 'k': k, 'm': m, 'q': q_approx,
                         'pass_rate': pass_rate, 'reps': reps})
    except:
        pass

for f in sorted(glob.glob(f'{RESULTS_DIR}/threshold_n*_k[23]0_rho9.csv')):
    try:
        df = pd.read_csv(f).drop_duplicates()
        n = int(df.n.iloc[0]); k = int(df.k.iloc[0])
        if k not in [20, 30]: continue
        reps = df.rep.nunique()
        thresh = 1e-4 if k >= 25 else 1e-3
        pass_rate = (df.cPHS >= thresh).sum() / len(df)
        sim_rows.append({'n': n, 'k': k, 'm': 50, 'q': 0.02,
                         'pass_rate': pass_rate, 'reps': reps})
    except:
        pass

# Also add m=50 files that have _m50_ in name
for f in sorted(glob.glob(f'{RESULTS_DIR}/threshold_n*_k*_m50_rho9.csv')):
    try:
        df = pd.read_csv(f).drop_duplicates()
        n = int(df.n.iloc[0]); k = int(df.k.iloc[0]); m = int(df.m.iloc[0])
        if k not in [20, 30]: continue
        reps = df.rep.nunique()
        thresh = 1e-4 if k >= 25 else 1e-3
        pass_rate = (df.cPHS >= thresh).sum() / len(df)
        sim_rows.append({'n': n, 'k': k, 'm': m, 'q': 0.02,
                         'pass_rate': pass_rate, 'reps': reps})
    except:
        pass

sim_df = pd.DataFrame(sim_rows)
sim_df = sim_df[sim_df.n <= 2000]
# Dedup (m=50 might appear twice from different file patterns)
sim_df = sim_df.drop_duplicates(subset=['n', 'k', 'm'])
print(f"  {len(sim_df)} simulation configurations loaded")

# ══════════════════════════════════════════════════════════════════════
# Load KP tumor data
# ══════════════════════════════════════════════════════════════════════

print("Loading KP tumor data...")
df_all = pd.read_csv(MERGED_CSV)
df_all = df_all[~df_all.tumor.isin(EXCLUDED)]
df_all = df_all[df_all.n_sites >= 7]

all_methods = ['cassiopeia', 'nj', 'maxcut', 'smj', 'spectral', 'spectral_greedy']

tumor_rows = []
for tumor in sorted(df_all.tumor.unique()):
    sub = df_all[df_all.tumor == tumor]
    info = sub.iloc[0]
    n = int(info.n_cells)
    k = int(info.n_sites)
    q = info.q
    thresh = get_threshold(k)
    is_reliable = q < Q_THRESHOLD and k > K_FLAG_CUTOFF

    method_pass = {}
    for m in all_methods:
        msub = sub[sub.method == m]
        if len(msub):
            method_pass[m] = msub.iloc[0].cPHS >= thresh
        else:
            method_pass[m] = False

    tumor_rows.append({
        'tumor': tumor, 'n': n, 'k': k, 'q': q,
        'reliable': is_reliable,
        **{f'{m}_pass': v for m, v in method_pass.items()}
    })

tdf = pd.DataFrame(tumor_rows)
tdf = tdf[tdf.reliable]
print(f"  {len(tdf)} reliable tumors")

# ══════════════════════════════════════════════════════════════════════
# Color scheme for simulation
# ══════════════════════════════════════════════════════════════════════

cmap = plt.cm.RdYlGn
norm = Normalize(vmin=0, vmax=1)

# ══════════════════════════════════════════════════════════════════════
# Plot: one per method × k value
# ══════════════════════════════════════════════════════════════════════

plot_methods = ['cassiopeia', 'smj', 'nj', 'maxcut']

for k_val, thresh_val, thresh_label, k_range_label, k_lo, k_hi in [
    (30, 1e-4, r'$t = 10^{-4}$', r'$k = 25$–$35$', 25, 35),
    (20, 1e-3, r'$t = 10^{-3}$', r'$k = 16$–$24$', 16, 24),
]:
    sim_k = sim_df[sim_df.k == k_val]
    kp = tdf[(tdf.k >= k_lo) & (tdf.k <= k_hi)]

    if len(kp) == 0:
        continue

    for method in plot_methods:
        pass_col = f'{method}_pass'
        if pass_col not in kp.columns:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(13, 8))

        # ── Simulation squares ──
        for _, row in sim_k.iterrows():
            color = cmap(norm(row.pass_rate))
            ax.scatter(row.n, row.q, c=[color], s=350, marker='s',
                       edgecolors='black', linewidth=1.0, zorder=2)
            ax.annotate(f'{row.pass_rate:.0%}', (row.n, row.q),
                        fontsize=13, ha='center', va='center',
                        fontweight='bold', zorder=3)

        # ── KP tumors: pass/fail for this method ──
        for _, row in kp.iterrows():
            passes = row[pass_col]
            if passes:
                color = '#228B22'
                marker = 'o'
            else:
                color = '#DC143C'
                marker = 'X'
            sz = 40 + 200 * (row.n - kp.n.min()) / max(kp.n.max() - kp.n.min(), 1)
            ax.scatter(row.n, row.q, c=color, marker=marker, s=sz,
                       edgecolors='black', linewidth=1.0, alpha=0.85, zorder=4)

        # ── Formatting ──
        ax.set_xscale('log')
        ax.set_xlabel('Number of cells (n)', fontsize=19)
        ax.set_ylabel('Collision probability (q)', fontsize=19)
        ax.set_title(f'{method.capitalize()} — cPHS pass rate\n'
                     f'Simulation: k={k_val}, ρ=0.9, {thresh_label}  |  '
                     f'KP tumors: {k_range_label}',
                     fontsize=18, fontweight='bold')
        ax.tick_params(labelsize=15)

        # q=0.35 line
        ax.axhline(Q_THRESHOLD, color='red', ls='--', alpha=0.4, lw=1.5)

        # ── Colorbar ──
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Simulation pass rate', fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        # ── Legend below the figure ──
        n_pass = kp[pass_col].sum()
        n_fail = len(kp) - n_pass
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#999999',
                   markersize=14, markeredgecolor='black', markeredgewidth=1.0,
                   label='Simulation (pass rate labeled)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#228B22',
                   markersize=11, markeredgecolor='black', markeredgewidth=1.0,
                   label=f'KP tumor passes ({n_pass})'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor='#DC143C',
                   markersize=11, markeredgecolor='black', markeredgewidth=1.0,
                   label=f'KP tumor fails ({n_fail})'),
            Line2D([0], [0], ls='', marker='o', color='w', markerfacecolor='gray',
                   markersize=7, markeredgecolor='black',
                   label='Circle size ∝ n'),
            Line2D([0], [0], ls='--', color='red', alpha=0.4,
                   label=f'q = {Q_THRESHOLD} cutoff'),
        ]
        fig.legend(handles=legend_elements, loc='lower center',
                   ncol=3, fontsize=15, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        fname = f'sim_vs_kp_k{k_val}_{method}.png'
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

# ══════════════════════════════════════════════════════════════════════
# Also make a combined 2×2 plot for k=30 (main methods)
# ══════════════════════════════════════════════════════════════════════

print("\nMaking combined 2x2 plots...")

for k_val, thresh_val, thresh_label, k_range_label, k_lo, k_hi in [
    (30, 1e-4, r'$t = 10^{-4}$', r'$k = 25$–$35$', 25, 35),
    (20, 1e-3, r'$t = 10^{-3}$', r'$k = 16$–$24$', 16, 24),
]:
    sim_k = sim_df[sim_df.k == k_val]
    kp = tdf[(tdf.k >= k_lo) & (tdf.k <= k_hi)]

    if len(kp) == 0:
        print(f"  No KP tumors for k={k_val} range, skipping combined")
        continue

    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    axes = axes.flatten()

    for idx, method in enumerate(['cassiopeia', 'smj', 'nj', 'maxcut']):
        ax = axes[idx]
        pass_col = f'{method}_pass'

        # Simulation squares
        for _, row in sim_k.iterrows():
            color = cmap(norm(row.pass_rate))
            ax.scatter(row.n, row.q, c=[color], s=350, marker='s',
                       edgecolors='black', linewidth=0.8, zorder=2)
            ax.annotate(f'{row.pass_rate:.0%}', (row.n, row.q),
                        fontsize=13, ha='center', va='center',
                        fontweight='bold', zorder=3)

        # KP tumors
        for _, row in kp.iterrows():
            passes = row[pass_col]
            color = '#228B22' if passes else '#DC143C'
            marker = 'o' if passes else 'X'
            sz = 30 + 150 * (row.n - kp.n.min()) / max(kp.n.max() - kp.n.min(), 1)
            ax.scatter(row.n, row.q, c=color, marker=marker, s=sz,
                       edgecolors='black', linewidth=0.8, alpha=0.85, zorder=4)

        ax.set_xscale('log')
        ax.set_xlabel('Number of cells (n)', fontsize=17)
        ax.set_ylabel('Collision probability (q)', fontsize=17)
        n_pass = kp[pass_col].sum()
        ax.set_title(f'{method.capitalize()} ({n_pass}/{len(kp)} pass)',
                     fontsize=18, fontweight='bold')
        ax.tick_params(labelsize=14)
        ax.axhline(Q_THRESHOLD, color='red', ls='--', alpha=0.4, lw=1)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Simulation pass rate', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # Legend below
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#999999',
               markersize=13, markeredgecolor='black', label='Simulation'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#228B22',
               markersize=11, markeredgecolor='black', label='KP passes'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#DC143C',
               markersize=11, markeredgecolor='black', label='KP fails'),
        Line2D([0], [0], ls='--', color='red', alpha=0.4, label=f'q = {Q_THRESHOLD}'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=15, bbox_to_anchor=(0.45, -0.01))

    fig.suptitle(f'cPHS pass rate: simulation vs KP tumors ({k_range_label}, {thresh_label}, ρ=0.9)',
                 fontsize=20, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.05, 0.91, 0.98])
    fname = f'sim_vs_kp_k{k_val}_combined.png'
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# ══════════════════════════════════════════════════════════════════════
# Print summary tables
# ══════════════════════════════════════════════════════════════════════

for k_val in [30, 20]:
    sim_k = sim_df[sim_df.k == k_val]
    print(f"\n=== Pass rate table (k={k_val}) ===")
    m_vals = sorted(sim_k.m.unique(), reverse=True)
    print(f"{'n':>6s}", end='')
    for m_val in m_vals:
        print(f'  m={m_val}(q≈{1/m_val:.2f})', end='')
    print()
    for n_val in sorted(sim_k.n.unique()):
        print(f'{n_val:6d}', end='')
        for m_val in m_vals:
            row = sim_k[(sim_k.n == n_val) & (sim_k.m == m_val)]
            if len(row):
                print(f'  {row.iloc[0].pass_rate:>14.0%}', end='')
            else:
                print(f'  {"—":>14s}', end='')
        print()

print(f"\nAll plots saved to {OUTPUT_DIR}")
