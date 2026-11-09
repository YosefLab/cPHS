#!/usr/bin/env python3
"""
Quality indicators vs cPHS performance — v2.
Each scatter plot has 2x2 layout:
  Top row: colored by pass/fail (green/red)
  Bottom row: colored by number of methods passing (gradient)
  Left column: 42 reliable low-q tumors
  Right column: 29 unreliable tumors (with k=15 and q=0.35 lines)
"""

import os, sys
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.stats import spearmanr, mannwhitneyu
import glob

# ── Paths (edit MERGED_CSV if needed) ──
MERGED_CSV = os.path.join(REPO, 'results', 'merged_all_tumors.csv')
TUMOR_STATS = os.path.join(os.environ.get("KPTRACER_DATA", os.path.expanduser("~/KPTracer-Data")), 'tumor_statistics.tsv')
TREES_DIR = os.path.join(os.environ.get("KPTRACER_DATA", os.path.expanduser("~/KPTracer-Data")), 'trees')
PLOT_DIR = os.path.join(REPO, 'output', 'figures')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Config ──
EXCLUDED = {'3430_NT_T1', '3434_NT_T1', '3777_Apc_T5'}
MIN_SITES = 7
Q_THRESHOLD = 0.35
K_FLAG_CUTOFF = 15

def get_threshold(k):
    if k <= K_FLAG_CUTOFF:
        return 0.05  # lenient reference threshold for flagged
    elif k <= 24:
        return 1e-3
    else:
        return 1e-4

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Load and merge all data
# ══════════════════════════════════════════════════════════════════════

print("Loading merged CSV...")
df = pd.read_csv(MERGED_CSV)
df = df[~df.tumor.isin(EXCLUDED)]
df = df[df.n_sites >= MIN_SITES]
tumors = sorted(df.tumor.unique())
print(f"  {len(tumors)} tumors after filtering")

print("Loading tumor_statistics.tsv...")
ts = pd.read_csv(TUMOR_STATS, sep='\t')
ts = ts.rename(columns={'Unnamed: 0': 'tumor'})
ts_dict = ts.set_index('tumor').to_dict('index')

print("Computing character matrix stats...")
cm_dict = {}
for tumor in tumors:
    cm_path = os.path.join(TREES_DIR, f'{tumor}_character_matrix.txt')
    if not os.path.exists(cm_path):
        continue
    c = pd.read_csv(cm_path, sep='\t', index_col=0, dtype=str)
    n_cm, k_cm = c.shape
    total = n_cm * k_cm
    missing = (c == '-').sum().sum()
    nunique_per_site = [c.iloc[:, j][c.iloc[:, j] != '-'].nunique() for j in range(k_cm)]
    profiles = c.apply(tuple, axis=1)
    cm_dict[tumor] = {
        'pct_missing': missing / total * 100,
        'mean_unique_states': np.mean(nunique_per_site),
        'pct_unique_cells': profiles.nunique() / n_cm * 100,
    }

# ── Build per-tumor summary ──
print("Building per-tumor table...")
methods = ['cassiopeia', 'nj', 'maxcut', 'smj', 'spectral', 'spectral_greedy']
rows = []
for tumor in tumors:
    sub = df[df.tumor == tumor]
    info = sub.iloc[0]
    n = int(info.n_cells)
    k = int(info.n_sites)
    q = info.q
    lam = info.lam
    rho = 1 - np.exp(-lam)
    genotype = info.genotype

    recon = sub[sub.category.isin(['precomputed', 'solver'])]
    best_cphs = recon.cPHS.max() if len(recon) else np.nan
    best_log_cphs = np.log10(max(best_cphs, 1e-30)) if pd.notna(best_cphs) else np.nan

    thresh = get_threshold(k)
    n_pass = sum(1 for m in methods
                 if len(sub[sub.method == m]) and sub[sub.method == m].iloc[0].cPHS >= thresh)

    # Best parsimony improvement
    rand_pars = sub[sub.category == 'random']['parsimony'].dropna()
    best_pars_imp = np.nan
    if len(rand_pars) > 0 and rand_pars.mean() > 0:
        imps = []
        for m in methods:
            ms = sub[sub.method == m]
            if len(ms) and pd.notna(ms.iloc[0].parsimony):
                imps.append((rand_pars.mean() - ms.iloc[0].parsimony) / rand_pars.mean() * 100)
        if imps:
            best_pars_imp = max(imps)

    # Derived
    k_over_n = k / n
    k_over_logn = k / np.log(n)
    k_eff = k * (1 - q)
    k_info = k * np.log(1 / q) if q > 0 else 0

    # Tumor statistics
    ts_info = ts_dict.get(tumor, {})
    pct_unique_ts = ts_info.get('PercentUnique', np.nan)
    if pd.notna(pct_unique_ts):
        pct_unique_ts *= 100
    pct_unsaturated = ts_info.get('PercentUnsaturatedTargets', np.nan)
    if pd.notna(pct_unsaturated):
        pct_unsaturated *= 100

    # Character matrix stats
    cm_info = cm_dict.get(tumor, {})

    # Classification
    is_lowq = q < Q_THRESHOLD
    is_flagged = k <= K_FLAG_CUTOFF
    is_reliable = is_lowq and not is_flagged
    group = 'reliable' if is_reliable else 'unreliable'
    any_pass = n_pass > 0

    rows.append({
        'tumor': tumor, 'n': n, 'k': k, 'q': q, 'lam': lam, 'rho': rho,
        'genotype': genotype,
        'best_cphs': best_cphs, 'best_log_cphs': best_log_cphs,
        'n_pass': n_pass, 'best_pars_imp': best_pars_imp, 'any_pass': any_pass,
        'k_over_n': k_over_n, 'k_over_logn': k_over_logn, 'k_eff': k_eff, 'k_info': k_info,
        'pct_missing': cm_info.get('pct_missing', np.nan),
        'mean_unique_states': cm_info.get('mean_unique_states', np.nan),
        'pct_unique_cells': cm_info.get('pct_unique_cells', np.nan),
        'pct_unique_ts': pct_unique_ts,
        'pct_unsaturated': pct_unsaturated,
        'group': group,
    })

tdf = pd.DataFrame(rows)
tdf.to_csv(os.path.join(PLOT_DIR, 'quality_indicators.csv'), index=False)

reliable = tdf[tdf.group == 'reliable']
unreliable = tdf[tdf.group == 'unreliable']
print(f"  {len(tdf)} tumors, {len(reliable)} reliable, {len(unreliable)} unreliable")
print(f"  Passes: {tdf.any_pass.sum()}, No pass: {(~tdf.any_pass).sum()}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Plotting helpers
# ══════════════════════════════════════════════════════════════════════

PASS_COLORS = {True: '#228B22', False: '#DC143C'}
PASS_MARKERS = {True: 'o', False: 'X'}
PASS_LABELS = {True: 'At least 1 passes', False: 'No method passes'}

# Gradient colormap for n_pass (0=red, 3=yellow, 6=green)
NPASS_CMAP = cm.RdYlGn
NPASS_NORM = Normalize(vmin=0, vmax=6)


def _scatter_pass_fail(ax, subset, xvar, yvar, annotate=True, sizes=None):
    """Scatter colored by pass/fail."""
    for status in [False, True]:
        s = subset[subset.any_pass == status]
        if len(s) == 0:
            continue
        x = s[xvar].values
        y = s[yvar].values
        mask = np.isfinite(x) & np.isfinite(y)
        if sizes is not None:
            sz = sizes.loc[s.index].values[mask]
        else:
            sz = 70
        ax.scatter(x[mask], y[mask],
                   c=PASS_COLORS[status], marker=PASS_MARKERS[status],
                   s=sz, alpha=0.8, edgecolors='black', linewidth=0.4,
                   label=f'{PASS_LABELS[status]} ({mask.sum()})', zorder=3)
        if annotate and mask.sum() > 0:
            y_valid = y[mask]
            x_valid = x[mask]
            names = s.tumor.values[mask]
            if len(y_valid) > 3:
                for ie in np.argsort(y_valid)[:2]:
                    ax.annotate(names[ie].split('_')[0] + '\n' + '_'.join(names[ie].split('_')[1:]),
                                (x_valid[ie], y_valid[ie]),
                                fontsize=10, alpha=0.6, ha='center', va='top')


def _scatter_npass(ax, subset, xvar, yvar, sizes=None):
    """Scatter colored by number of methods passing (0-6 gradient)."""
    x = subset[xvar].values
    y = subset[yvar].values
    npass = subset.n_pass.values
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(npass)
    if sizes is not None:
        sz = sizes.loc[subset.index].values[mask]
    else:
        sz = 70
    sc = ax.scatter(x[mask], y[mask], c=npass[mask], cmap=NPASS_CMAP, norm=NPASS_NORM,
                    s=sz, alpha=0.8, edgecolors='black', linewidth=0.4, zorder=3)
    return sc


def scatter_2x2(xvar, yvar, xlabel, ylabel, title, filename, log_x=False, size_by_n=False,
                add_hlines=None, add_vlines=None, legend_loc='lower right'):
    """
    2x2 scatter plot.
    Top row: pass/fail coloring. Bottom row: n_pass gradient.
    Left: reliable. Right: unreliable.
    Colorbar for bottom row is placed below the figure.
    """
    fig = plt.figure(figsize=(16, 15))
    # Create gridspec: 2 rows for plots + 1 narrow row for colorbar
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.06], hspace=0.80, wspace=0.3)

    axes_top = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes_bot = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    cbar_ax = fig.add_subplot(gs[2, :])

    for col, (subset, panel_label) in enumerate([
        (reliable, f'Reliable low-q ({len(reliable)} tumors)'),
        (unreliable, f'Unreliable ({len(unreliable)} tumors)')
    ]):
        # Compute sizes if needed
        if size_by_n:
            n_vals = subset['n'].values
            sz = 20 + 280 * (n_vals - n_vals.min()) / max(n_vals.max() - n_vals.min(), 1)
            sizes_series = pd.Series(sz, index=subset.index)
        else:
            sizes_series = None

        # Top: pass/fail
        ax = axes_top[col]
        _scatter_pass_fail(ax, subset, xvar, yvar, annotate=False, sizes=sizes_series)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{panel_label}\nColored by pass/fail', fontsize=19, fontweight='bold')
        ax.tick_params(labelsize=11)
        if log_x:
            ax.set_xscale('log')

        # Reference lines for unreliable panel
        if col == 1 and yvar == 'k':
            ax.axhline(K_FLAG_CUTOFF, color='purple', ls='--', alpha=0.5,
                       label=f'k = {K_FLAG_CUTOFF}')

        if add_hlines:
            for val, color, label in add_hlines:
                ax.axhline(val, color=color, ls='--', alpha=0.4, label=label)
        if add_vlines:
            for val, color, label in add_vlines:
                ax.axvline(val, color=color, ls='--', alpha=0.4, label=label)

        ax.legend(fontsize=15, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, framealpha=0.9)

        # Bottom: n_pass gradient
        ax = axes_bot[col]
        sc = _scatter_npass(ax, subset, xvar, yvar, sizes=sizes_series)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{panel_label}\nColored by # methods passing (0–6)',
                     fontsize=19, fontweight='bold')
        ax.tick_params(labelsize=11)
        if log_x:
            ax.set_xscale('log')

        if col == 1 and yvar == 'k':
            ax.axhline(K_FLAG_CUTOFF, color='purple', ls='--', alpha=0.5)

        if add_hlines:
            for val, color, label in add_hlines:
                ax.axhline(val, color=color, ls='--', alpha=0.4)
        if add_vlines:
            for val, color, label in add_vlines:
                ax.axvline(val, color=color, ls='--', alpha=0.4)

    # Colorbar in dedicated bottom axes
    cb = fig.colorbar(cm.ScalarMappable(norm=NPASS_NORM, cmap=NPASS_CMAP),
                      cax=cbar_ax, orientation='horizontal')
    cb.set_label('# methods passing cPHS threshold', fontsize=12)
    cb.set_ticks([0, 1, 2, 3, 4, 5, 6])
    cb.ax.tick_params(labelsize=11)

    fig.suptitle(title, fontsize=21, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Scatter plots
# ══════════════════════════════════════════════════════════════════════

print("\nGenerating scatter plots...")

# 1. n vs k
scatter_2x2('n', 'k', 'Number of cells (n)', 'Number of character sites (k)',
            'Tree size vs character sites', '01_n_vs_k.png', log_x=True,
            legend_loc='upper right')

# 2. pct_missing vs best log10(cPHS)
scatter_2x2('pct_missing', 'best_log_cphs',
            '% missing data in character matrix', 'Best log₁₀(cPHS)',
            'Missing data vs cPHS performance', '02_missing_vs_cphs.png')

# 3. PercentUnique vs best log10(cPHS)
scatter_2x2('pct_unique_ts', 'best_log_cphs',
            '% cells with unique indel profile (from KPTracer QC)', 'Best log₁₀(cPHS)',
            'Cell profile uniqueness vs cPHS performance', '03_unique_vs_cphs.png')

# 4. PercentUnsaturatedTargets vs best log10(cPHS)
scatter_2x2('pct_unsaturated', 'best_log_cphs',
            '% unsaturated target sites', 'Best log₁₀(cPHS)',
            'Target site saturation vs cPHS performance', '04_unsaturated_vs_cphs.png')

# 5. mean_unique_states vs best log10(cPHS)
scatter_2x2('mean_unique_states', 'best_log_cphs',
            'Mean unique alleles per character site', 'Best log₁₀(cPHS)',
            'Allele diversity vs cPHS performance', '05_diversity_vs_cphs.png')

# 6. pct_unique_cells vs best log10(cPHS)
scatter_2x2('pct_unique_cells', 'best_log_cphs',
            '% cells with unique character profile', 'Best log₁₀(cPHS)',
            'Cell distinguishability vs cPHS performance', '06_unique_cells_vs_cphs.png')

# 7. k/log(n) vs best log10(cPHS)
scatter_2x2('k_over_logn', 'best_log_cphs',
            'k / log(n)', 'Best log₁₀(cPHS)',
            'Information sufficiency ratio vs cPHS performance', '07_klogn_vs_cphs.png')

# 8. k_eff vs best log10(cPHS)
scatter_2x2('k_eff', 'best_log_cphs',
            'k · (1 − q)  (effective characters)', 'Best log₁₀(cPHS)',
            'Effective characters vs cPHS performance', '08_keff_vs_cphs.png')

# 8b. k/n vs best log10(cPHS) — the key predictor
scatter_2x2('k_over_n', 'best_log_cphs',
            'k / n  (characters per cell)', 'Best log₁₀(cPHS)',
            'Characters per cell vs cPHS performance', '08b_kn_vs_cphs.png', size_by_n=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Correlation heatmap (improved layout)
# ══════════════════════════════════════════════════════════════════════

print("\nGenerating correlation heatmap...")

quality_vars = ['n', 'k', 'q', 'rho', 'pct_missing', 'mean_unique_states',
                'pct_unique_cells', 'pct_unique_ts', 'pct_unsaturated',
                'k_over_n', 'k_over_logn', 'k_eff', 'k_info']
quality_labels = ['n (cells)', 'k (sites)', 'q (collision prob)', 'ρ (mutation prob)',
                  '% missing', 'unique alleles/site', '% unique cells (matrix)',
                  '% unique cells (KPTracer)', '% unsaturated sites',
                  'k/n', 'k/log(n)', 'k·(1−q)', 'k·log(1/q)']
outcome_vars = ['best_log_cphs', 'n_pass', 'best_pars_imp']
outcome_labels = ['best log₁₀(cPHS)', '# methods passing', 'best parsimony imp.']

fig = plt.figure(figsize=(20, 10))
gs_hm = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.7)
ax1 = fig.add_subplot(gs_hm[0, 0])
ax2 = fig.add_subplot(gs_hm[0, 1])
cbar_ax_hm = fig.add_subplot(gs_hm[0, 2])

for ax, subset, panel_title in [
    (ax1, reliable, f'Reliable low-q ({len(reliable)} tumors)'),
    (ax2, unreliable, f'Unreliable ({len(unreliable)} tumors)')
]:
    corr_matrix = np.full((len(quality_vars), len(outcome_vars)), np.nan)
    pval_matrix = np.full((len(quality_vars), len(outcome_vars)), np.nan)

    for i, qv in enumerate(quality_vars):
        for j, ov in enumerate(outcome_vars):
            x = subset[qv].values
            y = subset[ov].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 5:
                rho_val, pval = spearmanr(x[mask], y[mask])
                corr_matrix[i, j] = rho_val
                pval_matrix[i, j] = pval

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(outcome_vars)))
    ax.set_xticklabels(outcome_labels, rotation=45, ha='right', fontsize=12)
    ax.set_yticks(range(len(quality_vars)))
    ax.set_yticklabels(quality_labels, fontsize=12)
    ax.set_title(panel_title, fontsize=17, fontweight='bold')

    for i in range(len(quality_vars)):
        for j in range(len(outcome_vars)):
            if np.isfinite(corr_matrix[i, j]):
                sig = pval_matrix[i, j] < 0.05
                txt = f'{corr_matrix[i, j]:.2f}'
                color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                weight = 'bold' if sig else 'normal'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=17, fontweight=weight, color=color)

# Colorbar in dedicated right axes
cb_hm = fig.colorbar(im, cax=cbar_ax_hm)
cb_hm.set_label('Spearman ρ', fontsize=13)
cb_hm.ax.tick_params(labelsize=11)
fig.suptitle('Quality indicators vs cPHS outcomes (Spearman correlation)\nBold = p < 0.05',
             fontsize=18, fontweight='bold', y=0.98)
fig.savefig(os.path.join(PLOT_DIR, '09_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 09_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Box plots (3 columns × 3 rows per panel = 2 panels)
# ══════════════════════════════════════════════════════════════════════

print("\nGenerating box plots...")

box_vars = ['n', 'k', 'q', 'pct_missing', 'mean_unique_states',
            'pct_unique_cells', 'pct_unsaturated', 'k_over_logn', 'k_eff']
box_labels = ['n (cells)', 'k (sites)', 'q (collision prob)',
              '% missing data', 'unique alleles/site',
              '% unique cells', '% unsaturated sites', 'k / log(n)', 'k · (1−q)']

n_vars = len(box_vars)
ncols = 3
nrows = int(np.ceil(n_vars / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))

subset = reliable
panel_title = 'Reliable low-q tumors'
passes = subset[subset.any_pass]
no_pass = subset[~subset.any_pass]

for vi, (var, label) in enumerate(zip(box_vars, box_labels)):
    row = vi // ncols
    col = vi % ncols
    ax = axes[row, col] if nrows > 1 else axes[col]

    data_pass = passes[var].dropna().values
    data_nopass = no_pass[var].dropna().values

    box_data = []
    box_labels_plot = []
    box_colors = []
    if len(data_pass) > 0:
        box_data.append(data_pass)
        box_labels_plot.append(f'passes\n({len(data_pass)})')
        box_colors.append('#228B22')
    if len(data_nopass) > 0:
        box_data.append(data_nopass)
        box_labels_plot.append(f'no pass\n({len(data_nopass)})')
        box_colors.append('#DC143C')

    if len(box_data) > 0:
        bp = ax.boxplot(box_data, tick_labels=box_labels_plot, patch_artist=True)
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.3)

    ax.set_title(label, fontsize=17, fontweight='bold')
    ax.tick_params(labelsize=12)

    # Mann-Whitney p-value
    if len(data_pass) >= 3 and len(data_nopass) >= 3:
        try:
            _, pval = mannwhitneyu(data_pass, data_nopass, alternative='two-sided')
            sig = '**' if pval < 0.01 else ('*' if pval < 0.05 else '')
            ax.text(0.5, 0.97, f'p={pval:.3f}{sig}', transform=ax.transAxes,
                    ha='center', va='top', fontsize=18,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))
        except Exception:
            pass



# Hide empty subplots
for vi in range(n_vars, nrows * ncols):
    row = vi // ncols
    col = vi % ncols
    axes[row, col].axis('off')

fig.suptitle('Quality indicators: passes vs no-pass (42 reliable low-q tumors)\n(* p<0.05, ** p<0.01)',
             fontsize=19, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(PLOT_DIR, '10_boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 10_boxplots.png")


# ══════════════════════════════════════════════════════════════════════
# STEP 6: Print summary
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY — Reliable low-q tumors")
print("=" * 70)

r_pass = reliable[reliable.any_pass]
r_nopass = reliable[~reliable.any_pass]

print(f"\nReliable: {len(reliable)} tumors ({len(r_pass)} pass, {len(r_nopass)} no pass)")

all_vars = quality_vars + ['k_over_n']
print(f"\n{'Variable':<25s} {'Passes med':<14s} {'No pass med':<14s} {'Spearman r':<12s} {'p-value':<10s}")
print("-" * 75)

for var in all_vars:
    p_med = r_pass[var].median()
    np_med = r_nopass[var].median()
    x = reliable[var].values
    y = reliable['best_log_cphs'].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 5:
        rho_val, pval = spearmanr(x[mask], y[mask])
        sig = '**' if pval < 0.01 else ('*' if pval < 0.05 else '')
        print(f"{var:<25s} {p_med:<14.3f} {np_med:<14.3f} {rho_val:<12.3f} {pval:<10.4f}{sig}")
    else:
        print(f"{var:<25s} {p_med:<14.3f} {np_med:<14.3f} {'n/a':<12s} {'n/a':<10s}")

print(f"\nPlots saved to: {PLOT_DIR}")
print("Done.")
