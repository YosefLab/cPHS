#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import subprocess

DATA = os.environ.get("KPTRACER_DATA", os.path.expanduser("~/KPTracer-Data"))
if not os.path.isdir(os.path.join(DATA, "trees")):
    # Download the KPTracer dataset from Zenodo
    url = "https://zenodo.org/records/5847462/files/KPTracer-Data.tar.gz?download=1"
    output_path = os.path.join(os.path.dirname(DATA), "KPTracer-Data.tar.gz")

    print("Downloading KPTracer data (~1.3 GB)...")
    subprocess.run(["wget", "-q", "--show-progress", url, "-O", output_path], check=True)

    # Extract
    print("Extracting...")
    subprocess.run(["tar", "-xzf", output_path, "-C", os.path.dirname(DATA)], check=True)

    # Remove the archive to save space
    os.remove(output_path)

    # Show what was extracted
    for root, dirs, files in os.walk(DATA):
        level = root.replace(DATA, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:  # only show files for first 2 levels
            subindent = " " * 2 * (level + 1)
            for file in files:
                size_mb = os.path.getsize(os.path.join(root, file)) / 1e6
                print(f"{subindent}{file} ({size_mb:.1f} MB)")

# In[ ]:





# In[2]:


# ── Size filter ──
MIN_LEAVES = 1       # minimum cells to include a tumor
MAX_LEAVES = 1000     # maximum cells to include a tumor


# In[3]:


# ── Analysis parameters ──
N_RANDOM = 5                    # random trees per tumor
EXTRA_SOLVERS = ["maxcut", "spectral", "spectral_greedy", "smj"]
PERTURB_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50]

# ── Output file ──
CSV_FILE = f"phs_comparison_{MIN_LEAVES}to{MAX_LEAVES}.csv"

# ── Skip recomputation if CSV exists? ──
SKIP_IF_CSV_EXISTS = True


# In[ ]:


# %% [markdown]
# # Full PHS Comparison on KPTracer Data
# 
# Resume-friendly: saves one CSV per tumor, skips already-computed tumors on restart.
# Final step merges all per-tumor files into one big CSV for plotting.

# %% Cell 1 — CONFIGURATION (edit these)

# ── Size filter ──
#MIN_LEAVES = 1       # minimum cells to include a tumor
#MAX_LEAVES = 600     # maximum cells to include a tumor

# ── Analysis parameters ──
#N_RANDOM = 5                    # random trees per tumor
#EXTRA_SOLVERS = ["maxcut", "spectral", "spectral_greedy", "smj"]
#PERTURB_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50]

# ── Output directories ──
CHECKPOINT_DIR = f"checkpoints_{MIN_LEAVES}to{MAX_LEAVES}"
MERGED_CSV = f"phs_comparison_{MIN_LEAVES}to{MAX_LEAVES}.csv"


# %% Cell 2 — Imports

import os, re, glob, time, random, pickle, warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import binom, spearmanr
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from IPython.display import display, Image

import cassiopeia as cass
from cassiopeia.tools.branch_length_estimator.IIDExponentialMLE import IIDExponentialMLE

warnings.filterwarnings('ignore', message='Objective contains too many subexpressions')
warnings.filterwarnings('ignore', message='Solution may be inaccurate')
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA_DIR = os.environ.get("KPTRACER_DATA", os.path.expanduser("~/KPTracer-Data"))
eps = 1e-17

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Config: leaves {MIN_LEAVES}–{MAX_LEAVES}, solvers={EXTRA_SOLVERS}")
print(f"Checkpoints: {CHECKPOINT_DIR}/")
print(f"Merged output: {MERGED_CSV}")


# %% Cell 3 — Core PHS functions

def calc_phs_and_lca_times(tree):
    cm = tree.character_matrix
    n, k = cm.shape
    res       = pd.DataFrame(np.zeros((n, n)) - 1, index=cm.index, columns=cm.index)
    lca_times = pd.DataFrame(np.zeros((n, n)) - 1, index=cm.index, columns=cm.index)
    for lca in tree.internal_nodes:
        lca_states = tree.get_character_states(lca)
        lca_height = tree.get_time(lca)
        children = tree.children(lca)
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                for leaf_1 in tree.leaves_in_subtree(children[i]):
                    s1 = tree.get_character_states(leaf_1)
                    for leaf_2 in tree.leaves_in_subtree(children[j]):
                        s2 = tree.get_character_states(leaf_2)
                        phs = sum(1 for c in range(k)
                                  if lca_states[c] == 0 and s1[c] > 0 and s1[c] == s2[c])
                        res.loc[leaf_1, leaf_2]       = phs
                        res.loc[leaf_2, leaf_1]       = phs
                        lca_times.loc[leaf_1, leaf_2] = lca_height
                        lca_times.loc[leaf_2, leaf_1] = lca_height
    for leaf in tree.leaves:
        res.loc[leaf, leaf] = 0
        lca_times.loc[leaf, leaf] = 0
    return res, lca_times


def calc_cPHS(tree, lam, q):
    phs, lca_times = calc_phs_and_lca_times(tree)
    n, k = tree.character_matrix.shape
    alpha = np.exp(-lam * lca_times)
    beta  = 1 - np.exp(-lam * (1 - lca_times))
    prob  = alpha * beta**2 * q
    prob[lca_times.to_numpy() == 1] = 1
    cdf = binom.cdf(phs - 1, k, prob)
    pvalues_matrix = 1 - cdf
    off_diag = ~np.eye(n, dtype=bool)
    pvalues = np.array(pvalues_matrix[off_diag])
    pvalues[pvalues == 0] = np.finfo(float).eps
    N = len(pvalues)
    pvalues_sorted = np.sort(pvalues)
    adjusted = pvalues_sorted * N / np.arange(1, N + 1)
    return float(min(adjusted))


def get_lam_and_q(tree, known_priors=True):
    cm = tree.character_matrix.to_numpy()
    proportion_mutated = np.sum(cm > 0) / np.sum(cm >= 0)
    lam = -np.log(1.0 - proportion_mutated)
    if known_priors and tree.priors is not None:
        q_per_site = []
        for site_idx in tree.priors:
            vals = np.array(list(tree.priors[site_idx].values()))
            if vals.sum() > 0:
                vals_norm = vals / vals.sum()
                q_per_site.append(np.sum(vals_norm ** 2))
        q = np.mean(q_per_site) if q_per_site else 1.0 / max(cm.max(), 2)
    else:
        q = 1.0 / tree.character_matrix.stack().nunique()
    return lam, q


# %% Cell 4 — Data-loading helpers

def clean_nj_newick(nwk):
    return re.sub(r'\)([^:;,)(]+)', ')', nwk)


def load_cm_and_priors(tumor):
    cm_raw = pd.read_csv(f"{DATA_DIR}/trees/{tumor}_character_matrix.txt",
                         sep="\t", index_col=0)
    cm = cm_raw.apply(pd.to_numeric, errors='coerce').fillna(-1).astype(int)
    with open(f"{DATA_DIR}/trees/{tumor}_priors.pkl", "rb") as f:
        priors_raw = pickle.load(f)
    priors = {si: {int(k): v for k, v in sp.items()}
              for si, sp in priors_raw.items()}
    return cm, priors


def postprocess(tree):
    tree.collapse_unifurcations()
    tree.reconstruct_ancestral_characters()
    tree.set_character_states(tree.root, [0] * tree.n_character)
    tree.scale_to_unit_length()
    IIDExponentialMLE().estimate_branch_lengths(tree)
    return tree


def load_precomputed(tumor, tree_type="cassiopeia"):
    cm, priors = load_cm_and_priors(tumor)
    if tree_type == "cassiopeia":
        nwk_path = f"{DATA_DIR}/trees/{tumor}_tree.nwk"
    else:
        nwk_path = f"{DATA_DIR}/trees/neighbor_joining/{tumor}_tree_nj.processed.tree"
    with open(nwk_path) as f:
        nwk = f.read().strip()
    if tree_type == "nj":
        nwk = clean_nj_newick(nwk)
    tree = cass.data.CassiopeiaTree(character_matrix=cm, tree=nwk, priors=priors)
    return postprocess(tree), cm, priors


# %% Cell 5 — Tree generators

def random_newick(leaves, seed=None):
    if seed is not None:
        random.seed(seed)
    nodes = list(leaves); random.shuffle(nodes)
    while len(nodes) > 1:
        i = random.randrange(len(nodes))
        j = random.randrange(len(nodes) - 1)
        if j >= i: j += 1
        a, b = nodes[i], nodes[j]
        nodes = [n for idx, n in enumerate(nodes) if idx not in (i, j)]
        nodes.append(f"({a},{b})")
    return nodes[0] + ";"


def make_random_trees(cm, priors, n_random=5):
    trees = []
    for r in range(n_random):
        nwk = random_newick(list(cm.index), seed=r)
        t = cass.data.CassiopeiaTree(character_matrix=cm.copy(),
                                     tree=nwk, priors=priors)
        trees.append(postprocess(t))
    return trees


def perturb_topology_nni(tree, n_moves, seed=42):
    rng = np.random.RandomState(seed)
    perturbed = deepcopy(tree)
    G = perturbed.get_tree_topology()
    for _ in range(n_moves):
        internal_edges = []
        for u, v in G.edges():
            ch_u = list(G.successors(u))
            ch_v = list(G.successors(v))
            if len(ch_u) >= 2 and len(ch_v) >= 1 and v in ch_u:
                internal_edges.append((u, v))
        if not internal_edges:
            break
        parent, child = internal_edges[rng.randint(len(internal_edges))]
        siblings = [s for s in G.successors(parent) if s != child]
        grandchildren = list(G.successors(child))
        if not siblings or not grandchildren:
            continue
        sib = siblings[rng.randint(len(siblings))]
        gc = grandchildren[rng.randint(len(grandchildren))]
        G.remove_edge(parent, sib)
        G.remove_edge(child, gc)
        G.add_edge(parent, gc)
        G.add_edge(child, sib)

    def _to_newick(G, node):
        children = list(G.successors(node))
        if not children:
            return str(node)
        return "(" + ",".join(_to_newick(G, c) for c in children) + ")"

    new_nwk = _to_newick(G, perturbed.root) + ";"
    new_tree = cass.data.CassiopeiaTree(
        character_matrix=tree.character_matrix.copy(),
        tree=new_nwk, priors=tree.priors)
    return postprocess(new_tree)


def make_perturbed_trees(tree, n_leaves, fractions):
    results = []
    for frac in fractions:
        n_moves = max(1, int(frac * n_leaves))
        try:
            pt = perturb_topology_nni(tree, n_moves, seed=int(frac * 10000))
            results.append((frac, pt))
        except Exception as e:
            print(f"      NNI {frac:.0%} failed: {e}")
    return results


def reconstruct_with_solver(cm, priors, solver_name):
    try:
        tree = cass.data.CassiopeiaTree(character_matrix=cm.copy(), priors=priors)
        solver_map = {
            "greedy":          cass.solver.VanillaGreedySolver(),
            "maxcut":          cass.solver.MaxCutSolver(),
            "maxcut_greedy":   cass.solver.MaxCutGreedySolver(),
            "spectral":        cass.solver.SpectralSolver(),
            "spectral_greedy": cass.solver.SpectralGreedySolver(),
            "smj":             cass.solver.SharedMutationJoiningSolver(),
            "nj":              cass.solver.NeighborJoiningSolver(add_root=True),
        }
        solver_map[solver_name].solve(tree)
        return postprocess(tree)
    except Exception as e:
        print(f"      {solver_name} failed: {e}")
        return None


# %% Cell 6 — Scoring

def score_tree(tree, lam, q):
    cphs = calc_cPHS(tree, lam, q)
    try:
        pars = cass.tools.calculate_parsimony(tree)
    except:
        pars = np.nan
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            lik = cass.tools.calculate_likelihood_continuous(tree)
        if np.isinf(lik):
            lik = np.nan
    except:
        lik = np.nan
    return cphs, pars, lik


# %% Cell 7 — Process ONE tumor

def process_tumor(tumor, n_cells):
    """Process a single tumor, return list of row dicts."""
    genotype = tumor.split("_")[1] if len(tumor.split("_")) >= 2 else "?"

    tree_cass, cm, priors = load_precomputed(tumor, "cassiopeia")
    tree_nj,   _,  _     = load_precomputed(tumor, "nj")
    n_sites = cm.shape[1]
    lam, q  = get_lam_and_q(tree_cass)

    base = dict(tumor=tumor, genotype=genotype,
                n_cells=n_cells, n_sites=n_sites, lam=lam, q=q)
    rows = []

    # Pre-computed
    c, p, l = score_tree(tree_cass, lam, q)
    rows.append({**base, 'method': 'cassiopeia', 'category': 'precomputed',
                 'cPHS': c, 'parsimony': p, 'likelihood': l})
    c2, p2, l2 = score_tree(tree_nj, lam, q)
    rows.append({**base, 'method': 'nj', 'category': 'precomputed',
                 'cPHS': c2, 'parsimony': p2, 'likelihood': l2})

    # Other solvers
    for sname in EXTRA_SOLVERS:
        t_s = reconstruct_with_solver(cm, priors, sname)
        if t_s is not None:
            cs, ps, ls = score_tree(t_s, lam, q)
            rows.append({**base, 'method': sname, 'category': 'solver',
                         'cPHS': cs, 'parsimony': ps, 'likelihood': ls})

    # Random
    for rt in make_random_trees(cm, priors, N_RANDOM):
        cr, pr, lr = score_tree(rt, lam, q)
        rows.append({**base, 'method': 'random', 'category': 'random',
                     'cPHS': cr, 'parsimony': pr, 'likelihood': lr})

    # Perturbed
    for frac, pt in make_perturbed_trees(tree_cass, n_cells, PERTURB_FRACTIONS):
        cp, pp, lp = score_tree(pt, lam, q)
        pct = int(frac * 100)
        rows.append({**base, 'method': f'perturbed_{pct}pct',
                     'category': 'perturbed',
                     'cPHS': cp, 'parsimony': pp, 'likelihood': lp})

    return rows


# %% Cell 8 — Main loop with per-tumor checkpoints
#
# KEY FEATURE: Each tumor saves its own CSV file in CHECKPOINT_DIR.
# If the notebook crashes and you restart, already-done tumors are skipped.
# Just re-run this cell to continue where you left off.

cass_set = {os.path.basename(f).replace("_tree.nwk", "")
            for f in glob.glob(f"{DATA_DIR}/trees/*_tree.nwk")}
nj_set   = {os.path.basename(f).replace("_tree_nj.processed.tree", "")
            for f in glob.glob(f"{DATA_DIR}/trees/neighbor_joining/*_tree_nj.processed.tree")}
all_tumors = sorted(cass_set & nj_set)

eligible = []
for tumor in all_tumors:
    cm_check = pd.read_csv(f"{DATA_DIR}/trees/{tumor}_character_matrix.txt",
                           sep="\t", index_col=0)
    n = len(cm_check)
    if MIN_LEAVES <= n <= MAX_LEAVES:
        eligible.append((tumor, n))

# Check which are already done
already_done = {os.path.basename(f).replace(".csv", "")
                for f in glob.glob(f"{CHECKPOINT_DIR}/*.csv")}
todo = [(t, n) for t, n in eligible if t not in already_done]

print(f"Total eligible tumors: {len(eligible)}")
print(f"Already computed:      {len(already_done)}")
print(f"Remaining:             {len(todo)}")
print("=" * 80)

if len(todo) == 0:
    print("Nothing to compute — all tumors already have checkpoints!")
    print("Skip to the merge cell (Cell 9) to build the combined CSV.")
else:
    for idx, (tumor, n_cells) in enumerate(todo):
        print(f"\n[{idx+1}/{len(todo)}] {tumor}  n={n_cells}", end="", flush=True)
        try:
            t0 = time.time()
            rows = process_tumor(tumor, n_cells)

            # Save per-tumor checkpoint
            tumor_df = pd.DataFrame(rows)
            tumor_df.to_csv(f"{CHECKPOINT_DIR}/{tumor}.csv", index=False)

            elapsed = time.time() - t0
            print(f"  ✓ {len(rows)} rows, {elapsed:.0f}s")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")

    print(f"\n{'='*80}")
    print(f"Done! Checkpoints saved to: {CHECKPOINT_DIR}/")


# %% Cell 9 — Merge all checkpoints into one CSV
#
# Run this cell after all tumors are done (or after partial runs).
# It reads every CSV in the checkpoint folder and combines them.

checkpoint_files = sorted(glob.glob(f"{CHECKPOINT_DIR}/*.csv"))
print(f"Found {len(checkpoint_files)} checkpoint files in {CHECKPOINT_DIR}/")

dfs = []
for f in checkpoint_files:
    try:
        sub = pd.read_csv(f)
        if len(sub) > 0:
            dfs.append(sub)
    except Exception as e:
        print(f"  Warning: could not read {f}: {e}")

if len(dfs) == 0:
    raise RuntimeError("No checkpoint files found! Run Cell 8 first.")

df = pd.concat(dfs, ignore_index=True)
df.to_csv(MERGED_CSV, index=False)
print(f"Merged: {len(df)} rows, {df.tumor.nunique()} tumors → {MERGED_CSV}")


# %% Cell 10 — Data summary

print(f"{'='*60}")
print(f"DATASET: {len(df)} rows, {df.tumor.nunique()} tumors")
print(f"Cell range: {df.n_cells.min()}–{df.n_cells.max()}")
print(f"Methods: {sorted(df.method.unique())}")
print(f"{'='*60}")

for cat in ['precomputed', 'solver', 'perturbed', 'random']:
    sub = df[df.category == cat]
    print(f"\n{cat}: {len(sub)} rows")
    for col in ['cPHS', 'parsimony', 'likelihood']:
        vals = sub[col].dropna()
        if len(vals):
            print(f"  {col}: mean={vals.mean():.4g}  "
                  f"median={vals.median():.4g}  "
                  f"min={vals.min():.4g}  max={vals.max():.4g}")

lik_avail = df.likelihood.notna().sum()
print(f"\nLikelihood available: {lik_avail}/{len(df)} rows")
HAS_LIK = lik_avail > 10


# %% Cell 11 — Plotting setup

tumors = sorted(df.tumor.unique())
n_tumors = len(tumors)

METHOD_COLORS = {
    'cassiopeia': 'steelblue', 'nj': 'salmon',
    'maxcut': 'darkgreen', 'spectral': 'purple',
    'spectral_greedy': 'orange', 'smj': 'brown',
}

METHOD_ORDER = (['cassiopeia', 'nj'] + EXTRA_SOLVERS +
                [f'perturbed_{int(f*100)}pct' for f in PERTURB_FRACTIONS] +
                ['random'])

BAR_LABELS = (['Cass', 'NJ'] +
              [s.replace('_', '\n') for s in EXTRA_SOLVERS] +
              [f'Pert\n{int(f*100)}%' for f in PERTURB_FRACTIONS] +
              ['Random'])

_solver_colors = ['steelblue', 'salmon', 'darkgreen', 'purple', 'orange', 'brown']
_perturb_colors = [plt.cm.Reds(0.3 + 0.7 * f / max(PERTURB_FRACTIONS))
                   for f in PERTURB_FRACTIONS]
BAR_COLORS = _solver_colors[:2 + len(EXTRA_SOLVERS)] + _perturb_colors + ['gray']

PERTURB_LEVELS = ['cassiopeia'] + \
                 [f'perturbed_{int(f*100)}pct' for f in PERTURB_FRACTIONS] + \
                 ['random']
PERTURB_LABELS = ['Cass\n(0%)'] + \
                 [f'{int(f*100)}%' for f in PERTURB_FRACTIONS] + \
                 ['Random']

GENO_COLORS = {}
for g in df.genotype.unique():
    if g == 'NT':     GENO_COLORS[g] = 'steelblue'
    elif g == 'Lkb1': GENO_COLORS[g] = 'coral'
    elif g == 'Apc':  GENO_COLORS[g] = 'forestgreen'
    else:             GENO_COLORS[g] = 'gray'

def _get_vals(method, metric):
    if method == 'random':
        return df[df.category == 'random'][metric].dropna()
    else:
        return df[df.method == method][metric].dropna()

PLOT_DIR = f"plots_{MIN_LEAVES}to{MAX_LEAVES}"
os.makedirs(PLOT_DIR, exist_ok=True)
print(f"Plots will be saved to: {PLOT_DIR}/")


# %% Cell 12 — PLOT 1: Overview per tumor

n_panels = 3 if HAS_LIK else 2
metrics_to_plot = [('cPHS', 'log₁₀(cPHS)  [higher = better]', True),
                   ('parsimony', 'log₁₀(parsimony)  [lower = better]', True)]
if HAS_LIK:
    metrics_to_plot.append(('likelihood', 'likelihood  [higher = better]', False))

fig, axes = plt.subplots(n_panels, 1, figsize=(max(16, n_tumors * 0.5), 4 * n_panels), sharex=True)
if n_panels == 1: axes = [axes]

for ax, (metric, ylabel, log) in zip(axes, metrics_to_plot):
    for t_idx, tumor in enumerate(tumors):
        sub = df[df.tumor == tumor]
        rand = sub[sub.category == 'random'][metric].dropna()
        if len(rand):
            lo = np.log10(max(rand.min(), eps)) if log else rand.min()
            hi = np.log10(max(rand.max(), eps)) if log else rand.max()
            ax.fill_between([t_idx-0.3, t_idx+0.3], lo, hi, color='gray', alpha=0.25, zorder=1)
        for _, row in sub[sub.category.isin(['precomputed', 'solver'])].iterrows():
            v = row[metric]
            if pd.isna(v): continue
            y = np.log10(max(v, eps)) if log else v
            ax.scatter(t_idx, y, s=50, zorder=3, edgecolors='black', linewidth=0.4,
                      color=METHOD_COLORS.get(row.method, 'gray'))
        for _, row in sub[sub.category == 'perturbed'].iterrows():
            v = row[metric]
            if pd.isna(v): continue
            y = np.log10(max(v, eps)) if log else v
            pct = int(re.search(r'\d+', row.method).group())
            alpha_val = 0.3 + 0.7 * (pct / (max(PERTURB_FRACTIONS) * 100))
            ax.scatter(t_idx, y, s=20, marker='v', color='red', alpha=alpha_val, zorder=2)
    if metric == 'cPHS': ax.axhline(np.log10(0.05), color='red', ls=':', alpha=0.5)
    ax.set_ylabel(ylabel, fontsize=10)

handles = [Patch(fc=c, ec='black', lw=0.5, label=n) for n, c in METHOD_COLORS.items()]
handles.append(Patch(fc='gray', alpha=0.3, label='random range'))
handles.append(Line2D([0],[0], marker='v', ls='', color='red', alpha=0.6, ms=6, label='perturbed (▼)'))
axes[0].legend(handles=handles, fontsize=7, ncol=4, loc='lower left')
axes[-1].set_xticks(range(n_tumors))
axes[-1].set_xticklabels(tumors, rotation=90, fontsize=5)
fig.suptitle(f"All metrics across methods and tumors (n={MIN_LEAVES}–{MAX_LEAVES})", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/01_overview_per_tumor.png", dpi=150, bbox_inches='tight')
plt.show()


# %% Cell 13 — PLOT 2: Method ranking

n_bar_panels = 3 if HAS_LIK else 2
fig, axes = plt.subplots(1, n_bar_panels, figsize=(7 * n_bar_panels, 6))
x = np.arange(len(METHOD_ORDER))

ax = axes[0]
pass_rates = []
for m in METHOD_ORDER:
    vals = _get_vals(m, 'cPHS')
    pass_rates.append((vals >= 0.05).mean() * 100 if len(vals) else 0)
bars = ax.bar(x, pass_rates, color=BAR_COLORS, edgecolor='black', linewidth=0.5)
ax.axhline(50, color='gray', ls='--', alpha=0.3)
ax.set_xticks(x); ax.set_xticklabels(BAR_LABELS, fontsize=7, rotation=45)
ax.set_ylabel("% tumors passing cPHS ≥ 0.05"); ax.set_ylim(0, 110)
ax.set_title("cPHS pass rate (higher = better)")
for bar, rate in zip(bars, pass_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate:.0f}%', ha='center', va='bottom', fontsize=6)

ax = axes[1]
cass_mean_pars = df[df.method == 'cassiopeia']['parsimony'].mean()
norm_means, norm_stds = [], []
for m in METHOD_ORDER:
    vals = _get_vals(m, 'parsimony')
    normed = vals / cass_mean_pars if len(vals) else pd.Series([np.nan])
    norm_means.append(normed.mean()); norm_stds.append(normed.std())
ax.bar(x, norm_means, yerr=norm_stds, capsize=2, color=BAR_COLORS, edgecolor='black', linewidth=0.5)
ax.axhline(1.0, color='black', ls='--', alpha=0.3, label='Cassiopeia baseline')
ax.set_xticks(x); ax.set_xticklabels(BAR_LABELS, fontsize=7, rotation=45)
ax.set_ylabel("Parsimony (normalized, Cass = 1.0)"); ax.set_title("Parsimony (lower = better)"); ax.legend(fontsize=8)

if HAS_LIK:
    ax = axes[2]
    lik_means, lik_stds = [], []
    for m in METHOD_ORDER:
        vals = _get_vals(m, 'likelihood')
        lik_means.append(vals.mean() if len(vals) else np.nan)
        lik_stds.append(vals.std() if len(vals) else 0)
    ax.bar(x, lik_means, yerr=lik_stds, capsize=2, color=BAR_COLORS, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(BAR_LABELS, fontsize=7, rotation=45)
    ax.set_ylabel("Likelihood"); ax.set_title("Likelihood (higher = better)")

fig.suptitle(f"Method ranking (n={MIN_LEAVES}–{MAX_LEAVES}, {n_tumors} tumors)", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/02_method_ranking.png", dpi=150, bbox_inches='tight')
plt.show()


# %% Cell 14 — PLOT 3: Degradation curves

n_deg_panels = 3 if HAS_LIK else 2
fig, axes = plt.subplots(1, n_deg_panels, figsize=(6 * n_deg_panels, 5))
deg_configs = [
    ('cPHS', 'log₁₀(cPHS)', True, False, 'steelblue', 'darkblue'),
    ('parsimony', 'Parsimony (normalized to Cass=1)', False, True, 'salmon', 'darkred'),
]
if HAS_LIK:
    deg_configs.append(('likelihood', 'Likelihood', False, False, 'green', 'darkgreen'))

for ax, (metric, ylabel, log, normalize, line_color, mean_color) in zip(axes, deg_configs):
    all_curves = []
    for tumor in tumors:
        sub = df[df.tumor == tumor]
        curve = []
        for m in PERTURB_LEVELS:
            if m == 'random':
                vals = sub[sub.category == 'random'][metric].dropna()
                v = vals.mean() if len(vals) else np.nan
            else:
                vals = sub[sub.method == m][metric].values
                v = vals[0] if len(vals) else np.nan
            curve.append(np.log10(max(v, eps)) if log and not np.isnan(v) else v)
        if normalize and not np.isnan(curve[0]) and curve[0] > 0:
            curve = [c / curve[0] if not np.isnan(c) else np.nan for c in curve]
        all_curves.append(curve)
        ax.plot(range(len(PERTURB_LEVELS)), curve, '-o', alpha=0.15, ms=2, color=line_color)
    mean_c = np.nanmean(all_curves, axis=0)
    ax.plot(range(len(PERTURB_LEVELS)), mean_c, '-s', color=mean_color, lw=3, ms=8, label='Mean', zorder=5)
    if metric == 'cPHS': ax.axhline(np.log10(0.05), color='red', ls=':', alpha=0.5, label='p=0.05')
    if normalize: ax.axhline(1.0, color='black', ls='--', alpha=0.3)
    ax.set_xticks(range(len(PERTURB_LEVELS))); ax.set_xticklabels(PERTURB_LABELS, fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(metric); ax.legend(fontsize=9)

fig.suptitle("Metric degradation: intact → perturbed → random", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/03_degradation_curves.png", dpi=150, bbox_inches='tight')
plt.show()


# %% Cell 15 — PLOT 4: Reconstructed vs Random (4-panel)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
for t_idx, tumor in enumerate(tumors):
    sub = df[df.tumor == tumor]
    cv = sub[sub.method == 'cassiopeia']['cPHS'].values
    if len(cv): ax.scatter(t_idx, np.log10(max(cv[0], eps)), s=40, color='steelblue', zorder=3, edgecolors='k', linewidth=0.3)
    nv = sub[sub.method == 'nj']['cPHS'].values
    if len(nv): ax.scatter(t_idx, np.log10(max(nv[0], eps)), s=40, color='salmon', zorder=3, edgecolors='k', linewidth=0.3)
    rand = sub[sub.category == 'random']['cPHS'].dropna()
    if len(rand):
        ax.scatter(t_idx, np.log10(max(rand.mean(), eps)), s=20, color='gray', marker='x', zorder=2)
        ax.fill_between([t_idx-0.3, t_idx+0.3], np.log10(max(rand.min(), eps)), np.log10(max(rand.max(), eps)), color='gray', alpha=0.2)
ax.axhline(np.log10(0.05), color='red', ls=':', alpha=0.5)
ax.set_xticks(range(n_tumors)); ax.set_xticklabels(tumors, rotation=90, fontsize=4)
ax.set_ylabel("log₁₀(cPHS)"); ax.set_title("A. cPHS: Reconstructed vs Random")

ax = axes[0, 1]
for t_idx, tumor in enumerate(tumors):
    sub = df[df.tumor == tumor]
    cv = sub[sub.method == 'cassiopeia']['parsimony'].values
    if len(cv): ax.scatter(t_idx, cv[0], s=40, color='steelblue', zorder=3, edgecolors='k', linewidth=0.3)
    nv = sub[sub.method == 'nj']['parsimony'].values
    if len(nv): ax.scatter(t_idx, nv[0], s=40, color='salmon', zorder=3, edgecolors='k', linewidth=0.3)
    rand = sub[sub.category == 'random']['parsimony'].dropna()
    if len(rand):
        ax.scatter(t_idx, rand.mean(), s=20, color='gray', marker='x', zorder=2)
        ax.fill_between([t_idx-0.3, t_idx+0.3], rand.min(), rand.max(), color='gray', alpha=0.2)
ax.set_xticks(range(n_tumors)); ax.set_xticklabels(tumors, rotation=90, fontsize=4)
ax.set_ylabel("Parsimony"); ax.set_title("B. Parsimony: Reconstructed vs Random")

ax = axes[1, 0]
for t_idx, tumor in enumerate(tumors):
    sub = df[df.tumor == tumor]
    rand = sub[sub.category == 'random']['cPHS'].dropna()
    if len(rand) == 0: continue
    rand_mean_log = np.log10(max(rand.mean(), eps))
    cv = sub[sub.method == 'cassiopeia']['cPHS'].values
    if len(cv): ax.bar(t_idx - 0.15, np.log10(max(cv[0], eps)) - rand_mean_log, 0.3, color='steelblue', edgecolor='black', linewidth=0.3)
    nv = sub[sub.method == 'nj']['cPHS'].values
    if len(nv): ax.bar(t_idx + 0.15, np.log10(max(nv[0], eps)) - rand_mean_log, 0.3, color='salmon', edgecolor='black', linewidth=0.3)
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(range(n_tumors)); ax.set_xticklabels(tumors, rotation=90, fontsize=4)
ax.set_ylabel("Δlog₁₀(cPHS) over random"); ax.set_title("C. cPHS separation from random")

ax = axes[1, 1]
for t_idx, tumor in enumerate(tumors):
    sub = df[df.tumor == tumor]
    rand = sub[sub.category == 'random']['parsimony'].dropna()
    if len(rand) == 0: continue
    rand_mean = rand.mean()
    cv = sub[sub.method == 'cassiopeia']['parsimony'].values
    if len(cv) and rand_mean > 0: ax.bar(t_idx - 0.15, (rand_mean - cv[0]) / rand_mean * 100, 0.3, color='steelblue', edgecolor='black', linewidth=0.3)
    nv = sub[sub.method == 'nj']['parsimony'].values
    if len(nv) and rand_mean > 0: ax.bar(t_idx + 0.15, (rand_mean - nv[0]) / rand_mean * 100, 0.3, color='salmon', edgecolor='black', linewidth=0.3)
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(range(n_tumors)); ax.set_xticklabels(tumors, rotation=90, fontsize=4)
ax.set_ylabel("% improvement over random"); ax.set_title("D. Parsimony separation from random")

fig.suptitle(f"Reconstructed vs Random comparison (n={MIN_LEAVES}–{MAX_LEAVES})", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/04_reconstructed_vs_random.png", dpi=150, bbox_inches='tight')
plt.show()


# %% Cell 16 — PLOT 5: Metric separation scatter + winners

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sep_cphs, sep_pars, genos_list = [], [], []
for tumor in tumors:
    sub = df[df.tumor == tumor]
    genos_list.append(sub.genotype.iloc[0])
    recon = sub[sub.category.isin(['precomputed', 'solver'])]
    rand  = sub[sub.category == 'random']
    r_c = recon['cPHS'].dropna(); d_c = rand['cPHS'].dropna()
    sep_cphs.append(np.log10(max(r_c.max(), eps)) - np.log10(max(d_c.max(), eps)) if len(r_c) and len(d_c) else np.nan)
    r_p = recon['parsimony'].dropna(); d_p = rand['parsimony'].dropna()
    sep_pars.append((d_p.mean() - r_p.min()) / d_p.mean() * 100 if len(r_p) and len(d_p) and d_p.mean() > 0 else np.nan)
sep_cphs, sep_pars = np.array(sep_cphs), np.array(sep_pars)

ax = axes[0]
for g, gc in GENO_COLORS.items():
    m = np.array([x == g for x in genos_list])
    valid = m & ~np.isnan(sep_cphs) & ~np.isnan(sep_pars)
    ax.scatter(sep_pars[valid], sep_cphs[valid], c=gc, label=g, s=60, alpha=0.8, edgecolors='black', linewidth=0.4)
ax.axhline(0, color='gray', ls='--', alpha=0.3); ax.axvline(0, color='gray', ls='--', alpha=0.3)
ax.set_xlabel("Parsimony separation (%)"); ax.set_ylabel("cPHS separation (Δlog₁₀)")
ax.set_title("cPHS vs Parsimony: separation from random"); ax.legend(fontsize=8)

ax = axes[1]
sizes = [df[df.tumor == t].n_cells.iloc[0] for t in tumors]
for g, gc in GENO_COLORS.items():
    m = np.array([x == g for x in genos_list])
    valid = m & ~np.isnan(sep_cphs)
    ax.scatter(np.array(sizes)[valid], sep_cphs[valid], c=gc, label=g, s=60, alpha=0.8, edgecolors='black', linewidth=0.4)
ax.axhline(0, color='gray', ls='--', alpha=0.3)
ax.set_xlabel("Number of cells"); ax.set_ylabel("cPHS separation (Δlog₁₀)")
ax.set_title("cPHS separation vs tumor size"); ax.legend(fontsize=8)

ax = axes[2]
win_cphs, win_pars = {}, {}
for tumor in tumors:
    sub = df[(df.tumor == tumor) & df.category.isin(['precomputed', 'solver'])]
    if len(sub) < 2: continue
    best_c = sub.loc[sub.cPHS.idxmax(), 'method']
    win_cphs[best_c] = win_cphs.get(best_c, 0) + 1
    vp = sub.dropna(subset=['parsimony'])
    if len(vp):
        best_p = vp.loc[vp.parsimony.idxmin(), 'method']
        win_pars[best_p] = win_pars.get(best_p, 0) + 1
all_m = sorted(set(list(win_cphs) + list(win_pars)))
xw = np.arange(len(all_m)); w = 0.35
ax.bar(xw - w/2, [win_cphs.get(m, 0) for m in all_m], w, label='Best by cPHS', color='steelblue', edgecolor='black')
ax.bar(xw + w/2, [win_pars.get(m, 0) for m in all_m], w, label='Best by Parsimony', color='salmon', edgecolor='black')
ax.set_xticks(xw); ax.set_xticklabels(all_m, rotation=45, fontsize=8)
ax.set_ylabel("# tumors ranked #1"); ax.set_title("Which solver wins?"); ax.legend(fontsize=8)

fig.suptitle("Metric comparison", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/05_metric_comparison.png", dpi=150, bbox_inches='tight')
plt.show()


# %% Cell 17 — PLOT 6: Spearman rank correlation

fig, ax = plt.subplots(figsize=(8, 5))
rhos = []
for tumor in tumors:
    sub = df[(df.tumor == tumor)].dropna(subset=['cPHS', 'parsimony'])
    if len(sub) < 4: continue
    rho, _ = spearmanr(sub['cPHS'].values, -sub['parsimony'].values)
    rhos.append(rho)
if rhos:
    ax.hist(rhos, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(rhos), color='red', lw=2, label=f'mean ρ = {np.mean(rhos):.2f}')
    ax.axvline(np.median(rhos), color='orange', lw=2, ls='--', label=f'median ρ = {np.median(rhos):.2f}')
    ax.legend(fontsize=10)
    ax.set_xlabel("Spearman ρ (cPHS vs −parsimony)"); ax.set_ylabel("# tumors")
    ax.set_title("Do cPHS and parsimony rank trees the same way?"); ax.set_xlim(-1, 1)
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/06_metric_agreement.png", dpi=150, bbox_inches='tight')
plt.show()


# %% Cell 18 — PLOT 7: Genotype breakdown

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
genotype_list = sorted(df.genotype.unique())

ax = axes[0]
for gi, g in enumerate(genotype_list):
    sub_g = df[df.genotype == g]
    methods_here = ['cassiopeia', 'nj'] + EXTRA_SOLVERS + ['random']
    rates = []
    for m in methods_here:
        vals = sub_g[sub_g.category == 'random']['cPHS'].dropna() if m == 'random' else sub_g[sub_g.method == m]['cPHS'].dropna()
        rates.append((vals >= 0.05).mean() * 100 if len(vals) else 0)
    x = np.arange(len(methods_here))
    offset = (gi - len(genotype_list)/2 + 0.5) * 0.25
    ax.bar(x + offset, rates, 0.22, label=g, color=GENO_COLORS.get(g, 'gray'), edgecolor='black', linewidth=0.3)
ax.set_xticks(range(len(methods_here))); ax.set_xticklabels(methods_here, rotation=45, fontsize=8)
ax.set_ylabel("% passing cPHS ≥ 0.05"); ax.set_title("cPHS pass rate by genotype"); ax.legend(fontsize=8)

ax = axes[1]
for gi, g in enumerate(genotype_list):
    sub_g = df[df.genotype == g]
    cass_mean_g = sub_g[sub_g.method == 'cassiopeia']['parsimony'].mean()
    if np.isnan(cass_mean_g) or cass_mean_g == 0: continue
    methods_here = ['cassiopeia', 'nj'] + EXTRA_SOLVERS + ['random']
    ratios = []
    for m in methods_here:
        vals = sub_g[sub_g.category == 'random']['parsimony'].dropna() if m == 'random' else sub_g[sub_g.method == m]['parsimony'].dropna()
        ratios.append(vals.mean() / cass_mean_g if len(vals) else np.nan)
    x = np.arange(len(methods_here))
    offset = (gi - len(genotype_list)/2 + 0.5) * 0.25
    ax.bar(x + offset, ratios, 0.22, label=g, color=GENO_COLORS.get(g, 'gray'), edgecolor='black', linewidth=0.3)
ax.axhline(1.0, color='black', ls='--', alpha=0.3)
ax.set_xticks(range(len(methods_here))); ax.set_xticklabels(methods_here, rotation=45, fontsize=8)
ax.set_ylabel("Parsimony (normalized to Cass=1)"); ax.set_title("Parsimony by genotype"); ax.legend(fontsize=8)

fig.suptitle("Genotype breakdown", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/07_genotype_breakdown.png", dpi=150, bbox_inches='tight')
plt.show()


# %% Cell 19 — PLOT 8: Size effects

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sizes = np.array([df[df.tumor == t].n_cells.iloc[0] for t in tumors])

ax = axes[0]
cphs_cass = np.array([np.log10(max(df[(df.tumor == t) & (df.method == 'cassiopeia')]['cPHS'].values[0], eps))
                       if len(df[(df.tumor == t) & (df.method == 'cassiopeia')]) else np.nan for t in tumors])
for g, gc in GENO_COLORS.items():
    m = np.array([df[df.tumor == t].genotype.iloc[0] == g for t in tumors])
    valid = m & ~np.isnan(cphs_cass)
    ax.scatter(sizes[valid], cphs_cass[valid], c=gc, label=g, s=60, alpha=0.8, edgecolors='black', linewidth=0.4)
ax.axhline(np.log10(0.05), color='red', ls=':', alpha=0.5, label='p=0.05')
ax.set_xlabel("Number of cells"); ax.set_ylabel("log₁₀(cPHS) — Cassiopeia")
ax.set_title("Does tree size affect cPHS?"); ax.legend(fontsize=8)

ax = axes[1]
pars_sep_arr = []
for tumor in tumors:
    sub = df[df.tumor == tumor]
    cv = sub[sub.method == 'cassiopeia']['parsimony'].values
    rv = sub[sub.category == 'random']['parsimony'].dropna()
    pars_sep_arr.append((rv.mean() - cv[0]) / rv.mean() * 100 if len(cv) and len(rv) and rv.mean() > 0 else np.nan)
pars_sep_arr = np.array(pars_sep_arr)
for g, gc in GENO_COLORS.items():
    m = np.array([df[df.tumor == t].genotype.iloc[0] == g for t in tumors])
    valid = m & ~np.isnan(pars_sep_arr)
    ax.scatter(sizes[valid], pars_sep_arr[valid], c=gc, label=g, s=60, alpha=0.8, edgecolors='black', linewidth=0.4)
ax.axhline(0, color='gray', ls='--', alpha=0.3)
ax.set_xlabel("Number of cells"); ax.set_ylabel("Parsimony % improvement over random")
ax.set_title("Does tree size affect parsimony separation?"); ax.legend(fontsize=8)

fig.suptitle("Size effects", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f"{PLOT_DIR}/08_size_effects.png", dpi=150, bbox_inches='tight')
plt.show()


# %% Cell 20 — Summary statistics

print("=" * 80)
print(f"FINAL SUMMARY  (n={MIN_LEAVES}–{MAX_LEAVES}, {n_tumors} tumors)")
print("=" * 80)

print(f"\n--- cPHS (higher = better) ---")
for m in METHOD_ORDER:
    vals = _get_vals(m, 'cPHS')
    if len(vals) == 0: continue
    log_vals = np.log10(np.clip(vals, eps, None))
    pr = (vals >= 0.05).mean()
    print(f"  {m:20s}: mean log₁₀={log_vals.mean():7.2f} ± {log_vals.std():.2f}"
          f"  pass@0.05: {pr:.0%}  (n={len(vals)})")

print(f"\n--- Parsimony (lower = better) ---")
for m in METHOD_ORDER:
    vals = _get_vals(m, 'parsimony')
    if len(vals) == 0: continue
    print(f"  {m:20s}: mean={vals.mean():8.1f} ± {vals.std():.1f}  (n={len(vals)})")

print(f"\n--- Metric agreement ---")
agree, total = 0, 0
for tumor in tumors:
    sub = df[(df.tumor == tumor) & df.category.isin(['precomputed', 'solver'])]
    if len(sub) < 2: continue
    total += 1
    best_c = sub.loc[sub.cPHS.idxmax(), 'method']
    vp = sub.dropna(subset=['parsimony'])
    if len(vp):
        best_p = vp.loc[vp.parsimony.idxmin(), 'method']
        if best_c == best_p: agree += 1
if total:
    print(f"  cPHS & Parsimony agree on best solver: {agree}/{total} ({agree/total:.0%})")

print(f"\n--- Files ---")
print(f"  Checkpoints: {CHECKPOINT_DIR}/ ({len(glob.glob(f'{CHECKPOINT_DIR}/*.csv'))} files)")
print(f"  Merged CSV:  {MERGED_CSV}")
print(f"  Plots:       {PLOT_DIR}/")
for f in sorted(glob.glob(f"{PLOT_DIR}/*.png")):
    print(f"    {f}")


# %% Cell 21 — Display all plots inline

print("All plots:")
for f in sorted(glob.glob(f"{PLOT_DIR}/*.png")):
    print(f"\n{'─'*40}\n{f}")
    display(Image(filename=f))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




