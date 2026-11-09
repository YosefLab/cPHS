"""
Core cPHS functions, exactly as used for the KPTracer analysis in
Zilber et al., "Can you trust your reconstructed lineage tree?" (Section 6).

Copied verbatim (logic unchanged) from Cell 3 of kptracer/kp_phs_all.py;
prepare_tree() is postprocess() from Cell 4 of the same script.
Requires Cassiopeia 2.1.0 (see environment.yml).

Input: a cassiopeia.data.CassiopeiaTree with
  - character states at all nodes (leaves observed, inner nodes reconstructed),
  - branch lengths scaled so the root is at time 0 and the leaves at time 1,
  - per-site priors in tree.priors (for q).
"""
import numpy as np
import pandas as pd
from scipy.stats import binom
from cassiopeia.tools.branch_length_estimator.IIDExponentialMLE import IIDExponentialMLE


def calc_phs_and_lca_times(tree):
    """Number of homoplasies k*phs(u,v) and LCA birth time for every leaf pair (Def. 3)."""
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
    """cPHS test statistic (Algorithm 1): binomial tail p-values (Lemma 1), BH-style adjustment, minimum."""
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
    """lambda = -log(1 - rho_hat); q = mean over sites of sum_j q_ij^2, per-site priors normalized."""
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


def prepare_tree(tree):
    """Preprocessing applied to every tree in the paper: small-MP ancestral states (Alg. 2),
    unmutated root, unit time scale, ML branch lengths (Prillo et al.)."""
    tree.collapse_unifurcations()
    tree.reconstruct_ancestral_characters()
    tree.set_character_states(tree.root, [0] * tree.n_character)
    tree.scale_to_unit_length()
    IIDExponentialMLE().estimate_branch_lengths(tree)
    return tree
