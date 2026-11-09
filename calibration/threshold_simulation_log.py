#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

"""
Threshold calibration using fully log-scale cPHS.
All probability computations done in log space — no underflow at any k/rho.

Usage:
    python threshold_simulation_log.py --n 100 --k 70 --reps 30 --output results/log_n100_k70_rho9.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

import cassiopeia as cass
from cassiopeia.simulator import (
    BirthDeathFitnessSimulator,
    UniformLeafSubsampler,
    Cas9LineageTracingDataSimulator,
)
from cassiopeia.tools.branch_length_estimator.IIDExponentialMLE import IIDExponentialMLE


# ── Helper functions ──

def get_lam_and_q(tree):
    cm = tree.character_matrix.to_numpy()
    proportion_mutated = np.sum(cm > 0) / np.sum(cm >= 0)
    lam = -np.log(1.0 - min(proportion_mutated, 0.9999))
    priors = np.array(list(tree.priors[0].values()))
    q = np.sum(priors ** 2)
    return lam, q


def postprocess(tree):
    tree.collapse_mutationless_edges(infer_ancestral_characters=True)
    ble = IIDExponentialMLE()
    ble.estimate_branch_lengths(tree)
    return tree


def calc_phs_and_lca_times(tree):
    cm = tree.character_matrix
    leaves = cm.index.tolist()
    n = len(leaves)
    phs = np.zeros((n, n))
    lca_times = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            l1, l2 = leaves[i], leaves[j]
            s1 = cm.loc[l1].values
            s2 = cm.loc[l2].values
            shared = np.sum((s1 > 0) & (s2 > 0) & (s1 == s2))
            phs[i, j] = shared
            phs[j, i] = shared
            lca = tree.find_lca(l1, l2)
            h = tree.get_time(lca)
            lca_times[i, j] = h
            lca_times[j, i] = h
    return phs, lca_times


def _log1mexp(x):
    """Compute log(1 - exp(-x)) stably for x > 0."""
    return np.where(x < 0.6931,
                    np.log(-np.expm1(-x)),
                    np.log1p(-np.exp(-x)))


def _log_binom_sf(x, k, log_p, log_1mp):
    """
    Compute log(P[Binom(k,p) >= x]) using log_p and log_1mp directly.
    No intermediate computation of p or 1-p, so no underflow.
    """
    x = int(x)
    if x <= 0:
        return 0.0  # P[X >= 0] = 1
    if x > k:
        return -np.inf

    lgk1 = gammaln(k + 1)
    terms = []
    for j in range(x, k + 1):
        log_pmf = (lgk1 - gammaln(j + 1) - gammaln(k - j + 1)
                   + j * log_p + (k - j) * log_1mp)
        terms.append(log_pmf)

    if len(terms) == 0:
        return -np.inf

    return logsumexp(terms)


def calc_log_cPHS(tree, lam, q):
    """
    Compute log(cPHS) entirely in log space.
    prob = exp(-lam*tau) * (1-exp(-lam*(1-tau)))^2 * q
    log(prob) = -lam*tau + 2*log(1-exp(-lam*(1-tau))) + log(q)
    log(1-prob) computed via log1p(-exp(log_prob))
    """
    phs, lca_times = calc_phs_and_lca_times(tree)
    cm = tree.character_matrix
    n, k = cm.shape

    log_q = np.log(q)

    # Compute log(prob) directly — never compute prob itself
    # log(alpha) = -lam * tau
    log_alpha = -lam * lca_times

    # log(beta) = log(1 - exp(-lam*(1-tau)))
    # lam*(1-tau) is always >= 0
    arg = lam * (1.0 - lca_times)
    # Clamp to avoid log1mexp(0) = -inf for tau=1
    arg = np.clip(arg, 1e-300, None)
    log_beta = _log1mexp(arg)

    # log(prob) = log(alpha) + 2*log(beta) + log(q)
    log_prob = log_alpha + 2 * log_beta + log_q

    # log(1 - prob): since prob is typically very small, log(1-prob) ~ -prob ~ exp(log_prob)
    # More precisely: log(1-prob) = log1p(-exp(log_prob))
    # For very negative log_prob (prob ~ 0), this is essentially 0
    log_1mprob = np.log1p(-np.exp(np.minimum(log_prob, 0)))
    # Handle edge case where log_prob >= 0 (prob >= 1): set to very negative
    log_1mprob = np.where(log_prob >= 0, -1e10, log_1mprob)

    # For pairs with lca_times == 1 (leaves sharing parent), skip
    # They have tau=1, arg=0, which would give log_beta=-inf, log_prob=-inf
    # These pairs have no meaningful PHS test, mark as log_pv = 0 (p=1)
    skip_mask = (lca_times >= 1.0 - 1e-12)

    # Compute log p-values for each off-diagonal pair
    nn = phs.shape[0]
    log_pv = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(nn):
            if i == j or skip_mask[i, j]:
                log_pv[i, j] = 0.0  # p-value = 1, will be filtered
                continue
            log_pv[i, j] = _log_binom_sf(
                phs[i, j], k, log_prob[i, j], log_1mprob[i, j])

    off_diagonal = ~np.eye(nn, dtype=bool)
    log_pv_flat = log_pv[off_diagonal]

    # Keep only entries with actual homoplasies (log_pv < ~0)
    mask = np.isfinite(log_pv_flat) & (log_pv_flat < -1e-10)
    log_pv_flat = log_pv_flat[mask]

    N = len(log_pv_flat)
    if N == 0:
        return 0.0  # log(1.0) = 0, no homoplasies detected

    # BH correction in log space
    log_pv_sorted = np.sort(log_pv_flat)
    log_adj = log_pv_sorted + np.log(N) - np.log(np.arange(1, N + 1))

    return float(np.min(log_adj))


# ── Simulation ──

def simulate_one(rep_id, n, k, m, rho, N_orig):
    lam = -np.log(1.0 - rho)

    state_priors = np.array([np.random.exponential(1e-5) for _ in range(m)])
    state_priors /= np.sum(state_priors)
    priors = {i + 1: state_priors[i] for i in range(m)}

    topology_simulator = BirthDeathFitnessSimulator(
        birth_waiting_distribution=lambda scale: np.random.exponential(1 / scale),
        initial_birth_scale=2,
        num_extant=int(N_orig),
    )
    original_topology = topology_simulator.simulate_tree()

    leaf_subsampler = UniformLeafSubsampler(number_of_leaves=int(n))
    gt_tree = leaf_subsampler.subsample_leaves(original_topology)
    gt_tree.scale_to_unit_length()

    lt_simulator = Cas9LineageTracingDataSimulator(
        number_of_cassettes=k,
        size_of_cassette=1,
        number_of_states=m,
        mutation_rate=lam,
        state_priors=priors,
    )
    lt_simulator.overlay_data(gt_tree)
    gt_tree.set_character_states(gt_tree.root, [0] * gt_tree.n_character)
    gt_tree.priors = {i: priors for i in range(k)}

    gt_lam, gt_q = get_lam_and_q(gt_tree)
    try:
        gt_log_cPHS = calc_log_cPHS(gt_tree, gt_lam, gt_q)
    except Exception:
        gt_log_cPHS = np.nan

    rows = []

    solvers = {
        'cassiopeia': cass.solver.VanillaGreedySolver(),
        'smj': cass.solver.SharedMutationJoiningSolver(),
    }

    for solver_name, solver in solvers.items():
        try:
            recon_tree = cass.data.CassiopeiaTree(
                character_matrix=gt_tree.character_matrix.copy(),
                priors=gt_tree.priors,
            )
            solver.solve(recon_tree)
            recon_tree = postprocess(recon_tree)

            r_lam, r_q = get_lam_and_q(recon_tree)
            log_cphs = calc_log_cPHS(recon_tree, r_lam, r_q)

            rf, rf_max = cass.critique.compare.robinson_foulds(gt_tree, recon_tree)
            rf_norm = rf / rf_max if rf_max > 0 else 0

            try:
                triplets = cass.critique.compare.triplets_correct(gt_tree, recon_tree)
                triplets_score = np.mean(list(triplets[0].values()))
            except Exception:
                triplets_score = np.nan

            rows.append({
                'rep': rep_id,
                'solver': solver_name,
                'log_cPHS': log_cphs,
                'rf_norm': rf_norm,
                'triplets': triplets_score,
                'gt_log_cPHS': gt_log_cPHS,
                'n': n, 'k': k, 'm': m, 'rho': rho,
            })

        except Exception as e:
            print(f"    Rep {rep_id}, {solver_name}: FAILED — {e}")
            rows.append({
                'rep': rep_id, 'solver': solver_name,
                'log_cPHS': np.nan, 'rf_norm': np.nan,
                'triplets': np.nan, 'gt_log_cPHS': gt_log_cPHS,
                'n': n, 'k': k, 'm': m, 'rho': rho,
            })

    return rows


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='cPHS threshold calibration (fully log-scale)')
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--m', type=int, default=50)
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--N_orig', type=int, default=10000)
    parser.add_argument('--reps', type=int, default=30)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    existing_reps = set()
    if args.resume and os.path.exists(args.output):
        try:
            old = pd.read_csv(args.output)
            existing_reps = set(old.rep.unique())
            print(f"  Resuming: {len(existing_reps)} reps already done")
        except Exception:
            pass

    for rep in range(args.reps):
        if rep in existing_reps:
            continue
        print(f"  Rep {rep}/{args.reps} (n={args.n}, k={args.k}, rho={args.rho})")
        try:
            rows = simulate_one(rep, args.n, args.k, args.m, args.rho, args.N_orig)
            row_df = pd.DataFrame(rows)
            if os.path.exists(args.output):
                row_df.to_csv(args.output, mode='a', header=False, index=False)
            else:
                row_df.to_csv(args.output, index=False)
        except Exception as e:
            print(f"    Rep {rep}: FAILED — {e}")

    print(f"  Done. Output: {args.output}")


if __name__ == '__main__':
    main()
