#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
"""
cPHS threshold calibration simulation.
Stripped-down: only generates GT trees, reconstructs with 3 methods, computes cPHS + RF distance.
No parsimony, likelihood, random trees, or NNI perturbation.

Usage:
    python threshold_simulation.py --n 500 --k 30 --reps 100 --output results_n500_k30.csv

Parameters:
    --n        Number of leaves (subsampled tree)
    --k        Sequence length (number of CRISPR target sites)
    --m        Number of mutation states (default: 50)
    --rho      Mutation probability at a leaf (default: 0.5)
    --N_orig   Original tree size before subsampling (default: 10000)
    --reps     Number of repetitions
    --output   Output CSV path
    --resume   Resume from existing partial CSV
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import binom

import cassiopeia as cass
from cassiopeia.simulator import (
    BirthDeathFitnessSimulator,
    UniformLeafSubsampler,
    Cas9LineageTracingDataSimulator,
)
from cassiopeia.tools.branch_length_estimator.IIDExponentialMLE import IIDExponentialMLE


# ── Core functions ────────────────────────────────────────────────────

def calc_phs_and_lca_times(tree):
    cm = tree.character_matrix
    n, k = cm.shape
    res = pd.DataFrame(np.zeros((n, n)) - 1, index=cm.index, columns=cm.index)
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
                        res.loc[leaf_1, leaf_2] = phs
                        res.loc[leaf_2, leaf_1] = phs
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
    beta = 1 - np.exp(-lam * (1 - lca_times))
    prob = alpha * beta**2 * q
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


def get_lam_and_q(tree):
    cm = tree.character_matrix.to_numpy()
    proportion_mutated = np.sum(cm > 0) / np.sum(cm >= 0)
    lam = -np.log(1.0 - proportion_mutated)
    if tree.priors is not None:
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


def postprocess(tree):
    tree.collapse_unifurcations()
    tree.reconstruct_ancestral_characters()
    tree.set_character_states(tree.root, [0] * tree.n_character)
    tree.scale_to_unit_length()
    IIDExponentialMLE().estimate_branch_lengths(tree)
    return tree


# ── Simulation ────────────────────────────────────────────────────────

def simulate_one(rep_id, n, k, m, rho, N_orig):
    """Run one simulation rep: generate GT, reconstruct, compute cPHS + RF."""

    lam = -np.log(1.0 - rho)

    # Generate state priors (random exponential, normalized)
    state_priors = np.array([np.random.exponential(1e-5) for _ in range(m)])
    state_priors /= np.sum(state_priors)
    priors = {i + 1: state_priors[i] for i in range(m)}

    # Simulate ground-truth tree
    topology_simulator = BirthDeathFitnessSimulator(
        birth_waiting_distribution=lambda scale: np.random.exponential(1 / scale),
        initial_birth_scale=2,
        num_extant=int(N_orig),
    )
    original_topology = topology_simulator.simulate_tree()

    leaf_subsampler = UniformLeafSubsampler(number_of_leaves=int(n))
    gt_tree = leaf_subsampler.subsample_leaves(original_topology)
    gt_tree.scale_to_unit_length()

    # Generate lineage tracing data (k cassettes of size 1, matching original notebook)
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

    # Get GT metrics
    gt_lam, gt_q = get_lam_and_q(gt_tree)

    try:
        gt_cPHS = calc_cPHS(gt_tree, gt_lam, gt_q)
    except Exception:
        gt_cPHS = np.nan

    rows = []

    # Reconstruct with 3 methods
    solvers = {
        'cassiopeia': cass.solver.VanillaGreedySolver(),
        'maxcut': cass.solver.MaxCutSolver(),
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

            # cPHS
            r_lam, r_q = get_lam_and_q(recon_tree)
            recon_cPHS = calc_cPHS(recon_tree, r_lam, r_q)

            # RF distance to ground truth
            rf, rf_max = cass.critique.compare.robinson_foulds(gt_tree, recon_tree)
            rf_norm = rf / rf_max if rf_max > 0 else 0

            # Triplets distance
            try:
                triplets = cass.critique.compare.triplets_correct(gt_tree, recon_tree)
                triplets_score = np.mean(list(triplets[0].values()))
            except Exception:
                triplets_score = np.nan

            rows.append({
                'rep': rep_id,
                'solver': solver_name,
                'cPHS': recon_cPHS,
                'rf_norm': rf_norm,
                'triplets': triplets_score,
                'gt_cPHS': gt_cPHS,
            })

        except Exception as e:
            print(f"    Rep {rep_id}, {solver_name}: FAILED — {e}")
            rows.append({
                'rep': rep_id,
                'solver': solver_name,
                'cPHS': np.nan,
                'rf_norm': np.nan,
                'triplets': np.nan,
                'gt_cPHS': gt_cPHS,
            })

    return rows


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='cPHS threshold calibration simulation')
    parser.add_argument('--n', type=int, required=True, help='Number of leaves')
    parser.add_argument('--k', type=int, required=True, help='Sequence length')
    parser.add_argument('--m', type=int, default=50, help='Number of mutation states')
    parser.add_argument('--rho', type=float, default=0.5, help='Mutation probability at leaf')
    parser.add_argument('--N_orig', type=int, default=10000, help='Original tree size')
    parser.add_argument('--reps', type=int, default=100, help='Number of repetitions')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--resume', action='store_true', help='Resume from existing CSV')
    args = parser.parse_args()

    print(f"=== cPHS Threshold Calibration ===")
    print(f"  n={args.n}, k={args.k}, m={args.m}, rho={args.rho}")
    print(f"  N_orig={args.N_orig}, reps={args.reps}")
    print(f"  output={args.output}")
    print()

    # Resume logic
    done_reps = set()
    if args.resume and os.path.exists(args.output):
        existing = pd.read_csv(args.output)
        done_reps = set(existing['rep'].unique())
        print(f"  Resuming: {len(done_reps)} reps already done")

    all_rows = []
    for rep in range(args.reps):
        if rep in done_reps:
            continue

        t0 = time.time()
        print(f"  Rep {rep + 1}/{args.reps} ...", end="", flush=True)

        try:
            rows = simulate_one(rep, args.n, args.k, args.m, args.rho, args.N_orig)
            all_rows.extend(rows)
            elapsed = time.time() - t0
            cphs_vals = [r['cPHS'] for r in rows if not np.isnan(r.get('cPHS', np.nan))]
            print(f"  done ({elapsed:.0f}s, cPHS: {[f'{v:.2e}' for v in cphs_vals]})")
        except Exception as e:
            print(f"  FAILED: {e}")

        # Checkpoint every 10 reps
        if (rep + 1) % 10 == 0 and all_rows:
            _save(args, all_rows, done_reps)

    # Final save
    if all_rows:
        _save(args, all_rows, done_reps)

    print(f"\nDone! Results saved to {args.output}")


def _save(args, new_rows, done_reps):
    new_df = pd.DataFrame(new_rows)
    new_df['n'] = args.n
    new_df['k'] = args.k
    new_df['m'] = args.m
    new_df['rho'] = args.rho

    if args.resume and os.path.exists(args.output):
        existing = pd.read_csv(args.output)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
