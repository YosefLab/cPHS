# runs the simulations of section 4, and writes one csv row per scored tree (see README.md)
import os
import time
import argparse
import itertools
import pandas as pd

import cassiopeia as cass

from sim_functions import sim_functions as sim_funcs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[200], help="number of leaves of the GT tree")
    parser.add_argument("--subsampling-ratio", type=float, default=1.0,
                        help="the GT tree is grown to n / ratio cells, and n of them are sampled")
    parser.add_argument("--k", type=int, nargs="+", default=[30], help="sequence length")
    parser.add_argument("--m", type=int, nargs="+", default=[32], help="number of mutated states")
    parser.add_argument("--rho", type=float, nargs="+", default=[0.5], help="mutation probability at a leaf")
    parser.add_argument("--solvers", nargs="+", default=sim_funcs.solver_names, choices=sim_funcs.solver_names)
    parser.add_argument("--reps", type=int, default=10, help="realizations per setting")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="output csv file")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # realizations that are already in the output file are skipped, so a run can be resumed
    done = set()
    if os.path.exists(args.out):
        df_done = pd.read_csv(args.out)
        done = set(map(tuple, df_done[["n", "subsampling_ratio", "k", "m", "rho", "seed"]].values.tolist()))
        print("{} rows already in {}".format(len(df_done), args.out))

    max_counts = 100
    for n, k, m, rho in itertools.product(args.n, args.k, args.m, args.rho):
        print("n = {}, k = {}, m = {}, rho = {}".format(n, k, m, rho))
        for seed in range(args.first_seed, args.first_seed + args.reps):
            if (n, args.subsampling_ratio, k, m, rho, seed) in done:
                continue
            print("\nRealization with seed {}...".format(seed))
            start_time = time.time()

            # if the simulation fails, we retry with a new seed
            counter = 0
            while counter < max_counts:
                seed_used = seed + 1000000 * counter
                try:
                    scores = sim_funcs.simulate_and_calculate_metrics(
                        seed_used, n, args.subsampling_ratio, k, m, rho, args.solvers)
                    break
                except cass.mixins.errors.TreeSimulatorError:
                    counter += 1
                    print("All lineages died before stopping condition. Attempt #{}".format(counter))
                except cass.mixins.errors.IIDExponentialMLEError:
                    counter += 1
                    print("IIDExponentialMLE BLE failed when it shouldn't have. Attempt #{}".format(counter))
            if counter >= max_counts:
                print("No more attempts!")
                exit(1)

            # save scores (results are appended after every realization)
            setting = {"n": n, "subsampling_ratio": args.subsampling_ratio, "k": k, "m": m, "rho": rho,
                       "seed": seed, "seed_used": seed_used}
            df = pd.DataFrame([{**setting, **row} for row in scores])
            df.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
            print("Seed {} done in {:.0f} seconds".format(seed, time.time() - start_time))


if __name__ == "__main__":
    main()
