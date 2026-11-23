# runs the two simulations of section 4, and then plots figures 1-5 (see README.md)
import os
import sys
import argparse
import subprocess

# figures 1-4 and figure 5 (left): n = 200 leaves, m = 32 states, rho = 0.5
main_setting = "--n 200 --k 10 20 30 40 --m 32 --rho 0.5"
# figure 5 (right): n = 1000 leaves sampled from a tree of 10^6 cells, m = 50 states.
large_setting = "--n 1000 --subsampling-ratio 0.001 --k 30 40 50 60 70 80 --m 50 --rho 0.5"

simulations_dir = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name, args):
    print("python simulations/{} {}".format(script_name, " ".join(args)), flush=True)
    return_code = subprocess.call([sys.executable, "-u", os.path.join(simulations_dir, script_name)] + args,
                                  cwd=os.path.dirname(simulations_dir))
    if return_code != 0:
        sys.exit(return_code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10, help="realizations per setting (1000 in the paper)")
    parser.add_argument("--skip-large", action="store_true", help="skip the n = 1000 simulation of figure 5 (right)")
    args = parser.parse_args()

    main_csv = os.path.join("output", "simulations", "main.csv")
    large_csv = os.path.join("output", "simulations", "large.csv")
    plot_args = ["--csv", main_csv, "--k-ref", "30", "--outdir", os.path.join("output", "figures")]

    # figures 1-5, without the right panel of figure 5
    run_script("run_simulation.py", main_setting.split() + ["--reps", str(args.reps), "--out", main_csv])
    run_script("plot_figures.py", plot_args)

    # figure 5 again, with its right panel (this simulation takes much longer)
    if not args.skip_large:
        run_script("run_simulation.py", large_setting.split() + ["--reps", str(args.reps), "--out", large_csv])
        run_script("plot_figures.py", plot_args + ["--csv-large", large_csv])


if __name__ == "__main__":
    main()
