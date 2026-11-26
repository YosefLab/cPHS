# simulations

Simulations of Section 4: Figures 1 to 5 of the main text.

| File | Contents |
|---|---|
| `sim_functions.py` | One realization: a ground-truth tree (birth-death process with fitness, Appendix J), CRISPR-Cas9 mutations (Section 2.1), reconstructions by the seven solvers, and the scores of every tree: cPHS (Algorithm 1), parsimony, parsimony tail probability (Appendix D), likelihood, and the RF and triplets distances to the ground truth |
| `run_simulation.py` | Runs realizations over a grid of $n$, $k$, $m$, $\rho$ and writes one CSV row per scored tree |
| `plot_figures.py` | Figures 1 to 5 from those CSVs |
| `reproduce_figures.py` | The two simulation runs behind Figures 1 to 5, then the plots |

The cPHS score is computed with the functions in `../cphs` (the same code as for the KPTracer
analysis), so run the scripts from a checkout of the whole repository, in the environment of the
main README (Cassiopeia 2.1.0). The scripts were tested with Cassiopeia 2.1.0 installed from GitHub
(`pip install git+https://github.com/YosefLab/Cassiopeia@1ee5959eb9d3f8d4d26e2af5678234493bf54d6d`),
Python 3.10, numpy 1.26.4, pandas 2.3.3, scipy 1.15.3, scikit-learn 1.7.2, matplotlib 3.10.8.
Activate that environment before running the commands below; in a Python without Cassiopeia they
stop with `ModuleNotFoundError: No module named 'cassiopeia'`.

## Quick test

Ten realizations with small trees take about ten minutes and exercise the whole pipeline:

```bash
python simulations/run_simulation.py --n 50 --k 10 20 --m 32 --reps 5 --out output/simulations/test.csv
python simulations/plot_figures.py --csv output/simulations/test.csv --k-ref 20 --n-boot 20 --outdir output/figures_test
```

## Reproducing Figures 1 to 5

```bash
python simulations/reproduce_figures.py                # 10 realizations per setting
python simulations/reproduce_figures.py --reps 1000    # as in the paper
python simulations/reproduce_figures.py --skip-large   # without the n = 1000 run of Figure 5 (right)
```

The figures are written to `output/figures` once the $n = 200$ run is done, and again with the
right panel of Figure 5 after the $n = 1000$ run.

| Figure | Simulation setting | Analysis (`plot_figures.py`) |
|---|---|---|
| Fig 1 | $n = 200$, $k = 30$, $m = 32$, $\rho = 0.5$ | KDEs for GT, accurate and inaccurate trees, $d = d_{RF}$, $\epsilon = 1/4$ |
| Fig 2 | same | ROC and precision-recall curves, $d_{RF}$ with $\epsilon = 1/7$ and $d_P$ with $\epsilon = 0.01$ |
| Fig 3 | same, $k \in \{10, 20, 30, 40\}$ | AUC vs $k$ ($\epsilon = 1/7$) and vs $\epsilon$ ($k = 30$) |
| Fig 4 | same | balanced accuracy vs $k$ and vs $\epsilon$; cPHS with the fixed threshold $t = 10^{-3}$ |
| Fig 5 | left: same; right: $n = 1000$ sampled from $10^6$ cells (subsampling ratio $10^{-3}$), $m = 50$, $k \in \{30, \ldots, 80\}$ | fraction of realizations in which the GT tree gets the best score, i.e. no reconstruction scores strictly better (ties count as GT scoring best) |

All settings use the seven solvers (Neighbor-Joining, Cassiopeia-Greedy, Shared-Mutation-Joining,
MaxCut, MaxCut-Greedy, Spectral, Spectral-Greedy). Any other setting can be simulated by changing
the arguments of `run_simulation.py`:

| Argument | Meaning | Default |
|---|---|---|
| `--n` | number of leaves of the GT tree | 200 |
| `--subsampling-ratio` | fraction of the simulated cells kept as leaves: the tree is grown to $n$/ratio cells and $n$ of them are sampled | 1 (no subsampling) |
| `--k` | sequence length | 30 |
| `--m` | number of mutated states; $q_j$ drawn from an exponential distribution | 32 |
| `--rho` | mutation probability at a leaf, $\lambda = -\log(1-\rho)$ | 0.5 |
| `--solvers` | subset of `nj greedy smj maxcut maxcut_greedy spectral spectral_greedy` | all seven |
| `--reps`, `--first-seed` | realizations, seeded `first-seed`, ..., `first-seed + reps - 1` | 10, 0 |

`--n`, `--k`, `--m` and `--rho` accept several values; all combinations are simulated. The
birth-death parameters are set in `_get_tree` of `sim_functions.py`, and the cutoffs
$\epsilon$ of each figure at the top of `plot_figures.py`.

One realization with $n = 200$, $k = 30$ and the seven solvers takes about 10 minutes on one core:
a third is cPHS ($O(kn^2)$, pure Python), the rest the solvers, branch-length estimation,
likelihoods and triplets. With $n = 1000$ (Figure 5, right) a realization is many times slower,
since the cPHS time grows as $n^2$, and growing the $10^6$-cell tree from which the leaves are
sampled alone takes about 16 minutes and 7 GB of memory. Rows are appended to the CSV after every realization, so a run
can be interrupted and resumed with the same command. On a cluster, split the realizations over
jobs with `--first-seed` and a separate `--out` per job, and pass all the CSVs (or a quoted glob)
to `plot_figures.py`.

## Pipeline

For each realization:

1. **Ground truth.** A tree with $n$ leaves is grown by a birth-death process with heritable
   fitness (or with $n$/ratio leaves, of which $n$ are subsampled uniformly), and time is
   rescaled so the experiment ends at $\tau = 1$. The $k$ sites are mutated along the tree with
   rate $\lambda$ under the non-modifiable model, with no missing data.
2. **Reconstruction.** Each solver is run on the observed character matrix.
3. **Preprocessing**, identical for the GT tree and the reconstructions: mutationless edges are
   contracted, then `cphs.prepare_tree` sets the inner states by small maximum parsimony
   (Algorithm 2), the root to unmutated, and the times by the ML branch-length estimator
   (Appendix A.1).
4. **Scores.** $\lambda$ is estimated from the fraction of mutated characters and $q$ is the known
   collision probability (Appendix A.1). Each tree gets cPHS (Eq. 13), parsimony, the parsimony
   tail probability (Appendix D), the log-likelihood (Cassiopeia's
   `calculate_likelihood_continuous`, with no heritable missing data and the mutation rate
   estimated by Cassiopeia), and its normalized RF distance and triplets score with respect to the
   preprocessed GT tree.

## Output

`run_simulation.py` writes one row per tree: the setting (`n`, `subsampling_ratio`, `k`, `m`,
`rho`), the realization (`seed`; `seed_used` differs if the GT simulation had to be redrawn, e.g.
after extinction), `tree` (`gt` or the solver), and

| Column | Score |
|---|---|
| `rf`, `triplets` | normalized RF distance and triplets score w.r.t. the GT tree |
| `parsimony`, `parsimony_pvalue` | $M(\mathcal{T}^S)$ and the tail probability $P[M \geq M(\mathcal{T}^S) \mid \mathcal{T}]$ |
| `likelihood` | continuous-time log-likelihood (Cassiopeia's `calculate_likelihood_continuous`) |
| `cPHS`, `cPHS_A3` | cPHS of Eq. (13), and the variant of Appendix A.3 that excludes leaf pairs without homoplasy |
| `lam_hat`, `q` | estimated mutation rate and collision probability used for the scores |
| `time_s` | seconds spent scoring the tree |

`plot_figures.py` uses `cPHS` by default; `--cphs-column cPHS_A3` switches to the Appendix A.3
variant. The distances of Eqs. (4) to (7)
are derived from these columns: $d_{RF}$ = `rf`, $d_{tri}$ = 1 − `triplets`, and $d_P$, $d_L$
from the parsimony and log-likelihood of each tree relative to the GT tree of the same
realization.
