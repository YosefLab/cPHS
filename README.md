# cPHS

Code and data for the paper

> Zilber, Prillo, Neumeier, Yosef, Nadler. *Can you trust your reconstructed lineage tree?
> A homoplasy-based approach for irreversible evolution.*

This repository contains the implementation of the cPHS score and the analysis of the KPTracer
lung adenocarcinoma data (Section 6 and Appendices L to P).
A maintained implementation of cPHS is being integrated into
[Cassiopeia](https://github.com/YosefLab/Cassiopeia) (`calculate_cPHS`, PR #289).
The code here is the exact code that produced the numbers in the paper.

## Contents

| Folder | Contents |
|---|---|
| `cphs/` | Core cPHS functions used for all KPTracer results |
| `examples/` | Minimal example: score one tumor's tree |
| `kptracer/` | KPTracer pipeline: reconstructions, random trees, scores, NNI perturbation, per-site state counts |
| `calibration/` | Threshold-calibration simulations and Tables S1 to S4 (Appendix M) |
| `figures/` | Scripts for the figures and tables of Section 6 and Appendices L to P |
| `results/` | Result tables underlying the paper |
| `trees/` | Trees scored in the paper (Newick with branch lengths) |

## Installation

All results were produced with Python 3.10 and Cassiopeia 2.1.0:

```bash
conda env create -f environment.yml
conda activate cass
```

Cassiopeia 3.0 moved the branch-length estimator used here (`IIDExponentialMLE`) to the separate
ConvexML package and changed several solvers, so use the environment above to reproduce the paper.

## Data

The KPTracer data of Yang et al. (2022) are on Zenodo, DOI
[10.5281/zenodo.5847462](https://doi.org/10.5281/zenodo.5847462). Download and extract
`KPTracer-Data.tar.gz`. The scripts read the per-tumor character matrices, indel priors and
deposited trees from `KPTracer-Data/trees/`.

## Quick start

```bash
python examples/score_one_tree.py --data ~/KPTracer-Data --tumor 3435_NT_T3
# 3435_NT_T3: n=322, k=30, lambda=2.432, q=0.0966, cPHS=9.353e-10
```

This reproduces the value in `results/merged_all_tumors.csv` for the Cassiopeia tree of this tumor.

## Paper figures and tables

| Paper item | Script | Input |
|---|---|---|
| Section 6, all KP numbers | `kptracer/kp_phs_all.py` | KPTracer data, produces `results/merged_all_tumors.csv` |
| Section 6, number of mutated states $m$ | `kptracer/count_states.py` | KPTracer priors, produces `results/m_states/` |
| Fig 6 (method ranking) | `figures/regen_02.py` | `results/merged_all_tumors.csv` |
| Tables S1 to S4 (threshold calibration) | `calibration/tables_S1_S4.py` | outputs of `calibration/threshold_simulation.py` |
| Table S5 (adopted thresholds) | summary of Tables S1 to S4 | |
| Fig S17 (reliability vs $q$) | `figures/fig_S17_reliability_by_q.py` | `results/merged_all_tumors.csv` |
| Fig S18 (reliability vs $k$) | `figures/regen_19c_byk.py` | `results/merged_all_tumors.csv` |
| Tables S6, S7 (best-method concordance) | `figures/concordance_final.py` | `results/merged_all_tumors.csv` |
| Fig S19 (predictors of separation) | `figures/regen_10b.py` | `results/merged_all_tumors.csv` |
| Table S8 (quality factors) | `figures/quality_vs_cphs.py` | `results/merged_all_tumors.csv`, KPTracer data |
| Appendix O, simulation-based validation | `figures/sim_vs_kp_plot_v3.py` | calibration simulation outputs |
| Table S9 (flagged tumors under NNI) | `figures/table_S9_flagged.py` | `results/merged_all_tumors.csv` |
| Table S10 (NNI degradation) | `figures/table_S10_degradation.py` | `results/merged_all_tumors.csv` |
| Figs S20, S21 (NNI perturbation) | `kptracer/nni_analysis_v2.py`, `figures/nni_plots_v2.py` | KPTracer data; results in `results/nni_v2/` |
| Supplementary table `KPTracer_summary.xlsx` | | `results/KPTracer_summary.xlsx` |

Figures were rendered at 300 dpi with `figures/fig_render_300.py`.

## Results

| File | Contents |
|---|---|
| `results/merged_all_tumors.csv` | cPHS, parsimony and likelihood for every scored tree (six reconstructions, five random trees and NNI-perturbed Cassiopeia trees per tumor), with $n$, $k$, $\lambda$, $q$ |
| `results/table7_exclusion_criteria.csv` | The 63 analyzed tumors (Appendix L) |
| `results/KPTracer_summary.xlsx` | Per-tumor summary of the 63 analyzed tumors |
| `results/m_states/` | Number of mutated states per site (`m_per_site.csv`) and per tumor (`m_per_tumor.csv`) |
| `results/nni_v2/` | NNI perturbation results (Appendix P): one row per perturbed tree, including its Newick string; 34 tumors, Cassiopeia and SMJ, 8 levels, 20 replicates |

## Trees

`trees/{tumor}/` contains, for each of the 63 analyzed tumors, the MaxCut, SMJ, Spectral and
Spectral Greedy reconstructions and the five random trees, as Newick with branch lengths
(produced by `kptracer/save_kp_trees.py`). Inner-node states are not stored in Newick; they are
determined by small maximum parsimony on the character matrix (`cphs.prepare_tree`).

The Cassiopeia and Neighbor Joining trees are those deposited by Yang et al. (2022)
(`KPTracer-Data/trees/{tumor}_tree.nwk` and `neighbor_joining/{tumor}_tree_nj.processed.tree`)
and are not duplicated here. The perturbed trees of the NNI analysis are in `results/nni_v2/`.

## Implementation notes

* **Collision probability.** The deposited indel priors do not sum to one per site. They are
  normalized per site, $q_i = \sum_j q_{ij}^2$ is computed for each site, and $q$ is the mean over sites.
* **Stochastic solvers.** In Cassiopeia 2.1.0, MaxCut, SMJ and the two spectral solvers can return
  slightly different trees on repeated runs, and the original run did not fix a seed. MaxCut depends
  on the NumPy random state, so `save_kp_trees.py` seeds it and searches seeds until the tree's
  parsimony equals the value in `merged_all_tumors.csv`, and records the seed. For SMJ and the
  spectral solvers the variation is not controlled by seeding; their trees are deposited as
  regenerated. Trees whose parsimony differs from the scored tree are listed in
  `trees/unmatched.csv`, with their recomputed cPHS and whether the accept/reject decision changes.
* **Random trees** are fully reproducible (`random.seed(r)`, $r = 0, \ldots, 4$).
* **Calibration, $\varepsilon$.** The cutoff printed as 0.67 in Tables S1 to S4 is $2/3$ exactly.
* **Rounding in Tables S1 to S4.** The published values were rounded by hand from three-decimal
  output, so `calibration/tables_S1_S4.py` differs from them in a few cells by 0.01 in balanced
  accuracy or by one unit in the last digit of the median cPHS (for example, BA(0.05) $= 0.57$ for
  $n = 500$, $k = 15$ in Table S1, published as 0.58; exact value 0.5749).
* **Table S10** uses the uniform threshold $t = 0.05$ and the 48 low-$q$ tumors of the earlier
  71-tumor cohort (all tumors in `merged_all_tumors.csv` except three, with $k \geq 7$ and $q < 0.35$).
* **Script versions.** Scripts are deposited as they were run; the only later change is replacing
  the original absolute paths with `KPTRACER_DATA` and repository-relative paths (see git history).

## Running the scripts

All scripts read the KPTracer data from the directory given by the environment variable
`KPTRACER_DATA` (default `~/KPTracer-Data`; `kp_phs_all.py` downloads it there if it is missing).
Everything the scripts write goes to `output/`, so the deposited files in `results/` and `trees/`
are never overwritten; compare the two to verify a rerun.

```bash
export KPTRACER_DATA=~/KPTracer-Data
```

The full analysis, in order:

1. **Score all trees** (six reconstructions, random trees, NNI-perturbed Cassiopeia trees):
   `python kptracer/kp_phs_all.py`, which reproduces `results/merged_all_tumors.csv`.
   cPHS is $O(kn^2)$ per tree; the largest tumors take hours, so run tumors in parallel on a cluster.
2. **Save the trees**: `python kptracer/save_kp_trees.py --tumor <tumor>` for each tumor in
   `results/table7_exclusion_criteria.csv`, producing `output/trees/<tumor>/`.
3. **NNI perturbation** (Appendix P): `python kptracer/nni_analysis_v2.py --tumor <tumor> --reps 20`,
   producing `output/nni_v2/` (deposited as `results/nni_v2/`).
4. **Number of mutated states**: `python kptracer/count_states.py` (deposited as `results/m_states/`).
5. **Threshold calibration** (Appendix M):
   `python calibration/threshold_simulation.py --n <n> --k <k> --reps 100 --output <file>` for each
   configuration, then `python calibration/tables_S1_S4.py calibration/results`.
6. **Figures and tables**: the scripts in `figures/` (see the table above) read `results/` and write to `output/figures/`.

Steps 1 to 5 are slow; steps 6 and the table scripts run in seconds to minutes from the deposited
results, so the paper's figures and tables can be reproduced without rerunning the pipeline.

## Citation

If you use this code, please cite:

> Zilber P, Prillo S, Neumeier Y, Yosef N, Nadler B. Can you trust your reconstructed lineage tree?
> A homoplasy-based approach for irreversible evolution. *Systematic Biology* (in revision).

## License

MIT
