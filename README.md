# cPHS on KPTracer data

Code and data for the KPTracer analysis (Section 6 and Appendices L–P) of

> Zilber, Prillo, Neumeier, Yosef, Nadler. *Can you trust your reconstructed lineage tree?
> A homoplasy-based approach for irreversible evolution.*

The simulation code (Sections 4–5, Appendices A–K) is deposited separately: **[link to be added]**.
A maintained implementation of cPHS is being integrated into
[Cassiopeia](https://github.com/YosefLab/Cassiopeia) (`calculate_cPHS`, PR #289).

## Contents

| Folder | What |
|---|---|
| `cphs/` | Core cPHS functions used for all results in the paper |
| `examples/` | Minimal example: score one tree |
| `kptracer/` | KPTracer pipeline: reconstructions, random trees, scores, NNI perturbation |
| `calibration/` | Threshold-calibration simulations (Appendix M) |
| `figures/` | Scripts for the figures and tables of Section 6 and Appendices L–P |
| `results/` | Result tables underlying the paper |
| `trees/` | All trees scored in the paper (Newick with branch lengths) |

## Installation

All results were produced with Cassiopeia 2.1.0 and Python 3.10:

```bash
conda env create -f environment.yml
conda activate cass
```

## Data

The KPTracer data of Yang et al. (2022) are available on Zenodo
(DOI [10.5281/zenodo.5847462](https://doi.org/10.5281/zenodo.5847462)).
Download and extract `KPTracer-Data.tar.gz`; the scripts expect the extracted `KPTracer-Data/` folder.

## Quick start

```bash
python examples/score_one_tree.py --data ~/KPTracer-Data --tumor 3467_Lkb1_T1
```

## Paper figures and tables

*(to be completed)*

## License

MIT
