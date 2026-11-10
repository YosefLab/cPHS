#!/bin/bash
# Regenerate all figures (300 dpi) and tables of Section 6 and Appendices L-P from the deposited results.
# Figures go to output/figures/, tables to output/tables/.   Usage: bash figures/make_all.sh
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
export KPTRACER_DATA=${KPTRACER_DATA:-$HOME/KPTracer-Data}
mkdir -p output/figures output/tables
fig() { echo "== $1"; $PY figures/fig_render_300.py "$2"; }
tab() { echo "== $1"; shift; bash -c "$*"; }
fig "Fig 6"         figures/regen_02.py
fig "Fig S17"       figures/fig_S17_reliability_by_q.py
fig "Fig S18"       figures/regen_19c_byk.py
fig "Fig S19"       figures/regen_10b.py
fig "Figs S20, S21" figures/nni_plots_v2.py
tab "Tables S1-S4"  "$PY calibration/tables_S1_S4.py calibration/results > output/tables/tables_S1_S4.txt"
tab "Tables S6, S7" "$PY figures/concordance_final.py > output/tables/tables_S6_S7.txt"
tab "Table S8"      "$PY figures/quality_vs_cphs.py > output/tables/table_S8.txt"
tab "Table S9"      "$PY figures/table_S9_flagged.py results/merged_all_tumors.csv > output/tables/table_S9.tex"
tab "Table S10"     "$PY figures/table_S10_degradation.py results/merged_all_tumors.csv output/tables/table_S10.tex"
echo "done: $(ls output/figures/Fig*.png | wc -l) paper figures in output/figures/, $(ls output/tables | wc -l) table files in output/tables/"
