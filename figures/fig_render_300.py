#!/usr/bin/env python3
"""
fig_render_300.py  --  force 300 dpi (and larger titles where not hardcoded)
on any of the KP figure scripts, without editing them.

USAGE (on WEXAC):
  PY=/home/projects/nyosef/yaaraneu/.conda/envs/cass/bin/python
  $PY ~/threshold_calibration/fig_render_300.py ~/threshold_calibration/regen_02.py
  $PY ~/threshold_calibration/fig_render_300.py ~/threshold_calibration/kp_phs_all.py
  $PY ~/threshold_calibration/fig_render_300.py ~/quality_vs_cphs.py
  $PY ~/threshold_calibration/fig_render_300.py ~/threshold_calibration/nni_plots.py
  ... one call per script ...

It works by:
  1. setting matplotlib to Agg and applying title-font rcParams BEFORE the
     target script imports pyplot;
  2. monkey-patching Figure.savefig and pyplot.savefig so that EVERY save is
     forced to dpi=300, overriding any hardcoded dpi=150 in the script;
  3. exec'ing the target script in its own globals so all its paths,
     functions, and __main__ logic run unchanged.

Notebooks (.ipynb) cannot be run this way -- for 19a/19b/04b/10b run the
notebook cells, or export the relevant cells to a .py first (see note at end).
"""
import os
import sys
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

FORCE_DPI = 300

# --- dpi + a modest subplot-title size. axes.titlesize only affects
#     set_title calls that pass NO explicit fontsize; panels that hardcode
#     fontsize= (e.g. 04b's fontsize=20) still override this. 12 shrinks the
#     default-size subplot titles slightly without causing overlap. ---
plt.rcParams.update({
    "savefig.dpi": FORCE_DPI,
    "figure.dpi": FORCE_DPI,
    "axes.titlesize": 12,
})

# --- force dpi=300 on every save, no matter what the script requests ---
_orig_fig_savefig = Figure.savefig
def _patched_fig_savefig(self, *args, **kwargs):
    kwargs["dpi"] = FORCE_DPI
    return _orig_fig_savefig(self, *args, **kwargs)
Figure.savefig = _patched_fig_savefig

_orig_plt_savefig = plt.savefig
def _patched_plt_savefig(*args, **kwargs):
    kwargs["dpi"] = FORCE_DPI
    return _orig_plt_savefig(*args, **kwargs)
plt.savefig = _patched_plt_savefig

# --- nudge ONLY the figure-level suptitle up a little, regardless of what
#     fontsize the script passed. Subplot titles (ax.set_title) are untouched.
#     Scales the requested suptitle size by SUPTITLE_SCALE; if the script
#     passed no fontsize, falls back to a fixed bumped size. ---
SUPTITLE_SCALE = 1.25
_SUPTITLE_FALLBACK = 18  # used only when no fontsize was given

from matplotlib import rcParams as _rc
def _bump(kwargs):
    fs = kwargs.get("fontsize", kwargs.get("size", None))
    if fs is None:
        base = _rc.get("figure.titlesize", _SUPTITLE_FALLBACK)
        # rcParams can be a string like 'large'; only scale if numeric
        try:
            base = float(base)
        except (TypeError, ValueError):
            base = _SUPTITLE_FALLBACK
        kwargs["fontsize"] = base * SUPTITLE_SCALE
    else:
        try:
            kwargs["fontsize"] = float(fs) * SUPTITLE_SCALE
            kwargs.pop("size", None)
        except (TypeError, ValueError):
            pass  # named size like 'large'; leave as-is
    return kwargs

_orig_fig_suptitle = Figure.suptitle
def _patched_fig_suptitle(self, *args, **kwargs):
    return _orig_fig_suptitle(self, *args, **_bump(kwargs))
Figure.suptitle = _patched_fig_suptitle

_orig_plt_suptitle = plt.suptitle
def _patched_plt_suptitle(*args, **kwargs):
    return _orig_plt_suptitle(*args, **_bump(kwargs))
plt.suptitle = _patched_plt_suptitle

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: fig_render_300.py <path-to-figure-script.py>")
    target = sys.argv[1]
    # let the target see itself as the main program; strip our own argv
    sys.argv = [target] + sys.argv[2:]
    # run in a fresh namespace as __main__ so its `if __name__ == '__main__'` fires
    runpy.run_path(target, run_name="__main__")
    print(f"[fig_render_300] finished {target} at {FORCE_DPI} dpi")
