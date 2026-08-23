"""
Synthetic-vs-real comparison: distribution of observed log-futures prices
across maturities, as a 3D ridgeline ("joy plot"), one standalone figure per
data source:

  data_maturity_distribution_synthetic  -- the synthetic panel
  data_maturity_distribution_real       -- the real WTI panel

A third panel completing the "why is the synthetic setup still identifiable"
argument already exists and is NOT regenerated here: mc3b_identifiability_scatter
(figures/monte_carlo_sim/, from make_mc_figures.py) is the slope-delta map
these two are meant to sit alongside. Compose the three as a LaTeX subfigure
row/grid rather than here -- see the note below.

WHY TWO STANDALONE FIGURES, NOT ONE COMBINED PDF
---------------------------------------------------
make_mc_figures.py and make_real_data_figures.py both draw one figure per PDF
and rely on LaTeX \\subfigure to compose pairs -- every other multi-panel
figure in this thesis follows that convention, and an earlier combined-PNG
version of this figure (three panels drawn together in one matplotlib figure)
did not: cramped panels, mismatched 2D/3D aspect ratios, and axis labels
fighting each other for space. Splitting back into one-PDF-per-panel and
letting LaTeX (subcaption package, real page geometry) handle sizing and the
(a)/(b)/(c) labels is both more consistent with the rest of the document and
produces a cleaner result.

WHAT THESE SHOW AND WHY A RIDGELINE, NOT A BAR GRID
-------------------------------------------------------
Per maturity, a smoothed density (Gaussian KDE) of every observed log futures
price at that maturity slot: 100 paths x 1001 dates for the synthetic panel,
one value per trading date for the real WTI panel. Stacking these along the
maturity axis in 3D is the intuitive "grid" a first attempt would reach for
as solid histogram bars (bar3d) -- but solid bars occlude: with 8-12
maturities the back columns hide behind the front ones and the very shape
the figure exists to show (how the distribution's spread/location changes
with maturity) becomes unreadable. Filled ridge curves (PolyCollection via
add_collection3d) avoid this: each maturity's profile stays a legible
silhouette against the ones behind it.

WHY THIS SCRIPT IS SEPARATE FROM make_mc_figures.py / make_real_data_figures.py
-----------------------------------------------------------------------------------
Those two scripts each enforce a strict single-provenance rule in their own
docstrings (synthetic-only / real-only respectively, see PROVENANCE RULE
below). This figure's whole point is showing both side by side, so it cannot
honestly live in either without violating the rule that script enforces.

PROVENANCE RULE (do not relax this)
--------------------------------------
Synthetic panel: every mark comes from data/input/synthetic/mc_data.pkl
(log_F_obs), or a KDE-smoothed histogram of it. No resimulation.
Real panel: every mark comes from data/input/real/wti_analysis_ready.csv via
_wti_data.load_wti_panel (the same loader the Kalman filter and real-data
PINN figures use, so the maturity grid here matches those elsewhere in the
thesis), or a KDE-smoothed histogram of it. Not reproducible from code --
tracked as-is (see CLAUDE.md).

Run:  python scripts/figures/make_data_comparison_figures.py
Reads data/input/synthetic/mc_data.pkl and data/input/real/wti_analysis_ready.csv,
writes figures/data_comparison/data_maturity_distribution_{synthetic,real}.{pdf,png}.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wti_data import load_wti_panel  # noqa: E402

from gs_wamol.utils.thesis_style import (  # noqa: E402
    INK, SEQ_BLUE, figsize, save, use_thesis_style,
)

REPO = Path(__file__).resolve().parents[2]
MC_DATA = REPO / "data" / "input" / "synthetic" / "mc_data.pkl"
FIGDIR = REPO / "figures" / "data_comparison"


def load_synthetic():
    with open(MC_DATA, "rb") as f:
        d = pickle.load(f)
    taus = np.asarray(d["taus"])                        # (12,)
    logF = np.asarray(d["log_F_obs"])                    # (100, 1001, 12)
    return taus, logF.reshape(-1, logF.shape[-1])          # (100*1001, 12)


def load_real():
    _, _, _, taus, logF, _, _ = load_wti_panel()          # taus (8,), logF (n_dates, 8)
    return taus, logF


def ridge_polys(values: np.ndarray, taus: np.ndarray, n_pts: int = 200, bw_frac: float = 0.06):
    """One Gaussian-KDE density curve per maturity column of `values`, each
    returned as a closed polygon (baseline at z=0) ready for add_collection3d.

    SHAPES: values (n_obs, n_maturities); returns a list of (n_pts+2, 2) arrays.
    """
    lo, hi = np.nanmin(values), np.nanmax(values)
    grid = np.linspace(lo, hi, n_pts)
    bw = bw_frac * (hi - lo)

    polys, peak = [], 0.0
    for k in range(len(taus)):
        v = values[:, k][:, None]                        # (n_obs, 1)
        dens = np.exp(-0.5 * ((grid[None, :] - v) / bw) ** 2).sum(axis=0)
        dens /= v.shape[0] * bw * np.sqrt(2 * np.pi)      # normalise to a density
        peak = max(peak, dens.max())
        verts = np.column_stack([grid, dens])
        verts = np.vstack([[grid[0], 0.0], verts, [grid[-1], 0.0]])
        polys.append(verts)
    return polys, peak


def fig_ridge(taus, values):
    """Standalone full-width 3D ridgeline figure -- same sizing/view-angle
    family as the other standalone 3D figures in this thesis
    (mc2c_term_structure_3d, real_data_term_structure_3d)."""
    polys, peak = ridge_polys(values, taus)
    colors = [SEQ_BLUE(u) for u in np.linspace(0.15, 0.95, len(taus))]

    fig = plt.figure(figsize=figsize("full", ratio=0.62))
    ax = fig.add_subplot(111, projection="3d")

    coll = PolyCollection(polys, facecolors=colors, edgecolors=INK["primary"],
                           linewidths=0.5, alpha=0.88)
    ax.add_collection3d(coll, zs=taus, zdir="x")

    ax.set_xlim(taus[0], taus[-1])
    ax.set_ylim(values.min(), values.max())
    ax.set_zlim(0, peak * 1.05)

    # No x-axis label TEXT: mplot3d's tight-bbox extent estimate for a
    # rotated axis label at this view angle clips it against the figure edge
    # regardless of labelpad or margin -- the same mplot3d bug already
    # documented (and worked around the same way, tick numbers only +
    # caption names the axis) for the y/z labels of mc2c_term_structure_3d
    # and real_data_term_structure_3d. Tick numbers stay and are unambiguous
    # next to a caption stating the x-axis is maturity in years.
    # No y-axis label TEXT either, same reasoning as the dropped x-label
    # above: labelpad/margin changes do not touch this, since savefig's
    # tight bbox recomputes from painted content regardless of reserved
    # whitespace. Tick numbers (2.5-6.0) stay; the caption names both axes.
    ax.set_zticklabels([])
    ax.tick_params(labelsize=8, pad=1)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((2.2, 1, 0.85), zoom=1.15)
    ax.annotate("darker = longer maturity", xy=(0.02, 0.85), xycoords="axes fraction",
                fontsize=8, color=INK["secondary"])

    fig.subplots_adjust(left=0.02, right=0.90, top=0.92, bottom=0.05)
    return fig


def main():
    use_thesis_style()

    taus_syn, logF_syn = load_synthetic()
    fig = fig_ridge(taus_syn, logF_syn)
    path = save(fig, "data_maturity_distribution_synthetic", directory=FIGDIR)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO)}")

    taus_real, logF_real = load_real()
    fig = fig_ridge(taus_real, logF_real)
    path = save(fig, "data_maturity_distribution_real", directory=FIGDIR)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
