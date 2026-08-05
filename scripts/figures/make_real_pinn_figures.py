"""
AP-PEN-on-real-data figures for the real WTI application chapter.

Three standalone figures (no multi-panel composites -- see the sizing note
in make_kalman_figures.py, which this module follows too):

  real1  AP-PEN on real WTI data        -- delta_hat(t) for all three real-data
                                          variants (MLP, AP-PEN, AP-PEN (ARB)),
                                          crisis-regime shaded, no ground truth
                                          to score against (delta_t is genuinely
                                          latent here -- see module docstring)
  real2  AP-PEN vs Kalman filter        -- the actual benchmark comparison: all
                                          three variants overlaid against the
                                          Kalman filter's own delta_hat(t) on
                                          their common dates
  real2b AP-PEN vs Kalman, summary      -- companion to real2: corr/RMSE of
                                          each variant against that benchmark,
                                          sized to sit alongside a same-width
                                          partner figure rather than full width

Run:  python scripts/figures/make_real_pinn_figures.py
Reads data/output/real_data/results/results_{MLP,APPINN,APPINN_ARB}.pkl
(written by the real-WTI calibration cells of
notebooks/AP-PEN.ipynb, 40k epochs) and
data/output/kalman/results/results_kalman.pkl. Writes figures/real_pinn/{pdf,png}.

WHY NO ERROR PANEL AGAINST GROUND TRUTH
----------------------------------------
Every synthetic-data figure in this repo (pinn2a, pinn3) can plot a signed
error against delta_true because the Monte Carlo simulator retains it. Real
WTI data has no such thing -- delta_t is latent by construction (CLAUDE.md
S1: "the true delta_t path must never leak into anything the inversion
sees," and for real data there is no true path at all). The Kalman filter is
the only available reference, and it is itself an estimate, not ground
truth -- hence "vs Kalman filter", never "error vs truth", in real2's
labelling.

PROVENANCE
----------
real1 and real2 both read delta_hat and metrics straight from the results
pickles -- nothing is retrained or recomputed. real2's corr/RMSE summary is
a plain aggregation (correlation coefficient, root-mean-square difference)
of the two delta_hat series on their common dates, mirroring exactly the
comparison already printed by the notebook's own real-WTI cell.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gs_wamol.utils.thesis_style import (
    C_BASELINE, C_MODEL, INK, PALETTE,
    direct_label, figsize, save, use_thesis_style,
)

REPO = Path(__file__).resolve().parents[2]
REAL_RESULTS_DIR = REPO / "data" / "output" / "real_data" / "results"
KALMAN_RESULTS = REPO / "data" / "output" / "kalman" / "results" / "results_kalman.pkl"
FIGDIR = REPO / "figures" / "real_pinn"

TAGS = ["MLP", "APPINN", "APPINN_ARB"]
DISPLAY_NAME = {"MLP": "MLP", "APPINN": "AP-PEN", "APPINN_ARB": "AP-PEN (ARB)"}
# MLP in neutral grey, not a PALETTE hue: these figures also show the Kalman
# filter, which owns PALETTE["aqua"] (C_BASELINE) by the thesis-wide colour
# convention (thesis_style.py) -- reusing aqua for MLP here (as make_pinn_
# figures.py does in the no-Kalman synthetic-data figures) would put two
# different series in the same colour on the same plot.
COLOUR = {"MLP": INK["secondary"], "APPINN": C_MODEL, "APPINN_ARB": PALETTE["blue"]}
LINESTYLE = {"MLP": (0, ()), "APPINN": (0, (3, 1, 1, 1)), "APPINN_ARB": (0, (4, 1, 1, 1, 1, 1))}


def load_real(tag):
    with open(REAL_RESULTS_DIR / f"results_{tag}.pkl", "rb") as f:
        return pickle.load(f)


def load_kalman():
    with open(KALMAN_RESULTS, "rb") as f:
        kf = pickle.load(f)
    return {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in kf.items()}


def _is_crisis(dates):
    dt = pd.DatetimeIndex(dates)
    return np.asarray((dt.year == 2015) | (dt.year == 2016) | ((dt.year == 2020) & (dt.month == 4)))


def _shade_crisis(ax, dates, crisis):
    in_run = False
    for i, c in enumerate(crisis):
        if c and not in_run:
            start, in_run = dates[i], True
        elif not c and in_run:
            ax.axvspan(start, dates[i], color=INK["muted"], alpha=0.10, lw=0, zorder=0)
            in_run = False
    if in_run:
        ax.axvspan(start, dates[-1], color=INK["muted"], alpha=0.10, lw=0, zorder=0)


# ---------------------------------------------------------------------------
# Figure 1 -- AP-PEN on real WTI data
# ---------------------------------------------------------------------------
def fig_pinn_real_data():
    results = {tag: load_real(tag) for tag in TAGS}
    dates = pd.DatetimeIndex(results["MLP"]["dates"])
    crisis = _is_crisis(results["MLP"]["dates"])

    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.48))
    _shade_crisis(ax, dates, crisis)
    ax.axhline(0.0, color=INK["muted"], lw=0.7, ls=":", zorder=1)

    for tag in TAGS:
        d = results[tag]["delta_hat"]
        ax.plot(dates, d, color=COLOUR[tag], ls=LINESTYLE[tag], lw=1.0, alpha=0.9, zorder=3)

    ax.set_ylabel(r"$\hat\delta_\phi(t)$")
    ax.annotate("no ground truth exists for real data --\ncompare against the Kalman filter"
                " (Fig.~real2), not a true path",
                xy=(0.015, 0.05), xycoords="axes fraction", color=INK["secondary"],
                ha="left", va="bottom", fontsize=8)

    q = int(0.06 * len(dates))
    for tag in TAGS:
        direct_label(ax, dates[q], results[tag]["delta_hat"][q], DISPLAY_NAME[tag],
                     COLOUR[tag], dx=0, dy=10 if tag != "MLP" else -12, ha="left")

    fig.tight_layout()
    return fig


def _vs_kalman_stats():
    """Shared by fig_pinn_vs_kalman and fig_pinn_vs_kalman_summary so the two
    never compute the comparison independently and risk disagreeing."""
    results = {tag: load_real(tag) for tag in TAGS}
    kf = load_kalman()

    pinn_dates = pd.DatetimeIndex(results["MLP"]["dates"])
    kf_dates = pd.DatetimeIndex(kf["dates"])
    common = kf_dates.intersection(pinn_dates)
    kf_pos = kf_dates.get_indexer(common)
    pinn_pos = pinn_dates.get_indexer(common)
    kf_common = kf["delta_hat"][kf_pos]

    stats = {}
    for tag in TAGS:
        d = results[tag]["delta_hat"][pinn_pos]
        corr = float(np.corrcoef(d, kf_common)[0, 1])
        rmse = float(np.sqrt(np.mean((d - kf_common) ** 2)))
        stats[tag] = {"corr": corr, "rmse": rmse}

    return results, kf, common, kf_pos, pinn_pos, kf_common, stats


# ---------------------------------------------------------------------------
# Figure 2 -- AP-PEN vs Kalman filter, time series
# ---------------------------------------------------------------------------
def fig_pinn_vs_kalman():
    results, kf, common, kf_pos, pinn_pos, kf_common, _ = _vs_kalman_stats()

    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.48))
    crisis = _is_crisis(kf["dates"][kf_pos])
    _shade_crisis(ax, common, crisis)
    ax.plot(common, kf_common, color=C_BASELINE, lw=1.4, zorder=4)
    for tag in TAGS:
        ax.plot(common, results[tag]["delta_hat"][pinn_pos], color=COLOUR[tag],
                ls=LINESTYLE[tag], lw=1.0, alpha=0.85, zorder=3)
    ax.set_ylabel(r"$\hat\delta_t$")
    direct_label(ax, common[-1], kf_common[-1], "Kalman filter", C_BASELINE,
                 dx=4, dy=0, ha="left")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2b -- AP-PEN vs Kalman filter, corr/RMSE summary (companion to real2)
# ---------------------------------------------------------------------------
def fig_pinn_vs_kalman_summary():
    _, _, _, _, _, _, stats = _vs_kalman_stats()

    fig, ax = plt.subplots(1, 1, figsize=figsize("half", ratio=0.85))
    tag_order = sorted(TAGS, key=lambda t: -stats[t]["corr"])
    y = np.arange(len(tag_order))
    corr_vals = [stats[t]["corr"] for t in tag_order]
    ax.barh(y, corr_vals, color=[COLOUR[t] for t in tag_order], height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_NAME[t] for t in tag_order], fontsize=9)
    ax.invert_yaxis()  # best correlation at the top, reading down
    ax.set_xlabel("corr. vs Kalman", fontsize=9)
    ax.set_xlim(0, 1)
    for yi, t in zip(y, tag_order):
        ax.annotate(f"RMSE {stats[t]['rmse']:.3f}", xy=(stats[t]["corr"] + 0.02, yi),
                    va="center", ha="left", fontsize=8, color=INK["secondary"])

    fig.tight_layout()
    return fig, stats


def main():
    use_thesis_style()

    fig = fig_pinn_real_data()
    path = save(fig, "real1_pinn_recovery", directory=FIGDIR)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO)}")

    fig = fig_pinn_vs_kalman()
    path = save(fig, "real2_pinn_vs_kalman", directory=FIGDIR)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO)}")

    fig, stats = fig_pinn_vs_kalman_summary()
    path = save(fig, "real2b_pinn_vs_kalman_summary", directory=FIGDIR)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO)}")
    for tag, s in stats.items():
        print(f"  {tag:12s} corr {s['corr']:+.4f}  RMSE {s['rmse']:.4f}")


if __name__ == "__main__":
    main()
