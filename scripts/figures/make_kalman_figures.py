"""
Kalman filter benchmark figures for the real WTI application chapter.

Three standalone figures (no multi-panel composites -- each is its own
figure, full text-width; see module NOTE on this project's house style):

  kalman1  filtered convenience yield   -- delta_hat(t), crisis regime shading
  kalman1b innovation RMSE              -- per-date fit quality across the
                                          whole panel, companion to kalman1
  kalman2  term-structure fit, 3D       -- the Kalman innovation (observed -
                                          fitted log futures price) as a
                                          surface over (date, maturity): shows
                                          not just that the filter fits well
                                          on average, but exactly when/where
                                          across the curve it does not

NOTE ON FIGURE SIZING
----------------------
This thesis's figures are never internal multi-panel composites (no (a)/(b)
sub-axes sharing one float) -- each figure is either full text-width on its
own, or sized to sit next to ONE partner figure as two separate side-by-side
floats. kalman1/kalman1b were originally one two-panel figure; they are full
width here rather than a side-by-side pair because both are dense ~11-year
daily time series that compress unreadably at half width.

Run:  python scripts/figures/make_kalman_figures.py
Reads data/output/kalman/results/results_kalman.pkl (written by
notebooks/kalman_filter.ipynb) and data/input/real/wti_analysis_ready.csv
(via _wti_data.load_wti_panel, for the observed side of kalman2's fit
comparison -- the pickle stores innovations but not the raw observed panel
they were computed against). Writes figures/kalman/{pdf,png}.

PROVENANCE
----------
kalman1's delta_hat, delta_var and dates are read straight from the pickle;
the crisis shading is the same date-rule (2015, 2016, or April 2020) used
throughout the thesis, not fitted to anything. The bottom panel's per-date
innovation RMSE is a plain aggregation (root-mean-square over the 8 maturity
columns) of the pickle's own `innovations` array.

kalman2's "fitted" surface is derived, not stored directly: the pickle keeps
the innovation (observed - fitted) but not the fitted curve itself, so
fitted = observed - innovation, with "observed" read from the same WTI panel
the Kalman filter notebook itself loads (_wti_data.load_wti_panel mirrors its
loader exactly -- see that module's docstring). Both operations are linear
and exact, not model fits performed here.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

from _wti_data import load_wti_panel
from gs_wamol.utils.thesis_style import (
    C_BASELINE, DIV_BLUE_RED, INK,
    figsize, save, use_thesis_style,
)

REPO = Path(__file__).resolve().parents[2]
KALMAN_RESULTS = REPO / "data" / "output" / "kalman" / "results" / "results_kalman.pkl"
FIGDIR = REPO / "figures" / "kalman"


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
# Figure 1 -- filtered convenience yield
# ---------------------------------------------------------------------------
def fig_delta_recovery(kf):
    dates = pd.DatetimeIndex(kf["dates"])
    delta_hat = kf["delta_hat"]
    delta_sd = np.sqrt(kf["delta_var"])
    crisis = _is_crisis(kf["dates"])

    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.42))
    _shade_crisis(ax, dates, crisis)

    # The filter's steady-state posterior s.d. is ~3e-4 here -- four orders of
    # magnitude below delta_hat's own range (~0.5), so a +/-1.96sd band would
    # render as an invisible hairline. Rather than fake visibility, state the
    # number directly: at this scale, essentially all of the day-to-day
    # spread visible here is genuine variation in the fitted state, not
    # filter uncertainty -- that IS the finding, not a plotting compromise.
    ax.axhline(kf["psi_hat"]["alpha_P"], color=INK["secondary"], lw=1.0, ls="--", zorder=1)
    ax.plot(dates, delta_hat, color=C_BASELINE, lw=1.4, zorder=3)
    ax.set_ylabel(r"$\hat\delta_t$ (Kalman)")
    ax.annotate(rf"steady-state filter s.d. $\approx {delta_sd.mean():.1e}$"
                "\n(too small to render as a band at this scale)",
                xy=(0.015, 0.94), xycoords="axes fraction",
                color=INK["secondary"], ha="left", va="top", fontsize=8)
    ax.annotate(r"crisis regime (2015-16, Apr 2020)", xy=(0.015, 0.06),
                xycoords="axes fraction", color=INK["secondary"], ha="left", va="bottom",
                fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 1b -- innovation RMSE (companion to kalman1, own figure)
# ---------------------------------------------------------------------------
def fig_innovation_rmse(kf):
    dates = pd.DatetimeIndex(kf["dates"])
    crisis = _is_crisis(kf["dates"])
    innov_rmse = np.sqrt(np.mean(kf["innovations"] ** 2, axis=1))
    roll = pd.Series(innov_rmse, index=dates).rolling("30D").mean()

    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.35))
    _shade_crisis(ax, dates, crisis)
    ax.plot(dates, innov_rmse, color=INK["faint"], lw=0.6, zorder=1)
    ax.plot(roll.index, roll.values, color=INK["primary"], lw=1.4, zorder=2)
    ax.set_ylabel("innovation RMSE\n(log-price)", fontsize=9)
    ax.set_yscale("log")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 -- 3D term-structure fit
# ---------------------------------------------------------------------------
def fig_term_structure_fit_3d(kf, stride: int = 5):
    dates_panel, t_train, S, taus, log_F_obs, r, _ = load_wti_panel()

    kf_idx = pd.DatetimeIndex(kf["dates"])
    panel_idx = pd.DatetimeIndex(dates_panel)
    common = kf_idx.intersection(panel_idx)
    kf_pos = kf_idx.get_indexer(common)

    innov = kf["innovations"][kf_pos]              # (n_common, 8) observed - fitted

    # Subsample dates for a legible surface -- 2700+ dates x 8 maturities renders
    # as a dense, unreadable mesh; every `stride`-th date keeps the seasonal/
    # crisis shape visible while cutting vertex count by stride-fold.
    idx = np.arange(0, len(common), stride)
    t_years = (common[idx] - common[0]).days / 365.0
    T, K = np.meshgrid(t_years, taus, indexing="ij")
    err_full = innov[idx]

    # The April-2020 negative-WTI print produces one innovation spike roughly
    # an order of magnitude larger than everything else in the panel; plotted
    # at full height it swallows the entire z-range and flattens every other
    # date to a thin ribbon. Clip the SURFACE HEIGHT at the 99th percentile
    # (color uses the same clipped field, so the two never disagree) and say
    # so explicitly, rather than silently losing that event or letting it
    # dominate the figure.
    zmax = np.nanpercentile(np.abs(err_full), 99)
    peak = np.nanmax(np.abs(err_full))
    err = np.clip(err_full, -zmax, zmax)

    fig = plt.figure(figsize=figsize("full", ratio=0.72))
    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(-zmax, zmax)
    ax.plot_surface(T, K, err, facecolors=DIV_BLUE_RED(norm(err)),
                     rstride=1, cstride=1, linewidth=0.15,
                     edgecolor=(1, 1, 1, 0.25), antialiased=True, shade=False)

    ax.set_xlabel("years since " + str(common[0].date()), fontsize=9, labelpad=10)
    ax.set_ylabel(r"maturity $\tau$ (yrs)", fontsize=9, labelpad=8)
    # No separate z-axis label: height and colour both encode the same signed
    # innovation, and the colorbar label already names it -- a z-label here
    # only collided with the colorbar in practice, for no added information.
    ax.set_zticklabels([])
    ax.tick_params(labelsize=8, pad=2)
    ax.view_init(elev=24, azim=-55)
    ax.set_box_aspect((2.6, 1, 0.9))
    ax.text2D(0.02, 0.90, f"z clipped at $\\pm${zmax:.2f}; peak $|$innovation$|$ "
              f"= {peak:.2f} (Apr 2020)", transform=ax.transAxes, fontsize=8,
              color=INK["secondary"])

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=DIV_BLUE_RED)
    mappable.set_array(err)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.1, aspect=18)
    cbar.set_label("innovation (log-price)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    return fig


def main():
    use_thesis_style()
    kf = load_kalman()

    for name, builder in (
        ("kalman1_delta_recovery", fig_delta_recovery),
        ("kalman1b_innovation_rmse", fig_innovation_rmse),
        ("kalman2_term_structure_fit_3d", fig_term_structure_fit_3d),
    ):
        fig = builder(kf)
        path = save(fig, name, directory=FIGDIR)
        plt.close(fig)
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
