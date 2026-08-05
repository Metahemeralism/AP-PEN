"""
Data-chapter figures for the real WTI dataset.

Four standalone figures describing the market data that feeds the
Gibson-Schwartz AP-PEN inversion, BEFORE any model touches it (no multi-panel
composites -- see the sizing note in make_kalman_figures.py):

  real_data_market_state      -- S_t and r_t, the two exogenous pricing
                                 inputs, on a shared time axis (one plot, twin
                                 y-axes -- not a panel pair, see NOTE below)
  real_data_basis_surface     -- what the PINN actually observes: the futures
                                 term structure through time, contango vs
                                 backwardation, as a 2D heatmap
  real_data_delta_proxy       -- a model-free estimate of delta_t, the latent
                                 quantity this thesis inverts for, and the
                                 naive baseline the PINN is judged against
  real_data_term_structure_3d -- the same basis surface as
                                 real_data_basis_surface, redrawn as a literal
                                 3D surface (date x maturity x basis) rather
                                 than a flat heatmap -- makes the contango/
                                 backwardation shape, and how sharply it can
                                 invert within a single episode (e.g. the
                                 April 2020 collapse), directly visible as
                                 topography rather than colour alone

NOTE: real_data_market_state plots two series (spot, rate) on one Axes with
a twin y-axis, not two Axes tiled in a grid -- that is one plot with two
scales, the standard way to show two differently-scaled series against a
shared x-axis, and is not the kind of (a)/(b) panel grouping this project's
figures otherwise avoid.

Run:  python scripts/figures/make_real_data_figures.py
Reads data/input/real/{wti_daily_state,wti_futures_panel}.csv, writes
figures/raw_data/{real_data_market_state,real_data_basis_surface,
real_data_delta_proxy,real_data_term_structure_3d}.{pdf,png}.

PROVENANCE RULE
---------------
data/input/real/ is NOT reproducible from code (Bloomberg source, see
CLAUDE.md) -- it is tracked as-is. Every mark in both figures is either a raw
column from those CSVs or a plain aggregation of them (per-date OLS slope,
linear interpolation onto a common tau grid, rolling mean, date subsampling).
Nothing here is simulated or fitted beyond that one-line regression, and the
regression is the classical, model-free cost-of-carry slope -- not the PINN.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

from gs_wamol.utils.thesis_style import (
    C_DELTA, C_SPOT, DIV_BLUE_RED, INK,
    direct_label, figsize, save, use_thesis_style,
)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "input" / "real"
FIGDIR = REPO / "figures" / "raw_data"


def load():
    state = pd.read_csv(DATA / "wti_daily_state.csv", parse_dates=["date"])
    panel = pd.read_csv(DATA / "wti_futures_panel.csv", parse_dates=["date", "expiry"])
    state = state.sort_values("date").reset_index(drop=True)
    panel = panel.sort_values(["date", "tau"]).reset_index(drop=True)
    return state, panel


def per_date_features(g: pd.DataFrame) -> pd.Series:
    """Collapse one date's live curve into scalar term-structure features.

    Cost-of-carry: ln F(tau) ~= ln S + (r - delta) * tau, so the least-squares
    slope of ln F against tau across ALL live maturities that day estimates
    (r - delta). Fitting the whole curve (rather than just the front two
    contracts) is what keeps the slope from being hostage to one noisy quote.
    """
    tau = g["tau"].to_numpy()
    F = g["settle"].to_numpy()
    slope = np.polyfit(tau, np.log(F), 1)[0]          # [slope, intercept][0]
    front = F[np.argmin(tau)]
    back = F[np.argmax(tau)]
    return pd.Series({"logF_slope": slope, "curve_spread": back - front})


def build_features(state: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    feat = (
        panel.groupby("date").apply(per_date_features, include_groups=False)
        .reset_index()
    )
    df = feat.merge(state, on="date", how="left")
    # Model-free delta proxy: delta_hat = r - (r - delta) = r - slope.
    df["delta_proxy"] = df["rate"] - df["logF_slope"]
    return df


def build_basis_surface(state: pd.DataFrame, panel: pd.DataFrame):
    """Interpolate the daily curve F(tau)/S - 1 onto a common tau grid.

    NaN outside each day's observed tau span so the heatmap never fabricates
    curve where no contract actually traded (6-9 live maturities/day here).
    """
    tau_grid = np.linspace(0.05, 1.8, 60)
    spot_by_date = state.set_index("date")["spot"]
    rows, row_dates = [], []
    for d, g in panel.groupby("date"):
        g = g.sort_values("tau")
        if len(g) < 3 or d not in spot_by_date.index:
            continue
        f = np.interp(tau_grid, g["tau"], g["settle"], left=np.nan, right=np.nan)
        rows.append(f / spot_by_date[d] - 1.0)
        row_dates.append(d)
    basis = np.array(rows).T                          # (n_tau, n_dates)
    return tau_grid, pd.to_datetime(row_dates), basis


def fig_market_state(state):
    """Spot (left axis) + rate (right axis, twin) -- one plot, two scales."""
    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.42))

    # Colour is carried by the axis LABEL rather than an in-plot direct label:
    # both series end at the same right-hand edge of the shared x-axis, so any
    # end-of-line label collides with the twin axis's own tick labels there.
    ax.plot(state["date"], state["spot"], color=C_SPOT, linewidth=1.4)
    ax.set_ylabel("spot  $S_t$  (USD/bbl)", color=C_SPOT)
    ax.tick_params(axis="y", colors=C_SPOT)
    ax.set_ylim(0, state["spot"].max() * 1.08)

    ax2 = ax.twinx()
    ax2.plot(state["date"], state["rate"] * 100, color=INK["secondary"],
             linewidth=1.2, linestyle="--")
    ax2.set_ylabel("rate  $r_t$  (%)", color=INK["secondary"])
    ax2.tick_params(axis="y", colors=INK["secondary"])
    ax2.grid(False)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    save(fig, "real_data_market_state", directory=FIGDIR)
    return fig


def fig_basis_surface(state, panel):
    """F(tau)/S - 1 through time: contango (blue) vs backwardation (red)."""
    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.55))

    tau_grid, row_dates, basis = build_basis_surface(state, panel)
    vmax = np.nanpercentile(np.abs(basis), 98)
    mesh = ax.pcolormesh(row_dates, tau_grid, basis, cmap=DIV_BLUE_RED,
                          vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_ylabel(r"maturity  $\tau$  (yrs)")
    ax.grid(False)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.025)
    cbar.set_label(r"basis  $F(\tau)/S-1$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    save(fig, "real_data_basis_surface", directory=FIGDIR)
    return fig


def fig_delta_proxy(feat):
    """Model-free convenience-yield proxy, delta_hat = r - d(lnF)/d(tau)."""
    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.38))

    back = feat["curve_spread"] < 0
    ax.fill_between(feat["date"], 0, 1, where=back,
                     transform=ax.get_xaxis_transform(),
                     color=C_DELTA, alpha=0.08, step="mid")
    ax.axhline(0, color=INK["muted"], linewidth=0.7, linestyle=":")
    ax.plot(feat["date"], feat["delta_proxy"], color=INK["faint"], linewidth=0.8)
    roll = feat.set_index("date")["delta_proxy"].rolling("90D").mean()
    ax.plot(roll.index, roll.values, color=C_DELTA, linewidth=1.6)
    ax.set_ylabel(r"$\hat\delta_t$")
    direct_label(ax, roll.index[-1], roll.values[-1], r"90d mean $\hat\delta_t$",
                 C_DELTA, dx=6, dy=0)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    save(fig, "real_data_delta_proxy", directory=FIGDIR)
    return fig


def fig_term_structure_3d(state, panel, stride: int = 4):
    """The basis surface (panel b of the overview figure) as literal 3D
    topography instead of a flat heatmap.

    Reuses build_basis_surface exactly -- same interpolated tau grid, same
    NaN-outside-observed-span masking -- so this is a different projection of
    the identical data, not a second computation that could quietly diverge
    from panel (b).
    """
    tau_grid, row_dates, basis = build_basis_surface(state, panel)  # (n_tau, n_dates)

    # Subsample dates for a legible mesh -- 2900+ dates at full resolution
    # renders as an unreadable wall; every `stride`-th date keeps the
    # regime-to-regime shape (2015-16 contango glut, 2020 collapse, the
    # 2022+ backwardation stretch) visible.
    idx = np.arange(0, len(row_dates), stride)
    t_years = (row_dates[idx] - row_dates[0]).days / 365.0
    T, K = np.meshgrid(t_years, tau_grid, indexing="ij")
    Z = basis[:, idx].T                                   # (n_dates_sub, n_tau)

    vmax = np.nanpercentile(np.abs(Z), 98)
    norm = plt.Normalize(-vmax, vmax)

    fig = plt.figure(figsize=figsize("full", ratio=0.72))
    ax = fig.add_subplot(111, projection="3d")
    # NaN cells (outside that date's observed tau span, see build_basis_surface)
    # must not become 0 -- plot_surface handles NaN by leaving those facets out,
    # but facecolors needs the same NaNs mapped through the colormap, which
    # matplotlib's Normalize + cmap already does safely (NaN -> the cmap's
    # "bad" colour, invisible here since the corresponding vertices are masked).
    surf_colours = DIV_BLUE_RED(norm(Z))
    ax.plot_surface(T, K, Z, facecolors=surf_colours, rstride=1, cstride=1,
                     linewidth=0.1, edgecolor=(1, 1, 1, 0.2), antialiased=True,
                     shade=False)

    ax.set_xlabel("years since " + str(row_dates[0].date()), fontsize=9, labelpad=10)
    ax.set_ylabel(r"maturity $\tau$ (yrs)", fontsize=9, labelpad=8)
    ax.set_zticklabels([])  # height and colour both encode the basis; see kalman2's
    ax.tick_params(labelsize=8, pad=2)                      # note on this same choice
    ax.view_init(elev=24, azim=-55)
    ax.set_box_aspect((2.6, 1, 0.9))

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=DIV_BLUE_RED)
    mappable.set_array(Z)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.1, aspect=18)
    cbar.set_label(r"basis  $F(\tau)/S-1$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    save(fig, "real_data_term_structure_3d", directory=FIGDIR)
    return fig


def main():
    use_thesis_style()
    state, panel = load()
    feat = build_features(state, panel)

    fig_market_state(state)
    print(f"wrote {FIGDIR / 'real_data_market_state.pdf'}")
    fig_basis_surface(state, panel)
    print(f"wrote {FIGDIR / 'real_data_basis_surface.pdf'}")
    fig_delta_proxy(feat)
    print(f"wrote {FIGDIR / 'real_data_delta_proxy.pdf'}")
    fig_term_structure_3d(state, panel)
    print(f"wrote {FIGDIR / 'real_data_term_structure_3d.pdf'}")


if __name__ == "__main__":
    main()
