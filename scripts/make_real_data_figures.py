"""
Data-chapter figure for the real WTI dataset (three-panel).

One thesis-ready figure describing the market data that feeds the
Gibson-Schwartz PINN inversion:

  (a) market state       -- S_t and r_t, the two exogenous pricing inputs
  (b) basis surface       -- what the PINN actually observes: the futures
                             term structure through time, contango vs
                             backwardation
  (c) convenience-yield proxy -- a model-free estimate of delta_t, the
                             latent quantity this thesis inverts for, and
                             the naive baseline the PINN is judged against

Run:  python scripts/make_real_data_figures.py
Reads data/input/real/{wti_daily_state,wti_futures_panel}.csv, writes
figures/raw_data/real_data_overview.{pdf,png}.

PROVENANCE RULE
---------------
data/input/real/ is NOT reproducible from code (Bloomberg source, see
CLAUDE.md) -- it is tracked as-is. Every mark in this figure is either a raw
column from those CSVs or a plain aggregation of them (per-date OLS slope,
linear interpolation onto a common tau grid, rolling mean). Nothing here is
simulated or fitted beyond that one-line regression, and the regression is
the classical, model-free cost-of-carry slope -- not the PINN.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gs_wamol.utils.thesis_style import (
    C_DELTA, C_SPOT, DIV_BLUE_RED, INK,
    direct_label, figsize, panel_tag, save, use_thesis_style,
)

REPO = Path(__file__).resolve().parents[1]
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


def fig_real_data_overview(state, panel, feat):
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=figsize("full", ratio=1.05), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.5, 1.2], "hspace": 0.32},
    )

    # -- (a) market state: spot (left axis) + rate (right axis, twin) --------
    # Colour is carried by the axis LABEL rather than an in-plot direct label:
    # both series end at the same right-hand edge of the shared x-axis, so any
    # end-of-line label collides with the twin axis's own tick labels there.
    ax_a.plot(state["date"], state["spot"], color=C_SPOT, linewidth=1.2)
    ax_a.set_ylabel("spot  $S_t$  (USD/bbl)", color=C_SPOT)
    ax_a.tick_params(axis="y", colors=C_SPOT)
    ax_a.set_ylim(0, state["spot"].max() * 1.08)

    ax_a2 = ax_a.twinx()
    ax_a2.plot(state["date"], state["rate"] * 100, color=INK["secondary"],
               linewidth=1.0, linestyle="--")
    ax_a2.set_ylabel("rate  $r_t$  (%)", color=INK["secondary"])
    ax_a2.tick_params(axis="y", colors=INK["secondary"])
    ax_a2.grid(False)
    panel_tag(ax_a, "a")

    # -- (b) basis surface: F(tau)/S - 1, contango (blue) vs backwardation (red)
    tau_grid, row_dates, basis = build_basis_surface(state, panel)
    vmax = np.nanpercentile(np.abs(basis), 98)
    mesh = ax_b.pcolormesh(row_dates, tau_grid, basis, cmap=DIV_BLUE_RED,
                            vmin=-vmax, vmax=vmax, shading="auto")
    ax_b.set_ylabel(r"maturity  $\tau$  (yrs)")
    ax_b.grid(False)
    cbar = fig.colorbar(mesh, ax=ax_b, pad=0.015, fraction=0.025)
    cbar.set_label(r"basis  $F(\tau)/S-1$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    panel_tag(ax_b, "b")

    # -- (c) model-free convenience-yield proxy, delta_hat = r - d(lnF)/d(tau) -
    back = feat["curve_spread"] < 0
    ax_c.fill_between(feat["date"], 0, 1, where=back,
                       transform=ax_c.get_xaxis_transform(),
                       color=C_DELTA, alpha=0.08, step="mid")
    ax_c.axhline(0, color=INK["muted"], linewidth=0.7, linestyle=":")
    ax_c.plot(feat["date"], feat["delta_proxy"], color=INK["faint"], linewidth=0.8)
    roll = feat.set_index("date")["delta_proxy"].rolling("90D").mean()
    ax_c.plot(roll.index, roll.values, color=C_DELTA, linewidth=1.6)
    ax_c.set_ylabel(r"$\hat\delta_t$")
    direct_label(ax_c, roll.index[-1], roll.values[-1], r"90d mean $\hat\delta_t$",
                 C_DELTA, dx=6, dy=0)
    panel_tag(ax_c, "c")

    ax_c.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in (ax_a, ax_b):
        ax.tick_params(axis="x", labelbottom=False)

    save(fig, "real_data_overview", directory=FIGDIR)
    return fig


def main():
    use_thesis_style()
    state, panel = load()
    feat = build_features(state, panel)
    fig_real_data_overview(state, panel, feat)
    print(f"wrote {FIGDIR / 'real_data_overview.pdf'}")


if __name__ == "__main__":
    main()
