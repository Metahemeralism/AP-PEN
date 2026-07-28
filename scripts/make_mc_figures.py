"""
Methodology-chapter figures for the Gibson-Schwartz Monte Carlo simulator.

Each figure is a SIDE-BY-SIDE pair, because each one makes a comparative claim
that the methodology text needs to assert:

  mc1  the two P-measure state processes    -- S_t diffuses, delta_t mean-reverts
  mc2  the observation layer                -- clean prices vs noisy quotes
  mc3  why the inversion is well posed      -- delta maps affinely to curve slope
  mc4  the basis by maturity, small multiples -- one panel per tau, same diagram
       repeated twelve times, shared axes, showing the band widening with tau

Run:  python scripts/make_mc_figures.py
Reads data/input/synthetic/mc_data.pkl, writes figures/monte_carlo_sim/{pdf,png}.

PROVENANCE RULE (do not relax this)
-----------------------------------
Every plotted mark must come from data/mc_data.pkl -- the pickle written by
monte_carlo.ipynb -- or be a plain aggregation of it (percentile, median, log,
difference, stride). This script must NOT:

  * re-run or re-simulate any part of the model,
  * draw fresh random numbers at plot time,
  * overlay analytically-derived curves computed here.

The reason is evidentiary, not stylistic. These figures are the thesis's claim
about what the simulator produced; if the plotting layer can generate its own
numbers, the figure stops being evidence of that and no reader (or examiner) can
tell which marks came from the simulation and which from the plotting code.

Model PARAMETERS read from the pickle (alpha_P, alpha_Q, kappa) are fine to draw
as reference lines -- they are the notebook's own stored inputs, and each is
labelled as a parameter rather than passed off as simulated output.

Consequence: this script is only as current as the pickle. Re-run the notebook
end-to-end before regenerating figures for submission.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from gs_wamol.utils.thesis_style import (
    C_DELTA, C_SPOT, DIV_BLUE_RED, INK, SEQ_BLUE,
    despine_shared_y, direct_label, figsize, panel_tag, save, use_thesis_style,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "input" / "synthetic" / "mc_data.pkl"

# One path is followed through every figure so the reader can build a mental
# thread across the chapter. Picking it once, here, is the whole point.
HERO_PATH = 7


def load():
    with open(DATA, "rb") as f:
        d = pickle.load(f)
    # JAX arrays -> numpy; matplotlib handles numpy natively and we want no
    # device transfers hiding inside the plotting code.
    return {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Figure 1 -- the two state processes under P
# ---------------------------------------------------------------------------
def fig_state_paths(d):
    """S_t (left) vs delta_t (right): the simulator's two latent state variables.

    FORM: emphasis, not categorical. The 100 Monte Carlo paths are exchangeable
    draws -- path #3 has no more meaning than path #57 -- so giving each its own
    colour (the viridis default) encodes an ordering that does not exist. The
    honest encoding is: the ensemble is faint grey CONTEXT, one path is the
    coloured SUBJECT, and the analytic moments are the reference.

    SHAPES: t (N+1,) = (1001,);  S, delta_true (M, N+1) = (100, 1001).
    """
    t, S, delta = d["t_grid"], d["S"], d["delta_true"]
    p = d["params_P"]
    alpha_P, kappa, sigma2 = p["alpha_P"], p["kappa"], p["sigma2"]

    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=figsize("full", ratio=0.40))

    # -- (a) spot ----------------------------------------------------------
    ax_s.plot(t, S.T, color=INK["faint"], lw=0.35, alpha=0.7, zorder=1)
    # Empirical 5-95% band. S_t is lognormal-ish with a long right tail, so the
    # mean sits above the bulk; quantiles describe where paths actually are.
    lo, hi = np.percentile(S, [5, 95], axis=0)
    ax_s.fill_between(t, lo, hi, color=C_SPOT, alpha=0.10, lw=0, zorder=2)
    ax_s.plot(t, np.median(S, axis=0), color=INK["secondary"], lw=1.0, ls="--", zorder=3)
    ax_s.plot(t, S[HERO_PATH], color=C_SPOT, lw=1.4, zorder=4)
    ax_s.set_ylim(S.min() * 0.75, S.max() * 1.6)  # headroom so labels never sit on data

    ax_s.set_yscale("log")  # geometric process -> equal proportional moves get
                            # equal vertical space; on a linear axis the early
                            # low-price section is visually squashed to nothing
    ax_s.set_xlabel("Time $t$ (years)")
    ax_s.set_ylabel(r"Spot price $S_t$")
    panel_tag(ax_s, "a")
    # Labels are placed at DIFFERENT x positions and pushed to opposite sides of
    # their lines. Anchoring both at the same t is what made them collide.
    q = len(t) // 4
    direct_label(ax_s, t[q], S[HERO_PATH, q], "single path", C_SPOT, dy=-11, ha="center")
    direct_label(ax_s, t[-1], np.median(S, axis=0)[-1],
                 "median", INK["secondary"], dx=-4, dy=9, ha="right")
    direct_label(ax_s, t[-1], hi[-1], "5–95%", C_SPOT, dx=-4, dy=4, ha="right")

    # -- (b) convenience yield ---------------------------------------------
    ax_d.plot(t, delta.T, color=INK["faint"], lw=0.35, alpha=0.7, zorder=1)
    # Empirical 5-95% band across the simulated ensemble -- the SAME treatment as
    # the spot panel, so the two panels are read on equal terms.
    #
    # This deliberately does NOT overlay the analytic OU standard deviation
    # sigma2*sqrt((1-exp(-2*kappa*t))/(2*kappa)). That curve would be a
    # theoretical quantity computed here at plot time, not an output of the
    # simulation, and mixing derived theory into a figure whose caption says
    # "simulated paths" invites the reader to mistake one for the other. Every
    # mark in this figure is now the notebook's own output or a plain
    # aggregation of it.
    d_lo, d_hi = np.percentile(delta, [5, 95], axis=0)
    ax_d.fill_between(t, d_lo, d_hi, color=C_DELTA, alpha=0.13, lw=0, zorder=2)
    ax_d.axhline(alpha_P, color=INK["secondary"], lw=1.0, ls="--", zorder=3)
    ax_d.plot(t, delta[HERO_PATH], color=C_DELTA, lw=1.4, zorder=4)

    ax_d.set_xlabel("Time $t$ (years)")
    ax_d.set_ylabel(r"Convenience yield $\delta_t$")
    panel_tag(ax_d, "b")
    # Sits INSIDE the axes: an offset label at the far-right data edge is clipped
    # by bbox="tight", which trims to the axes rather than growing to fit it.
    direct_label(ax_d, t[-1], alpha_P, r"$\alpha^{\mathbb{P}}$", INK["secondary"],
                 dx=-4, dy=7, ha="right")
    direct_label(ax_d, t[len(t) // 3], d_hi[len(t) // 3],
                 "5–95%", C_DELTA, dy=4, ha="center")

    # The mean-reversion timescale tau* = 1/kappa is the identifiability-critical
    # constant of the whole thesis, so it is annotated rather than left implicit.
    # The LINE stays muted (it is context), but its LABEL does not: 8pt type at
    # 3.5:1 contrast is under the AA floor, and this is the smallest text here.
    ax_d.axvline(1.0 / kappa, color=INK["muted"], lw=0.7, ls=":", zorder=3)
    direct_label(ax_d, 1.0 / kappa, ax_d.get_ylim()[1], r"$\tau^{*}=1/\kappa$",
                 INK["secondary"], dx=4, dy=-8, va="top")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 -- the observation layer
# ---------------------------------------------------------------------------
def fig_observation_layer(d):
    """Clean closed-form prices (left) vs noisy observed quotes (right).

    This is the figure that justifies the Option-A architecture: the left panel
    carries NO sampling error at all, so every deviation visible on the right is
    the measurement-noise layer and nothing else. Sharing the y-axis is what
    makes the comparison honest -- two independently scaled panels could make
    any noise level look like any other.

    SHAPES: taus (K,) = (12,);  F_clean, log_F_obs (M, N+1, K) = (100, 1001, 12).
    """
    taus, tgrid = d["taus"], d["t_grid"]
    logF_clean = np.log(d["F_clean"][HERO_PATH])  # (N+1, K) one path, all dates
    logF_obs = d["log_F_obs"][HERO_PATH]          # (N+1, K)
    # Read from the pickle, never hardcoded: a mislabelled noise level is worse
    # than a crash, because nothing in the output signals that it is wrong.
    noise_std = d["noise_std"]

    # Eight observation dates spread over the horizon. Date is an ORDERED
    # variable, so it gets the sequential ramp: darker = later. The reader
    # recovers the ordering from the ink alone, with no legend lookup.
    date_idx = np.linspace(0, len(tgrid) - 1, 8).astype(int)
    shades = SEQ_BLUE(np.linspace(0.15, 0.95, len(date_idx)))

    fig, (ax_c, ax_o) = plt.subplots(1, 2, figsize=figsize("full", ratio=0.40), sharey=True)

    for colour, i in zip(shades, date_idx):
        ax_c.plot(taus, logF_clean[i], color=colour, lw=1.2)
        # Right panel: the clean curve stays as a faint spine so the reader can
        # see what the scatter is deviating FROM. Without it the right panel is
        # just an unreadable cloud.
        ax_o.plot(taus, logF_clean[i], color=colour, lw=0.7, alpha=0.45)
        ax_o.plot(taus, logF_obs[i], color=colour, lw=0, marker="o", ms=2.6)

    ax_c.set_ylabel(r"Log futures price $\log \hat{F}(\tau)$")
    for ax, tag in ((ax_c, "a"), (ax_o, "b")):
        ax.set_xlabel(r"Maturity $\tau$ (years)")
        panel_tag(ax, tag)
    despine_shared_y(ax_o)

    # No legend box: with eight curves filling the panel there is nowhere to put
    # one that does not cover data. A sequential ramp only needs its two ENDS
    # named -- the reader interpolates the middle for free, which is exactly the
    # property that makes an ordered ramp better than eight categorical hues.
    # Extra x-headroom is reserved first so the labels have somewhere to live.
    # The headroom must be applied to BOTH panels. Widening only (a) leaves the
    # pair with different x-scales, so the curves render at different physical
    # widths and the eye reads a shape difference that is not in the data --
    # which would defeat the entire purpose of a side-by-side comparison.
    # Padding is a FRACTION of the maturity span, not an absolute number of
    # years, so a pickle with a different tau grid still gets label room that
    # looks the same. The right pad is larger because it holds the date labels.
    span = taus[-1] - taus[0]
    # Ticks come from matplotlib's own "nice number" locator applied to the DATA
    # range, then clipped so none lands in the label headroom -- a tick out there
    # would imply the curves extend further than they do.
    ticks = MaxNLocator(nbins=5, steps=[1, 2, 5, 10]).tick_values(taus[0], taus[-1])
    ticks = ticks[(ticks >= taus[0]) & (ticks <= taus[-1])]
    for ax in (ax_c, ax_o):
        ax.set_xlim(taus[0] - 0.03 * span, taus[-1] + 0.30 * span)
        ax.set_xticks(ticks)
    for j in (0, len(date_idx) - 1):
        i = date_idx[j]
        direct_label(ax_c, taus[-1], logF_clean[i, -1],
                     f"$t={tgrid[i]:.0f}$", shades[j], dx=5)
    # Sits low-left where the earliest curve runs, so it needs the same white
    # halo the direct labels get -- and readable ink, not the line grey.
    ax_c.annotate("darker = later date", xy=(0.02, 0.03), xycoords="axes fraction",
                  fontsize=8, color=INK["secondary"],
                  path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    # sigma_eps is a data fact (the pickle's own noise_std), not narrative, so it
    # stays on the panel it applies to rather than moving to the caption -- a
    # caption fixed at submission time would go stale if the pickle is ever
    # regenerated with a different noise level.
    ax_o.annotate(rf"$\sigma_\varepsilon = {noise_std:g}$", xy=(0.02, 0.03),
                  xycoords="axes fraction", fontsize=8, color=INK["secondary"],
                  path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 -- why the inverse problem is well posed
# ---------------------------------------------------------------------------
def fig_identifiability(d):
    """Curve shape vs delta (left) and the affine slope-delta map (right).

    The methodological claim: because the closed form is exponential-affine,
        log F(tau_2) - log F(tau_1) = [B(tau_2) - B(tau_1)] * delta + const,
    the term-structure slope is an EXACT affine function of delta. So in the
    noiseless case the inversion is trivially well posed, and the entire
    difficulty is the noise -- which panel (b) shows as vertical scatter about
    that line. This is the figure that motivates the identifiability analysis.

    FORM: panel (a) is diverging, not sequential. delta relative to alpha^Q is a
    POLARITY (backwardation above, contango below) with a meaningful centre, and
    a diverging ramp with a neutral grey midpoint is the encoding that says
    "the middle is nothing happening".
    """
    taus, delta = d["taus"], d["delta_true"]
    logF_clean = np.log(d["F_clean"])
    logF_obs = d["log_F_obs"]
    alpha_Q = d["alpha_Q"]

    fig, (ax_curve, ax_slope) = plt.subplots(1, 2, figsize=figsize("full", ratio=0.40))

    # -- (a) curves coloured by the delta that produced them ----------------
    # Subsample (path, date) states on a fixed STRIDE rather than a fresh random
    # draw. A `default_rng` here would introduce numbers generated at plot time:
    # harmless statistically, but it means the figure is no longer a pure
    # function of the notebook's output, and re-running with a different seed
    # would silently change which states are shown. Striding is deterministic,
    # reproducible, and reads every path in the ensemble.
    # The stride is OFFSET to start at HERO_PATH so the path followed through
    # mc1 and mc2 is actually among the states shown here. A plain
    # arange(0, M, 5) starts at 0 and skips 7 entirely, which quietly broke the
    # cross-figure thread the chapter tells the reader to follow.
    M, N1 = delta.shape
    n_paths, n_dates = len(range(HERO_PATH, M, 5)), 8
    pi = np.repeat(np.arange(HERO_PATH, M, 5), n_dates)
    di = np.tile(np.linspace(0, N1 - 1, n_dates).astype(int), n_paths)
    dv = delta[pi, di]

    # Normalise delta about alpha_Q so 0.5 on the colormap lands exactly on the
    # neutral midpoint -- symmetric limits are what make a diverging map truthful.
    half = np.abs(dv - alpha_Q).max()
    norm = (dv - alpha_Q) / (2 * half) + 0.5

    # Curves are shown as deviations from their own front-month value, isolating
    # SHAPE from level: without this the spot-price spread dominates and every
    # curve just looks like a different horizontal line.
    for p_i, d_i, u in zip(pi, di, norm):
        curve = logF_clean[p_i, d_i] - logF_clean[p_i, d_i, 0]
        ax_curve.plot(taus, curve, color=DIV_BLUE_RED(u), lw=0.6, alpha=0.75)
    ax_curve.axhline(0.0, color=INK["muted"], lw=0.7)
    # Reserve an empty BAND above and below the curve envelope for the two regime
    # labels, rather than letting matplotlib pad and then squeezing the text into
    # whatever is left. The fan converges toward tau_1, so its extreme curves
    # sweep upward across any label parked just under them -- a gap measured only
    # at the right-hand edge does not stay a gap. Reserving a full band keeps the
    # labels clear across their whole width, and the (a) tag clear of the ticks.
    ends = logF_clean[pi, di, -1] - logF_clean[pi, di, 0]
    band = 0.16 * (ends.max() - ends.min())   # ~2x an 8pt line at this figure size
    ax_curve.set_ylim(ends.min() - band, ends.max() + band)

    ax_curve.set_xlabel(r"Maturity $\tau$ (years)")
    ax_curve.set_ylabel(r"$\log \hat{F}(\tau) - \log \hat{F}(\tau_1)$")
    panel_tag(ax_curve, "a")
    # Text, not colour, names the two regimes -- colour alone would be lost in
    # greyscale and to a colour-blind reader.
    # Each label wears the colour of the ramp arm it names. The ramp runs
    # blue (low delta) -> grey -> red (high delta), and low delta is the CONTANGO
    # (upward-sloping) regime -- so contango is blue and backwardation is red.
    # Getting this pairing backwards silently teaches the reader the wrong
    # mapping, so it is worth stating the direction explicitly rather than
    # trusting that the hex constants were typed in the right order.
    #
    # Each label is centred in the band reserved for it above, so placement is
    # DERIVED from the curve envelope rather than being a fraction of the axis.
    # The old 0.72*ylim dropped the backwardation label straight into the dense
    # red fan, because the fan is asymmetric -- it reaches further down than up,
    # so the same fraction means different things at the two ends.
    direct_label(ax_curve, taus[-1], ends.max() + band / 2,
                 r"contango ($\delta < \alpha^{\mathbb{Q}}$)", "#184f95", dx=-4, ha="right")
    direct_label(ax_curve, taus[-1], ends.min() - band / 2,
                 r"backwardation ($\delta > \alpha^{\mathbb{Q}}$)", "#a32020", dx=-4, ha="right")

    # -- (b) the slope-delta map, clean vs noisy ----------------------------
    slope_clean = logF_clean[pi, di, -1] - logF_clean[pi, di, 0]
    slope_obs = logF_obs[pi, di, -1] - logF_obs[pi, di, 0]

    ax_slope.scatter(dv, slope_obs, s=5, color=C_DELTA, alpha=0.55,
                     lw=0, label="with measurement noise")
    ax_slope.plot(np.sort(dv), slope_clean[np.argsort(dv)],
                  color=INK["primary"], lw=1.2, label="exact (closed form)")
    ax_slope.axvline(alpha_Q, color=INK["muted"], lw=0.7, ls=":")  # line: context

    ax_slope.set_xlabel(r"True convenience yield $\delta_t$")
    ax_slope.set_ylabel(r"Slope $\log \hat{F}(\tau_{12}) - \log \hat{F}(\tau_{1})$")
    ax_slope.legend(loc="upper right")
    panel_tag(ax_slope, "b")
    direct_label(ax_slope, alpha_Q, ax_slope.get_ylim()[0],
                 r"$\alpha^{\mathbb{Q}}$", INK["secondary"], dx=3, dy=6, va="bottom")

    fig.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# Figure 4 -- the basis by maturity, one small-multiple panel per tau
# ---------------------------------------------------------------------------
def fig_basis_by_maturity(d):
    """The log basis log(F/S), same diagram repeated at 4 representative maturities.

    WHY 4, NOT ALL 12: the ensemble spread at a fixed date is |B(tau)| times the
    spread of delta itself, and B(tau) = -(1-e^{-kappa tau})/kappa saturates
    fast -- with kappa ~= 1.9 the mean-reversion timescale is tau* = 1/kappa
    ~= 0.53y, well inside the 1-year contract grid. An earlier 12-panel version
    of this figure showed that directly: panels past tau ~= 0.5 were visually
    indistinguishable from each other, so eight of the twelve panels were
    redundant ink repeating the same "already saturated" band width. Four
    maturities -- shortest, near tau*, mid, longest -- carry the whole story
    (narrow -> widening -> flat) without asking the reader to notice that panels
    5-12 stopped changing.

    SHAPES: taus (K,) = (12,); S, (M, N+1) = (100, 1001);
            F_clean (M, N+1, K) = (100, 1001, 12); basis is a plain log-and-
            difference of those two pickle outputs, not a new simulation.
    """
    t, taus = d["t_grid"], d["taus"]
    S, F = d["S"], d["F_clean"]
    basis = np.log(F) - np.log(S)[:, :, None]  # (M, N+1, K)
    kappa = float(d["params_P"]["kappa"])
    tau_star = 1.0 / kappa

    # Indices into the 12-point tau grid: shortest, the maturity closest to
    # tau* (the mean-reversion timescale -- the natural place for the band to
    # stop widening), a mid maturity past that, and the longest quoted contract.
    sel = [0, int(np.argmin(np.abs(taus - tau_star))), 8, len(taus) - 1]
    sel = sorted(set(sel))

    fig, axes = plt.subplots(
        1, len(sel), figsize=figsize("full", ratio=0.30),
        sharex=True, sharey=True,
    )

    # Shared y-limits computed ONCE across the selected maturities, not per
    # panel. Autoscaling each panel independently would rescale away exactly
    # the amplitude growth this figure exists to show -- the longest maturity
    # would look no wider than the shortest, just re-zoomed.
    sel_basis = basis[:, :, sel]
    lo_all, hi_all = np.nanpercentile(sel_basis, [1, 99])
    pad = 0.08 * (hi_all - lo_all)

    for ax, k in zip(axes, sel):
        b = basis[:, :, k]
        ax.plot(t, b.T, color=INK["faint"], lw=0.3, alpha=0.6, zorder=1)
        lo, hi = np.percentile(b, [5, 95], axis=0)
        ax.fill_between(t, lo, hi, color=C_DELTA, alpha=0.16, lw=0, zorder=2)
        ax.plot(t, np.median(b, axis=0), color=INK["secondary"], lw=0.8,
                 ls="--", zorder=3)
        ax.plot(t, b[HERO_PATH], color=C_DELTA, lw=1.0, zorder=4)
        ax.axhline(0.0, color=INK["muted"], lw=0.5, ls=":", zorder=0)
        ax.set_title(rf"$\tau={taus[k]:.2f}$", fontsize=8,
                     color=INK["secondary"], pad=2)
        ax.set_ylim(lo_all - pad, hi_all + pad)
        ax.set_xlabel("$t$ (yrs)")

    axes[0].set_ylabel(r"$\log(F_\tau/S_t)$")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    # ONE legend for the whole row rather than direct labels repeated four
    # times -- the encoding (grey ensemble / band / median / hero path) is
    # identical in every panel, so it only needs to be stated once.
    legend_elems = [
        Line2D([0], [0], color=INK["faint"], lw=1.2, label=f"ensemble ($M={S.shape[0]}$)"),
        Patch(facecolor=C_DELTA, alpha=0.16, label="5–95%"),
        Line2D([0], [0], color=INK["secondary"], lw=0.8, ls="--", label="median"),
        Line2D([0], [0], color=C_DELTA, lw=1.4, label="single path"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=8)

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return fig


def main():
    use_thesis_style()
    d = load()
    for name, builder in (
        ("mc1_state_paths", fig_state_paths),
        ("mc2_observation_layer", fig_observation_layer),
        ("mc3_identifiability", fig_identifiability),
        ("mc4_basis_by_maturity", fig_basis_by_maturity),
    ):
        fig = builder(d)
        path = save(fig, name)
        plt.close(fig)
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
