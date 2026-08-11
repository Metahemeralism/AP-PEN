"""
Methodology-chapter figures for the Gibson-Schwartz Monte Carlo simulator.

No multi-panel composites (see the sizing note in make_kalman_figures.py) --
each figure below is standalone, sized either full text-width or "half" so it
can sit next to its named partner as two separate side-by-side floats:

  mc1a/mc1b   the two P-measure state processes -- S_t diffuses (mc1a),
              delta_t mean-reverts (mc1b); a side-by-side pair
  mc2a/mc2b   the observation layer -- clean prices (mc2a) vs noisy quotes
              (mc2b); a side-by-side pair, SAME y-limits so the comparison
              stays honest
  mc2c        the observation layer, as a literal 3D surface (companion to
              mc2a, full width)
  mc3a/mc3b   why the inversion is well posed -- curve shape vs delta (mc3a),
              the affine slope-delta map (mc3b); a side-by-side pair
  mc4a-mc4d   the basis by maturity, one figure per representative tau
              (shortest, near tau*, mid, longest) -- was one 4-panel small-
              multiples figure; now four standalone full-width figures. Split
              per this project's no-panels rule, though the "band widens with
              tau" comparison across all four was easier to see side by side
              as small multiples -- flagging that trade-off rather than
              hiding it.

Run:  python scripts/figures/make_mc_figures.py
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

from gs_wamol.utils.thesis_style import (
    C_DELTA, C_SPOT, DIV_BLUE_RED, INK, SEQ_BLUE,
    direct_label, figsize, save, use_thesis_style,
)

REPO = Path(__file__).resolve().parents[2]
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
# Figure 1a/1b -- the two state processes under P (side-by-side pair)
# ---------------------------------------------------------------------------
def fig_spot_paths(d):
    """S_t: the simulator's spot-price ensemble, one path emphasised.

    FORM: emphasis, not categorical. The 100 Monte Carlo paths are exchangeable
    draws -- path #3 has no more meaning than path #57 -- so giving each its own
    colour (the viridis default) encodes an ordering that does not exist. The
    honest encoding is: the ensemble is faint grey CONTEXT, one path is the
    coloured SUBJECT, and the analytic moments are the reference.

    SHAPES: t (N+1,) = (1001,);  S (M, N+1) = (100, 1001).
    """
    t, S = d["t_grid"], d["S"]

    fig, ax = plt.subplots(1, 1, figsize=figsize("half", ratio=0.85))

    ax.plot(t, S.T, color=INK["faint"], lw=0.35, alpha=0.7, zorder=1)
    # Empirical 5-95% band. S_t is lognormal-ish with a long right tail, so the
    # mean sits above the bulk; quantiles describe where paths actually are.
    lo, hi = np.percentile(S, [5, 95], axis=0)
    ax.fill_between(t, lo, hi, color=C_SPOT, alpha=0.10, lw=0, zorder=2)
    ax.plot(t, np.median(S, axis=0), color=INK["secondary"], lw=1.0, ls="--", zorder=3)
    ax.plot(t, S[HERO_PATH], color=C_SPOT, lw=1.4, zorder=4)
    ax.set_ylim(S.min() * 0.75, S.max() * 1.6)  # headroom so labels never sit on data

    ax.set_yscale("log")  # geometric process -> equal proportional moves get
                          # equal vertical space; on a linear axis the early
                          # low-price section is visually squashed to nothing
    ax.set_xlabel("Time $t$ (years)")
    ax.set_ylabel(r"Spot price $S_t$")
    # Labels are placed at DIFFERENT x positions and pushed to opposite sides of
    # their lines. Anchoring both at the same t is what made them collide.
    q = len(t) // 4
    direct_label(ax, t[q], S[HERO_PATH, q], "single path", C_SPOT, dy=-11, ha="center")
    direct_label(ax, t[-1], np.median(S, axis=0)[-1],
                 "median", INK["secondary"], dx=-4, dy=9, ha="right")
    direct_label(ax, t[-1], hi[-1], "5–95%", C_SPOT, dx=-4, dy=4, ha="right")

    fig.tight_layout()
    return fig


def fig_delta_paths(d):
    """delta_t: the simulator's convenience-yield ensemble, one path emphasised.

    Companion to fig_spot_paths -- same ensemble/band/median/hero-path
    encoding, so the pair reads on equal terms when placed side by side.

    SHAPES: t (N+1,) = (1001,);  delta_true (M, N+1) = (100, 1001).
    """
    t, delta = d["t_grid"], d["delta_true"]
    p = d["params_P"]
    alpha_P, kappa = p["alpha_P"], p["kappa"]

    fig, ax = plt.subplots(1, 1, figsize=figsize("half", ratio=0.85))

    ax.plot(t, delta.T, color=INK["faint"], lw=0.35, alpha=0.7, zorder=1)
    # Empirical 5-95% band across the simulated ensemble -- the SAME treatment as
    # the spot-path figure, so the two are read on equal terms.
    #
    # This deliberately does NOT overlay the analytic OU standard deviation
    # sigma2*sqrt((1-exp(-2*kappa*t))/(2*kappa)). That curve would be a
    # theoretical quantity computed here at plot time, not an output of the
    # simulation, and mixing derived theory into a figure whose caption says
    # "simulated paths" invites the reader to mistake one for the other. Every
    # mark in this figure is now the notebook's own output or a plain
    # aggregation of it.
    d_lo, d_hi = np.percentile(delta, [5, 95], axis=0)
    ax.fill_between(t, d_lo, d_hi, color=C_DELTA, alpha=0.13, lw=0, zorder=2)
    ax.axhline(alpha_P, color=INK["secondary"], lw=1.0, ls="--", zorder=3)
    ax.plot(t, delta[HERO_PATH], color=C_DELTA, lw=1.4, zorder=4)

    ax.set_xlabel("Time $t$ (years)")
    ax.set_ylabel(r"Convenience yield $\delta_t$")
    # Sits INSIDE the axes: an offset label at the far-right data edge is clipped
    # by bbox="tight", which trims to the axes rather than growing to fit it.
    direct_label(ax, t[-1], alpha_P, r"$\alpha^{\mathbb{P}}$", INK["secondary"],
                 dx=-4, dy=7, ha="right")
    direct_label(ax, t[len(t) // 3], d_hi[len(t) // 3],
                 "5–95%", C_DELTA, dy=4, ha="center")

    # The mean-reversion timescale tau* = 1/kappa is the identifiability-critical
    # constant of the whole thesis, so it is annotated rather than left implicit.
    # The LINE stays muted (it is context), but its LABEL does not: 8pt type at
    # 3.5:1 contrast is under the AA floor, and this is the smallest text here.
    ax.axvline(1.0 / kappa, color=INK["muted"], lw=0.7, ls=":", zorder=3)
    direct_label(ax, 1.0 / kappa, ax.get_ylim()[1], r"$\tau^{*}=1/\kappa$",
                 INK["secondary"], dx=4, dy=-8, va="top")

    fig.tight_layout()
    return fig


def _obs_layer_ylim(d):
    """Shared y-limits for fig_clean_curves/fig_noisy_curves, computed once
    from BOTH series so two independently-scaled figures can't make the noise
    level look bigger or smaller than it is -- the same honesty the original
    single-figure sharey=True pairing enforced, now done explicitly since the
    two are separate figures."""
    logF_clean = np.log(d["F_clean"][HERO_PATH])
    logF_obs = d["log_F_obs"][HERO_PATH]
    lo = min(logF_clean.min(), logF_obs.min())
    hi = max(logF_clean.max(), logF_obs.max())
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def _obs_layer_xticks(taus):
    """Shared x-limit/tick logic for fig_clean_curves and fig_noisy_curves, so
    the pair renders at the SAME x-scale even though they are now two
    independent figures -- computed once here rather than duplicated, so the
    two can't quietly drift apart.

    Padding is a FRACTION of the maturity span, not an absolute number of
    years, so a pickle with a different tau grid still gets label room that
    looks the same. The right pad is larger because it holds the date labels.
    """
    span = taus[-1] - taus[0]
    ticks = MaxNLocator(nbins=5, steps=[1, 2, 5, 10]).tick_values(taus[0], taus[-1])
    ticks = ticks[(ticks >= taus[0]) & (ticks <= taus[-1])]
    return (taus[0] - 0.03 * span, taus[-1] + 0.30 * span), ticks


# ---------------------------------------------------------------------------
# Figure 2a/2b -- the observation layer (side-by-side pair)
# ---------------------------------------------------------------------------
def fig_clean_curves(d):
    """Clean closed-form log-futures curves, eight sampled dates, NO sampling
    error -- the reference companion to fig_noisy_curves, which shows the same
    dates with measurement noise added. Comparing the two is what justifies
    the Option-A architecture: every deviation visible in fig_noisy_curves is
    the measurement-noise layer and nothing else.

    SHAPES: taus (K,) = (12,);  F_clean (M, N+1, K) = (100, 1001, 12).
    """
    taus, tgrid = d["taus"], d["t_grid"]
    logF_clean = np.log(d["F_clean"][HERO_PATH])  # (N+1, K) one path, all dates

    # Eight observation dates spread over the horizon. Date is an ORDERED
    # variable, so it gets the sequential ramp: darker = later. The reader
    # recovers the ordering from the ink alone, with no legend lookup.
    date_idx = np.linspace(0, len(tgrid) - 1, 8).astype(int)
    shades = SEQ_BLUE(np.linspace(0.15, 0.95, len(date_idx)))

    fig, ax = plt.subplots(1, 1, figsize=figsize("half", ratio=0.95))
    for colour, i in zip(shades, date_idx):
        ax.plot(taus, logF_clean[i], color=colour, lw=1.2)

    ax.set_ylabel(r"Log futures price $\log \hat{F}(\tau)$")
    ax.set_xlabel(r"Maturity $\tau$ (years)")
    ax.set_ylim(*_obs_layer_ylim(d))
    xlim, ticks = _obs_layer_xticks(taus)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    for j in (0, len(date_idx) - 1):
        i = date_idx[j]
        direct_label(ax, taus[-1], logF_clean[i, -1],
                     f"$t={tgrid[i]:.0f}$", shades[j], dx=5)
    ax.annotate("darker = later date", xy=(0.02, 0.03), xycoords="axes fraction",
                fontsize=8, color=INK["secondary"],
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    fig.tight_layout()
    return fig


def fig_noisy_curves(d):
    """The same eight dates as fig_clean_curves, with the clean curve kept as
    a faint spine and the noisy observed quotes overlaid as points.

    SHAPES: taus (K,) = (12,);  F_clean, log_F_obs (M, N+1, K) = (100, 1001, 12).
    """
    taus, tgrid = d["taus"], d["t_grid"]
    logF_clean = np.log(d["F_clean"][HERO_PATH])
    logF_obs = d["log_F_obs"][HERO_PATH]
    # Read from the pickle, never hardcoded: a mislabelled noise level is worse
    # than a crash, because nothing in the output signals that it is wrong.
    noise_std = d["noise_std"]

    date_idx = np.linspace(0, len(tgrid) - 1, 8).astype(int)
    shades = SEQ_BLUE(np.linspace(0.15, 0.95, len(date_idx)))

    fig, ax = plt.subplots(1, 1, figsize=figsize("half", ratio=0.95))
    for colour, i in zip(shades, date_idx):
        # The clean curve stays as a faint spine so the reader can see what
        # the scatter is deviating FROM -- without it this is just an
        # unreadable cloud.
        ax.plot(taus, logF_clean[i], color=colour, lw=0.7, alpha=0.45)
        ax.plot(taus, logF_obs[i], color=colour, lw=0, marker="o", ms=2.6)

    ax.set_ylabel(r"Log futures price $\log \hat{F}(\tau)$")
    ax.set_xlabel(r"Maturity $\tau$ (years)")
    ax.set_ylim(*_obs_layer_ylim(d))
    xlim, ticks = _obs_layer_xticks(taus)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    # sigma_eps is a data fact (the pickle's own noise_std), not narrative, so it
    # stays on the figure it applies to rather than moving to the caption -- a
    # caption fixed at submission time would go stale if the pickle is ever
    # regenerated with a different noise level.
    ax.annotate(rf"$\sigma_\varepsilon = {noise_std:g}$", xy=(0.02, 0.03),
                xycoords="axes fraction", fontsize=8, color=INK["secondary"],
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    fig.tight_layout()
    return fig


def _identifiability_states(d):
    """Shared (path, date) subsample for fig_identifiability_curves and
    fig_identifiability_scatter, so the two figures always show exactly the
    same underlying states.

    Subsample on a fixed STRIDE rather than a fresh random draw. A
    `default_rng` here would introduce numbers generated at plot time:
    harmless statistically, but it means the figure is no longer a pure
    function of the notebook's output, and re-running with a different seed
    would silently change which states are shown. Striding is deterministic,
    reproducible, and reads every path in the ensemble.
    The stride is OFFSET to start at HERO_PATH so the path followed through
    mc1 and mc2 is actually among the states shown here. A plain
    arange(0, M, 5) starts at 0 and skips 7 entirely, which quietly broke the
    cross-figure thread the chapter tells the reader to follow.
    """
    delta = d["delta_true"]
    M, N1 = delta.shape
    n_paths, n_dates = len(range(HERO_PATH, M, 5)), 8
    pi = np.repeat(np.arange(HERO_PATH, M, 5), n_dates)
    di = np.tile(np.linspace(0, N1 - 1, n_dates).astype(int), n_paths)
    return pi, di, delta[pi, di]


# ---------------------------------------------------------------------------
# Figure 3a/3b -- why the inverse problem is well posed (side-by-side pair)
# ---------------------------------------------------------------------------
def fig_identifiability_curves(d):
    """Curve shape vs delta: each curve is one (path, date) state's log-futures
    curve, coloured by the delta that produced it.

    The methodological claim this pair makes: because the closed form is
    exponential-affine,
        log F(tau_2) - log F(tau_1) = [B(tau_2) - B(tau_1)] * delta + const,
    the term-structure slope is an EXACT affine function of delta. This
    figure shows the curve-shape side of that claim; fig_identifiability_
    scatter shows the resulting slope-delta map directly.

    FORM: diverging, not sequential. delta relative to alpha^Q is a POLARITY
    (backwardation above, contango below) with a meaningful centre, and a
    diverging ramp with a neutral grey midpoint is the encoding that says
    "the middle is nothing happening".
    """
    taus = d["taus"]
    logF_clean = np.log(d["F_clean"])
    alpha_Q = d["alpha_Q"]
    pi, di, dv = _identifiability_states(d)

    # Normalise delta about alpha_Q so 0.5 on the colormap lands exactly on the
    # neutral midpoint -- symmetric limits are what make a diverging map truthful.
    half = np.abs(dv - alpha_Q).max()
    norm = (dv - alpha_Q) / (2 * half) + 0.5

    fig, ax = plt.subplots(1, 1, figsize=figsize("half", ratio=0.95))

    # Curves are shown as deviations from their own front-month value, isolating
    # SHAPE from level: without this the spot-price spread dominates and every
    # curve just looks like a different horizontal line.
    for p_i, d_i, u in zip(pi, di, norm):
        curve = logF_clean[p_i, d_i] - logF_clean[p_i, d_i, 0]
        ax.plot(taus, curve, color=DIV_BLUE_RED(u), lw=0.6, alpha=0.75)
    ax.axhline(0.0, color=INK["muted"], lw=0.7)
    # Reserve an empty BAND above and below the curve envelope for the two regime
    # labels, rather than letting matplotlib pad and then squeezing the text into
    # whatever is left. The fan converges toward tau_1, so its extreme curves
    # sweep upward across any label parked just under them -- a gap measured only
    # at the right-hand edge does not stay a gap. Reserving a full band keeps the
    # labels clear across their whole width.
    ends = logF_clean[pi, di, -1] - logF_clean[pi, di, 0]
    band = 0.16 * (ends.max() - ends.min())   # ~2x an 8pt line at this figure size
    ax.set_ylim(ends.min() - band, ends.max() + band)

    ax.set_xlabel(r"Maturity $\tau$ (years)")
    ax.set_ylabel(r"$\log \hat{F}(\tau) - \log \hat{F}(\tau_1)$")
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
    direct_label(ax, taus[-1], ends.max() + band / 2,
                 r"contango ($\delta < \alpha^{\mathbb{Q}}$)", "#184f95", dx=-4, ha="right")
    direct_label(ax, taus[-1], ends.min() - band / 2,
                 r"backwardation ($\delta > \alpha^{\mathbb{Q}}$)", "#a32020", dx=-4, ha="right")

    fig.tight_layout()
    return fig


def fig_identifiability_scatter(d):
    """The slope-delta map itself: exact affine line (closed form) vs the
    same relationship with measurement noise added. Companion to
    fig_identifiability_curves -- same (path, date) states.
    """
    delta = d["delta_true"]
    logF_clean = np.log(d["F_clean"])
    logF_obs = d["log_F_obs"]
    alpha_Q = d["alpha_Q"]
    pi, di, dv = _identifiability_states(d)

    slope_clean = logF_clean[pi, di, -1] - logF_clean[pi, di, 0]
    slope_obs = logF_obs[pi, di, -1] - logF_obs[pi, di, 0]

    fig, ax = plt.subplots(1, 1, figsize=figsize("half", ratio=0.95))
    ax.scatter(dv, slope_obs, s=5, color=C_DELTA, alpha=0.55,
               lw=0, label="with measurement noise")
    ax.plot(np.sort(dv), slope_clean[np.argsort(dv)],
            color=INK["primary"], lw=1.2, label="exact (closed form)")
    ax.axvline(alpha_Q, color=INK["muted"], lw=0.7, ls=":")  # line: context

    ax.set_xlabel(r"True convenience yield $\delta_t$")
    ax.set_ylabel(r"Slope $\log \hat{F}(\tau_{12}) - \log \hat{F}(\tau_{1})$")
    ax.legend(loc="upper right")
    direct_label(ax, alpha_Q, ax.get_ylim()[0],
                 r"$\alpha^{\mathbb{Q}}$", INK["secondary"], dx=3, dy=6, va="bottom")

    fig.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# Figure 4a-4d -- the basis by maturity, one standalone figure per tau
# ---------------------------------------------------------------------------
# WHY 4, NOT ALL 12: the ensemble spread at a fixed date is |B(tau)| times the
# spread of delta itself, and B(tau) = -(1-e^{-kappa tau})/kappa saturates
# fast -- with kappa ~= 1.9 the mean-reversion timescale is tau* = 1/kappa
# ~= 0.53y, well inside the 1-year contract grid. An earlier 12-figure version
# showed that directly: figures past tau ~= 0.5 were visually indistinguishable
# from each other, so eight of the twelve were redundant ink repeating the same
# "already saturated" band width. Four maturities -- shortest, near tau*, mid,
# longest -- carry the whole story (narrow -> widening -> flat).
#
# These were originally one 4-panel small-multiples figure sharing one y-scale
# across all four, which is exactly what let the reader see the band widen
# with tau at a glance. Split into four standalone figures per this project's
# no-panels rule; each one fixes its y-limits to the SAME shared range
# (_basis_by_maturity_ylim) computed once across all four maturities, so a
# reader placing the four PDFs next to each other still sees the same honest
# comparison the small-multiples layout gave directly.
def _basis_by_maturity_selection(d):
    taus = d["taus"]
    kappa = float(d["params_P"]["kappa"])
    tau_star = 1.0 / kappa
    # Indices into the 12-point tau grid: shortest, the maturity closest to
    # tau* (the mean-reversion timescale -- the natural place for the band to
    # stop widening), a mid maturity past that, and the longest quoted contract.
    sel = [0, int(np.argmin(np.abs(taus - tau_star))), 8, len(taus) - 1]
    return sorted(set(sel))


def _basis_by_maturity_ylim(d):
    t, taus = d["t_grid"], d["taus"]
    S, F = d["S"], d["F_clean"]
    basis = np.log(F) - np.log(S)[:, :, None]
    sel = _basis_by_maturity_selection(d)
    lo, hi = np.nanpercentile(basis[:, :, sel], [1, 99])
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def fig_basis_by_maturity(d, k: int):
    """The log basis log(F/S) at one maturity slot `k`.

    SHAPES: taus (K,) = (12,); S, (M, N+1) = (100, 1001);
            F_clean (M, N+1, K) = (100, 1001, 12); basis is a plain log-and-
            difference of those two pickle outputs, not a new simulation.
    """
    t, taus = d["t_grid"], d["taus"]
    S, F = d["S"], d["F_clean"]
    basis = np.log(F) - np.log(S)[:, :, None]  # (M, N+1, K)
    b = basis[:, :, k]

    fig, ax = plt.subplots(1, 1, figsize=figsize("half", ratio=0.85))

    ax.plot(t, b.T, color=INK["faint"], lw=0.3, alpha=0.6, zorder=1)
    lo, hi = np.percentile(b, [5, 95], axis=0)
    ax.fill_between(t, lo, hi, color=C_DELTA, alpha=0.16, lw=0, zorder=2)
    ax.plot(t, np.median(b, axis=0), color=INK["secondary"], lw=0.8,
             ls="--", zorder=3)
    ax.plot(t, b[HERO_PATH], color=C_DELTA, lw=1.4, zorder=4)
    ax.axhline(0.0, color=INK["muted"], lw=0.5, ls=":", zorder=0)
    ax.annotate(rf"$\tau={taus[k]:.2f}$", xy=(0.06, 0.94), xycoords="axes fraction",
                 color=INK["secondary"], ha="left", va="top", fontsize=9)
    ax.set_ylim(*_basis_by_maturity_ylim(d))
    ax.set_xlabel("$t$ (yrs)")
    ax.set_ylabel(r"$\log(F_\tau/S_t)$")

    legend_elems = [
        Line2D([0], [0], color=INK["faint"], lw=1.2, label=f"ensemble ($M={S.shape[0]}$)"),
        Patch(facecolor=C_DELTA, alpha=0.16, label="5–95%"),
        Line2D([0], [0], color=INK["secondary"], lw=0.8, ls="--", label="median"),
        Line2D([0], [0], color=C_DELTA, lw=1.4, label="single path"),
    ]
    # Unlike every other legend in this module, this one needs an opaque
    # backing: the single path is active enough at some maturities (tau near
    # tau*, especially) to run directly through the upper-right corner, and a
    # frameless legend there becomes illegible against it.
    ax.legend(handles=legend_elems, loc="upper right", fontsize=7,
              frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2c -- the observation layer, as a literal 3D surface (companion to
# mc2a, full width -- see module docstring)
# ---------------------------------------------------------------------------
def fig_term_structure_3d(d, stride: int = 4):
    """The hero path's clean log-futures curve (fig_observation_layer's left
    panel, eight sampled dates) redrawn as a full surface over every
    simulated date instead of eight coloured lines -- shows the same "the
    curve tilts as delta_t moves through its OU cycle" story panel (a) makes,
    but as continuous topography rather than a handful of snapshots.

    Same PROVENANCE rule as the rest of this module: F_clean is read straight
    from the pickle, nothing is recomputed or resimulated.
    """
    taus, tgrid = d["taus"], d["t_grid"]
    logF_clean = np.log(d["F_clean"][HERO_PATH])   # (N+1, K)

    idx = np.arange(0, len(tgrid), stride)
    T, K = np.meshgrid(tgrid[idx], taus, indexing="ij")
    Z = logF_clean[idx]                              # (n_dates_sub, K)

    fig = plt.figure(figsize=figsize("full", ratio=0.72))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T, K, Z, cmap=SEQ_BLUE, rstride=1, cstride=1,
                            linewidth=0.1, edgecolor=(1, 1, 1, 0.25), antialiased=True)

    ax.set_xlabel("Time $t$ (years)", fontsize=8, labelpad=10)
    # No y-axis label TEXT, same reasoning as the dropped z-label: at this
    # view angle mplot3d's tight-bbox extent estimate for the rotated y-axis
    # label clips it against the figure edge regardless of labelpad, margin,
    # or shortening the string. The y-tick numbers (0.2-1.0) are kept and are
    # unambiguous read alongside the x-axis's "Time t (years)"; the axis is
    # named in the caption instead (maturity tau, years).
    # No z-axis label OR tick numbers: height and colour both encode log F,
    # same redundant-channel call as kalman2 and real_data_term_structure_3d
    # -- and the colorbar below now carries the axis name and the exact
    # values, which also sidesteps mplot3d's z-label clipping bug entirely
    # (a colorbar lives in its own 2D axes, never subject to the 3D axes'
    # extent-estimate-before-draw problem that broke a bare z-label here).
    # This used to push the axis name into the external LaTeX caption
    # instead; the colorbar is strictly better since the figure now reads
    # standalone, without depending on caption text to interpret it.
    ax.set_zticklabels([])
    ax.tick_params(labelsize=7, pad=2)
    ax.view_init(elev=24, azim=-55)
    ax.set_box_aspect((2.2, 1, 0.9))
    fig.subplots_adjust(left=0.02, right=0.80, top=0.95, bottom=0.05)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1, aspect=18)
    cbar.set_label(r"$\log \hat F(\tau)$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return fig


def main():
    use_thesis_style()
    d = load()

    for name, builder in (
        ("mc1a_spot_paths", fig_spot_paths),
        ("mc1b_delta_paths", fig_delta_paths),
        ("mc2a_clean_curves", fig_clean_curves),
        ("mc2b_noisy_curves", fig_noisy_curves),
        ("mc2c_term_structure_3d", fig_term_structure_3d),
        ("mc3a_identifiability_curves", fig_identifiability_curves),
        ("mc3b_identifiability_scatter", fig_identifiability_scatter),
    ):
        fig = builder(d)
        path = save(fig, name)
        plt.close(fig)
        print(f"wrote {path.relative_to(REPO)}")

    tau_tag = {0: "shortest", 8: "mid"}  # descriptive suffixes; the tau*-nearest
    # and longest slots are computed from the data, not fixed indices, so their
    # tags are resolved below instead of hardcoded here.
    sel = _basis_by_maturity_selection(d)
    kappa = float(d["params_P"]["kappa"])
    tau_star_idx = int(np.argmin(np.abs(d["taus"] - 1.0 / kappa)))
    tau_tag[tau_star_idx] = "near_taustar"
    tau_tag[len(d["taus"]) - 1] = "longest"
    for k in sel:
        fig = fig_basis_by_maturity(d, k)
        path = save(fig, f"mc4_basis_{tau_tag.get(k, f'tau{k}')}")
        plt.close(fig)
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
