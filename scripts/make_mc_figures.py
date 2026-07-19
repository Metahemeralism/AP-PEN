"""
Methodology-chapter figures for the Gibson-Schwartz Monte Carlo simulator.

Each figure is a SIDE-BY-SIDE pair, because each one makes a comparative claim
that the methodology text needs to assert:

  mc1  the two P-measure state processes    -- S_t diffuses, delta_t mean-reverts
  mc2  the observation layer                -- clean prices vs noisy quotes
  mc3  why the inversion is well posed      -- delta maps affinely to curve slope

Run:  python scripts/make_mc_figures.py
Reads data/mc_data.pkl, writes notebooks/figures/methodology/{pdf,png}.

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

import matplotlib.pyplot as plt
import numpy as np

from gs_wamol.utils.thesis_style import (
    C_DELTA, C_SPOT, DIV_BLUE_RED, INK, SEQ_BLUE,
    despine_shared_y, direct_label, figsize, panel_tag, save, use_thesis_style,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "mc_data.pkl"

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
    ax_s.set_title(r"Geometric diffusion under $\mathbb{P}$", color=INK["secondary"])
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
    ax_d.set_title("Mean reversion under $\\mathbb{P}$", color=INK["secondary"])
    panel_tag(ax_d, "b")
    # Sits INSIDE the axes: an offset label at the far-right data edge is clipped
    # by bbox="tight", which trims to the axes rather than growing to fit it.
    direct_label(ax_d, t[-1], alpha_P, r"$\alpha^{\mathbb{P}}$", INK["secondary"],
                 dx=-4, dy=7, ha="right")
    direct_label(ax_d, t[len(t) // 3], d_hi[len(t) // 3],
                 "5–95%", C_DELTA, dy=4, ha="center")

    # The mean-reversion timescale tau* = 1/kappa is the identifiability-critical
    # constant of the whole thesis, so it is annotated rather than left implicit.
    ax_d.axvline(1.0 / kappa, color=INK["muted"], lw=0.7, ls=":", zorder=3)
    direct_label(ax_d, 1.0 / kappa, ax_d.get_ylim()[1], r"$\tau^{*}=1/\kappa$",
                 INK["muted"], dx=4, dy=-8, va="top")

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
    for ax, tag, title in ((ax_c, "a", "Closed form (exact)"),
                           (ax_o, "b", r"Observed, $\sigma_\varepsilon = 0.01$")):
        ax.set_xlabel(r"Maturity $\tau$ (years)")
        ax.set_title(title, color=INK["secondary"])
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
    for ax in (ax_c, ax_o):
        ax.set_xlim(taus[0] - 0.03, taus[-1] + 0.30)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    for j in (0, len(date_idx) - 1):
        i = date_idx[j]
        direct_label(ax_c, taus[-1], logF_clean[i, -1],
                     f"$t={tgrid[i]:.0f}$", shades[j], dx=5)
    ax_c.annotate("darker = later date", xy=(0.02, 0.03), xycoords="axes fraction",
                  fontsize=8, color=INK["muted"])

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
    M, N1 = delta.shape
    pi = np.repeat(np.arange(0, M, 5), 8)                    # 20 paths
    di = np.tile(np.linspace(0, N1 - 1, 8).astype(int), 20)  # 8 dates each
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
    # Headroom so the topmost tick label never sits under the (a) panel tag.
    # Needed because the data range here depends on which states get sampled --
    # any change to the stride shifts it, so the padding is set relatively
    # rather than as a hardcoded limit.
    ax_curve.margins(y=0.10)

    ax_curve.set_xlabel(r"Maturity $\tau$ (years)")
    ax_curve.set_ylabel(r"$\log \hat{F}(\tau) - \log \hat{F}(\tau_1)$")
    ax_curve.set_title("Curve shape is set by $\\delta$", color=INK["secondary"])
    panel_tag(ax_curve, "a")
    # Text, not colour, names the two regimes -- colour alone would be lost in
    # greyscale and to a colour-blind reader.
    # Each label wears the colour of the ramp arm it names. The ramp runs
    # blue (low delta) -> grey -> red (high delta), and low delta is the CONTANGO
    # (upward-sloping) regime -- so contango is blue and backwardation is red.
    # Getting this pairing backwards silently teaches the reader the wrong
    # mapping, so it is worth stating the direction explicitly rather than
    # trusting that the hex constants were typed in the right order.
    direct_label(ax_curve, taus[-1], ax_curve.get_ylim()[1] * 0.72,
                 r"contango ($\delta < \alpha^{\mathbb{Q}}$)", "#184f95", dx=-4, ha="right")
    direct_label(ax_curve, taus[-1], ax_curve.get_ylim()[0] * 0.72,
                 r"backwardation ($\delta > \alpha^{\mathbb{Q}}$)", "#a32020", dx=-4, ha="right")

    # -- (b) the slope-delta map, clean vs noisy ----------------------------
    slope_clean = logF_clean[pi, di, -1] - logF_clean[pi, di, 0]
    slope_obs = logF_obs[pi, di, -1] - logF_obs[pi, di, 0]

    ax_slope.scatter(dv, slope_obs, s=5, color=C_DELTA, alpha=0.55,
                     lw=0, label="with measurement noise")
    ax_slope.plot(np.sort(dv), slope_clean[np.argsort(dv)],
                  color=INK["primary"], lw=1.2, label="exact (closed form)")
    ax_slope.axvline(alpha_Q, color=INK["muted"], lw=0.7, ls=":")

    ax_slope.set_xlabel(r"True convenience yield $\delta_t$")
    ax_slope.set_ylabel(r"Slope $\log \hat{F}(\tau_{12}) - \log \hat{F}(\tau_{1})$")
    ax_slope.set_title("The map the PINN must invert", color=INK["secondary"])
    ax_slope.legend(loc="upper right")
    panel_tag(ax_slope, "b")
    direct_label(ax_slope, alpha_Q, ax_slope.get_ylim()[0],
                 r"$\alpha^{\mathbb{Q}}$", INK["muted"], dx=3, dy=6, va="bottom")

    fig.tight_layout()
    return fig


def main():
    use_thesis_style()
    d = load()
    for name, builder in (
        ("mc1_state_paths", fig_state_paths),
        ("mc2_observation_layer", fig_observation_layer),
        ("mc3_identifiability", fig_identifiability),
    ):
        fig = builder(d)
        path = save(fig, name)
        plt.close(fig)
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
