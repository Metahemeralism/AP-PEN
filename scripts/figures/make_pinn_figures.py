"""
PINN-performance figures for the Gibson-Schwartz convenience-yield inversion.

Four figures, none of which a results table can substitute for:

  pinn0   training history        -- per-channel training-loss curves (e_data,
                                    e_ODE, e_SDE, e_cac, e_rcac), one panel per
                                    channel, all five variants overlaid where
                                    each actually trains that channel
  pinn1   parameter convergence   -- learned psi vs epoch, MLP/PINN/AP-PINN vs
                                    truth, with the final relative error % read
                                    directly off each curve's endpoint
  pinn2a  delta recovery, 2D      -- true delta_t and AP-PINN's delta_hat(t)
                                    overlaid, with a signed-error panel
                                    underneath
  pinn3   path-repeat validation -- 20 independent Monte Carlo noise draws,
                                    each refit from scratch: signed-error
                                    traces overlaid (all 20, faint, + mean)
                                    alongside the per-path RMSE spread. Answers
                                    "is recovery consistent across draws," which
                                    a 20-panel small-multiples grid (one subplot
                                    per path) leaves the reader to eyeball and
                                    aggregate by hand.

(pinn2b, a 3D ribbon variant of pinn2a, and an earlier pinn3, a collocation-
vs-data-coverage scatter, were dropped -- kept as an idea in git history if
ever wanted again, not carried forward as figures.)

Run:  python scripts/figures/make_pinn_figures.py
Reads data/input/synthetic/mc_data.pkl, data/output/mc_simulated/results/
results_{MLP,PINN,APPINN_nobal,APPINN,APPINN_ARB}.pkl, the orbax checkpoint at
data/output/mc_simulated/checkpoints/APPINN, and (for pinn3 only)
data/output/mc_simulated/results/path_repeat.pkl -- the same AP-PINN
(config_APPINN: loss-balanced, no arb constraints) 20-path repeat validation,
so pinn1/pinn2a and pinn3 are the same model at two granularities (single
path vs. 20-path repeat), not two different models. Writes figures/pinn/{pdf,png}.

PROVENANCE
----------
pinn0 reads the same per-epoch history as pinn1, just the loss-channel
values (history[2]) rather than the psi snapshots (history[3]) -- nothing
is recomputed.

pinn1 reads the psi values the training loop itself logged every epoch
(notebooks/PINN_implementation.ipynb, cell 18) -- nothing is recomputed.

pinn2a plots delta_true straight from mc_data.pkl (the ground truth the notebook
never shows the model) against delta_hat(t), a bare forward pass of the trained
AP-PINN checkpoint; no weights are touched or retrained here.

pinn3 reads data/output/mc_simulated/results/path_repeat.pkl, written by the
notebook's 20-path repeat-validation cell (each path's own delta_true/delta_hat
already computed there) -- nothing is retrained or recomputed here either.

ARCHITECTURE NOTE -- why the network classes are duplicated here
------------------------------------------------------------------
notebooks/PINN_implementation.ipynb defines its own Dense/MLP (no output
activation) rather than importing src/gs_wamol/models/mlp.MLP, which appends
nn.softplus() to force positive outputs. delta_hat and A_hat are both
signed quantities here, so the softplus-constrained package MLP would silently
produce the wrong network and could not load these checkpoint weights back
correctly (shapes match; the forward pass would not). This script therefore
mirrors the notebook's own MLP definition, not the package one.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
from flax import linen as nn
from jax import random, vmap
from jax.nn.initializers import glorot_normal, normal, zeros
from matplotlib.lines import Line2D

from gs_wamol.physics.gibson_schwartz import GSParams
from gs_wamol.utils.thesis_style import (
    C_MODEL, INK, PALETTE,
    direct_label, figsize, save, use_thesis_style,
)

REPO = Path(__file__).resolve().parents[2]
MC_DATA = REPO / "data" / "input" / "synthetic" / "mc_data.pkl"
RESULTS_DIR = REPO / "data" / "output" / "mc_simulated" / "results"
CKPT_APPINN = REPO / "data" / "output" / "mc_simulated" / "checkpoints" / "APPINN"
PATH_REPEAT = REPO / "data" / "output" / "mc_simulated" / "results" / "path_repeat.pkl"
FIGDIR = REPO / "figures" / "pinn"

ORDER = ["kappa", "sigma1", "sigma2", "rho", "alpha_Q", "alpha_P"]
LABEL = {
    "kappa": r"$\kappa$", "sigma1": r"$\sigma_1$", "sigma2": r"$\sigma_2$",
    "rho": r"$\rho$", "alpha_Q": r"$\alpha^{\mathbb{Q}}$", "alpha_P": r"$\alpha^{\mathbb{P}}$",
}
# One colour/display-name per model variant, fixed across every figure in this
# module. AP-PINN gets C_MODEL (violet) -- the same "fitted/recovered quantity"
# hue used document-wide -- because it is the thesis's proposed model, not a
# baseline. AP-PINN (nobal) gets a lighter tint of the same violet (it's the
# same architecture minus one mechanism, not a different model); AP-PINN (ARB)
# gets blue, the remaining PALETTE hue, since it genuinely adds new loss terms.
# Internal tag (matches run_tag(config) / checkpoint dir names) vs. display name
# (what actually renders in the legend) are kept separate so a display-name-only
# change never risks touching a file path.
MODEL_COLOUR = {
    "MLP": PALETTE["aqua"], "PINN": PALETTE["orange"],
    "APPINN_nobal": "#aea6d7", "APPINN": C_MODEL, "APPINN_ARB": PALETTE["blue"],
}
DISPLAY_NAME = {
    "MLP": "MLP", "PINN": "PINN",
    "APPINN_nobal": "AP-PINN (nobal)", "APPINN": "AP-PINN", "APPINN_ARB": "AP-PINN (ARB)",
}


def load_mc():
    with open(MC_DATA, "rb") as f:
        d = pickle.load(f)
    return {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Figure 0 -- per-channel training history
# ---------------------------------------------------------------------------
def load_history(tag: str):
    with open(RESULTS_DIR / f"results_{tag}.pkl", "rb") as f:
        return pickle.load(f)["history"]


CHANNELS = ["e_data", "e_ode", "e_sde"]
CHANNEL_LABEL = {
    "e_data": r"$e_{\mathrm{data}}$", "e_ode": r"$e_{\mathrm{ODE}}$", "e_sde": r"$e_{\mathrm{SDE}}$",
}
# e_sde is a signed OU transition NLL (can go negative) -- log-scale would clip it.
CHANNEL_YSCALE = {"e_data": "log", "e_ode": "log", "e_sde": "linear"}


def _rolling_mean(x, window):
    """Trailing rolling mean, same length as x (early points average over
    however many samples are available). Per-epoch losses are extremely
    jagged (each epoch is one full-batch gradient step, not an average over
    many minibatches), so the raw curves are mostly visual noise that hides
    the actual convergence trend and any real separation between variants --
    smoothing is what makes the trend legible, not a stylistic choice.
    """
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    out[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
    for i in range(window - 1):
        out[i] = x[: i + 1].mean()
    return out


# Which loss channels each variant actually trains on. error() (the notebook's
# eval function) always computes all five channels for every variant, whether
# or not they're in that variant's loss -- e.g. MLP's e_ode/e_cac/e_rcac are
# real numbers in its results pickle, but they're incidental (never optimized),
# not evidence about MLP's training dynamics. Plotting them would be misleading,
# so each panel only shows the variants for which that channel is a live loss
# term. Mirrors VARIANT_COMPONENTS in PINN_implementation.ipynb (cell e8d1d583).
TRAINED_CHANNELS = {
    "MLP": {"e_data"},
    "PINN": {"e_data", "e_ode"},
    "APPINN_nobal": {"e_data", "e_ode", "e_sde"},
    "APPINN": {"e_data", "e_ode", "e_sde"},
    "APPINN_ARB": {"e_data", "e_ode", "e_sde", "e_cac", "e_rcac"},
}
# Several variants' e_data/e_ode trajectories land almost exactly on top of one
# another (they converge to near-identical final values -- docs/project_summary.md's
# psi table), so colour alone would leave whichever tag is drawn last fully
# occluding the rest, no matter the alpha. A distinct dash pattern per variant
# means every curve stays identifiable even where it's pixel-for-pixel identical
# to another -- same "never rely on colour alone" rule the TikZ figures follow.
LINESTYLE = {
    "MLP": (0, ()), "PINN": (0, (5, 2)), "APPINN_nobal": (0, (1, 1)),
    "APPINN": (0, (3, 1, 1, 1)), "APPINN_ARB": (0, (4, 1, 1, 1, 1, 1)),
}


def fig_training_history():
    """Per-channel training-loss curves: e_data/e_ODE/e_SDE, one panel each,
    all five model variants overlaid where that variant actually trains the
    channel, plus a fourth panel for the no-arbitrage hinges (e_cac/e_rcac),
    which only AP-PINN (ARB) ever trains.

    Two problems with a naive "one raw panel per channel, 5 channels" version:
    each per-epoch loss is one full-batch gradient step, not a minibatch
    average, so the raw curves are dominated by point-to-point noise that
    buries both the real convergence trend and any real separation between
    variants -- fixed by rolling-mean smoothing (_rolling_mean). And e_cac/
    e_rcac each collapse to exactly 0 within ~200 epochs and stay there for
    the remaining ~39800 (checked directly against the results pickle), so a
    full-width 0-40000 panel for either is >99% dead space -- fixed by
    cropping to the actual transient and merging both into one panel, since
    they're the same story (a hinge constraint satisfied almost immediately)
    on the same model.
    """
    tags = ["MLP", "PINN", "APPINN_nobal", "APPINN", "APPINN_ARB"]
    hists = {tag: load_history(tag) for tag in tags}

    LOG_FLOOR = 1e-9  # display floor: an exact-zero hinge loss can't sit on a
    # log axis and would otherwise vanish instead of reading as "at zero."
    SMOOTH_WINDOW = 200  # ~200 effective points across 40k epochs -- enough
    # to cancel per-epoch noise without flattening the real convergence shape.
    ARB_CROP = 500  # epochs -- both e_cac and e_rcac are firmly at the floor
    # well before this (last nonzero epoch: 199 and 109 respectively).

    # Single-column thesis width like every other pinn* figure (figsize("full", ...),
    # TEXT_WIDTH_IN=6.3in) -- NOT the notebook's own figsize=(24, 4.5), which only
    # reads as reasonably sized because a notebook cell has no page-width limit.
    # Stacked vertically (4x1), not side by side (1x4): 40k noisy epochs need
    # real horizontal room to read, and four panels crammed into 6.3in width
    # left each one too narrow regardless of height. Full width per panel,
    # stacked down the page, is the actual single-column-figure convention.
    fig, axes = plt.subplots(4, 1, figsize=figsize("full", ratio=1.8))
    *chan_axes, ax_arb = axes

    for ax, ch in zip(chan_axes, CHANNELS):
        for tag in tags:
            if ch not in TRAINED_CHANNELS[tag]:
                continue
            hist = hists[tag]
            epochs = [h[0] for h in hist]
            vals = np.asarray([h[2][ch] for h in hist])
            if CHANNEL_YSCALE[ch] == "log":
                vals = np.clip(vals, LOG_FLOOR, None)
            vals = _rolling_mean(vals, SMOOTH_WINDOW)
            ax.plot(epochs, vals, color=MODEL_COLOUR[tag], ls=LINESTYLE[tag],
                     lw=1.3, zorder=3)
        ax.set_yscale(CHANNEL_YSCALE[ch])
        ax.set_ylabel(CHANNEL_LABEL[ch], color=INK["secondary"], fontsize=12)
        ax.set_xlabel("epoch", fontsize=11)
        ax.tick_params(labelsize=10)

    # Fourth panel: the no-arbitrage hinges, AP-PINN (ARB) only, raw (not
    # smoothed) -- the sharp initial spike-and-decay IS the story here, and
    # smoothing a transient this short would blur exactly that.
    hist = hists["APPINN_ARB"]
    epochs = np.array([h[0] for h in hist])
    crop = epochs <= ARB_CROP
    e_cac = np.clip(np.array([h[2]["e_cac"] for h in hist]), LOG_FLOOR, None)
    e_rcac = np.clip(np.array([h[2]["e_rcac"] for h in hist]), LOG_FLOOR, None)
    ax_arb.plot(epochs[crop], e_cac[crop], color=MODEL_COLOUR["APPINN_ARB"],
                 ls="solid", lw=1.3, zorder=3)
    ax_arb.plot(epochs[crop], e_rcac[crop], color=MODEL_COLOUR["APPINN_ARB"],
                 ls=(0, (1, 1)), lw=1.3, zorder=3)
    ax_arb.set_yscale("log")
    ax_arb.set_ylabel(r"$e_{\mathrm{cac}},\,e_{\mathrm{rcac}}$", color=INK["secondary"], fontsize=12)
    ax_arb.set_xlabel("epoch", fontsize=11)
    ax_arb.tick_params(labelsize=10)
    # Neither the top (collides with the "$10^2$" y-tick label) nor the bottom
    # (both curves sit AT the floor across the full x-range once settled, so a
    # flat line runs along the entire bottom edge) is actually empty. The only
    # clear space is mid-height on the right, past the epoch~200 spike and
    # well above the floor line.
    ax_arb.annotate(r"$e_{\mathrm{cac}}$ solid, $e_{\mathrm{rcac}}$ dotted"
                      "\n(AP-PINN ARB only)",
                      xy=(0.97, 0.55), xycoords="axes fraction", color=INK["secondary"],
                      ha="right", va="center", fontsize=9)

    legend_elems = [Line2D([0], [0], color=MODEL_COLOUR[t], ls=LINESTYLE[t], lw=1.6, label=DISPLAY_NAME[t])
                     for t in tags]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.024), frameon=False, fontsize=10)

    # Bottom margin as a FRACTION of total figure height, not an absolute size --
    # re-tuned each time ratio changes (currently 1.7) so the legend gutter
    # stays a constant ~0.38in regardless of how tall the figure gets.
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return fig


# ---------------------------------------------------------------------------
# Figure 1 -- parameter convergence
# ---------------------------------------------------------------------------
def fig_psi_convergence(psi_true):
    """Learned psi vs epoch, one panel per parameter, all three model variants.

    Each panel ends with the relative error |theta_hat - theta_true|/|theta_true|
    annotated directly at the curve's own endpoint -- the reviewer reads
    calibration accuracy off the plot without cross-referencing a table.
    """
    tags = ["MLP", "PINN", "APPINN"]
    hists = {tag: load_history(tag) for tag in tags}

    fig, axes = plt.subplots(2, 3, figsize=figsize("full", ratio=0.62))

    for ax, p in zip(axes.ravel(), ORDER):
        true_v = psi_true[p]
        ax.axhline(true_v, color=INK["secondary"], lw=1.0, ls="--", zorder=1)

        endpoints = {}
        for tag in tags:
            hist = hists[tag]
            epochs = [h[0] for h in hist]
            vals = [h[3][p] for h in hist]
            ax.plot(epochs, vals, color=MODEL_COLOUR[tag], lw=1.3, zorder=3)
            endpoints[tag] = (epochs[-1], vals[-1])

        # Declutter the endpoint labels: MLP and PINN share every loss channel
        # except e_ode, so on several parameters (sigma1, sigma2, rho) they
        # converge to nearly the same value and their raw endpoint y's would
        # print two labels on top of each other. Push apart in axes units
        # (computed from the CURRENT autoscaled ylim, before the labels
        # themselves add any headroom) rather than in data units, so the same
        # minimum gap looks the same across panels with very different scales.
        ylo, yhi = ax.get_ylim()
        min_gap = 0.09 * (yhi - ylo)
        label_y, prev = {}, None
        for tag in sorted(tags, key=lambda t: endpoints[t][1]):
            y = endpoints[tag][1]
            if prev is not None and y - prev < min_gap:
                y = prev + min_gap
            label_y[tag] = y
            prev = y

        for tag in tags:
            x_end, final = endpoints[tag]
            rel_err = abs(final - true_v) / abs(true_v) * 100
            direct_label(ax, x_end, label_y[tag], f"{rel_err:.0f}%",
                         MODEL_COLOUR[tag], dx=6, dy=0, size=7)

        ax.set_ylabel(LABEL[p], color=INK["secondary"], fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        # Headroom on the right so the endpoint % labels never clip the axes.
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0, x1 + 0.22 * (x1 - x0))

    legend_elems = [Line2D([0], [0], color=INK["secondary"], lw=1.0, ls="--", label="true")]
    legend_elems += [Line2D([0], [0], color=MODEL_COLOUR[t], lw=1.6, label=DISPLAY_NAME[t]) for t in tags]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# ---------------------------------------------------------------------------
# Figure 2 -- price-surface reconstruction
# ---------------------------------------------------------------------------
def _weight_fact(init_fn, mean, stddev):
    def init(key, shape):
        key1, key2 = random.split(key)
        w = init_fn(key1, shape)
        g = mean + normal(stddev)(key2, (shape[-1],))
        v = w / jnp.exp(g)
        return jnp.exp(g), v
    return init


class Dense(nn.Module):
    """Bit-for-bit copy of the notebook's Dense layer (see module docstring).

    kernel_init/bias_init MUST carry a type annotation (Callable) here: flax's
    Module dataclass transform only registers annotated attributes as fields,
    and an unannotated `kernel_init = glorot_normal()` breaks self.features
    resolution inside self.param(...) in a way that surfaces as an opaque
    "iteration over a 0-d array" from the shape canonicaliser.
    """
    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    reparam: dict | None = None

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param("kernel", self.kernel_init, (x.shape[-1], self.features))
        else:
            g, v = self.param("kernel", _weight_fact(self.kernel_init, **self.reparam), (x.shape[-1], self.features))
            kernel = g * v
        bias = self.param("bias", self.bias_init, (self.features,))
        return jnp.dot(x, kernel) + bias


class NotebookMLP(nn.Module):
    """No output activation -- unlike gs_wamol.models.mlp.MLP, whose trailing
    softplus would be wrong here (delta_hat and A_hat are signed)."""
    hidden_dim: tuple
    out_dim: int = 1
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act = jnp.tanh if self.activation == "tanh" else jnp.sin
        for dim in self.hidden_dim:
            x = act(Dense(features=dim)(x))
        return Dense(features=self.out_dim)(x)


def restore_appinn(t_scale: float, tau_scale: float):
    """Restore the trained AP-PINN (config_APPINN) checkpoint and return
    (delta_hat_fn, A_hat_fn, gsp).

    Reconstructs psi (learned, unconstrained) with the same softplus/tanh
    constraints the notebook trains under (build_nets, cell 8) -- these are
    the ONLY place the true parameter values could leak into the reconstructed
    functions, and they do not: psi comes entirely from the checkpoint.
    """
    ckpt = orbax.checkpoint.PyTreeCheckpointer().restore(str(CKPT_APPINN))
    params = ckpt["params"]

    drift_ann = NotebookMLP(hidden_dim=(64, 64, 64, 64), out_dim=1, activation="tanh")
    path_ann = NotebookMLP(hidden_dim=(32, 32, 32), out_dim=1, activation="tanh")

    psi_raw = params["psi"]
    psi = {
        "kappa": jax.nn.softplus(psi_raw["kappa"]),
        "sigma1": jax.nn.softplus(psi_raw["sigma1"]),
        "sigma2": jax.nn.softplus(psi_raw["sigma2"]),
        "rho": jnp.tanh(psi_raw["rho"]),
        "alpha_Q": psi_raw["alpha_Q"],
        "alpha_P": psi_raw["alpha_P"],
    }
    gsp = {"p_Q": GSParams(r=0.05, kappa=float(psi["kappa"]), alpha_Q=float(psi["alpha_Q"]),
                            sigma1=float(psi["sigma1"]), sigma2=float(psi["sigma2"]), rho=float(psi["rho"])),
           "alpha_P": float(psi["alpha_P"])}

    def A_hat_fn(tau):
        return tau * drift_ann.apply(params["drift"], jnp.atleast_1d(tau / tau_scale))[0]

    def delta_hat_fn(t):
        return path_ann.apply(params["path"], jnp.atleast_1d(t / t_scale))[0]

    return delta_hat_fn, A_hat_fn, gsp


def _delta_recovery(d):
    """Shared true/recovered delta_t series + fit-quality metrics for both
    the 3D ribbon figure and its plain 2D companion, so the two never drift
    apart by using two separately-computed forward passes."""
    t = d["t_grid"]
    delta_true = d["delta_true"][0]   # path_id=0, matching what the notebook trained on
    taus = d["taus"]

    delta_hat_fn, _, _ = restore_appinn(t_scale=float(t[-1]), tau_scale=float(taus.max()))
    delta_hat = np.asarray(vmap(delta_hat_fn)(t))

    # Fit-quality decomposition, same identity the notebook's eval cell uses:
    # MSE = bias^2 + (sd_hat - sd_true)^2 + 2 sd_hat sd_true (1 - corr).
    metrics = {
        "bias": float(np.mean(delta_hat - delta_true)),
        "sd_true": float(np.std(delta_true)),
        "sd_hat": float(np.std(delta_hat)),
        "corr": float(np.corrcoef(delta_hat, delta_true)[0, 1]),
        "rmse": float(np.sqrt(np.mean((delta_hat - delta_true) ** 2))),
    }
    return t, delta_true, delta_hat, metrics


def fig_delta_recovery_2d(d):
    """True vs. delta_hat overlay for a single path, plus the signed error
    underneath. When the two curves are this close -- corr > 0.97 here -- the
    strongest way to show it is the most boring one: overlay them directly and
    let them visibly coincide. The error panel underneath then answers "close
    by how much, and where does it stop being close."
    """
    t, delta_true, delta_hat, m = _delta_recovery(d)
    err = delta_hat - delta_true

    fig, (ax_top, ax_err) = plt.subplots(
        2, 1, figsize=figsize("full", ratio=0.55), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08},
    )

    ax_top.plot(t, delta_true, color=INK["primary"], lw=1.4, zorder=3, label="true")
    ax_top.plot(t, delta_hat, color=C_MODEL, lw=1.2, ls="--", zorder=4, label=DISPLAY_NAME["APPINN"])
    ax_top.set_ylabel(r"$\delta_t$")
    ax_top.legend(loc="upper right", ncol=2, frameon=False)

    ax_err.axhline(0.0, color=INK["muted"], lw=0.7, ls=":", zorder=1)
    ax_err.fill_between(t, 0.0, err, color=C_MODEL, alpha=0.25, lw=0, zorder=2)
    ax_err.plot(t, err, color=C_MODEL, lw=0.8, zorder=3)
    ax_err.axhline(m["rmse"], color=INK["secondary"], lw=0.6, ls="--", zorder=1)
    ax_err.axhline(-m["rmse"], color=INK["secondary"], lw=0.6, ls="--", zorder=1)
    ax_err.set_ylabel(r"$\hat\delta-\delta$")
    ax_err.set_xlabel("$t$ (yrs)")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 -- path-repeat validation (AP-PINN, 20 independent MC noise draws)
# ---------------------------------------------------------------------------
def load_path_repeat():
    with open(PATH_REPEAT, "rb") as f:
        return pickle.load(f)


def fig_path_repeat_validation(pr):
    """Recovery consistency across 20 independently-trained AP-PINN fits, one
    per Monte Carlo noise draw.

    WHY NOT A 20-PANEL GRID: the notebook's own ad hoc version of this (one
    subplot per path, true vs. recovered) makes the reader eyeball 20 near-
    identical mini time series and mentally aggregate "are these consistently
    good." That aggregation is exactly what a thesis figure should do FOR the
    reader, not ask them to redo by eye. Two panels here answer the two
    questions the grid only implies: (a) does the recovery error stay small
    and centred on zero across every draw -- all 20 signed-error traces
    overlaid, faint, with the mean traced boldly; and (b) how tight is that
    across draws -- the per-path RMSE as a strip plot (20 points is too few
    for a histogram to read as anything but noisy bars).
    """
    t = pr["t_train"]
    results = pr["results"]
    errs = np.stack([r["delta_hat"] - r["delta_true"] for r in results])   # (20, N)
    rmses = np.array([r["delta_rmse"] for r in results])
    mean_err = errs.mean(axis=0)
    pooled_rmse = float(np.sqrt(np.mean(errs ** 2)))

    fig, (ax_err, ax_rmse) = plt.subplots(
        1, 2, figsize=figsize("full", ratio=0.42),
        gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.3},
    )

    # (a) every path's signed error, faint; the mean, bold; pooled RMSE band dashed
    for e in errs:
        ax_err.plot(t, e, color=C_MODEL, lw=0.5, alpha=0.18, zorder=1)
    ax_err.axhline(0.0, color=INK["muted"], lw=0.7, ls=":", zorder=2)
    ax_err.axhline(pooled_rmse, color=INK["secondary"], lw=0.6, ls="--", zorder=2)
    ax_err.axhline(-pooled_rmse, color=INK["secondary"], lw=0.6, ls="--", zorder=2)
    ax_err.plot(t, mean_err, color=C_MODEL, lw=1.6, zorder=3)
    # Axes-fraction placement (not direct_label's data-coord + halo convention):
    # bottom-left corner, plain black text, no white halo stroke.
    ax_err.annotate(f"RMSE {pooled_rmse:.3f}", xy=(0.02, 0.04), xycoords="axes fraction",
                     color=INK["primary"], ha="left", va="bottom", fontsize=7)
    ax_err.set_xlabel("$t$ (yrs)")
    ax_err.set_ylabel(r"$\hat\delta_t-\delta_t$")

    # (b) per-path RMSE, strip plot -- too few points (20) for a histogram
    rng = np.random.default_rng(0)   # jitter only, not part of the model's own randomness
    jitter = rng.uniform(-0.15, 0.15, size=len(rmses))
    ax_rmse.scatter(jitter, rmses, s=14, color=C_MODEL, alpha=0.7, zorder=3, lw=0)
    ax_rmse.errorbar([0], [rmses.mean()], yerr=[rmses.std()], fmt="_", color=INK["primary"],
                      markersize=20, markeredgewidth=1.8, capsize=4, elinewidth=1.4, zorder=4)
    ax_rmse.set_xlim(-0.5, 0.5)
    ax_rmse.set_xticks([])
    ax_rmse.set_ylabel("per-path RMSE")
    # Corner annotation, not a title (no text above the axes box) -- same plain
    # black, no-halo convention as ax_err's RMSE label. The point cloud spans
    # nearly the full y-range (top AND bottom corners both have a point close
    # by), but jitter is confined to x in [-0.15, 0.15] out of an xlim of
    # [-0.5, 0.5] -- so the side margin is clear at any y. Rotated to fit the
    # narrow panel width.
    ax_rmse.annotate(f"mean {rmses.mean():.3f} $\\pm$ {rmses.std():.3f}", xy=(0.30, 0.5),
                      xycoords=("data", "axes fraction"), color=INK["primary"],
                      ha="center", va="center", rotation=90, fontsize=7)

    fig.tight_layout()
    return fig


def main():
    use_thesis_style()
    d = load_mc()
    p_Q = d["params_Q"] if not isinstance(d["params_Q"], dict) else GSParams(**d["params_Q"])
    psi_true = {
        "kappa": float(p_Q.kappa), "sigma1": float(p_Q.sigma1), "sigma2": float(p_Q.sigma2),
        "rho": float(p_Q.rho), "alpha_Q": float(p_Q.alpha_Q),
        "alpha_P": float(d["params_P"]["alpha_P"]),
    }

    for name, builder, args in (
        ("pinn0_training_history", fig_training_history, ()),
        ("pinn1_psi_convergence", fig_psi_convergence, (psi_true,)),
        ("pinn2a_delta_recovery_2d", fig_delta_recovery_2d, (d,)),
    ):
        fig = builder(*args)
        path = save(fig, name, directory=FIGDIR)
        plt.close(fig)
        print(f"wrote {path.relative_to(REPO)}")

    if PATH_REPEAT.exists():
        fig = fig_path_repeat_validation(load_path_repeat())
        path = save(fig, "pinn3_path_repeat_validation", directory=FIGDIR)
        plt.close(fig)
        print(f"wrote {path.relative_to(REPO)}")
    else:
        print(f"skipped pinn3_path_repeat_validation -- {PATH_REPEAT.relative_to(REPO)} not found yet")


if __name__ == "__main__":
    main()
