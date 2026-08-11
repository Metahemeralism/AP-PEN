"""
AP-PEN performance figures for the Gibson-Schwartz convenience-yield inversion.

No multi-panel composites (see the sizing note in make_kalman_figures.py) --
eleven standalone figures, full width or "half" (sized to sit next to a named
partner as two separate floats), none of which a results table can substitute
for:

  pinn0a/b/c  training history       -- per-channel training-loss curves,
                                       one figure per channel (e_data, e_SDE,
                                       and the three no-arbitrage hinges
                                       e_cac/e_rcac/e_delta_floor together),
                                       all four variants overlaid where each
                                       actually trains that channel
  pinn1_*     parameter convergence  -- six figures, one per psi parameter,
                                       learned value vs epoch for all four
                                       variants vs truth, with the final
                                       relative error % read directly off
                                       each curve's endpoint
  pinn2a/b    delta recovery         -- true delta_t and AP-PEN's delta_hat(t)
                                       overlaid (pinn2a), and the signed error
                                       between them (pinn2b), for a single path
  pinn2c      delta recovery, all    -- the ablation companion to pinn2a: true
              variants                 delta_t vs delta_hat for MLP/APPINN/
                                       APPINN_ARB together on the same single
                                       path (full domain, no holdout).
                                       APPINN_nobal excluded -- PROVABLY
                                       identical to APPINN under this
                                       architecture, not just empirically
                                       close (see DELTA_RECOVERY_TAGS)
  pinn3a-c    path-repeat validation -- 20 independent Monte Carlo noise
                                       draws, each refit from scratch:
                                       signed-error traces overlaid (all 20,
                                       faint, + mean, pinn3a), the per-path
                                       RMSE spread (pinn3b), and true-vs-
                                       recovered delta pooled across all 20
                                       paths as a hexbin density scatter
                                       against the y=x line (pinn3c, with
                                       path 0 -- pinn2a/b's single-path
                                       illustration -- traced through it).
                                       Answers "is recovery consistent
                                       across draws," which a 20-panel
                                       small-multiples grid (one subplot per
                                       path) leaves the reader to eyeball and
                                       aggregate by hand.

(pinn2b as a 3D ribbon variant of pinn2a, and an earlier pinn3, a collocation-
vs-data-coverage scatter, were dropped -- kept as an idea in git history if
ever wanted again, not carried forward as figures. The pinn2b name above was
reused for the delta-recovery error figure once the ribbon idea was dropped.)

Run:  python scripts/figures/make_pinn_figures.py
Reads data/input/synthetic/mc_data.pkl, data/output/mc_simulated_single_net/
results/results_{MLP,APPINN_nobal,APPINN,APPINN_ARB}.pkl,
data/output/mc_simulated_single_net/results/path_repeat.pkl (for pinn2a/b and
pinn3a-c -- all come from that file, see PROVENANCE), and
data/output/mc_simulated_single_net/results/delta_recovery_all_variants.pkl
(for pinn2c). Writes figures/pinn/{pdf,png}.

ARCHITECTURE (single-network AP-PEN, not the two-network predecessor)
-----------------------------------------------------------------------
notebooks/AP-PEN.ipynb trains ONE network, the latent path
$\\hat\\delta_\\phi$; $A(\\tau)$ and $B(\\tau)$ are both analytic closed forms,
not learned, so there is no drift/ODE-residual channel and no second network
to restore. Four variants, no `PINN` tier (that tier existed only to isolate
the ODE-residual channel the two-network architecture had; nothing here does).

PROVENANCE
----------
pinn0 reads the same per-epoch history as pinn1, just the loss-channel
values (history[2]) rather than the psi snapshots (history[3]) -- nothing
is recomputed.

pinn1 reads the psi values the training loop itself logged every epoch
(notebooks/AP-PEN.ipynb, cell 19's `calibration`) -- nothing
is recomputed.

pinn2a and pinn3 both read data/output/mc_simulated_single_net/results/
path_repeat.pkl, written by the notebook's 20-path repeat-validation cell,
which already stores each path's own delta_true/delta_hat/delta_rmse (path
0 is one of the 20 draws, used here as the single-path illustration; nothing
is retrained, recomputed, or restored from a checkpoint by this script).
Earlier versions of this script restored an orbax checkpoint and replayed a
forward pass to get pinn2a's delta_hat; that machinery assumed a second
(drift) network that no longer exists in this architecture and has been
removed -- path_repeat.pkl already carries everything pinn2a needs.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from gs_wamol.physics.gibson_schwartz import GSParams
from gs_wamol.utils.thesis_style import (
    C_MODEL, INK, PALETTE, SEQ_BLUE,
    direct_label, figsize, save, use_thesis_style,
)

REPO = Path(__file__).resolve().parents[2]
MC_DATA = REPO / "data" / "input" / "synthetic" / "mc_data.pkl"
RESULTS_DIR = REPO / "data" / "output" / "mc_simulated_single_net" / "results"
PATH_REPEAT = REPO / "data" / "output" / "mc_simulated_single_net" / "results" / "path_repeat.pkl"
DELTA_RECOVERY_ALL = REPO / "data" / "output" / "mc_simulated_single_net" / "results" / "delta_recovery_all_variants.pkl"
FIGDIR = REPO / "figures" / "pinn"

ORDER = ["kappa", "sigma1", "sigma2", "rho", "alpha_Q", "alpha_P"]
LABEL = {
    "kappa": r"$\kappa$", "sigma1": r"$\sigma_1$", "sigma2": r"$\sigma_2$",
    "rho": r"$\rho$", "alpha_Q": r"$\alpha^{\mathbb{Q}}$", "alpha_P": r"$\alpha^{\mathbb{P}}$",
}
# One colour/display-name per model variant, fixed across every figure in this
# module. AP-PEN gets C_MODEL (violet) -- the same "fitted/recovered quantity"
# hue used document-wide -- because it is the thesis's proposed model, not a
# baseline. AP-PEN (no balancing) gets a lighter tint of the same violet (it's
# the same architecture minus one mechanism, not a different model); AP-PEN
# (ARB) gets blue, the remaining PALETTE hue, since it genuinely adds new loss
# terms. Internal tag (matches run_tag(config) / results-pickle filenames) vs.
# display name (what actually renders in the legend) are kept separate so a
# display-name-only change never risks touching a file path.
MODEL_COLOUR = {
    "MLP": PALETTE["aqua"],
    "APPINN_nobal": "#aea6d7", "APPINN": C_MODEL, "APPINN_ARB": PALETTE["blue"],
}
DISPLAY_NAME = {
    "MLP": "MLP",
    "APPINN_nobal": "AP-PEN (no balancing)", "APPINN": "AP-PEN", "APPINN_ARB": "AP-PEN (ARB)",
}
TAGS = ["MLP", "APPINN_nobal", "APPINN", "APPINN_ARB"]


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


CHANNELS = ["e_data", "e_sde"]
CHANNEL_LABEL = {
    "e_data": r"$e_{\mathrm{data}}$", "e_sde": r"$e_{\mathrm{SDE}}$",
}
# e_sde is a signed OU transition NLL (can go negative) -- log-scale would clip it.
CHANNEL_YSCALE = {"e_data": "log", "e_sde": "linear"}


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
# or not they're in that variant's loss -- e.g. MLP's e_sde/e_cac/e_rcac/
# e_delta_floor are real numbers in its results pickle, but they're incidental
# (never optimised), not evidence about MLP's training dynamics. Plotting them
# would be misleading, so each panel only shows the variants for which that
# channel is a live loss term. Mirrors VARIANT_COMPONENTS in
# "AP-PEN.ipynb" (cell 17); APPINN_nobal shares APPINN's
# component set (same loss_str, only the balancing flag differs -- see run_tag).
TRAINED_CHANNELS = {
    "MLP": {"e_data"},
    "APPINN_nobal": {"e_data", "e_sde"},
    "APPINN": {"e_data", "e_sde"},
    "APPINN_ARB": {"e_data", "e_sde", "e_cac", "e_rcac", "e_delta_floor"},
}
LINESTYLE = {
    "MLP": (0, ()), "APPINN_nobal": (0, (1, 1)),
    "APPINN": (0, (3, 1, 1, 1)), "APPINN_ARB": (0, (4, 1, 1, 1, 1, 1)),
}


def _training_legend(fig, tags=TAGS):
    legend_elems = [Line2D([0], [0], color=MODEL_COLOUR[t], ls=LINESTYLE[t], lw=1.6, label=DISPLAY_NAME[t])
                     for t in tags]
    fig.legend(handles=legend_elems, loc="lower center", ncol=len(tags),
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=9)


def _fig_training_channel(ch: str):
    """One channel's training-loss curve, all variants that train it
    overlaid. Shared by fig_training_data and fig_training_sde -- e_data and
    e_sde are otherwise identical in construction (smoothed, full 40k-epoch
    width), so this avoids the two drifting apart.

    Per-epoch loss is one full-batch gradient step, not a minibatch average,
    so the raw curve is dominated by point-to-point noise that buries the
    real convergence trend -- fixed by rolling-mean smoothing (_rolling_mean).
    """
    hists = {tag: load_history(tag) for tag in TAGS}
    LOG_FLOOR = 1e-9
    SMOOTH_WINDOW = 200  # ~200 effective points across 40k epochs -- enough
    # to cancel per-epoch noise without flattening the real convergence shape.

    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.42))
    for tag in TAGS:
        if ch not in TRAINED_CHANNELS[tag]:
            continue
        hist = hists[tag]
        epochs = [h[0] for h in hist]
        vals = np.asarray([h[2][ch] for h in hist])
        if CHANNEL_YSCALE[ch] == "log":
            vals = np.clip(vals, LOG_FLOOR, None)
        vals = _rolling_mean(vals, SMOOTH_WINDOW)
        ax.plot(epochs, vals, color=MODEL_COLOUR[tag], ls=LINESTYLE[tag], lw=1.4, zorder=3)
    ax.set_yscale(CHANNEL_YSCALE[ch])
    ax.set_ylabel(CHANNEL_LABEL[ch], color=INK["secondary"], fontsize=12)
    ax.set_xlabel("epoch", fontsize=11)

    _training_legend(fig, [t for t in TAGS if ch in TRAINED_CHANNELS[t]])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def fig_training_data():
    return _fig_training_channel("e_data")


def fig_training_sde():
    return _fig_training_channel("e_sde")


def fig_training_hinges():
    """The three no-arbitrage hinges (e_cac/e_rcac/e_delta_floor), AP-PEN
    (ARB) only -- the other variants never train these. Raw (not smoothed):
    the sharp initial spike-and-decay IS the story here, and smoothing a
    transient this short would blur exactly that. Companion to
    fig_training_data/fig_training_sde.
    """
    hists = {"APPINN_ARB": load_history("APPINN_ARB")}
    LOG_FLOOR = 1e-9
    ARB_CROP = 1000  # epochs -- generous crop; the hinge transient is checked
    # directly against the results pickle and this is adjusted if needed.

    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.42))
    hist = hists["APPINN_ARB"]
    epochs = np.array([h[0] for h in hist])
    crop = epochs <= ARB_CROP
    e_cac = np.clip(np.array([h[2]["e_cac"] for h in hist]), LOG_FLOOR, None)
    e_rcac = np.clip(np.array([h[2]["e_rcac"] for h in hist]), LOG_FLOOR, None)
    e_floor = np.clip(np.array([h[2]["e_delta_floor"] for h in hist]), LOG_FLOOR, None)
    ax.plot(epochs[crop], e_cac[crop], color=MODEL_COLOUR["APPINN_ARB"],
            ls="solid", lw=1.4, zorder=3)
    ax.plot(epochs[crop], e_rcac[crop], color=MODEL_COLOUR["APPINN_ARB"],
            ls=(0, (1, 1)), lw=1.4, zorder=3)
    ax.plot(epochs[crop], e_floor[crop], color=MODEL_COLOUR["APPINN_ARB"],
            ls=(0, (4, 1, 1, 1)), lw=1.4, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel(r"$e_{\mathrm{cac}},\,e_{\mathrm{rcac}},\,e_{\delta\mathrm{-floor}}$",
                  color=INK["secondary"], fontsize=11)
    ax.set_xlabel("epoch", fontsize=11)
    ax.annotate(r"$e_{\mathrm{cac}}$ solid, $e_{\mathrm{rcac}}$ dotted, "
                r"$e_{\delta\mathrm{-floor}}$ dash-dot"
                "\n(AP-PEN (ARB) only)",
                xy=(0.97, 0.75), xycoords="axes fraction", color=INK["secondary"],
                ha="right", va="center", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 1 -- parameter convergence
# ---------------------------------------------------------------------------
def fig_psi_param(p: str, psi_true: dict):
    """Learned psi[p] vs epoch, all four model variants vs truth.

    Ends with the relative error |theta_hat - theta_true|/|theta_true|
    annotated directly at each curve's own endpoint -- the reviewer reads
    calibration accuracy off the plot without cross-referencing a table.
    One standalone figure per parameter (six total, see main()) -- was one
    panel of a 2x3 grid; same per-panel content, now sized to sit two-up.
    """
    hists = {tag: load_history(tag) for tag in TAGS}
    true_v = psi_true[p]

    fig, ax = plt.subplots(1, 1, figsize=figsize("half", ratio=0.85))
    ax.axhline(true_v, color=INK["secondary"], lw=1.0, ls="--", zorder=1)

    endpoints = {}
    for tag in TAGS:
        hist = hists[tag]
        epochs = [h[0] for h in hist]
        vals = [h[3][p] for h in hist]
        ax.plot(epochs, vals, color=MODEL_COLOUR[tag], ls=LINESTYLE[tag], lw=1.4, zorder=3)
        endpoints[tag] = (epochs[-1], vals[-1])

    # Declutter the endpoint labels: several variants share most loss
    # channels and can converge to nearly the same value on several
    # parameters, so their raw endpoint y's would print two labels on
    # top of each other. Push apart in axes units (computed from the
    # CURRENT autoscaled ylim, before the labels themselves add any
    # headroom) rather than in data units, so the same minimum gap
    # looks the same across parameters with very different scales.
    ylo, yhi = ax.get_ylim()
    min_gap = 0.09 * (yhi - ylo)
    label_y, prev = {}, None
    for tag in sorted(TAGS, key=lambda t: endpoints[t][1]):
        y = endpoints[tag][1]
        if prev is not None and y - prev < min_gap:
            y = prev + min_gap
        label_y[tag] = y
        prev = y

    for tag in TAGS:
        x_end, final = endpoints[tag]
        rel_err = abs(final - true_v) / abs(true_v) * 100
        direct_label(ax, x_end, label_y[tag], f"{rel_err:.0f}%",
                     MODEL_COLOUR[tag], dx=6, dy=0, size=8)

    ax.set_ylabel(LABEL[p], color=INK["secondary"], fontsize=11)
    ax.set_xlabel("epoch", fontsize=9)
    # Headroom on the right so the endpoint % labels never clip the axes.
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.24 * (x1 - x0))

    legend_elems = [Line2D([0], [0], color=INK["secondary"], lw=1.0, ls="--", label="true")]
    legend_elems += [Line2D([0], [0], color=MODEL_COLOUR[t], ls=LINESTYLE[t], lw=1.6, label=DISPLAY_NAME[t]) for t in TAGS]
    ax.legend(handles=legend_elems, loc="best", fontsize=7, frameon=True,
              facecolor="white", framealpha=0.9, edgecolor="none")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2a -- delta recovery, single path
# ---------------------------------------------------------------------------
def load_path_repeat():
    with open(PATH_REPEAT, "rb") as f:
        return pickle.load(f)


def _path0(pr):
    """path_id 0 of the 20-path repeat-validation set -- see module
    PROVENANCE. Shared by fig_delta_recovery and fig_delta_recovery_error so
    the two never disagree about which path they're showing."""
    r0 = next(r for r in pr["results"] if r["path_id"] == 0)
    return (pr["t_train"], np.asarray(r0["delta_true"]),
            np.asarray(r0["delta_hat"]), r0["delta_rmse"])


def fig_delta_recovery(pr):
    """True vs. delta_hat overlay for a single path (path 0 of the 20-path
    repeat-validation set, see PROVENANCE).

    Sized for a half-width pairing with fig_path_repeat_recovery (pinn3c),
    which generalises this single-path story to all 20 draws -- NOT drawn at
    "full" width and shrunk by LaTeX's \\includegraphics, which is exactly
    the font-scaling mistake thesis_style.py's own docstring warns about
    (a figure sized for 6.3in and displayed at half that width takes its 9pt
    labels down to an unreadable ~4.5pt on the page). fig_delta_recovery_error
    (pinn2b) still reads the same path 0 data even though it no longer sits
    next to this figure in the thesis.
    """
    t, delta_true, delta_hat, _ = _path0(pr)

    fig, ax = plt.subplots(1, 1, figsize=figsize("half"))
    ax.plot(t, delta_true, color=INK["primary"], lw=1.4, zorder=3, label="true")
    ax.plot(t, delta_hat, color=C_MODEL, lw=1.2, ls="--", zorder=4, label=DISPLAY_NAME["APPINN"])
    ax.set_ylabel(r"$\delta_t$")
    ax.set_xlabel("$t$ (yrs)")
    ax.legend(loc="upper right", ncol=2, frameon=False)

    fig.tight_layout()
    return fig


def fig_delta_recovery_error(pr):
    """Signed error delta_hat - delta_true for the same single path shown in
    fig_delta_recovery."""
    t, delta_true, delta_hat, rmse = _path0(pr)
    err = delta_hat - delta_true

    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.28))
    ax.axhline(0.0, color=INK["muted"], lw=0.7, ls=":", zorder=1)
    ax.fill_between(t, 0.0, err, color=C_MODEL, alpha=0.25, lw=0, zorder=2)
    ax.plot(t, err, color=C_MODEL, lw=0.9, zorder=3)
    ax.axhline(rmse, color=INK["secondary"], lw=0.6, ls="--", zorder=1)
    ax.axhline(-rmse, color=INK["secondary"], lw=0.6, ls="--", zorder=1)
    ax.set_ylabel(r"$\hat\delta-\delta$")
    ax.set_xlabel("$t$ (yrs)")

    fig.tight_layout()
    return fig


def load_delta_recovery_all():
    with open(DELTA_RECOVERY_ALL, "rb") as f:
        return pickle.load(f)


# APPINN_nobal excluded, not just left out of the plot call: e_sde is stop-
# gradiented from the path net (error(), AP-PEN.ipynb), so for APPINN
# specifically, Step 1 (the only step WamOL balances) has exactly one
# component with a nonzero path-net gradient -- e_data; e_sde's is always 0.
# The balancer's own zero-handling (jnp.where(x==0, 1., sum/x)) then forces
# both weights to exactly 1.0 regardless of balancing on/off. Confirmed
# directly (grad_norm_dict = {'e_data': 6.3e-4, 'e_sde': 0.0}) and against
# results_APPINN_nobal.pkl/results_APPINN.pkl on disk, which are bit-
# identical at epoch 0, 100, and 40000 -- APPINN_nobal is PROVABLY identical
# to APPINN under this architecture, not just empirically close, so it was
# never trained into delta_recovery_all_variants.pkl in the first place
# (notebook cell between "Model Evals" and "Run"). APPINN_ARB is unaffected:
# its Step 1 has e_cac/e_rcac/e_delta_floor too, which DO have live path-net
# gradients, so its balancer does real work.
DELTA_RECOVERY_TAGS = ["MLP", "APPINN", "APPINN_ARB"]


def fig_delta_recovery_all_variants(dr):
    """True vs. delta_hat overlay for all three distinguishable variants
    (see DELTA_RECOVERY_TAGS), single path (path_id=0, full domain, no
    holdout) -- the ablation companion to fig_delta_recovery (pinn2a), which
    only shows AP-PEN alone. Answers "does e_sde / do the no-arb hinges
    actually help delta-recovery over plain data-misfit," which pinn2a
    structurally can't on its own.
    """
    t = dr["t_train"]
    delta_true = dr["delta_true"]

    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.46))
    # true UNDERNEATH (lowest zorder), variants layered on top -- same
    # convention as pinn2a (true zorder=3 < model zorder=4 there). Recovery
    # is tight enough (RMSE ~0.05-0.07) that the two nearly coincide almost
    # everywhere; with true drawn last it would sit on top and visually
    # swallow the thinner dashed variant lines across most of the domain,
    # leaving only "true" visible. Drawing true first, thicker and solid,
    # lets it peek through the dash gaps of the variant lines on top of it
    # instead of the other way around.
    ax.plot(t, delta_true, color=INK["primary"], lw=1.6, zorder=3, label="true")
    for tag in DELTA_RECOVERY_TAGS:
        v = dr["variants"][tag]
        ax.plot(t, v["delta_hat"], color=MODEL_COLOUR[tag], ls=LINESTYLE[tag], lw=1.3, zorder=4,
                label=f"{DISPLAY_NAME[tag]} (RMSE {v['rmse']:.3f})")
    ax.set_ylabel(r"$\delta_t$")
    ax.set_xlabel("$t$ (yrs)")
    # In-axes legend collides with delta_true's own peaks/troughs -- the true
    # path swings across nearly its full range at multiple points along t, so
    # no in-plot corner is actually clear (unlike pinn2a's 2-entry legend,
    # this one has 4 entries and needs the room). Placed below the axes
    # instead, one row, same convention as _training_legend elsewhere in this
    # module for multi-variant comparisons.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 -- path-repeat validation (AP-PEN, 20 independent MC noise draws)
# ---------------------------------------------------------------------------
def fig_path_repeat_errors(pr):
    """Signed-error traces across 20 independently-trained AP-PEN fits, one
    per Monte Carlo noise draw -- all 20 overlaid, faint, with the mean
    traced boldly. Companion to fig_path_repeat_rmse.

    WHY NOT A 20-PANEL GRID: the notebook's own ad hoc version of this (one
    subplot per path, true vs. recovered) makes the reader eyeball 20 near-
    identical mini time series and mentally aggregate "are these consistently
    good." That aggregation is exactly what a thesis figure should do FOR the
    reader, not ask them to redo by eye. Answers: does the recovery error
    stay small and centred on zero across every draw?
    """
    t = pr["t_train"]
    results = pr["results"]
    errs = np.stack([r["delta_hat"] - r["delta_true"] for r in results])   # (20, N)
    mean_err = errs.mean(axis=0)
    pooled_rmse = float(np.sqrt(np.mean(errs ** 2)))

    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.35))
    for e in errs:
        ax.plot(t, e, color=C_MODEL, lw=0.5, alpha=0.18, zorder=1)
    ax.axhline(0.0, color=INK["muted"], lw=0.7, ls=":", zorder=2)
    ax.axhline(pooled_rmse, color=INK["secondary"], lw=0.6, ls="--", zorder=2)
    ax.axhline(-pooled_rmse, color=INK["secondary"], lw=0.6, ls="--", zorder=2)
    ax.plot(t, mean_err, color=C_MODEL, lw=1.6, zorder=3)
    ax.annotate(f"RMSE {pooled_rmse:.3f}", xy=(0.02, 0.04), xycoords="axes fraction",
                color=INK["primary"], ha="left", va="bottom", fontsize=8)
    ax.set_xlabel("$t$ (yrs)")
    ax.set_ylabel(r"$\hat\delta_t-\delta_t$")

    fig.tight_layout()
    return fig


def fig_path_repeat_rmse(pr):
    """Per-path RMSE across the same 20 draws as fig_path_repeat_errors, as a
    strip plot (20 points is too few for a histogram to read as anything but
    noisy bars)."""
    results = pr["results"]
    rmses = np.array([r["delta_rmse"] for r in results])

    fig, ax = plt.subplots(1, 1, figsize=figsize("half", ratio=1.15))
    rng = np.random.default_rng(0)   # jitter only, not part of the model's own randomness
    jitter = rng.uniform(-0.15, 0.15, size=len(rmses))
    ax.scatter(jitter, rmses, s=16, color=C_MODEL, alpha=0.7, zorder=3, lw=0)
    ax.errorbar([0], [rmses.mean()], yerr=[rmses.std()], fmt="_", color=INK["primary"],
                markersize=22, markeredgewidth=1.8, capsize=4, elinewidth=1.4, zorder=4)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("per-path RMSE")
    ax.annotate(f"mean {rmses.mean():.3f} $\\pm$ {rmses.std():.3f}", xy=(0.30, 0.5),
                xycoords=("data", "axes fraction"), color=INK["primary"],
                ha="center", va="center", rotation=90, fontsize=8)

    fig.tight_layout()
    return fig


def fig_path_repeat_recovery(pr):
    """True vs. recovered delta, pooled across all 20 path-repeat draws, as
    a density scatter against the y=x perfect-recovery line.

    WHY THIS INSTEAD OF 20 fig_delta_recovery-STYLE OVERLAYS: each of the 20
    draws is its OWN simulated path with its own delta_true (see PROVENANCE
    -- these are 20 independent paths, not 20 noise redraws of one path), so
    the 20 true/hat time series don't share a common phase or scale to
    overlay meaningfully -- 40 lines would just be spaghetti, and a 20-panel
    small-multiples grid is exactly the "make the reader eyeball 20 near-
    identical mini plots" failure mode fig_path_repeat_errors already avoids
    (see its own docstring). Pooling every (true, hat) pair from every path
    into one true-vs-hat scatter sidesteps both: recovery quality across the
    full ensemble reads directly off how tightly the cloud hugs the diagonal,
    with hexbin density standing in for the 20,020 individual points.

    An earlier version also traced path 0 (pinn2a/b's single-path
    illustration) through the cloud as a connected line -- dropped, since a
    time-ordered path threading through a density plot reads as a scribble
    with no clear direction, adding clutter without adding to the one thing
    this figure is for: showing how tightly the ensemble hugs the diagonal.
    """
    results = pr["results"]
    true_all = np.concatenate([np.asarray(r["delta_true"]) for r in results])
    hat_all = np.concatenate([np.asarray(r["delta_hat"]) for r in results])

    lo = min(true_all.min(), hat_all.min())
    hi = max(true_all.max(), hat_all.max())
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)

    fig, ax = plt.subplots(1, 1, figsize=figsize("half", ratio=1.0))
    hb = ax.hexbin(true_all, hat_all, gridsize=45, cmap=SEQ_BLUE, mincnt=1,
                    linewidths=0.1, extent=(*lim, *lim), zorder=1)
    ax.plot(lim, lim, color=INK["muted"], lw=0.9, ls=":", zorder=2)

    corr = np.corrcoef(true_all, hat_all)[0, 1]
    ax.annotate(f"$r = {corr:.3f}$", xy=(0.05, 0.93), xycoords="axes fraction",
                color=INK["primary"], ha="left", va="top", fontsize=9)

    ax.set_xlabel(r"$\delta_t$ (true)")
    ax.set_ylabel(r"$\hat\delta_t$ (recovered)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    # set_aspect("equal") + the colorbar eating into the axes' pixel width
    # (added after aspect is set) leave x and y with different effective
    # pixel spans, so matplotlib's default per-axis locator picks a coarser
    # tick set for x than y (0, 1 vs 0.0, 0.5, 1.0) despite identical data
    # ranges -- force both to the same step so the shared scale reads as
    # shared.
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    cbar = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.03, aspect=22)
    cbar.set_label("point density", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

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
        ("pinn0a_training_data", fig_training_data, ()),
        ("pinn0b_training_sde", fig_training_sde, ()),
        ("pinn0c_training_hinges", fig_training_hinges, ()),
    ):
        fig = builder(*args)
        path = save(fig, name, directory=FIGDIR)
        plt.close(fig)
        print(f"wrote {path.relative_to(REPO)}")

    PARAM_TAG = {"alpha_Q": "alphaQ", "alpha_P": "alphaP"}  # file-safe suffixes
    for p in ORDER:
        fig = fig_psi_param(p, psi_true)
        path = save(fig, f"pinn1_{PARAM_TAG.get(p, p)}", directory=FIGDIR)
        plt.close(fig)
        print(f"wrote {path.relative_to(REPO)}")

    if PATH_REPEAT.exists():
        pr = load_path_repeat()

        for name, builder in (
            ("pinn2a_delta_recovery", fig_delta_recovery),
            ("pinn2b_delta_recovery_error", fig_delta_recovery_error),
            ("pinn3a_path_repeat_errors", fig_path_repeat_errors),
            ("pinn3b_path_repeat_rmse", fig_path_repeat_rmse),
            ("pinn3c_path_repeat_recovery", fig_path_repeat_recovery),
        ):
            fig = builder(pr)
            path = save(fig, name, directory=FIGDIR)
            plt.close(fig)
            print(f"wrote {path.relative_to(REPO)}")
    else:
        print(f"skipped pinn2a/b and pinn3a/b -- "
              f"{PATH_REPEAT.relative_to(REPO)} not found yet")

    if DELTA_RECOVERY_ALL.exists():
        dr = load_delta_recovery_all()
        fig = fig_delta_recovery_all_variants(dr)
        path = save(fig, "pinn2c_delta_recovery_all_variants", directory=FIGDIR)
        plt.close(fig)
        print(f"wrote {path.relative_to(REPO)}")
    else:
        print(f"skipped pinn2c -- {DELTA_RECOVERY_ALL.relative_to(REPO)} not found yet")


if __name__ == "__main__":
    main()
