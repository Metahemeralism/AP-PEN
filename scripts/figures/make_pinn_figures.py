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
  pinn3a/b    path-repeat validation -- 20 independent Monte Carlo noise
                                       draws, each refit from scratch:
                                       signed-error traces overlaid (all 20,
                                       faint, + mean, pinn3a) and the per-path
                                       RMSE spread (pinn3b). Answers "is
                                       recovery consistent across draws,"
                                       which a 20-panel small-multiples grid
                                       (one subplot per path) leaves the
                                       reader to eyeball and aggregate by hand.

(pinn2b as a 3D ribbon variant of pinn2a, and an earlier pinn3, a collocation-
vs-data-coverage scatter, were dropped -- kept as an idea in git history if
ever wanted again, not carried forward as figures. The pinn2b name above was
reused for the delta-recovery error figure once the ribbon idea was dropped.)

Run:  python scripts/figures/make_pinn_figures.py
Reads data/input/synthetic/mc_data.pkl, data/output/mc_simulated_single_net/
results/results_{MLP,APPINN_nobal,APPINN,APPINN_ARB}.pkl, and
data/output/mc_simulated_single_net/results/path_repeat.pkl (for pinn2a/b and
pinn3a/b -- all four come from that file now, see PROVENANCE). Writes
figures/pinn/{pdf,png}.

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

from gs_wamol.physics.gibson_schwartz import GSParams
from gs_wamol.utils.thesis_style import (
    C_MODEL, INK, PALETTE,
    direct_label, figsize, save, use_thesis_style,
)

REPO = Path(__file__).resolve().parents[2]
MC_DATA = REPO / "data" / "input" / "synthetic" / "mc_data.pkl"
RESULTS_DIR = REPO / "data" / "output" / "mc_simulated_single_net" / "results"
PATH_REPEAT = REPO / "data" / "output" / "mc_simulated_single_net" / "results" / "path_repeat.pkl"
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
    """True vs. delta_hat overlay for a single path. Companion to
    fig_delta_recovery_error, which plots the signed error between them."""
    t, delta_true, delta_hat, _ = _path0(pr)

    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.40))
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
        ):
            fig = builder(pr)
            path = save(fig, name, directory=FIGDIR)
            plt.close(fig)
            print(f"wrote {path.relative_to(REPO)}")
    else:
        print(f"skipped pinn2a/b and pinn3a/b -- "
              f"{PATH_REPEAT.relative_to(REPO)} not found yet")


if __name__ == "__main__":
    main()
