"""
PINN-performance figures for the Gibson-Schwartz convenience-yield inversion.

Three figures, none of which a results table can substitute for:

  pinn1  parameter convergence   -- learned psi vs epoch, MLP/PINN/DCPINN vs
                                    truth, with the final relative error % read
                                    directly off each curve's endpoint
  pinn2  delta recovery, in 3D       -- true delta_t and DCPINN's delta_hat(t)
                                    as two curves joined by a ruled surface
                                    coloured by signed error; the surface's
                                    twist is the recovery-quality signal
  pinn3  collocation vs data coverage -- the ODE residual is enforced on 4096
                                    continuous tau points; the price data only
                                    ever anchors 12 discrete contract maturities

Run:  python scripts/make_pinn_figures.py
Reads data/input/synthetic/mc_data.pkl, data/output/results/results_{MLP,
PINN,DCPINN}.pkl, and the orbax checkpoint at
data/output/checkpoints/DCPINN. Writes figures/pinn/{pdf,png}.

PROVENANCE
----------
pinn1 reads the psi values the training loop itself logged every epoch
(notebooks/PINN_implementation.ipynb, cell 18) -- nothing is recomputed.

pinn2 plots delta_true straight from mc_data.pkl (the ground truth the notebook
never shows the model) against delta_hat(t), a bare forward pass of the trained
DCPINN checkpoint; no weights are touched or retrained here.

pinn3's collocation cloud is regenerated with the identical
jax.random.PRNGKey(0) the notebook used for tau_mesh, so it is bit-identical
to what the model actually trained against, not a fresh random draw.

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
    C_DELTA, C_MODEL, DIV_BLUE_RED, INK, PALETTE,
    direct_label, figsize, save, use_thesis_style,
)

REPO = Path(__file__).resolve().parents[1]
MC_DATA = REPO / "data" / "input" / "synthetic" / "mc_data.pkl"
RESULTS_DIR = REPO / "data" / "output" / "results"
CKPT_DCPINN = REPO / "data" / "output" / "checkpoints" / "DCPINN"
FIGDIR = REPO / "figures" / "pinn"

ORDER = ["kappa", "sigma1", "sigma2", "rho", "alpha_Q", "alpha_P"]
LABEL = {
    "kappa": r"$\kappa$", "sigma1": r"$\sigma_1$", "sigma2": r"$\sigma_2$",
    "rho": r"$\rho$", "alpha_Q": r"$\alpha^{\mathbb{Q}}$", "alpha_P": r"$\alpha^{\mathbb{P}}$",
}
# One colour per model variant, fixed for both figures in this module.
# DCPINN gets C_MODEL (violet) -- the same "fitted/recovered quantity" hue used
# document-wide -- because it is the thesis's proposed model, not a baseline.
MODEL_COLOUR = {"MLP": PALETTE["aqua"], "PINN": PALETTE["orange"], "DCPINN": C_MODEL}


def load_mc():
    with open(MC_DATA, "rb") as f:
        d = pickle.load(f)
    return {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Figure 1 -- parameter convergence
# ---------------------------------------------------------------------------
def load_history(tag: str):
    with open(RESULTS_DIR / f"results_{tag}.pkl", "rb") as f:
        return pickle.load(f)["history"]


def fig_psi_convergence(psi_true):
    """Learned psi vs epoch, one panel per parameter, all three model variants.

    Each panel ends with the relative error |theta_hat - theta_true|/|theta_true|
    annotated directly at the curve's own endpoint -- the reviewer reads
    calibration accuracy off the plot without cross-referencing a table.
    """
    tags = ["MLP", "PINN", "DCPINN"]
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

        ax.set_title(LABEL[p], color=INK["secondary"], fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        # Headroom on the right so the endpoint % labels never clip the axes.
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0, x1 + 0.22 * (x1 - x0))

    legend_elems = [Line2D([0], [0], color=INK["secondary"], lw=1.0, ls="--", label="true")]
    legend_elems += [Line2D([0], [0], color=MODEL_COLOUR[t], lw=1.6, label=t) for t in tags]
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


def restore_dcpinn(t_scale: float, tau_scale: float):
    """Restore the trained DCPINN checkpoint and return (delta_hat_fn, A_hat_fn, gsp).

    Reconstructs psi (learned, unconstrained) with the same softplus/tanh
    constraints the notebook trains under (build_nets, cell 8) -- these are
    the ONLY place the true parameter values could leak into the reconstructed
    functions, and they do not: psi comes entirely from the checkpoint.
    """
    ckpt = orbax.checkpoint.PyTreeCheckpointer().restore(str(CKPT_DCPINN))
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

    delta_hat_fn, _, _ = restore_dcpinn(t_scale=float(t[-1]), tau_scale=float(taus.max()))
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
    """Plain 2D companion to fig_delta_recovery_3d: true vs. delta_hat overlay,
    plus the signed error underneath.

    WHY THIS EXISTS ALONGSIDE THE 3D VERSION: a static 3D render cannot be
    rotated on a printed page, so depth is ambiguous and the exact gap between
    two close curves is hard to read off precisely (see fig_delta_recovery_3d's
    own docstring for why the ribbon was tried anyway). When the two curves are
    this close -- corr > 0.97 here -- the strongest way to show it is the most
    boring one: overlay them directly and let them visibly coincide. The error
    panel underneath then answers "close by how much, and where does it stop
    being close" without asking the reader to infer a distance from a 3D twist.
    """
    t, delta_true, delta_hat, m = _delta_recovery(d)
    err = delta_hat - delta_true

    fig, (ax_top, ax_err) = plt.subplots(
        2, 1, figsize=figsize("full", ratio=0.55), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08},
    )

    ax_top.plot(t, delta_true, color=INK["primary"], lw=1.4, zorder=3, label="true")
    ax_top.plot(t, delta_hat, color=C_MODEL, lw=1.2, ls="--", zorder=4, label="DCPINN")
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


def fig_delta_recovery_3d(d):
    """True vs. DCPINN-recovered convenience yield, as a 3D 'ribbon' surface.

    WHY A RIBBON, NOT A 2D OVERLAY: when recovery is good the true and recovered
    curves sit almost on top of each other, and a flat line-plot overlay makes a
    good fit and a bad one look nearly the same -- the eye has to hunt for a thin
    vertical gap between two overlapping lines. Here the gap IS the surface: a
    ruled surface connects delta_true(t) (front edge, y=0) to delta_hat(t) (back
    edge, y=1) at every t, so recovery quality becomes the surface's own shape --
    flat and untwisted where the fit is good, buckled where it is not -- and the
    face colour repeats the same signal as signed error, so the two encodings
    (geometry and colour) confirm each other rather than requiring a legend
    lookup to interpret either one alone.

    SHAPES: t (N+1,) downsampled by `stride`; delta_true, delta_hat same shape.
    The ribbon is a (2, n) grid -> (1, n-1) quad faces, one signed-error colour
    per face.
    """
    t, delta_true, delta_hat, _ = _delta_recovery(d)

    # Downsample for the mesh: 1001 raw points make an overly fine ribbon whose
    # facets are individually invisible; a coarser mesh renders the same twist
    # with visible facets and a much smaller PDF.
    stride = 4
    ts, dt_true, dt_hat = t[::stride], delta_true[::stride], delta_hat[::stride]

    X = np.vstack([ts, ts])
    Y = np.vstack([np.zeros_like(ts), np.ones_like(ts)])
    Z = np.vstack([dt_true, dt_hat])

    err = dt_hat - dt_true
    face_err = 0.5 * (err[:-1] + err[1:])       # one value per quad face
    vmax_err = np.abs(err).max()
    norm = plt.Normalize(-vmax_err, vmax_err)
    facecolors = DIV_BLUE_RED(norm(face_err))[None, :, :]

    fig = plt.figure(figsize=figsize("full", ratio=0.75))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1,
                     linewidth=0, antialiased=True, shade=False)
    # Bold outlines on top of the ribbon: shading alone makes the exact curve
    # position ambiguous, especially near the twisted regions.
    ax.plot(ts, np.zeros_like(ts), dt_true, color=INK["primary"], lw=1.8, zorder=10)
    ax.plot(ts, np.ones_like(ts), dt_hat, color=C_MODEL, lw=1.8, zorder=10)

    ax.set_xlabel("$t$ (yrs)", labelpad=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["true", "DCPINN"])
    ax.set_zlabel(r"$\delta_t$", labelpad=18)
    ax.view_init(elev=20, azim=-58)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIV_BLUE_RED)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.08, fraction=0.035, shrink=0.6)
    cbar.set_label(r"signed error $\hat\delta-\delta$", fontsize=8)

    fig.subplots_adjust(top=0.95, bottom=0.05)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 -- collocation (physics loss) vs. quoted maturities (data loss)
# ---------------------------------------------------------------------------
def fig_collocation_map(d):
    """Where the ODE residual is enforced vs. where price data actually exists.

    The ODE residual for A(tau) (error(), e_ode) is a pure function of tau --
    it carries no time coordinate -- so the honest domain to compare against
    the 12 quoted contracts is tau alone, not a (t, tau) plane. Vertical
    position is jitter for visual separation only and carries no data; the
    axis is deliberately left unlabelled to say so rather than implying a
    second variable that does not exist.
    """
    taus = d["taus"]
    N_COLLOCATION = 4096
    key = random.PRNGKey(0)  # identical seed to the notebook's tau_mesh (cell 13)
    tau_mesh = np.asarray(random.uniform(key, (N_COLLOCATION,), minval=0.0, maxval=float(taus.max())))

    rng = np.random.default_rng(0)  # jitter only, not part of the model's own randomness
    jitter = rng.uniform(-1, 1, size=N_COLLOCATION)

    fig, ax = plt.subplots(1, 1, figsize=figsize("full", ratio=0.30))
    ax.scatter(tau_mesh, jitter, s=3, color=INK["faint"], alpha=0.5, lw=0, zorder=1)
    for tau_i in taus:
        ax.axvline(tau_i, color=C_DELTA, lw=1.4, zorder=2)

    ax.set_xlabel(r"Maturity $\tau$ (years)")
    ax.set_ylim(-1.3, 1.3)
    ax.set_yticks([])
    ax.text(0.01, -1.18, "(vertical position: jitter for visibility only, not a variable)",
            fontsize=7, color=INK["muted"], transform=ax.get_yaxis_transform())

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK["faint"],
               markersize=5, label=f"ODE collocation ($N={N_COLLOCATION}$)"),
        Line2D([0], [0], color=C_DELTA, lw=1.4, label=f"quoted contracts ($K={len(taus)}$)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", frameon=False, fontsize=8)
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
        ("pinn1_psi_convergence", fig_psi_convergence, (psi_true,)),
        ("pinn2a_delta_recovery_2d", fig_delta_recovery_2d, (d,)),
        ("pinn2b_delta_recovery_3d", fig_delta_recovery_3d, (d,)),
        ("pinn3_collocation_map", fig_collocation_map, (d,)),
    ):
        fig = builder(*args)
        path = save(fig, name, directory=FIGDIR)
        plt.close(fig)
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
