"""
Visualisation functions for calibration results.
"""

import pickle

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from gs_wamol.physics.gibson_schwartz import call_derivatives
from gs_wamol.data.synthetic import sabr_vol_hagan


# ---------------------------------------------------------------------------
# Volatility surface
# ---------------------------------------------------------------------------

def plot_volatility_surface(model, config, save_path=None):
    Ks, Ts = np.meshgrid(np.linspace(0, 2.5, 101), np.linspace(0, 5, 101))
    x_val  = np.array([Ks.flatten(), Ts.flatten()]).T

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), subplot_kw=dict(projection="3d"))
    fig.suptitle(f"Volatility Surface ({config.loss_str})")

    vol_surface = model(x_val).reshape(101, 101)
    ax.plot_surface(Ks, Ts, vol_surface, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel("Strike"); ax.set_ylabel("Time")
    ax.view_init(30, -70)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show(); plt.close()


# ---------------------------------------------------------------------------
# No-arbitrage heatmaps
# ---------------------------------------------------------------------------

def plot_arbitrage_heatmaps(model, config, r: float, save_path=None):
    K_mesh = np.linspace(0, 2.5, 201)
    T_mesh = np.linspace(0, 5, 201)
    Ks, Ts = np.meshgrid(K_mesh, T_mesh)
    x_mesh = np.array([Ks.flatten(), Ts.flatten()]).T
    K, T   = x_mesh[:, 0], x_mesh[:, 1]

    # Pull s_0 from config or use 1.0
    s_0 = getattr(config, "s_0", 1.0)
    dK_C, d2K_C, dT_C = call_derivatives(model, x_mesh, s_0, r)

    e_arb = {
        "dK":  jnp.where(dK_C  > 0,            dK_C  ** 2,
               jnp.where(dK_C  < -jnp.exp(-r * T), dK_C ** 2, 0)),
        "d2K": jnp.where(d2K_C < 0, d2K_C ** 2, 0),
        "dT":  jnp.where(dT_C  < 0, dT_C  ** 2, 0),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f"Arbitrage Condition Heatmaps ({config.loss_str})")
    titles = {"dK": "∂C/∂K", "d2K": "∂²C/∂K²", "dT": "∂C/∂T"}

    for ax, (key, title) in zip(axes, titles.items()):
        im = ax.pcolormesh(
            Ks, Ts, e_arb[key].reshape(len(T_mesh), len(K_mesh)),
            cmap="bwr", vmin=-1e-50, vmax=1e-50, shading="auto", alpha=0.7,
        )
        ax.set_title(title)
        ax.set_xlabel("Strike"); ax.set_ylabel("Time")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show(); plt.close()


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_training_history(history_path: str, save_path=None):
    with open(history_path, "rb") as f:
        results = pickle.load(f)
    history = results["history"]
    epochs, losses, metrics = zip(*history)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(epochs, losses, "b-", label="Total Loss")

    colors = {
        "e_acc": "r", "e_pde": "g",
        "e_arb_dK": "m", "e_arb_d2K": "c", "e_arb_dT": "y",
    }
    for key in metrics[0].keys():
        vals = [m[key] for m in metrics]
        ax.plot(epochs, vals, f"{colors.get(key, 'k')}--", label=key)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_yscale("log"); ax.grid(True); ax.legend()
    plt.title("Training History")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show(); plt.close()


# ---------------------------------------------------------------------------
# SABR comparison
# ---------------------------------------------------------------------------

def compare_with_sabr(
    model, s_0: float, r: float,
    alpha: float, beta: float, rho: float, nu: float,
    save_path=None,
):
    x_mesh = np.linspace(0.0, 2.5, 101)
    t_mesh = np.linspace(0.0, 5.0, 101)
    Ks, Ts = np.meshgrid(x_mesh, t_mesh)
    x_vec  = np.array([Ks.flatten(), Ts.flatten()]).T

    sabr_vols = []
    for K, tau in x_vec:
        F = s_0 * np.exp(r * tau)
        vol = np.nan if (K == 0.0 or tau == 0.0) else sabr_vol_hagan(F, K, tau, alpha, beta, nu, rho)
        sabr_vols.append(vol)
    sabr_vols  = np.array(sabr_vols)
    model_vols = model(x_vec).flatten()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw=dict(projection="3d"))
    fig.suptitle("SABR vs Calibrated Volatility Surface")

    pairs = [
        (axes[0], sabr_vols,  "SABR"),
        (axes[1], model_vols, "Calibrated"),
        (axes[2], model_vols - sabr_vols, "Difference"),
    ]
    for ax, vals, title in pairs:
        surf = ax.plot_surface(Ks, Ts, vals.reshape(len(t_mesh), len(x_mesh)), cmap=cm.coolwarm)
        ax.set_title(title)
        ax.set_xlabel("Strike"); ax.set_ylabel("Time"); ax.set_zlabel("Vol")
        ax.view_init(30, -70)
        plt.colorbar(surf, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show(); plt.close()
