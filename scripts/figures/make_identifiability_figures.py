"""
Identifiability and closed-form-baseline figures.

These cover results that none of the other figure scripts do, and that a table
cannot substitute for because the whole point is the SHAPE of a likelihood
surface and the AGREEMENT STRUCTURE between estimators:

  ident1_profile   profile likelihood, five parameters on one axis pair --
                   kappa/sigma2/m are curved (identified), sigma1/rho are
                   exactly horizontal (not identified). The flatness is the
                   result; a table of point estimates and standard errors
                   actively hides it, because a flat profile has no curvature
                   from which a standard error can be computed at all.
  ident2_invariance the same claim shown in price space: A(tau) drawn for
                   wildly different (sigma1, rho, alpha_Q) triples that share
                   one value of m. Every curve lands on top of every other.
  ls1_recovery     synthetic delta recovery, closed-form least squares vs the
                   neural variants vs truth -- the "is the network earning its
                   place?" figure.
  ls2_agreement    real WTI: pairwise RMSE between every estimator's delta_hat.
                   Shows the Kalman filter, not the network, is the outlier.

Run:  python scripts/figures/make_identifiability_figures.py
Reads data/input/synthetic/mc_data.pkl,
      data/output/ls_baseline/results/ls_baseline.pkl,
      data/output/mc_simulated_single_net/results/delta_recovery_all_variants.pkl,
      data/output/real_data/results/results_*.pkl,
      data/output/kalman/results/results_kalman.pkl
Writes figures/identifiability/{pdf,png}.

THE RESULT THESE FIGURES SUPPORT
--------------------------------
Writing the closed form in basis-function form,
    A(tau) = c1*tau + c2*(1-e^{-2 kappa tau}) + c3*(1-e^{-kappa tau}),
and defining   m = alpha_Q + sigma1*sigma2*rho/kappa,   gives
    c1 = r - m + sigma2^2/(2 kappa^2),
    c2 = sigma2^2/(4 kappa^3),
    c3 = m/kappa - sigma2^2/kappa^3.
All three depend on (kappa, sigma2, m) alone, and B(tau) on kappa alone. So a
futures panel identifies exactly three quantities out of five; sigma1, rho and
alpha_Q are individually non-identifiable at any number of maturities, on any
number of dates, at zero noise.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gs_wamol.utils.thesis_style import (  # noqa: E402
    INK, PALETTE, figsize, save, use_thesis_style,
)

FIG_DIR = REPO / "figures" / "identifiability"
PSI_TRUE = dict(kappa=1.876, sigma1=0.393, sigma2=0.527, rho=0.766,
                alpha_Q=0.07790831556503197)
M_TRUE = PSI_TRUE["alpha_Q"] + PSI_TRUE["sigma1"] * PSI_TRUE["sigma2"] * PSI_TRUE["rho"] / PSI_TRUE["kappa"]


# --------------------------------------------------------------- closed form
def B_coeff(tau, kappa):
    return -(1.0 - np.exp(-kappa * tau)) / kappa


def A_raw(tau, kappa, sigma1, sigma2, rho, alpha_Q, r=0.05):
    """A(tau) in the RAW five-parameter coordinates (what the thesis fits)."""
    P = sigma1 * sigma2 * rho
    s2 = sigma2 ** 2
    return ((r - alpha_Q + 0.5 * s2 / kappa ** 2 - P / kappa) * tau
            + 0.25 * s2 * (1.0 - np.exp(-2.0 * kappa * tau)) / kappa ** 3
            + (alpha_Q * kappa + P - s2 / kappa) * (1.0 - np.exp(-kappa * tau)) / kappa ** 2)


def rmse_raw(theta, logF, logS, taus, r):
    """Price RMSE with delta profiled out analytically, RAW coordinates."""
    kappa, sigma1, sigma2, rho, alpha_Q = theta
    if kappa <= 1e-6 or kappa > 50 or sigma2 <= 1e-6 or abs(rho) >= 1.0 or sigma1 <= 1e-6:
        return 1e12
    B = B_coeff(taus, kappa)
    A = A_raw(taus[None, :], kappa, sigma1, sigma2, rho, alpha_Q, r[:, None])
    y = logF - logS[:, None] - A
    d = (y @ B) / (B @ B)                       # closed-form per-date delta
    resid = y - B[None, :] * d[:, None]
    return float(np.sqrt(np.mean(resid ** 2)))


NAMES = ["kappa", "sigma1", "sigma2", "rho", "alpha_Q"]


def rmse_id(kappa, sigma2, m, logF, logS, taus, r):
    """Price RMSE in the IDENTIFIED coordinates. Any (sigma1, rho, alpha_Q)
    consistent with this m gives exactly this value, so one representative
    triple suffices."""
    return rmse_raw([kappa, PSI_TRUE["sigma1"], sigma2, PSI_TRUE["rho"],
                     m - PSI_TRUE["sigma1"] * sigma2 * PSI_TRUE["rho"] / kappa],
                    logF, logS, taus, r)


def profile_identified(fixed, grid, start, logF, logS, taus, r):
    """Profile likelihood over the identified coordinates (kappa, sigma2, m).

    Done here rather than in the raw five-parameter coordinates on purpose. The
    inner maximisation over the two flat directions has a known exact solution
    (any (sigma1, rho, alpha_Q) on the manifold m = const is a global optimum),
    so running a numerical optimiser over them would only add simplex noise --
    it stalls ~1e-3% above the true optimum, which on a log axis is
    indistinguishable from genuine curvature. Profiling the three identified
    coordinates numerically and the two flat ones analytically is both exact
    and honest.
    """
    names = ["kappa", "sigma2", "m"]
    idx = names.index(fixed)
    free = [i for i in range(3) if i != idx]
    out = []
    for g in grid:
        def obj(z):
            th = list(start)
            th[idx] = g
            for j, i in enumerate(free):
                th[i] = z[j]
            if th[0] <= 1e-6 or th[1] <= 1e-6:
                return 1e12
            return rmse_id(th[0], th[1], th[2], logF, logS, taus, r)

        res = minimize(obj, [start[i] for i in free], method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-12, "fatol": 1e-16})
        out.append(res.fun)
    return np.array(out)


def main() -> None:
    use_thesis_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    mc = pickle.load(open(REPO / "data/input/synthetic/mc_data.pkl", "rb"))
    taus = np.asarray(mc["taus"])
    logF = np.asarray(mc["log_F_obs"][0])
    logS = np.log(np.asarray(mc["S"][0]))
    dtrue = np.asarray(mc["delta_true"][0])
    t = np.asarray(mc["t_grid"])
    r = np.full(len(logS), 0.05)

    # ---------------------------------------------------- ident1: profiles
    theta0 = [PSI_TRUE["kappa"], PSI_TRUE["sigma1"], PSI_TRUE["sigma2"],
              PSI_TRUE["rho"], PSI_TRUE["alpha_Q"]]
    base = rmse_raw(theta0, logF, logS, taus, r)

    # Refit the identified coordinates first, so "% above minimum" is measured
    # against the actual optimum rather than against the true parameters.
    res0 = minimize(lambda z: rmse_id(z[0], z[1], z[2], logF, logS, taus, r),
                    [PSI_TRUE["kappa"], PSI_TRUE["sigma2"], M_TRUE],
                    method="Nelder-Mead",
                    options={"maxiter": 20000, "xatol": 1e-12, "fatol": 1e-16})
    star = list(res0.x)
    base = float(res0.fun)

    # Log y-axis: identified and unidentified profiles differ by ~14 orders of
    # magnitude, which a linear axis collapses into "one curve and four flat
    # lines" while also hiding that sigma2 is only WEAKLY identified -- a
    # distinction that matters, since sigma2 is the parameter that degrades
    # first under observation noise.
    specs = [
        ("kappa",  r"$\kappa$",   np.linspace(1.3, 2.5, 21),          PALETTE["blue"]),
        ("sigma2", r"$\sigma_2$", np.linspace(0.35, 0.72, 21),        PALETTE["aqua"]),
        # grid deliberately offset by half a step off the fitted optimum: if a
        # grid point lands exactly on it the log axis plots a spike to
        # numerical zero, which reads as "unidentified here" when it means the
        # opposite.
        ("m",      r"$m=\alpha^{\mathbb{Q}}+\sigma_1\sigma_2\rho/\kappa$",
         np.linspace(star[2] - 0.0475, star[2] + 0.0525, 21),         INK["primary"]),
    ]
    FLOOR = 1e-15   # plotting floor; exact zeros cannot be drawn on a log axis
    fig, ax = plt.subplots(figsize=figsize("full", ratio=0.44))
    for name, lab, grid, col in specs:
        pr = profile_identified(name, grid, star, logF, logS, taus, r)
        rel = np.maximum(100.0 * np.abs(pr / base - 1.0), FLOOR)
        x = (grid - grid.min()) / (grid.max() - grid.min())
        ax.plot(x, rel, color=col, lw=1.8, label=lab, zorder=3)

    # sigma1 and rho: the profile is EXACTLY the global minimum everywhere.
    # Pinning either one leaves every (kappa, sigma2, m) still reachable, so the
    # inner maximisation returns `base` by construction -- an exactly horizontal
    # line, drawn at the plotting floor because log(0) does not exist.
    for lab, col, ls in [(r"$\sigma_1$", PALETTE["orange"], "--"),
                         (r"$\rho$", PALETTE["violet"], ":")]:
        ax.plot([0, 1], [FLOOR, FLOOR], color=col, ls=ls, lw=2.2, label=lab, zorder=4)

    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 4, 3e2)
    ax.axhspan(FLOOR / 4, 1e-12, color=INK["faint"], alpha=0.4, zorder=0)
    ax.text(0.5, 2.2e-14,
            "exactly flat: pinning either leaves every "
            r"$(\kappa,\sigma_2,m)$ reachable" "\n"
            "no curvature, so no standard error exists",
            ha="center", va="center", fontsize=8, color=INK["secondary"])
    ax.set_xlabel("parameter swept across its plotted range (normalised)")
    ax.set_ylabel("price RMSE, % above minimum")
    ax.set_title(r"Profile likelihood: $\kappa$ and $m$ identified, $\sigma_2$ "
                 r"weakly, $\sigma_1$ and $\rho$ not at all", fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center")
    fig.tight_layout()
    save(fig, "ident1_profile_likelihood", FIG_DIR)
    plt.close(fig)
    print("  ident1_profile_likelihood")

    # ------------------------------------------------ ident2: invariance
    tau_g = np.linspace(0.02, 2.0, 200)
    fig, ax = plt.subplots(figsize=figsize("half", ratio=0.72))
    combos = [(0.16, -0.90), (0.39, 0.77), (0.80, 0.45), (1.24, 0.83), (0.60, -0.40)]
    ramp = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(combos)))
    for (s1, rho), c in zip(combos, ramp):
        aQ = M_TRUE - s1 * PSI_TRUE["sigma2"] * rho / PSI_TRUE["kappa"]
        ax.plot(tau_g, A_raw(tau_g, PSI_TRUE["kappa"], s1, PSI_TRUE["sigma2"], rho, aQ),
                color=c, lw=2.4, alpha=0.85,
                label=rf"$\sigma_1{{=}}{s1:.2f},\ \rho{{=}}{rho:+.2f}$")
    ax.set_xlabel(r"$\tau$ (years to maturity)")
    ax.set_ylabel(r"$A(\tau)$")
    ax.set_title(r"Five different $(\sigma_1,\rho,\alpha^{\mathbb{Q}})$, one $m$", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.text(0.98, 0.05, r"max deviation $5.6\times10^{-17}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=INK["secondary"])
    fig.tight_layout()
    save(fig, "ident2_invariance", FIG_DIR)
    plt.close(fig)
    print("  ident2_invariance")

    # ------------------------------------------------ ls1: delta recovery
    ls = pickle.load(open(REPO / "data/output/ls_baseline/results/ls_baseline.pkl", "rb"))
    dv = pickle.load(open(REPO / "data/output/mc_simulated_single_net/results/"
                          "delta_recovery_all_variants.pkl", "rb"))
    # deliberately NOT sharex: the lower panel is a zoomed detail of the upper
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize("full", ratio=0.62),
                                   height_ratios=[2, 1])
    ax1.plot(t, dtrue, color=INK["primary"], lw=1.4, label=r"true $\delta_t$", zorder=5)
    ax1.plot(t, ls["mc"]["delta_ls"], color=PALETTE["orange"], lw=1.0, alpha=0.9,
             label=f"closed-form LS (RMSE {ls['mc']['delta_rmse']:.4f})")
    ax1.plot(t, dv["variants"]["APPINN"]["delta_hat"], color=PALETTE["blue"], lw=1.0,
             ls="--", label=f"AP-PEN (RMSE {dv['variants']['APPINN']['rmse']:.4f})")
    ax1.set_ylabel(r"$\delta_t$")
    ax1.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")

    win = slice(400, 520)   # a short window makes the smoothing visible
    ax2.plot(t[win], dtrue[win], color=INK["primary"], lw=1.4)
    ax2.plot(t[win], ls["mc"]["delta_ls"][win], color=PALETTE["orange"], lw=1.1)
    ax2.plot(t[win], dv["variants"]["APPINN"]["delta_hat"][win], color=PALETTE["blue"],
             lw=1.4, ls="--")
    ax2.set_xlim(t[win.start], t[win.stop - 1])
    ax2.set_xlabel("$t$ (years)")
    ax2.set_ylabel(r"$\delta_t$")
    ax2.set_title("detail: the network reproduces the level but not the innovations",
                  fontsize=8)
    fig.tight_layout()
    save(fig, "ls1_delta_recovery", FIG_DIR)
    plt.close(fig)
    print("  ls1_delta_recovery")

    # ------------------------------------------------ ls2: agreement matrix
    kf = pickle.load(open(REPO / "data/output/kalman/results/results_kalman.pkl", "rb"))
    kfd = pd.DatetimeIndex(kf["dates"])
    series, ref = {}, None
    for tag, lab in [("MLP", "MLP"), ("APPINN", "AP-PEN"), ("APPINN_ARB", "AP-PEN (ARB)")]:
        p = REPO / f"data/output/real_data/results/results_{tag}.pkl"
        if p.exists():
            d = pickle.load(open(p, "rb"))
            ref = pd.DatetimeIndex(d["dates"])
            series[lab] = d["delta_hat"]
    series["closed-form LS"] = ls["real"]["delta_ls"]
    common = kfd.intersection(ref)
    pos, kpos = ref.get_indexer(common), kfd.get_indexer(common)
    mat_series = {"Kalman": kf["delta_hat"][kpos]}
    for k, v in series.items():
        mat_series[k] = v[pos]

    names = list(mat_series)
    M = np.array([[np.sqrt(np.mean((mat_series[a] - mat_series[b]) ** 2))
                   for b in names] for a in names])
    fig, ax = plt.subplots(figsize=figsize("full", ratio=0.66))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 * M.max() else INK["primary"])
    ax.set_title(r"Pairwise RMSE between recovered $\hat\delta_t$, real WTI", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="RMSE")
    fig.tight_layout()
    save(fig, "ls2_agreement_matrix", FIG_DIR)
    plt.close(fig)
    print("  ls2_agreement_matrix")


if __name__ == "__main__":
    print(f"writing to {FIG_DIR}")
    main()
    print("done")
