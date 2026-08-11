"""Constraint activity and violation rates for the no-arbitrage hinge terms.

WHY
---
The evaluation scheme promises a "constraint activity rate" as one of its
headline metrics, but no table in the thesis reports one. That metric is the
most direct evidence available for whether the DC-PINN hinge mechanism -- the
part of this work inherited from Hoshisashi et al. and extended to
commodity-specific inequalities -- actually does anything. Without it the
hinges are only ever visible indirectly, through their cost to path recovery.

WHAT IS MEASURED
----------------
Three rates, each on the (date x maturity) grid or the date axis as
appropriate:

  observed      the model-free violation rate of the OBSERVED panel itself.
                Needs no model at all -- it only references (F_obs, S, tau, r).
                This is the baseline that determines whether a constraint can
                do any work: if the data never violates the bound, a hinge
                penalising violations has nothing to bind against.

  fitted        the violation rate of each variant's own fitted prices F_hat.
                The difference between this and `observed` is what the
                constraint bought.

  delta-floor   the fraction of dates on which the recovered path itself sits
                below delta_min, which is a statement about the latent state
                rather than about prices.

Run:  conda run -n pinns python scripts/experiments/constraint_activity.py
"""
import os, pickle

import numpy as np
import pandas as pd

REPO = "/Users/evanlynch/Developer/DC-PINNs"
U_LEVEL, DELTA_MAX, DELTA_MIN = 0.30, 0.75, -0.30


def B_coeff(tau, kappa):
    return -(1.0 - np.exp(-kappa * tau)) / kappa


def A_coeff(tau, r, kappa, alpha_Q, sigma1, sigma2, rho):
    P = sigma1 * sigma2 * rho
    s2 = sigma2 ** 2
    return ((r - alpha_Q + 0.5 * s2 / kappa ** 2 - P / kappa) * tau
            + 0.25 * s2 * (1.0 - np.exp(-2.0 * kappa * tau)) / kappa ** 3
            + (alpha_Q * kappa + P - s2 / kappa) * (1.0 - np.exp(-kappa * tau)) / kappa ** 2)


def rates(logF, logS, taus, r, delta=None, psi=None):
    """Violation rates. If psi/delta given, score the FITTED curve; else the observed."""
    if psi is not None:
        A = A_coeff(taus[None, :], r[:, None], psi["kappa"], psi["alpha_Q"],
                    psi["sigma1"], psi["sigma2"], psi["rho"])
        lf = logS[:, None] + B_coeff(taus, psi["kappa"])[None, :] * delta[:, None] + A
    else:
        lf = logF
    ceiling = logS[:, None] + (r[:, None] + U_LEVEL) * taus[None, :]
    floor = logS[:, None] + (r[:, None] - DELTA_MAX) * taus[None, :]
    return (100 * float(np.mean(lf > ceiling)),      # cac violated
            100 * float(np.mean(lf < floor)))        # rcac violated


# ---------------------------------------------------------------- real WTI
df = pd.read_csv(f"{REPO}/data/input/real/wti_analysis_ready.csv").dropna(subset=["spot", "rate"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "tau"])
df["slot"] = df.groupby("date").cumcount()
df = df[df["slot"] < 8]
df = df[df.groupby("date")["slot"].transform("count") == 8]
taus_r = df.pivot(index="date", columns="slot", values="tau").sort_index().to_numpy().mean(axis=0)
logF_r = np.log(df.pivot(index="date", columns="slot", values="settle").sort_index().to_numpy())
S_r = df.groupby("date")["spot"].first().sort_index().to_numpy()
r_r = df.groupby("date")["rate"].first().sort_index().to_numpy()
dates_r = df.pivot(index="date", columns="slot", values="tau").sort_index().index.to_numpy()

print("=" * 92)
print("CONSTRAINT ACTIVITY -- REAL WTI (2,770 dates x 8 maturities = 22,160 points)")
print("=" * 92)
c, rc = rates(logF_r, np.log(S_r), taus_r, r_r)
print(f"{'series':>34} | {'cac viol %':>11} | {'rcac viol %':>12} | {'delta<min (% dates)':>20}")
print("-" * 92)
print(f"{'OBSERVED PANEL (model-free)':>34} | {c:11.2f} | {rc:12.2f} | {'--':>20}")

kf = pickle.load(open(f"{REPO}/data/output/real_data/results/kappa_floor_comparison.pkl", "rb"))
for (kmin, tag), v in sorted(kf["results"].items()):
    if kmin != 0.1:
        continue
    d, psi = v["delta_hat"], v["psi"]
    c, rc = rates(logF_r, np.log(S_r), taus_r, r_r, delta=d, psi=psi)
    below = 100 * float(np.mean(d < DELTA_MIN - 1e-9))
    name = {"MLP": "MLP (no hinges)", "APPINN": "AP-PEN (no hinges)",
            "APPINN_ARB": "AP-PEN (ARB) -- hinges ON"}[tag]
    print(f"{name:>34} | {c:11.2f} | {rc:12.2f} | {below:20.2f}")

# where does the observed panel violate?
ceil_r = np.log(S_r)[:, None] + (r_r[:, None] + U_LEVEL) * taus_r[None, :]
viol_dates = pd.DatetimeIndex(dates_r)[(logF_r > ceil_r).any(axis=1)]
if len(viol_dates):
    print(f"\n  observed cac violations fall on {len(viol_dates)} dates, "
          f"{viol_dates.year.value_counts().sort_index().to_dict()}")

# ---------------------------------------------------------------- synthetic
mc = pickle.load(open(f"{REPO}/data/input/synthetic/mc_data.pkl", "rb"))
taus_m = np.asarray(mc["taus"])
logF_m = np.asarray(mc["log_F_obs"][0])
S_m = np.asarray(mc["S"][0])
r_m = np.full(len(S_m), 0.05)
dtrue = np.asarray(mc["delta_true"][0])

print("\n" + "=" * 92)
print("CONSTRAINT ACTIVITY -- SYNTHETIC (1,001 dates x 12 maturities = 12,012 points)")
print("=" * 92)
c, rc = rates(logF_m, np.log(S_m), taus_m, r_m)
print(f"{'series':>34} | {'cac viol %':>11} | {'rcac viol %':>12} | {'delta<min (% dates)':>20}")
print("-" * 92)
print(f"{'OBSERVED PANEL (model-free)':>34} | {c:11.2f} | {rc:12.2f} | "
      f"{100*float(np.mean(dtrue < DELTA_MIN)):20.2f}")

dv = pickle.load(open(f"{REPO}/data/output/mc_simulated_single_net/results/"
                      "delta_recovery_all_variants.pkl", "rb"))
for tag, v in dv["variants"].items():
    d, psi = v["delta_hat"], v["psi_hat"]
    c, rc = rates(logF_m, np.log(S_m), taus_m, r_m, delta=d, psi=psi)
    below = 100 * float(np.mean(d < DELTA_MIN - 1e-9))
    name = {"MLP": "MLP (no hinges)", "APPINN": "AP-PEN (no hinges)",
            "APPINN_ARB": "AP-PEN (ARB) -- hinges ON"}[tag]
    print(f"{name:>34} | {c:11.2f} | {rc:12.2f} | {below:20.2f}")

print(f"\n  TRUE delta on the synthetic panel: min {dtrue.min():+.4f}, "
      f"{100*float(np.mean(dtrue < DELTA_MIN)):.2f}% of dates below delta_min={DELTA_MIN}")
print("  -> the floor hinge penalises states the data-generating process genuinely visits.")
