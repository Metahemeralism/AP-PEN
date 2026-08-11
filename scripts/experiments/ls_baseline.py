"""Per-date analytic least-squares inversion baseline.

Why this exists: log F(t,tau) = log S_t + B(tau)delta_t + A(tau) is LINEAR in
delta_t. So for any fixed psi the optimal delta_t at each date has a closed form
-- one scalar regression per date, no training, no network:

    delta_hat_t = sum_k B_k (log F_tk - log S_t - A_tk) / sum_k B_k^2

Profiling psi out on top of that gives the exact MLE of the whole model under
Gaussian log-price noise. This is the baseline the thesis needs: if the neural
parameterisation does not beat it, the network is unjustified.

Parameterised directly in the IDENTIFIED coordinates (kappa, sigma2, m) with
m = alpha_Q + sigma1*sigma2*rho/kappa -- see docs/results_and_discussion.md S0.
"""
import os, pickle, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO = "/Users/evanlynch/Developer/DC-PINNs"
OUT = f"{REPO}/data/output/ls_baseline"
os.makedirs(f"{OUT}/results", exist_ok=True)


# ---------------------------------------------------------------- closed form
def B_coeff(tau, kappa):
    return -(1.0 - np.exp(-kappa * tau)) / kappa


def A_coeff(tau, r, kappa, sigma2, m):
    """A(tau) in identified coordinates. tau: (K,), r: scalar or (n,1)."""
    s2sq = sigma2 ** 2
    lin = (r - m + 0.5 * s2sq / kappa ** 2) * tau
    two = 0.25 * s2sq * (1.0 - np.exp(-2.0 * kappa * tau)) / kappa ** 3
    one = (m / kappa - s2sq / kappa ** 3) * (1.0 - np.exp(-kappa * tau))
    return lin + two + one


def solve_delta(logF, logS, taus, r, kappa, sigma2, m):
    """Per-date closed-form LS solve for delta_t. logF (n,K), logS (n,), r (n,)."""
    B = B_coeff(taus, kappa)                            # (K,)
    A = A_coeff(taus[None, :], r[:, None], kappa, sigma2, m)   # (n,K)
    y = logF - logS[:, None] - A                        # (n,K)
    return (y @ B) / (B @ B)                            # (n,)


def sse(theta_raw, logF, logS, taus, r):
    """Concentrated objective: delta profiled out analytically."""
    kappa = np.exp(theta_raw[0])          # kappa > 0
    sigma2 = np.exp(theta_raw[1])         # sigma2 > 0
    m = theta_raw[2]
    if not np.isfinite(kappa) or kappa < 1e-6 or kappa > 50:
        return 1e12
    d = solve_delta(logF, logS, taus, r, kappa, sigma2, m)
    B = B_coeff(taus, kappa)
    A = A_coeff(taus[None, :], r[:, None], kappa, sigma2, m)
    resid = logF - (logS[:, None] + B[None, :] * d[:, None] + A)
    return float(np.mean(resid ** 2))


def fit(logF, logS, taus, r, x0=(1.0, 0.4, 0.1), n_starts=12, seed=0):
    """Multi-start profiled MLE over (kappa, sigma2, m)."""
    rng = np.random.default_rng(seed)
    best, best_v = None, np.inf
    starts = [np.array([np.log(x0[0]), np.log(x0[1]), x0[2]])]
    for _ in range(n_starts - 1):
        starts.append(np.array([np.log(rng.uniform(0.05, 4.0)),
                                np.log(rng.uniform(0.05, 1.5)),
                                rng.uniform(-0.4, 0.6)]))
    for s in starts:
        try:
            res = minimize(sse, s, args=(logF, logS, taus, r), method="Nelder-Mead",
                           options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-14})
            if res.fun < best_v:
                best_v, best = res.fun, res.x
        except Exception:
            continue
    return dict(kappa=float(np.exp(best[0])), sigma2=float(np.exp(best[1])),
                m=float(best[2])), float(np.sqrt(best_v))


def profile(logF, logS, taus, r, fixed, grid, theta_hat):
    """Profile likelihood: fix one identified param, re-optimise the other two."""
    idx = {"kappa": 0, "sigma2": 1, "m": 2}[fixed]
    out = []
    base = np.array([np.log(theta_hat["kappa"]), np.log(theta_hat["sigma2"]), theta_hat["m"]])
    for g in grid:
        gv = np.log(g) if fixed in ("kappa", "sigma2") else g
        free = [i for i in range(3) if i != idx]

        def obj(z):
            th = base.copy(); th[idx] = gv
            th[free[0]], th[free[1]] = z
            return sse(th, logF, logS, taus, r)

        res = minimize(obj, base[free], method="Nelder-Mead",
                       options={"maxiter": 8000, "xatol": 1e-9, "fatol": 1e-13})
        out.append(np.sqrt(res.fun))
    return np.array(out)


# ---------------------------------------------------------------- experiments
results = {}

print("=" * 84)
print("A. SYNTHETIC (MC path_id=0) -- scored against known delta_true")
print("=" * 84)
mc = pickle.load(open(f"{REPO}/data/input/synthetic/mc_data.pkl", "rb"))
t_mc = np.asarray(mc["t_grid"]); taus_mc = np.asarray(mc["taus"])
S_mc = np.asarray(mc["S"][0]); logF_mc = np.asarray(mc["log_F_obs"][0])
dtrue = np.asarray(mc["delta_true"][0]); r_mc = np.full(len(t_mc), 0.05)

th, rmse_price = fit(logF_mc, np.log(S_mc), taus_mc, r_mc, x0=(1.8, 0.5, 0.16))
d_ls = solve_delta(logF_mc, np.log(S_mc), taus_mc, r_mc, th["kappa"], th["sigma2"], th["m"])
rm = float(np.sqrt(np.mean((d_ls - dtrue) ** 2)))
corr = float(np.corrcoef(d_ls, dtrue)[0, 1])
m_true = 0.07790831556503197 + 0.393 * 0.527 * 0.766 / 1.876
print(f"  fitted: kappa {th['kappa']:.4f} (true 1.8760)   sigma2 {th['sigma2']:.4f} (true 0.5270)   "
      f"m {th['m']:.5f} (true {m_true:.5f})")
print(f"  price RMSE {rmse_price:.5f}   |   delta RMSE {rm:.5f}   corr {corr:+.4f}")
print(f"  delta increment std {np.diff(d_ls).std():.5f}  vs true {np.diff(dtrue).std():.5f} "
      f"({100*np.diff(d_ls).std()/np.diff(dtrue).std():.1f}%)")
print(f"  --- AP-PEN (40k epochs, neural) delta RMSE 0.04994, corr +0.9791, incr 32.0% ---")
print(f"  --- MLP    (40k epochs, neural) delta RMSE 0.05708, corr +0.9726, incr 27.4% ---")
print(f"  --- no-skill baseline (constant) delta RMSE {dtrue.std():.5f} ---")
results["mc"] = dict(psi=th, price_rmse=rmse_price, delta_rmse=rm, corr=corr,
                     delta_ls=d_ls, delta_true=dtrue, t=t_mc, m_true=m_true)

print("\n  profile likelihoods (price RMSE as each identified param is fixed):")
for name, grid in [("kappa", np.linspace(1.2, 2.6, 15)),
                   ("sigma2", np.linspace(0.3, 0.8, 15)),
                   ("m", np.linspace(0.10, 0.23, 15))]:
    pr = profile(logF_mc, np.log(S_mc), taus_mc, r_mc, name, grid, th)
    rel = pr / rmse_price - 1.0
    print(f"    {name:7s} range over grid: {100*rel.min():+.2f}% to {100*rel.max():+.2f}% of min RMSE")
    results.setdefault("profiles_mc", {})[name] = (grid, pr)

print("\n" + "=" * 84)
print("B. THE FLAT DIRECTIONS -- sigma1/rho profiles are EXACTLY flat by construction")
print("=" * 84)
print("  For any (sigma1, rho), setting alpha_Q = m_hat - sigma1*sigma2*rho/kappa gives")
print("  an IDENTICAL fit. Profile likelihood over sigma1 or rho is therefore exactly")
print("  horizontal -- there is no curvature to estimate a standard error from.")
for s1 in [0.15, 0.393, 0.9, 1.4]:
    for rho in [-0.5, 0.766]:
        aQ = th["m"] - s1 * th["sigma2"] * rho / th["kappa"]
        print(f"    sigma1={s1:5.3f} rho={rho:+.3f} -> alpha_Q={aQ:+.4f}  "
              f"(price RMSE identical at {rmse_price:.6f})")

print("\n" + "=" * 84)
print("C. REAL WTI -- vs Kalman filter and the neural variants")
print("=" * 84)
df = pd.read_csv(f"{REPO}/data/input/real/wti_analysis_ready.csv").dropna(subset=["spot", "rate"])
df["date"] = pd.to_datetime(df["date"]); df = df.sort_values(["date", "tau"])
df["slot"] = df.groupby("date").cumcount(); df = df[df["slot"] < 8]
df = df[df.groupby("date")["slot"].transform("count") == 8]
tau_m = df.pivot(index="date", columns="slot", values="tau").sort_index()
settle = df.pivot(index="date", columns="slot", values="settle").sort_index()
spot = df.groupby("date")["spot"].first().sort_index()
rate = df.groupby("date")["rate"].first().sort_index()
dates_r = tau_m.index.to_numpy()
taus_r = tau_m.to_numpy().mean(axis=0)
logF_r = np.log(settle.to_numpy()); S_r = spot.to_numpy(); r_r = rate.to_numpy()
t_r = ((dates_r - dates_r[0]) / np.timedelta64(365, "D")).astype(float)

th_r, rmse_r = fit(logF_r, np.log(S_r), taus_r, r_r, x0=(0.43, 0.17, 0.08))
d_ls_r = solve_delta(logF_r, np.log(S_r), taus_r, r_r, th_r["kappa"], th_r["sigma2"], th_r["m"])
print(f"  fitted: kappa {th_r['kappa']:.4f}   sigma2 {th_r['sigma2']:.4f}   m {th_r['m']:.5f}")
print(f"  price RMSE {rmse_r:.5f}   (Kalman in-sample innovation RMSE 0.0263)")
print(f"  delta_hat: mean {d_ls_r.mean():+.4f} std {d_ls_r.std():.4f} "
      f"range [{d_ls_r.min():+.4f}, {d_ls_r.max():+.4f}]")

kf = pickle.load(open(f"{REPO}/data/output/kalman/results/results_kalman.pkl", "rb"))
kfd = pd.DatetimeIndex(kf["dates"]); rd = pd.DatetimeIndex(dates_r)
common = kfd.intersection(rd)
k = kf["delta_hat"][kfd.get_indexer(common)]
l = d_ls_r[rd.get_indexer(common)]
print(f"  vs Kalman: corr {np.corrcoef(l,k)[0,1]:+.4f}  RMSE {np.sqrt(np.mean((l-k)**2)):.4f}   "
      f"(null=constant: {k.std():.4f})")
results["real"] = dict(psi=th_r, price_rmse=rmse_r, delta_ls=d_ls_r, dates=dates_r,
                       corr_kf=float(np.corrcoef(l, k)[0, 1]),
                       rmse_kf=float(np.sqrt(np.mean((l - k) ** 2))))

print("\n" + "=" * 84)
print("D. TEMPORAL HOLDOUT -- decomposing the neural model's out-of-sample error")
print("=" * 84)
cut = np.datetime64("2024-01-01")
tr, te = dates_r < cut, dates_r >= cut
th_ho, rmse_in = fit(logF_r[tr], np.log(S_r[tr]), taus_r, r_r[tr], x0=(0.43, 0.17, 0.08))
# (i) psi from train, delta re-solved per date on test  -> psi-generalization only
d_te = solve_delta(logF_r[te], np.log(S_r[te]), taus_r, r_r[te], th_ho["kappa"], th_ho["sigma2"], th_ho["m"])
B = B_coeff(taus_r, th_ho["kappa"]); A_te = A_coeff(taus_r[None, :], r_r[te][:, None], th_ho["kappa"], th_ho["sigma2"], th_ho["m"])
rmse_out_solve = float(np.sqrt(np.mean((logF_r[te] - (np.log(S_r[te])[:, None] + B[None, :] * d_te[:, None] + A_te)) ** 2)))
# (ii) psi from train, delta FROZEN at its last in-sample value -> mimics a non-extrapolating path
d_tr = solve_delta(logF_r[tr], np.log(S_r[tr]), taus_r, r_r[tr], th_ho["kappa"], th_ho["sigma2"], th_ho["m"])
d_frozen = np.full(te.sum(), d_tr[-1])
rmse_out_frozen = float(np.sqrt(np.mean((logF_r[te] - (np.log(S_r[te])[:, None] + B[None, :] * d_frozen[:, None] + A_te)) ** 2)))
print(f"  LS, psi fit on <2024:  in-sample {rmse_in:.5f}")
print(f"    (i)  delta re-solved per test date : out {rmse_out_solve:.5f}  ratio {rmse_out_solve/rmse_in:.2f}x"
      "   <- pure psi-generalization (the floor)")
print(f"    (ii) delta frozen at last train val: out {rmse_out_frozen:.5f}  ratio {rmse_out_frozen/rmse_in:.2f}x"
      "   <- what a non-extrapolating path costs")
print(f"  --- neural, same split: MLP 8.86x, AP-PEN 1.69x, AP-PEN(ARB) 1.09x ---")
results["holdout"] = dict(psi=th_ho, rmse_in=rmse_in, rmse_out_solve=rmse_out_solve,
                          rmse_out_frozen=rmse_out_frozen)

with open(f"{OUT}/results/ls_baseline.pkl", "wb") as f:
    pickle.dump(results, f)
print(f"\nwrote {OUT}/results/ls_baseline.pkl")
