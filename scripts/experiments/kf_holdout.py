"""Kalman filter refit on the 2024 temporal holdout.

WHY
---
Table `tab:holdout` reports out-of-sample price RMSE ratios for the neural
variants, but had no incumbent baseline to be judged against: the Kalman filter
was only ever fitted on the full panel. This refits it on dates before
2024-01-01 and evaluates on the held-out window, so the holdout comparison
finally has a reference point.

MAKING THE INFORMATION SETS COMPARABLE
--------------------------------------
The obvious KF out-of-sample number -- its one-step-ahead innovation RMSE on
test dates -- is NOT comparable to the neural rows, because the filter updates
on each test observation as it goes, while the path network sees no test data
at all. Reporting only that would flatter the filter.

So two test-window numbers are computed:

  (a) FILTERED  one-step-ahead innovations, updating through the test window.
      The filter's own natural out-of-sample metric, and an upper bound on what
      it can do. Uses test observations up to t-1.

  (b) FORECAST  psi and the state are taken at the cutoff and delta is
      propagated forward by the OU transition mean alone, with NO measurement
      updates. The observed spot is used (as the neural models do, since S_t is
      an input rather than a latent there), so only the latent delta is
      forecast. This matches the path network's information set exactly and is
      the like-for-like row.

(b) is the honest comparator; (a) is reported because the gap between them is
itself informative -- it measures how much of the filter's advantage comes from
being allowed to keep looking at the data.

Run:  conda run -n pinns python scripts/experiments/kf_holdout.py
"""
import json, os, pickle, time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import jax
jax.config.update("jax_enable_x64", True)   # CLAUDE.md S5: long sequential recursion
import jax.numpy as jnp
from jax import lax

REPO = "/Users/evanlynch/Developer/DC-PINNs"
CUTOFF = np.datetime64("2024-01-01")
OUT = f"{REPO}/data/output/kalman/results"
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ pricer
def B_coeff(tau, kappa):
    return -(1.0 - jnp.exp(-kappa * tau)) / kappa


def A_coeff(tau, r, kappa, alpha_Q, sigma1, sigma2, rho):
    s1s2rho = sigma1 * sigma2 * rho
    s2sq = sigma2 ** 2
    lin = (r - alpha_Q + 0.5 * s2sq / kappa ** 2 - s1s2rho / kappa) * tau
    two = 0.25 * s2sq * (1.0 - jnp.exp(-2.0 * kappa * tau)) / kappa ** 3
    one = (alpha_Q * kappa + s1s2rho - s2sq / kappa) * (1.0 - jnp.exp(-kappa * tau)) / kappa ** 2
    return lin + two + one


# ------------------------------------------------------------------ data
df = pd.read_csv(f"{REPO}/data/input/real/wti_analysis_ready.csv").dropna(subset=["spot", "rate"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "tau"])
df["slot"] = df.groupby("date").cumcount()
df = df[df["slot"] < 8]
df = df[df.groupby("date")["slot"].transform("count") == 8]

tau_m = df.pivot(index="date", columns="slot", values="tau").sort_index()
settle = df.pivot(index="date", columns="slot", values="settle").sort_index()
spot = df.groupby("date")["spot"].first().sort_index().to_numpy()
rate = df.groupby("date")["rate"].first().sort_index().to_numpy()

dates = tau_m.index.to_numpy()
taus = tau_m.to_numpy().mean(axis=0)
logF = np.log(settle.to_numpy())
K = len(taus)

train = dates < CUTOFF
n_tr = int(train.sum())
print(f"train {n_tr} dates (< {CUTOFF})   test {int((~train).sum())} "
      f"({100*(~train).mean():.1f}% held out)\n", flush=True)

taus_j = jnp.asarray(taus)
N_PARAMS = 7 + K


def unpack_psi(raw):
    return dict(kappa=jax.nn.softplus(raw[0]) + 1e-4, mu=raw[1], alpha_P=raw[2],
                lam=raw[3], sigma1=jax.nn.softplus(raw[4]) + 1e-4,
                sigma2=jax.nn.softplus(raw[5]) + 1e-4, rho=jnp.tanh(raw[6]),
                h=jax.nn.softplus(raw[7:7 + K]) + 1e-8)


def make_filter(logF_arr, r_arr, dt_arr, logS0, delta0):
    """Filter over an arbitrary window. Returns (loglik, a_path, innovs)."""
    logF_j = jnp.asarray(logF_arr[1:])
    r_j = jnp.asarray(r_arr[1:])
    dt_j = jnp.asarray(dt_arr)

    def run(raw):
        p = unpack_psi(raw)
        kappa, mu, alpha_P, lam = p["kappa"], p["mu"], p["alpha_P"], p["lam"]
        s1, s2, rho, h = p["sigma1"], p["sigma2"], p["rho"], p["h"]
        alpha_Q = alpha_P - s2 * lam / kappa
        Z = jnp.stack([jnp.ones(K), B_coeff(taus_j, kappa)], axis=1)
        R = jnp.diag(h)
        a0 = jnp.array([logS0, delta0])
        P0 = jnp.diag(jnp.array([0.05 ** 2, 0.1 ** 2]))

        def step(carry, xs):
            a_prev, P_prev, ll = carry
            dt_t, y_t, r_val = xs
            Q_t = jnp.array([[1.0, -dt_t], [0.0, 1.0 - kappa * dt_t]])
            c_t = jnp.array([(mu - 0.5 * s1 ** 2) * dt_t, kappa * alpha_P * dt_t])
            V_t = dt_t * jnp.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
            a_pred = c_t + Q_t @ a_prev
            P_pred = Q_t @ P_prev @ Q_t.T + V_t
            d_t = A_coeff(taus_j, r_val, kappa, alpha_Q, s1, s2, rho)
            innov = y_t - (d_t + Z @ a_pred)
            S_t = Z @ P_pred @ Z.T + R
            S_inv = jnp.linalg.inv(S_t)
            gain = P_pred @ Z.T @ S_inv
            a_new = a_pred + gain @ innov
            P_new = P_pred - gain @ Z @ P_pred
            _, logdet = jnp.linalg.slogdet(S_t)
            ll_t = -0.5 * (K * jnp.log(2 * jnp.pi) + logdet + innov @ S_inv @ innov)
            return (a_new, P_new, ll + ll_t), (a_new, innov)

        (_, _, ll), (a_path, innovs) = lax.scan(
            step, (a0, P0, 0.0), (dt_j, logF_j, r_j))
        return ll, a_path, innovs
    return run


def dt_of(d):
    return np.diff(d).astype("timedelta64[D]").astype(float) / 365.0


# ---- delta_0 seed (Krul eq. 4.9), from the first TRAINING date -------------
t1, t2 = taus[0], taus[1]
delta0 = (rate[0] * t1 - rate[0] * t2) / (t1 - t2) - (logF[0, 0] - logF[0, 1]) / (t1 - t2)
logS0 = float(np.log(spot[0]))

# ============================================================ fit on train
run_tr = make_filter(logF[train], rate[train], dt_of(dates[train]), logS0, delta0)
nll_grad = jax.jit(jax.value_and_grad(lambda raw: -run_tr(raw)[0]))


def inv_softplus(y):
    return np.log(np.expm1(np.asarray(y, dtype=np.float64)))


with open(f"{REPO}/config/p_params.json") as f:
    p0 = json.load(f)
with open(f"{REPO}/config/q_params.json") as f:
    q0 = json.load(f)
raw0 = np.concatenate([
    np.array([inv_softplus(p0["kappa"]), p0["mu"], p0["alpha_P"], q0["lambda2"],
              inv_softplus(p0["sigma1"]), inv_softplus(p0["sigma2"]),
              float(np.arctanh(np.clip(p0["rho"], -0.999, 0.999)))]),
    inv_softplus(np.maximum(logF[train].var(axis=0), 1e-4))])

t0 = time.time()
res = minimize(lambda x: tuple(np.asarray(v, dtype=np.float64) if i else float(v)
                               for i, v in enumerate(nll_grad(jnp.asarray(x)))),
               raw0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
print(f"MLE on train window: {time.time()-t0:.1f}s, converged={res.success}, "
      f"nll={res.fun:.1f}", flush=True)

raw_star = jnp.asarray(res.x)
psi = {k: (float(v) if np.ndim(v) == 0 else np.asarray(v))
       for k, v in unpack_psi(raw_star).items()}
psi["alpha_Q"] = psi["alpha_P"] - psi["sigma2"] * psi["lam"] / psi["kappa"]
print("psi_hat (train only):", {k: round(v, 4) for k, v in psi.items() if np.ndim(v) == 0})

_, a_tr, innov_tr = run_tr(raw_star)
rmse_in = float(np.sqrt(np.mean(np.asarray(innov_tr) ** 2)))
print(f"\nin-sample innovation RMSE (train window): {rmse_in:.5f}")

# ============================== (a) FILTERED through the test window
# Continue the recursion from the last training state, updating on test obs.
a_cut = np.asarray(a_tr)[-1]
bridge = np.r_[np.where(train)[0][-1], np.where(~train)[0]]   # last train date + all test
run_te = make_filter(logF[bridge], rate[bridge], dt_of(dates[bridge]),
                     float(a_cut[0]), float(a_cut[1]))
_, a_te, innov_te = run_te(raw_star)
rmse_out_filt = float(np.sqrt(np.mean(np.asarray(innov_te) ** 2)))

# ============================== (b) FORECAST: propagate delta, no updates
kappa, alpha_P = psi["kappa"], psi["alpha_P"]
dts = dt_of(dates[bridge])
delta_f = float(a_cut[1])
deltas = []
for h in dts:                       # OU/EM transition mean only
    delta_f = delta_f + kappa * (alpha_P - delta_f) * h
    deltas.append(delta_f)
deltas = np.array(deltas)                                     # (n_test,)

test_idx = bridge[1:]
Bv = np.asarray(B_coeff(taus_j, kappa))
Av = np.asarray(jax.vmap(lambda rr: A_coeff(taus_j, rr, kappa, psi["alpha_Q"],
                                            psi["sigma1"], psi["sigma2"], psi["rho"])
                         )(jnp.asarray(rate[test_idx])))       # (n_test, K)
logF_hat = np.log(spot[test_idx])[:, None] + Bv[None, :] * deltas[:, None] + Av
rmse_out_fc = float(np.sqrt(np.mean((logF_hat - logF[test_idx]) ** 2)))

print(f"\n{'metric':>44} | {'RMSE':>9} | {'ratio':>7}")
print("-" * 68)
print(f"{'in-sample (train window)':>44} | {rmse_in:9.5f} | {1.0:7.2f}x")
print(f"{'(a) test, FILTERED (updates on test obs)':>44} | {rmse_out_filt:9.5f} | {rmse_out_filt/rmse_in:7.2f}x")
print(f"{'(b) test, FORECAST (no test obs) <- comparable':>44} | {rmse_out_fc:9.5f} | {rmse_out_fc/rmse_in:7.2f}x")
print(f"\n  delta forecast over test window: {deltas[0]:+.4f} -> {deltas[-1]:+.4f} "
      f"(decaying toward alpha_P = {alpha_P:+.4f})")
print("\n  For comparison, same split, neural + closed-form (5 seeds where applicable):")
print("    MLP           5.73x +/- 6.81      AP-PEN        6.71x +/- 4.83")
print("    AP-PEN (ARB)  0.93x +/- 0.33      LS, delta frozen  2.72x")
print("    LS, delta re-solved per test date 0.69x")

with open(f"{OUT}/results_kalman_holdout.pkl", "wb") as f:
    pickle.dump({"psi_hat": psi, "cutoff": CUTOFF, "n_train": n_tr,
                 "rmse_in": rmse_in, "rmse_out_filtered": rmse_out_filt,
                 "rmse_out_forecast": rmse_out_fc, "delta_forecast": deltas,
                 "test_dates": dates[test_idx], "converged": bool(res.success),
                 "note": "(b) is the like-for-like comparator with the neural holdout"}, f)
print(f"\nwrote {OUT}/results_kalman_holdout.pkl")
