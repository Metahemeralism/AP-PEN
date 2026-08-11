"""Follow-up: pin down exactly WHICH 3 quantities the futures curve identifies,
and check whether the trained variants agree on those even where raw psi differs.

Claim to test:
    A(tau) depends on (kappa, sigma1, sigma2, rho, alpha_Q) ONLY through
        (kappa, sigma2, m)   with   m = alpha_Q + sigma1*sigma2*rho/kappa
    because both the tau-linear coefficient and the (1-e^{-kappa tau})
    coefficient are functions of m alone.
"""
import os, pickle
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
os.chdir("/Users/evanlynch/Developer/DC-PINNs")
R = 0.05


def B_coeff(tau, kappa):
    return -(1.0 - jnp.exp(-kappa * tau)) / kappa


def A_coeff(tau, kappa, sigma1, sigma2, rho, alpha_Q, r=R):
    P = sigma1 * sigma2 * rho
    s2sq = sigma2 ** 2
    lin = (r - alpha_Q + 0.5 * s2sq / kappa ** 2 - P / kappa) * tau
    two = 0.25 * s2sq * (1.0 - jnp.exp(-2.0 * kappa * tau)) / kappa ** 3
    one = (alpha_Q * kappa + P - s2sq / kappa) * (1.0 - jnp.exp(-kappa * tau)) / kappa ** 2
    return lin + two + one


def m_of(kappa, sigma1, sigma2, rho, alpha_Q):
    return alpha_Q + sigma1 * sigma2 * rho / kappa


TAUS = jnp.linspace(0.05, 2.0, 40)
PT = dict(kappa=1.876, sigma1=0.393, sigma2=0.527, rho=0.766, alpha_Q=0.07790831556503197)

print("=" * 78)
print("TEST 1: hold (kappa, sigma2, m) fixed, move sigma1/rho/alpha_Q wildly")
print("=" * 78)
base = A_coeff(TAUS, **PT)
m0 = m_of(**PT)
print(f"true m = alpha_Q + sigma1*sigma2*rho/kappa = {m0:.6f}\n")
print(f"{'sigma1':>9} {'rho':>9} {'alpha_Q':>10} {'m':>10} {'max|dA|':>12}")
rng = np.random.default_rng(0)
for _ in range(6):
    s1 = float(rng.uniform(0.1, 1.5))
    rho = float(rng.uniform(-0.95, 0.95))
    # solve alpha_Q so that m is preserved
    aQ = m0 - s1 * PT["sigma2"] * rho / PT["kappa"]
    alt = A_coeff(TAUS, PT["kappa"], s1, PT["sigma2"], rho, aQ)
    print(f"{s1:9.4f} {rho:9.4f} {aQ:10.4f} {m_of(PT['kappa'], s1, PT['sigma2'], rho, aQ):10.6f} "
          f"{float(jnp.max(jnp.abs(alt - base))):12.3e}")

print("\n" + "=" * 78)
print("TEST 2: is the LEVEL of delta confounded with these too?")
print("  (shift delta by c, refit m) -- log F = log S + B(tau)delta + A(tau)")
print("=" * 78)
delta0 = 0.10
for c in [0.02, 0.05, 0.10]:
    obs0 = B_coeff(TAUS, PT["kappa"]) * delta0 + A_coeff(TAUS, **PT)
    # shift delta by c, then search m-shift that best re-fits (kappa,sigma2 held)
    def resid(dm):
        alt = dict(PT); alt["alpha_Q"] = PT["alpha_Q"] + dm
        return np.asarray(B_coeff(TAUS, PT["kappa"]) * (delta0 + c) + A_coeff(TAUS, **alt) - obs0)
    grid = np.linspace(-0.3, 0.3, 60001)
    errs = np.array([np.sqrt(np.mean(resid(d) ** 2)) for d in grid[::200]])
    d_best = grid[::200][errs.argmin()]
    fine = np.linspace(d_best - 0.01, d_best + 0.01, 2001)
    errs2 = np.array([np.sqrt(np.mean(resid(d) ** 2)) for d in fine])
    print(f"  delta shift +{c:.2f}: best alpha_Q shift {fine[errs2.argmin()]:+.4f}  "
          f"residual RMSE in log-price {errs2.min():.3e}   (obs noise std = 0.01)")

print("\n" + "=" * 78)
print("TEST 3: do the trained variants agree on the IDENTIFIED composites?")
print("=" * 78)
with open("data/output/mc_simulated_single_net/results/delta_recovery_all_variants.pkl", "rb") as f:
    dv = pickle.load(f)
TRUE = dict(PT)
rows = [("TRUE", TRUE)]
for tag, v in dv["variants"].items():
    p = v["psi_hat"]
    rows.append((tag, dict(kappa=p["kappa"], sigma1=p["sigma1"], sigma2=p["sigma2"],
                           rho=p["rho"], alpha_Q=p["alpha_Q"])))
print(f"{'variant':>12} | {'kappa':>8} {'sigma2':>8} | {'sigma1':>8} {'rho':>8} {'alpha_Q':>9} | "
      f"{'P=s1s2rho':>10} {'m (identified)':>15}")
print("-" * 100)
for tag, p in rows:
    P = p["sigma1"] * p["sigma2"] * p["rho"]
    print(f"{tag:>12} | {p['kappa']:8.4f} {p['sigma2']:8.4f} | {p['sigma1']:8.4f} {p['rho']:8.4f} "
          f"{p['alpha_Q']:9.4f} | {P:10.4f} {m_of(**p):15.6f}")

print("\n--- same, for the REAL-DATA runs + Kalman filter ---")
real = {}
for tag in ["MLP", "APPINN", "APPINN_ARB"]:
    with open(f"data/output/real_data/results/results_{tag}.pkl", "rb") as f:
        real[tag] = pickle.load(f)["psi"]
with open("data/output/kalman/results/results_kalman.pkl", "rb") as f:
    kfp = pickle.load(f)["psi_hat"]
real["KALMAN"] = kfp
print(f"{'variant':>12} | {'kappa':>8} {'sigma2':>8} | {'sigma1':>8} {'rho':>8} {'alpha_Q':>9} | "
      f"{'P=s1s2rho':>10} {'m (identified)':>15}")
print("-" * 100)
for tag, p in real.items():
    q = dict(kappa=float(p["kappa"]), sigma1=float(p["sigma1"]), sigma2=float(p["sigma2"]),
             rho=float(p["rho"]), alpha_Q=float(p["alpha_Q"]))
    P = q["sigma1"] * q["sigma2"] * q["rho"]
    print(f"{tag:>12} | {q['kappa']:8.4f} {q['sigma2']:8.4f} | {q['sigma1']:8.4f} {q['rho']:8.4f} "
          f"{q['alpha_Q']:9.4f} | {P:10.4f} {m_of(**q):15.6f}")

print("\n" + "=" * 78)
print("TEST 4: 20-path repeats -- is m stable even though sigma1/rho are not?")
print("=" * 78)
for f_ in ["data/output/mc_simulated_single_net/results/path_repeat.pkl",
           "data/output/mc_simulated_single_net/results/path_repeat_noise0.10.pkl",
           "data/output/mc_simulated/results/path_repeat.pkl"]:
    with open(f_, "rb") as fh:
        d = pickle.load(fh)
    ms, Ps, k, s2 = [], [], [], []
    for r in d["results"]:
        p = r["psi_hat"]
        ms.append(m_of(p["kappa"], p["sigma1"], p["sigma2"], p["rho"], p["alpha_Q"]))
        Ps.append(p["sigma1"] * p["sigma2"] * p["rho"])
        k.append(p["kappa"]); s2.append(p["sigma2"])
    pt = d["psi_true"]
    m_true = m_of(pt["kappa"], pt["sigma1"], pt["sigma2"], pt["rho"], pt["alpha_Q"])
    lbl = os.path.basename(f_) + "  [" + os.path.basename(os.path.dirname(os.path.dirname(f_))) + "]"
    print(f"\n{lbl}")
    for name, arr, tv in [("kappa", k, pt["kappa"]), ("sigma2", s2, pt["sigma2"]),
                          ("P=s1s2rho", Ps, pt["sigma1"] * pt["sigma2"] * pt["rho"]),
                          ("m", ms, m_true)]:
        a = np.array(arr)
        print(f"   {name:>10}: true {tv:8.4f}  mean {a.mean():8.4f}  std {a.std():8.4f}  "
              f"CV {a.std()/abs(a.mean()):6.2%}")
