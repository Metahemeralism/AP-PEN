"""Fix 1 + Fix 2: real-WTI experiments.

  1. kappa-floored re-run of all three variants on the full real panel.
     KAPPA_MIN=0.1 is not arbitrary: below kappa=0.1 the mean-reversion
     half-life (ln2/kappa = 6.9yr) exceeds the longest observed maturity
     (1.88yr) by ~3.7x, so B(tau) ~= -tau over the entire observable range
     and the second factor is indistinguishable from deterministic carry.
     kappa >= 0.1 is the boundary of "the two-factor structure is identifiable
     from this maturity range at all."

  2. seed repeats (5 seeds x 3 variants) on the 2024 temporal holdout, and on
     the full-panel fit, so the headline claims get error bars.
"""
import os, pickle, sys
import numpy as np
import pandas as pd
import appen as A

OUT = f"{A.REPO}/data/output/real_data"
os.makedirs(f"{OUT}/results", exist_ok=True)
SEEDS = [42, 7, 123, 2024, 31337]
EPOCHS = 40000

kf = pickle.load(open(f"{A.REPO}/data/output/kalman/results/results_kalman.pkl", "rb"))
dates, t, S, taus, logF, r = A.get_data_wti()
psi_init = {k: float(kf["psi_hat"][k]) for k in ["kappa", "sigma1", "sigma2", "rho", "alpha_P"]}
lam = float(kf["psi_hat"]["lam"])

kfd = pd.DatetimeIndex(kf["dates"]); rd = pd.DatetimeIndex(dates)
common = kfd.intersection(rd)
kf_delta = kf["delta_hat"][kfd.get_indexer(common)]
wti_pos = rd.get_indexer(common)
NULL_RMSE = float(kf_delta.std())


def vs_kf(d):
    dc = d[wti_pos]
    return (float(np.corrcoef(dc, kf_delta)[0, 1]),
            float(np.sqrt(np.mean((dc - kf_delta) ** 2))))


# ============================================================ 1. kappa floor
print("=" * 92, flush=True)
print("FIX 1: real-WTI full panel, KAPPA_MIN=0.1 (vs the unfloored collapse)", flush=True)
print("=" * 92, flush=True)
floored = {}
for kmin in [0.0, 0.1]:
    A.configure(taus, t[-1], psi_init, lam, len(t), kappa_min=kmin)
    data = (t, S, logF, r)
    cfgs = A.make_variant_configs(num_epochs=EPOCHS, seed=42)
    for cfg in [cfgs[0], cfgs[2], cfgs[3]]:
        tag = A.run_tag(cfg)
        fns, _ = A.calibration(cfg, data)
        _, dfn, gsp = fns
        d = np.asarray(A.vmap(dfn)(t))
        psi = A.psi_dict(gsp["p_Q"], gsp["alpha_P"])
        err, metrics = A.error(fns, data)
        c, rm = vs_kf(d)
        floored[(kmin, tag)] = dict(psi=psi, delta_hat=d, corr_kf=c, rmse_kf=rm,
                                    m=A.m_of(psi),
                                    metrics={k: float(v) for k, v in metrics.items()})
        print(f"  kmin={kmin:.1f} {tag:11s} kappa {psi['kappa']:7.4f} sigma2 {psi['sigma2']:6.4f} "
              f"m {A.m_of(psi):+8.4f} | dmin {d.min():+.3f} dstd {d.std():.3f} "
              f"| vs KF corr {c:+.4f} RMSE {rm:.4f}", flush=True)
print(f"  [null: constant at KF mean -> RMSE {NULL_RMSE:.4f}]", flush=True)

with open(f"{OUT}/results/kappa_floor_comparison.pkl", "wb") as f:
    pickle.dump({"results": floored, "null_rmse": NULL_RMSE, "kf_delta": kf_delta,
                 "common_dates": np.asarray(common), "kappa_min_rationale":
                 "half-life ln2/0.1=6.9yr vs longest maturity 1.88yr"}, f)

# ============================================================ 2. seed repeats
print("\n" + "=" * 92, flush=True)
print(f"FIX 2: seed repeats ({len(SEEDS)} seeds), temporal holdout train<2024 / test>=2024", flush=True)
print("=" * 92, flush=True)
cut = np.datetime64("2024-01-01")
tr, te = dates < cut, dates >= cut
print(f"  train {tr.sum()} dates, test {te.sum()} ({100*te.mean():.1f}%)", flush=True)

A.configure(taus, t[tr][-1], psi_init, lam, int(tr.sum()), kappa_min=0.1)
data_ho = (t[tr], S[tr], logF[tr], r[tr])

ho = {}
for seed in SEEDS:
    cfgs = A.make_variant_configs(num_epochs=EPOCHS, seed=seed)
    for cfg in [cfgs[0], cfgs[2], cfgs[3]]:
        tag = A.run_tag(cfg)
        fns, _ = A.calibration(cfg, data_ho, verbose=False)
        rin, _ = A.price_rmse(fns, t[tr], S[tr], logF[tr], r[tr])
        rout, _ = A.price_rmse(fns, t[te], S[te], logF[te], r[te])
        _, dfn, gsp = fns
        ho[(seed, tag)] = dict(rmse_in=rin, rmse_out=rout, ratio=rout / rin,
                               psi=A.psi_dict(gsp["p_Q"], gsp["alpha_P"]),
                               delta_full=np.asarray(A.vmap(dfn)(t)))
        print(f"  seed {seed:6d} {tag:11s} in {rin:.5f}  out {rout:.5f}  ratio {rout/rin:6.2f}x", flush=True)

print("\n  --- SUMMARY across seeds ---", flush=True)
print(f"  {'variant':>12} | {'in-sample':>18} | {'out-of-sample':>18} | {'ratio':>18}", flush=True)
summary = {}
for tag in ["MLP", "APPINN", "APPINN_ARB"]:
    ri = np.array([ho[(s, tag)]["rmse_in"] for s in SEEDS])
    ro = np.array([ho[(s, tag)]["rmse_out"] for s in SEEDS])
    rt = np.array([ho[(s, tag)]["ratio"] for s in SEEDS])
    summary[tag] = dict(rmse_in=(ri.mean(), ri.std()), rmse_out=(ro.mean(), ro.std()),
                        ratio=(rt.mean(), rt.std()), ratio_all=rt.tolist())
    print(f"  {tag:>12} | {ri.mean():.5f} +/- {ri.std():.5f} | {ro.mean():.5f} +/- {ro.std():.5f} | "
          f"{rt.mean():6.2f}x +/- {rt.std():.2f}  [{rt.min():.2f}-{rt.max():.2f}]", flush=True)

with open(f"{OUT}/results/holdout_seed_repeats.pkl", "wb") as f:
    pickle.dump({"runs": ho, "summary": summary, "seeds": SEEDS, "dates": dates,
                 "train_mask": tr, "test_mask": te}, f)

# ================================== 3. seed repeats on the FULL-panel real fit
print("\n" + "=" * 92, flush=True)
print("FIX 2b: seed repeats on the full-panel real fit (kappa floored) vs Kalman", flush=True)
print("=" * 92, flush=True)
A.configure(taus, t[-1], psi_init, lam, len(t), kappa_min=0.1)
data = (t, S, logF, r)
full = {}
for seed in SEEDS:
    cfgs = A.make_variant_configs(num_epochs=EPOCHS, seed=seed)
    for cfg in [cfgs[0], cfgs[2], cfgs[3]]:
        tag = A.run_tag(cfg)
        fns, _ = A.calibration(cfg, data, verbose=False)
        _, dfn, gsp = fns
        d = np.asarray(A.vmap(dfn)(t))
        psi = A.psi_dict(gsp["p_Q"], gsp["alpha_P"])
        c, rm = vs_kf(d)
        full[(seed, tag)] = dict(psi=psi, corr_kf=c, rmse_kf=rm, m=A.m_of(psi),
                                 dmin=float(d.min()), dstd=float(d.std()))
        print(f"  seed {seed:6d} {tag:11s} corr {c:+.4f} RMSE {rm:.4f} kappa {psi['kappa']:.4f} "
              f"m {A.m_of(psi):+.4f} dmin {d.min():+.3f}", flush=True)

print(f"\n  {'variant':>12} | {'corr vs KF':>18} | {'RMSE vs KF':>18} | {'kappa':>16} | {'m':>16}", flush=True)
fsum = {}
for tag in ["MLP", "APPINN", "APPINN_ARB"]:
    cc = np.array([full[(s, tag)]["corr_kf"] for s in SEEDS])
    rr = np.array([full[(s, tag)]["rmse_kf"] for s in SEEDS])
    kk = np.array([full[(s, tag)]["psi"]["kappa"] for s in SEEDS])
    mm = np.array([full[(s, tag)]["m"] for s in SEEDS])
    fsum[tag] = dict(corr=(cc.mean(), cc.std()), rmse=(rr.mean(), rr.std()),
                     kappa=(kk.mean(), kk.std()), m=(mm.mean(), mm.std()))
    print(f"  {tag:>12} | {cc.mean():+.4f} +/- {cc.std():.4f} | {rr.mean():.4f} +/- {rr.std():.4f} | "
          f"{kk.mean():.3f} +/- {kk.std():.3f} | {mm.mean():+.3f} +/- {mm.std():.3f}", flush=True)
print(f"  [null = constant at KF mean: RMSE {NULL_RMSE:.4f}]", flush=True)

with open(f"{OUT}/results/full_panel_seed_repeats.pkl", "wb") as f:
    pickle.dump({"runs": full, "summary": fsum, "seeds": SEEDS, "null_rmse": NULL_RMSE}, f)
print("\nDONE", flush=True)
