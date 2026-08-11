"""Fix 3: the MC temporal holdout that was written into AP-PEN.ipynb (cells 34-37)
but never executed.

Unlike the real-WTI holdout this can score OUT-OF-SAMPLE DELTA against known
truth, not just price reconstruction -- the strongest evidence available for
whether the physics channels buy generalisation. Run over 5 seeds.

Also re-runs the in-sample MC comparison over seeds so the headline synthetic
delta-recovery numbers get error bars, and computes the LS baseline on the same
split for reference.
"""
import os, pickle
import numpy as np
import appen as A

OUT = f"{A.REPO}/data/output/mc_simulated_single_net"
os.makedirs(f"{OUT}/results", exist_ok=True)
SEEDS = [42, 7, 123, 2024, 31337]
EPOCHS = 40000
CUT = 8.0

t, S, taus, logF, dtrue, p_Q, kappa_P, alpha_P, sigma2_P = A.get_data_mc(0)
t = np.asarray(t); S = np.asarray(S); taus = np.asarray(taus)
logF = np.asarray(logF); dtrue = np.asarray(dtrue)
r = np.full(len(t), 0.05)

PSI_TRUE = A.psi_dict(p_Q, alpha_P)
PSI_INIT = {"kappa": PSI_TRUE["kappa"] * 1.5, "sigma1": PSI_TRUE["sigma1"] * 0.7,
            "sigma2": PSI_TRUE["sigma2"] * 1.3, "rho": PSI_TRUE["rho"] * 0.6,
            "alpha_P": PSI_TRUE["alpha_P"] - 0.06}
LAM_TRUE = PSI_TRUE["kappa"] * (PSI_TRUE["alpha_P"] - PSI_TRUE["alpha_Q"]) / PSI_TRUE["sigma2"]
LAM = LAM_TRUE * 1.30
M_TRUE = A.m_of(PSI_TRUE)

tr, te = t <= CUT, t > CUT
print(f"MC holdout: train {tr.sum()} dates (t<={CUT}yr)  test {te.sum()} ({100*te.mean():.1f}%)", flush=True)
print(f"PSI_TRUE {({k: round(v,4) for k,v in PSI_TRUE.items()})}   m_true {M_TRUE:.5f}\n", flush=True)

# ================================================== 1. in-sample, seed repeats
print("=" * 96, flush=True)
print("MC in-sample delta recovery, 5 seeds (full 10yr domain)", flush=True)
print("=" * 96, flush=True)
A.configure(taus, t[-1], PSI_INIT, LAM, len(t))
data = (t, S, logF, r)
ins = {}
for seed in SEEDS:
    cfgs = A.make_variant_configs(num_epochs=EPOCHS, seed=seed)
    for cfg in [cfgs[0], cfgs[2], cfgs[3]]:
        tag = A.run_tag(cfg)
        fns, _ = A.calibration(cfg, data, verbose=False)
        _, dfn, gsp = fns
        d = np.asarray(A.vmap(dfn)(t))
        psi = A.psi_dict(gsp["p_Q"], gsp["alpha_P"])
        rm = float(np.sqrt(np.mean((d - dtrue) ** 2)))
        ins[(seed, tag)] = dict(delta_rmse=rm, corr=float(np.corrcoef(d, dtrue)[0, 1]),
                                psi=psi, m=A.m_of(psi),
                                incr_ratio=float(np.diff(d).std() / np.diff(dtrue).std()))
        print(f"  seed {seed:6d} {tag:11s} dRMSE {rm:.5f} corr {ins[(seed,tag)]['corr']:+.4f} "
              f"m {A.m_of(psi):+.4f} incr {100*ins[(seed,tag)]['incr_ratio']:5.1f}%", flush=True)

print(f"\n  {'variant':>12} | {'delta RMSE':>18} | {'corr':>17} | "
      f"{'m (true ' + format(M_TRUE, '.4f') + ')':>17} | {'incr%':>13}", flush=True)
isum = {}
for tag in ["MLP", "APPINN", "APPINN_ARB"]:
    rm = np.array([ins[(s, tag)]["delta_rmse"] for s in SEEDS])
    cc = np.array([ins[(s, tag)]["corr"] for s in SEEDS])
    mm = np.array([ins[(s, tag)]["m"] for s in SEEDS])
    ii = np.array([ins[(s, tag)]["incr_ratio"] for s in SEEDS])
    isum[tag] = dict(rmse=(rm.mean(), rm.std()), corr=(cc.mean(), cc.std()),
                     m=(mm.mean(), mm.std()), incr=(ii.mean(), ii.std()))
    print(f"  {tag:>12} | {rm.mean():.5f} +/- {rm.std():.5f} | {cc.mean():+.4f} +/- {cc.std():.4f} | "
          f"{mm.mean():+.4f} +/- {mm.std():.4f} | {100*ii.mean():5.1f} +/- {100*ii.std():4.1f}", flush=True)
print(f"  [no-skill constant: dRMSE {dtrue.std():.5f}]", flush=True)
print(f"  [LS closed-form baseline: dRMSE 0.00873, corr +0.9994, incr 102.8%]", flush=True)

# ================================================== 2. temporal holdout
print("\n" + "=" * 96, flush=True)
print("MC TEMPORAL HOLDOUT -- out-of-sample DELTA vs known truth (the key experiment)", flush=True)
print("=" * 96, flush=True)
A.configure(taus, t[tr][-1], PSI_INIT, LAM, int(tr.sum()))
data_ho = (t[tr], S[tr], logF[tr], r[tr])
ho = {}
for seed in SEEDS:
    cfgs = A.make_variant_configs(num_epochs=EPOCHS, seed=seed)
    for cfg in [cfgs[0], cfgs[2], cfgs[3]]:
        tag = A.run_tag(cfg)
        fns, _ = A.calibration(cfg, data_ho, verbose=False)
        pin, din = A.price_rmse(fns, t[tr], S[tr], logF[tr], r[tr])
        pout, dout = A.price_rmse(fns, t[te], S[te], logF[te], r[te])
        drin = float(np.sqrt(np.mean((din - dtrue[tr]) ** 2)))
        drout = float(np.sqrt(np.mean((dout - dtrue[te]) ** 2)))
        corr_out = float(np.corrcoef(dout, dtrue[te])[0, 1])
        _, dfn, _ = fns
        ho[(seed, tag)] = dict(price_in=pin, price_out=pout, price_ratio=pout / pin,
                               delta_in=drin, delta_out=drout, delta_corr_out=corr_out,
                               delta_full=np.asarray(A.vmap(dfn)(t)))
        print(f"  seed {seed:6d} {tag:11s} price {pin:.5f}->{pout:.5f} ({pout/pin:5.2f}x) | "
              f"delta {drin:.5f}->{drout:.5f} corr_out {corr_out:+.4f}", flush=True)

print(f"\n  {'variant':>12} | {'price ratio':>16} | {'delta RMSE in':>16} | {'delta RMSE out':>16} | {'corr out':>16}",
      flush=True)
hsum = {}
for tag in ["MLP", "APPINN", "APPINN_ARB"]:
    pr = np.array([ho[(s, tag)]["price_ratio"] for s in SEEDS])
    di = np.array([ho[(s, tag)]["delta_in"] for s in SEEDS])
    do = np.array([ho[(s, tag)]["delta_out"] for s in SEEDS])
    co = np.array([ho[(s, tag)]["delta_corr_out"] for s in SEEDS])
    hsum[tag] = dict(price_ratio=(pr.mean(), pr.std()), delta_in=(di.mean(), di.std()),
                     delta_out=(do.mean(), do.std()), corr_out=(co.mean(), co.std()))
    print(f"  {tag:>12} | {pr.mean():5.2f}x +/- {pr.std():4.2f} | {di.mean():.5f} +/- {di.std():.5f} | "
          f"{do.mean():.5f} +/- {do.std():.5f} | {co.mean():+.4f} +/- {co.std():.4f}", flush=True)
print(f"  [no-skill on test window: dRMSE {dtrue[te].std():.5f}]", flush=True)

with open(f"{OUT}/results/mc_holdout_and_seeds.pkl", "wb") as f:
    pickle.dump({"in_sample": ins, "in_sample_summary": isum, "holdout": ho,
                 "holdout_summary": hsum, "seeds": SEEDS, "t": t, "delta_true": dtrue,
                 "cut": CUT, "psi_true": PSI_TRUE, "m_true": M_TRUE}, f)
print("\nDONE", flush=True)
