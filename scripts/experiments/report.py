"""Fix 5/6: consolidated reporting.

Everything reported in the IDENTIFIED coordinates (kappa, sigma2, m) alongside
the raw psi, with a null baseline next to every agreement metric. Reads
whatever exists on disk; skips sections whose inputs are not there yet.
"""
import os, pickle, glob
import numpy as np
import pandas as pd

REPO = "/Users/evanlynch/Developer/DC-PINNs"


def m_of(p):
    return float(p["alpha_Q"] + p["sigma1"] * p["sigma2"] * p["rho"] / p["kappa"])


def load(p):
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def hdr(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


PSI_TRUE = dict(kappa=1.876, sigma1=0.393, sigma2=0.527, rho=0.766,
                alpha_Q=0.07790831556503197, alpha_P=0.106)
M_TRUE = m_of(PSI_TRUE)

# ============================================================ 1. synthetic
hdr("TABLE 1 -- SYNTHETIC: parameter recovery in raw vs identified coordinates")
dv = load(f"{REPO}/data/output/mc_simulated_single_net/results/delta_recovery_all_variants.pkl")
ls = load(f"{REPO}/data/output/ls_baseline/results/ls_baseline.pkl")
if dv:
    dtrue = dv["delta_true"]
    print(f"{'method':>14} | {'kappa':>7} {'sigma2':>7} {'m':>9} | {'sigma1':>7} {'rho':>7} {'alphaQ':>8} |"
          f" {'dRMSE':>7} {'corr':>7} {'incr%':>6}")
    print("-" * 100)
    print(f"{'TRUE':>14} | {PSI_TRUE['kappa']:7.4f} {PSI_TRUE['sigma2']:7.4f} {M_TRUE:9.5f} | "
          f"{PSI_TRUE['sigma1']:7.4f} {PSI_TRUE['rho']:7.4f} {PSI_TRUE['alpha_Q']:8.4f} | "
          f"{'-':>7} {'-':>7} {'100.0':>6}")
    if ls:
        L = ls["mc"]
        print(f"{'LS closed-form':>14} | {L['psi']['kappa']:7.4f} {L['psi']['sigma2']:7.4f} "
              f"{L['psi']['m']:9.5f} | {'n/a':>7} {'n/a':>7} {'n/a':>8} | "
              f"{L['delta_rmse']:7.5f} {L['corr']:+7.4f} "
              f"{100*np.diff(L['delta_ls']).std()/np.diff(L['delta_true']).std():6.1f}")
    for tag, v in dv["variants"].items():
        p = v["psi_hat"]
        print(f"{tag:>14} | {p['kappa']:7.4f} {p['sigma2']:7.4f} {m_of(p):9.5f} | "
              f"{p['sigma1']:7.4f} {p['rho']:7.4f} {p['alpha_Q']:8.4f} | "
              f"{v['rmse']:7.5f} {np.corrcoef(v['delta_hat'], dtrue)[0,1]:+7.4f} "
              f"{100*np.diff(v['delta_hat']).std()/np.diff(dtrue).std():6.1f}")
    print(f"{'NULL (const)':>14} | {'-':>7} {'-':>7} {'-':>9} | {'-':>7} {'-':>7} {'-':>8} | "
          f"{dtrue.std():7.5f} {0.0:+7.4f} {0.0:6.1f}")

# ============================================================ 2. repeats
hdr("TABLE 2 -- 20-PATH REPEATS: identified vs unidentified quantities")
for lbl, path in [("single-net, noise 0.01", "mc_simulated_single_net/results/path_repeat.pkl"),
                  ("single-net, noise 0.10", "mc_simulated_single_net/results/path_repeat_noise0.10.pkl"),
                  ("dual-net,   noise 0.01", "mc_simulated/results/path_repeat.pkl")]:
    d = load(f"{REPO}/data/output/{path}")
    if not d:
        continue
    pt = d["psi_true"]
    rows = {"kappa": [], "sigma2": [], "P=s1s2rho": [], "m": [], "sigma1": [], "rho": []}
    for r in d["results"]:
        p = r["psi_hat"]
        rows["kappa"].append(p["kappa"]); rows["sigma2"].append(p["sigma2"])
        rows["sigma1"].append(p["sigma1"]); rows["rho"].append(p["rho"])
        rows["P=s1s2rho"].append(p["sigma1"] * p["sigma2"] * p["rho"])
        rows["m"].append(m_of(p))
    truth = {"kappa": pt["kappa"], "sigma2": pt["sigma2"], "sigma1": pt["sigma1"],
             "rho": pt["rho"], "P=s1s2rho": pt["sigma1"] * pt["sigma2"] * pt["rho"],
             "m": m_of(pt)}
    print(f"\n  {lbl}   (delta RMSE {np.mean([r['delta_rmse'] for r in d['results']]):.5f} "
          f"+/- {np.std([r['delta_rmse'] for r in d['results']]):.5f})")
    print(f"    {'quantity':>11} {'true':>9} {'mean':>9} {'std':>9} {'CV':>8}  identified?")
    for k in ["kappa", "sigma2", "m", "sigma1", "rho", "P=s1s2rho"]:
        a = np.array(rows[k])
        ident = "YES" if k in ("kappa", "sigma2", "m") else "NO"
        print(f"    {k:>11} {truth[k]:9.4f} {a.mean():9.4f} {a.std():9.4f} "
              f"{a.std()/abs(a.mean()):7.1%}  {ident}")

# ============================================================ 3. real data
hdr("TABLE 3 -- REAL WTI: agreement with the Kalman filter, against a null baseline")
kf = load(f"{REPO}/data/output/kalman/results/results_kalman.pkl")
kfd = pd.DatetimeIndex(kf["dates"])
rows = []
for tag in ["MLP", "APPINN", "APPINN_ARB"]:
    d = load(f"{REPO}/data/output/real_data/results/results_{tag}.pkl")
    if d:
        rows.append((tag + " (unfloored)", d["psi"], d["delta_hat"], pd.DatetimeIndex(d["dates"])))
kfl = load(f"{REPO}/data/output/real_data/results/kappa_floor_comparison.pkl")
if kfl:
    for (kmin, tag), v in kfl["results"].items():
        if kmin > 0:
            rows.append((tag + " (kappa>=0.1)", v["psi"], v["delta_hat"], None))

# LS on real data
if ls and "real" in ls:
    rows.append(("LS closed-form", None, ls["real"]["delta_ls"], pd.DatetimeIndex(ls["real"]["dates"])))

# reference dates
ref_dates = None
for _, _, _, dts in rows:
    if dts is not None:
        ref_dates = dts
        break
common = kfd.intersection(ref_dates)
kdel = kf["delta_hat"][kfd.get_indexer(common)]
NULL = float(kdel.std())
pos = ref_dates.get_indexer(common)

print(f"{'method':>24} | {'kappa':>7} {'sigma2':>7} {'m':>9} | {'dmin':>7} {'dstd':>6} | "
      f"{'corr KF':>8} {'RMSE KF':>8}  vs null {NULL:.4f}")
print("-" * 108)
kfp = kf["psi_hat"]
kfpsi = {k: float(kfp[k]) for k in ["kappa", "sigma1", "sigma2", "rho", "alpha_Q"]}
print(f"{'KALMAN (benchmark)':>24} | {kfpsi['kappa']:7.4f} {kfpsi['sigma2']:7.4f} "
      f"{m_of(kfpsi):9.5f} | {kdel.min():+7.3f} {kdel.std():6.3f} | {'-':>8} {'-':>8}")
for tag, psi, dh, dts in rows:
    dc = dh[pos] if dts is not None else dh[pos]
    c = float(np.corrcoef(dc, kdel)[0, 1])
    rm = float(np.sqrt(np.mean((dc - kdel) ** 2)))
    verdict = "BEATS null" if rm < NULL else "** WORSE than null **"
    if psi:
        print(f"{tag:>24} | {psi['kappa']:7.4f} {psi['sigma2']:7.4f} {m_of(psi):9.5f} | "
              f"{dh.min():+7.3f} {dh.std():6.3f} | {c:+8.4f} {rm:8.4f}  {verdict}")
    else:
        pm = ls["real"]["psi"]
        print(f"{tag:>24} | {pm['kappa']:7.4f} {pm['sigma2']:7.4f} {pm['m']:9.5f} | "
              f"{dh.min():+7.3f} {dh.std():6.3f} | {c:+8.4f} {rm:8.4f}  {verdict}")

# who agrees with whom?
hdr("TABLE 3b -- PAIRWISE AGREEMENT: is the Kalman filter the outlier?")
series = {"KALMAN": kdel}
for tag, _, dh, dts in rows:
    series[tag] = dh[pos]
names = list(series)
print(f"{'':>24} " + " ".join(f"{n[:12]:>13}" for n in names))
for a in names:
    print(f"{a:>24} " + " ".join(
        f"{np.sqrt(np.mean((series[a]-series[b])**2)):13.4f}" for b in names))
print("\n  (entries are RMSE between each pair of delta_hat series)")

# crisis decomposition
crisis = (common >= "2020-03-01") & (common <= "2020-06-30")
print(f"\n  Excluding Mar-Jun 2020 ({crisis.sum()} of {len(common)} dates):")
for a in names:
    if a == "KALMAN":
        continue
    r_all = np.sqrt(np.mean((series[a] - kdel) ** 2))
    r_ex = np.sqrt(np.mean((series[a][~crisis] - kdel[~crisis]) ** 2))
    print(f"    {a:>22} vs KF: all {r_all:.4f}  excl-crisis {r_ex:.4f}  "
          f"(crisis-only {np.sqrt(np.mean((series[a][crisis]-kdel[crisis])**2)):.4f})")

# ============================================================ 4. holdouts
hdr("TABLE 4 -- TEMPORAL HOLDOUT, real WTI (5 seeds), with the LS decomposition")
h = load(f"{REPO}/data/output/real_data/results/holdout_seed_repeats.pkl")
if h:
    print(f"{'variant':>14} | {'in-sample':>19} | {'out-of-sample':>19} | {'ratio':>22}")
    print("-" * 84)
    for tag, s in h["summary"].items():
        print(f"{tag:>14} | {s['rmse_in'][0]:.5f} +/- {s['rmse_in'][1]:.5f} | "
              f"{s['rmse_out'][0]:.5f} +/- {s['rmse_out'][1]:.5f} | "
              f"{s['ratio'][0]:5.2f}x +/- {s['ratio'][1]:4.2f} "
              f"[{min(s['ratio_all']):.2f}-{max(s['ratio_all']):.2f}]")
if ls and "holdout" in ls:
    L = ls["holdout"]
    print(f"{'LS (delta re-solved)':>14} | {L['rmse_in']:.5f}{'':13} | {L['rmse_out_solve']:.5f}{'':13} | "
          f"{L['rmse_out_solve']/L['rmse_in']:5.2f}x   <- psi-generalization floor")
    print(f"{'LS (delta frozen)':>14} | {L['rmse_in']:.5f}{'':13} | {L['rmse_out_frozen']:.5f}{'':13} | "
          f"{L['rmse_out_frozen']/L['rmse_in']:5.2f}x   <- cost of not extrapolating")

hdr("TABLE 5 -- MC TEMPORAL HOLDOUT (5 seeds): out-of-sample DELTA vs known truth")
mh = load(f"{REPO}/data/output/mc_simulated_single_net/results/mc_holdout_and_seeds.pkl")
if mh:
    print("  in-sample (full domain), 5 seeds:")
    print(f"    {'variant':>12} | {'delta RMSE':>19} | {'corr':>18} | {'m':>18} | {'incr%':>13}")
    for tag, s in mh["in_sample_summary"].items():
        print(f"    {tag:>12} | {s['rmse'][0]:.5f} +/- {s['rmse'][1]:.5f} | "
              f"{s['corr'][0]:+.4f} +/- {s['corr'][1]:.4f} | {s['m'][0]:+.4f} +/- {s['m'][1]:.4f} | "
              f"{100*s['incr'][0]:5.1f} +/- {100*s['incr'][1]:4.1f}")
    print(f"    {'LS':>12} | {ls['mc']['delta_rmse']:.5f}{'':13} | {ls['mc']['corr']:+.4f}{'':12} | "
          f"{ls['mc']['psi']['m']:+.4f}{'':12} | {100*np.diff(ls['mc']['delta_ls']).std()/np.diff(ls['mc']['delta_true']).std():5.1f}")
    print(f"\n  temporal holdout (train t<=8yr, test t>8yr), 5 seeds:")
    print(f"    {'variant':>12} | {'price ratio':>16} | {'dRMSE in':>17} | {'dRMSE out':>17} | {'corr out':>17}")
    for tag, s in mh["holdout_summary"].items():
        print(f"    {tag:>12} | {s['price_ratio'][0]:5.2f}x +/- {s['price_ratio'][1]:4.2f} | "
              f"{s['delta_in'][0]:.5f} +/- {s['delta_in'][1]:.5f} | "
              f"{s['delta_out'][0]:.5f} +/- {s['delta_out'][1]:.5f} | "
              f"{s['corr_out'][0]:+.4f} +/- {s['corr_out'][1]:.4f}")
    te = mh["t"] > mh["cut"]
    print(f"    [no-skill on test window: dRMSE {mh['delta_true'][te].std():.5f}]")

hdr("TABLE 6 -- FULL-PANEL REAL FIT, 5 seeds (kappa floored)")
fp = load(f"{REPO}/data/output/real_data/results/full_panel_seed_repeats.pkl")
if fp:
    print(f"{'variant':>14} | {'corr vs KF':>19} | {'RMSE vs KF':>19} | {'kappa':>17} | {'m':>17}")
    for tag, s in fp["summary"].items():
        print(f"{tag:>14} | {s['corr'][0]:+.4f} +/- {s['corr'][1]:.4f} | "
              f"{s['rmse'][0]:.4f} +/- {s['rmse'][1]:.4f} | {s['kappa'][0]:.3f} +/- {s['kappa'][1]:.3f} | "
              f"{s['m'][0]:+.3f} +/- {s['m'][1]:.3f}")
    print(f"  [null = constant at KF mean: RMSE {fp['null_rmse']:.4f}]")
print()
