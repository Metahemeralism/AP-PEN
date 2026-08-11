"""Local identifiability analysis of the GS closed form from a futures term structure.

Question: given only log F(tau_k) = log S + B(tau)delta + A(tau) observed on a
maturity grid, which directions in psi = (kappa, sigma1, sigma2, rho, alpha_Q)
are actually pinned down, and which are flat?

Method: build the Jacobian of the observation vector w.r.t. psi (and delta),
then SVD it. Singular values near zero = flat directions = non-identified
combinations. This is the standard local (Fisher-information) identifiability
test; it says nothing about global identifiability but is exactly what an
optimiser feels.
"""
import os, pickle
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # CLAUDE.md S5: never do this in float32
os.chdir("/Users/evanlynch/Developer/DC-PINNs")

PSI_TRUE = dict(kappa=1.876, sigma1=0.393, sigma2=0.527, rho=0.766,
                alpha_Q=0.07790831556503197, alpha_P=0.106)
R = 0.05


def B_coeff(tau, kappa):
    return -(1.0 - jnp.exp(-kappa * tau)) / kappa


def A_coeff(tau, kappa, sigma1, sigma2, rho, alpha_Q, r=R):
    s1s2rho = sigma1 * sigma2 * rho
    s2sq = sigma2 ** 2
    lin = (r - alpha_Q + 0.5 * s2sq / kappa ** 2 - s1s2rho / kappa) * tau
    two = 0.25 * s2sq * (1.0 - jnp.exp(-2.0 * kappa * tau)) / kappa ** 3
    one = (alpha_Q * kappa + s1s2rho - s2sq / kappa) * (1.0 - jnp.exp(-kappa * tau)) / kappa ** 2
    return lin + two + one


# MC maturity grid (12 maturities, 1 month to 1 year), matching mc_data.pkl
with open("data/input/synthetic/mc_data.pkl", "rb") as f:
    mc = pickle.load(f)
TAUS_MC = jnp.asarray(mc["taus"])
print("MC taus:", np.round(np.asarray(TAUS_MC), 4))
print("noise std in mc_data:", mc.get("noise_std", "<not stored>"), " keys:", list(mc.keys()))

# real WTI tau grid (8 slots) for comparison
TAUS_REAL = jnp.asarray([0.1247, 0.2, 0.3, 0.45, 0.7, 1.0, 1.4, 1.8756])


def obs_vector(theta, taus, delta):
    """log F - log S at one date, as a function of the 5 Q-parameters."""
    kappa, sigma1, sigma2, rho, alpha_Q = theta
    return B_coeff(taus, kappa) * delta + A_coeff(taus, kappa, sigma1, sigma2, rho, alpha_Q)


def analyse(taus, delta, label, names=("kappa", "sigma1", "sigma2", "rho", "alpha_Q")):
    theta0 = jnp.array([PSI_TRUE["kappa"], PSI_TRUE["sigma1"], PSI_TRUE["sigma2"],
                        PSI_TRUE["rho"], PSI_TRUE["alpha_Q"]])
    J = jax.jacfwd(lambda th: obs_vector(th, taus, delta))(theta0)   # (K, 5)
    J = np.asarray(J)
    # scale columns by the parameter magnitude -> relative sensitivity, so the SVD
    # compares "1% change in kappa" against "1% change in rho" rather than raw units
    scale = np.abs(np.asarray(theta0))
    Js = J * scale[None, :]
    U, s, Vt = np.linalg.svd(Js, full_matrices=False)
    print(f"\n{'='*78}\n{label}   (delta={delta}, K={len(taus)} maturities)\n{'='*78}")
    print("singular values (relative-scaled):", np.array2string(s, precision=3e0 and 4))
    print("condition number: %.3e" % (s[0] / s[-1]))
    print("\ndirections, weakest last:")
    for i in range(len(s)):
        v = Vt[i]
        terms = "  ".join(f"{n}:{v[j]:+.3f}" for j, n in enumerate(names))
        print(f"  sv={s[i]:9.3e}  |  {terms}")
    return s, Vt


# ---- 1. single date, MC grid -------------------------------------------------
analyse(TAUS_MC, 0.0, "MC maturity grid, single date, delta=0")
analyse(TAUS_REAL, 0.0, "Real WTI maturity grid (8 slots), single date, delta=0")

# ---- 2. is sigma1 x rho the flat direction? ---------------------------------
print("\n" + "=" * 78)
print("DIRECT TEST: does the price depend on sigma1,rho only via the product s1*s2*rho?")
print("=" * 78)
base = A_coeff(TAUS_MC, PSI_TRUE["kappa"], PSI_TRUE["sigma1"], PSI_TRUE["sigma2"],
               PSI_TRUE["rho"], PSI_TRUE["alpha_Q"])
# hold the product fixed, move sigma1 and rho in opposite directions
for f in [0.5, 0.8, 1.25, 2.0]:
    s1 = PSI_TRUE["sigma1"] * f
    rho = PSI_TRUE["rho"] / f
    alt = A_coeff(TAUS_MC, PSI_TRUE["kappa"], s1, PSI_TRUE["sigma2"], rho, PSI_TRUE["alpha_Q"])
    print(f"  sigma1 x{f:<5} ={s1:.4f}, rho /{f:<5} ={rho:.4f}  ->  "
          f"max|dA| = {float(jnp.max(jnp.abs(alt - base))):.3e}   (rho valid: {abs(rho) < 1})")

# ---- 3. the alpha_Q = alpha_P - sigma2*lambda2/kappa pole --------------------
print("\n" + "=" * 78)
print("THE 1/kappa POLE in the fixed-lambda2 reparameterisation")
print("=" * 78)
LAM = 0.4389990889936291      # the Kalman-fitted lambda used on real data
print(f"alpha_Q = alpha_P - sigma2*lambda2/kappa,  lambda2 = {LAM:.4f} (KF-fitted)")
print(f"{'kappa':>10} {'sigma2':>9} {'alpha_P':>9} -> {'sigma2*lam/kappa':>17} {'alpha_Q':>10}")
for kap, s2, aP in [(0.9301, 0.5444, 0.0120), (0.4257, 0.1726, 0.1007),
                    (0.1, 0.0557, 4.3741), (0.01, 0.0557, 4.3741),
                    (0.0027683686930686235, 0.05570824444293976, 4.374133110046387)]:
    shift = s2 * LAM / kap
    print(f"{kap:10.5f} {s2:9.4f} {aP:9.4f} -> {shift:17.4f} {aP - shift:10.4f}")

# ---- 4. what the collapsed real-data ARB solution actually prices ------------
print("\n" + "=" * 78)
print("WHAT THE COLLAPSED APPINN_ARB (real data) PRICING FUNCTION LOOKS LIKE")
print("=" * 78)
collapsed = dict(kappa=0.0027683686930686235, sigma1=0.005076534114778042,
                 sigma2=0.05570824444293976, rho=-0.8197784423828125,
                 alpha_Q=-4.459902763366699)
healthy = dict(kappa=0.930088222026825, sigma1=0.7660226821899414,
               sigma2=0.5444256067276001, rho=0.9675610661506653,
               alpha_Q=-0.24500508606433868)
kf = dict(kappa=0.42571276446419076, sigma1=0.4397451068598283,
          sigma2=0.17260585756996108, rho=0.8882750108441715, alpha_Q=-0.0772655658370586)
print(f"{'tau':>8} | {'B collapsed':>12} {'B APPINN':>10} {'B kalman':>10} | "
      f"{'A collapsed':>12} {'A APPINN':>10} {'A kalman':>10} | {'-tau':>8}")
for tau in [0.125, 0.25, 0.5, 1.0, 1.875]:
    row = [f"{tau:8.3f} |"]
    for p in (collapsed, healthy, kf):
        row.append(f"{float(B_coeff(tau, p['kappa'])):12.4f}")
    row.append("|")
    for p in (collapsed, healthy, kf):
        row.append(f"{float(A_coeff(tau, **p)):12.4f}")
    row.append(f"| {-tau:8.3f}")
    print(" ".join(row))
print("\nB(tau) -> -tau as kappa -> 0: the two-factor model degenerates to a")
print("deterministic cost-of-carry curve log F = log S + (r - delta_const)*tau.")
