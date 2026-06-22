"""
End-to-end synthetic-data generation example.

Wires together the three layers:
    P-measure path simulation  ->  Q-measure closed-form pricing  ->  noisy panel

and emits a dataset dict ready for PINN training / Kalman benchmarking. The true
latent delta_t path is retained under 'delta_true' as the evaluation target.

Run:
    JAX_ENABLE_X64=1 python3 -m gs_wamol.data.generate
"""

from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gs_wamol.physics.gibson_schwartz import GSParams
from gs_wamol.data.paths import PhysicalParams, simulate_paths
from gs_wamol.data.observe import ObservationConfig, build_panel


def generate(seed: int = 0):
    # --- Parameters -------------------------------------------------------
    # P and Q share kappa, sigma1, sigma2, rho. They differ in:
    #   - spot drift:        mu (P) vs r (Q)
    #   - long-run CY level: alpha_P (P) vs alpha_Q (Q)
    # alpha_Q = alpha_P - sigma2 * lambda2 / kappa  (the measure-change shift).
    kappa, sigma1, sigma2, rho = 1.5, 0.35, 0.40, 0.80
    r, mu = 0.05, 0.12
    alpha_P = 0.15
    lambda2 = 0.10                       # market price of CY risk (illustrative)
    alpha_Q = alpha_P - sigma2 * lambda2 / kappa

    p_Q = GSParams(r=r, kappa=kappa, alpha_Q=alpha_Q,
                   sigma1=sigma1, sigma2=sigma2, rho=rho)
    p_P = PhysicalParams(mu=mu, kappa=kappa, alpha_P=alpha_P,
                         sigma1=sigma1, sigma2=sigma2, rho=rho)

    # --- Simulation grid --------------------------------------------------
    key = jax.random.PRNGKey(seed)
    k_paths, k_noise = jax.random.split(key)

    t_grid, S, delta = simulate_paths(
        k_paths, p_P,
        S0=50.0, delta0=alpha_P,
        T=10.0, n_steps=2520,            # ~daily over 10y
        n_paths=64,
    )

    # --- Observable panel -------------------------------------------------
    # CL1..CL12-style liquid maturities, in years.
    taus = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) / 12.0
    cfg = ObservationConfig(taus=taus, noise_std=0.005)   # ~0.5% log-price noise

    data = build_panel(k_noise, t_grid, S, delta, p_Q, cfg)
    data['params_Q'] = p_Q
    data['params_P'] = p_P
    data['alpha_Q'] = alpha_Q
    data['lambda2'] = lambda2
    return data


if __name__ == "__main__":
    d = generate()
    print("=== Synthetic dataset generated ===")
    print(f"paths           : {d['S'].shape[0]}")
    print(f"dates per path  : {d['S'].shape[1]}")
    print(f"maturities (M)  : {d['taus'].shape[0]}  -> {[f'{t:.2f}' for t in d['taus']]}")
    print(f"panel shape     : {d['log_F_obs'].shape}  (paths, dates, maturities)")
    print(f"delta_true shape: {d['delta_true'].shape}  (retained as eval target)")
    print(f"alpha_Q         : {d['alpha_Q']:.4f}  (from alpha_P={d['params_P'].alpha_P}, "
          f"lambda2={d['lambda2']})")
    # quick sanity: clean vs observed RMS gap should be ~ noise_std
    gap = float(jnp.sqrt(jnp.mean((d['log_F_obs'] - d['log_F_clean'])**2)))
    print(f"obs-clean RMS   : {gap:.4f}  (target ~ noise_std = 0.005)")
