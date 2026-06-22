"""
Physical-measure (P) path simulation for Gibson-Schwartz Model 2.

Simulates the *real-world* joint dynamics of the spot price S_t and the latent
convenience yield delta_t via correlated Euler-Maruyama:

    dS_t     = (mu - delta_t) S_t dt + sigma1 S_t dW1^P
    ddelta_t = kappa (alpha_P - delta_t) dt + sigma2 dW2^P
    dW1 dW2  = rho dt

The crucial output for PINN validation is the *true latent delta_t path*, which
is unobservable from real market data and serves as the ground-truth target the
inversion is evaluated against.

Design choice (Option A architecture): paths are simulated under P here; futures
prices are then computed from these paths using the *closed-form* Q-measure
pricer (physics/gibson_schwartz.py), NOT by nested Monte Carlo. The closed form
gives exact (noise-free) prices conditional on the simulated state, so any
pricing "error" is introduced deliberately and controllably in the observation
layer.

Notes on log-spot integration
------------------------------
S_t is simulated in log space (X = ln S) to (a) guarantee positivity and
(b) remove the multiplicative-noise discretisation bias of a naive E-M on S.
Applying Ito to X = ln S under P:

    dX = (mu - delta - 1/2 sigma1^2) dt + sigma1 dW1^P

which is exact-in-drift for the log and integrates cleanly with E-M.
"""

from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PhysicalParams:
    """Physical-measure parameters for path simulation.

    mu       : real-world spot drift (total expected return on the spot)
    kappa    : convenience-yield mean-reversion speed (shared with Q)
    alpha_P  : physical long-run convenience yield (alpha^P)
    sigma1   : spot volatility (shared with Q)
    sigma2   : convenience-yield volatility (shared with Q)
    rho      : Brownian correlation (shared with Q)

    Only the long-run level (alpha_P) and the spot drift (mu) differ between the
    P and Q descriptions; kappa, sigma1, sigma2, rho are measure-invariant.
    """
    mu: float
    kappa: float
    alpha_P: float
    sigma1: float
    sigma2: float
    rho: float


def simulate_paths(key: jax.Array,
                   pp: PhysicalParams,
                   *,
                   S0: float,
                   delta0: float,
                   T: float,
                   n_steps: int,
                   n_paths: int):
    """Simulate correlated (S_t, delta_t) paths under P via Euler-Maruyama.

    Parameters
    ----------
    key      : JAX PRNG key
    pp       : PhysicalParams
    S0       : initial spot price
    delta0   : initial convenience yield
    T        : horizon in years
    n_steps  : number of time steps over [0, T]
    n_paths  : number of independent paths

    Returns
    -------
    t_grid : (n_steps + 1,) time grid
    S      : (n_paths, n_steps + 1) spot paths
    delta  : (n_paths, n_steps + 1) latent convenience-yield paths (GROUND TRUTH)

    The correlation is imposed by a 2x2 Cholesky factor of [[1, rho], [rho, 1]]
    applied to independent standard normals at each step.
    """
    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)
    t_grid = jnp.linspace(0.0, T, n_steps + 1)

    # Cholesky of the correlation matrix [[1, rho],[rho, 1]].
    # L = [[1, 0], [rho, sqrt(1 - rho^2)]]; correlated = L @ iid.
    rho = pp.rho
    L21 = rho
    L22 = jnp.sqrt(1.0 - rho ** 2)

    X0 = jnp.log(S0)
    half_s1sq = 0.5 * pp.sigma1 ** 2

    # Draw all increments up front: shape (n_paths, n_steps, 2).
    z = jax.random.normal(key, (n_paths, n_steps, 2))
    # Correlate: dW1 = z0, dW2 = rho z0 + sqrt(1-rho^2) z1, scaled by sqrt(dt).
    dW1 = z[..., 0] * sqrt_dt
    dW2 = (L21 * z[..., 0] + L22 * z[..., 1]) * sqrt_dt

    def step(carry, dWs):
        X, d = carry            # log-spot, convenience yield (each (n_paths,))
        dW1_t, dW2_t = dWs
        # Convenience yield: O-U Euler-Maruyama under P.
        d_next = d + pp.kappa * (pp.alpha_P - d) * dt + pp.sigma2 * dW2_t
        # Log-spot: drift uses the *current* delta (explicit E-M).
        X_next = X + (pp.mu - d - half_s1sq) * dt + pp.sigma1 * dW1_t
        return (X_next, d_next), (X_next, d_next)

    # scan over the time axis; dWs feed in as (n_steps, n_paths) per channel.
    dW1_T = dW1.T   # (n_steps, n_paths)
    dW2_T = dW2.T
    init = (jnp.full((n_paths,), X0), jnp.full((n_paths,), delta0))
    _, (X_path, d_path) = jax.lax.scan(step, init, (dW1_T, dW2_T))

    # X_path, d_path are (n_steps, n_paths); prepend initial state and transpose.
    X_path = jnp.concatenate([jnp.full((1, n_paths), X0), X_path], axis=0).T
    d_path = jnp.concatenate([jnp.full((1, n_paths), delta0), d_path], axis=0).T

    S = jnp.exp(X_path)
    return t_grid, S, d_path
