"""
Validation harness for the Gibson-Schwartz synthetic simulator.

Before any synthetic data is used to evaluate the PINN inversion, these checks
establish that the simulator is internally consistent and reproduces the
analytic properties of the model. Each check is a guard against a specific
class of bug.

Checks
------
1. Initial condition of the closed form: F(0, S, delta) = S exactly.
2. PDE residual: the closed form substituted into GS-PDE-tau via autodiff is
   zero to floating-point tolerance -- the definitive check that A(tau), B(tau)
   are correct.
3. Convenience-yield mean-reversion statistics match O-U theory: mean -> alpha_P
   and variance -> sigma2^2/(2 kappa).
4. Term-structure shape: backwardation when delta is high, contango when low,
   consistent with B(tau) < 0.

Closed-form verification needs double precision: the PDE residual floor in
float32 is ~1e-5 (worth remembering for float32 PINN training), whereas in
float64 it drops to machine epsilon, confirming the analytic coefficients.
"""

from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from gs_wamol.physics.gibson_schwartz import GSParams, futures, log_futures
from gs_wamol.data.paths import PhysicalParams, simulate_paths


# Representative oil-like parameters (loosely from Schwartz Table VI).
@pytest.fixture
def p_Q():
    return GSParams(r=0.05, kappa=1.5, alpha_Q=0.10,
                    sigma1=0.35, sigma2=0.40, rho=0.80)


@pytest.fixture
def p_P():
    return PhysicalParams(mu=0.12, kappa=1.5, alpha_P=0.15,
                          sigma1=0.35, sigma2=0.40, rho=0.80)


def test_initial_condition(p_Q):
    """F(tau=0, S, delta) must equal S for all states."""
    n = 2000
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    S = jax.random.uniform(k1, (n,), minval=10.0, maxval=120.0)
    delta = jax.random.uniform(k2, (n,), minval=-0.3, maxval=0.5)
    F0 = futures(jnp.zeros(n), S, delta, p_Q)
    max_abs_err = float(jnp.max(jnp.abs(F0 - S)))
    assert max_abs_err < 1e-10, f"max|F(0,S,d) - S| = {max_abs_err:.3e}"


def test_pde_residual(p_Q):
    """Autodiff PDE residual of the closed form against GS-PDE-tau.

    GS-PDE-tau:
       F_tau = (r - delta) S F_S + kappa (alpha_Q - delta) F_delta
               + 1/2 sigma1^2 S^2 F_SS + rho sigma1 sigma2 S F_Sdelta
               + 1/2 sigma2^2 F_deltadelta

    Should be ~1e-10 or smaller if A(tau), B(tau) are exactly right.
    """
    n = 2000
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
    S = jax.random.uniform(k1, (n,), minval=10.0, maxval=120.0)
    delta = jax.random.uniform(k2, (n,), minval=-0.3, maxval=0.5)
    tau = jax.random.uniform(k3, (n,), minval=1e-3, maxval=2.0)

    def F_scalar(tau_, S_, d_):
        return futures(tau_, S_, d_, p_Q)

    # First derivatives
    F_tau = jax.vmap(jax.grad(F_scalar, argnums=0))(tau, S, delta)
    F_S = jax.vmap(jax.grad(F_scalar, argnums=1))(tau, S, delta)
    F_d = jax.vmap(jax.grad(F_scalar, argnums=2))(tau, S, delta)

    # Second derivatives
    F_SS = jax.vmap(jax.grad(jax.grad(F_scalar, argnums=1), argnums=1))(tau, S, delta)
    F_dd = jax.vmap(jax.grad(jax.grad(F_scalar, argnums=2), argnums=2))(tau, S, delta)
    F_Sd = jax.vmap(jax.grad(jax.grad(F_scalar, argnums=1), argnums=2))(tau, S, delta)

    rhs = ((p_Q.r - delta) * S * F_S
           + p_Q.kappa * (p_Q.alpha_Q - delta) * F_d
           + 0.5 * p_Q.sigma1 ** 2 * S ** 2 * F_SS
           + p_Q.rho * p_Q.sigma1 * p_Q.sigma2 * S * F_Sd
           + 0.5 * p_Q.sigma2 ** 2 * F_dd)

    residual = float(jnp.max(jnp.abs(F_tau - rhs)))
    assert residual < 1e-10, f"max|F_tau - RHS| = {residual:.3e}"


def test_mean_reversion(p_P):
    """Compare empirical delta statistics to O-U theory.

    Terminal ensemble mean should approach alpha_P (for T >> 1/kappa) and the
    terminal variance should approach the stationary variance sigma2^2/(2 kappa).
    """
    T = 10.0
    _, _, delta = simulate_paths(jax.random.PRNGKey(2), p_P, S0=50.0,
                                 delta0=-0.10, T=T, n_steps=2520, n_paths=4000)
    delta_T = delta[:, -1]
    emp_mean = float(jnp.mean(delta_T))
    emp_var = float(jnp.var(delta_T))
    theo_var = float(p_P.sigma2 ** 2 / (2.0 * p_P.kappa)
                     * (1.0 - jnp.exp(-2 * p_P.kappa * T)))

    # Monte Carlo tolerance with 4000 paths: a few percent on the mean,
    # ~10% on the variance is comfortable.
    assert emp_mean == pytest.approx(p_P.alpha_P, abs=0.02)
    assert emp_var == pytest.approx(theo_var, rel=0.10)


def test_term_structure_shape(p_Q):
    """Curve slopes down (backwardation) for high delta, up for low delta.

    Since B(tau) < 0, ln F = ln S + B(tau) delta + A(tau); the delta-dependence
    of the short-end slope in tau is driven by B'(tau) delta.
    """
    taus = jnp.linspace(0.05, 2.0, 40)
    S0 = 50.0
    hi = jax.vmap(lambda t: log_futures(t, S0, 0.40, p_Q))(taus)
    lo = jax.vmap(lambda t: log_futures(t, S0, -0.10, p_Q))(taus)
    hi_slope = float(hi[1] - hi[0])   # short-end slope
    lo_slope = float(lo[1] - lo[0])
    assert hi_slope < 0.0, f"expected backwardation (slope<0), got {hi_slope:.3e}"
    assert lo_slope > 0.0, f"expected contango (slope>0), got {lo_slope:.3e}"
