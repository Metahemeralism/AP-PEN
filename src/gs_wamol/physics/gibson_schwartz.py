"""
Gibson-Schwartz Model 2: closed-form futures pricer.

Implements the exponential-affine futures price derived analytically:

    F_hat(tau, S, delta) = S * exp[ B(tau) * delta + A(tau) ]

with

    B(tau) = -(1 - e^{-kappa tau}) / kappa

    A(tau) = ( r - alpha_Q + (1/2) sigma2^2 / kappa^2 - sigma1 sigma2 rho / kappa ) tau
             + (1/4) sigma2^2 (1 - e^{-2 kappa tau}) / kappa^3
             + ( alpha_Q kappa + sigma1 sigma2 rho - sigma2^2 / kappa ) (1 - e^{-kappa tau}) / kappa^2
"""

from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GSParams:
    """Risk-neutral Gibson-Schwartz Model 2 parameters.

    Attributes
    ----------
    r        : constant risk-free rate
    kappa    : convenience-yield mean-reversion speed (kappa > 0)
    alpha_Q  : risk-neutral long-run convenience yield (alpha^Q)
    sigma1   : spot volatility
    sigma2   : convenience-yield volatility
    rho      : correlation between the two Brownian motions, in (-1, 1)

    Notes
    -----
    alpha_Q is the *risk-neutral* long-run level, alpha_Q = alpha_P - sigma2 lambda2 / kappa.
    For pricing only alpha_Q enters; the physical alpha_P and lambda2 are needed
    only when simulating the physical-measure paths (see data/paths.py).
    """
    r: float
    kappa: float
    alpha_Q: float
    sigma1: float
    sigma2: float
    rho: float


def B_coeff(tau: jnp.ndarray, kappa: float) -> jnp.ndarray:
    """B(tau) = -(1 - e^{-kappa tau}) / kappa.
    """
    return -(1.0 - jnp.exp(-kappa * tau)) / kappa


def A_coeff(tau: jnp.ndarray, p: GSParams) -> jnp.ndarray:
    """A(tau) for the Gibson-Schwartz Model 2 closed form.

    Written in the three-group form matching Schwartz (1997) eq. (20):
      - a tau-linear group,
      - a (1 - e^{-2 kappa tau}) group,
      - a (1 - e^{-kappa tau}) group.
    """
    kappa = p.kappa
    s1s2rho = p.sigma1 * p.sigma2 * p.rho
    s2sq = p.sigma2 ** 2

    lin = (p.r - p.alpha_Q
           + 0.5 * s2sq / kappa ** 2
           - s1s2rho / kappa) * tau

    two_kappa = 0.25 * s2sq * (1.0 - jnp.exp(-2.0 * kappa * tau)) / kappa ** 3

    one_kappa = (p.alpha_Q * kappa + s1s2rho - s2sq / kappa) \
        * (1.0 - jnp.exp(-kappa * tau)) / kappa ** 2

    return lin + two_kappa + one_kappa


def log_futures(tau: jnp.ndarray,
                S: jnp.ndarray,
                delta: jnp.ndarray,
                p: GSParams) -> jnp.ndarray:
    """ln F_hat(tau, S, delta) = ln S + B(tau) delta + A(tau)."""
    return jnp.log(S) + B_coeff(tau, p.kappa) * delta + A_coeff(tau, p)


def futures(tau: jnp.ndarray,
            S: jnp.ndarray,
            delta: jnp.ndarray,
            p: GSParams) -> jnp.ndarray:
    """F_hat(tau, S, delta) = S exp[B(tau) delta + A(tau)]."""
    return jnp.exp(log_futures(tau, S, delta, p))


# whole futures curve. vmap over the tau axis.
def term_structure(taus: jnp.ndarray,
                   S: jnp.ndarray,
                   delta: jnp.ndarray,
                   p: GSParams) -> jnp.ndarray:
    """Futures curve F(tau_i) for a fixed state (S, delta) across maturities."""
    return jax.vmap(lambda tau: futures(tau, S, delta, p))(taus)
