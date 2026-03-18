"""
Pricing model physics.

Currently implements Black-Scholes / local-vol formulas used by the
SABR-synthetic calibration experiments.

TODO: Extend / replace with Gibson-Schwartz (1990) two-factor commodity
      model: closed-form futures price and the corresponding PDE residual
      under the GS convenience-yield dynamics.
"""

import jax.numpy as jnp
from jax import grad, vmap
import jax.scipy.stats as jss

eps = 1e-15

N       = jss.norm.cdf
N_prime = jss.norm.pdf
N_inv   = jss.norm.ppf


# ---------------------------------------------------------------------------
# Numerical safeguards
# ---------------------------------------------------------------------------

def bound(x):
    return jnp.maximum(1e-15, x)


# ---------------------------------------------------------------------------
# Black-Scholes building blocks
# ---------------------------------------------------------------------------

def d1_(s, k, r, sigma, tau):
    s, k, tau, sigma = bound(s), bound(k), bound(tau), bound(sigma)
    return (jnp.log(s / k) + (r + sigma * sigma / 2) * tau) / (sigma * jnp.sqrt(tau))


def d2_(s, k, r, sigma, tau):
    s, k, tau, sigma = bound(s), bound(k), bound(tau), bound(sigma)
    return d1_(s, k, r, sigma, tau) - sigma * jnp.sqrt(tau)


def bs(s, k, tau, r, cp, sigma):
    """Black-Scholes call/put price."""
    s, k, tau, sigma = bound(s), bound(k), bound(tau), bound(sigma)
    d1 = d1_(s, k, r, sigma, tau)
    d2 = d2_(s, k, r, sigma, tau)
    return cp * s * N(cp * d1) - cp * k * jnp.exp(-r * tau) * N(cp * d2)


def black(fwd, k, tau, r, cp, sigma):
    """Black-76 price (forward as underlying)."""
    fwd, k, tau, sigma = bound(fwd), bound(k), bound(tau), bound(sigma)
    return bs(fwd, k, tau, 0.0, cp, sigma) * jnp.exp(-r * tau)


def bs_vega(s, k, tau, r, sigma):
    return s * jnp.sqrt(tau) * N_prime(d1_(s, k, r, sigma, tau))


def bs_iv(C, s, k, tau, cp, r=0.0,
          tol=1e-15, tol_vega=1e-15, ini=0.5, thr=2.0, max_it=20):
    """Newton-Raphson implied vol inversion."""
    sigma = ini * jnp.ones_like(C)
    for _ in range(max_it):
        diff = bs(s, k, tau, r, cp, sigma) - C
        vega = bs_vega(s, k, tau, r, sigma)
        end = jnp.logical_or(jnp.abs(diff) < tol, vega < tol_vega)
        sigma = (sigma - diff / vega) * jnp.logical_not(end) + sigma * end
    return jnp.minimum(jnp.maximum(sigma, 0.0), thr)


# ---------------------------------------------------------------------------
# Local-volatility (Dupire / Gatheral total-variance form)
# ---------------------------------------------------------------------------

def lv_var(vol, tau):
    return (vol ** 2) * tau


def lv_fwd_sqr(k, w, tau, dk_w, d2k_w, dt_w):
    """Local-vol squared in forward log-strike / total-variance coordinates.

    k := ln(K/F), w := sigma_BS^2 * tau
    Reference: https://quant.stackexchange.com/questions/16343
    """
    w = bound(w)
    A = -k / w * dk_w + 0.25 * (-0.25 - 1 / w + (k ** 2) / (w ** 2)) * (dk_w ** 2)
    return dt_w / (1.0 + A + 0.5 * d2k_w)


def lv_fwd_pde(lv_fwd_sq, k, d2k_v, dt_v):
    """Dupire PDE residual in forward coordinates."""
    return dt_v - 0.5 * lv_fwd_sq * (k ** 2) * d2k_v


def lv_sqr(s, k, r, v, tau, dk_v, d2k_v, dtau_v):
    """Local-vol squared from BS implied vol and its derivatives."""
    A = v ** 2 + 2.0 * v * tau * (dtau_v + r * k * dk_v)
    y = jnp.log(k / (jnp.exp(r * tau) * s))
    B = (1 - k * y / v * dk_v) ** 2
    C = k * v * tau * (dk_v - 0.25 * k * v * tau * (dk_v ** 2) + k * d2k_v)
    D = B + C + (B + C == 0.0) * eps
    return A / D


# ---------------------------------------------------------------------------
# Automatic-differentiation helpers
# ---------------------------------------------------------------------------

def derivatives(fn, x):
    """Compute (dK, d2K, dT) of fn w.r.t. inputs x = [K, T]."""
    def f(x):   return fn(x)[0]
    def f_dK(x): return grad(f)(x)[0]

    dx  = vmap(grad(f), 0)(x)
    d2x = vmap(grad(f_dK), 0)(x)
    dx1, d2x1, dx2 = dx.T[0], d2x.T[0], dx.T[1]
    return dx1, d2x1, dx2


def call_derivatives(fn, x, s_0, r):
    """Compute (dK, d2K, dT) of the BS call price implied by fn's vol output."""
    def f(x):    return bs(s_0, x.T[0], x.T[1], r, 1, fn(x).flatten())[0]
    def f_dK(x): return grad(f)(x)[0]

    dx  = vmap(grad(f), 0)(x)
    d2x = vmap(grad(f_dK), 0)(x)
    dx1, d2x1, dx2 = dx.T[0], d2x.T[0], dx.T[1]
    return dx1, d2x1, dx2
