"""
Synthetic data generation via SABR Monte Carlo.

Generates option premium observations on a truncated-normal grid of
(strike, maturity) pairs, priced under the Hagan et al. (2002) SABR
approximation with additive lognormal vol noise.
"""

import math

import jax.numpy as jnp
import numpy as np
from scipy.stats import truncnorm

# Default SABR / market parameters (Cell 6 of original notebook)
DEFAULT_S0    = 1.0
DEFAULT_R     = 0.05
DEFAULT_ALPHA = 0.3
DEFAULT_BETA  = 0.7
DEFAULT_RHO   = -0.6
DEFAULT_NU    = 0.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_truncated_normal(mean=0.0, sd=1.0, low=0.0, upp=10.0):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def _black_np(F, K, T, r, sigma):
    """Numpy Black-76 call price (used only in data generation)."""
    from scipy.stats import norm
    if T <= 0.0 or K <= 0.0:
        return np.exp(-r * T) * max(F - K, 0.0)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def _chi(z, rho):
    if rho != 1.0:
        s = math.sqrt(1.0 - 2 * rho * z + z * z)
        return math.log((s + z - rho) / (1.0 - rho))
    return math.log(1.0 + z / abs(1.0 - z))


def sabr_vol_hagan(F, K, T, alpha, beta, nu, rho, h=0.0):
    """Hagan et al. (2002) SABR implied vol approximation."""
    if nu == 0.0 and rho == 0.0:
        return alpha
    if K <= 0.0:
        return 0.0

    sub_beta = 1.0 - beta
    if abs(F - K) > 1e-4:
        logFK = math.log(F / K)
        a0 = sub_beta ** 2 / 24 * logFK ** 2
        a1 = sub_beta ** 4 / 1920 * logFK ** 4
        c0 = (F * K) ** (sub_beta / 2) * (1 + a0 + a1)
        z  = nu / alpha * (F * K) ** (sub_beta / 2) * logFK
        c1 = z / _chi(z, rho) * math.log((F + h) / (K + h)) / logFK
        coef_vol = alpha / c0 * c1
    else:
        coef_vol = alpha * F ** beta / (F + h)

    FK_sb  = (F * K) ** sub_beta
    sqrtFK = math.sqrt(F * K)
    y0 = sub_beta ** 2 / 24.0 * alpha ** 2 / FK_sb
    y1 = 0.25 * (alpha * beta * rho * nu) / (F * K) ** (sub_beta / 2)
    y2 = (2 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    y3 = h * (2.0 * sqrtFK + h) * alpha ** 2
    y4 = 24.0 * (sqrtFK + h) ** 2 * FK_sb

    return coef_vol * (1 + (y0 + y1 + y2 - y3 / y4) * T)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def get_data(
    n_pts: int,
    n_h: int,
    s_0: float   = DEFAULT_S0,
    r: float     = DEFAULT_R,
    alpha: float = DEFAULT_ALPHA,
    beta: float  = DEFAULT_BETA,
    rho: float   = DEFAULT_RHO,
    nu: float    = DEFAULT_NU,
    vol_noise_std: float = 0.1,
    seed: int    = 0,
):
    """Generate (x_train, y_train, x_mesh) for IVS calibration.

    x  columns: [strike K, maturity T]
    y  columns: [Black-76 call premium]
    """
    rng = np.random.default_rng(seed)
    eps_val = 1e-3

    # Mesh for PDE / constraint evaluation
    Ks, Ts = np.meshgrid(
        np.linspace(eps_val, 2.5 + eps_val, n_h),
        np.linspace(eps_val, 5.0 + eps_val, n_h),
    )
    x_mesh = np.array([Ks.flatten(), Ts.flatten()]).T

    # Training points sampled from truncated normals
    K_samp = get_truncated_normal(mean=s_0, sd=0.4, low=eps_val, upp=2.5).rvs(n_pts)
    T_samp = get_truncated_normal(mean=eps_val, sd=2, low=eps_val, upp=5.0).rvs(n_pts)
    x_train = np.stack([K_samp, T_samp], axis=1)

    premiums = []
    for K, tau in x_train:
        F   = s_0 * np.exp(r * tau)
        vol = sabr_vol_hagan(F, K, tau, alpha, beta, nu, rho) * (
            1 + rng.normal(0, vol_noise_std)
        )
        premiums.append(_black_np(F, K, tau, r, vol))

    y_train = np.array(premiums)[..., np.newaxis]

    return jnp.array(x_train), jnp.array(y_train), jnp.array(x_mesh)
