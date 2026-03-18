"""
No-arbitrage / economic constraint bounds on option prices.

Currently: standard static-arbitrage conditions on the call-price surface
  * Monotone decreasing in K  →  dC/dK ∈ [-exp(-rT), 0]
  * Convex in K               →  d²C/dK² ≥ 0
  * Non-decreasing in T       →  dC/dT  ≥ 0

TODO: Add Gibson-Schwartz convenience-yield economic bounds (e.g. positivity
      of the convenience yield, mean-reversion region constraints).
"""

import jax.numpy as jnp


def arb_dK(dK_C, r, T):
    """Penalty for violations of -exp(-rT) ≤ dC/dK ≤ 0."""
    upper_viol = jnp.where(dK_C > 0,            dK_C ** 2,                    0.0)
    lower_viol = jnp.where(dK_C < -jnp.exp(-r * T), dK_C ** 2, 0.0)
    return upper_viol + lower_viol


def arb_d2K(d2K_C):
    """Penalty for d²C/dK² < 0 (butterfly arbitrage)."""
    return jnp.where(d2K_C < 0, d2K_C ** 2, 0.0)


def arb_dT(dT_C):
    """Penalty for dC/dT < 0 (calendar-spread arbitrage)."""
    return jnp.where(dT_C < 0, dT_C ** 2, 0.0)


def all_arb_penalties(dK_C, d2K_C, dT_C, r, T):
    """Return dict of all three penalty arrays."""
    return {
        "e_arb_dK":  arb_dK(dK_C, r, T),
        "e_arb_d2K": arb_d2K(d2K_C),
        "e_arb_dT":  arb_dT(dT_C),
    }
