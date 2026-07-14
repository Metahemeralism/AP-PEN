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


# Model-free bounds
def cash_and_carry_bound(K: jnp.ndarray, T: jnp.ndarray, S: jnp.ndarray, r: float) -> jnp.ndarray:
    """Cash-and-carry bound on European call prices.

    C(K, T) >= max(0, S - K exp(-rT)).

    Parameters
    ----------
    K : strike price(s)
    T : time to maturity (in years)
    S : spot price
    r : risk-free rate

    Returns
    -------
    lower_bound : the cash-and-carry lower bound on the call price(s)
    """
    return jnp.maximum(0.0, F - S_t * jnp.exp(r + u)*tau)

def reverse_cash_and_carry_bound():
    pass


def yield_floor():
    pass
