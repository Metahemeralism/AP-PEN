"""
Observation layer and dataset assembly for the synthetic Gibson-Schwartz data.

Takes the clean P-measure state paths (S_t, delta_t) and the closed-form
Q-measure pricer, and produces an observable dataset of the form a PINN (or the
Kalman benchmark) would actually consume:

    - a panel of futures log-prices across a fixed set of maturities tau,
      observed at each date along the simulated path,
    - with deliberate, controllable measurement noise added in log-price space
      (mirroring Schwartz's serially- and cross-sectionally-uncorrelated
      disturbances on the measurement equation),
    - alongside the *true latent delta_t path*, retained separately as the
      evaluation target and never exposed to the inversion.

Why noise lives here and not in the pricer
------------------------------------------
The closed form gives exact prices given the state. Real futures quotes carry
bid-ask spread, non-simultaneity, and data error. By injecting noise here, on
top of exact prices, the signal-to-noise ratio is a single dial we control,
which is exactly what the identifiability experiments need.
"""

from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from gs_wamol.physics.gibson_schwartz import GSParams, log_futures


@dataclass(frozen=True)
class ObservationConfig:
    """Configuration for the observable futures panel.

    taus       : (M,) fixed maturities (time-to-delivery) for the M contracts
    noise_std  : std of the additive Gaussian noise in log-price space
                 (set 0.0 for clean/noise-free data)
    """
    taus: jnp.ndarray
    noise_std: float


def build_panel(key: jax.Array,
                t_grid: jnp.ndarray,
                S: jnp.ndarray,
                delta: jnp.ndarray,
                p: GSParams,
                cfg: ObservationConfig):
    """Construct the observable futures log-price panel.

    Parameters
    ----------
    key     : PRNG key for the measurement noise
    t_grid  : (N+1,) observation dates (from the path simulator)
    S       : (P, N+1) spot paths
    delta   : (P, N+1) true latent convenience-yield paths
    p       : GSParams (risk-neutral, for pricing)
    cfg     : ObservationConfig

    Returns
    -------
    dict with:
      'taus'        : (M,) maturities
      'log_F_clean' : (P, N+1, M) exact closed-form log futures prices
      'log_F_obs'   : (P, N+1, M) noisy observed log futures prices
      'S'           : (P, N+1) spot paths (echoed through)
      'delta_true'  : (P, N+1) ground-truth latent path (evaluation target)
      't_grid'      : (N+1,) dates

    Note: at each (path, date) we price the *whole* maturity vector taus from the
    state (S, delta) at that date. tau is the fixed time-to-delivery of each
    contract, held constant across dates here; if you later want dated contracts
    with tau shrinking toward delivery, replace cfg.taus with a per-date matrix.
    """
    taus = cfg.taus  # (M,)

    # Price the term structure at every (path, date) state.
    # log_futures broadcasts over leading axes; vmap over the maturity axis.
    def price_curve(S_pt, d_pt):
        # S_pt, d_pt are scalars (one path, one date)
        return jax.vmap(lambda tau: log_futures(tau, S_pt, d_pt, p))(taus)

    # vmap over dates, then over paths.
    price_over_dates = jax.vmap(price_curve, in_axes=(0, 0))       # (N+1, M)
    price_over_paths = jax.vmap(price_over_dates, in_axes=(0, 0))  # (P, N+1, M)
    log_F_clean = price_over_paths(S, delta)

    if cfg.noise_std > 0.0:
        eps = cfg.noise_std * jax.random.normal(key, log_F_clean.shape)
        log_F_obs = log_F_clean + eps
    else:
        log_F_obs = log_F_clean

    return {
        'taus': taus,
        'log_F_clean': log_F_clean,
        'log_F_obs': log_F_obs,
        'S': S,
        'delta_true': delta,
        't_grid': t_grid,
    }
