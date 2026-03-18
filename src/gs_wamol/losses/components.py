"""
Individual loss term computation.

Each term returns a per-point array (before reduction), so that
self-adaptive weights can be applied point-wise.
"""

import jax.numpy as jnp

from gs_wamol.physics.gibson_schwartz import (
    bs, lv_sqr, derivatives, call_derivatives,
)
from gs_wamol.physics.constraints import all_arb_penalties


def error(fn, data, s_0: float, r: float):
    """Compute all loss-relevant errors for model fn on data.

    Parameters
    ----------
    fn    : callable (x -> vol), the neural network
    data  : (x_train, y_train, x_mesh)
    s_0   : spot price / forward reference level
    r     : risk-free rate

    Returns
    -------
    err     : dict of per-point squared error arrays
    metrics : dict of scalar mean errors
    """
    x_train, y_train, x_mesh = data
    K, T = x_mesh[:, 0], x_mesh[:, 1]

    # Derivatives of implied vol (on mesh)
    dK_V, d2K_V, dT_V = derivatives(fn, x_mesh)
    # Derivatives of BS call price (on mesh)
    dK_C, d2K_C, dT_C = call_derivatives(fn, x_mesh, s_0, r)

    V  = fn(x_mesh).flatten()
    LV_sqr = lv_sqr(s_0, K, r, V, T, dK_V, d2K_V, dT_V)

    # Data-fit error (on training points)
    pred   = bs(s_0, x_train.T[0], x_train.T[1], r, 1, fn(x_train).flatten())
    e_acc  = (pred - y_train.ravel()) ** 2

    # PDE residual: Dupire forward equation (on mesh)
    e_pde  = (dT_C + r * K * dK_C - 0.5 * LV_sqr * (K ** 2) * d2K_C) ** 2

    # No-arbitrage penalty terms (on mesh)
    e_arb  = all_arb_penalties(dK_C, d2K_C, dT_C, r, T)

    err = {"e_acc": e_acc, "e_pde": e_pde, **e_arb}
    metrics = {k: jnp.mean(v) for k, v in err.items()}
    return err, metrics


def adj(loss_arr, weight: float, sa_weights):
    """Scale a per-point loss array by its loss weight and self-adaptive weights."""
    return weight * jnp.mean(sa_weights * loss_arr)
