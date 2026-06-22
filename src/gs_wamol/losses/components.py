"""
Individual loss term computation.

Each term returns a per-point array (before reduction), so that
self-adaptive weights can be applied point-wise.

STATUS: GS-ready stub. The previous implementation here computed the
*IVS local-volatility* loss (Black-Scholes data fit + Dupire forward-equation
PDE residual + call-surface no-arbitrage penalties); it was removed along with
the legacy `physics/local_vol.py` and `data/synthetic.py` modules.

The Gibson-Schwartz loss terms belong here and are not yet implemented:
  * e_acc : fit to the observed log-futures panel
            (gs_wamol.data.observe.build_panel -> 'log_F_obs')
  * e_pde : GS-PDE-tau residual of the network's futures map, using the
            closed form / PDE in gs_wamol.physics.gibson_schwartz
  * e_arb : GS economic constraints on the convenience yield
            (see physics/constraints.py TODO)

`adj` is generic and is retained as-is for the WAMOL weighting in
losses/factory.py and training/trainer.py.
"""

import jax.numpy as jnp


def error(fn, data, s_0: float, r: float):
    """Compute all loss-relevant errors for model fn on data.

    Not yet implemented for Gibson-Schwartz. See the module docstring for the
    intended e_acc / e_pde / e_arb terms.

    Returns
    -------
    err     : dict of per-point squared error arrays
    metrics : dict of scalar mean errors
    """
    raise NotImplementedError(
        "Gibson-Schwartz loss terms are not implemented yet. The legacy IVS "
        "local-vol loss was removed; implement e_acc/e_pde/e_arb against the "
        "GS futures panel and GS-PDE-tau here."
    )


def adj(loss_arr, weight: float, sa_weights):
    """Scale a per-point loss array by its loss weight and self-adaptive weights."""
    return weight * jnp.mean(sa_weights * loss_arr)
