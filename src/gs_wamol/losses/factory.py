"""
Loss-function construction and initialisation helpers.

Three model variants:
  MLP     – data fit only
  PINN    – data fit + PDE residual
  ACPINN  – data fit + PDE residual + no-arbitrage constraints
"""

import jax.numpy as jnp

from gs_wamol.losses.components import error, adj


# ---------------------------------------------------------------------------
# Static loss-weight initialisations
# ---------------------------------------------------------------------------

INIT_L_WS = {
    "MLP":    {"e_acc": 1.0},
    "PINN":   {"e_acc": 1.0, "e_pde": 1.0},
    "ACPINN": {"e_acc": 1.0, "e_pde": 1.0,
               "e_arb_dK": 1.0, "e_arb_d2K": 1.0, "e_arb_dT": 1.0},
}

LOSS_COMPONENTS = {
    "MLP":    ["e_acc"],
    "PINN":   ["e_acc", "e_pde"],
    "ACPINN": ["e_acc", "e_pde", "e_arb_dK", "e_arb_d2K", "e_arb_dT"],
}


def init_params_sa(loss_str: str, data):
    """Initialise per-point self-adaptive weight arrays (all ones)."""
    x_train, _, x_mesh = data
    n_train, n_mesh = len(x_train), len(x_mesh)
    mesh_ones  = jnp.ones(n_mesh)
    train_ones = jnp.ones(n_train)
    return {
        "MLP":    {"e_acc": train_ones},
        "PINN":   {"e_acc": train_ones, "e_pde": mesh_ones},
        "ACPINN": {"e_acc": train_ones, "e_pde": mesh_ones,
                   "e_arb_dK": mesh_ones, "e_arb_d2K": mesh_ones,
                   "e_arb_dT": mesh_ones},
    }[loss_str]


# ---------------------------------------------------------------------------
# Loss-function builders
# ---------------------------------------------------------------------------

def make_loss_lb(components: list, s_0: float, r: float):
    """Build a function returning per-component losses + metrics dict."""
    def loss_fn(fn, data, l_ws, params_sa):
        err, metrics = error(fn, data, s_0, r)
        loss = {k: adj(err[k], l_ws[k], params_sa[k]) for k in components}
        return loss, metrics
    return loss_fn


def make_loss_fn(loss_str: str, s_0: float, r: float):
    """Build a scalar loss function for a given model variant."""
    components = LOSS_COMPONENTS[loss_str]
    lb_fn = make_loss_lb(components, s_0, r)

    def loss_fn(fn, data, l_ws, params_sa):
        loss_dict, metrics = lb_fn(fn, data, l_ws, params_sa)
        return sum(loss_dict.values()), metrics

    return loss_fn


def build_loss_fns(s_0: float, r: float):
    """Return (loss_fn_lb, loss_fn) dicts for all model variants."""
    loss_fn_lb = {k: make_loss_lb(LOSS_COMPONENTS[k], s_0, r) for k in LOSS_COMPONENTS}
    loss_fn    = {k: make_loss_fn(k, s_0, r) for k in LOSS_COMPONENTS}
    return loss_fn_lb, loss_fn
