"""
WAMOL: Whack-A-Mole Online Learning loss-balancing mechanisms.

Three "whacks":
  1. Self-adaptive (SA) weights   – per-point weights updated via SGD ascent
                                    to up-weight high-residual regions
  2. Grad-norm balancing          – rescales each loss term so its gradient
                                    norm matches the mean across all terms
  3. Time-decay online update     – (see training/online.py, not yet impl.)

References: Wang et al. (2022) "When and why PINNs fail to train"
"""

from jax import jit, lax
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves, tree_map
import jax
import jax.numpy as jnp
import jax.example_libraries.optimizers as jax_optimizers

from gs_wamol.losses.factory import LOSS_COMPONENTS


# ---------------------------------------------------------------------------
# Whack 1 – Self-adaptive weight update (SGD ascent on per-point weights)
# ---------------------------------------------------------------------------

def make_sa_optimizer(lr: float = 1.0):
    """Return (init, update, get_params) for the SA-weight optimiser."""
    return jax_optimizers.sgd(lr)


def make_train_step_sa(ann, loss_fn_lb_dict, loss_str: str):
    """Return a JIT-compiled SA-weight update step."""
    ofunc_lb = loss_fn_lb_dict[loss_str]

    @jit
    def train_step_sa(step, state, data, l_ws, state_sa, get_params_sa):
        params = state.params

        def fn(x):
            return ann.apply(params, x)

        def loss_fn_sa(params_sa):
            return ofunc_lb(fn, data, l_ws, params_sa)[0]

        params_sa = get_params_sa(state_sa)
        _, grads_sa = jax.value_and_grad(
            lambda ps: sum(loss_fn_sa(ps).values())
        )(params_sa)

        # Ascent: negate gradients so weights increase for high-residual points
        grads_sa = tree_map(lambda g: -g, grads_sa)
        state_sa = jax_optimizers.sgd(1.0)[1](step, grads_sa, state_sa)
        return state_sa

    return train_step_sa


# ---------------------------------------------------------------------------
# Whack 2 – Gradient-norm loss balancing
# ---------------------------------------------------------------------------

def _flatten(pytree):
    return ravel_pytree(pytree)[0]


def make_update_loss_weights(ann, loss_fn_lb_dict, loss_str: str, momentum: float):
    """Return a JIT-compiled loss-weight update based on gradient-norm ratios."""
    ofunc_lb = loss_fn_lb_dict[loss_str]

    @jit
    def update_loss_weights(state, data, l_ws, state_sa, get_params_sa):
        params     = state.params
        params_sa  = get_params_sa(state_sa)

        def per_loss_fn(params):
            def fn(x): return ann.apply(params, x)
            return ofunc_lb(fn, data, l_ws, params_sa)[0]

        from jax import jacrev
        grads = jacrev(per_loss_fn)(params)

        # Mean absolute gradient per loss term
        grad_norm_dict = {k: jnp.abs(_flatten(v)).mean() for k, v in grads.items()}

        sum_norm = jnp.sum(jnp.stack(tree_leaves(grad_norm_dict)))
        w = tree_map(
            lambda x: jnp.where(x == 0.0, 1.0, sum_norm / x), grad_norm_dict
        )

        # Exponential moving average
        weights = tree_map(
            lambda old, new: old * momentum + (1 - momentum) * new, l_ws, w
        )
        return lax.stop_gradient(weights)

    return update_loss_weights
