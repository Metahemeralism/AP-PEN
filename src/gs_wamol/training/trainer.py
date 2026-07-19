"""
Single-batch calibration loop.

Runs the full WAMOL training procedure (whacks 1 & 2) for a fixed dataset.
"""

import os
import pickle
import time

import jax
import jax.numpy as jnp
from jax import jit
import jax.example_libraries.optimizers as jax_optimizers
from flax.training import train_state

from gs_wamol.models.network import ann_gen
from gs_wamol.losses.factory import (
    INIT_L_WS, init_params_sa, build_loss_fns,
)
from gs_wamol.losses.wamol import (
    make_sa_optimizer,
    make_train_step_sa,
    make_update_loss_weights,
)
from gs_wamol.utils.checkpointing import save_checkpoint


def calibration(config, data, tx, s_0: float, r: float):
    """Train a model for one experiment and return (fn, history).

    Parameters
    ----------
    config : ml_collections.ConfigDict
    data   : (x_train, y_train, x_mesh)
    tx     : optax optimiser
    s_0    : spot reference price
    r      : risk-free rate
    """
    ann = ann_gen(config)
    loss_fn_lb, loss_fn = build_loss_fns(s_0, r)
    ofunc = loss_fn[config.loss_str]

    l_ws      = dict(INIT_L_WS[config.loss_str])
    params_sa = init_params_sa(config.loss_str, data)

    # Initialise network
    key, key_init = jax.random.split(jax.random.PRNGKey(config.seed))
    dummy = jnp.ones((1, config.ann_in_dim), dtype=jnp.float32)
    state = train_state.TrainState.create(
        apply_fn=ann.apply,
        params=ann.init(key_init, dummy),
        tx=tx,
    )

    # Self-adaptive weight optimiser (SGD ascent)
    opt_init_sa, opt_update_sa, get_params_sa = make_sa_optimizer(lr=config.self_adaptive_lr)
    state_sa = opt_init_sa(params_sa)

    momentum = config.loss_balancing_momentum

    # Build WAMOL helpers
    update_loss_weights = make_update_loss_weights(
        ann, loss_fn_lb, config.loss_str, momentum
    )

    # -----------------------------------------------------------------------
    @jit
    def train_step(state, data, l_ws, state_sa):
        params_sa_cur = get_params_sa(state_sa)

        def _loss(params):
            def fn(x): return ann.apply(params, x)
            return ofunc(fn, data, l_ws, params_sa_cur)

        (loss, metric), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, metric, state_sa

    @jit
    def _train_step_sa(step, state, data, l_ws, state_sa):
        params_sa_cur = get_params_sa(state_sa)

        def fn(x): return ann.apply(state.params, x)
        def loss_sa(ps): return sum(loss_fn_lb[config.loss_str](fn, data, l_ws, ps)[0].values())

        _, grads = jax.value_and_grad(loss_sa)(params_sa_cur)
        grads     = jax.tree_util.tree_map(lambda g: -g, grads)  # ascent
        state_sa  = opt_update_sa(step, grads, state_sa)
        return state_sa

    # -----------------------------------------------------------------------
    hist_loss  = []
    start_time = time.time()

    print(f"{config.loss_str} calibration ──────────────>")
    for epoch in range(config.num_epochs):

        # Whack 2 (every 100 epochs for ACPINN)
        if config.loss_str == "ACPINN" and epoch % 100 == 0:
            l_ws = update_loss_weights(state, data, l_ws, state_sa, get_params_sa)

        # Whack 1 (offset by 50 epochs, every 100)
        if config.loss_str == "ACPINN" and (epoch + 50) % 100 == 0:
            state_sa = _train_step_sa(epoch, state, data, l_ws, state_sa)

        state, loss, metric, state_sa = train_step(state, data, l_ws, state_sa)

        if epoch % 1000 == 0:
            print(f"  Epoch {epoch:>6d}: loss = {loss:.6f}", end="\r")
            hist_loss.append((epoch, float(loss), metric))

    print(f"\n<────────── completed in {time.time() - start_time:.2f}s")

    # Save checkpoint
    ckpt = {
        "params": state.params,
        "ms":     get_params_sa(state_sa),
        "ls":     l_ws,
    }
    save_checkpoint(ckpt, os.path.abspath("checkpoints"))

    def fn(x): return ann.apply(state.params, x)
    return fn, hist_loss


def run_experiment(config, data_fn, tx, s_0: float, r: float):
    """End-to-end experiment: generate data, calibrate, persist results."""
    data  = data_fn(config.pts_num, 101, s_0=s_0, r=r)
    model, hist_loss = calibration(config, data, tx, s_0, r)

    results = {"config": config, "history": hist_loss}
    with open("results_latest.pkl", "wb") as f:
        pickle.dump(results, f)

    return model
