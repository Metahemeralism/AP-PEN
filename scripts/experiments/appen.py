"""AP-PEN training code extracted verbatim from notebooks/AP-PEN.ipynb.

Faithful reproduction so experiments run headless. The notebook's module-level
globals (normalize_time, T_SCALE, taus, STORAGE_COST_U, DELTA_MAX, PSI_INIT,
LAMBDA2_FIXED) are module globals here too, set via configure_* helpers --
same late-binding behaviour calibration()/error()/make_fns() rely on.

ONE addition over the notebook: KAPPA_MIN, a lower bound on the learned kappa
(default 0.0 == notebook behaviour exactly). See docs/results_and_discussion.md
S2.4 for why this is needed on real data.
"""
import os, sys, types, pickle, json, time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Tuple, Union

import jax
import jax.numpy as jnp
from jax import jacrev, jit, lax, random, vmap
from jax.nn.initializers import glorot_normal, normal, zeros
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves, tree_map

import optax
import ml_collections
from flax import linen as nn
from flax.training import train_state

import numpy as np
import pandas as pd

REPO = "/Users/evanlynch/Developer/DC-PINNs"


# ----------------------------------------------------------------- pricer
@dataclass(frozen=True)
class GSParams:
    r: float
    kappa: float
    alpha_Q: float
    sigma1: float
    sigma2: float
    rho: float


def B_coeff(tau, kappa):
    return -(1.0 - jnp.exp(-kappa * tau)) / kappa


def A_coeff(tau, p: GSParams):
    kappa = p.kappa
    s1s2rho = p.sigma1 * p.sigma2 * p.rho
    s2sq = p.sigma2 ** 2
    lin = (p.r - p.alpha_Q + 0.5 * s2sq / kappa ** 2 - s1s2rho / kappa) * tau
    two_kappa = 0.25 * s2sq * (1.0 - jnp.exp(-2.0 * kappa * tau)) / kappa ** 3
    one_kappa = (p.alpha_Q * kappa + s1s2rho - s2sq / kappa) * (1.0 - jnp.exp(-kappa * tau)) / kappa ** 2
    return lin + two_kappa + one_kappa


_gs_shim = types.ModuleType("gs_wamol.physics.gibson_schwartz")
_gs_shim.GSParams = GSParams
sys.modules.setdefault("gs_wamol", types.ModuleType("gs_wamol"))
sys.modules.setdefault("gs_wamol.physics", types.ModuleType("gs_wamol.physics"))
sys.modules["gs_wamol.physics.gibson_schwartz"] = _gs_shim


# ----------------------------------------------------------------- networks
activation_fn = {"tanh": jnp.tanh, "sin": jnp.sin}


class Dense(nn.Module):
    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", self.kernel_init, (x.shape[-1], self.features))
        bias = self.param("bias", self.bias_init, (self.features,))
        return jnp.dot(x, kernel) + bias


class MLP(nn.Module):
    hidden_dim: Tuple[int] = (32, 16)
    out_dim: int = 1
    activation: str = "tanh"

    def setup(self):
        self.activation_fn = activation_fn[self.activation]

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.hidden_dim)):
            x = Dense(features=self.hidden_dim[i])(x)
            x = self.activation_fn(x)
        return Dense(features=self.out_dim)(x)


def build_nets(config):
    return MLP(hidden_dim=config.path_hidden_dim, out_dim=1,
               activation=config.ann_activation_str)


# ----------------------------------------------------------------- psi
R_FIXED = 0.05
KAPPA_MIN = 0.0          # NEW: floor on learned kappa. 0.0 == notebook behaviour.
LAMBDA2_FIXED = None
PSI_INIT = None
T_SCALE = None
taus = None
STORAGE_COST_U = None
DELTA_MAX = None
DELTA_MIN = -0.3
U_LEVEL = 0.30
DELTA_MAX_LEVEL = 0.75


def _softplus_inv(y):
    return float(np.log(np.expm1(y)))


def _kappa_fwd(x):
    return KAPPA_MIN + jax.nn.softplus(x)


def _kappa_inv(y):
    return _softplus_inv(y - KAPPA_MIN)


PSI_TRANSFORMS = {
    "kappa":  (_kappa_fwd, _kappa_inv),
    "sigma1": (jax.nn.softplus, _softplus_inv),
    "sigma2": (jax.nn.softplus, _softplus_inv),
    "rho":    (jnp.tanh, lambda y: float(np.arctanh(y))),
    "alpha_P": (lambda x: x, lambda y: y),
}


def constrain_psi(raw):
    return {k: fwd(raw[k]) for k, (fwd, _) in PSI_TRANSFORMS.items()}


def constrain_psi_full(raw):
    psi = constrain_psi(raw)
    psi["alpha_Q"] = psi["alpha_P"] - psi["sigma2"] * LAMBDA2_FIXED / psi["kappa"]
    return psi


def raw_psi_init(psi0):
    return {k: jnp.asarray(inv(psi0[k])) for k, (_, inv) in PSI_TRANSFORMS.items()}


def psi_to_gsparams(psi):
    p_Q_eff = GSParams(r=R_FIXED, kappa=psi["kappa"], alpha_Q=psi["alpha_Q"],
                       sigma1=psi["sigma1"], sigma2=psi["sigma2"], rho=psi["rho"])
    return {"p_Q": p_Q_eff, "kappa_P": psi["kappa"],
            "alpha_P": psi["alpha_P"], "sigma2_P": psi["sigma2"]}


def normalize_time(t):
    return jnp.atleast_1d(t / T_SCALE)


def make_fns(path_ann, params):
    psi = constrain_psi_full(params["psi"])
    gsp = psi_to_gsparams(psi)
    p = gsp["p_Q"]

    def log_F_hat_fn(S, delta, tau):
        return jnp.log(S) + B_coeff(tau, p.kappa) * delta + A_coeff(tau, p)

    def F_hat_fn(S, delta, tau):
        return jnp.exp(log_F_hat_fn(S, delta, tau))

    def delta_hat_fn(t):
        return path_ann.apply(params["path"], normalize_time(t))[0]

    return F_hat_fn, delta_hat_fn, gsp


Q_KEYS = ["kappa", "sigma1", "sigma2", "rho"]
P_KEYS = ["alpha_P"]


def split_psi(raw_psi):
    return {k: raw_psi[k] for k in Q_KEYS}, {k: raw_psi[k] for k in P_KEYS}


def merge_psi(q, p):
    return {**q, **p}


def init_nets(config, key):
    path_ann = build_nets(config)
    path_params0 = path_ann.init(key, jnp.ones((1,)))
    q0, p0 = split_psi(raw_psi_init(PSI_INIT))
    return path_ann, path_params0, q0, p0


def guardrail_arrays(n):
    return jnp.full((n,), U_LEVEL), jnp.full((n,), DELTA_MAX_LEVEL)


# ----------------------------------------------------------------- loss
def error(fns, data):
    F_hat_fn, delta_hat_fn, gsp = fns
    p_Q = gsp["p_Q"]
    kappa_P, alpha_P, sigma2_P = gsp["kappa_P"], gsp["alpha_P"], gsp["sigma2_P"]
    t_train, S_train, log_F_obs_train, r_train = data

    delta_hat = vmap(delta_hat_fn)(t_train)

    def log_F_hat_fn(S, delta, tau, r):
        p_Q_r = GSParams(r=r, kappa=p_Q.kappa, alpha_Q=p_Q.alpha_Q,
                         sigma1=p_Q.sigma1, sigma2=p_Q.sigma2, rho=p_Q.rho)
        return jnp.log(S) + B_coeff(tau, p_Q_r.kappa) * delta + A_coeff(tau, p_Q_r)

    log_F_hat = vmap(lambda S, d, r: vmap(lambda tau: log_F_hat_fn(S, d, tau, r))(taus))(
        S_train, delta_hat, r_train)
    e_data = (log_F_hat - log_F_obs_train) ** 2

    dt = jnp.diff(t_train)
    kappa_P_sde = lax.stop_gradient(kappa_P)
    sigma2_P_sde = lax.stop_gradient(sigma2_P)

    def sde_nll(delta_i, delta_ip1, dt_i):
        mean = alpha_P + (delta_i - alpha_P) * jnp.exp(-kappa_P_sde * dt_i)
        var = (sigma2_P_sde ** 2 / (2.0 * kappa_P_sde)) * (1.0 - jnp.exp(-2.0 * kappa_P_sde * dt_i))
        return 0.5 * jnp.log(2.0 * jnp.pi * var) + 0.5 * (delta_ip1 - mean) ** 2 / var

    delta_hat_sde = lax.stop_gradient(delta_hat)
    e_sde = vmap(sde_nll)(delta_hat_sde[:-1], delta_hat_sde[1:], dt)

    F_hat_grid = jnp.exp(log_F_hat)
    ceiling = S_train[:, None] * jnp.exp((r_train[:, None] + STORAGE_COST_U[:, None]) * taus[None, :])
    rcac_floor = S_train[:, None] * jnp.exp((r_train[:, None] - DELTA_MAX[:, None]) * taus[None, :])
    e_cac = jnp.maximum(0., F_hat_grid - ceiling) ** 2
    e_rcac = jnp.maximum(0., rcac_floor - F_hat_grid) ** 2
    e_delta_floor = jnp.maximum(0., DELTA_MIN - delta_hat) ** 2

    err = {"e_data": e_data, "e_sde": e_sde, "e_cac": e_cac,
           "e_rcac": e_rcac, "e_delta_floor": e_delta_floor}
    return err, {k: jnp.mean(v) for k, v in err.items()}


VARIANT_COMPONENTS = {
    "MLP": ["e_data"],
    "APPINN": ["e_data", "e_sde"],
    "APPINN_ARB": ["e_data", "e_sde", "e_cac", "e_rcac", "e_delta_floor"],
}


def adj(loss, lw):
    return lw * jnp.mean(loss)


def make_loss_lb(components):
    def loss_fn(fns, data, l_ws):
        err, metrics = error(fns, data)
        return {k: adj(err[k], l_ws[k]) for k in components}, metrics
    return loss_fn


loss_fn_lb = {k: make_loss_lb(v) for k, v in VARIANT_COMPONENTS.items()}


def make_loss_fn(components):
    def loss_fn(fns, data, l_ws):
        loss, metrics = loss_fn_lb[components](fns, data, l_ws)
        return sum(loss.values()), metrics
    return loss_fn


loss_fn = {k: make_loss_fn(k) for k in loss_fn_lb}
init_l_ws = {k: {c: 1. for c in v} for k, v in VARIANT_COMPONENTS.items()}

with open(f"{REPO}/config/hp.sweep.json") as f:
    adam_hp = json.load(f)


def make_optimizer(hp):
    lr = optax.exponential_decay(init_value=hp["learning_rate"],
                                 transition_steps=hp["decay_steps"],
                                 decay_rate=hp["decay_rate"])
    return optax.adam(learning_rate=lr, b1=hp["beta1"], b2=hp["beta2"], eps=hp["adam_eps"])


def flatten_pytree(pytree):
    return ravel_pytree(pytree)[0]


# ----------------------------------------------------------------- training
def calibration(config, data, hp=None, verbose=True, keep_history=False):
    hp = hp or adam_hp
    ofunc = loss_fn[config.loss_str]
    components = VARIANT_COMPONENTS[config.loss_str]
    has_sde = "e_sde" in components
    l_ws = dict(init_l_ws[config.loss_str])

    key = random.PRNGKey(config.seed)
    key, key_init = random.split(key, 2)
    path_ann, path_params0, q0, p0 = init_nets(config, key_init)
    state_path = train_state.TrainState.create(apply_fn=None, params=path_params0, tx=make_optimizer(hp))
    state_q = train_state.TrainState.create(apply_fn=None, params=q0, tx=make_optimizer(hp))
    state_p = train_state.TrainState.create(apply_fn=None, params=p0, tx=make_optimizer(hp))

    def assemble(path_p, q_p, p_p):
        return {"path": path_p, "psi": merge_psi(q_p, p_p)}

    @jit
    def step1(state_path, q_p, p_p, l_ws):
        def loss_fn_(path_p):
            return ofunc(make_fns(path_ann, assemble(path_p, q_p, p_p)), data, l_ws)
        (loss, metric), grads = jax.value_and_grad(loss_fn_, has_aux=True)(state_path.params)
        return state_path.apply_gradients(grads=grads), loss, metric

    @jit
    def step2a(state_q, path_p, p_p):
        def loss_fn_(q_p):
            err, _ = error(make_fns(path_ann, assemble(path_p, q_p, p_p)), data)
            return jnp.mean(err["e_data"])
        loss, grads = jax.value_and_grad(loss_fn_)(state_q.params)
        return state_q.apply_gradients(grads=grads)

    @jit
    def step2b(state_p, path_p, q_p):
        def loss_fn_(p_p):
            err, _ = error(make_fns(path_ann, assemble(path_p, q_p, p_p)), data)
            return jnp.mean(err["e_sde"])
        loss, grads = jax.value_and_grad(loss_fn_)(state_p.params)
        return state_p.apply_gradients(grads=grads)

    @jit
    def update_loss_weights(state_path, q_p, p_p, l_ws):
        def loss_fn_(path_p):
            err, _ = error(make_fns(path_ann, assemble(path_p, q_p, p_p)), data)
            return {c: jnp.mean(err[c]) for c in components}
        grads = jacrev(loss_fn_)(state_path.params)
        gn = {c: jnp.abs(flatten_pytree(grads[c])).mean() for c in components}
        s = jnp.sum(jnp.stack(tree_leaves(gn)))
        w = tree_map(lambda x: jnp.where(x == 0., 1., s / x), gn)
        return lax.stop_gradient(tree_map(lambda o, n: o * config.loss_balancing_momentum
                                          + (1 - config.loss_balancing_momentum) * n, l_ws, w))

    use_balancing = config.get("balancing", True) and config.loss_str in ("APPINN", "APPINN_ARB")
    hist = []
    t0 = time.time()
    for epoch in range(config.num_epochs):
        if use_balancing and epoch % 100 == 0:
            l_ws = update_loss_weights(state_path, state_q.params, state_p.params, l_ws)
        state_path, loss, metric = step1(state_path, state_q.params, state_p.params, l_ws)
        state_q = step2a(state_q, state_path.params, state_p.params)
        if has_sde:
            state_p = step2b(state_p, state_path.params, state_q.params)
        if keep_history and epoch % 100 == 0:
            hist.append((epoch, float(loss),
                         {k: float(v) for k, v in constrain_psi_full(
                             merge_psi(state_q.params, state_p.params)).items()}))
    if verbose:
        print(f"    {config.loss_str} seed={config.seed} done in {time.time()-t0:.1f}s", flush=True)
    return make_fns(path_ann, assemble(state_path.params, state_q.params, state_p.params)), hist


def run_tag(config):
    return f"{config.loss_str}{'' if config.get('balancing', True) else '_nobal'}"


def make_variant_configs(num_epochs=5000, path_hidden_dim=(32, 32, 32), seed=42):
    base = ml_collections.ConfigDict({
        "loss_str": "MLP", "num_epochs": num_epochs, "path_hidden_dim": path_hidden_dim,
        "ann_activation_str": "tanh", "loss_balancing_momentum": 0.5, "seed": seed,
        "ann_reparam": False, "balancing": True})
    appinn = ml_collections.ConfigDict(base.to_dict()); appinn.loss_str = "APPINN"
    nobal = ml_collections.ConfigDict(appinn.to_dict()); nobal.balancing = False
    arb = ml_collections.ConfigDict(appinn.to_dict()); arb.loss_str = "APPINN_ARB"
    return base, nobal, appinn, arb


# ----------------------------------------------------------------- data
def get_data_mc(path_id=0):
    with open(f"{REPO}/data/input/synthetic/mc_data.pkl", "rb") as f:
        mc = pickle.load(f)
    return (mc["t_grid"], mc["S"][path_id], mc["taus"], mc["log_F_obs"][path_id],
            mc["delta_true"][path_id], GSParams(**asdict(mc["params_Q"])),
            mc["params_P"]["kappa"], mc["params_P"]["alpha_P"], mc["params_P"]["sigma2"])


def get_data_wti(n_maturities=8):
    df = pd.read_csv(f"{REPO}/data/input/real/wti_analysis_ready.csv")
    df = df.dropna(subset=["spot", "rate"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "tau"])
    df["slot"] = df.groupby("date").cumcount()
    df = df[df["slot"] < n_maturities]
    df = df[df.groupby("date")["slot"].transform("count") == n_maturities]
    tau_m = df.pivot(index="date", columns="slot", values="tau").sort_index()
    settle = df.pivot(index="date", columns="slot", values="settle").sort_index()
    spot = df.groupby("date")["spot"].first().sort_index()
    rate = df.groupby("date")["rate"].first().sort_index()
    dates = tau_m.index.to_numpy()
    t = ((dates - dates[0]) / np.timedelta64(365, "D")).astype(float)
    return (dates, t, spot.to_numpy(), tau_m.to_numpy().mean(axis=0),
            np.log(settle.to_numpy()), rate.to_numpy())


def configure(taus_, T_scale_, psi_init_, lambda2_, n_dates, kappa_min=0.0):
    """Set the module globals calibration()/error()/make_fns() read by name."""
    global taus, T_SCALE, PSI_INIT, LAMBDA2_FIXED, STORAGE_COST_U, DELTA_MAX, KAPPA_MIN
    taus = jnp.asarray(taus_)
    T_SCALE = float(T_scale_)
    PSI_INIT = dict(psi_init_)
    LAMBDA2_FIXED = float(lambda2_)
    KAPPA_MIN = float(kappa_min)
    STORAGE_COST_U, DELTA_MAX = guardrail_arrays(n_dates)


def price_rmse(fns, t_arr, S_arr, logF_arr, r_arr):
    _, delta_hat_fn, gsp = fns
    p = gsp["p_Q"]
    d = vmap(delta_hat_fn)(t_arr)

    def lf(S, delta, tau, r):
        pr = GSParams(r=r, kappa=p.kappa, alpha_Q=p.alpha_Q, sigma1=p.sigma1,
                      sigma2=p.sigma2, rho=p.rho)
        return jnp.log(S) + B_coeff(tau, pr.kappa) * delta + A_coeff(tau, pr)

    lfh = vmap(lambda S, dd, r: vmap(lambda tau: lf(S, dd, tau, r))(taus))(S_arr, d, r_arr)
    return float(jnp.sqrt(jnp.mean((lfh - logF_arr) ** 2))), np.asarray(d)


def psi_dict(p_Q, alpha_P):
    return {"kappa": float(p_Q.kappa), "sigma1": float(p_Q.sigma1),
            "sigma2": float(p_Q.sigma2), "rho": float(p_Q.rho),
            "alpha_Q": float(p_Q.alpha_Q), "alpha_P": float(alpha_P)}


def m_of(psi):
    """The identified composite: m = alpha_Q + sigma1*sigma2*rho/kappa."""
    return psi["alpha_Q"] + psi["sigma1"] * psi["sigma2"] * psi["rho"] / psi["kappa"]
