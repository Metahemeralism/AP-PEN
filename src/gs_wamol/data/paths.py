from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PhysicalParams:
    mu: float
    kappa: float
    alpha_P: float
    sigma1: float
    sigma2: float
    rho: float


def simulate_paths(key, pp, *, S0, delta0, T, n_steps, n_paths):
    dt = ...
    sqrt_dt = ...
    t_grid = ...    # the time grid over [0, T]
    ...