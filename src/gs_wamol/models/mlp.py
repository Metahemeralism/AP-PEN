"""MLP and ModifiedMLP architectures with custom Dense layer."""

from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from jax import random as jran
from jax.nn.initializers import glorot_normal, normal, zeros
import ml_collections
from flax import linen as nn


activation_fn = {
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}


def _get_activation(name: str) -> Callable:
    if name in activation_fn:
        return activation_fn[name]
    raise NotImplementedError(f"Activation '{name}' not supported.")


def _weight_fact(init_fn, mean, stddev):
    def init(key, shape):
        key1, key2 = jran.split(key)
        w = init_fn(key1, shape)
        g = mean + normal(stddev)(key2, (shape[-1],))
        g = jnp.exp(g)
        v = w / g
        return g, v
    return init


class Dense(nn.Module):
    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param("kernel", self.kernel_init, (x.shape[-1], self.features))
        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(
                    self.kernel_init,
                    mean=self.reparam["mean"],
                    stddev=self.reparam["stddev"],
                ),
                (x.shape[-1], self.features),
            )
            kernel = g * v
        bias = self.param("bias", self.bias_init, (self.features,))
        return jnp.dot(x, kernel) + bias


class MLP(nn.Module):
    arch_name: Optional[str] = "MLP"
    hidden_dim: Tuple[int] = (32, 16)
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dim:
            x = Dense(features=dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)
        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        x = nn.softplus(x)  # enforces positivity
        return x


class ModifiedMLP(nn.Module):
    """MLP with multiplicative gating via two parallel encoders (u, v)."""

    arch_name: Optional[str] = "ModifiedMLP"
    hidden_dim: Tuple[int] = (32, 16)
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        u = Dense(features=self.hidden_dim[0], reparam=self.reparam)(x)
        v = Dense(features=self.hidden_dim[0], reparam=self.reparam)(x)
        u = self.activation_fn(u)
        v = self.activation_fn(v)

        for dim in self.hidden_dim:
            x = Dense(features=dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)
            x = x * u + (1 - x) * v

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        x = nn.softplus(x)  # enforces positivity
        return x
