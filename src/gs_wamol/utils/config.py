"""
ml_collections default configurations and optimiser factory.
"""

import ml_collections
import optax


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

def make_optimizer(
    learning_rate: float = 1e-3,
    decay_rate: float    = 0.9,
    decay_steps: int     = 2000,
    b1: float            = 0.9,
    b2: float            = 0.999,
    eps: float           = 1e-8,
):
    """Adam with exponential LR decay (mirrors Cell 3 of original notebook)."""
    lr_schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
    )
    return optax.adam(learning_rate=lr_schedule, b1=b1, b2=b2, eps=eps)


# ---------------------------------------------------------------------------
# Experiment configs  (Cell 11 of original notebook)
# ---------------------------------------------------------------------------

def get_mlp_config() -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
        "data_source":             "SABR_syn",
        "pts_num":                 300,
        "ann_str":                 "MLP",
        "loss_str":                "MLP",
        "num_epochs":              5000,
        "ann_in_dim":              2,
        "ann_out_dim":             1,
        "ann_activation_str":      "tanh",
        "self_adaptive_lr":        1.0,
        "loss_balancing_momentum": 0.5,
        "seed":                    42,
        "ann_periodicity":         None,
        "ann_fourier_emb":         None,
        "ann_reparam":             False,
        "ann_hidden_dim":          (16, 16, 16, 16),
    })


def get_pinn_config() -> ml_collections.ConfigDict:
    cfg = get_mlp_config()
    cfg.loss_str = "PINN"
    return cfg


def get_dcpinn_config() -> ml_collections.ConfigDict:
    cfg = get_mlp_config()
    cfg.loss_str = "DCPINN"
    return cfg
