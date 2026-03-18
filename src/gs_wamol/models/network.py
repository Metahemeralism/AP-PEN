"""Factory function for constructing network architectures from config."""

import ml_collections

from gs_wamol.models.mlp import MLP, ModifiedMLP


def ann_gen(config: ml_collections.ConfigDict):
    """Instantiate an ANN from a ml_collections config."""
    reparam = None
    if config.ann_reparam == "weight_fact":
        reparam = ml_collections.ConfigDict(
            {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
        )

    kwargs = dict(
        arch_name=config.ann_str,
        hidden_dim=config.ann_hidden_dim,
        out_dim=config.ann_out_dim,
        activation=config.ann_activation_str,
        periodicity=config.ann_periodicity,
        fourier_emb=config.ann_fourier_emb,
        reparam=reparam,
    )

    if config.ann_str == "MLP":
        return MLP(**kwargs)
    elif config.ann_str == "ModifiedMLP":
        return ModifiedMLP(**kwargs)
    else:
        raise NotImplementedError(f"Architecture '{config.ann_str}' not supported.")
