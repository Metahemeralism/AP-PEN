"""
Checkpoint save / load utilities (orbax + pickle fallback).
"""

import pickle

import orbax.checkpoint
from flax.serialization import from_state_dict, to_state_dict
from flax.training import orbax_utils


def save_checkpoint(ckpt: dict, path: str) -> None:
    """Save a pytree checkpoint with orbax."""
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpointer.save(path, ckpt, force=True, save_args=save_args)


def load_checkpoint(path: str) -> dict:
    """Load a pytree checkpoint with orbax."""
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return checkpointer.restore(path)


# ---------------------------------------------------------------------------
# Pickle fallback (lighter-weight, used for flat param dicts)
# ---------------------------------------------------------------------------

def save_params(params, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(to_state_dict(params), f)


def load_params(params_template, path: str):
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    return from_state_dict(loaded, params_template)
