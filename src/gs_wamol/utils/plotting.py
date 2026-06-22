"""
Visualisation functions for calibration results.

STATUS: GS-ready. The legacy IVS plots (implied-vol surface, call-surface
no-arbitrage heatmaps, and the SABR comparison) were removed together with
`physics/local_vol.py` and `data/synthetic.py`. Only the loss-component-agnostic
training-history plotter remains; GS-specific plots (e.g. recovered vs. true
delta_t path, futures term-structure fits) belong here.
"""

import pickle

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_training_history(history_path: str, save_path=None):
    """Plot total loss and per-component metrics over epochs (log scale).

    Loss-component agnostic: it reads whatever metric keys are present in the
    saved history, so it works for any model variant.
    """
    with open(history_path, "rb") as f:
        results = pickle.load(f)
    history = results["history"]
    epochs, losses, metrics = zip(*history)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(epochs, losses, "b-", label="Total Loss")

    for key in metrics[0].keys():
        vals = [m[key] for m in metrics]
        ax.plot(epochs, vals, "--", label=key)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_yscale("log"); ax.grid(True); ax.legend()
    plt.title("Training History")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show(); plt.close()
