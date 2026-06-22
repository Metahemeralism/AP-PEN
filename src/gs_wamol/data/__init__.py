"""Synthetic and market data generation for the GS DC-PINN.

The Gibson-Schwartz Monte Carlo simulator is built from three layers:
    paths    : P-measure (S_t, delta_t) path simulation
    observe  : Q-measure closed-form pricing + measurement-noise panel
    generate : end-to-end dataset builder tying the two together

Real-data loading (CME/Bloomberg futures) will live in `market.py`.
"""

# NOTE: `generate` is intentionally NOT imported here. gs_wamol.data.generate
# sets jax_enable_x64 at module level (the simulator needs float64), and we do
# not want `import gs_wamol.data` to silently flip global precision. Import it
# explicitly:
#     from gs_wamol.data.generate import generate
from gs_wamol.data.paths import PhysicalParams, simulate_paths
from gs_wamol.data.observe import ObservationConfig, build_panel

__all__ = [
    "PhysicalParams",
    "simulate_paths",
    "ObservationConfig",
    "build_panel",
]
