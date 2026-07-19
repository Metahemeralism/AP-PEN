"""Pricing-model physics for the GS AC-PINN.

`gibson_schwartz.py` is the Gibson-Schwartz Model 2 closed-form futures pricer
(the Q-measure pricing layer of the synthetic simulator). `constraints.py` holds
the economic / no-arbitrage penalty terms used by the loss.
"""

from gs_wamol.physics.gibson_schwartz import (
    GSParams,
    B_coeff,
    A_coeff,
    log_futures,
    futures,
    term_structure,
)

__all__ = [
    "GSParams",
    "B_coeff",
    "A_coeff",
    "log_futures",
    "futures",
    "term_structure",
]
