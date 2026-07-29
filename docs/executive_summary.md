# Executive summary — Gibson–Schwartz PINN thesis

*One page, for anyone who wants the gist without the session logs. Detailed
technical narrative lives in `docs/project_summary.md`; this file mirrors its
"Where things stand" section but written for a non-specialist reader. Updated
alongside it — last updated 2026-07-28.*

## The problem, in one paragraph

Oil futures prices imply a hidden quantity called the **convenience yield**
— the benefit of holding physical crude rather than a paper contract on it —
which drives the shape of the futures curve (contango vs. backwardation) but
can never be observed directly. The standard model for this (Gibson &
Schwartz, 1990) treats it as a mean-reverting stochastic process alongside
the spot price, which gives a closed-form pricing formula and, classically,
lets a **Kalman filter** back it out from market data. **No published work
has tried inverting for it with a neural network instead** — that's the gap
this thesis fills, using a **Physics-Informed Neural Network (PINN)**: a
network trained not just to fit prices, but to also obey the model's own
pricing PDE and stochastic dynamics.

## Approach

1. **Synthetic data first.** A Monte Carlo simulator generates price paths
   from *known* parameters and a *known* convenience-yield path — something
   real markets can never provide — so model performance can be scored
   against ground truth before touching real data.
2. **Five models, increasing physics content:** a plain data-fitting network
   (MLP baseline), a PINN with an added consistency penalty on the pricing
   ODE, an "AP-PINN" that adds a further penalty on the recovered path's own
   stochastic dynamics (with and without automatic loss-channel balancing),
   and a fifth variant that adds no-arbitrage (cash-and-carry) constraints
   on top.
3. **A classical Kalman filter baseline**, based on the standard
   literature approach (Krul, 2008), fit on ten years of real WTI crude
   futures data — both a sanity check and a like-for-like benchmark for the
   PINN.

## Where things stand today

- **On synthetic data, the PINN approach works.** All five models recover
  the hidden convenience-yield path without collapsing; adding the pricing-ODE
  penalty (PINN) is what delivers the big jump in accuracy over the plain
  MLP baseline, and the more physics-constrained AP-PINN variants match or
  slightly improve on that without giving any of it back — including the
  variant with no-arbitrage constraints added on top, which costs almost
  nothing in recovery accuracy while enforcing economic consistency the
  simpler models don't.
- **A real finding along the way, not just an engineering bug:** an earlier,
  more constrained model design collapsed during training, and digging into
  *why* showed the constraint itself was economically wrong — real WTI
  market data was found to violate that exact constraint too, confirming it
  wasn't a quirk of the simulation.
- **The Kalman filter baseline is now running on real market data**,
  producing the first real-data parameter estimates this project has and a
  benchmark convenience-yield path to compare the PINN's own output against.
- **One genuine limit of the model itself, not the method:** two of the six
  model parameters turn out to be mathematically inseparable from futures
  prices alone — they only ever appear multiplied together, in the pricing
  formula itself, regardless of what fits the data. Worth stating plainly in
  the thesis rather than looking like an unconverged result.

## What's left

- Run the PINN itself on real WTI data (validated on synthetic so far).
- Handle real contracts' maturities shrinking day by day, rather than the
  current fixed-maturity simplification (shared by both the PINN and the
  Kalman baseline, so they stay comparable).
- Settle on one final model variant and freeze the comparison figures for
  the thesis write-up.
