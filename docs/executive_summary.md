# Executive summary — Gibson–Schwartz PINN thesis

*One page, for anyone who wants the gist without the session logs. Detailed
technical narrative lives in `docs/project_summary.md`; this file mirrors its
"Where things stand" section but written for a non-specialist reader. Updated
alongside it — last updated 2026-08-01.*

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
- **The Kalman filter baseline runs on real market data**, producing
  real-data parameter estimates and a benchmark convenience-yield path.
- **A simpler PINN variant (single network, no drift net) now also runs on
  real WTI data — and beats the plainer versions of itself against that
  Kalman benchmark.** Freezing one previously-free parameter (the market
  price of convenience-yield risk) turned out to fix an identifiability
  problem that had persisted throughout the project; the resulting model
  tracks the Kalman filter's own estimate more closely (correlation 0.89)
  than simpler versions of the same network, and a genuine out-of-sample test
  (train on early years, check pricing accuracy on held-out later years)
  shows the more physics-constrained version generalizing far better —
  barely any accuracy loss out-of-sample, versus the plain data-fitting
  version, which visibly drifts once asked to extrapolate.
- **One genuine limit of the model itself, not the method:** some model
  parameters turn out to be mathematically inseparable from futures prices
  alone in certain configurations — they only ever appear multiplied
  together in the pricing formula, regardless of what fits the data. Worth
  stating plainly in the thesis rather than looking like an unconverged
  result; the fix above resolves it in the newer model variant.

## What's left

- Reconcile the two PINN variants (the original two-network design and the
  newer single-network one) — no head-to-head comparison exists yet.
- Handle real contracts' maturities shrinking day by day, rather than the
  current fixed-maturity simplification (shared by both PINN variants and the
  Kalman baseline, so all three stay comparable).
- Settle on one final model variant and freeze the comparison figures for
  the thesis write-up.
