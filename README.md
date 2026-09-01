# AP-PEN

Code for my MSc thesis, *AP-PEN: A Hard-Constrained, Physics-Enforced Network for
Convenience-Yield Inversion from Commodity Futures*, UCL Mechanical Engineering.

The convenience yield of crude oil cannot be observed directly; it has to be
inferred from the futures curve. This code does that for WTI under the
two-factor Gibson-Schwartz model. The affine coefficients A(tau) and B(tau) are
imposed analytically rather than learned, so the network only has to parameterise
the latent path, and the pricing PDE holds by construction. A Kalman filter is
used as the benchmark.

## Layout

```
notebooks/
  monte_carlo_simulation.ipynb   generates the synthetic dataset
  AP-PEN.ipynb                   the model: losses, training, evaluation
  kalman_filter.ipynb            Kalman filter benchmark

src/gs_wamol/physics/gibson_schwartz.py   closed-form GS pricer
config/                                   parameter sets the notebooks load
data/input/real/                          WTI panel (Refinitiv), rates (FRED)
data/input/synthetic/                     written by the simulator notebook
```

`data/output/` and `figures/` are gitignored. Re-running the notebooks
regenerates them.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
jupyter lab
```

Run the notebooks from inside `notebooks/`, in this order: the simulator, then
AP-PEN, then the Kalman filter.

One note on precision. JAX defaults to float32, which floors the PDE residual at
about 1e-5. That is a precision limit, not a training failure.
`kalman_filter.ipynb` switches to float64 because its likelihood recursion runs
over roughly 2,700 sequential steps and float32 error compounds.

## Attribution

The WamOL loss balancing and the Dense/MLP network are adapted from the DC-PINN
and WamOL work of Hoshisashi, Phelan and Barucca, ported from their reference
notebook at <https://github.com/khoshisashi/DC-PINNs>.

- Hoshisashi, Phelan and Barucca (2024). *Whack-a-mole Online Learning:
  Physics-Informed Neural Network for Intraday Implied Volatility Surface.*
  ICAIF '24, 847-855. <https://doi.org/10.1145/3677052.3698601>
- Hoshisashi, Phelan and Barucca (2024). *Physics-Informed Neural Networks for
  Derivative-Constrained PDEs.* ICML AI for Science Workshop.
  <https://openreview.net/forum?id=9pFHmyx4Sh>

The pricing model follows Schwartz (1997), Model 2. The Kalman filter follows
Krul (2008).

## Data

The WTI futures data comes from Refinitiv Workspace and is included so the thesis
can be reviewed. Interest rates are FRED series DGS3MO. Redistributing the
Refinitiv data beyond that purpose may be restricted by its licence.
