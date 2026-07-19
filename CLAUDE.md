# CLAUDE.md — Project Context for the Gibson–Schwartz PINN Thesis

This file orients an AI coding assistant (Claude Code) working in this
repository. Read it before editing. It explains what the project is, the
mathematical conventions the code depends on, the simulator architecture, and
the non-obvious pitfalls that have already cost time.

---

## 0. File map — standalone names → where they live now

The simulator was authored as standalone scripts and then wired into the
`gs_wamol` package. The thesis prose refers to the original names; this is where
each one actually lives in the repo:

| Original standalone name | Location in this repo |
|--------------------------|------------------------|
| `gs_closed_form.py` | `src/gs_wamol/physics/gibson_schwartz.py` |
| `gs_paths.py` | `src/gs_wamol/data/paths.py` |
| `gs_observe.py` | `src/gs_wamol/data/observe.py` |
| `gs_generate.py` | `src/gs_wamol/data/generate.py` |
| `gs_validate.py` | `tests/test_synthetic_gs.py` (now pytest-style asserts) |

The pricing core deliberately occupies the canonical `physics/gibson_schwartz.py`
name. A separate `physics/local_vol.py` holds **legacy** Black–Scholes / Dupire
local-vol helpers from an earlier IVS-calibration line of work; **the thesis does
not use it** — this project is Gibson–Schwartz only. Do not build new work on
`local_vol.py`.

---

## 1. What this project is

An MSc thesis applying a **Physics-Informed Neural Network (PINN)** to invert
for the latent **convenience yield** $\delta_t$ from observed WTI crude oil
futures term structures, under the **Gibson–Schwartz two-factor model**
(Schwartz 1997, Model 2). The inversion is benchmarked against a **Kalman
filter** baseline.

The identified research gap: no published work uses a PINN to invert for latent
convenience yield from market data.

The implementation stack is **JAX / Flax**.

**Why a simulator exists at all:** $\delta_t$ is unobservable from real market
data, so there is no ground truth to score an inversion against. The synthetic
Monte Carlo simulator generates data from *known* parameters and a *stored true*
$\delta_t$ path, which is the evaluation target the PINN is judged on. This is
the single most important thing to preserve: **the true $\delta_t$ path must
always be retained and must never leak into anything the inversion sees.**

---

## 2. The model and its two measures

Two probability measures are in play. Keeping them straight is essential; mixing
them is the most likely source of a subtle bug.

### Physical measure ℙ (real-world dynamics — used for SIMULATING paths)
$$
dS_t = (\mu - \delta_t) S_t\,dt + \sigma_1 S_t\,dW_1^{\mathbb{P}}
$$
$$
d\delta_t = \kappa(\alpha^{\mathbb{P}} - \delta_t)\,dt + \sigma_2\,dW_2^{\mathbb{P}},
\qquad dW_1^{\mathbb{P}}dW_2^{\mathbb{P}} = \rho\,dt
$$

### Risk-neutral measure ℚ (pricing dynamics — used for PRICING futures)
$$
dS_t = (r - \delta_t) S_t\,dt + \sigma_1 S_t\,dW_1^{\mathbb{Q}}
$$
$$
d\delta_t = \kappa(\alpha^{\mathbb{Q}} - \delta_t)\,dt + \sigma_2\,dW_2^{\mathbb{Q}}
$$

### What changes between measures, and what does not
- **Changes:** the spot drift ($\mu \to r$) and the convenience-yield long-run
  level ($\alpha^{\mathbb{P}} \to \alpha^{\mathbb{Q}}$).
- **Invariant:** $\kappa$, $\sigma_1$, $\sigma_2$, $\rho$ are the same under both
  measures. Do not "convert" them.

### The measure-change relationship (critical)
$$
\alpha^{\mathbb{Q}} = \alpha^{\mathbb{P}} - \frac{\sigma_2 \lambda_2}{\kappa}
$$
where $\lambda_2$ is the market price of convenience-yield risk. There is an
**asymmetry** that matters for the identifiability analysis:
- $\lambda_1$ (spot risk premium) is pinned by no-arbitrage: $\lambda_1 = (\mu - r)/\sigma_1$.
- $\lambda_2$ is **not** pinned by any no-arbitrage condition (δ is not a traded
  asset). It is only identifiable from the futures term structure.

**Rule for the code:** simulate paths with the ℙ parameters
($\mu$, $\alpha^{\mathbb{P}}$); price with the ℚ parameter ($\alpha^{\mathbb{Q}}$).
When you need $\alpha^{\mathbb{Q}}$ for pricing, compute it from
$\alpha^{\mathbb{P}}$, $\lambda_2$, $\sigma_2$, $\kappa$ using the formula above.
Never silently reuse $\alpha^{\mathbb{P}}$ where $\alpha^{\mathbb{Q}}$ belongs.

---

## 3. The closed-form futures price (the pricing core)

Derived analytically from the pricing PDE and **verified term-by-term against
Schwartz (1997) Model 2, eqs. (18)–(20)**. Lives in
`src/gs_wamol/physics/gibson_schwartz.py` (the original standalone
`gs_closed_form.py`). Do not "simplify" or rewrite these coefficients without
re-running the PDE-residual check in `tests/test_synthetic_gs.py` — they are
exact and any change will break it.

The PDE (in time-to-maturity $\tau = T - t$), GS-PDE-tau:
$$
\frac{\partial \hat{F}}{\partial \tau}
= (r-\delta)S\,\hat{F}_S + \kappa(\alpha^{\mathbb{Q}}-\delta)\,\hat{F}_\delta
+ \tfrac12\sigma_1^2 S^2 \hat{F}_{SS}
+ \rho\sigma_1\sigma_2 S\,\hat{F}_{S\delta}
+ \tfrac12\sigma_2^2 \hat{F}_{\delta\delta}
$$
with initial condition $\hat{F}(0, S, \delta) = S$.

The solution (exponential-affine in $\delta$, $S$ as a multiplicative prefactor):
$$
\hat{F}(\tau, S, \delta) = S \exp\!\big[B(\tau)\delta + A(\tau)\big]
$$
$$
B(\tau) = -\frac{1 - e^{-\kappa\tau}}{\kappa}
$$
$$
A(\tau) = \left( r - \alpha^{\mathbb{Q}} + \tfrac12\frac{\sigma_2^2}{\kappa^2}
          - \frac{\sigma_1\sigma_2\rho}{\kappa} \right)\tau
        + \tfrac14\sigma_2^2\frac{1 - e^{-2\kappa\tau}}{\kappa^3}
        + \left( \alpha^{\mathbb{Q}}\kappa + \sigma_1\sigma_2\rho
          - \frac{\sigma_2^2}{\kappa} \right)\frac{1 - e^{-\kappa\tau}}{\kappa^2}
$$

Note Schwartz writes his maturity argument as $T$; it plays the role of our
$\tau$. His $\hat{\alpha}$ equals our $\alpha^{\mathbb{Q}}$.

**Known sharp edge in the derivation (for reference, not for the code):** the
$\sigma_2^2/2$ term in $A'(\tau)$ carries $B(\tau)^2$, not $B(\tau)$, because it
comes from $\hat{F}_{\delta\delta} = B^2\hat{F}$. The $B^2$ expansion is what
produces the $e^{-2\kappa\tau}$ term. If you ever re-derive or refactor symbolic
coefficients, this is where a power gets dropped.

---

## 4. Simulator architecture (Option A)

Pipeline: **simulate ℙ paths → price under ℚ with the closed form → add
measurement noise**. Files, in dependency order:

| File | Role |
|------|------|
| `src/gs_wamol/physics/gibson_schwartz.py` | `GSParams`, `B_coeff`, `A_coeff`, `log_futures`, `futures`, `term_structure`. The pricing core. (orig. `gs_closed_form.py`) |
| `src/gs_wamol/data/paths.py` | `PhysicalParams`, `simulate_paths`. Correlated Euler–Maruyama under ℙ. Returns `(t_grid, S, delta)`; `delta` is GROUND TRUTH. (orig. `gs_paths.py`) |
| `src/gs_wamol/data/observe.py` | `ObservationConfig`, `build_panel`. Prices the maturity panel from each state and adds log-price noise. (orig. `gs_observe.py`) |
| `tests/test_synthetic_gs.py` | Four-check validation harness. Run this after touching the pricer or paths. (orig. `gs_validate.py`) |
| `src/gs_wamol/data/generate.py` | End-to-end example producing a dataset dict. (orig. `gs_generate.py`) |

**Why Option A (closed form) and not nested Monte Carlo:** the closed form gives
*exact* prices given the state, so there is no pricing sampling error. All noise
is introduced deliberately in the observation layer, making signal-to-noise a
single controllable dial. This is required for the identifiability experiments.
Do not replace the closed-form pricing step with nested simulation.

### On-disk layout — where files belong

Data is separated by *provenance* (can it be regenerated by running code?) so
that `.gitignore` can draw a single clean line at `data/output/`.

```
data/
  input/
    real/        WTI CSVs from Bloomberg. NOT reproducible from code — always tracked.
    synthetic/   mc_data.pkl, written by notebooks/monte_carlo_simulation.ipynb.
  output/        Model artefacts. GITIGNORED — regenerate by re-running the notebook.
    checkpoints/<run_tag>/    orbax checkpoints, one dir per run_tag(config)
    results/results_<tag>.pkl training history + config
figures/
  monte_carlo_sim/   written by scripts/make_mc_figures.py (thesis_style.FIGURE_DIR)
  raw_data/          written by notebooks/data_exploration.ipynb
  pinn/              written by notebooks/PINN_implementation.ipynb
```

Rules:
- **Never put data files inside `src/gs_wamol/`.** The package is code; the WTI
  CSVs used to live in `src/gs_wamol/data/processed_data/` and that made the
  installable package carry 3.5 MB of thesis data.
- **`data/output/` is disposable.** Anything you cannot regenerate by re-running
  a notebook does not belong there.
- Notebook paths are **repo-relative** (`../data/input/...`), never absolute —
  an absolute `/Users/...` path breaks the moment the repo is cloned elsewhere.
- Checkpoint and result filenames are derived from `run_tag(config)`, which is
  derived from `config.loss_str`. Renaming a model variant renames its artefacts;
  move the old dirs or the next run silently starts from scratch.

### Data conventions
- Spot is simulated in **log space** ($X = \ln S$) for positivity and to remove
  multiplicative-noise discretisation bias. The log-spot SDE picks up the
  $-\tfrac12\sigma_1^2$ Itô term: $dX = (\mu - \delta - \tfrac12\sigma_1^2)dt + \sigma_1 dW_1$.
- Correlation is imposed by a 2×2 Cholesky factor of $[[1,\rho],[\rho,1]]$.
- The observable panel is **log futures prices**, shape `(paths, dates,
  maturities)`. Noise is additive Gaussian in log-price space.
- `delta_true` is carried in the dataset dict separately and is the evaluation
  target. **Never feed it to the inversion.**

### Maturity handling — open extension point
Currently each contract's $\tau$ is held **fixed across dates**. Real Bloomberg
dated contracts (e.g. `CLZ5 Comdty`) have $\tau$ **shrinking toward delivery**:
$\tau = (\text{delivery date} - \text{observation date})/365$. To support this,
replace the fixed `cfg.taus` vector with a per-date $\tau$ matrix. The pricing
layer already vmaps over $\tau$, so this is a localized change in
`src/gs_wamol/data/observe.py`.

---

## 5. Numerical precision — important gotcha

JAX defaults to **float32**. In float32:
- The PDE residual floor is ~$10^{-5}$.
- The closed-form initial-condition error is ~$10^{-5}$.

In **float64** (`jax.config.update("jax_enable_x64", True)`) both drop to machine
epsilon (~$10^{-14}$), confirming the analytic coefficients are exact.

**Consequences:**
- Validation / verification code must enable x64, or the checks will appear to
  "fail" when the maths is actually correct. `tests/test_synthetic_gs.py` enables
  it.
- A **float32 PINN cannot drive its PDE-residual loss below ~$10^{-5}$**. This is
  a precision floor, not a training failure. Set convergence tolerances
  accordingly, and consider float64 for residual-sensitive experiments.

---

## 6. Conventions and invariants to preserve when editing

1. **Measure discipline:** ℙ params simulate, ℚ params price. Compute
   $\alpha^{\mathbb{Q}}$ from $\alpha^{\mathbb{P}}$; never conflate them.
2. **Don't touch the closed-form coefficients** without re-running
   `tests/test_synthetic_gs.py`. The PDE residual must stay at machine epsilon
   (float64).
3. **The true $\delta_t$ path is sacred:** retained for evaluation, never an
   input to the inversion.
4. **Pure, jittable functions:** the model functions are written pure for JAX.
   Keep them side-effect-free and vmap/jit-friendly. Pass parameters via the
   frozen dataclasses (`GSParams`, `PhysicalParams`), not globals.
5. **Noise lives in the observation layer only**, never in the pricer.
6. **Mean-reversion timescale:** the identifiability-critical timescale is
   $\tau^\ast = 1/\kappa$. The $\kappa$–$\lambda$ entanglement and the $1/T$ bias
   in $\kappa$ estimation are known failure modes; code that estimates or
   stresses these should be aware of them.

---

## 7. Glossary

| Symbol | Meaning |
|--------|---------|
| $S_t$ | spot price |
| $\delta_t$ | instantaneous convenience yield (the latent inversion target) |
| $\kappa$ | convenience-yield mean-reversion speed ($\kappa > 0$) |
| $\alpha^{\mathbb{P}}, \alpha^{\mathbb{Q}}$ | long-run convenience yield under ℙ / ℚ |
| $\sigma_1, \sigma_2$ | spot vol, convenience-yield vol |
| $\rho$ | Brownian correlation, $\in(-1,1)$ |
| $\mu$ | real-world spot drift (ℙ only) |
| $r$ | risk-free rate (ℚ pricing) |
| $\lambda_1, \lambda_2$ | market prices of spot / convenience-yield risk |
| $\tau$ | time to maturity, $\tau = T - t$ |
| $A(\tau), B(\tau)$ | closed-form coefficients |
| $\hat{F}$ | futures price as a function of $(\tau, S, \delta)$ |

---

## 8. Key references

- **Schwartz (1997)**, *The Stochastic Behavior of Commodity Prices*, J. Finance
  52(3). Model 2 is the two-factor model used here; eqs. (18)–(20) are the
  closed-form futures price.
- **Gibson & Schwartz (1990)**, *Stochastic convenience yield and the pricing of
  oil contingent claims*, J. Finance 45.
- Hoshisashi, Phelan & Barucca (ICAIF 2024) for the WamOL framework this work
  adapts.
