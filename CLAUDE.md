# CLAUDE.md — Project Context for the Gibson–Schwartz PINN Thesis

This file orients an AI coding assistant (Claude Code) working in this
repository. Read it before editing. It explains what the project is, the
mathematical conventions the code depends on, the simulator architecture, and
the non-obvious pitfalls that have already cost time.

---

## 0. File map — what's actually in the repo (updated 2026-07-28)

**The `gs_wamol` installable package is now minimal on purpose.** A repo
cleanup (commit `a8c32b6`, "cleaned repo") removed everything under
`src/gs_wamol/` except the pricing core, plus the entire `tests/` directory.
What used to be package modules now mostly lives **inline inside the
notebook that needs it** — this project's established pattern (see below),
not an oversight.

| What | Where it actually is today |
|------|------------------------------|
| Closed-form GS pricer (`GSParams`, `B_coeff`, `A_coeff`, `futures`, `term_structure`) | `src/gs_wamol/physics/gibson_schwartz.py` — the **only** module left in the package |
| ℙ-path simulator (Euler–Maruyama) | Inline in `notebooks/monte_carlo_simulation.ipynb` (no longer a package module) |
| PDE/pricer validation harness | **Gone.** `tests/test_synthetic_gs.py` was deleted along with the rest of `tests/` in the cleanup. There is currently **no automated check** that the closed-form coefficients are still correct — see §3 and §6. |
| PINN networks/loss/training code | Inline in `notebooks/PINN_implementation.ipynb`. The notebook defines its own `Dense`/`MLP` rather than importing a package version (`src/gs_wamol/models/` no longer exists) — this matters because the notebook's networks have no output activation (`δ̂`/`Â` are signed), so a generic package MLP with a forced-positive output would silently be the wrong network. |
| Kalman filter benchmark | `notebooks/kalman_filter.ipynb` — self-contained, inlines its own copy of `B_coeff`/`A_coeff` rather than importing the package, for the same "keep the training/estimation code differentiable and dependency-free" reason as the PINN notebook. |
| Thesis-ready figures | `scripts/figures/make_{mc,real_data,pinn}_figures.py` — see §4's on-disk layout. |

All three figure scripts live together under `scripts/figures/` (not loose in
`scripts/`), so the folder reads as "the thesis-figure pipeline" as a unit if
`scripts/` ever grows non-figure tooling later.

`src/gs_wamol/utils/thesis_style.py` — deleted in the `a8c32b6` cleanup along
with the rest of `gs_wamol/utils/` — has been **restored** (it's
self-contained, no dependency on the other deleted `utils/` modules), so all
three scripts run again.

There is no more `physics/local_vol.py` (legacy Black–Scholes/Dupire
IVS-calibration helpers) either — it was removed in the same cleanup. It was
never used by this thesis anyway (Gibson–Schwartz only), so nothing to
migrate.

**For current project status** (what's trained, what's validated, open
items), see `docs/project_summary.md` (detailed) or
`docs/executive_summary.md` (one page). This file (`CLAUDE.md`) covers only
the durable stuff: what the project is, the math conventions, and pitfalls —
it does not track day-to-day progress.

---

## 1. What this project is

An MSc thesis applying a **Physics-Informed Neural Network (PINN)** to invert
for the latent **convenience yield** $\delta_t$ from observed WTI crude oil
futures term structures, under the **Gibson–Schwartz two-factor model**
(Schwartz 1997, Model 2). The inversion (`notebooks/PINN_implementation.ipynb`)
is benchmarked against a classical **Kalman filter** baseline
(`notebooks/kalman_filter.ipynb`, based on Krul 2008), fit on real WTI data.

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
`src/gs_wamol/physics/gibson_schwartz.py` — the only module left in the
package (§0). Do not "simplify" or rewrite these coefficients without a
PDE-residual check first. **That check no longer exists as runnable code**
(`tests/test_synthetic_gs.py` was deleted in the `a8c32b6` cleanup) — if you
touch these coefficients, either restore that test or re-derive the
float64 PDE-residual check from scratch (§5) before trusting a change.

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
measurement noise**. As of the `a8c32b6` cleanup this whole pipeline lives
**inline in `notebooks/monte_carlo_simulation.ipynb`** (its markdown cell has
a full ASCII flowchart of the steps below) rather than as separate package
modules — only the pricing core is still a package import:

| Step | Where |
|------|-------|
| Closed-form pricer: `GSParams`, `B_coeff`, `A_coeff`, `log_futures`, `futures`, `term_structure` | `src/gs_wamol/physics/gibson_schwartz.py` (package import) |
| `simulate_paths` — correlated Euler–Maruyama under ℙ, returns `(t, X, delta)` with `delta` GROUND TRUTH | Inline in `notebooks/monte_carlo_simulation.ipynb` |
| Price the maturity panel under ℚ with the closed form, add log-price noise | Inline in the same notebook |
| Save `data/input/synthetic/mc_data.pkl` (`S`, `log_F_obs`, `params_P`, `params_Q`, `delta_true`) | Inline in the same notebook |

**Why Option A (closed form) and not nested Monte Carlo:** the closed form gives
*exact* prices given the state, so there is no pricing sampling error. All noise
is introduced deliberately in the observation layer, making signal-to-noise a
single controllable dial. This is required for the identifiability experiments.
Do not replace the closed-form pricing step with nested simulation.

### On-disk layout — where files belong

Data is separated by *provenance* (can it be regenerated by running code?) so
that `.gitignore` can draw a single clean line at `data/output/`. Within
`data/output/`, artefacts are further split by *data source/method* —
`mc_simulated/` for anything trained on the synthetic simulator, `real_data/`
for a PINN run trained on real WTI data via `get_data_wti()` (none exists yet
— create this the first time one does), and `kalman/` for the classical
Kalman filter baseline (`notebooks/kalman_filter.ipynb`), kept separate from
`real_data/` since it's a different estimation method (MLE, not gradient
training) even though both fit on the same real WTI panel. There is
deliberately no flat `data/output/checkpoints/` or `data/output/results/` any
more — every artefact lives under one of these subfolders, never loose at the
top level.

```
data/
  input/
    real/        WTI CSVs from Bloomberg. NOT reproducible from code — always tracked.
    synthetic/   mc_data.pkl, written by notebooks/monte_carlo_simulation.ipynb.
  output/        Model artefacts. GITIGNORED — regenerate by re-running the notebook.
    mc_simulated/
      checkpoints/<run_tag>/    orbax checkpoints, one dir per run_tag(config)
      results/results_<tag>.pkl training history + config, one per MC-trained variant
    real_data/
      checkpoints/<run_tag>/    orbax checkpoints for a PINN trained on real WTI data (none yet)
      results/results_<tag>.pkl matching training history + config
    kalman/
      results/results_kalman.pkl the KF baseline's fitted psi_hat + filtered delta_t path
figures/
  monte_carlo_sim/   written by scripts/figures/make_mc_figures.py
  raw_data/          written by scripts/figures/make_real_data_figures.py;
                      the old notebooks/data_exploration.ipynb this used to come from is deleted
  pinn/              written by scripts/figures/make_pinn_figures.py — NOT by the notebook.
                      PINN_implementation.ipynb's own model-comparison plots render inline
                      only and are deliberately not saved here anymore (they were the source
                      of most of the churn in this directory's history — every rename of the
                      model variants left a new set of orphaned PNGs behind)
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
Currently each contract's $\tau$ is held **fixed across dates** — each real
contract's $\tau$ is averaged down to one static value per maturity "slot."
Real Bloomberg dated contracts (e.g. `CLZ5 Comdty`) have $\tau$ **shrinking
toward delivery**: $\tau = (\text{delivery date} - \text{observation
date})/365$. To support this, replace the fixed per-slot mean $\tau$ vector
with a per-date $\tau$ matrix. The pricing layer already vmaps over $\tau$, so
this is a localized change — but it now has to be made in **two places**
that both currently share the same simplification and should stay in sync:
`get_data_wti` in `notebooks/PINN_implementation.ipynb` and `load_wti_panel`
in `notebooks/kalman_filter.ipynb`.

---

## 5. Numerical precision — important gotcha

JAX defaults to **float32**. In float32:
- The PDE residual floor is ~$10^{-5}$.
- The closed-form initial-condition error is ~$10^{-5}$.

In **float64** (`jax.config.update("jax_enable_x64", True)`) both drop to machine
epsilon (~$10^{-14}$), confirming the analytic coefficients are exact.

**Consequences:**
- Validation / verification code must enable x64, or the checks will appear to
  "fail" when the maths is actually correct. `notebooks/kalman_filter.ipynb`
  enables it globally for exactly this reason (its MLE recursion compounds
  float32 error over ~2700 sequential steps). `PINN_implementation.ipynb`
  currently does **not** enable it — keep that in mind before reading any
  absolute `e_ode`/PDE-residual level out of that notebook as more than "at
  the float32 floor."
- A **float32 PINN cannot drive its PDE-residual loss below ~$10^{-5}$**. This is
  a precision floor, not a training failure. Set convergence tolerances
  accordingly, and consider float64 for residual-sensitive experiments.

---

## 6. Conventions and invariants to preserve when editing

1. **Measure discipline:** ℙ params simulate, ℚ params price. Compute
   $\alpha^{\mathbb{Q}}$ from $\alpha^{\mathbb{P}}$; never conflate them.
2. **Don't touch the closed-form coefficients** without a PDE-residual check
   confirming machine epsilon (float64) — **there is currently no automated
   test for this** (§0, §3); write one before changing `B_coeff`/`A_coeff`,
   don't just eyeball it.
3. **The true $\delta_t$ path is sacred:** retained for evaluation, never an
   input to the inversion.
4. **Pure, jittable functions:** the model functions are written pure for JAX.
   Keep them side-effect-free and vmap/jit-friendly. Pass parameters via the
   frozen `GSParams` dataclass (or the plain-scalar-argument style the PINN
   and Kalman filter notebooks use so `jax.grad` can differentiate through
   them during training/MLE — see each notebook's own pricer cell), not
   globals.
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
- **Krul, A. (2008)**, *Calibration of Stochastic Convenience Yield Models for
  Crude Oil Using the Kalman Filter*, MSc thesis, TU Delft / ING Wholesale
  Banking — the Kalman filter benchmark's own methodological basis (Ch. 4).
- Hoshisashi, Phelan & Barucca (ICAIF 2024) for the WamOL framework this work
  adapts.
