# Project summary — Gibson–Schwartz PINN thesis

**One doc, updated as we go**, replacing the six separate markdown files that
used to live in `docs/`. Their detailed, session-by-session content is
preserved verbatim in `docs/archive/` for provenance; this file is the
current, standing summary — read it first, dig into the archive only when you
need the blow-by-blow of *how* a conclusion below was reached.

For a one-page, non-technical version of this same "where things stand"
picture, see `docs/executive_summary.md`.

See `CLAUDE.md` (repo root) for the durable conventions this doc assumes:
measure discipline (ℙ simulates, ℚ prices), the closed-form pricer, the
on-disk layout, and the float32 precision gotcha.

---

## Where things stand today (2026-07-28)

- **Data.** Two sources feed the model, both behind a shared loader pattern:
  the synthetic Monte Carlo simulator (`data/input/synthetic/mc_data.pkl`,
  ground-truth `δ_t` retained for evaluation only) and the real WTI futures
  panel (`data/input/real/`, no ground truth — this is the actual thesis
  target). See §1.
- **Kalman filter baseline** (`notebooks/kalman_filter.ipynb`, added this
  session) — a classical joint `(ln S_t, δ_t)` Kalman filter fit by MLE on the
  real WTI panel, based directly on Krul (2008). This resolves a
  long-standing open item (§4, "Real WTI has no calibrated GS parameters" —
  first flagged 2026-07-19): the KF now supplies a `ψ̂` and a `δ̂_t` path
  fit on real data, which is both a benchmark for the PINN's inversion and a
  literature-grounded initialization source. See §2.
- **PINN.** `notebooks/PINN_implementation.ipynb` — drift net `Â_θ(τ)` +
  analytic `B(τ;κ)` + path net `δ̂_φ(t)`, GS parameters `ψ` learnable, `e_sde`
  restricted to update `α^ℙ` only (the fix for the variance-collapse failure
  mode in §3.6). The model is now named **AP-PINN** (not DCPINN/AC-PINN —
  those are earlier names from this project's own past naming eras; DCPINN
  was itself renamed to AP-PINN this session). Five variants now trained to
  40k epochs on synthetic data, each a loss-channel superset of the last
  (`e_data` ⊂ `+e_ode` ⊂ `+e_sde` ⊂ `+e_cac,e_rcac`), none collapsing:

  | Variant | Loss channels | `δ̂` RMSE | `δ̂` corr |
  |---|---|---|---|
  | MLP | `e_data` | 0.384 | 0.979 |
  | PINN | `+e_ode` | 0.049 | 0.980 |
  | AP-PINN (nobal) | `+e_sde`, fixed λ=1 | 0.049 | 0.980 |
  | **AP-PINN** | `+e_sde`, WamOL-balanced | **0.049** | **0.980** |
  | AP-PINN (ARB) | `+e_cac,e_rcac` (no-arbitrage hinges) | 0.051 | 0.979 |

  PINN and AP-PINN (nobal) are numerically near-identical by construction:
  `e_sde`'s gradient is stop-gradiented to update only `α^ℙ`, so adding it
  cannot move the drift net, path net, or any other `ψ` component. AP-PINN is
  the current "final" pick (confirmed via Optuna hyperparameter search that
  the vanilla-tuned defaults already used here are ≈ the tuned optimum on a
  corrected, non-gameable `e_data` objective — see §3.9). The no-arbitrage
  variant (AP-PINN ARB) costs almost nothing in recovery accuracy while
  additionally enforcing cash-and-carry/reverse-cash-and-carry consistency.
  A 20-path repeat-validation (independent MC noise draws, each refit from
  scratch) confirms `σ₁`/`ρ` are structurally non-identifiable from the
  futures term structure alone — they enter the closed-form `A(τ)` only as
  the product `σ₁σ₂ρ` (§3.7) — tight std relative to bias across repeats
  rules out this being a noise/undertraining artifact. Full narrative in §3.
- **Figures.** Two-tier pipeline, tightened today: dedicated scripts, now
  grouped together under `scripts/figures/`
  (`make_{mc,real_data,pinn}_figures.py`), write the thesis-ready PDF+PNG
  pairs into `figures/`, styled via `gs_wamol.utils.thesis_style` (also
  restored today after being caught up in the same cleanup commit).
  Ad hoc model-comparison plots inside the notebooks now render inline only
  (`plt.show()`) and are **not** written to `figures/` — they were stacking
  up as orphaned files every time the model-naming scheme churned (see the
  `figures/pinn/` history in §3's ACPINN→DCPINN rename). The three composite
  comparison PNGs already on disk (`training_history_composite.png`,
  `psi_trajectories.png`, `delta_recovery_composite.png`) are a frozen
  snapshot of the last comparison run, not auto-regenerated; delete them
  whenever they go stale.
- **Docs.** This consolidation. `docs/archive/` holds the five source docs
  (`pinn_master_history.md`, `pinn_implementation_provenance.md`,
  `dcpinn_vs_gs_pinn_comparison.md`, `pinn_rewrite_todo.md`,
  `pinn_session_findings_2026-07-19.md`, `pinn_session_findings_2026-07-22.md`)
  unchanged, for anyone who wants the detailed provenance behind §3.

### Open items (carried forward, still true as of today)

- **`σ₁`/`ρ` are not separately identifiable** from futures term-structure
  data in this model, at any epoch count — they enter the closed-form `A(τ)`
  only as the product `σ₁σ₂ρ` (§3.7). Worth stating explicitly in the thesis
  rather than looking like an unconverged run. Not yet fixed: pin one
  externally, or reparameterise to learn the identified composite directly.
- **Per-date `τ` is still a fixed mean-per-slot vector**, not the true
  per-date shrinking `τ` real dated contracts have (CLAUDE.md's "Maturity
  handling" extension point). This was already flagged as "on the critical
  path" on 2026-07-19 and hasn't moved — both `PINN_implementation.ipynb`'s
  `get_data_wti` and `kalman_filter.ipynb`'s `load_wti_panel` share the same
  simplification, so the two baselines stay comparable, but neither reflects
  real dated-contract roll-down yet.
- **`x64` is still off** in `PINN_implementation.ipynb` — per CLAUDE.md §5
  this floors any PDE/ODE-residual reading at ~1e-5. (The Kalman filter
  notebook does enable it.) Turn it on before reading absolute residual
  levels as anything other than "at the float32 floor."
- **The `ψ` naming collision** (WamOL's `ψ=(θ,φ)` — the whole trainable
  network-weight vector — vs. this project's `params["psi"]` — the six
  physical GS constants) is still unresolved in the notebook's comments.
- **AP-PINN (loss-balanced, no arb) is the current "final" pick**, not yet
  finalized against AP-PINN (ARB) — the no-arbitrage variant is very close in
  recovery accuracy and may end up preferred for the economic-consistency
  argument alone. `scripts/figures/make_pinn_figures.py` now reads the
  `mc_simulated/checkpoints/APPINN` checkpoint (fixed 2026-07-28 — it
  previously read a stale `DCPINN` checkpoint from before this session's
  rename, resolved along with the naming below).

---

## 1. Data pipeline

- **Synthetic**: `notebooks/monte_carlo_simulation.ipynb` simulates ℙ-measure
  paths, prices under ℚ with the closed form, adds observation noise, and
  saves `data/input/synthetic/mc_data.pkl` — the only place `δ_true` exists,
  and it's retained purely as an evaluation target (CLAUDE.md §1). Figures:
  `scripts/figures/make_mc_figures.py` → `figures/monte_carlo_sim/mc{1-4}_*`.
- **Real**: `data/input/real/wti_{daily_state,futures_panel,analysis_ready}.csv`
  — Bloomberg-sourced, not reproducible from code, tracked as-is. No ground
  truth `δ_t` exists for real data; that's the entire reason a PINN inversion
  and a Kalman filter baseline both need to be judged some other way (curve
  fit quality, parameter plausibility, cross-model agreement). Figures:
  `scripts/figures/make_real_data_figures.py` → `figures/raw_data/real_data_overview.*`.

## 2. Kalman filter baseline (this session)

`notebooks/kalman_filter.ipynb`, based on Krul, A. (2008), *Calibration of
Stochastic Convenience Yield Models for Crude Oil Using the Kalman Filter*
(MSc thesis, TU Delft / ING Wholesale Banking), Chapter 4.

- Joint state `(x_t, δ_t)`, `x_t=ln S_t`, both latent and inferred purely from
  the futures curve (real spot is used only to initialize `x_0`).
- Euler-Maruyama ℙ-transition (matches this project's own synthetic simulator
  convention), exact affine GS closed-form ℚ-measurement — linear in the
  state, so a plain (not Extended) Kalman filter is exact.
- `λ` (market price of convenience-yield risk) estimated directly so
  `α^ℚ=α^ℙ-σ₂λ/κ` (CLAUDE.md §2) is enforced as a constraint, not fit as two
  independently free parameters.
- Fit by MLE (prediction-error decomposition log-likelihood) via `jax.grad` +
  L-BFGS-B, on the full 2015–2026 real WTI panel.
- Result: `κ̂=0.43, σ̂₁=0.44, σ̂₂=0.17, ρ̂=0.89, α̂^ℚ=-0.077`; filtered `δ̂_t`
  stays in a plausible `[-0.29, 0.23]` range; futures-curve-only inferred
  spot tracks real WTI cash prices to ~6.9% RMSE as a sanity check.
- Saves `data/output/kalman/results/results_kalman.pkl` for downstream
  comparison against the PINN's own `δ̂_φ(t)`.

## 3. PINN model development

*(Adapted from the former `pinn_master_history.md`, which reconstructed this
narrative from `docs/archive/pinn_implementation_provenance.md`,
`docs/archive/pinn_rewrite_todo.md`,
`docs/archive/dcpinn_vs_gs_pinn_comparison.md`, and the two
`pinn_session_findings_*.md` docs — see the archive for the primary sources.
One gap is flagged honestly below: between the 07-19 findings and the 07-22
cleanup, the architecture changed a second time with **no doc recording the
transition**; §3.3 reconstructs it by diffing the two docs against the
notebook as it exists, and should be read as inference, not witnessed record.)*

### Timeline at a glance

| Era | What the model was | Why it ended |
|---|---|---|
| 3.1 Ancestor | AC-PINN/DCPINN on SABR vol surfaces (Hoshisashi et al.) | Different problem entirely; ported as scaffolding only |
| 3.2 Pre-07-19 | One net `δ̂(t,S)`, GS closed form composed in and differentiated | PDE residual is zero *by construction* for any δ̂ — degenerates to penalised least squares |
| 3.2 07-19 rewrite | Two free nets: surface `F̂_θ(S,δ,τ)` + path `δ̂_φ(t)`, closed form excluded entirely, arbitrage-inequality constraints | The cash-and-carry ceiling contradicts the true contango states — collapses the path; confirmed on real WTI too |
| 3.3 *(undocumented)* | Drift net `Â(τ)` + **analytic** `B(τ)` + path net; arbitrage terms replaced by `e_ode` (ODE residual) + `e_sde` (OU transition NLL) | Not abandoned — **this is the architecture used for the rest of this section and still in the notebook today** |
| 3.5–3.7 | Same skeleton, diagnosed term-by-term, then GS parameters ψ made learnable jointly with the networks | Landed at a working three-model comparison with a documented identifiability boundary |

### 3.1 Ancestor: the AC-PINN/DCPINN SABR notebook

The starting point was a colleague's notebook (Hoshisashi et al., ICAIF 2024
WamOL framework) that calibrated a **Dupire local-volatility surface** from
SABR-simulated option prices: one network `σ_θ(K,τ) → ℝ⁺` fed into a
closed-form Black–Scholes price, differentiated for a Dupire-PDE residual,
plus three no-arbitrage inequality terms on the call-price surface.

**What carried over wholesale** (verbatim or near-verbatim, none of it
problem-specific): the `Dense`/`MLP` network primitives, the Adam +
exponential-decay optimiser, the WamOL gradient-norm loss-balancing machinery,
orbax checkpointing, and the `error() → dict → weighted sum` loss
architecture.

**What didn't transfer at all**: every financial primitive (`bs`, `black`,
`lv_sqr`, `SabrVolHagan`, the implied-vol Newton solver, ~350 lines total) —
none of it has a Gibson–Schwartz analogue and all of it was deleted.

### 3.2 Pre-07-19 broken state → the 07-19 two-network rewrite

**What was broken.** The notebook inherited before 07-19 mirrored the SABR
structure too literally: one network `δ̂(t,S)`, fed into the **GS closed
form** `S·exp(B(τ)δ̂+A(τ))`, differentiated to get a PDE residual. Session
findings measured this directly: the "self-consistent" residual (δ treated
as an independent variable) sits at machine epsilon (**4.73e-13**), while the
notebook's own composed version reported **3.68e+02** — the entire non-zero
value was a derivative-consistency artefact (`∂δ̂/∂S` leaking into `F_S`),
not physics. Raw unnormalised inputs (`S∈[36,165]`, `t∈[0,10]`) also
saturated 96% of the first tanh layer, so `δ̂` was flat to `5e-3` across the
whole panel at init.

**Root cause, stated precisely:** the GS exponential-affine solution solves
its own pricing PDE *identically for every value of δ*. Composing it with a
network and differentiating can never produce a non-trivial residual — the
method degenerates to penalised least squares no matter how it's dressed up.

**The rewrite (07-19).** The fix mandated by the revised derivation
(`mathematical_derivations-6.pdf`, §6): split into **two independently free
networks**, and remove the closed form from the training objective entirely.

- **Surface net** `F̂_θ(S,δ,τ) → ℝ⁺` — no longer fed through the closed form.
  Positivity and the initial condition `F̂(S,δ,0)=S` both came for free from
  the output transform `F̂_θ = S·exp(τ·N_θ)`, so the undefined `L_b` term was
  never needed.
- **Path net** `δ̂_φ(t) → ℝ` — deliberately **t alone**, no `S` input, per the
  derivation's explicit warning that admitting `S` "would let the network
  launder price information into the yield estimate."
- **PDE residual** (`e_pde`) differentiated the *free surface net* on a
  **separate collocation mesh** over `(S,δ,τ)` with δ sampled as an
  independent coordinate, giving the residual real physics content.
- **Three inequality terms** priced through the surface net: cash-and-carry
  ceiling (`e_arb_cac`), reverse-cash-and-carry floor (`e_arb_rcac`), a floor
  on the path itself (`e_delta_floor`). Named `ACPINN`.

**Result:** PINN (data + PDE only) recovered the path at RMSE 0.172 / corr
+0.72, close to a measured identifiability floor of ~0.10–0.14 set by
`∂ln F̂/∂δ ≈ B(τ)`, which vanishes as τ→0. **ACPINN collapsed to the no-skill
baseline** (RMSE 0.246 = std of the truth, corr +0.10).

**Diagnosing the collapse.** Two hypotheses tested by controlled ablation:
1. WamOL balancer — refuted (forcing it off with fixed λ=1 collapsed just as
   hard).
2. A single inequality term — confirmed: `e_arb_cac` alone reproduced the
   full collapse; `rcac` was inert; `e_delta_floor` alone *helped* slightly.

**Root cause:** at `STORAGE_COST_U=0`, the cac ceiling maps to a lower bound
`δ≥−0.0127` at the tightest maturity — but the true path dips below that on
35.3% of dates. **In Gibson–Schwartz, negative convenience yield *is* the
model's encoding of storage cost; a cash-and-carry ceiling at u=0 forbids the
exact states the δ-dynamics produce.** Validated model-free against real
WTI: the same ceiling is violated by **31.9%** of actual 2015–2026 contracts
— not a simulator artefact. The cac-term fix was **left deferred** at the end
of that session.

### 3.3 The undocumented pivot: drift net + analytic B(τ)

**No session doc records this transition.** It happened between the 07-19
findings and the 07-22 cleanup, which opens by describing the drift/path net
split as already the *existing* state. What follows is reconstructed by
diffing the two docs and the notebook as it exists — read as inference.

| §3.2's design (07-19) | Reconstructed current design |
|---|---|
| Surface net `F̂_θ(S,δ,τ)→ℝ⁺`, full 3-D pricing function | **Drift net** `Â_θ(τ)`, function of **τ alone** |
| Closed form fully excluded from training | **`B(τ)` reintroduced analytically** — only `A(τ)` stays learned |
| `e_arb_cac`, `e_arb_rcac`, `e_delta_floor` (3 inequality terms) | **`e_ode`** (ODE residual tying `Â'(τ)` to the analytic RHS) **+ `e_sde`** (OU transition NLL under ℙ) |
| Named `ACPINN` | Renamed **`DCPINN`** (Derivative-Constrained PINN) |

This is a plausible, well-motivated redesign given what the collapse
diagnosis found: the inequality constraints were the specific failure
mechanism, priced through a full 3-D surface net that the ODE/SDE
reformulation no longer needs. Reintroducing `B(τ)` analytically also makes
the ODE residual a statement about `A(τ)` alone, which is what let this
session isolate each loss term's effect so cleanly (§3.5).

The pricing transform, used by every subsequent section:

$$\log \hat F(t,\tau) = \log S + B(\tau)\,\hat\delta_\phi(t) + \hat A_\theta(\tau), \qquad B(\tau) = -\frac{1-e^{-\kappa\tau}}{\kappa}$$

with three loss channels: `e_data` (price MSE), `e_ode` (ties `Â'(τ)` to the
analytic ODE RHS derived from `κ,α^ℚ,σ₁,σ₂,ρ`), `e_sde` (OU transition NLL
disciplining the path's dynamics under ℙ).

### 3.4 The 07-22 cleanup (no architecture change)

A hygiene pass on the already-pivoted architecture: removed dead code, fixed
a real silent bug (`ann_reparam` was compared against the string
`"weight_fact"` but every config set it as a **boolean**, so
`True == "weight_fact"` was always `False` — the reparameterisation feature
had never actually activated), and removed the second-level per-point
self-adaptive loss weighting inherited from the SABR original (kept only the
component-level WamOL balancing).

### 3.5 Isolating which loss terms actually help

Starting point: `num_epochs=1000` default, GS parameters still **hardcoded
constants** — nothing in `ψ` was learnable yet.

- **Capacity test.** Path net trained alone on plain MSE against `δ_true`: at
  40k epochs, corr **0.9665**, R²=0.934. Capacity was never the bottleneck.
- **Pricing-channel test.** Drift frozen at exact `A_coeff`: corr **0.963**.
  Drift free, trained jointly: corr **0.924**. `B(τ)`'s attenuation
  (`|B|≤1/κ≈0.53`) doesn't matter — Adam absorbs the constant gradient scale.
- **The "~0.68 ceiling" was undertraining, not a wall.** `num_epochs=1000`
  was a smoke-test setting; trained to 40k, corr **0.918**, still rising
  slowly (spectral bias — high-frequency path structure fits late). Any
  reading of δ-recovery at the 1000-epoch default is not representative.
- **`e_ode` is inert for path recovery.** PINN (`e_data`+`e_ode`, 45k): corr
  **0.919** ≈ MLP's 0.918 (`e_data` only). `e_ode` is a pure function of the
  drift net alone and never touches the path net.
- **`e_sde` (fixed ψ) actively hurts in this data regime.** DCPINN corr
  **0.899** < PINN's **0.930**; recovered path std collapses (truth 0.246 →
  DCPINN 0.185 → DCPINN_nobal 0.070) as SDE influence grows. The OU
  transition NLL has no term rewarding innovation variance — minimising it
  drags `δ̂` toward the smooth conditional mean. With 12 maturities and low
  noise, the data already pins δ and a smoothness prior can only remove
  signal (it only helps in an ill-posed, few/noisy-contract regime).

### 3.6 Making ψ learnable — a collapse, and its fix

**The design.** Six GS parameters made learnable: `κ,σ₁,σ₂,ρ,α^ℚ,α^ℙ`
(`r` held fixed). Constrained by reparameterisation (softplus for the three
positive scalars, tanh for ρ). `κ` and `σ₂` are **shared across measures**
(confirmed directly from `mc_data.pkl`) — a single learned leaf feeds both
the ℚ pricing side and the ℙ transition side; only `α` splits into `α^ℚ`/`α^ℙ`.

**First run: DCPINN collapses catastrophically.** MLP corr 0.931, PINN corr
0.907, **DCPINN corr 0.259**. `σ₂: 0.685 → 0.0001`, path flattening onto a
near-constant line.

**Mechanism — variance collapse.** The OU transition NLL contains
`0.5·log(2π·var)` with `var=σ₂²/(2κ)·(1−e^{−2κΔt})`. As `σ₂→0`, `log(var)→
−∞`: an unbounded-below free lunch. Textbook variance collapse (the same
singularity as free-variance GMMs/VAEs) — invisible in every earlier
experiment purely because `σ₂` had always been a fixed constant.

**Fix 1 — stop-gradient the path inside `e_sde`.** `e_sde` shouldn't be able
to reshape the path to make its own likelihood better. Result: DCPINN
δ-recovery **0.259 → 0.938**. But this exposed a second degeneracy: with the
path fixed, the OU-MLE hits a random-walk boundary (`κ→0`), and because `κ`
is a *shared* leaf, this dragged the well-identified pricing-side `κ` down
too (MLP/PINN held `κ≈1.8`; DCPINN corrupted it to `κ≈0.008–0.07`).

**Fix 2 (final) — restrict `e_sde` to update only `α^ℙ`.** Since `κ` and
`σ₂` are already well-identified by the pricing channels, and `α^ℙ` is the
one parameter no other channel touches, stop-gradient `κ` and `σ₂` *inside*
the SDE NLL too — it still *reads* them, can no longer *write* them.

**Standalone validation, 40k epochs:**

| | κ (true 1.876) | σ₂ (true 0.527) | α^ℙ (true 0.106) | δ-recovery corr |
|---|---|---|---|---|
| MLP | 1.768 | 0.685 (frozen at init) | 0.046 (frozen at init) | 0.931 |
| PINN | 1.806 | 0.499 | 0.046 (frozen at init) | 0.907 |
| **DCPINN** | **1.644** (stable) | 0.365 | **0.094** (toward truth) | **0.935** |

**Ported into the notebook, 10k epochs (the state the notebook is in today):**

| | RMSE | κ | α^ℙ |
|---|---|---|---|
| MLP | 0.136 | 1.874 | 0.046 (frozen) |
| PINN | 0.061 | 1.830 | 0.046 (frozen) |
| **DCPINN** | **0.054** (lowest) | 1.809 (stable) | 0.076 (converging) |

At 10k epochs DCPINN is the best-performing model, and κ never breaks away
from the pricing-identified value.

### 3.7 What's still structurally unidentified (not a bug — a property of the model)

`σ₁` and `ρ` sit near their init values in every run above and don't move
meaningfully toward truth, no matter how long training runs. Checked directly
against the closed-form `A(τ)` coefficient (CLAUDE.md §3):

$$A(\tau) = \Big(r-\alpha^{\mathbb Q}+\tfrac12\tfrac{\sigma_2^2}{\kappa^2}-\tfrac{\sigma_1\sigma_2\rho}{\kappa}\Big)\tau + \tfrac14\sigma_2^2\tfrac{1-e^{-2\kappa\tau}}{\kappa^3} + \Big(\alpha^{\mathbb Q}\kappa+\sigma_1\sigma_2\rho-\tfrac{\sigma_2^2}{\kappa}\Big)\tfrac{1-e^{-\kappa\tau}}{\kappa^2}$$

**`σ₁` never appears alone anywhere in this formula — only as the product
`σ₁σ₂ρ`.** This is a structural fact about the Gibson–Schwartz futures price
itself (checked and confirmed the exact closed form has the same
product-only dependence — switching from the free drift net wouldn't fix
it). Futures term-structure data alone cannot separate `σ₁` from `ρ`; only
their covariance-term product is observable, in any architecture. Two honest
ways forward, not yet implemented: fix one of `{σ₁,ρ}` externally and let the
other absorb the whole product, or reparameterise to learn the identified
composite quantity directly.

---

## Where to look for detail

- `docs/archive/pinn_master_history.md` — the original, unedited version of
  §3 above, including its own source-attribution footnotes.
- `docs/archive/pinn_implementation_provenance.md`,
  `docs/archive/dcpinn_vs_gs_pinn_comparison.md` — line-by-line diffs against
  the SABR ancestor notebook.
- `docs/archive/pinn_rewrite_todo.md` — the action list for the 07-19
  two-network rewrite (superseded by §3.3's pivot; kept for the surface-net
  design's own record).
- `docs/archive/pinn_session_findings_2026-07-19.md`,
  `docs/archive/pinn_session_findings_2026-07-22.md` — full session logs,
  including the real-WTI cac-violation check (§3.2) and file-touched lists.
- `docs/mathematical_derivations-6.pdf` — the derivation this architecture
  implements.
- `CLAUDE.md` — durable repo conventions (read this first, always).
