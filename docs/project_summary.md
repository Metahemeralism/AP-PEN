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

## Where things stand today (2026-08-01)

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
- **PINN.** `notebooks/archive/AP-PINN dual network.ipynb` (archived; renamed from
  `PINN_implementation.ipynb` this session) — drift net `Â_θ(τ)` +
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
- **Single-network AP-PINN (new this session, `notebooks/AP-PINN single
  network.ipynb` — since renamed to `notebooks/AP-PEN.ipynb`, see §4's
  note)** — a second architecture: path net only, `A(τ)` and `B(τ)`
  both analytic (no drift net, no `e_ode`). Fixing the market price of
  convenience-yield risk `λ₂` and segregating `ψ` into Q/P optimizer groups
  resolved an identifiability failure that persisted under every other
  configuration tried, including this project's own dual-network model
  (§3.7's `σ₁σ₂ρ` product problem) — `σ₁`/`σ₂`/`ρ` all move meaningfully
  toward truth for the first time. The notebook now also runs end-to-end on
  **real WTI data**, benchmarked against the Kalman filter baseline (§2)
  instead of synthetic ground truth: `APPINN_ARB` reaches corr **+0.888** /
  RMSE **0.051** against the KF's own `δ̂_t`, best of the three real-data
  variants tried. A temporal holdout (train `<2024`, test `≥2024`) shows the
  SDE-regularized variant generalizing almost perfectly out-of-sample (RMSE
  ratio 1.01×) while the plain data-fit model degrades sharply (1.96×) and
  visibly drifts once extrapolating past its trained time domain — direct
  evidence for what the physics term buys, not just an in-sample fit
  improvement. Full narrative in the new §4.
- **`PINN_implementation.ipynb` no longer exists** — renamed to
  `notebooks/AP-PINN dual network.ipynb` this session (the single-network
  notebook above is its new sibling, not a replacement); that dual-network
  notebook has since been archived to `notebooks/archive/AP-PINN dual
  network.ipynb` (see §4's note). `CLAUDE.md`'s file map and cross-references
  were updated accordingly; grep for the old name before trusting any other
  doc that still has it.

### Open items (carried forward, still true as of today)

- **`σ₁`/`ρ` are not separately identifiable in the dual-network model**
  from futures term-structure data at any epoch count — they enter the
  closed-form `A(τ)` only as the product `σ₁σ₂ρ` (§3.7). The single-network
  model's fixed-`λ₂` fix (§4.2) resolves this in *that* architecture, but the
  dual-network notebook itself is untouched — worth stating explicitly in the
  thesis rather than looking like an unconverged run if that model is what
  ships, or porting the same fix across if it isn't.
- **Per-date `τ` is still a fixed mean-per-slot vector**, not the true
  per-date shrinking `τ` real dated contracts have (CLAUDE.md's "Maturity
  handling" extension point). This was already flagged as "on the critical
  path" on 2026-07-19 and hasn't moved — `get_data_wti` in *both* PINN
  notebooks and `kalman_filter.ipynb`'s `load_wti_panel` share the same
  simplification, so all three baselines stay comparable, but none reflects
  real dated-contract roll-down yet.
- **`x64` is still off** in both PINN notebooks — per CLAUDE.md §5 this
  floors any PDE/ODE-residual reading (dual notebook) or price/no-arb-hinge
  reading (either notebook) at ~1e-5. (The Kalman filter notebook does enable
  it.) Turn it on before reading absolute residual levels as anything other
  than "at the float32 floor."
- **The `ψ` naming collision** (WamOL's `ψ=(θ,φ)` — the whole trainable
  network-weight vector — vs. this project's `params["psi"]` — the six
  physical GS constants) is still unresolved in the notebooks' comments.
- **AP-PINN (loss-balanced, no arb) is the current "final" pick for the
  dual-network model**, not yet finalized against AP-PINN (ARB) — the
  no-arbitrage variant is very close in recovery accuracy and may end up
  preferred for the economic-consistency argument alone.
  `scripts/figures/make_pinn_figures.py` now reads the
  `mc_simulated/checkpoints/APPINN` checkpoint (fixed 2026-07-28 — it
  previously read a stale `DCPINN` checkpoint from before this session's
  rename, resolved along with the naming below). Separately, the two PINN
  *architectures* (dual- vs single-network) aren't yet reconciled either —
  see §4's open items.

---

## 1. Data pipeline

- **Synthetic**: `notebooks/monte_carlo_simulation.ipynb` simulates ℙ-measure
  paths, prices under ℚ with the closed form, adds observation noise, and
  saves `data/input/synthetic/mc_data.pkl` — the only place `δ_true` exists,
  and it's retained purely as an evaluation target (CLAUDE.md §1). Figures:
  `scripts/figures/make_mc_figures.py` → `figures/monte_carlo_sim/mc{1-4}_*`.
- **Real**: `data/input/real/wti_{daily_state,futures_panel,analysis_ready}.csv`
  — Refinitiv-sourced, not reproducible from code, tracked as-is. No ground
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

## 4. Single-network AP-PINN — fixed risk premium, real no-arb bounds, first real-data result

**Naming note (later session):** this notebook has since been renamed to
`notebooks/AP-PEN.ipynb` ("AP-PEN" = Affine-Partitioned Physics-Embedded
Network — the model's display name; internal config/variant tags like
`APPINN`/`APPINN_ARB` and on-disk paths were left unchanged, see `CLAUDE.md`).
Its dual-network sibling from §3 has been archived to `notebooks/archive/AP-PINN
dual network.ipynb` and is no longer maintained. The narrative below describes
the notebook as it was at the time of this session, under its original name.

New notebook this session, `notebooks/AP-PINN single network.ipynb`, cloned
from the dual-network notebook (§3) and renamed alongside it (`AP-PINN dual
network.ipynb` ← `PINN_implementation.ipynb`). Original question: does the
drift net `Â(τ)` buy anything over the exact closed form? Short answer: no —
`A(τ)`/`B(τ)` can both be analytic, which also removes `e_ode` outright
(`A_coeff` satisfies its own defining ODE identically for any `ψ`, so the
residual is ~0 by construction, not by training) — but chasing that question
led to fixing a real identifiability problem, a real numerical instability,
and the project's first end-to-end real-data run.

### 4.1 Fixed risk premium + parameter segregation — the identifiability fix

`α^ℚ`/`α^ℙ` had always been learned as fully independent `ψ` leaves, even
though the true model has only one free real-world/risk-neutral gap
(`λ₂`, via `α^ℚ=α^ℙ-σ₂λ₂/κ`, CLAUDE.md §2). That redundant degree of freedom
is what was keeping `σ₁`/`σ₂`/`ρ` pinned near their init values under *every*
configuration tried, in both notebooks (§3.7's `σ₁σ₂ρ` finding is the same
symptom). Fix: freeze `λ₂` at a "historical estimate" (`LAMBDA2_FIXED`, not
learned) and derive `α^ℚ` from it each step, removing `α^ℚ` as an
independent leaf entirely. Paired with **parameter segregation** — `ψ` split
into a Q-group `{κ,σ₁,σ₂,ρ}` and a P-group `{α^ℙ}`, each with its own optax
`TrainState` (plus the path net's own), three gradient steps per epoch
instead of one joint step, so a "frozen" group gets literally zero update
rather than fighting stale Adam momentum.

Validated standalone before folding in, including a robustness sweep
perturbing `λ₂` by ±30% off its true value (oracle case, then two
misestimated cases) — the `σ₁`/`σ₂`/`ρ` recovery gain holds up under a
realistic wrong `λ₂`, not just the oracle one:

| `λ₂` setting | `ρ` (true 0.766) | `σ₂` (true 0.527) | `δ̂` MSE |
|---|---|---|---|
| oracle | 0.80–0.83 | 0.45–0.50 | 0.0042–0.0054 |
| +30% error | 0.80–0.84 | 0.48–0.51 | 0.0052–0.0078 |
| −30% error | 0.77–0.84 | 0.45–0.53 | 0.0047–0.0050 |

vs. every free-`α^ℚ` configuration tried before this (joint or segregated),
where `ρ` sat at 0.44–0.46 (near its perturbed init, true 0.766) regardless
of epoch count.

> **⚠️ CORRECTED 2026-08-07 — this did not resolve the identifiability
> failure; it converted bias into variance.**
>
> The futures term structure identifies only **three** combinations of the
> five ℚ-parameters: `(kappa, sigma2, m)` with
> **`m = alpha_Q + sigma1*sigma2*rho/kappa`**. Substituting `m` into the
> closed form collapses all three `A(tau)` basis coefficients onto those
> three quantities, so `sigma1`, `rho` and `alpha_Q` are *individually*
> non-identifiable from futures prices — exactly, at any number of
> maturities, at zero noise. Confirmed by SVD (two singular values at ~1e-17,
> rank 3 / nullity 2) and by direct invariance test (moving `sigma1` over
> [0.16, 1.24] and `rho` over [−0.94, +0.83] at fixed `m` changes `A(tau)` by
> ≤5.6e-17).
>
> The 20-path repeats show what the fix actually did to `P = sigma1*sigma2*rho`
> (true 0.1586): dual-network **0.0648 ± 0.0092** (stuck at init — large bias,
> tiny variance) → single-network **0.2424 ± 0.1712** (CV **71%** — small bias,
> huge variance). `m` is recovered at CV **3.8%** in *both*. The ρ≈0.80 figure
> in the table above is one draw from a distribution with std 0.36; a ±1σ
> interval for ρ is [0.34, 1.06], wider than ρ's own admissible range.
> See `docs/results_and_discussion.md` §0 and §1.4.

### 4.2 A collapse, deliberately not repeated: `e_sde` into the path net

The architecture-diagram spec called for `e_sde`'s gradient to also train the
path net directly (not just `α^ℙ`), routed via the alternating steps above.
Tested exactly as specified: it collapses `κ→0` and flattens `δ̂_φ(t)` to the
OU conditional mean — the identical failure mode as the dual-network model's
`σ₂→0` variance collapse (§3.6), just reached by a different route (here `κ`
is Q-owned so `σ₂` can't collapse, but the path itself still gets flattened
and `B(τ,κ)≈-1/κ` blows up to compensate in the price fit). Reverted: `e_sde`
stays stop-gradiented from the path net (as in the dual-network model),
confirming this protection generalizes across both architectures rather than
being specific to the dual-network's own derivation.

### 4.3 Literature-grounded no-arbitrage bounds, and a new loss term

The placeholder `STORAGE_COST_U=0.5` (50%/yr) was roughly 5× even the most
generous literature estimate. Replaced with `u=0.10`/yr baseline (Stancu,
Symeonidis, Wese Simen & Zhao 2022 report ~6%/yr average, ~9%/yr in
contango — 0.10 sits safely above both as a ceiling that shouldn't spuriously
bind). `DELTA_MAX=0.75` was already empirically reasonable but re-justified
against Gibson & Schwartz (1990) Table VI's realized +65.5% instantaneous
convenience yield and Figure 2's ~100% turmoil excursions, rather than
resting on an ad hoc Monte Carlo percentile.

Also implemented `e_delta_floor` (`mathematical_derivations-6.pdf` §7.3, eq.
38 — previously listed as "not implemented"): a hinge directly on the latent
path `δ̂_φ(t_i)` at observation times, `DELTA_MIN=-0.3`. Deliberately *not*
the Liu & Tang (2010) no-arbitrage floor of exactly 0 — the derivation itself
argues a hard structural floor there would fight the OU dynamics' Gaussian
marginals and bias `δ̂` upward in exactly the deep-contango regime the model
exists to capture (eq. 39's own reasoning, carried over unchanged).

### 4.4 The no-arb ceiling needed to be time-varying, discovered empirically

Checking the new `u=0.10` ceiling directly against real WTI data (model-free,
no training required — both the cash-and-carry and reverse-cash-and-carry
bounds reference only observed `(F,S,τ,r)`) found it violated on **63%/47%/
21%** of 2015/2016/2020 rows respectively vs **0–2%** every other year
2017–2026. Not spread evenly — a single constant is provably wrong for that
window, not just imprecise. The two crisis clusters are the 2015–16
shale-oversupply contango glut and April 2020 (the day after the negative-WTI
print, where the front-month spot was a physically-distressed, non-
representative price — storage capacity was exhausted that week, so the
cash-and-carry mechanism was mechanically unavailable, not just expensive).

Fix: `STORAGE_COST_U`/`DELTA_MAX` are now `(n,)` arrays via a regime step
function (`is_crisis`, flagged for all of 2015–2016 and April 2020;
`U_CRISIS=0.30`, `DELTA_MAX_CRISIS=1.0`), all-`False`/baseline-only for
MC-simulated data. Cuts real-data violations from 11.93%→2.72%; the residual
2.72% concentrates on the most extreme days inside the crisis window itself
(April 20–21, 2020), which no finite `u` can rationalize for the reason
above. **Caveat for the thesis**: the crisis events themselves are public,
well-documented history (legitimate to use, like any econometric crisis
dummy), but the exact window boundaries and magnitude were cross-checked
against this same evaluation dataset — recommend also reporting the
flat-`u` comparison as a robustness check, not just the regime-aware result.

### 4.5 Real WTI data, benchmarked against the Kalman filter (§2)

First end-to-end real-data run for either PINN notebook. No `δ_true`/`ψ_true`
exists for real data, so the Kalman filter (§2) is the benchmark instead — an
independent, non-PINN estimate from the same panel. Its own MLE-fitted `ψ̂`
also replaces what had been a literature guess: `PSI_INIT` is now the KF's
converged `κ,σ₁,σ₂,ρ,α^ℙ`, and `LAMBDA2_FIXED` is the KF's own fitted `λ≈0.44`
— the actual "historical estimate" §4.1's design wanted, not a placeholder.

First attempt at this diverged completely on `APPINN_ARB` (loss exploding,
`κ` running to 25+, `ρ` pinned at the tanh boundary) — traced to a **missing
WamOL balancer** in the standalone test script used to check it before
folding into the notebook (dropped for speed). With `κ` starting low (0.43,
from the KF fit), `B(τ)≈-1/κ≈-2.3` amplifies price-space gradients into the
path net heavily; without the balancer equalizing channel gradient
magnitudes, that snowballs into a runaway. The notebook's actual
`calibration()` already had the balancer — folding in with the real
production code (not the simplified test script) fixed it outright.

Result — `δ̂_φ(t)` vs. the Kalman filter's own `δ̂_t`, common dates:

| variant | corr vs. KF | RMSE vs. KF | `κ` (KF 0.43) | `ρ` (KF 0.89) |
|---|---|---|---|---|
| MLP | 0.881 | 0.098 | 1.08 | 0.97 |
| APPINN | 0.875 | 0.091 | 1.00 | 0.96 |
| **APPINN_ARB** | **0.888** | **0.051** | 0.54 | **0.90** |

> **⚠️ CORRECTED 2026-08-07 — this table describes a run that no longer
> exists on disk.** `data/output/real_data/results/results_APPINN_ARB.pkl`
> (and the notebook's own cell-41 output) show `APPINN_ARB` **collapsed**:
> `kappa=0.0028`, `sigma1=0.0051`, `sigma2=0.0557`, `rho=-0.820`,
> `alpha_Q=-4.460`, `alpha_P=+4.374`. Current vs-KF figures are MLP corr
> +0.9180 / RMSE 0.1087, APPINN +0.9163 / 0.1094, APPINN_ARB +0.9399 /
> 0.0395.
>
> Two things this section got wrong beyond the numbers:
> 1. **A constant at the KF's own mean scores RMSE 0.1078** — so MLP and
>    APPINN both have *negative skill* on this metric. The high correlations
>    (+0.92) are scale-free and hide the −128%/yr excursion.
> 2. `APPINN_ARB` "winning" is an artefact: the hinges squash its path into
>    the same narrow band the KF occupies while its parameters run to a
>    degenerate corner. The mechanism is a **1/kappa pole** in
>    `alpha_Q = alpha_P - sigma2*lambda2/kappa` — the §4.1 fix and this
>    failure are the same line of code. See `docs/results_and_discussion.md`
>    §2.4–2.5.

`APPINN_ARB`'s RMSE against the KF is roughly half the plain variants' —
and it visibly avoids a real failure the other two don't: `MLP`/`APPINN`
both hit an implausible **-129%/yr** minimum somewhere in the series (far
beyond G&S's own most extreme documented value of -13.7%), which the plot
shows landing exactly on the April 2020 crisis date. `e_delta_floor` (§4.3)
correctly reins this in for `APPINN_ARB` (contained to -15%, `e_delta_floor`
driven to exactly 0) — the constraint doing its intended job on real data,
not just in the synthetic ablation it was designed against.

### 4.6 Temporal holdout — what the SDE term actually buys

Not a classical ML train/test split — there's no observable `δ_t` to hold
out and score against. Instead: calibrate on dates before 2024-01-01, hold
out 2024-01-01 onward (~19% of the panel), and check whether the fitted `ψ`
and path net still price *observed futures* well on unseen later dates. This
also directly measures a limitation the derivation names but doesn't
quantify (`mathematical_derivations-6.pdf` §12.3): the path net is a plain
function of `normalize_time(t)`, so any date past the trained cutoff is
neural extrapolation.

| variant | in-sample RMSE | out-of-sample RMSE | ratio |
|---|---|---|---|
| MLP | 0.031 | 0.061 | **1.96×** |
| **APPINN** | 0.037 | 0.037 | **1.01×** |
| APPINN_ARB | 0.172 | 0.060 | 0.35× |

`APPINN` (the SDE-regularized variant, no hinge constraints) generalizes
almost perfectly. The recovered-path plot shows why: extrapolating past the
cutoff, `MLP` drifts steadily downward with nothing to anchor it, while
`APPINN`/`APPINN_ARB` stay flat — the OU-consistency prior gives the path
something to revert toward even outside its trained domain, which a plain
price-fit objective has no mechanism for. `APPINN_ARB`'s much higher
in-sample RMSE is the same fit-vs-constraint trade-off as §4.5; it doesn't
drift like `MLP` either, since it still carries the same `e_sde` anchor.

> **⚠️ CORRECTED 2026-08-07 — both the numbers and the mechanism above are wrong.**
>
> **Numbers.** A fresh run of the same cells gives MLP 0.02619→0.23201
> (**8.86×**), APPINN 0.02298→0.03875 (**1.69×**), APPINN_ARB
> 0.04243→0.04634 (**1.09×**). The qualitative ordering survives; the
> magnitudes do not.
>
> **Mechanism.** `e_sde` cannot "anchor" the path: `error()` applies
> `lax.stop_gradient(delta_hat)` before computing it, so its gradient into
> the path net is *identically zero*, in-domain and out. `step2b` optimises
> it w.r.t. `alpha_P` only. MLP and APPINN therefore differ by exactly one
> scalar — whether `alpha_P` (hence `alpha_Q = alpha_P - sigma2*lambda2/kappa`,
> hence the level of `A(tau)`) is frozen at init or fitted by OU-MLE. The
> generalisation gap operates through the risk-neutral *level*, not through
> any regularisation of the latent path. See
> `docs/results_and_discussion.md` §4.4.

### Open items specific to this section

- The two PINN architectures (dual-network §3, single-network above) are not
  reconciled — different identifiability behavior (§4.1 fixes what §3.7
  calls structural), different real-data results, no head-to-head run yet.
- `R_FIXED=0.05` is still a constant in the single-network notebook even
  though real `r` ranges 0–5.6% over 2015–2026 (unlike `u`/`δ_max`, this
  wasn't made time-varying this session).
- The crisis-regime lookahead caveat in §4.4 — recommend running the
  flat-`u` comparison before treating the regime-aware real-data numbers in
  §4.5 as final.
- `DELTA_MIN=-0.3` is a single constant, not time-varying like `u`/`δ_max` —
  no literature figure was found to anchor a crisis-regime value to it, but
  April 2020's implied convenience yield is exactly the kind of episode a
  fixed floor might not accommodate; worth checking whether it ever binds
  before trusting it's inert the way `DELTA_MAX_BASELINE` turned out to be.

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
- §4 (single-network AP-PINN) has no separate archive doc — it's a single
  session's work, written directly into this summary at full detail; there's
  no shorter "primary source" to point to underneath it.
