# PINN master history — from broken notebook to working DC-PINN

A single narrative spine through `notebooks/PINN_implementation.ipynb`'s entire
development: four architectural eras, why each one was abandoned or kept, and
the specific failures and fixes that produced the model as it stands today
(2026-07-25).

**Sources.** The first three eras are reconstructed from the existing session
docs in this folder — `pinn_implementation_provenance.md`,
`pinn_rewrite_todo.md`, `dcpinn_vs_gs_pinn_comparison.md`,
`pinn_session_findings_2026-07-19.md`, `pinn_session_findings_2026-07-22.md`.
The fourth era (§4 onward) is this session's work and has no separate
findings doc yet — it is written up here directly, with numbers pulled from
the actual runs (standalone validation scripts and the notebook's own saved
outputs), not reconstructed from memory.

**One gap, flagged honestly.** Between the 07-19 findings doc and the 07-22
cleanup doc, the architecture changed a second time — from the "two free
networks, no closed form, arbitrage-inequality constraints" design of §2 to
the "drift net + analytic B(τ) + ODE/SDE physics terms" design that is the
starting point of everything else in this document. **No doc records that
transition.** §3 reconstructs what changed by diffing the two findings docs
against each other and against the notebook as it exists today; treat it as
inference, not as something directly witnessed the way §4–§7 are.

---

## Timeline at a glance

| Era | What the model was | Why it ended |
|---|---|---|
| §1 Ancestor | AC-PINN/DCPINN on SABR vol surfaces (Hoshisashi et al.) | Different problem entirely; ported as scaffolding only |
| §2 Pre-07-19 | One net `δ̂(t,S)`, GS closed form composed in and differentiated | PDE residual is zero *by construction* for any δ̂ — degenerates to penalised least squares |
| §2 07-19 rewrite | Two free nets: surface `F̂_θ(S,δ,τ)` + path `δ̂_φ(t)`, closed form excluded entirely, arbitrage-inequality constraints | The cash-and-carry ceiling contradicts the true contango states — collapses the path; confirmed on real WTI too |
| §3 *(undocumented)* | Drift net `Â(τ)` + **analytic** `B(τ)` + path net; arbitrage terms replaced by `e_ode` (ODE residual) + `e_sde` (OU transition NLL) | Not abandoned — **this is the architecture used for the rest of the document and still in the notebook today** |
| §4–§7 This session | Same skeleton, diagnosed term-by-term, then GS parameters ψ made learnable jointly with the networks | Landed at a working three-model comparison with a documented identifiability boundary |

---

## §1. Ancestor: the AC-PINN/DCPINN SABR notebook

*(`dcpinn_vs_gs_pinn_comparison.md`)*

The starting point was a colleague's notebook (Hoshisashi et al., ICAIF 2024
WamOL framework) that calibrated a **Dupire local-volatility surface** from
SABR-simulated option prices: one network `σ_θ(K,τ) → ℝ⁺` fed into a
closed-form Black–Scholes price, differentiated for a Dupire-PDE residual,
plus three no-arbitrage inequality terms on the call-price surface
(`∂C/∂K`, `∂²C/∂K²`, `∂C/∂T` bounds).

**What carried over wholesale** (verbatim or near-verbatim, none of it
problem-specific): the `Dense`/`MLP` network primitives, the Adam +
exponential-decay optimiser, the WamOL gradient-norm loss-balancing machinery
(`update_loss_weights`, the whack-a-mole λ-update schedule), orbax
checkpointing, and the `error() → dict → weighted sum` loss architecture.

**What didn't transfer at all**: every financial primitive (`bs`, `black`,
`lv_sqr`, `SabrVolHagan`, the implied-vol Newton solver, ~350 lines total) —
none of it has a Gibson–Schwartz analogue and all of it was deleted.

---

## §2. Pre-07-19 broken state → the 07-19 two-network rewrite

### 2a. What was broken

*(`pinn_implementation_provenance.md`, `pinn_session_findings_2026-07-19.md` §1)*

The notebook inherited before 07-19 mirrored the SABR structure too literally:
one network `δ̂(t,S)`, fed into the **GS closed form** `S·exp(B(τ)δ̂+A(τ))`,
differentiated to get a PDE residual. Session findings measured this directly:
the "self-consistent" residual (δ treated as an independent variable) sits at
machine epsilon (**4.73e-13**), while the notebook's own composed version
reported **3.68e+02** — the entire non-zero value was a derivative-consistency
artefact (`∂δ̂/∂S` leaking into `F_S`), not physics. On top of that, raw
unnormalised inputs (`S∈[36,165]`, `t∈[0,10]`) saturated 96% of the first
tanh layer, so `δ̂` was flat to `5e-3` across the whole panel at init.

**Root cause, stated precisely:** the GS exponential-affine solution solves
its own pricing PDE *identically for every value of δ*. Composing it with a
network and differentiating can never produce a non-trivial residual — the
method degenerates to penalised least squares no matter how it's dressed up.

### 2b. The `-6.pdf` rewrite (07-19)

*(`pinn_rewrite_todo.md`, `pinn_session_findings_2026-07-19.md` §2–§4)*

The fix mandated by the revised derivation (`mathematical_derivations-6.pdf`,
§6): split into **two independently free networks**, and remove the closed
form from the training objective entirely.

- **Surface net** `F̂_θ(S,δ,τ) → ℝ⁺` — the pricing surface itself, no longer
  fed through the closed form. Positivity and the initial condition
  `F̂(S,δ,0)=S` both came for free from the output transform
  `F̂_θ = S·exp(τ·N_θ)` (eq. 47), so the `L_b` term the derivation left
  undefined was simply never needed.
- **Path net** `δ̂_φ(t) → ℝ` — deliberately **t alone**, no `S` input, per the
  derivation's explicit warning that admitting `S` "would let the network
  launder price information into the yield estimate."
- **PDE residual** (`e_pde`) now differentiated the *free surface net* on a
  **separate collocation mesh** over `(S,δ,τ)` with δ sampled as an
  independent coordinate — not `δ̂_φ(t)` — so the residual had real physics
  content instead of being trivially zero.
- **Three inequality terms**, priced through the surface net: a cash-and-carry
  ceiling (`e_arb_cac`), a reverse-cash-and-carry floor (`e_arb_rcac`), and a
  floor on the path itself (`e_delta_floor`). This variant was named `ACPINN`.

**Result:** PINN (data + PDE only) recovered the path at RMSE 0.172 / corr
+0.72, close to a measured identifiability floor of ~0.10–0.14 set by
`∂ln F̂/∂δ ≈ B(τ)`, which vanishes as τ→0 (short tenors carry almost no
information about δ). **ACPINN collapsed to the no-skill baseline**
(RMSE 0.246 = std of the truth, corr +0.10).

### 2c. Diagnosing and confirming the ACPINN collapse

*(`pinn_session_findings_2026-07-19.md` §4–§5)*

Two hypotheses were tested by controlled ablation:

1. **WamOL balancer** — refuted. Running ACPINN with the balancer forced off
   (fixed λ=1) collapsed *just as hard* (RMSE 0.245, std(δ̂) 0.0095).
2. **A single inequality term** — confirmed. Adding each of the three terms to
   PINN individually: `e_arb_cac` alone reproduced the full collapse; `rcac`
   was inert (its bound never bound); `e_delta_floor` alone *helped* slightly.

**Root cause:** at `STORAGE_COST_U = 0`, the cac ceiling maps to a lower bound
`δ ≥ −0.0127` at the tightest maturity — but the true path dips below that on
35.3% of dates. The penalty forces δ̂ up until it flattens at the truth's mean
with its downswings clipped. Clearing the observed minimum honestly would need
`u ≈ 0.45–0.90`/yr — economically absurd for crude. **In Gibson–Schwartz,
negative convenience yield *is* the model's encoding of storage cost; a
cash-and-carry ceiling at u=0 forbids the exact states the δ-dynamics
produce.** This was then validated model-free (no network, no GS parameters
at all) against real WTI settles: the same ceiling is violated by **31.9%** of
actual 2015–2026 contracts, confirming it wasn't a simulator artefact.

The cac-term decision was **left deferred by the user** at the end of that
session — no fix chosen yet.

---

## §3. The undocumented pivot: drift net + analytic B(τ), physics terms replace inequality constraints

**No session doc records this transition.** It happened sometime between the
07-19 findings doc and the 07-22 cleanup doc, which opens by describing "the
identical drift net (`A_hat_theta(tau)`), path net (`delta_hat_phi(t)`)" as
the *existing* state — i.e. the pivot from §2's surface-net design had
already happened by the time that session started. What follows is
reconstructed by diffing the two docs and the current notebook; it should be
read as inference, not as a witnessed record.

**What changed, concretely:**

| §2's design (07-19) | Reconstructed current design |
|---|---|
| Surface net `F̂_θ(S,δ,τ)→ℝ⁺`, full 3-D pricing function | **Drift net** `Â_θ(τ)`, function of **τ alone** |
| Closed form fully excluded from training | **`B(τ)` reintroduced analytically** — only `A(τ)` stays learned |
| `e_arb_cac`, `e_arb_rcac`, `e_delta_floor` (3 inequality terms) | **`e_ode`** (ODE residual tying `Â'(τ)` to the analytic RHS) **+ `e_sde`** (OU transition NLL on the recovered path under ℙ) |
| Named `ACPINN` | Renamed **`DCPINN`** (Derivative-Constrained PINN — the notebook's current title) |

This is a plausible, well-motivated redesign given what §2c found: the
inequality constraints were the specific mechanism of collapse, and they were
priced through a full 3-D surface net that the ODE/SDE reformulation no
longer needs. Replacing "forbid price states that violate a static bound"
with "penalise inconsistency with the model's own dynamics" sidesteps the
exact failure mode §2c diagnosed — a bound that can flatly contradict the
true data-generating process. Reintroducing `B(τ)` analytically also removes
one whole source of freedom (the surface net no longer has to learn
`∂F̂/∂δ` from scratch) and makes the ODE residual a *statement about `A(τ)`
alone*, which is what let this session isolate each loss term's effect so
cleanly (§4).

The pricing transform is now:

$$\log \hat F(t,\tau) = \log S + B(\tau)\,\hat\delta_\phi(t) + \hat A_\theta(\tau), \qquad B(\tau) = -\frac{1-e^{-\kappa\tau}}{\kappa}$$

with the three loss channels: `e_data` (price MSE), `e_ode` (ties `Â'(τ)` to
the analytic ODE RHS derived from `κ,α^Q,σ₁,σ₂,ρ`), `e_sde` (OU transition
NLL disciplining the path's dynamics under ℙ). **This is the architecture
every subsequent section works with.**

---

## §4. The 07-22 cleanup (no architecture change)

*(`pinn_session_findings_2026-07-22.md`)*

A pure hygiene pass on the already-pivoted architecture: removed dead code
(unused `ModifiedMLP`, the superseded `ann_gen`, unused config fields, unused
imports), fixed a real silent bug (`ann_reparam` was compared against the
string `"weight_fact"` but every config set it as a **boolean**, so
`True == "weight_fact"` was always `False` — the reparameterisation feature
had never actually activated), and removed the second-level *per-point*
self-adaptive loss weighting inherited from the SABR original (kept only the
component-level WamOL balancing). Confirmed the optimiser was unaffected
(Adam + exponential decay, unchanged) and clarified PINN vs DCPINN differ
only in loss composition (`e_data`+`e_ode` vs. `+e_sde`) and in which variant
the balancer runs for (`DCPINN` only).

---

## §5. This session — isolating which loss terms actually help

Starting point: the §3 architecture, `num_epochs=1000` by default, GS
parameters (`κ, σ₁, σ₂, ρ, α^Q, α^P`) still **hardcoded constants** read
straight from `mc_data.pkl` — nothing in `ψ` was learnable yet.

### 5a. Capacity test — is the path net even big enough?

Trained the path net **alone** on plain MSE against `δ_true` — no pricing, no
physics. At 40k epochs: **corr 0.9665, R²=0.934**. The target is genuinely
hard (`T·κ ≈ 18.8` mean-reversion lengths over the 10-year window), but a
tanh 3×32 net (2,209 params) gets there anyway. **Capacity was never the
bottleneck.**

### 5b. Pricing-channel test — does `B(τ)δ` transmit enough gradient?

Two conditions, both `e_data` only: (A) drift **frozen** at the exact
analytic `A_coeff` — corr **0.963**, essentially matching the direct-MSE
baseline; (B) drift **free**, trained jointly — corr **0.924** (a small cost
from δ↔A aliasing, not a collapse). **The `B(τ)` attenuation
(`|B|≤1/κ≈0.53`) doesn't matter** — Adam's per-parameter step-size
normalisation absorbs a constant gradient scale factor completely.

### 5c. The "~0.68 ceiling" was undertraining, not a wall

The notebook's default `num_epochs=1000` was a smoke-test setting. Corr at
epoch 1000 **is** ~0.69 — that's not a ceiling, it's a snapshot mid-climb.
Trained to 40k with the identical setup: **corr 0.918, R²=0.842**, still
rising slowly (spectral bias: the net fits the path's high-frequency
structure late, over tens of thousands of epochs). **Any reading of this
model's δ-recovery at the 1000-epoch default is not representative.**

### 5d. `e_ode` is inert for path recovery

PINN (`e_data`+`e_ode`, λ=1, 45k): corr **0.919** ≈ MLP's 0.918 (`e_data`
only). `e_ode` collapses to ~1e-8 by epoch ~8000 and stays there — it's a
pure function of the **drift net only** (`Â(τ)`'s own derivative vs. an
analytic RHS), so it re-derives what `e_data` already pins on the observable
side and never touches the path net at all.

### 5e. `e_sde` (fixed ψ) actively hurts in this data regime

User's own 45k run: DCPINN corr **0.899** < PINN's **0.930**. Diagnosed via
the recovered path's standard deviation collapsing as SDE influence grows:
truth 0.246 → PINN 0.227 → DCPINN 0.185 → DCPINN\_nobal 0.070. The OU
transition NLL, as a trajectory penalty, has no term rewarding innovation
variance — minimising it drags `δ̂` toward the smooth conditional mean and
kills the high-frequency structure that correlates with the true rough path.
Consistent with an earlier finding (a prior notebook, `05_pinn_inversion_
testrun.ipynb`, not covered by this document) that the OU prior only helps in
an **ill-posed** regime (few/noisy contracts); with 12 maturities and low
noise, the data already pins δ and a smoothness prior can only remove signal.

**Per-channel loss plotting** was also added in this session (the
`plot_training_history_composite` cell), replacing a single total-loss curve
with one panel per channel (log-scale for `e_data`/`e_ode`, linear for
`e_sde` since the NLL can go negative) — the diagnostic tool that made 5d–5e
legible in the first place.

---

## §6. Making ψ learnable — a collapse, and its fix

**The trigger.** Every result up to this point used ψ as constants read
straight from `mc_data.pkl` — `p_Q`, `kappa_P`, `alpha_P`, `sigma2_P` were
never part of the trainable parameter pytree (`{"drift", "path"}` only). This
was flagged directly: *"ψ is not free anywhere."*

### 6a. The design

Six GS parameters made learnable as a new `params["psi"]` leaf:
`κ, σ₁, σ₂, ρ, α^Q, α^P`, with `r` held fixed (observable). Constrained by
reparameterisation (`softplus` for the three positive scalars, `tanh` for
`ρ`). **κ and σ₂ are shared across measures** — confirmed directly from
`mc_data.pkl`: `params_P.kappa == params_Q.kappa == 1.876`,
`params_P.sigma2 == params_Q.sigma2 == 0.527` exactly — so a *single* learned
leaf feeds both the ℚ pricing side (via `B(τ)`, `e_ode`) and the ℙ transition
side (`e_sde`); only `α` splits into `α^Q`/`α^P`. Initialised 30–50% off
truth, mixed directions, to make this a genuine recovery test.

### 6b. First run: DCPINN collapses catastrophically

MLP corr 0.931, PINN corr 0.907, **DCPINN corr 0.259** — a full collapse, far
below even the fixed-ψ DCPINN's 0.899 (§5e). Trajectory plots showed
`σ₂: 0.685 → 0.0001` and the recovered path flattening onto a near-constant
line.

**Mechanism — variance collapse.** The OU transition NLL contains
`0.5·log(2π·var)` with `var = σ₂²/(2κ)·(1−e^{−2κΔt})`. As `σ₂→0`, `var→0` and
`log(var)→−∞`: an unbounded-below free lunch. The optimiser found it —
shrink `σ₂` toward zero *and* flatten `δ̂` (so the quadratic term stays
finite) → `e_sde → −∞`. This was invisible in every earlier experiment
(§5e included) purely because `σ₂` had always been a fixed constant; the
instant it became learnable under a raw NLL, the objective had a hole in the
floor and training fell through it. Textbook variance collapse (the same
singularity that plagues free-variance GMMs/VAEs).

### 6c. Fix 1 — stop-gradient the path inside `e_sde`

`e_sde` should not be able to reshape the path to make its own likelihood
better — that's the "cheating" mechanism. Applied
`delta_hat_sde = lax.stop_gradient(delta_hat)` before the OU NLL, and
restricted its WamOL support accordingly. **Result:** DCPINN δ-recovery
**0.259 → 0.938** (now the best of the three, no path collapse, `σ₂` no
longer degenerate).

**But this exposed a second, separate degeneracy.** With the path fixed, the
OU-MLE hit a random-walk boundary: the NN-smoothed `δ̂` has tiny,
highly-autocorrelated increments at the data's Δt≈0.01, so it *looks like* a
random walk to the likelihood → `κ→0` (the `φ=e^{−κΔt}→1` limit). Once
`κ→0`, `α^P` becomes unidentified and wanders (validation run: `α^P→−1.54`).
**Worst part: because `κ` is a *shared* leaf, this ℙ-side degeneracy dragged
the well-identified pricing-side `κ` down with it** — MLP/PINN held `κ≈1.8`;
DCPINN corrupted it to `κ≈0.008–0.07` across two separate validation runs.

### 6d. Fix 2 (final) — restrict `e_sde` to update only `α^P`

Since `κ` and `σ₂` are already well-identified by the pricing channels
(`e_data` via `B(τ)`, `e_ode` via the ODE RHS), and `α^P` is the **one
parameter no other channel touches at all** (absent from `B(τ)`, `A(τ)`, and
the ODE RHS), the fix was to stop-gradient `κ` and `σ₂` *inside* the `sde_nll`
calculation too — `e_sde` still *reads* them (needed to evaluate the OU
mean/variance) but can no longer *write* them. (`σ₁` and `ρ` never appeared
in the SDE term to begin with, so nothing extra was needed there.)

**Standalone validation, 40k epochs, all three models:**

| | κ (true 1.876) | σ₂ (true 0.527) | α^P (true 0.106) | δ-recovery corr |
|---|---|---|---|---|
| MLP | 1.768 | 0.685 (frozen at init) | 0.046 (frozen at init) | 0.931 |
| PINN | 1.806 | 0.499 | 0.046 (frozen at init) | 0.907 |
| **DCPINN** | **1.644** (stable, no collapse) | 0.365 | **0.094** (moving toward truth) | **0.935** |

Trajectory plots confirmed stability *throughout* training, not just at the
final epoch — DCPINN's κ tracks alongside PINN's the entire 40k run.

**Ported into the notebook and re-run at the user's chosen 10k epochs**
(shorter than the 40k validated standalone, run directly in the IDE):

| | RMSE | κ | α^P |
|---|---|---|---|
| MLP | 0.136 | 1.874 | 0.046 (frozen) |
| PINN | 0.061 | 1.830 | 0.046 (frozen) |
| **DCPINN** | **0.054** (lowest of the three) | 1.809 (stable) | 0.076 (converging toward 0.106) |

At 10k epochs DCPINN is not just non-collapsed — it's the **best-performing
model**, and κ never breaks away from the pricing-identified value. This is
the state the notebook is in today.

---

## §7. What's still structurally unidentified (not a bug — a property of the model)

`σ₁` and `ρ` sit near their init values in every run above (MLP, PINN, and
DCPINN alike) and don't move meaningfully toward truth (`σ₁: 0.393`,
`ρ: 0.766`) no matter how long training runs. This was checked directly
against the closed-form `A(τ)` coefficient (`CLAUDE.md` §3):

$$A(\tau) = \Big(r-\alpha^{\mathbb Q}+\tfrac12\tfrac{\sigma_2^2}{\kappa^2}-\tfrac{\sigma_1\sigma_2\rho}{\kappa}\Big)\tau + \tfrac14\sigma_2^2\tfrac{1-e^{-2\kappa\tau}}{\kappa^3} + \Big(\alpha^{\mathbb Q}\kappa+\sigma_1\sigma_2\rho-\tfrac{\sigma_2^2}{\kappa}\Big)\tfrac{1-e^{-\kappa\tau}}{\kappa^2}$$

**`σ₁` never appears alone anywhere in this formula — only as the product
`σ₁σ₂ρ`.** This is a structural fact about the Gibson–Schwartz futures price
itself, not an artefact of using a free drift net instead of the exact
closed form (an earlier hypothesis in this session that switching to the
closed form would fix it was checked against this formula and found to be
**wrong** — the closed form has exactly the same product-only dependence, so
it wouldn't help). Futures term-structure data alone cannot separate `σ₁`
from `ρ`; only their covariance-term product is observable, in any
architecture. Two honest ways forward, not yet implemented: fix one of
`{σ₁, ρ}` externally and let the other absorb the whole product, or
reparameterise to learn the identified composite quantity directly instead
of pretending the individual factors are recoverable.

**A naming collision, also flagged but not yet fixed:** a pre-existing
comment in the calibration cell uses "ψ" in the original WamOL paper's sense
— `ψ = (θ, φ)`, the *entire* trainable network-weight vector — which now
collides with this session's own `params["psi"]` key (the six physical GS
constants). Same symbol, two unrelated meanings, in the same notebook.

---

## Where things stand today

- **Architecture:** drift net `Â_θ(τ)` + analytic `B(τ;κ)` + path net
  `δ̂_φ(t)`, GS parameters ψ learnable and reparameterised, `e_sde`
  restricted to update `α^P` only.
- **Working, in the sense that matters for the thesis:** all three models
  (MLP/PINN/DCPINN) recover δ without collapse; DCPINN is the best performer
  at 10k epochs and its shared parameters no longer get corrupted by the OU
  likelihood.
- **Known, documented limitation:** `σ₁` and `ρ` are not separately
  identifiable from futures data in this model, at any epoch count. This is
  worth stating explicitly in the thesis rather than leaving it looking like
  an unconverged run.
- **Open, not yet actioned:** the `σ₁`/`ρ` composite-reparameterisation fix
  (§7); the ψ-naming collision (§7); a full 40k in-notebook run of the final
  fix (only validated standalone at 40k and in-notebook at 10k so far); the
  §2c cash-and-carry question technically never got resolved, it was
  superseded by the §3 pivot away from inequality constraints entirely.
