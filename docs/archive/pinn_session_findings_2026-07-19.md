# PINN implementation — session findings (2026-07-19)

Working session on `notebooks/PINN_implementation.ipynb`. Started from a broken
notebook (`data` undefined at the Run cell), ended with the two-network rewrite
of `docs/mathematical_derivations-6.pdf` implemented, trained, and diagnosed.

This document records **what changed**, **what was verified**, and **what the
experiments found**. It does not itself change any code beyond what is described.

Pre-session notebook is backed up at
`scratchpad/PINN_implementation.ipynb.bak` (session scratch; copy elsewhere if
you want it to persist).

---

## 0. TL;DR

1. The original `NameError: data` was a missing `get_data()` call plus `error()`
   closing over notebook globals — not a data-loading problem. Fixed.
2. Inputs were **unnormalised**: 96% of the first tanh layer was saturated and
   `δ̂` was constant across the whole panel. Fixed with input normalisation.
3. The old objective composed the **closed form** with the network, which by
   construction drives the PDE residual to zero (verified: 4.7e-13). Per §6 of
   the derivations this collapses the method to penalised least squares. Rewrote
   to the **two-network** formulation (eq. 30): a free surface `F̂_θ(S,δ,τ)` and
   a latent path `δ̂_φ(t)`.
4. After the rewrite: **PINN recovers** the convenience-yield path (RMSE 0.172,
   corr +0.72), close to the identifiability floor (~0.10–0.14). **ACPINN
   collapses** to the no-skill baseline (RMSE 0.246 = std of the truth).
5. Diagnosed the ACPINN collapse by controlled experiment. It is **not** the
   WamOL balancer (hypothesis tested and refuted). It is a **single loss term**:
   the cash-and-carry ceiling `e_arb_cac` at `STORAGE_COST_U = 0`, which forbids
   the true deep-contango (δ<0) states on 35% of the panel.
6. **Validated on real WTI data (§5).** The collapse reproduces (std ratio 0.067),
   and the cac ceiling at `u=0` is violated by **31.9% of actual market
   observations** — confirmed model-free, with no network and no GS parameters.
   The finding is therefore not a simulator artefact.
7. Decision on the cac term **deferred** by the user — no fix applied yet.

---

## 1. Bugs found and fixed

### 1.1 `data` undefined at the Run cell
`get_data()` was defined but never called, and `error()` referenced
`taus`, `p_Q`, `kappa_P`, … as **free variables** resolved from notebook
globals. Returning them from `get_data` does not bind them at module level.

**Fix:** a new cell after `get_data` unpacks all nine returns into globals and
builds `data = (t_train, S_train, log_F_obs_train)`. `delta_true` is
deliberately excluded from `data` (CLAUDE.md §6.3 — the true δ path is the
evaluation target and must never enter a loss term).

**Design note:** the free-variable coupling is the underlying smell — a function
that took its pricing params as arguments would have been impossible to call
wrong. Left as globals for now (matches the ported style, keeps signatures short
across the `jit` boundary); flagged for a future frozen-dataclass refactor.

### 1.2 Unnormalised network inputs → tanh saturation
Raw inputs span `t ∈ [0,10]`, `S ∈ [36,165]`. `tanh` saturates outside ≈[-3,3].

Measured on the old single net at init:

| inputs | mean \|z\| | saturated (\|tanh\|>0.99) | mean tanh′ | δ̂ range |
|---|---|---|---|---|
| raw | 25.8 | **96.1%** | 0.0075 | [0.429, 0.494] |
| normalised | 0.31 | 0.0% | 0.875 | [-0.844, 0.144] |

At init the raw net's δ̂ was constant to 5e-3 across all 1001 dates — a
deceptively plausible local optimum given the truth only spans [-0.5, 0.79].

**Fix:** normalisation applied **at the network boundary** (not to the stored
arrays), so autodiff still differentiates the pricer w.r.t. **physical** S and δ.
- surface: `S → (log S − mean)/std`, `δ → δ/0.5`, `τ → τ/τ_max`
- path: `t → t/T`

### 1.3 PDE residual differentiated *through* `δ̂(t,S)` (old formulation)
The old `pde_residual` built `F_hat = S·exp(B·δ̂(t,S)+A)` and took
`grad(·, S)`, which leaked `∂δ̂/∂S` into the "partial" `F_S`. Verified at x64
over 200 dates × 12 maturities:

| residual definition | mean squared residual |
|---|---|
| self-consistent (δ independent, eq. 44) | **4.73e-13** (machine ε) |
| old notebook (through `δ̂`) | **3.68e+02** |

The 368 the old notebook reported for `e_pde` was **entirely** the
derivative-inconsistency artefact, not physics. This finding is what triggered
reading the derivations and the full rewrite.

---

## 2. The two-network rewrite (steps 1–7 of `pinn_rewrite_todo.md`)

Source of truth: `docs/mathematical_derivations-6.pdf`.

### What was ADDED
- **New cell — two networks (eq. 30):** `build_nets`, `make_fns`, `init_nets`.
  - `F̂_θ : (S,δ,τ) → ℝ⁺` — pricing surface, input dim 3, 12,801 params.
  - `δ̂_φ : t → ℝ` — latent path, input dim 1, 2,209 params, sign-unconstrained.
  - Both in one parameter pytree under a single `TrainState` (trained jointly).
- **Hard initial condition (eq. 47):** `F̂_θ = S·exp(τ·N_θ)`. Gives
  `F̂(S,δ,0)=S` identically (verified exact), positivity for free, and
  `log F̂ = log S + τ·N_θ` linear in the raw output. `L_b` is therefore dropped
  from the objective by construction — there is no `e_b` term.
- **Collocation mesh for `e_pde`:** 4,096 points over `(S,δ,τ)` with **δ an
  independent coordinate** (eq. 44 — the latent path does not appear in `L_f`).
  Disjoint from the data manifold, so the PDE loss has its own resolution.
- **`balancing` config flag** (added later, §4): gates the WamOL whack-a-mole so
  ACPINN can be run with it off. `run_tag()` keys result/checkpoint files by
  variant so bal-on vs bal-off don't clobber each other.

### What was REMOVED
- The single `MLP : (t,S) → δ̂` network and `ann_in_dim`.
- The closed form (`B_coeff`/`A_coeff`) from **every training loss**. Per §6
  remark 2: because the closed form solves the pricing PDE identically for every
  δ, composing it with `δ̂_φ` drives `L_f` and `L_b` to zero and reduces the
  method to penalised least squares. It is now used only in data-gen and
  validation.
- `S` as an input to the path network. Per §6 remark 1, admitting `S` would "let
  the network launder price information into the yield estimate"; the S–δ link is
  already carried by ρ in the dynamics that `L_f` enforces.

### What was CHANGED
- `error()` rewritten to eq. 42–45, 49. `e_acc` prices through the surface net at
  the inferred state `(S_i, δ̂_φ(t_i), τ_i)`; `e_pde` is the eq. 44 residual on
  the mesh; carry bounds stay on the data manifold (§7 "Domain of enforcement").
- `init_params_sa` self-adaptive weight shapes now split by **domain**:
  `e_acc`/carry terms `(n,K)`, `e_pde` `(N_f,)`, `e_delta_floor` `(n,)`.
- Configs replace `ann_in_dim` with `surface_hidden_dim` / `path_hidden_dim`.
- `plot_delta_recovery` drops the `S` argument (path net is 1-D).

### Constants (step 7)
| constant | was | now | note |
|---|---|---|---|
| `DELTA_MIN` | −3.0 | **−0.3** | eq. 39 value; **but see §5.3** — contradicts the data |
| `DELTA_MAX` | 3.0 | 1.5 | rcac wedge; paper gives no number |
| `STORAGE_COST_U` | 0.0 | 0.0 | **root cause of the ACPINN collapse — see §5** |

---

## 3. Post-rewrite results (user's 3000-epoch run)

| model | e_acc | e_pde | δ̂ RMSE | corr | verdict |
|---|---|---|---|---|---|
| MLP | 6.96e-4 | 5885 | — | — | data-fit only (no physics) |
| **PINN** | 3.36e-3 | 0.287 | **0.1717** | **+0.716** | recovers the path |
| ACPINN | 6.55e-3 | 0.436 | 0.2460 | +0.100 | collapsed |

`e_pde` now starts at ~4600 and is genuinely driven down — real physics content,
where the old formulation's value was a derivative artefact (§1.3).

### Identifiability floor — why PINN's 0.172 is good
The futures panel low-pass-filters δ. The data residual couples to the path
through `∂ln F̂/∂δ ≈ B(τ) = −(1−e^{−κτ})/κ`, which **vanishes as τ→0** (§12.1,
eq. 54): short tenors carry almost no information about δ, so only the slow
component (timescale τ\* = 1/κ ≈ 0.53 yr) is identifiable.

| reference | δ̂ RMSE |
|---|---|
| no-skill (constant mean = std of truth) | 0.2455 |
| **ACPINN** | 0.2460 (= no-skill; learned nothing) |
| **PINN** | **0.1717** (skill +30% over no-skill) |
| identifiability floor (envelope over 1/κ) | ~0.10–0.14 |

PINN recovers the low-frequency envelope to near the information limit and misses
only the near-floor jitter. **This is the thesis result** for the PINN inversion.

---

## 4. Diagnosing the ACPINN collapse

### 4.1 Hypothesis (WamOL balancer) — TESTED AND REFUTED
Initial read: the loss-balancer (`update_loss_weights`) averages `|grad|` over
the full `ψ=(θ,φ)` instead of on-support (§12.2, eq. 55–56 — a known deviation),
starving `e_acc`, the sole channel that disciplines the path (§9).

Added the `balancing` flag and ran ACPINN with the whack-a-mole fully off (fixed
λ=1, m=1), same seed:

| run | δ̂ RMSE | corr | std(δ̂) |
|---|---|---|---|
| PINN | 0.1717 | +0.716 | 0.165 |
| ACPINN, balancer **ON** | 0.2460 | +0.100 | 0.021 |
| ACPINN, balancer **OFF** | 0.2447 | +0.112 | **0.0095** |

Balancer off collapses **just as hard**. Hypothesis refuted — the balancer is
exonerated. (The on-support balancing deviation is still real and still worth
fixing eventually, but it is not what breaks recovery here.)

### 4.2 Isolation — one term is responsible
ACPINN-nobal differs from PINN only by three inequality terms at weight 1. Added
each to PINN singly:

| variant | δ̂ RMSE | corr | std(δ̂) | verdict |
|---|---|---|---|---|
| PINN (acc+pde) | 0.1717 | +0.716 | 0.165 | baseline |
| **+cac only** | **0.2447** | **+0.112** | **0.0095** | **collapse** |
| +rcac only | 0.1717 | +0.716 | 0.165 | inert (DELTA_MAX=1.5 never binds) |
| +delta_floor only | 0.1656 | +0.740 | 0.170 | **mildly helps** |

The cash-and-carry ceiling `e_arb_cac` — alone — flattens the path. rcac is
inert; the δ-floor at −0.3 is actually beneficial.

### 4.3 Root cause — cac at u=0 contradicts the data-generating process
The cac ceiling `F ≤ S·e^{(r+u)τ}` maps (eq. 35, B<0 flips it) to a **lower
bound on δ**: `δ ≥ ((r+u)τ − A(τ))/B(τ)`.

At `u=0` the tightest-maturity bound is **δ ≥ −0.0127**, and the true path dips
below it on **35.3% of dates**. The penalty forces δ̂ up until it can't move →
flat line at ≈+0.13 (the truth's mean with its downswings clipped).

`u` needed to stop cac firing on the true contango:

| clear... | required `u` |
|---|---|
| path-0 minimum (−0.50) | 0.45/yr |
| all-100-paths minimum (−0.98) | 0.90/yr |

45–90%/yr storage cost is economically absurd for crude. The deeper point:
**in Gibson–Schwartz, negative convenience yield *is* the model's storage cost.**
A cac ceiling with u=0 forbids exactly the contango the δ-dynamics produce — the
two encodings of storage contradict. The paper's own §7.1 hints at this
(the δ-inequality form "retained as an internal consistency check… rather than a
training identity"); the experiment shows that on this DGP the price-form cac
term does not gently reinforce — it dominates and destroys recovery.

---

## 5. Real-data validation (WTI)

Everything above was measured on the Monte Carlo panel. To test whether the cac
finding is an artefact of the simulator, it was re-run against the real WTI
panel in `data/input/real/wti_analysis_ready.csv`
(22,985 rows, 2,891 dates, 52 contracts, 2015-01-02 → 2026-06-29).

**The notebook was not modified.** The run lives entirely in a scratch script
that `exec`s the notebook cells into a private globals dict and overrides the
data there, so there was nothing to revert.

### 5.1 Model-free check — the market itself violates the cac bound
The cash-and-carry ceiling `F ≤ S·e^{(r+u)τ}` is model-free (§7.1): it needs no
GS parameters, no network, no calibration. So it can be tested directly against
observed settles. This is the strongest possible form of the §4.3 result.

| `u` | contracts violating | dates with ≥1 violation |
|---|---|---|
| **0.00** | **31.89%** | **38.33%** |
| 0.05 | 20.42% | 27.64% |
| 0.10 | 11.64% | 18.44% |
| 0.20 | 3.67% | 7.58% |
| 0.30 | 1.67% | 4.53% |
| 0.50 | 0.63% | 1.73% |

Implied carry `ln(F/S)/τ − r` (the `u` each observation would need):

| percentile | 50th | 75th | 90th | 95th | 99th | max |
|---|---|---|---|---|---|---|
| implied `u` | −0.057 | +0.033 | +0.113 | +0.168 | +0.383 | 3.590 |

Median is backwardation (−0.057, normal), but the right tail is thick. By year,
mean implied carry: **2015 +0.126, 2016 +0.119, 2020 +0.119** (max 3.59 during
the April-2020 storage crisis) — the known contango episodes. 2018–19 and
2021–26 are backwardated and untroubled by the bound.

**`STORAGE_COST_U = 0.0` is contradicted by the actual market on ~32% of
observations.** This is now a statement about WTI, not about the simulator.

### 5.2 Training run — the collapse reproduces
Panel restricted to the 8-contract dates and subsampled to match MC scale:
**1383 dates × 8 contracts, 2015-01-02 → 2026-02-19.**

| | std(δ̂) | δ̂ range | e_acc | e_cac |
|---|---|---|---|---|
| **PINN** | **0.1564** | [−0.574, −0.011] | 8.3e-3 | 5.38 |
| **ACPINN** (bal off) | **0.0105** | [−0.033, +0.003] | 1.4e-2 | 2.8e-4 |

**std ratio ACPINN/PINN = 0.067** (MC was 0.058) — the collapse reproduces.

The mechanism is more legible here than on MC data:
- PINN infers δ̂ **entirely negative** (−0.57 to −0.01), the economically correct
  reading of a contango-dominated 2015–2026 WTI sample.
- PINN's `e_cac = 5.38` — it substantially violates the ceiling, because the
  market does.
- ACPINN drives `e_cac` to 2.8e-4 and in doing so **pins δ̂ to ≈0**. It buys
  constraint satisfaction by denying the contango that actually occurred.

### 5.3 Caveats on the real-data run
1. **No `δ_true`** — RMSE and correlation are *undefined* on real data. Collapse
   (std → 0) is the only transferable diagnostic; no pseudo-RMSE was computed.
2. **Q-parameters are the MC ones** (κ=1.876, α_Q=0.0779, σ₁=0.393, σ₂=0.527,
   ρ=0.766), reused as a stand-in. Real WTI has no calibrated GS parameters yet —
   that is the Kalman-filter baseline's job. `e_pde` on real data is only as
   meaningful as those borrowed parameters.
3. **Structural accommodations**, each a real difference from the MC panel:
   - panel is **not rectangular** (6–9 contracts/date, mode 8) → restricted to
     the 8-contract dates;
   - **τ shrinks per date** (real dated contracts) → `taus` had to become an
     `(n, K)` **matrix**, and `e_acc` vmaps over per-date τ rows. This is exactly
     the extension point flagged in CLAUDE.md §4 and it is *required* for real
     data — the current notebook's fixed `taus` vector cannot represent it;
   - **r varies per date** (0–5.63%) → scalar mean `r = 2.074%` used throughout.

---

## 6. Open items (not actioned this session)

- **cac term — DECISION DEFERRED (user).** Three candidate fixes:
  1. drop cac from training, keep it as a validation-time check (would make
     ACPINN beat PINN, corr ~0.74); 2. raise `u` (economically strained, only
     renders it inert); 3. soften to a log-space / slack-margin penalty.
  §5.1 raises the stakes: on real WTI, clearing the observed contango would need
  `u ≳ 0.38` (99th pct) and `u ≳ 3.6` to clear every observation, so option 2 is
  not defensible on real data either. Option 1 now looks strongest.
- **Per-date τ is REQUIRED for real data** (§5.3). The fixed `taus` vector cannot
  represent dated contracts; `e_acc` needs an `(n,K)` τ matrix. Prototyped in the
  scratch script, **not** ported into the notebook. This is the CLAUDE.md §4
  extension point and it is now on the critical path, not optional.
- **Real WTI has no calibrated GS parameters.** §5.2 borrowed the MC Q-params.
  Any real-data result is provisional until the Kalman baseline supplies them.
- **`DELTA_MIN = −0.3` still contradicts the data.** §7.3/eq. 40 call it the
  "empirical lower edge… zero on all admissible states," but δ_true spans
  [−0.98, 1.36] and 6.3% of true states fall below −0.3. It happens to *help*
  recovery slightly here, but the write-up claim is false as stated. `−0.51`
  (1st pct) is the defensible value; sits commented in the constants.
- **WamOL on-support gradient restriction** (§12.2, eq. 55–56) — still averages
  over full `ψ`. Parked in `pinn_rewrite_todo.md`. The measured dim θ/dim φ ratio
  is 5.8×, not the "one to two orders of magnitude" the paper claims — worth
  correcting in the text.
- **x64 is off** in the notebook. Per CLAUDE.md §5 this floors the PDE residual
  at ~1e-5; enable `jax_enable_x64` before reading anything into absolute `e_pde`
  levels.
- **Epoch count.** Runs above are 3000 epochs (config default), not the 100k in
  the earlier config. PINN's 0.172 may still be improving toward the floor.

---

## 7. Files touched

- `notebooks/PINN_implementation.ipynb` — cells: added data/normalisation/mesh
  cell and two-network cell; rewrote `error()`, loss composition, calibration
  (+ `balancing` flag, `run_tag`), run configs (+ `config_ACPINN_nobal`
  control), metrics, and plotting cells.
- `docs/pinn_session_findings_2026-07-19.md` — this file.
- Backup: `scratchpad/PINN_implementation.ipynb.bak` (pre-rewrite).

**Not modified:** `data/input/real/*.csv` (read-only), and the
notebook was **not** touched by the §5 real-data work — that ran entirely in
session-scratch scripts (`cac_real.py`, `real_run.py`) which `exec` the notebook
cells into a private globals dict. Those scripts live in session scratch and will
not persist; §5.1 is cheap to reproduce (pure pandas over the CSV), §5.2 needs
the per-date-τ `error()` variant rewritten.
