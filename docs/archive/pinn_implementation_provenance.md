# `PINN_implementation.ipynb` vs. the original AC-PINN notebook

What was carried over unchanged, what was adapted, what's new, what got deleted,
and what's still a stub. "Original" = the colleague's SABR / Dupire-local-vol
AC-PINN calibration notebook this was ported from.

---

## ⚠️ OUTSTANDING: notebook is out of sync with `mathematical_derivations-6.pdf` (2026-07-16)

The derivation was revised from `-5.pdf` (7 pp.) to `-6.pdf` (12 pp.). The new
version adds a **Section 6 "Network parameterisation"** and two page-4 remarks
that **overturn the two central design choices the notebook currently rests
on**. What's implemented now is exactly the degenerate "penalised least
squares" case the new derivation was written to rule out. Diagnosis only — the
notebook was **not** changed (user is doing the rewrite). Nothing below this
section reflects `-6.pdf`; it documents the `-5.pdf` implementation as built.

### 🔴 Blocking — architectural

| # | Notebook now | `-6.pdf` requires | Source |
|---|---|---|---|
| 1 | **One** network → `delta_hat`; closed form supplies prices | **Two** networks trained jointly (eq. 30): surface `F̂_θ(S,δ,τ)→ℝ⁺` **and** path `δ̂_φ(t)→ℝ`. Surface net is what the PDE acts on; path net enters only via the data loss. | §6, eq. 30 |
| 2 | `e_pde` differentiates the **closed-form composite** `S·exp(B·δ̂+A)` | Closed form **excluded from training** — composing it with `δ̂` "would render `L̂_f` and `L̂_b` identically zero and reduce the method to penalised least squares." `L_f` must act on the *free* surface net. | §6 "Role of the closed form"; §10 eq. 44 |
| 3 | `delta_hat_fn(t, S)` — takes **spot as input** | "`δ̂_φ` takes **t alone** as input. Admitting `S_t`… would let the network launder price information into the yield estimate." | §6 p.4 remark 1 |
| 4 | `e_pde` reuses the **data (t,S,τ) grid**, δ = `δ̂(t)` | `L_f` on a **separate (S,δ,τ) collocation mesh**, δ an *independent coordinate*; "the latent path `δ̂_φ` does not appear." | §10 eq. 44–45 |

**Empirical fingerprint already observed:** in the last smoke test `e_pde` sat
at ~`1e-8` throughout — not convergence, but the exact degeneracy §6 predicts
(you can't get PDE-residual signal by differentiating the PDE's own closed-form
solution).

### 🟠 Now specified in `-6.pdf` (were stubs/guesses)

- **`L_b` initial-condition loss** — now given (§11): either soft
  `(F̂_θ(S,δ,0)−S)²` on the τ=0 slice (eq. 46), **or** the hard output transform
  `F̂_θ = S·exp(τ·N_θ(S,δ,τ))` (eq. 47), which also gives ℝ⁺ positivity and makes
  `ln F̂` linear in the raw output, and then `L_b` is dropped. (Recommended.)
- **Surface-net positivity** — the softplus removal is correct for the *path*
  net but the *surface* net needs ℝ⁺; eq. 47 handles it.
- **`δ_min = −0.3`** is now fixed by the paper (eq. 39). Notebook currently has
  `DELTA_MIN = -3.0` — off by 10×, so the floor never fires. `δmax` (rcac wedge)
  still has no numeric value in the paper.

### 🟡 Refinements

- **WamOL balancing must use per-category *support*** (§12.2, eq. 55–56):
  gradient scales on `ψ_t=ψ_cac=ψ_rcac=(θ,φ)`, `ψ_b=ψ_f=θ`, `ψ_δ-floor=φ`.
  Current `update_loss_weights` averages over the full param vector, which the
  paper warns dilutes the scalar path-net weight "by one to two orders of
  magnitude."
- **Carry penalties** should price through `F̂_θ(x_i)`, not the closed form.
- **Online/time-decay ζ** (§9, §12.3) — optional; `ζ≡1` static case is fine.

### ✅ Still valid under `-6.pdf`
Network building blocks (`Dense`/`MLP`/`ModifiedMLP`), optax setup, WamOL loop
skeleton, `get_data` loader, and the closed form in `gibson_schwartz.py` — now
used only for data-gen + validation benchmark (§5 "Role of the closed form"),
not in the objective.

---

## Kept as-is (problem-agnostic)

Nothing about these referred to volatility, options, or SABR in the first
place, so they carried over verbatim.

| Piece | What it is |
|---|---|
| `Dense`, `_weight_fact` | Custom dense layer with optional weight-factorized reparameterisation |
| `MLP`, `ModifiedMLP` | Network architectures (the multiplicative-gate variant) |
| `ann_gen` | Config-driven architecture factory |
| `activation_fn`, `_get_activation` | Activation lookup (`tanh`, `sin`) |
| optax Adam + exponential LR decay | Optimizer setup |
| `adj` | Per-point weighted-mean loss reduction |
| `make_loss_lb`, `make_loss_fn` | Build a per-component loss dict / a scalar total loss from it |
| WAMOL training loop | `TrainState`, self-adaptive per-point weights (SGD ascent), gradient-norm loss balancing ("Whack-a-mole"), the `train_step` / `train_step_sa` / `update_loss_weights` jitted functions |
| `save_params`, `load_params`, `flatten_pytree`, orbax checkpointing | Persistence utilities |
| `plot_training_history` | Loss-curve plotter — reads whatever metric keys are in the pickled history, so it needed no changes |

**One exception to "everything ported into the notebook":** the closed-form
GS pricing coefficients (`GSParams`, `B_coeff`, `A_coeff`) are *imported*
from `gs_wamol.physics.gibson_schwartz`, not retyped. `CLAUDE.md` sec. 3 is
explicit that those coefficients are exact and verified against Schwartz
(1997) eqs. (18)–(20); retyping them risks a silent transcription bug, and
`monte_carlo.ipynb` already depends on that exact module.

---

## Changed subtly (existing function, adapted logic)

| Original | Changed to | Why |
|---|---|---|
| `MLP`/`ModifiedMLP` forced `x = nn.softplus(x)` on the output | Softplus removed | The original output was a volatility (must be positive). `delta_hat` (convenience yield) is not sign-constrained — nothing in the model requires `delta_t >= 0`. |
| `error()` computed Black-Scholes/Dupire/SABR terms | Rewritten around the GS closed form and the PDE derivation | See "New" below — the *shape* of the function (return `(err_dict, metrics_dict)`) is unchanged, but every formula inside it is GS's, not SABR's. |
| `run_experiment` wrote one shared `results_latest.pkl` | Writes `results_{loss_str}.pkl` | The original silently overwrote that file on the second call in the same cell (e.g. MLP's history lost as soon as ACPINN ran). Keying by `loss_str` was a one-line fix to an existing footgun, not a new feature. |
| `run_experiment(config)` re-sampled data via `get_data(config.pts_num, ...)` each call | `run_experiment(config, data)` takes data explicitly | The GS dataset is a fixed loaded panel, not something to redraw per experiment the way the SABR sampler was. |
| `init_l_ws` / `init_params_sa` keyed on `x_train`/`x_mesh` shapes | Reshaped to `error()`'s actual component shapes: `e_acc (n,K)`, `e_pde (n,K)`, `e_arb_cac (n,K)`, `e_arb_rcac (n,K)`, `e_delta_floor (n,)` | Mechanical consequence of the new loss terms, same dict-of-arrays pattern as the original. |

---

## New (GS-specific, mechanical plumbing — not modeling decisions)

| Piece | Replaces | Notes |
|---|---|---|
| Input Data loader (`mc_params.json` + `mc_data.parquet` → `t_train`, `S_train`, `log_F_obs_train`, `delta_true`) | `get_data()`'s SABR truncated-normal sampler over a `(K,T)` grid | Reads the panel `monte_carlo.ipynb` already built, for a single path (`PATH_ID = 0`). `delta_true` is loaded but deliberately kept out of `data` — eval-only (CLAUDE.md: "the true delta_t path is sacred"). |
| `plot_delta_recovery` | `compare_with_sabr` | The GS analogue of "learned vs. true" — plots `delta_hat(t,S)` against `delta_true(t)`. The only place `delta_true` is used anywhere in the notebook. |

---

## Loss terms — implemented from `docs/mathematical_derivations-5.pdf`

The original's `error()` was entirely Black-Scholes/SABR-specific (option
price, vega, implied vol, Dupire local-vol PDE, call-surface arbitrage
bounds). None of it transferred. What's in `error()` now is a direct port of
the math in the derivation PDF, not a guess:

| Component | PDF source | What it does |
|---|---|---|
| `e_acc` | $\mathcal{L}_t$, eq. 32 | Log-price MSE of `ln S + B(tau) delta_hat(t,S) + A(tau)` against `ln F_obs`, priced through the exact closed form. |
| `e_pde` | $\mathcal{L}_f$, eq. 33/35 | PDE residual, obtained by automatically differentiating the *composite* `F_hat(t,S,tau) := S * exp(B(tau) delta_hat(t,S) + A(tau))` through the network — the same trick the original's `call_derivatives` used (differentiate `bs(..., vol=network(x))` through the network, not the closed form's own abstract arguments, which would make the residual identically zero). |
| `e_arb_cac` | $\mathcal{L}_{h,\text{cac}}$, eq. 22 | Cash-and-carry upper bound on the futures price. |
| `e_arb_rcac` | $\mathcal{L}_{h,\text{rcac}}$, eq. 25 | Reverse-cash-and-carry lower bound, with the convenience-yield wedge `delta_max`. |
| `e_delta_floor` | $\mathcal{L}_{h,\delta\text{-floor}}$, eq. 28 | Convenience-yield floor. |
| — | $\mathcal{L}_b$ | **Not implemented.** The PDF's own footnote says "Definition to be supplied" — there's nothing to port. |

### Design decision that superseded an earlier guess

An earlier version of this notebook improvised an "OU drift-consistency"
physics term because the network's I/O contract hadn't been pinned down yet.
That guess was wrong — it collapsed `delta_hat` to a flat line in testing —
and the PDF's actual answer (`(t, S) -> delta_hat`, and a genuine autodiff
PDE residual rather than a hand-rolled SDE-consistency check) replaced it
entirely once the derivation was available.

### Constants the PDF names but doesn't fix numerically

| Constant | PDF role | What was used here | Why |
|---|---|---|---|
| `STORAGE_COST_U` | eq. 21/22, per-unit-time physical storage cost | `0.0` | No value given anywhere in the derivation; this is the simplest default, not a considered choice. |
| `DELTA_MIN` / `DELTA_MAX` | eq. 24–28, convenience-yield floor/ceiling | `alpha_P ± 3·(sigma2_P / sqrt(2·kappa_P))` — a stationary-OU band | The PDF fixes `delta_min = -0.3` from *their* simulated draw (eq. 27). `PATH_ID = 0` here reaches about -0.5, so copying `-0.3` would bind on genuinely-simulated states. Re-deriving the edge from `delta_true` directly would fix that but leaks eval-only ground truth into a training hyperparameter — so the band is derived from `kappa_P`/`sigma2_P` alone (assumed-known physical parameters, not the realised path). |

All three are marked `TODO` in the notebook — placeholders, not settled values.

---

## Stripped entirely (SABR / Black-Scholes / Dupire — none of it applies to GS)

`bs`, `black`, `d1_`, `d2_`, `bs_vega`, `bs_iv`, `lv_var`, `lv_fwd_sqr`,
`lv_fwd_pde`, `lv_sqr`, `SabrVolHagan`, `chi`, `get_truncated_normal`,
`get_data`'s SABR sampler, `call_derivatives`, `derivatives` (the original's
2-input mesh AD helper — not applicable once the network's inputs became
`(t, S)` with a different differentiation pattern), `plot_volatility_surface`,
`plot_arbitrage_heatmaps`, `compare_with_sabr`.

---

## Left for you (explicitly, not an oversight)

- **$\mathcal{L}_b$**: undefined in the source PDF itself.
- **Collocation mesh**: `e_pde` currently reuses the observed `(date, maturity)`
  data grid as its collocation points. The original AC-PINN used a separate,
  denser `x_mesh` independent of `x_train` for its PDE term — an equivalent
  here (extra `(t,S,tau)` points beyond the observed panel) would give
  `e_pde` its own resolution instead of being tied to data density.
- **`STORAGE_COST_U`, `DELTA_MIN`, `DELTA_MAX`**: see table above.
- **`PATH_ID`**: inverts a single simulated path end to end. Real-data
  loading is `gs_wamol/data/market.py`, currently unimplemented.
- **ACPINN training quality**: in the 3000-epoch smoke test, all three
  inequality terms sat at exactly `0.0` (never violated, given the band
  width above) — so they contributed no gradient, yet ACPINN's `e_acc` still
  tracked *worse* than plain PINN's. Likely the gradient-norm rebalancing
  ("Whack-a-mole", which only runs for `loss_str == "ACPINN"`) is diluting
  `e_acc`'s effective weight by rebalancing against three structurally-inactive
  terms. Not fixed — worth investigating before trusting ACPINN results.
