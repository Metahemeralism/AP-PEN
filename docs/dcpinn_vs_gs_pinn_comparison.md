# Original DCPINN notebook vs. `notebooks/PINN_implementation.ipynb`

A line-by-line provenance audit of the current notebook against the original
**AC-PINN / DCPINN IVS-calibration notebook** (Hoshisashi et al., SABR data →
Dupire local-vol surface). Three verdicts per item:

- **SAME** — carried over verbatim or near-verbatim (renames only).
- **CHANGED** — the same construct survives, but its contents/contract differ.
- **DIFFERENT** — no counterpart; either deleted from the original or new here.

Naming note: the original's third loss variant is `"DCPINN"`; the notebook calls
the same slot `"ACPINN"`. Same role (data + PDE + inequality constraints), same
position in the `loss_fn_lb` dict — a pure rename, tracked below as CHANGED.

Supersedes the architecture section of
[pinn_implementation_provenance.md](pinn_implementation_provenance.md), which
documents the pre-rewrite (`-5.pdf`, single-network) state.

---

## 1. Executive summary

| Layer | Verdict | One-line |
|---|---|---|
| Network building blocks (`Dense`, `_weight_fact`, `MLP`, `ModifiedMLP`) | SAME | Identical, minus the output softplus |
| Optimiser (Adam + exponential decay) | SAME | Hyperparameters unchanged |
| WamOL machinery (self-adaptive `m`, grad-norm λ, whack-a-mole schedule) | SAME | Same algorithm, extra threading args |
| Checkpoint / persistence utilities | SAME | Paths changed only |
| Model topology | **DIFFERENT** | 1 net → **2 jointly-trained nets** |
| Loss residuals (`error`) | **DIFFERENT** | Every formula replaced; Dupire → Gibson–Schwartz |
| PDE term | **DIFFERENT** | Dupire local-vol → GS 2-factor, on its own mesh |
| Constraint terms | **DIFFERENT** | 3 call-surface no-arb → 2 carry bounds + δ-floor |
| Data source | **DIFFERENT** | SABR sampler → loaded MC panel |
| Financial primitives (BS/Black/SABR/Dupire) | **DIFFERENT** (deleted) | ~350 lines gone, nothing replaces them 1:1 |
| Visualisation | mixed | 1 SAME, 3 deleted, 1 new |

**The single biggest structural divergence:** the original has one network
(→ implied volatility) that is *fed into* a closed-form pricer (`bs`) which is
then differentiated. The notebook once mirrored that exactly — network → `δ̂`,
fed into the GS closed form, then differentiated — and it was **wrong**, because
the GS exponential-affine solution satisfies the pricing PDE identically for
*every* value of `δ`, so the residual is zero by construction (measured:
`4.7e-13` at x64). The rewrite therefore splits into two networks and drops the
closed form from the objective entirely. That asymmetry with the original is
**not an oversight — it is forced by the maths**, and is the reason most of the
"same shape" analogies below break down at the `error()` boundary.

---

## 2. SAME — carried over essentially verbatim

Nothing in this group ever referenced options, volatility, or SABR, so it
transferred untouched.

| Piece | Delta from original |
|---|---|
| `activation_fn` dict, `_get_activation` | Arg renamed `str` → `name` (the original shadowed the builtin); `else:` branch flattened to a bare `raise`. Behaviour identical. |
| `_weight_fact` | Verbatim. `jran.split` → `random.split` (import alias only). |
| `Dense` | Verbatim, including the latent bug that `kernel` is undefined if `reparam["type"] != "weight_fact"`. |
| `MLP` / `ModifiedMLP` | Verbatim **except** the final `nn.softplus` — see §3. |
| `ann_gen` | Verbatim. Still present, but now dead code for the main path (`build_nets` replaced it). |
| optax setup (`Adam`, `b1=0.9`, `b2=0.999`, `eps=1e-8`, `lr=1e-3`, `decay_rate=0.9`, `decay_steps=2000`, `exponential_decay`) | Verbatim; module-level `eps` renamed `adam_eps` to stop it shadowing the `1e-15` numerical guard (a real latent bug in the original — the second assignment silently redefined `eps` for every function defined above it). |
| `adj(loss, lw, m)` | Verbatim: `lw * mean(m * loss)`. |
| `make_loss_lb` / `make_loss_fn` / `loss_fn_lb` / `loss_fn` | Same two-level factory pattern (dict of components → scalar). Signature gained a `mesh` argument. |
| `save_params`, `load_params`, `flatten_pytree` | Verbatim. |
| `train_step` | Same structure: `value_and_grad(..., has_aux=True)` → `apply_gradients`. |
| `train_step_sa` | Verbatim, including the sign flip `grads_sa[k] *= -1.` that turns descent into **ascent** on the per-point weights (the saddle-point formulation). |
| `update_loss_weights` | Verbatim: `jacrev` per component, `mean(|g|)` as the scale, `sum/scale` weighting, momentum running average, `stop_gradient`. Including the original's commented-out `linalg.norm` alternative. |
| Whack-a-mole schedule | Verbatim: λ-update at `epoch % 100 == 0`, `m`-update at `(epoch+50) % 100 == 0`. |
| Progress logging | Verbatim: print + `hist_loss.append` every 1000 epochs. |
| orbax checkpointing block | Verbatim (`PyTreeCheckpointer`, `save_args_from_target`, `force=True`); only `CKPT_DIR` changed. |
| `plot_training_history` | Verbatim in substance — it reads whatever metric keys exist in the pickle, so new loss names needed no edit. Only the hardcoded per-key colour dict was dropped and the title now names the variant. |

---

## 3. CHANGED — same construct, different contents

| # | Original | Now | Why |
|---|---|---|---|
| 1 | `MLP`/`ModifiedMLP` end with `x = nn.softplus(x)  # CAUTION NOT PLAIN MLP` | Softplus removed from **both** | Original output was an implied vol (must be > 0). Neither current output needs it: `δ̂` is sign-unconstrained (contango ⇒ δ can be negative), and the surface net's positivity comes from the eq. 47 output transform instead. |
| 2 | `"DCPINN"` key in `loss_fn_lb`, `init_l_ws`, `init_params_sa` | `"ACPINN"` | Rename only; identical role. |
| 3 | Components `['e_acc','e_pde','e_arb_dK','e_arb_d2K','e_arb_dT']` | `['e_acc','e_pde','e_arb_cac','e_arb_rcac','e_delta_floor']` | Constraint set is problem-specific — see §4.3. Count coincidentally also 5. |
| 4 | `init_params_sa` shapes keyed on `len(data[0])` / `len(data[2])` — every component on one of two flat point sets | Shapes keyed on **domain**: `e_acc`, `e_arb_*` → `(n, K)` data manifold; `e_pde` → `(N_f,)` mesh; `e_delta_floor` → `(n,)` path | Substantive, not cosmetic: three distinct domains now exist where the original had two. |
| 5 | `error(fn, data)` — one callable, one data tuple | `error(fns, data, mesh)` — a 3-tuple of bound callables, plus a separate mesh | Two networks, two domains. Return contract `(err_dict, metrics_dict)` is unchanged. |
| 6 | `calibration` builds one `ann`, `TrainState.create(apply_fn=ann.apply, params=ann.init(...))` | `init_nets` builds two, `TrainState.create(apply_fn=None, params={"surface":…, "path":…})` | One `TrainState` over the joint pytree is what "trained jointly" means operationally — a single `apply_gradients` updates θ and φ. |
| 7 | `run_experiment(config)` calls `get_data(config.pts_num, 101)` internally | `run_experiment(config, data, mesh)` takes both explicitly | The GS panel is a fixed loaded dataset, not something to redraw per run. `pts_num` is gone entirely. |
| 8 | Writes `results_latest.pkl` (overwritten by every run) | Writes `results_{run_tag(config)}.pkl`; `CKPT_DIR` likewise per-tag | Fixes a real footgun: in the original, running MLP then DCPINN in one cell destroyed the MLP history and the MLP checkpoint. |
| 9 | Balancer gated on `config.loss_str == "DCPINN"` inline | Gated on `use_balancing = config.get("balancing", True) and loss_str == "ACPINN"` | Adds the `ACPINN_nobal` control arm — same losses, fixed λ=1/m=1 — to test whether the balancer or the constraints causes path collapse. |
| 10 | `s_0`, `r`, `alpha/beta/rho/nu` as bare module globals used inside `error` | `p_Q` (`GSParams`), `kappa_P`, `alpha_P`, `sigma2_P` closed over from the loaded panel | Same closure-over-globals pattern (still not injected — see §7), but the values are now loaded, not hardcoded SABR constants. |
| 11 | `config`: `ann_in_dim=2`, `ann_hidden_dim=(16,16,16,16)`, `pts_num`, `data_source` | `surface_hidden_dim=(64,64,64,64)`, `path_hidden_dim=(32,32,32)`, `balancing`; no `ann_in_dim`/`pts_num`/`data_source` | Two nets have different input dims, so a single `ann_in_dim` is meaningless. Surface net widened — it carries a whole 3-D pricing function, not a scalar. |
| 12 | No input normalisation; raw `(K, τ)` already ≈ `O(1)` (`s_0 = 1.0`) | `normalize_surface` / `normalize_time` applied at the network boundary | GS inputs are `S ∈ [36,165]`, `t ∈ [0,10]`, `τ ∈ [0.08,2]`. Measured on the pre-fix single net: 96% of first-layer tanh units saturated (mean `tanh' = 0.008`) and `δ̂` was flat to `5e-3`. The original never needed this because SABR was already unit-scaled — a hidden assumption in the inherited code. |

---

## 4. DIFFERENT — no counterpart in the original

### 4.1 Model topology (the central divergence)

| | Original | Now |
|---|---|---|
| Networks | **1** | **2**, jointly trained |
| Signature | `(K, τ) → σ_impl ∈ ℝ⁺` | `F̂_θ(S, δ, τ) → ℝ⁺` **and** `δ̂_φ(t) → ℝ` |
| Pricing | Network output fed into closed-form `bs(...)`, then autodiffed (`call_derivatives`) | Surface net **is** the price; closed form absent from the objective |
| Positivity | Output softplus | Output transform `F̂ = S·exp(τ·N_θ)` (eq. 47) |
| Initial condition | Not enforced | Enforced **structurally** by eq. 47 (`τ=0 ⇒ F̂=S` identically), so `L_b` is dropped from the objective rather than penalised |

Why the original's trick doesn't transfer: `call_derivatives` differentiates
`bs(s_0, K, τ, r, 1, net(x))` — the Black–Scholes price *is* a free surface once
its vol argument is a network, so a Dupire residual on it has content. The GS
analogue does **not**: `S·exp(B(τ)δ̂ + A(τ))` solves the GS PDE exactly for any
`δ̂` whatsoever, so the residual is identically zero and the method degenerates
to penalised least squares. Hence a free surface net.

### 4.2 The physics term

| | Original `e_pde` | Now `e_pde` |
|---|---|---|
| Equation | Dupire forward local-vol PDE, `dT_C + rK·dK_C − ½σ²_LV K²·d2K_C` | GS two-factor pricing PDE in τ: `F_τ − (r−δ)S F_S − κ(α^ℚ−δ)F_δ − ½σ₁²S²F_SS − ρσ₁σ₂S F_Sδ − ½σ₂²F_δδ` |
| Local vol | Computed via `lv_sqr` (Dupire in K–T space) | n/a — GS has no local-vol object |
| Domain | `x_mesh`: a dense `101×101` `(K,T)` grid, separate from `x_train` | `mesh`: **4096 uniform random** `(S, δ, τ)` points, δ an **independent coordinate** sampled over `[-1.0, 1.5]` |
| Derivative order | 1st and 2nd in K, 1st in T (2 inputs) | Full 2nd-order set including the **cross term** `F_Sδ` (3 inputs) |
| AD mechanism | `vmap(grad(f))` + a manual "grad of grad-component" trick, with the `# CAUTION NOT hessian` warning | Direct nested `grad(F_hat_fn, i)` per argument — clearer, since the function takes named scalar args rather than a packed vector |

The **separate collocation mesh** is conceptually inherited from the original
(`x_mesh` ≠ `x_train`), but the meaning of the extra dimension is new: sampling δ
independently of `δ̂_φ(t)` is what keeps the PDE a statement about the whole
state space rather than about the data manifold.

### 4.3 The inequality terms

Zero overlap. The original's are static-arbitrage bounds on a **call price
surface**; the notebook's are **carry bounds on a futures price** plus a bound
on the latent state itself.

| Original | Now |
|---|---|
| `e_arb_dK`: `∂C/∂K ∈ [−e^{−rT}, 0]` (two-sided) | `e_arb_cac`: `F̂ ≤ S·e^{(r+u)τ}` cash-and-carry ceiling |
| `e_arb_d2K`: `∂²C/∂K² ≥ 0` (butterfly / density positivity) | `e_arb_rcac`: `F̂ ≥ S·e^{(r−δ_max)τ}` reverse-cash-and-carry floor |
| `e_arb_dT`: `∂C/∂T ≥ 0` (calendar) | `e_delta_floor`: `δ̂_φ(t) ≥ δ_min` |
| All three are **derivative** constraints, evaluated on the mesh via AD | Two are **level** constraints on the price, one is a level constraint on the path; **no autodiff** involved |
| All three on `x_mesh` | All three on the **data manifold**, deliberately not the mesh — the cac ceiling makes no reference to δ, and the true GS surface violates it wherever `δ < ((r+u)τ − A)/B`, so enforcing it mesh-wide would contradict `e_pde` |

The penalty *form* is also different: original uses `jnp.where(cond, x**2, 0)`,
the notebook uses `jnp.maximum(0, viol)**2`. Equivalent in value; the hinge form
is the standard one and avoids the nested `where` the original needed for its
two-sided `dK` bound.

### 4.4 Data

| Original | Now |
|---|---|
| `get_data(n_pts, n_h)` synthesises on the fly | `get_data(path_id, path)` unpickles `data/input/synthetic/mc_data.pkl` |
| Truncated-normal sampling of `(K, τ)` via `scipy.stats.truncnorm` | Fixed `(t_grid × taus)` panel — no sampling |
| Labels from `SabrVolHagan` → `black`, with `×(1 + N(0, 0.1))` **multiplicative vol noise** | Labels are `log_F_obs`, noise added upstream in `gs_wamol.data.observe` (additive in log-price space) |
| No ground truth retained — SABR vol is recomputed at plot time for comparison | `delta_true` loaded and **deliberately excluded from `data`** (CLAUDE.md §6.3) |
| `x_mesh` returned as part of `data` | `mesh` built separately, independent of the panel |

### 4.5 Deleted outright (~350 lines, no replacement)

Every financial primitive in the original's first code cell is Black–Scholes-
or SABR-specific and has no Gibson–Schwartz analogue:

`bound`, `d1_`, `d2_`, `bs`, `black` (both definitions — the JAX one and the
NumPy one that silently shadowed it), `lv_var`, `lv_fwd_sqr`, `lv_fwd_pde`,
`lv_sqr`, `bs_vega`, `bs_iv` (Newton implied-vol solver), `chi`,
`SabrVolHagan`, `get_truncated_normal`, `derivatives`, `call_derivatives`,
`N`/`N_prime`/`N_inv`, the SABR parameter cell (`alpha, beta, rho, nu`), the
premium-surface scatter plot, `plot_volatility_surface`,
`plot_arbitrage_heatmaps`, `compare_with_sabr`.

The closed-form GS coefficients (`GSParams`, `B_coeff`, `A_coeff`) are
**imported** from `gs_wamol.physics.gibson_schwartz` rather than retyped —
they're verified against Schwartz (1997) eqs. 18–20 and are used only for data
generation and post-hoc benchmarking, never inside the objective.

### 4.6 New with no ancestor

| Piece | Role |
|---|---|
| `build_nets` / `make_fns` / `init_nets` | Two-network construction, parameter binding, joint pytree init |
| `normalize_surface` / `normalize_time` | Input rescaling at the network boundary (§3 item 12) |
| Eq. 47 output transform `F̂ = S·exp(τ·N_θ)` | Hard initial condition + positivity, replacing both softplus and an `L_b` term |
| Collocation-mesh construction (`N_COLLOCATION`, `DELTA_MESH_RANGE`) | δ as an independent sampling coordinate |
| `STORAGE_COST_U`, `DELTA_MIN`, `DELTA_MAX` | Constraint constants (see §6) |
| `run_tag(config)` | Artefact naming, enabling the `ACPINN_nobal` control arm |
| `config_ACPINN_nobal` | Balancer-off control |
| `plot_delta_recovery` | `δ̂_φ(t)` vs. `delta_true` — the only place ground truth is touched |
| Metrics-summary cell | Post-hoc `error()` call across all four variants |

---

## 5. Side-by-side: the objective

```
ORIGINAL (DCPINN, SABR/Dupire)          NOW (ACPINN, Gibson–Schwartz)
────────────────────────────────        ──────────────────────────────────────
σ_θ(K,τ)                                F̂_θ(S,δ,τ) = S·exp(τ·N_θ)   [surface]
        │                               δ̂_φ(t)                       [path]
        ▼
C = bs(s₀,K,τ,r,1,σ_θ)                  ── data manifold (n × K) ──
        │                               e_acc  = (log F̂_θ(Sᵢ, δ̂_φ(tᵢ), τⱼ)
   ── x_train ──                                  − log F_obs)²
   e_acc = (C − y)²                     e_arb_cac  = max(0, F̂ − S e^{(r+u)τ})²
                                        e_arb_rcac = max(0, S e^{(r−δmax)τ} − F̂)²
   ── x_mesh (101×101 K,T) ──           e_delta_floor = max(0, δmin − δ̂_φ)²
   e_pde     = Dupire residual²
   e_arb_dK  = hinge(∂C/∂K)²            ── collocation mesh (4096 S,δ,τ) ──
   e_arb_d2K = hinge(∂²C/∂K²)²          e_pde = GS-PDE-τ residual²
   e_arb_dT  = hinge(∂C/∂T)²                    (δ independent of δ̂_φ)

   L_b: absent                          L_b: absent — enforced structurally
```

---

## 6. Constants: original vs. now

| | Original | Now |
|---|---|---|
| Source | Hardcoded SABR truth: `s_0=1.0, r=0.05, α=0.3, β=0.7, ρ=−0.6, ν=0.6` | Loaded from `mc_data.pkl` (`p_Q`, `kappa_P`, `alpha_P`, `sigma2_P`) |
| Free constraint constants | none | `STORAGE_COST_U = 0.0`, `DELTA_MAX = 1.5`, `DELTA_MIN = −0.3` |

`DELTA_MIN` carries a documented, unresolved conflict: `-6.pdf` eq. 39 asserts
`δ_min = −0.3` returns zero on all admissible states, but the panel says
otherwise — `delta_true` spans `[−0.98, 1.36]` across 100 paths, 1st pct `−0.51`,
and **6.27%** of true states fall below `−0.3` (4.30% on path 0). At `−0.3` the
floor penalises the ground truth and biases `δ̂_φ` upward in exactly the
deep-contango regime the term is meant to describe. The paper value is kept so
code matches write-up; `−0.51` is commented as the alternative. **Resolve before
citing eq. 39.**

---

## 7. Issues inherited from the original and still present

Flagged, not fixed — all are faithful reproductions of the source notebook:

1. **`Dense` with an unrecognised `reparam["type"]`** leaves `kernel` unbound →
   `UnboundLocalError`. Needs an `else: raise`.
2. **Globals-as-closure** (`p_Q`, `taus`, `S_train`, `DELTA_*` read directly
   inside `error`) — untestable in isolation, and reordering cells silently
   changes results. The original did the same with `s_0`/`r`. The fix is to pass
   a frozen config/params object, as `CLAUDE.md` §6.4 already requires of the
   `gs_wamol` package code.
3. **`update_loss_weights` measures gradient scale over the full parameter
   vector** `ψ = (θ, φ)`. `-6.pdf` §12.2 (eq. 55–56) requires per-category
   support: `ψ_f = θ`, `ψ_δ-floor = φ`, `ψ_t = ψ_cac = ψ_rcac = (θ, φ)`. With
   `dim φ ≪ dim θ` this dilutes `g_δ-floor` and inflates its λ by the reciprocal
   — the paper puts the distortion at 1–2 orders of magnitude. Parked in
   [pinn_rewrite_todo.md](pinn_rewrite_todo.md).
4. **History logged only every 1000 epochs** — at `num_epochs=3000` that is
   three points, which is not a loss curve. Original had the same cadence at
   5000 epochs.
5. **No validation split anywhere.** Both notebooks report training loss only.
   Here it's one path, full panel; a second path is the obvious held-out set (and
   would then require freezing the normalisation stats).
6. **`m` (self-adaptive weights) is unbounded above** and updated by pure
   gradient ascent with `lr = 1.0`. Inherited verbatim; worth watching if a
   component's `m` diverges.

---

## 8. Verdict counts

| | Count |
|---|---|
| SAME (verbatim / rename-only) | 15 constructs |
| CHANGED (same construct, new contents) | 12 |
| DIFFERENT — new | 9 |
| DIFFERENT — deleted | 22 functions/cells |

Roughly: **the training infrastructure is inherited, the model and the physics
are not.** What survives from the original is the *scaffolding* — Flax modules,
optax, the WamOL saddle-point balancer, checkpointing, the
`error → dict → weighted sum` loss architecture. Everything downstream of
"what does the network represent and what equation constrains it" was rebuilt.
