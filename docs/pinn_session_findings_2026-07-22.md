# PINN implementation — session findings (2026-07-22)

Working session on `notebooks/PINN_implementation.ipynb`, following on from the
2026-07-19 two-network (drift net + analytic `B(τ)` + path net) rewrite. This
session was a cleanup pass: remove dead/redundant code, fix a silent
config bug found along the way, and simplify the WamOL loss-balancing to a
single level. No architecture or loss-term changes were made.

---

## 0. TL;DR

1. Removed dead code across the notebook: unused imports, an unused pricer
   function, an unused network class + its builder function, unused dataclass
   fields, unused config fields, and two unused variables. See §1.
2. Found and fixed a real (silent) bug: the weight-factorization reparam flag
   was compared against the wrong type and so was **always disabled**
   regardless of config. See §2.
3. Removed the second-level **per-point** self-adaptive loss weighting
   (`params_sa`, SGD ascent on `m`) inherited from the original AC-PINN code,
   per explicit request. Kept the first-level **component-level** WamOL
   gradient-norm balancing (`l_ws` across `e_data`/`e_ode`/`e_sde`). See §3.
4. Confirmed the optimizer is unchanged: still `optax.adam` with an
   exponential-decay learning-rate schedule. A dead `optimizer = "Adam"`
   string (never read anywhere) was removed as part of the cleanup, which is
   not the same thing as changing the optimizer. See §4.
5. No new errors introduced; residual `get_errors` output is pre-existing
   notebook forward-reference lint noise (functions referencing globals bound
   in later cells), not real problems.

---

## 1. Dead code removed

| Item | Where | Why it was dead |
|---|---|---|
| `import json`, `import pandas as pd` | imports cell | never referenced |
| `term_structure()` | closed-form pricer cell | never called (only `B_coeff`/`A_coeff`/`futures`/`log_futures` are used, by the pricing transform and the self-check) |
| `ModifiedMLP` class | network cell | never instantiated — `build_nets()` only ever builds `MLP` |
| `ann_gen(config)` function | network cell | superseded by `build_nets()`; never called; referenced config fields (`ann_hidden_dim`, `ann_periodicity`, `ann_fourier_emb`, …) that don't exist in any of the run configs — would have raised `AttributeError` if it were ever called |
| `periodicity`, `fourier_emb`, `arch_name` fields | `MLP`/`ModifiedMLP` | declared, never read inside `__call__` |
| `"ann_str"` config field | `config_MLP` | only ever read by the dead `ann_gen()` |
| `"self_adaptive_lr"` config field | `config_MLP` | fed a hardcoded `jax_optimizers.sgd(1.0)` call, itself removed in §3 |
| `optimizer = "Adam"` variable | optimizer cell | set once, never read anywhere (see §4) |
| `n_dates, n_taus = log_F_obs_train.shape` | data-binding cell | computed, never used again |
| `save_params()` / `load_params()` functions | calibration cell | defined, never called — checkpointing goes through `orbax_checkpointer` instead |
| `to_state_dict`/`from_state_dict` imports | imports cell | only used by the now-removed `save_params`/`load_params` |
| `Optional` import (twice — added back, then removed again) | imports cell | only used by the now-removed `arch_name` fields |

---

## 2. Bug found and fixed: `ann_reparam` silently always off

`build_nets()` (and the now-removed `ann_gen()`) gated the weight-factorization
reparameterisation on:

```python
if config.ann_reparam == "weight_fact":
```

but every run config sets `"ann_reparam": True` — a **boolean**, not the string
`"weight_fact"`. `True == "weight_fact"` is `False`, so `reparam` was always
`None` and the reparam feature never activated for any of `MLP`/`PINN`/`DCPINN`,
regardless of the config value.

**Fix:** changed the check to `if config.ann_reparam:` so the boolean config
flag is respected.

---

## 3. Loss weighting: kept component-level, removed per-point

The original (ported AC-PINN) code balanced the loss at two nested levels:

1. **Component-level** `l_ws` — one scalar weight per loss channel
   (`e_data`, `e_ode`, `e_sde`), updated every 100 epochs via WamOL
   gradient-norm balancing (`update_loss_weights`).
2. **Per-point (piecewise)** `params_sa` / `m` — one weight *per collocation
   point / per observation*, trained via a separate SGD-ascent optimizer
   (`train_step_sa`, on a saddle-point objective) inherited verbatim from the
   original SABR/Dupire AC-PINN.

Per explicit request, level 2 is now fully removed; level 1 is kept:

- `adj(loss, lw, m)` → `adj(loss, lw)` (no more per-point multiply).
- `make_loss_lb` / `make_loss_fn` / `calibration` no longer thread `params_sa`
  or `state_sa` through anywhere.
- Removed: `init_params_sa()`, `train_step_sa()`, the
  `jax_optimizers.sgd(...)` self-adaptive optimizer, and the corresponding
  `jax.example_libraries.optimizers` import.
- Removed the now-dead `"self_adaptive_lr"` config field (nothing reads it
  anymore).
- Checkpoint payload no longer includes the `'ms'` (per-point weights) key,
  only `'params'` and `'ls'` (the component weights).

`update_loss_weights` (component-level balancing) is unchanged in logic, only
in signature (drops `params_sa`).

---

## 4. Optimizer clarification

A user question during the session: *"what optimizer am I using if not Adam?"*
— prompted by the removal of `optimizer = "Adam"` in §1.

That variable was a **dead descriptive string**, never read by any code in the
notebook. The actual optimizer, unaffected by the cleanup, is still:

```python
lr = optax.exponential_decay(init_value=learning_rate,
                              transition_steps=decay_steps,
                              decay_rate=decay_rate)
tx = optax.adam(learning_rate=lr, b1=beta1, b2=beta2, eps=adam_eps)
```

i.e. Adam with an exponentially-decaying learning-rate schedule — unchanged.

---

## 5. PINN vs DCPINN — clarified during session

Both variants share the identical drift net (`A_hat_theta(tau)`), path net
(`delta_hat_phi(t)`), and pricing transform. They differ only in:

- **Loss composition** (`loss_fn_lb`):
  - `PINN` = `e_data` (price misfit) + `e_ode` (ODE residual on the learned
    `A_hat(tau)` against the analytic `B(tau)`).
  - `DCPINN` = `PINN`'s two terms **+ `e_sde`** (OU transition-density NLL
    under the physical measure, disciplining the recovered path's dynamics,
    not just its price fit).
- **Loss balancing**: `use_balancing = config.get("balancing", True) and
  config.loss_str == "DCPINN"` — the WamOL component-weight balancer only ever
  runs for `DCPINN`; `MLP` and `PINN` always train with fixed unit weights.
- `config_DCPINN_nobal` is a control ablation: same three loss terms as
  `DCPINN`, balancer forced off, to isolate whether the balancer (vs. the
  extra `e_sde` term) drives any path-recovery collapse.

---

## 6. Notes for next session

- The `get_errors` output on the notebook shows a handful of Pylance
  false positives (`p_Q`, `normalize_tau`, `normalize_time` "not defined";
  `fig`/`F_hat_fn` "not accessed"). These are forward-reference / unused-in-cell
  lint noise from analyzing cells out of their runtime dependency order, not
  real bugs — safe to ignore.
- Two markdown cells still describe a stale/pre-rewrite API: "Input Data"
  (mentions `(t, S)` pairs and `delta_hat(t, S)`) and "Run" (mentions
  `ann_in_dim=2`, `pts_num`). Flagged but not yet corrected — do this next
  time markdown cells are touched.
- Lesson reconfirmed this session: prefer `edit_notebook_file` for full-cell
  rewrites of `.ipynb` files; small, uniquely-matched snippet edits via
  `replace_string_in_file`/`multi_replace_string_in_file` are fine, but do not
  use them for large multi-line cell bodies — the underlying JSON does not
  diff cleanly and has previously produced duplicated/corrupted cell content.
