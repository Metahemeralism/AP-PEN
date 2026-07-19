# PINN rewrite — action list (`-6.pdf` alignment)

Adjustments to make `PINN_implementation.ipynb` match
`mathematical_derivations-6.pdf`. **WamOL loss-balancing changes are parked**
(tracked separately) — everything below is the architecture + loss-term work.

Ordered roughly in the sequence to implement. Full diagnosis with severities
lives in `pinn_implementation_provenance.md` (⚠️ OUTSTANDING section).

---

## 1. Split into two networks (§6, eq. 30)

- [ ] Add a **surface network** `F̂_θ : (S, δ, τ) → ℝ⁺`. Input dim **3**.
- [ ] Keep a **path network** `δ̂_φ : t → ℝ`. Input dim **1**, unconstrained sign.
- [ ] Train them **jointly** — one `TrainState` holding both param sets, or two
      states stepped together. `ann_in_dim` is no longer a single value; the two
      nets have different input dims.
- [ ] Update `config_*` accordingly (the current single `ann_in_dim=2` is wrong
      for both nets).

## 2. Fix the path-network input (§6, p.4 remark 1)

- [ ] `δ̂_φ` takes **`t` alone** — drop `S` from its input.
- [ ] In `error()`, change `delta_hat_fn(t, S)` → `delta_hat_fn(t)`; the network
      call becomes `fn_phi(jnp.atleast_1d(t))[0]`.

## 3. Remove the closed form from the training objective (§6 "Role of the closed form")

This is the core correctness fix. The closed form (`B_coeff`/`A_coeff`) must
**not** appear in any training loss — only in data-gen (`monte_carlo.ipynb`) and
validation.

- [ ] `e_acc`: price with the **surface net**, not `ln S + B·δ̂ + A`. Per eq. 42–43:
      `F̂_i = F̂_θ(S_i, δ̂_φ(t_i), τ_i)`, then `e_acc = (ln F̂_i − ln F_obs_i)²`.
- [ ] `e_pde`: compute the residual of the **surface net** `F̂_θ` (see §4), not
      the differentiated closed-form composite. Delete the current `F_hat_fn` /
      `pde_residual` block that wraps `S·exp(B·δ̂+A)`.
- [ ] Carry penalties (`e_arb_cac`, `e_arb_rcac`): price through `F̂_θ(x_i)` at
      inferred states `x_i = (S_i, δ̂_φ(t_i), τ_i)`, not the closed form (§7.1–7.2,
      eq. 34/37).
- [ ] The `from gs_wamol.physics.gibson_schwartz import ... B_coeff, A_coeff`
      import is no longer needed in the objective — keep it only if used for a
      validation/benchmark plot.

## 4. PDE-residual loss on a collocation mesh (§10, eq. 44–45)

- [ ] Build a **separate collocation mesh** over `(S, δ, τ)` — δ is an
      **independent coordinate here**, sampled over a range, *not* `δ̂_φ(t)`.
- [ ] `L_f` = mean of `rPDE[F̂_θ](S,δ,τ)²` over mesh points, where `rPDE` is
      eq. 44 with all partials of `F̂_θ` taken by autodiff. The latent path does
      **not** appear in `L_f`.
- [ ] Decide mesh ranges/resolution for `S`, `δ`, `τ` (paper leaves the sampling
      scheme to you; the old notebook's `x_mesh` is the analogue).

## 5. Initial-condition loss `L_b` (§11) — pick one

**Recommended: hard enforcement (eq. 47).**
- [ ] Parameterise the surface net as `F̂_θ(S,δ,τ) = S·exp(τ · N_θ(S,δ,τ))`,
      where `N_θ` is the raw MLP output.
- [ ] This gives the initial condition `F̂_θ(S,δ,0)=S` and positivity for free,
      and makes `ln F̂_θ = ln S + τ·N_θ` linear in the raw output.
- [ ] Then `L_b` is **dropped** from the objective entirely.

**Alternative: soft loss (eq. 46).**
- [ ] Plain positive-output surface net (softplus/exp on output for ℝ⁺).
- [ ] Add `L_b = mean((F̂_θ(S_j,δ_j,0) − S_j)²)` on a `τ=0` slice of the mesh.
- [ ] Register `e_b` as a loss component (adds a weight to balance).

## 6. Surface-network positivity

- [ ] The surface net output must be ℝ⁺. Handled automatically by the eq. 47
      transform (step 5, recommended path). If you go soft-`L_b` instead, add an
      explicit positive output activation to the surface net only.
- [ ] The **path net stays sign-unconstrained** — the softplus removal already
      done is correct for it, keep it.

## 7. Constants

- [ ] `DELTA_MIN = -0.3` (§7.3, eq. 39) — currently `-3.0`, off by 10×, so the
      floor never fires. Set to `-0.3`.
- [ ] `DELTA_MAX` (rcac wedge, δmax) — paper gives no number; still your call.
- [ ] `STORAGE_COST_U` (u ≥ 0) — paper gives no number; still your call.

## 8. Downstream / bookkeeping

- [ ] `plot_delta_recovery`: `δ̂_φ` now takes `t` alone — drop the `S` arg from
      the eval call.
- [ ] `init_l_ws` / `init_params_sa`: component set changes (add `e_b` if using
      soft IC; per-point shapes follow the new mesh vs. data-manifold split).
- [ ] Loss-component lists (`loss_fn_lb` MLP/PINN/ACPINN): re-derive which terms
      belong to each variant under the two-net setup.
- [ ] Update `pinn_implementation_provenance.md` once the rewrite lands (the
      ⚠️ OUTSTANDING section can then move to "done").

---

## Removed: the `configs/` directory

`configs/{mlp,pinn,acpinn}.py` were deleted (2026-07-19, commit `5d9bec8` has the
originals — `git show 5d9bec8:configs/pinn.py`). Each was a four-line
`ml_collections` shim re-exporting `get_*_config()` from
`src/gs_wamol/utils/config.py`, existing to be passed as
`--config=configs/pinn.py` to an `absl` `config_flags` launcher. **That launcher
was never written** — nothing in the repo imported them, so they were pure
indirection.

**Re-add them when** a `scripts/train.py` CLI lands. The pattern is worth having
then: `config_flags.DEFINE_config_file("config")` gives reproducible, flag-logged
runs (`--config.num_epochs=20000`), which matters for regenerating thesis figures
from a recorded command. Under the two-net rewrite the presets also stop being
one-knob variants — `ann_in_dim` splits into separate surface/path dims (step 1)
and the loss-component sets diverge (step 8) — so there will be real content to
hold.

The defaults themselves still live in `src/gs_wamol/utils/config.py` and are
untouched. Note `data_source: "SABR_syn"` there is stale legacy IVS-calibration
naming, unrelated to Gibson–Schwartz.

---

## Parked (not in this list)

- WamOL per-category gradient-support balancing (§12.2, eq. 55–56).
- Online/time-decay ζ streaming (§9, §12.3) — optional even under `-6.pdf`.
