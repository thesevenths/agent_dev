# P2-03A — Action-conditioned JEPA: Architecture Specification & Staged Plan

> **Design document only (Phase 0 + Phase 0.1 review amendments).** No dataset,
> simulator, model or training code is implemented here. Every interface below
> is derived from the audited source; see `P2_03A_ARCHITECTURE_AUDIT.md` for
> `file:line` evidence.
>
> **Phase 0.1 (this revision):** the multi-step target-window formula, E0/E1
> definitions, Parameter-OOD grouping, G2/G3 metrics, collapse criteria, physics
> config and model dispatch were corrected/locked per the review. The full
> item-by-item change log (with the original text preserved) is
> `P2_03A_REVIEW_AMENDMENT.md`. Historical Phase 0 audit evidence is NOT
> deleted; corrections are marked **AMENDED (Phase 0.1)** inline.
> Implementation begins only after review of the audit + this plan.

## 1. Research objective

Move from unconditional latent prediction to intervention-conditioned latent
prediction on a **controlled physical system**, keeping SciNet++
domain-independent:

```
legacy / A2 infrastructure:   z_{t+1} = f_theta(z_t)
P2-03A:                        z_{t+1} = f_theta(z_t, u_t)
```

| | P2-02 (done) | P2-03A (this phase) |
|---|---|---|
| Space | raw state `x` | latent `z` |
| Input | `x_t, u_t` | `z_t, u_t` |
| Output | `x_hat_{t+1}` | `z_hat_{t+1}` |
| Model | `ResidualStatePredictor` | `Encoder + ConditionalPredictor + EMA target` |
| Question | does action add predictive value? | can a **representation** learn action-conditioned dynamics? |
| Domain | Pusher-v1 (arm kinematics) | torque-controlled pendulum (controlled physics) |

P2-02 already answered its question positively (B2 skill 0.8749). P2-03A must
**not** be justified by that latent loss alone; the Linear Probe and the G4
gate are what make the comparison meaningful.

## 2. Data contract and EXACT window indexing

### 2.1 Episode layout

For one episode of length `L`:

```
observations : x[0], x[1], ..., x[L-1]        x_t ∈ R^D   (D = 2: theta, omega)
conditions   : u[0], u[1], ..., u[L-2]        u_t ∈ R^U   (U = 1: torque)
```

Fixed A1 relation: **`(x_t, u_t) → x_{t+1}`**. `u_t` is the action applied at
state `x_t`; it is NOT the action that led to `x_t`.

Physical parameters `(m, L, b)` are per-episode constants recorded in the
manifest; the integrator is deterministic RK4 with fixed `dt`.

### 2.2 Adjacent shifted windows (the critical alignment contract)

Let `K = block_size`, and let a window start at `s` with `0 ≤ s ≤ L-K-1`.
Use A1's `dynamics_batch_from_windows(window_len=K+1, stride=1)`, which
returns, for each row, `obs[s : s+K+1]` and `cond[s : s+K]` and guarantees the
window never crosses an episode boundary (`contracts.py:241-312`).

Define `t = s + K - 1` (the LAST state of the context block). Then:

| Tensor | Definition | Shape | Meaning |
|---|---|---|---|
| `context_block` | `flatten(obs[s : s+K])` | `[B, K*D]` | `x_{t-K+1} … x_t` |
| `action_t` | `cond[s+K-1]` | `[B, U]` | **`u_t`**, applied at `x_t` |
| `target_block` | `flatten(obs[s+1 : s+K+1])` | `[B, K*D]` | `x_{t-K+2} … x_{t+1}` (shifted +1) |

Identities that MUST hold and MUST be unit-tested:

```
context_block[..., -D:]      == x_t
target_block[..., -D:]       == x_{t+1}
target_block[..., :-D]       == context_block[..., D:]      (shift by exactly 1)
action_t                     == u_t  where (x_t, u_t) -> x_{t+1}
every window's episode_id is single-valued (no cross-episode stitching)
```

**AMENDED (Phase 0.1, review item 1) — multi-step target window.** The Phase 0
draft wrote `target_k_block = flatten(obs[s+k : s+k+K+1])`, which slices **K+1**
frames and therefore cannot be fed to an `Encoder` that requires `[B, K*D]`.
Corrected to exactly K frames:

```
target_k_block = flatten(obs[s+k : s+k+K])         # EXACTLY K frames -> [B, K*D]
action_seq     = cond[s+K-1 : s+K-1+k]             # u_t ... u_{t+k-1}  (k actions)
z_pred_k       = ConditionalPredictor.rollout(z_context, action_seq)[:, k-1]
z_target_k     = stop_gradient(target_encoder(target_k_block))
```

With `t = s+K-1`, the last frame of `target_k_block` is `obs[s+k+K-1] = x_{t+k}`,
so `z_pred_k` (the k-th autoregressive step) and `z_target_k` are aligned.

**Specification-level shape assertions** to be implemented as unit tests in P2
(`tests/dynamics/test_trajectory_windows.py`, NOT yet implemented — this phase
is documentation + validation hardening only). Verified numerically against a
synthetic `obs[L,D]`, `cond[L-1,U]` with `K=10, D=2, L=60, s=3, t=12`:

| Horizon k | `target_k_block` shape | `action_seq` shape | last frame index | expected `x_{t+k}` | equals one-step target? |
|---|---|---|---|---|---|
| 1 | `(B, K*D)` = (1,20) | (1,1,U) | 13 | `x_13` | **YES (elementwise identical)** |
| 2 | `(B, K*D)` = (1,20) | (1,2,U) | 14 | `x_14` | no |
| 5 | `(B, K*D)` = (1,20) | (1,5,U) | 17 | `x_17` | no |
| 10 | `(B, K*D)` = (1,20) | (1,10,U) | 22 | `x_22` | no |
| 20 | `(B, K*D)` = (1,20) | (1,20,U) | 32 | `x_32` | no |

The erroneous `K+1` formula yields `(1,22)` for every horizon, confirming the
dimensionality bug. **Horizon 1 MUST be asserted elementwise equal to the
single-step `target_block`** (`obs[s+1:s+K+1]`), which the corrected formula
satisfies.

**Additional boundary discovered during the amendment (not in the review):**
a valid multi-step window requires `s+k+K <= L` and `s+K-1+k <= L-2`, i.e.

```
0 <= s <= L - K - k_max        (single-step bound L-K-1 is only valid for k=1)
```

With `K=10, L=60, k_max=20` the start bound is `s <= 30` (not 49). P1 MUST
derive windows using `k_max`, otherwise horizon-20 slices are silently
truncated by NumPy and become short `[B, <K*D]` tensors. This is a required
test case, not an optional detail.

`ConditionalPredictor.rollout` (`conditional_predictor.py:148-197`) already feeds
predicted latents back and preserves the graph, so **no ground-truth latent is
re-injected**. Ground-truth latents appear only as EMA-target labels.

### 2.3 Why a new module is still needed

`dynamics_batch_from_windows` returns a `DynamicsBatch` of `(obs-window,
cond-window)` pairs, not the triple `(context_block, action_t, target_block)`,
and does not flatten to `[B, K*D]`. `trajectory_windows.py` is therefore a
**thin adapter** over it — no duplicate window/placement logic, no new
cross-episode risk. `BlockSequenceDataset` (`data/dataset.py:286`) is NOT used
because its blocks are non-overlapping and action-free.

## 3. Model: `ActionConditionedWorldModel` (new, independent)

```
models/action_conditioned_world_model.py   (NEW — never edit WorldModel)
```

Constructor (proposed, all configurable):

**AMENDED (Phase 0.1, review item 8) — explicit model dispatch.** The Phase 0
skeleton contained an incomplete `self.is_conditioned = isinstance(predictor, ...)`
and an ambiguous `predictor or ConditionalPredictor(...)` default. Replaced by a
single explicit, config-driven `predictor_mode` string. Neither existing predictor
implementation is modified; they are only instantiated/called.

```python
PREDICTOR_MODES = ("conditional", "unconditioned")

class ActionConditionedWorldModel(nn.Module):
    def __init__(self, block_size, state_dim, condition_dim,
                 latent_dim=8, encoder_hidden=(64, 32),
                 predictor_hidden=(64, 64), encoder_type="mlp",
                 predictor_mode="conditional",              # EXPLICIT, not isinstance
                 ema_momentum=0.996):
        if predictor_mode not in PREDICTOR_MODES:
            raise ValueError(f"predictor_mode must be one of {PREDICTOR_MODES}")
        self.predictor_mode = predictor_mode
        self.encoder = Encoder(block_size, state_dim, encoder_hidden,
                              latent_dim, kind=encoder_type)        # REUSE
        if predictor_mode == "conditional":
            self.predictor = ConditionalPredictor(                 # REUSE, no edits
                latent_dim, condition_dim, predictor_hidden)
        else:                       # E1 unconditioned JEPA ablation
            self.predictor = Predictor(                            # REUSE, no edits
                latent_dim, predictor_hidden, kind="mlp")
        self.target_encoder = copy.deepcopy(self.encoder)           # EMA PATTERN
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
```

Methods:

| Method | Behaviour |
|---|---|
| `encode(block_flat)` | online encoder, differentiable |
| `encode_target(block_flat)` | `@torch.no_grad()` EMA encoder (mirrors `world_model.py:76-79`) |
| `predict_step(z, condition)` | dispatch on `self.predictor_mode`: `conditional` → `self.predictor(z, condition)` (condition MUST be a tensor, never `None`); `unconditioned` → `self.predictor(z)` (condition MUST be `None`/ignored, and passing a tensor raises `ValueError`) |
| `rollout_latent(z0, conditions)` | only valid for `predictor_mode="conditional"`; delegates to `ConditionalPredictor.rollout`; raises otherwise |
| `update_target_encoder(momentum)` | `@torch.no_grad()` EMA copy (`world_model.py:128-134`) |
| `forward(context, action, targets_by_horizon, horizons)` | returns `(preds_by_horizon, targets_by_horizon, z_context)` |

Dispatch tests required in P2: both modes produce `[B,L]`; a tensor passed to
`unconditioned` and `None` passed to `conditional` are both rejected; E1 and E2
constructors differ ONLY in the predictor submodule. Capturing exact parameter
counts is mandatory (review item 4 / §5 below).

**Checkpoint isolation:** new checkpoints are separate files with a
`model_class = "ActionConditionedWorldModel"` tag; a two-way load test must
prove legacy checkpoints still load into `WorldModel` with `strict=True` and
that new checkpoints cannot be silently loaded into `WorldModel` (map §4).

**E1 unconditioned ablation:** instantiate legacy `Predictor(z)` and inject it.
Instantiation is not modification, so §8 is respected. `condition=None` must
NEVER be passed to A2's `ConditionalPredictor` (it rejects it by design,
`conditional_predictor.py:102-108`). E4 (zero action) is a DIFFERENT experiment:
same conditional architecture with a genuine all-zero tensor.

## 4. Training

### 4.1 Single-step first (P2)

```python
z_context = model.encode(context_block)                  # [B, L]
z_target  = model.encode_target(target_block).detach()   # [B, L], no grad
z_pred    = model.predictor(z_context, action_t)        # [B, L]

pred_l, _ = prediction_loss({1: z_pred}, {1: z_target}, [1])   # utils/losses.py:27
vic_l = vicreg_loss(z_context, var_weight, cov_weight)          # utils/losses.py:83
loss  = pred_weight * pred_l + vic_l

optimizer.zero_grad(); loss.backward(); optimizer.step()
model.update_target_encoder(ema_momentum(step, warmup_steps, base_momentum))
```

Only `requires_grad=True` parameters enter the optimizer (mirrors
`train.py:141-142`). VICReg acts on `z_context` only, as in the legacy loop
(`train.py:174-178`).

### 4.2 Multi-step only after single-step passes (P4)

Use `preds_by_horizon = {k: rollout[:, k-1] for k in horizons}` and
`targets_by_horizon = {k: encode_target(target_k_block).detach()}` with
`horizons = [1, 2, 5, 10]`, then `prediction_loss(preds, targets, horizons)`
with weights `w_k` from config. Keep VICReg unchanged. Do **not** introduce
Transformer encoders or extra regularisers in this phase.

### 4.3 Logged every epoch

EMA momentum, VICReg coefficients, per-dimension latent variance, covariance
off-diagonal magnitude, prediction loss per horizon, gradient norms, seed,
device, dataset version, config hash, git commit, duration. Exactly one unique
`experiment_id` per `(config_hash, variant, seed)`; never overwrite.

### 4.4 First-round config (design values, NOT validated results)

| Key | Value |
|---|---|
| `latent_dim` | 8 (or 16 — Q4) |
| `encoder_hidden` / `predictor_hidden` | `(64,32)` / `(64,64)` |
| `block_size` | 10 |
| `batch_size` | 128 |
| `lr` / `weight_decay` | `1e-3` / `1e-4` |
| `seeds` | 5 independent model seeds (dataset_seed strictly separate) |
| `device` | CPU smoke first; GPU optional |

**AMENDED (Phase 0.1, review item 7) — complete physics configuration.** All of
the following MUST be explicit config keys and recorded in the dataset manifest.
Values below are **pilot starting points, NOT validated results**; P1-A must run
20 pilot episodes and prove numerical stability BEFORE full collection (review §6).

| Key | Pilot value | Status |
|---|---|---|
| `gravity_g` | `9.81` | unverified starting value |
| `dt` | `0.02` | unverified starting value |
| `integrator` | deterministic RK4, fixed `dt` | required |
| `episode_steps` | `1000` | unverified starting value |
| `theta` representation | **unwrapped** (never wrapped to ±π) | required — avoids artificial discontinuities |
| `theta0_range` / `omega0_range` | e.g. `[-π, π]` / `[-1, 1]` | unverified starting values |
| `torque_bounds` | `[-1.0, 1.0]` | unverified starting value |
| `action_hold_steps` | piecewise-constant hold duration, configurable | required |
| `action_profiles` | random / step / sinusoid / sign-alternating, diverse in sign & duration | required |
| `mass` / `length` / `damping` distributions | uniform over the train ranges in §5.1 | required |
| `parameter_ood_ranges` | see §5.1 four-group design | required |
| `collision / NaN / Inf guards` | assert finite state at every RK4 step | required |

Pilot stability checks (must all pass before full collection): RK4 finite over
all 20 pilot episodes; zero-torque trajectory matches the expected damped
pendulum behaviour; positive vs negative torque produce distinguishable state
trajectories; exact-action replay error within declared tolerance; no NaN/Inf or
unintended blow-up. These are **gates, not assumptions**.

**AMENDED (Phase 0.1, review item 6) — pre-registered collapse criteria.** The
following thresholds MUST live in `configs/dynamics/controlled_pendulum_jepa.yaml`
and MUST be frozen BEFORE any `frozen_test` / `parameter_ood_test` result is
inspected (no post-hoc threshold selection):

| Criterion | Config key | Pilot threshold | Gate |
|---|---|---|---|
| per-dimension latent std | `collapse.min_latent_std` | `0.1` | G1 |
| fraction of collapsed dims | `collapse.max_collapsed_fraction` | `0.25` | G1 |
| covariance off-diagonal magnitude | `collapse.max_cov_offdiag` | `0.1` | G1 |
| finite latent check | `collapse.require_finite` | `true` | G1 |

These values are also unverified starting points; they may be revised ONLY on
`train`/`validation` evidence and every revision must be logged with its reason.

## 5. Six experiments (E0–E5)

| ID | Encoder | Predictor | Action | Role |
|---|---|---|---|---|
| E0 | — | Persistence | — | lowest baseline (locked below) |
| E1 | MLP | unconditioned (legacy `Predictor(z)`) | none | ordinary JEPA |
| E2 | MLP | conditional | correct | **target model** |
| E3 | MLP | conditional | shuffled | negative control |
| E4 | MLP | conditional | zero tensor | action-contribution check |
| E5 | MLP | conditional | correct | multi-step rollout |

### 5.0 AMENDED (Phase 0.1, review item 2) — E0 Persistence locked

```
x_hat[t+h] = x[t]        for every horizon h >= 1     (raw state, unwrapped theta)
```

- E0 uses **no Encoder, no Predictor, no Linear Probe, no training**.
- Its raw-state MSE / MAE are computed on **exactly the same evaluation episodes
  and timesteps** as every learned model (identical mask, identical split).
- `Skill = 1 - MSE_model / MSE_persistence`. Report **the denominator
  `MSE_persistence` explicitly** in every table. If `MSE_persistence` is near
  zero (below a config-declared `min_denominator`), report `skill=None` with an
  explicit "baseline degenerate" flag instead of dividing (no Inf, no silent
  clipping). `Skill > 0` = better than standing still, `= 0` = equal, `< 0` =
  worse.
- E0 is the denominator of G4 and of all skill scores; it is NOT a latent
  persistence model with a random encoder.

### 5.1 AMENDED (Phase 0.1, review item 3) — Parameter OOD split into four groups

The Phase 0 design changed all three physical parameters simultaneously, making
it impossible to attribute any degradation. Retain the ranges but partition the
300 OOD episodes into four **disjoint, labelled** groups of 75 episodes each:

| OOD group | Mass | Length | Damping | Episodes |
|---|---|---|---|---|
| `mass_ood` | **1.4–1.6** | train 0.8–1.2 | train 0.05–0.15 | 75 |
| `length_ood` | train 0.8–1.2 | **1.4–1.6** | train 0.05–0.15 | 75 |
| `damping_ood` | train 0.8–1.2 | train 0.8–1.2 | **0.20–0.30** | 75 |
| `joint_ood` | **1.4–1.6** | **1.4–1.6** | **0.20–0.30** | 75 |

For each single-parameter group the other two MUST remain inside their train
ranges. Per-episode `(m, L, b)` values, the four group ranges, and the group
episode counts MUST be written into `manifest.json`; the audit script must
assert the OOD ranges are disjoint from train (R8). Report G6 separately per
group, so a failure can be attributed to one physical parameter.

### 5.2 AMENDED (Phase 0.1, review item 4) — capacity comparability

- E1 = legacy `Predictor(z)`; E2 = `ConditionalPredictor(z, u)`. Their first
  layers have different input widths (`L` vs `L+U`), so **parameter counts
  differ by construction**. This is expected and MUST be reported, not hidden.
- E1 and E2 share the **identical Encoder architecture**, identical
  `latent_dim`, identical `predictor_hidden`, identical epochs / batch / seeds /
  train-validation protocol. Record `sum(p.numel())` for encoder, predictor and
  total for every variant.
- **E2, E3 and E4 MUST have exactly the same architecture and parameter count**
  (they differ only in the action tensor supplied). This makes `E2 vs E3` and
  `E2 vs E4` clean action-information tests.
- Consequence: a win of E2 over E1 cannot be attributed *solely* to action
  conditioning; the primary evidence for action information is `E2 vs E3` / `E2
  vs E4`, with `E2 vs E1` reported as a secondary comparison with the capacity
  difference stated explicitly.

E1 and E2 use **identical training budgets** (epochs, batch, seeds,
train/validation protocol) and evaluation protocols, but their **parameter
counts differ by construction** (see the AMENDED note above). E2, E3 and E4
share **identical architecture and parameter count**; the capacity-comparability
rules in the AMENDED note above supersede this paragraph. E3 uses a fixed-seed
`shuffle_conditions` (generic helper from `models/state_dynamics.py`, no domain
constants).

## 6. Evaluation

1. **Representation quality:** per-dimension latent std, covariance
   off-diagonal magnitude, collapse flag (G1), latent prediction MSE.
2. **Dynamics quality:** horizon errors `[1,2,5,10,20]`, open-loop stability,
   action-conditioned vs unconditioned gap.
3. **Raw-state Linear Probe (fit on TRAIN ONLY):** fit a linear/ridge map
   `z_pred → x_{t+1}` using TRAIN representations with the trained encoder
   frozen; evaluate frozen_test and parameter_ood_test WITHOUT refitting.
   Report RMSE / MAE / `Skill = 1 − MSE_probe/MSE_persistence` against E0.
4. **Physical generalisation:** report by `(m, L, b)` groups and explicitly
   quantify Parameter-OOD degradation (G6 does not require high OOD skill, only
   an honest measurement).
5. **Statistics:** 5 seeds, episode-level paired bootstrap (10,000 resamples,
   95% CI), mean AND median. Correlated timesteps are never independent samples.

### 6.1 AMENDED (Phase 0.1, review item 5) — G2/G3 primary metrics are raw-state

Different models learn latent spaces with different scales, rotations and
coordinate systems, so a lower **latent** MSE alone does not establish better
physical prediction. Therefore:

| Comparison | PRIMARY metric (gate decision) | SECONDARY (diagnostic only) |
|---|---|---|
| G2: E2 vs E1 | **train-only Linear-Probe raw-state RMSE** + Frozen-Test raw-state MAE | latent prediction MSE |
| G3: E2 vs E3 | **train-only Linear-Probe raw-state RMSE** + Frozen-Test raw-state MAE | latent prediction MSE |
| all | paired per-episode errors, 95% episode-level paired bootstrap CI | — |

Protocol constraints (all MUST hold):

- Fit **each** variant's probe on TRAIN representations only.
- Use **VALIDATION** for probe hyper-parameter selection (ridge alpha, etc.).
- NEVER fit or tune a probe on `frozen_test` or `parameter_ood_test`.
- Evaluate both test splits **without refitting**.
- Compare E0 on the identical episodes/timesteps and report the denominator.
- Latent prediction loss remains a **training diagnostic**, never the sole
  cross-model physical-capability metric. This directly addresses the fact that
  E1/E2 have different capacity (§5.2).

## 7. Acceptance gates (operational criteria — to be reviewed)

| Gate | Criterion |
|---|---|
| G0 | RK4 replay with the exact persisted action reproduces `x[t+1]`; manifest SHA-256 + parameter ranges + four OOD group ranges/counts + split isolation verified |
| G1 | No latent collapse against the **pre-registered** thresholds in config (§4.4): per-dim std, collapsed fraction, covariance off-diagonal |
| G2 | E2 **train-only Linear-Probe raw-state RMSE/MAE** < E1, paired 95% episode-level CI excludes 0; capacity difference reported |
| G3 | E2 **train-only Linear-Probe raw-state RMSE/MAE** < E3 shuffled-action, paired 95% episode-level CI excludes 0 |
| G4 | E2 Linear-Probe raw-state RMSE < E0 Persistence, skill > 0, denominator reported |
| G5 | Horizons 1/2/5/10/20 finite, no divergence, open-loop feedback of predicted latents |
| G6 | Parameter OOD measured **per four groups** and degradation reported honestly (no pass/fail on magnitude) |

**Failed gates must be reported, never hidden.** A low latent prediction loss
alone is not evidence of learned physics, nor of planning/control ability.
**Stage gates:** if P1's G0 fails, P2 training MUST NOT start; if P2's
single-step G1–G3 fail, P4 multi-step MUST NOT start.

## 8. Staged development order (task §6)

| Stage | Work | Acceptance |
|---|---|---|
| **P0** | Architecture audit (read-only) | **DONE** (`144a300`), 4 documents |
| **P0.1** | Review amendments: fix window formula, lock E0/E1, four OOD groups, G2/G3 metrics, collapse criteria, physics config, model dispatch | **current phase** — docs only, Commit A |
| **P0-fix** | Narrow validation hardening in `state_dynamics.py` + `normalization.py`, tests, full regression | **current phase** — Commit B; compare against the **actual** baseline (210 passed + 22 subtests, measured at `144a300`) |
| P1-A | ODE + RK4 + control signal generator + 20 pilot episodes | numerical stability |
| P1-B | Full dataset (train/val/frozen/four-group OOD) + manifest | manifest complete |
| P1-C | Data acceptance: exact-action replay, physics sanity, split isolation | **G0** (STOP GATE) |
| P2 | Single-step `ActionConditionedWorldModel` + E0–E4, 5 seeds | G1–G3 |
| P3 | EMA + VICReg diagnostics | no collapse |
| P4 | Multi-step JEPA + E5, horizons 1/2/5/10/20 | G4–G5 |
| P5 | Linear Probe (train-only) + four-group Parameter OOD | G6 |
| P6 | `P2_03A_FINAL_REPORT.md` + feature-branch commits | independent reproduction |

**P0 / P0.1 / P0-fix are separate commits.** Single-step JEPA must pass before
any multi-step loss is introduced (one variable at a time). Per review §5, P1
starts with **20 pilot episodes only**; full collection follows only after the
pilot stability checks pass. Stage gates: G0 failure ⇒ no P2; P2 failure ⇒ no P4.

## 9. Proposed new files (all NET-NEW)

```
dynamics/trajectory_windows.py
models/action_conditioned_world_model.py
data/controlled_pendulum.py
configs/dynamics/controlled_pendulum_jepa.yaml
scripts/collect_controlled_pendulum.py
scripts/audit_controlled_pendulum.py
scripts/train_action_jepa.py
scripts/evaluate_action_jepa.py
tests/models/test_action_conditioned_world_model.py
tests/dynamics/test_trajectory_windows.py
datasets/controlled_pendulum_v1/{train,validation,frozen_test,parameter_ood_test}/manifest.json
docs/architecture/P2_03_ACTION_JEPA.md            (this file)
docs/architecture/P2_03A_FINAL_REPORT.md           (P6)
```

## 10. Restrictions & scientific boundaries

- Do NOT modify `models/world_model.py`, `models/predictor.py`,
  `models/conditional_predictor.py`, `utils/losses.py` behaviour,
  `train.py`, `data/pendulum.py`, existing datasets, or historical checkpoints.
- No `robotics` imports and no Pusher constants in generic `models/` / `dynamics/`.
- `pusher_v1` stays byte-identical; P2-02B contact-rich data is a **parallel,
  separate** effort and must not be mixed into this Core work.
- Frozen / Parameter-OOD splits never touch training, early stopping,
  hyper-parameter selection or Linear-Probe fitting.
- No MPC, no closed-loop control, no planning claims (that is P2-04).
- No claim of "physical law discovery" from a low latent loss.
- Commit to `feature/p2-03a-action-jepa` first; **do not auto-merge to `main`**.
- Full existing regression suite must remain green (baseline at `e4eccc0`:
  210 passed + 22 subtests).

## 11. Stop condition

Phase 0 delivered the audit, dependency map, gap analysis and this specification
with **zero production code changes**. Phase 0.1 (review amendments) + P0-fix
deliver the corrected specification, one narrow validation-hardening commit and
the full regression evidence, also with **no dataset, simulator, model or
training code**.

**STOP after P0-fix.** Per the review task book §7, do NOT implement
`controlled_pendulum.py`, do NOT generate the dataset, do NOT train JEPA, do NOT
implement MPC, do NOT modify robotics, do NOT merge `main`, and do NOT open a
PR. Wait for explicit review approval before starting P1. The remaining
unresolved scientific decisions (Q3 physics-range stability, Q4 latent_dim,
Q5 `conditional_predictor.py` validation, Q6 child naming) stay explicitly
documented in `P2_03A_GAP_ANALYSIS.md`; Q7 (push feature branch) is CLOSED.
