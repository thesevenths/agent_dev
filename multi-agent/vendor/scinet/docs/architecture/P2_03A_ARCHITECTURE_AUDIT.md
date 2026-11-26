# P2-03A Phase 0 — Architecture Audit

> **Status: Phase 0 ONLY.** This is a read-only audit. NO production code was
> modified, NO dataset was created, NO model was trained, and NO merge to
> `main` was performed. All findings below are grounded in actual source code
> with `file:line` references, not README claims or recollection.
>
> Branch: `feature/p2-03a-action-jepa` (cut from `main` @ `e4eccc0`, P2-02 final).

## 0. Executive conclusion

**The architecture is ready for P2-03A, but only as NET-NEW modules.**

The existing Core is cleanly isolated from Robotics, and all three JEPA building
blocks already exist (`Encoder`, EMA target encoder pattern, VICReg loss,
`ConditionalPredictor`). What is missing is a **glue layer**: an independent
model that assembles them with an action-conditioned predictor, plus a
controlled-physics data generator and adjacent-window builder.

Three findings dominate the design:

| # | Finding | Consequence |
|---|---|---|
| F1 | `WorldModel` is unconditional (`self.predictor(z)`, `world_model.py:94`) and is on the **do-not-modify** list | P2-03A MUST be a new independent class, not a modification of `WorldModel` |
| F2 | `Encoder.forward` consumes a **flattened time block** `[B, K*D]` (`encoder.py:139-150`), and `BlockSequenceDataset` builds **non-overlapping** blocks (`data/dataset.py:286`) with no actions | A new adjacent-window builder is required; neither existing class can be reused directly |
| F3 | `data/pendulum.py` is an **analytic damped oscillator** `A·e^(−γt)·cos(ωt+φ)`, `state_dim=1`, no control (`:26-57`) | It is a DIFFERENT physical system from the torque-controlled pendulum; a new simulator is required, and the §8 ban on modifying it is correct |

**Legacy checkpoint compatibility was re-verified read-only**: all three 2026-09
checkpoints load with `strict=True`, `missing=[] unexpected=[]`.

**Stop condition:** per the task specification §1 and §10, implementation begins
only after this audit and the implementation plan are reviewed. Three minor
P2-02 validation issues are **identified with exact fixes below, but NOT applied
in this phase** (see §6 and the constraint conflict in §7).

---

## 1. Audit scope and method

Read in full or in the cited ranges:

| Area | Files |
|---|---|
| Models | `models/encoder.py`, `predictor.py`, `conditional_predictor.py`, `world_model.py`, `state_dynamics.py` |
| Losses / training | `utils/losses.py`, `train.py`, `utils/config.py`, `utils/seed.py` |
| Legacy data | `data/pendulum.py`, `data/dataset.py`, `configs/pendulum.yaml` |
| A1/A2 + P2-02 infra | `dynamics/contracts.py`, `validation.py`, `normalization.py`, `metrics.py`, `evaluation.py`, `__init__.py` |
| P2-02 experiment layer | `configs/dynamics/pusher_v1_baselines.yaml`, `scripts/train_dynamics.py`, `scripts/evaluate_dynamics.py`, `scripts/audit_pusher_dynamics.py` |
| Robotics (for the dtype finding) | `robotics/data/collector.py`, `robotics/data/trajectory.py` |

Read-only verification command (executed, output in §5):

```python
torch.load(p, map_location="cpu", weights_only=False)
m = WorldModel.from_config(ck["config"], ck["state_dim"])
m.load_state_dict(ck["model_state"], strict=True)
```

---

## 2. Real interfaces (verified signatures)

| Component | Location | Actual contract |
|---|---|---|
| `build_mlp(dims, activation=nn.ReLU)` | `models/encoder.py:41-48` | `nn.Sequential`, no activation after final projection |
| `Encoder(block_size, state_dim, hidden_dims=(64,32), latent_dim=3, kind="mlp", ...)` | `encoder.py:79-93` | `input_dim = block_size * state_dim` (`:99`) |
| `Encoder.forward(x)` | `encoder.py:139-150` | **`[B, K*D]` flat block → `[B, latent_dim]`**; conv/transformer reshape internally |
| `Predictor(latent_dim, hidden_dims, kind)` | `predictor.py:42-50` | **unconditional** |
| `Predictor.forward(z)` | `predictor.py:80-86` | `[B,L] → [B,L]`, no condition argument |
| `ConditionalPredictor(latent_dim, condition_dim, hidden_dims=(256,256))` | `conditional_predictor.py:79-98` | `net = build_mlp([L+U, *hidden, L])` (`:98`) |
| `ConditionalPredictor.forward(z, condition)` | `conditional_predictor.py:138-145` | `[B,L]+[B,U]` or `[B,T,L]+[B,T,U]`; rejects `None` (`:102-108`), exact leading dims (`:127`), NaN/Inf (`:132-135`) |
| `ConditionalPredictor.rollout(z0, conditions)` | `conditional_predictor.py:148-197` | `[B,L] + [B,H,U] → [B,H,L]`, autoregressive, step `t` uses `conditions[:,t]` (`:194-196`), **graph preserved** |
| `WorldModel(block_size, state_dim, latent_dim=3, ...)` | `world_model.py:34-49` | children: `encoder`, `predictor`, `target_encoder` |
| `WorldModel.target_encoder` | `world_model.py:67-69` | `deepcopy(encoder)` + `requires_grad_(False)` |
| `WorldModel.encode_target` | `world_model.py:76-79` | `@torch.no_grad()` |
| `WorldModel.predict_multistep(z0, horizons)` | `world_model.py:82-97` | `z = self.predictor(z)` (`:94`) — **no action** |
| `WorldModel.forward(blocks, horizons, context_block)` | `world_model.py:99-125` | `preds/targets` keyed by horizon, targets under `no_grad` (`:121-124`) |
| `WorldModel.update_target_encoder(momentum)` | `world_model.py:128-134` | EMA `target = m*target + (1−m)*online` |
| `WorldModel.from_config(cfg, state_dim)` | `world_model.py:137-159` | reads `cfg["model"]` + `cfg["data"]["block_size"]` |
| `prediction_loss(preds: Dict[int,Tensor], targets, horizons)` | `utils/losses.py:27-50` | **expects dict-of-horizons**, returns `(scalar, per_horizon)` |
| `variance_loss(z, target_std=1.0, eps=1e-4)` | `losses.py:53-62` | hinge on per-dim std |
| `covariance_loss(z)` | `losses.py:65-80` | off-diagonal squared covariance / d |
| `vicreg_loss(z, var_weight, cov_weight)` | `losses.py:83-85` | **variance + covariance only**; invariance is supplied by `prediction_loss` |
| `ema_momentum(step, warmup_steps, base_momentum)` | `train.py:69-80` | cosine ramp |
| JEPA train step | `train.py:162-187` | `prediction_loss` + `vicreg_loss` → `loss.backward()` → `optimizer.step()` → `update_target_encoder(momentum)` |
| JEPA config keys | `configs/pendulum.yaml:43-49` | `rollout_steps`, `ema_momentum`, `ema_warmup_epochs`, `pred_weight`, `vicreg_var_weight`, `vicreg_cov_weight` |
| `DynamicsBatch` | `dynamics/contracts.py:49-58` | `observations [B,T+1,D]`, `conditions [B,T,U]` or `None`, `valid_mask [B,T] bool`, `episode_ids List[str]` |
| `to_training_pairs()` | `contracts.py:109-118` | `(x_t, u_t, x_{t+1}, mask)` |
| `dynamics_batch_from_windows(observations, conditions, episode_ids, window_len=11, stride=1)` | `contracts.py:241-312` | sliding windows **strictly inside one episode**; `obs[s:e]` (`:297`) + `cond[s:e-1]` (`:301`); every row tagged with source `episode_id` |
| `fit_normalization(train_batch, eps)` | `normalization.py:118+` | train-only, masked, `state_std` clamped to `eps` |
| `NormalizationStats.from_dict(d)` | `normalization.py:95-111` | **no schema validation** |
| `paired_bootstrap_ci(a, b, n_boot, alpha, seed)` | `metrics.py:134+` | episode-level paired CI; `one_step_metrics` `:67`, `episode_errors` `:96`, `describe` `:120` |
| `open_loop_rollout(model, x0, actions, stats, horizon)` | `evaluation.py:65+` | **raw-state** rollout; denormalises with `NormalizationStats` |
| `_valid_rollout_windows(mask, horizon)` | `evaluation.py:135-149` | all valid in-episode starts, aggregated back to episode level |
| `set_seed(seed)` | `utils/seed.py:19-27` | python / numpy / torch / cuDNN |
| `load_config(path)` | `utils/config.py:32-35` | `yaml.safe_load` |
| `ensure_output_dir(cfg)` | `utils/config.py:66-103` | ⚠️ **reuses the most recent run dir** (`:98`) |
| `PendulumGenerator` | `data/pendulum.py:26-78` | analytic oscillation, `state_dim=1` (`:30`), `(gamma, omega)` labels, **no control** |
| `BlockSequenceDataset` | `data/dataset.py:278-301` | `reshape(n, nb, block_size*state_dim)` (`:286`) — **non-overlapping** blocks, no actions |
| `_GENERATORS` factory / `SPLITS` | `data/dataset.py:59-63`, `:66` | hardcoded 3 generators; `("train","validation","adapt","test")` |
| `dataset_version` / `derive_split_seed` / `load_frozen_split` | `data/dataset.py:99-131`, `:134-155`, `:162-217` | content-addressed id + SHA-256 frozen manifest + per-split `RandomState` — **reusable pattern** |
| `robot...collect_episode` | `robotics/data/collector.py:38-57` | `action` generated in float64 → `env.step(action)` → stored via `RobotTransition` |
| `RobotTransition` dtype | `robotics/data/trajectory.py:25`, `:42-44` | `DEFAULT_DTYPE = np.float32` — action **downcast on persistence** |

---

## 3. Reusable components (no changes needed)

| Component | How P2-03A uses it |
|---|---|
| `Encoder` (`mlp` kind) | Instantiate with `block_size=K`, `state_dim=2`, `latent_dim ∈ {8,16}`, `hidden_dims=(64,32)`; feed flattened K-step blocks |
| `build_mlp` | Used internally by Encoder / Predictor / ConditionalPredictor; do not duplicate |
| `ConditionalPredictor` | Directly instantiate as the **conditional** predictor; `forward` for 1-step, `rollout` for multi-step open-loop latent prediction (already autoregressive, already preserves gradients) |
| `Predictor` (legacy) | **Instantiate only** for the E1 unconditioned-JEPA ablation (see §4, I8). Instantiation is not modification, so §8 is respected |
| `vicreg_loss`, `variance_loss`, `covariance_loss`, `prediction_loss` | Direct reuse; single-step JEPA wraps `{1: z_pred}` / `{1: z_target}` / `horizons=[1]` |
| EMA pattern (`deepcopy` + `requires_grad_(False)` + `@no_grad` + momentum update) | Re-implemented **inside the new class**; `WorldModel` itself is untouched |
| `ema_momentum` cosine warmup | Reuse the formula / `train.py:69-80` as reference |
| `DynamicsBatch` + `validate_dynamics_batch` | Data contract for the controlled-pendulum episodes |
| `dynamics_batch_from_windows` | **Underlying** builder for `trajectory_windows.py` (already guarantees episode isolation, see §4 I2) |
| `fit_normalization` / `NormalizationStats` | train-only normalisation for raw states and actions |
| `dynamics.metrics` (`masked_mse_loss`, `one_step_metrics`, `episode_errors`, `paired_bootstrap_ci`, `describe`) | Evaluation statistics & 95% CIs |
| `_valid_rollout_windows` pattern | Episode-internal multi-start rollout aggregation |
| `set_seed`, `load_config` | Reproducibility & config loading |
| `data/dataset.py` content-addressing / SHA-256 frozen manifest / per-split RNG | **Pattern** for `datasets/controlled_pendulum_v1/manifest.json` (new implementation, not a call into legacy methods) |

---

## 4. Incompatible interfaces / gaps (the actual work)

| ID | Gap | Evidence | Required resolution |
|---|---|---|---|
| I1 | `WorldModel` predictor is unconditional and cannot take `u_t` | `world_model.py:94`; `predictor.py:80` | New `ActionConditionedWorldModel`; **never** insert `ConditionalPredictor` into `WorldModel` |
| I2 | Existing window builder yields a `DynamicsBatch`, not a `(context_block, action_t, target_block)` triple, and does not flatten to `[B,K*D]` | `contracts.py:241-312` | New `dynamics/trajectory_windows.py` as a **thin adapter over** `dynamics_batch_from_windows` — no duplicate window logic |
| I3 | `Encoder` requires a flattened K-step block, not a single frame | `encoder.py:99`, `:139-150` | Block builder must flatten `[K,D] → [K*D]` |
| I4 | `BlockSequenceDataset` uses non-overlapping blocks and has no action channel | `data/dataset.py:286`, `:297-301` | Do not reuse; new adjacent shifted-window dataset |
| I5 | `PendulumGenerator` is an analytic oscillator with `state_dim=1`, no control, no RK4, no (m,L,b) params | `data/pendulum.py:26-57` | New `data/controlled_pendulum.py` implementing `θ̇=ω`, `ω̇=−(g/L)sinθ − bω + u/(mL²)` with deterministic RK4 |
| I6 | `_GENERATORS` factory is hardcoded to 3 legacy generators; `SPLITS` names differ | `data/dataset.py:59-63`, `:66` | New generator/config/manifest outside the legacy factory; use `train/validation/frozen_test/parameter_ood_test` |
| I7 | `dynamics.evaluation.open_loop_rollout` denormalises **raw states** with `NormalizationStats` | `evaluation.py:65+` | JEPA multi-step rollout is in **latent** space → use `ConditionalPredictor.rollout` directly; do not force raw-state evaluation utilities |
| I8 | A2 `ConditionalPredictor` **rejects** `condition=None` | `conditional_predictor.py:102-108` | E1 (unconditioned JEPA) must inject legacy `Predictor(z)`, not pass `None`; E4 (zero action) is a distinct experiment |
| I9 | `prediction_loss` expects `Dict[int, Tensor]` keyed by horizon | `losses.py:27-50` | Single-step JEPA must wrap tensors into `{1: ...}`; multi-step passes `{1,2,5,10}` dicts |
| I10 | `ensure_output_dir` reuses the most recent run directory | `utils/config.py:98` | Adopt P2-02's `config_hash + experiment + seed` unique-directory policy (never overwrite) |
| I11 | Legacy `jepa` config schema is bound to `train.py` | `configs/pendulum.yaml:43-49`; `train.py:116-117` | New `configs/dynamics/controlled_pendulum_jepa.yaml` may reuse **key names** but must be consumed by the new script only |
| I12 | P2-02 action persistence is float32 while env received float64 | `collector.py:41-42`; `trajectory.py:25`, `:42-44` | New datasets must persist the **exact action passed to the simulator** (float64), see §6.3 |

---

## 5. Legacy checkpoint compatibility (read-only verification, executed)

```
outputs/pendulum/20260918_235258/checkpoint.pt
  sha256=fe7b0b412fdff1dc...  strict=True missing=[] unexpected=[]  prefixes=['encoder','predictor','target_encoder']  30 keys
outputs/double_pendulum/20260919_093324/checkpoint.pt
  sha256=d42db8e9551d122d...  strict=True missing=[] unexpected=[]  prefixes=['encoder','predictor','target_encoder']  30 keys
outputs/newton/20260919_201912/checkpoint.pt
  sha256=03098584ffd54267...  strict=True missing=[] unexpected=[]  prefixes=['encoder','predictor','target_encoder']  30 keys
```

Also verified: `WorldModel.named_children() == ['encoder','predictor','target_encoder']`,
`hasattr(wm, 'conditional_predictor') == False`, and neither A2's
`ConditionalPredictor` nor P2-02's `ResidualStatePredictor` state_dict keys
collide with the legacy keys.

**Implication:** as long as P2-03A adds an independent class and does not edit
`models/world_model.py`, `models/predictor.py` or `models/conditional_predictor.py`,
legacy checkpoints remain loadable. A regression test must lock this in P2.

---

## 6. The three P2-02 validation issues (identified; fixes proposed, NOT applied)

### 6.1 Strict integer validation for `hidden_dims`

- **Location:** `models/state_dynamics.py:72-83`, specifically `:78`
  `hidden = tuple(int(h) for h in hidden_dims)`.
- **Defect:** `int(3.7) → 3`, `int(True) → 1`, `int(np.float32(2.9)) → 2` — silently
  truncates, so a mistyped config produces a different architecture with no error.
- **Same pattern also at** `models/conditional_predictor.py:91` — **but §8 forbids
  modifying that file** (see §7).
- **Proposed fix (P0-fix, to be reviewed):** in `state_dynamics.py`, require
  `isinstance(h, (int, np.integer)) and not isinstance(h, bool)` for every entry;
  raise `TypeError` otherwise; keep `> 0` check. Add tests for float/bool/
  numpy-float rejection. **Do not touch the A2 file in this phase.**

### 6.2 `NormalizationStats` deserialization validation

- **Location:** `dynamics/normalization.py:95-111`.
- **Defect:** no checks at all. A crafted/edited `normalization.json` can inject
  mismatched lengths (`state_mean` vs `state_std`), zero/negative std (→ division
  by zero → Inf), NaN/Inf, `cond_mean`/`cond_std` out of sync (one `None`),
  `eps <= 0`, or `source_split="frozen_test"`.
- **Proposed fix (P0-fix, to be reviewed):** add schema validation in
  `from_dict`: required keys, 1-D equal-length float sequences, all finite,
  `state_std > 0`, `cond_*` both `None` or both finite equal-length,
  `eps > 0`, `source_split` restricted to `train`, `n_valid_transitions >= 0`.
  Add tests for each rejection path plus a round-trip test.

### 6.3 Exact preservation of environment-applied action dtype

- **Root cause (already proved in P2-02 Phase 0):** `collect_episode` generates
  float64 actions and calls `env.step(action)` (`collector.py:41-42`), but
  `RobotTransition` casts to `np.float32` on persistence
  (`trajectory.py:25`, `:42-44`). Replaying stored float32 actions diverges on
  146/30,000 steps (max 0.58) after ~6e-8 quantisation is amplified by 100-step
  nonlinear dynamics. P2-02 worked around it by rebuilding float64 actions from
  the deterministic policies (exact replay max error 0.0).
- **Constraint:** §8 forbids modifying existing datasets, so **`pusher_v1` must
  stay byte-identical**; do not "fix" it in place.
- **Proposed fix (new data pipelines only):** `datasets/controlled_pendulum_v1`
  and the future `pusher_v2` persist the exact action tensor handed to the
  simulator (float64) as a separate field (e.g. `action_exact`), record its dtype
  in the manifest, and make the audit script verify replay with that field
  directly — no policy reconstruction needed.

---

## 7. Constraint conflict that must be resolved by review

The task specification contains two instructions in tension:

1. §1: "Identify **and fix** the minor P2-02 validation issues".
2. §1: "**Do not modify existing production modules until the audit and
   implementation plan are reviewed**"; §8 forbids modifying
   `models/conditional_predictor.py`, `models/world_model.py`,
   `models/predictor.py`, `data/pendulum.py`, existing datasets and historical
   checkpoints; §10 says commit to the feature branch and do not auto-merge.

**Decision taken in this phase:** audit only. All three issues are identified with
exact locations and ready-to-apply fixes, but **zero production code was changed**.
This preserves the "review before implementation" gate. The P0-fix commit should
be the first implementation action AFTER approval, and must explicitly state
that the §8-banned A2 file is left untouched (recorded as a known residual risk).

---

## 8. Compatibility risks register

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | New model accidentally inserted into `WorldModel` → legacy ckpt keys change | High | Independent class + strict-loading regression test in P2 |
| R2 | Context/target windows misaligned by one step → action leaks future | High | Explicit index contract (see `P2_03_ACTION_JEPA.md`) + unit tests asserting `(x_t,u_t)→x_{t+1}` |
| R3 | Cross-episode window stitching → train/test leakage | High | Build on `dynamics_batch_from_windows` (guarantees episode isolation) |
| R4 | Latent collapse despite VICReg | Medium | Log per-dim variance/covariance; G1 gate; EMA target under `no_grad` |
| R5 | E1 unconditioned ablation implemented by passing `condition=None` to A2 | Medium | Inject legacy `Predictor(z)` instead (I8) |
| R6 | `prediction_loss` dict interface violated for single-step | Low | Wrap `{1: ...}`; dedicated test |
| R7 | Normalisation fitted on validation/test or parameter-OOD | High | Reuse `fit_normalization(train_batch)`; persist stats; assert `source_split == "train"` |
| R8 | Parameter OOD is only a seed change (the P2-01 mistake) | High | Physical parameters (m, L, b) MUST be written into the manifest and audited; A0 already proved seed-only "OOD" is not OOD |
| R9 | Linear Probe fitted on frozen test | High | Fit on TRAIN representations only; evaluate frozen/OOD without refit |
| R10 | Reusing `ensure_output_dir` overwrites a previous experiment | Medium | Use content-addressed `experiment_id`; never overwrite |
| R11 | CPU/GPU non-determinism across 5 seeds | Low | `set_seed`, record device, prefer CPU for the smoke run |
| R12 | Claiming "physical law discovery" from low latent loss | High | Explicitly forbidden by §10; final report must separate verified results / negative results / limitations |

---

## 9. Phase 0 deliverables and stop condition

Delivered (documents only, no code):

1. `P2_03A_ARCHITECTURE_AUDIT.md` — this file.
2. `P2_03A_MODULE_DEPENDENCY_MAP.md` — dependency graphs + reuse/adapter matrix.
3. `P2_03A_GAP_ANALYSIS.md` — requirement-by-requirement gap table.
4. `P2_03_ACTION_JEPA.md` — architecture specification + staged implementation plan + gates.

**STOP.** No production module was modified, no dataset generated, no model
trained, no merge to `main`. Await review of the module boundary and the
implementation plan before P0-fix / P1.

---

## 10. AMENDMENT — Phase 0.1 review + P0-fix (historical Phase 0 text above preserved)

The Phase 0 audit (§1–§9) remains valid as read-only evidence. This section
records only what changed afterwards; it does not rewrite the audit record.

### 10.1 Specification defect found during review (not caught in Phase 0)

A reviewer found that `P2_03_ACTION_JEPA.md` §2.2 wrote the multi-step target as
`flatten(obs[s+k : s+k+K+1])` — **K+1 frames**, which is incompatible with
`Encoder.forward`'s `[B, K*D]` contract (`encoder.py:139-150`, F2/I3 above).
Corrected to `flatten(obs[s+k : s+k+K])`. This is exactly the kind of
off-by-one that F2/R2 warned about, and it validates the decision to require
an explicit, unit-tested index contract.

Empirical verification performed during the amendment (synthetic `obs[60,2]`,
`cond[59,1]`, `K=10, s=3`): the corrected formula yields shape `(1,20)` for
all horizons k ∈ {1,2,5,10,20}, with k=1 elementwise identical to the
single-step target; the erroneous formula yields `(1,22)` for all k.

**Additional finding not in the review:** a valid multi-step window also needs
`s <= L - K - k_max`. With `K=10, L=60, k_max=20` the bound is `s <= 30`,
whereas the single-step bound `L-K-1 = 49` would let NumPy silently produce
short windows. P1 MUST use the `k_max` bound. This is now a required test.

### 10.2 Status of the three §6 validation issues after P0-fix

| §6 issue | Phase 0 | P0-fix (Commit B) |
|---|---|---|
| 6.1 `int(h)` truncation in `state_dynamics.py:78` | proposed | **FIXED** — `numbers.Integral` + explicit bool exclusion; accepts Python int and numpy integer scalars; rejects float (incl. 3.0/3.7), bool, `np.bool_`, str; zero new dependencies |
| 6.2 `NormalizationStats.from_dict` schema | proposed | **FIXED** — required fields, 1-D, equal lengths, finite values, strictly positive std, paired `cond_*`, positive finite eps, `source_split=="train"`, non-negative non-bool integer count; the already-persisted `outputs/dynamics/pusher_v1/normalization.json` was verified to still load |
| 6.3 exact action dtype | proposed | **NOT APPLIED (by design)** — `pusher_v1` / `RobotTransition` stay byte-identical; the requirement belongs to the NEW controlled-pendulum pipeline in P1 (persist the exact float64 action handed to the simulator) |

**Residual risk (unchanged):** `models/conditional_predictor.py:91` has the same
`int(h)` pattern but is on the §8 do-not-modify list. Q5 was closed as "no
change in P0-fix"; the residual risk remains explicitly documented.

### 10.3 Boundary invariants re-verified for Phase 0.1 / P0-fix

- No §8-banned file was modified (`world_model.py`, `predictor.py`,
  `conditional_predictor.py`, `train.py`, `utils/losses.py`, `data/pendulum.py`,
  `datasets/pusher_v1`, historical checkpoints).
- `main` was NOT merged; no PR was created; work stays on
  `feature/p2-03a-action-jepa`.
- No dataset, simulator, model or JEPA training code was implemented.
- The already-collected P2-02 evidence (`normalization.json`, 20 checkpoints,
  metrics) is untouched and remains loadable.
