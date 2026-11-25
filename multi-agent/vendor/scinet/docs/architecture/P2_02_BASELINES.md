# P2-02 — Raw-State Dynamics Baselines: Design & Protocol

> Status: infrastructure + protocol document. **Experimental numbers live in
> `P2_02_FINAL_REPORT.md`** and `outputs/dynamics/pusher_v1/evaluation/`.
> This document contains no results and makes no performance claims.

## 1. Research question

Given a physical state `x_t` and an external control `u_t`, can a model predict
`x_{t+1}`, and **does knowing the action add predictive value**?

```
x_hat(t+1) = x_t + g_theta(x_t, u_t)
```

Residual (delta) prediction is used because short-time physical state changes
are small, and it makes the comparison against the Persistence baseline exact.

This is deliberately **Track B (raw-state)** and is NOT the A2 latent predictor
`z_{t+1}=f(z_t,u_t)`, nor a JEPA world model. The two tracks stay separate
until P2-03 is justified by these experiments.

## 2. Boundary with the existing Core

| Module | Role | Modified by P2-02? |
|---|---|---|
| `models/predictor.py`, `models/world_model.py`, `models/encoder.py` | legacy unconditional latent JEPA | **No** |
| `models/conditional_predictor.py` | A2 conditional *latent* predictor | **No** |
| `dynamics/contracts.py`, `dynamics/validation.py` | A1 DynamicsBatch contract | **No** (only additive re-exports in `__init__.py`) |
| `data/pendulum.py`, `data/*` | physics data generation semantics | **No** |
| `datasets/pusher_v1/**` | frozen dataset + manifest | **No** (read-only) |
| `models/state_dynamics.py` | **new** raw-state baselines | new |
| `dynamics/normalization.py`, `metrics.py`, `evaluation.py` | **new** generic infra | new |
| `configs/dynamics/`, `scripts/train_dynamics.py`, `scripts/evaluate_dynamics.py` | **new** experiment layer | new |
| `scripts/audit_pusher_dynamics.py` | **new** Phase 0 STOP GATE | new |

No generic core module (`dynamics/`, `models/`) imports `robotics`. The only
`robotics` import is in the Phase 0 audit script, which uses the collection
*policy* classes to rebuild the original float64 actions for the exact-replay
verification; it lives in the `scripts/` experiment layer.

## 3. Data contract

`DynamicsBatch` (A1) rows are whole episodes:

| field | shape | meaning |
|---|---|---|
| `observations` | `[B, T+1, D]` | `x_0..x_T`, one contiguous episode per row |
| `conditions` | `[B, T, U]` | `u_0..u_{T-1}` |
| `valid_mask` | `[B, T]` | True for real transitions |
| `episode_ids` | `[B]` | string identifiers |

For Pusher-v1, `obs_seq = concat(obs, next_obs[-1:])` gives `T=100`, `D=23`,
`U=7`. Phase 0 proved `obs[t+1] == next_obs[t]` bitwise for all 30,000
transitions, so the final state is well defined and **no transition crosses an
episode boundary**. Splits are assigned at the episode level before any window
is built; rollout windows are generated strictly inside each episode.

`conditions=None` in A1 means "the system has no control variable". Pusher-v1 is
always conditioned. The B1/B0 baselines *deliberately ignore* that channel;
that is a model ablation, not a data-contract statement.

### Observation semantics (verified, not guessed)

From `gymnasium 1.2.1 / pusher_v5.PusherEnv._get_obs`:

| slice | meaning |
|---|---|
| `[0:7]` | `data.qpos[:7]` — 7 arm joint angles |
| `[7:14]` | `data.qvel[:7]` — 7 arm joint velocities |
| `[14:17]` | `body_com("tips_arm")` — fingertip xyz |
| `[17:20]` | `body_com("object")` — object xyz |
| `[20:23]` | `body_com("goal")` — goal xyz (fixed) |

`robot_kinematics = [0:17]` is the primary evaluation group. Phase 0 measured
`mean |delta|`: qvel 0.438, qpos 0.030, fingertip 0.0075, object 3e-05, goal
0.0 — i.e. **only arm kinematics actually moves**.

## 4. Normalisation (train-only)

Fitted on the TRAIN split only, persisted to `normalization.json`, and reused
unchanged for validation / frozen_test / ood_test:

```
x_norm     = (x     - state_mean) / state_std
u_norm     = (u     - cond_mean)  / cond_std
delta_norm = (x_next - x)        / state_std        <-- state_std, not delta_std
=>  x_norm_next = x_norm + delta_norm               (exact identity)
```

Using `state_std` for the increment is the key detail: a residual model, the
Persistence baseline, the ridge baseline and the skill score then all live in
one consistent coordinate system. Constant dimensions are clamped to `eps`.
Predictions are mapped back with `denormalize_state` **before** computing all
physical metrics.

## 5. Five baselines

| ID | Model | Training |
|---|---|---|
| B0 | `x_next = x` (Persistence) | none |
| B1 | `x + f_theta(x)` — `ResidualStatePredictor(condition_dim=None)` | AdamW, early stopping |
| B2 | `x + g_theta(x,u)` — `ResidualStatePredictor(condition_dim=U)` | AdamW, early stopping |
| B3 | `delta = W[x,u]+b` — `RidgeDynamics`, closed form | alpha grid selected on validation |
| B4 | B2 architecture trained with **shuffled** actions (`shuffle_conditions`) | AdamW, early stopping |

B4 is a negative control: only the *training* action↔state correspondence is
destroyed; validation and evaluation always use correctly aligned actions. If
B2 ≈ B4 ≈ B1, action conditioning contributed nothing.

All baselines share the same architecture capacity (where applicable), the same
train/validation protocol, the same normalisation, and the same evaluation
code path. B0 and B1 are fed no condition tensor at all.

## 6. Training protocol

- `hidden_dims [256, 256]`, AdamW `lr=1e-3`, `weight_decay=1e-4`,
  `batch_size=256`, `max_epochs=200`, `early_stopping_patience=20`.
- Masked MSE on the normalised next state: padded transitions never contribute.
- Validation is used **only** for early stopping / ridge alpha selection.
- `frozen_test` is quarantined: never used for training, early stopping or
  model selection; it appears only in final reporting.
- Model seeds `[11, 22, 33, 44, 55]` are independent of `dataset_seed=20261110`.
- Each `(baseline, seed)` gets a unique `experiment_id =
  <config_hash8>_<baseline>_seed<seed>`; existing experiments are never
  overwritten without `--force`.
- `repro.json` records Python / torch / CUDA / git commit / dataset version /
  config hash / seed / duration.

The training script refuses to run unless the Phase 0 audit for the exact
manifest reports `integrity_pass=True`.

## 7. Evaluation protocol

- **One step**: teacher-forced, every valid transition.
- **Multi step**: open-loop autoregressive rollout. Recorded (true) actions are
  used, but **predicted states are fed back**; ground-truth states are never
  re-injected. Horizons `[1, 2, 5, 10, 20]`.
- Rollouts start at **every valid in-episode position** (not just `t=0`), then
  are collapsed back to one value per episode. This makes horizon-1 open-loop
  Persistence equal to teacher-forced Persistence (verified) while preserving
  enough samples for episode-level statistics.
- Metrics (original units): MSE, RMSE, MAE, per-dimension RMSE,
  `Skill = 1 - MSE_model/MSE_persistence`, goal-distance error.
- Group metrics for `joint_qpos / joint_qvel / fingertip_com / object_com /
  goal_com / robot_kinematics`. Constant groups get `skill=None` (no division
  by a zero Persistence MSE).
- **Statistical significance**: 5 seeds, episode-level paired bootstrap
  (10,000 resamples, 95% CI), mean **and** median reported. Correlated
  timesteps are never treated as independent samples.
- Comparisons: B2 vs B0, B2 vs B1, B2 vs B3, B2 vs B4, B1 vs B0, B4 vs B1.

## 8. Acceptance gates

| Gate | Criterion |
|---|---|
| G0 | Phase 0 data integrity + temporal alignment pass |
| G1 | B2 one-step MSE < Persistence on `robot_kinematics` (frozen_test) |
| G2 | B2 one-step MSE < B1 (action conditioning value) |
| G3 | B2 one-step MSE < B4 shuffled-action control |
| G4 | Horizon 5/10 rollouts finite and skill > 0 vs Persistence |

Gates use mean improvements on `frozen_test`; paired bootstrap CIs are reported
as separate statistical support, never substituted for a claim. Failed gates are
reported honestly.

## 9. Scientific limitations (must not be violated in interpretation)

1. Pusher-v1 contains **0 / 30,000 fingertip-object contacts**. It can support
   arm-kinematics / action-conditioning experiments, but **cannot** establish
   contact-rich object-pushing dynamics. No result may be described as
   "successful pushing" or a pushing World Model.
2. `ood_test` differs from train only by reset seed, with no explicit
   distribution-shift mechanism. It is an additional held-out split, **not**
   evidence of OOD generalisation.
3. Object and goal state dimensions are essentially static, so aggregate
   23-dim metrics can look good while saying nothing about manipulation.
4. A low latent loss or a low one-step MSE does not imply planning/control
   ability; multi-step rollout and action sensitivity are reported separately.
5. If G2 fails (B2 not better than B1), the correct next step is to investigate
   action information, policy diversity, state representation and training
   objective — **not** to proceed to P2-03 JEPA.

## 10. Reproduction commands

```bash
python scripts/audit_pusher_dynamics.py \
    --dataset datasets/pusher_v1 --output outputs/dynamics/pusher_v1
python scripts/train_dynamics.py \
    --config configs/dynamics/pusher_v1_baselines.yaml
python scripts/evaluate_dynamics.py \
    --config configs/dynamics/pusher_v1_baselines.yaml
```
