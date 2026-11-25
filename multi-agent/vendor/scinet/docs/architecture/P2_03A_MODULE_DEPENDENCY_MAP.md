# P2-03A — Module Dependency Map

> Companion to `P2_03A_ARCHITECTURE_AUDIT.md`. All arrows are derived from
> actual `import` statements / constructor calls in source (see file:line in
> the audit). Phase 0 read-only document.

## 1. Current three-layer architecture (verified)

```
LEGACY SCIENTIFIC CORE  (unchanged by P2-03A)
    data/pendulum.py ──► data/dataset.py ──► BlockSequenceDataset
    data/double_pendulum.py ──┘      │        (non-overlapping blocks, no actions)
    data/newton.py ──────────────────┘              │
                                                      ▼
    configs/pendulum.yaml ──► utils/config.py ──► train.py
                                                      │
             ┌────────────────────────────────────────┤
             ▼                                        ▼
    models/world_model.py                    utils/losses.py
        ├── models/encoder.py  (build_mlp)       prediction_loss
        ├── models/predictor.py                 vicreg_loss
        └── deepcopy → target_encoder (EMA)     variance_loss / covariance_loss
             │
             └──► outputs/*/checkpoint.pt  (encoder.* / predictor.* / target_encoder.*)

A1/A2 GENERIC INFRASTRUCTURE  (unchanged by P2-03A except additive modules)
    dynamics/contracts.py ──► dynamics/validation.py
    dynamics/normalization.py ──► dynamics/evaluation.py ──► dynamics/metrics.py
    models/conditional_predictor.py ──► models/encoder.py (build_mlp)
    models/state_dynamics.py ────────► models/encoder.py (build_mlp)

P2-02 EXPERIMENT LAYER
    configs/dynamics/pusher_v1_baselines.yaml
        ──► scripts/audit_pusher_dynamics.py ──(replay only)──► robotics.data.policies
        ──► scripts/train_dynamics.py  ──► dynamics.* + models.state_dynamics
        ──► scripts/evaluate_dynamics.py ──► dynamics.* + models.state_dynamics
    datasets/pusher_v1/**  (frozen, read-only)
```

**Boundary invariants re-verified in Phase 0:** `models/` and `dynamics/`
contain zero `import robotics` statements and, after stripping docstrings via
AST, zero Pusher/field constants. The only `robotics` import in the whole
P2-02 work sits in `scripts/audit_pusher_dynamics.py` (collection policies for
exact float64 replay).

## 2. Target P2-03A dependency graph (NEW nodes only)

```
datasets/controlled_pendulum_v1/**        (NEW, frozen, content-addressed manifest)
   ▲
   │ writes
scripts/collect_controlled_pendulum.py   (NEW)
   │ imports
   ├── data/controlled_pendulum.py      (NEW simulator: RK4, theta/omega/torque)
   └── dynamics/contracts.py            (REUSE: DynamicsBatch, episode isolation)

scripts/audit_controlled_pendulum.py     (NEW, STOP GATE)
   ├── data/controlled_pendulum.py     (replay with EXACT persisted action)
   └── datasets/.../manifest.json      (SHA-256 + physical parameter ranges)

dynamics/trajectory_windows.py          (NEW thin adapter)
   ├── dynamics/contracts.py            (REUSE: dynamics_batch_from_windows)
   └── models/encoder.py                (contract: flat [B, K*D] blocks)

models/action_conditioned_world_model.py (NEW, independent class)
   ├── models/encoder.py                (REUSE: Encoder, build_mlp)
   ├── models/conditional_predictor.py  (REUSE: forward/rollout — NO edits)
   ├── models/predictor.py              (REUSE by INSTANTIATION for E1 ablation — NO edits)
   └── copy.deepcopy / requires_grad_(False) / EMA pattern from world_model.py (PATTERN ONLY)

scripts/train_action_jepa.py            (NEW)
   ├── models/action_conditioned_world_model.py
   ├── utils/losses.py                 (REUSE: prediction_loss, vicreg_loss — NO edits)
   ├── utils/seed.py / utils/config.py (REUSE)
   ├── dynamics/trajectory_windows.py + dynamics/normalization.py (train-only)
   └── configs/dynamics/controlled_pendulum_jepa.yaml (NEW)

scripts/evaluate_action_jepa.py         (NEW)
   ├── dynamics/metrics.py             (REUSE: one_step_metrics, episode_errors,
   │                                    paired_bootstrap_ci, describe — NO edits)
   ├── dynamics/evaluation.py           (REUSE: _valid_rollout_windows PATTERN only)
   └── Linear Probe (fit on TRAIN representations only)

tests/models/test_action_conditioned_world_model.py   (NEW)
tests/dynamics/test_trajectory_windows.py           (NEW)
docs/architecture/P2_03_ACTION_JEPA.md             (NEW)
```

**No arrow** points from any NEW generic module (`models/`, `dynamics/`) to
`robotics/`, `datasets/pusher_v1`, or Pusher constants. The controlled pendulum
is a domain adapter feeding the generic contract, exactly like Robotics.

## 3. Reuse / adapter decision matrix

| Existing component | Reused? | How | Modified? |
|---|---|---|---|
| `Encoder` / `build_mlp` | ✅ direct | instantiate with `state_dim=2`, `block_size=K` | ❌ |
| `ConditionalPredictor` | ✅ direct | `forward` (1-step), `rollout` (multi-step) | ❌ forbidden by §8 |
| `Predictor` (legacy) | ✅ instantiate only | E1 unconditioned-JEPA ablation | ❌ forbidden by §8 |
| `WorldModel` | ⚠️ **pattern only** | EMA + `@no_grad` + momentum update re-implemented in the new class | ❌ forbidden by §8 |
| `utils/losses.py` | ✅ direct | `prediction_loss`, `vicreg_loss`, variance/covariance | ❌ forbidden by §8 |
| `ema_momentum` formula | ✅ reference | cosine warmup | ❌ |
| `train.py` JEPA loop | ⚠️ pattern only | new script, does not import or alter `train.py` | ❌ |
| `configs/pendulum.yaml` schema | ✅ key names only | new independent YAML | ❌ |
| `DynamicsBatch` / validators | ✅ direct | data contract | ❌ |
| `dynamics_batch_from_windows` | ✅ direct | underlying builder for `trajectory_windows.py` | ❌ |
| `BlockSequenceDataset` | ❌ | non-overlapping blocks + no actions → incompatible | ❌ |
| `data/pendulum.py` | ❌ | analytic oscillator, different physics → new simulator | ❌ forbidden by §8 |
| `_GENERATORS` factory / `build_dataset` | ❌ | hardcoded legacy generators → new dataset pipeline | ❌ |
| `dataset_version` / `load_frozen_split` | ✅ pattern | content-addressed manifest with SHA-256 | ❌ |
| `fit_normalization` / `NormalizationStats` | ✅ direct | train-only stats | ⚠️ schema fix proposed (§6.2) |
| `dynamics/metrics.py` | ✅ direct | metrics + paired bootstrap | ❌ |
| `dynamics/evaluation.py::open_loop_rollout` | ❌ | raw-state + denormalise → latent rollout uses A2 `rollout` | ❌ |
| `_valid_rollout_windows` | ✅ pattern | episode-internal multi-start rollout | ❌ |
| `set_seed` / `load_config` | ✅ direct | reproducibility / YAML | ❌ |
| `ensure_output_dir` | ❌ | reuses most recent run dir → unique `experiment_id` instead | ❌ |
| `datasets/pusher_v1/**` | ❌ | untouched, unrelated domain | ❌ forbidden by §8 |
| historical `checkpoint.pt` | ✅ read-only verify | strict-loading regression only | ❌ forbidden by §8 |

## 4. Checkpoint / state_dict boundary

```
LEGACY (must stay byte-compatible)
  WorldModel.state_dict() = { encoder.*, predictor.*, target_encoder.* }  (30 keys)

A2  ConditionalPredictor.state_dict() = { net.* }        (no collision)
P2-02 ResidualStatePredictor / RidgeDynamics = { net.* | W,b,fitted }  (no collision)

P2-03A  ActionConditionedWorldModel.state_dict()  (NEW, SEPARATE FILE)
        = { encoder.*, predictor.*, target_encoder.*, ... }
        ⚠️ SAME PREFIX NAMES as legacy but a DIFFERENT CLASS / DIFFERENT CHECKPOINT.
```

**Risk R1:** reusing the same submodule names is fine (it is a different
checkpoint file), but a regression test MUST assert that legacy checkpoints still
load into `WorldModel` with `strict=True`, and that the new model's checkpoint
does **not** accidentally get loaded into `WorldModel`. Document both directions.

Recommended new-model key naming to make confusion impossible:
`encoder.*`, `predictor.*`, `target_encoder.*` are natural, so additionally
store a `model_class = "ActionConditionedWorldModel"` field in every new
checkpoint and verify it on load.

## 5. Split / manifest dependency

```
controlled_pendulum_jepa.yaml
   ├─ physics ranges (mass/length/damping) for train + parameter_ood
   ├─ integrator (RK4), dt, episode length, action profile
   ├─ dataset_seed (data identity) ≠ model seeds
   └─► dataset_version = sha256(canonical spec)[:12]
           └─► datasets/controlled_pendulum_v1/manifest.json
                  ├─ split_episode_ids
                  ├─ split_file_hashes (SHA-256)
                  ├─ physical parameter ranges PER SPLIT  ← G6 evidence
                  ├─ action dtype (float64, exact)
                  └─ integrator / g / dt / version
```

**R8 guard:** `parameter_ood_test` must differ in `(m, L, b)` ranges, not merely
in seed. The manifest is the evidence; the audit script must assert the ranges
are disjoint from train (A0 proved that Pusher-v1's seed-only `ood_test` was NOT
out-of-distribution).

## 6. Boundary rules (carry over from A0, unchanged)

1. Generic `models/` / `dynamics/` never import `robotics` or a domain dataset.
2. Domain semantics (state layout, physical ranges, action bounds) live in
   `configs/` and `scripts/`, passed in as data — never hardcoded in Core.
3. Legacy `WorldModel` / `Predictor` / JEPA training objective are never edited.
4. New models are independent classes with their own checkpoints.
5. Frozen / OOD test data is never used for normalisation, early stopping,
   hyper-parameter selection or Linear-Probe fitting.
6. Every new experiment gets a unique `experiment_id` (config hash + variant +
   seed); nothing is silently overwritten.

## 7. AMENDED (Phase 0.1) — config / manifest dependency changes

The graph in §1–§5 is preserved as historical evidence. Review amendments add
the following required nodes/edges (no new code, no new cross-domain import):

```
configs/dynamics/controlled_pendulum_jepa.yaml   (NEW, P1 — not yet implemented)
   ├── physics: g, dt, integrator=RK4, episode_steps, unwrapped theta,
   │            theta0/omega0 ranges, torque_bounds, action_hold_steps,
   │            action_profiles, mass/length/damping distributions
   ├── collapse: min_latent_std, max_collapsed_fraction,
   │             max_cov_offdiag, require_finite       ← PRE-REGISTERED (G1)
   ├── evaluation: min_denominator for Persistence skill (E0 near-zero guard)
   ├── probe: train-only fit, VALIDATION-only hyper-param selection
   └── predictor_mode: "conditional" | "unconditioned"  ← replaces isinstance(...)

datasets/controlled_pendulum_v1/manifest.json     (NEW, P1 — not yet implemented)
   ├── split_episode_ids
   ├── split_file_hashes (SHA-256)
   ├── per-episode (m, L, b) values
   ├── parameter_ood_groups: {mass_ood:75, length_ood:75,
   │                          damping_ood:75, joint_ood:75}  ← review item 3
   ├── train/param-OOD physical ranges (asserted disjoint by audit script)
   ├── exact action dtype (float64, action passed to the simulator)
   └── integrator / g / dt / version
```

**Boundary impact: none.** All physics semantics still live in `configs/`,
`scripts/` and the domain adapter; generic `models/` and `dynamics/` gain no
new import of a domain module, and no arrow points to `robotics/` or
`datasets/pusher_v1`. The only production files touched in P0-fix are
`models/state_dynamics.py` and `dynamics/normalization.py` (validation
hardening only); their dependency edges are unchanged.
