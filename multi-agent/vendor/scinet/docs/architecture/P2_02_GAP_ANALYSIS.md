# P2_02_GAP_ANALYSIS — Gap between SciNet++ today and P2-02 requirements

**Phase 0 deliverable (audit only).** Companion: `ARCHITECTURE_AUDIT.md`,
`MODULE_DEPENDENCY_MAP.md`, `CORE_ADAPTER_BOUNDARY.md`.

This doc maps each P2-02 requirement (from the task brief) to what exists today,
what is missing, and the concrete change needed. It is the input to the **A1 gate**
("data-contract tests pass → allow model-interface changes").

---

## 0. P2-02 需求清单 (Requirement → status matrix)

| # | P2-02 requirement | Status | Where it lands |
|---|---|---|---|
| 1 | Generic `DynamicsBatch`: obs `[B,T+1,D]`, conditions `[B,T,U]`, valid_mask `[B,T]`, episode_ids `[B]` | **MISSING** | new `dynamics/` package |
| 2 | Support unconditioned **and** conditioned dynamics | **MISSING** | `conditions` optional (`U=0` for Pendulum) |
| 3 | Variable-length episodes | **MISSING** | `valid_mask` + per-episode packing |
| 4 | Never create transitions across episode boundaries | **MISSING** | loader must respect `episode_ids` |
| 5 | Guarantee `x_t` + `u_t` predicts `x_{t+1}` | **MISSING** | contract invariant in loader |
| 6 | No Pusher-specific assumptions in generic contract | **MISSING** (must enforce) | `D`,`U` runtime, not hardcoded |
| 7 | `ConditionalPredictor`: `z_next = f(z, condition)` | **MISSING** | new `models/conditional_predictor.py` |
| 8 | Configurable `latent_dim`, `condition_dim` | **MISSING** | constructor args |
| 9 | Sequence + batch inputs; validate dims | **MISSING** | forward overloads + asserts |
| 10 | Preserve existing `Predictor` interface | **OK** (keep separate) | do not modify `Predictor` |
| 11 | Preserve legacy checkpoint compatibility | **OK** (if separate) | separate checkpoint key |
| 12 | Do not rewrite JEPA training pipeline unnecessarily | **OK** | keep `train.py` untouched |
| 13 | Verify loading/inference with actual legacy checkpoints | **MISSING** (test) | new test loads a real `WorldModel` ckpt |
| 14 | Three baselines: Persistence / Action-MLP / Action-ablation | **MISSING** | new `dynamics/` training script |
| 15 | Evaluation: 1-step MSE/RMSE, multi-step rollout, per-dim, per-episode, action-sensitivity, val vs frozen | **MISSING** | new `dynamics/eval.py` |
| 16 | Training-only normalization | **MISSING** | new `dynamics/normalize.py` |
| 17 | Reproducibility manifest | **PARTIAL** | reuse `robotics/data/manifest` pattern |
| 18 | Updated docs + commit hashes + test results | **THIS PHASE** | audit docs delivered; code later |

---

## 1. Generic data contract — the central gap

> **A1 AMENDMENT (supersedes the `U=0` / `episode_ids: Tensor` lines below).**
> This A0 audit draft sketched `U=0 => unconditioned` and
> `episode_ids: Tensor [B] (long)`. Both were **rejected** during A1 to keep a
> single unambiguous semantics:
> - `conditions=None` is the **only** unconditioned representation; a `U=0`
>   (zero-width) condition tensor is rejected by `validate_dynamics_batch`.
>   An all-zero tensor with `U>=1` stays valid (control channel exists, zero
>   input) and is a different experiment from `None`.
> - `episode_ids` is a length-`B` **sequence of strings / integer IDs** (sample
>   identity metadata), never a `torch.Tensor`.
> The authoritative spec is `DYNAMICS_DATA_CONTRACT.md` §2; the implementation
> is `dynamics/contracts.py` + `dynamics/validation.py`. The sketch below is
> retained verbatim as the A0 audit record, not as the current contract.

**What exists:** `data/dataset.BlockSequenceDataset` shaped `(n, num_blocks,
block_dim)` + `labels`. It assumes a *fixed* `sequence_length`, *uniform* validity,
*no* actions, *no* episode grouping. It is fundamentally the wrong shape for
P2-02's `(observations, conditions, valid_mask, episode_ids)`.

**What to build (net-new, additive):**
```
dynamics/batch.py
  class DynamicsBatch:
      observations: Tensor [B, T+1, D]   # x_0..x_T
      conditions:   Tensor [B, T,   U]   # u_0..u_{T-1}  (U=0 => unconditioned)
      valid_mask:   Tensor [B, T]        # 1 if transition (x_t,u_t)->x_{t+1} is real
      episode_ids:  Tensor [B] (long)    # prevents cross-episode transitions
      # invariants enforced in __post_init__ / builder:
      #   valid_mask[b,t] == 0  at the last step of each episode
      #   no transition ever connects episode e's last step to episode e+1's first
```

**Loader from Pusher-v1 (reuses robotics, no core import):**
```
dynamics/loaders.py
  def pusher_to_dynamics(output_dir, split, *, max_len=None) -> DynamicsBatch:
      # uses robotics.data.dataset.load_split (path-based, no robotics import in core)
      # stacks obs/action/next_obs per episode, builds valid_mask by episode length
```

**Contract tests (required for A1):**
- `test_dynamics_batch.py`:
  - builds a batch with `U=0` (Pendulum-like) and `U=7` (Pusher) from the same
    code path → proves no Pusher hardcoding.
  - asserts `valid_mask` is 0 at every episode boundary; asserts no transition
    spans two `episode_ids`.
  - asserts `observations[:, t+1]` equals the stored `next_obs` for every
    `valid_mask==1` index (the `x_t,u_t → x_{t+1}` guarantee).
  - asserts variable-length episodes pad with `valid_mask==0` and are not trained on.

---

## 2. ConditionalPredictor — design & compatibility

**Existing `Predictor` (`models/predictor.py:39-86`):** `forward(z)` → `(B, latent_dim)`,
no condition. This is the legacy interface + checkpoint contract. **Do not change it.**

**New `ConditionalPredictor` (separate module, e.g. `models/conditional_predictor.py`):**
```
class ConditionalPredictor(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dims=(256,256), kind="mlp"):
        # latent_dim, condition_dim both configurable
    def forward(self, z, condition):           # z:[B,latent_dim], condition:[B,condition_dim]
        # supports condition:[B,U] (batch) and [T,U] (sequence, applied per-step)
        # validates shapes; raises on dim mismatch
    # optional: forward_sequence(z_seq, cond_seq) for autoregressive rollout
```

Compatibility requirements:
- It reuses `build_mlp` from `models/encoder.py` (no duplication).
- Its `state_dict` keys are `conditional_predictor.*` — distinct from
  `predictor.*`, so loading a legacy `WorldModel` checkpoint into `WorldModel`
  is unaffected.
- A new test (`test_conditional_predictor.py`) loads a **real legacy
  `WorldModel` checkpoint** (`checkpoint.pt` from a prior run, or a committed
  fixture) and asserts `WorldModel.from_config(cfg, state_dim).load_state_dict(
  ckpt["model_state"])` still succeeds. This is the A2 gate evidence
  ("old interface + old checkpoint compatible → allow new-model training").

Note on scope: the task lists both a *latent* `ConditionalPredictor`
(`z_next=f(z,condition)`) and *raw-state* baselines (`x_next = x + g(x,action)`).
These are **two tracks** sharing the `DynamicsBatch` contract:
- Latent track → `ConditionalPredictor` conditions the JEPA latent transition
  (future work: action-conditioned JEPA).
- Raw-state track → the three P2-02 baselines operate directly on `observations`.
Keep them separate to avoid confusing "conditioned latent dynamics" with
"conditioned raw-state dynamics". Both consume `DynamicsBatch`.

---

## 3. Three baseline experiments — what's missing

| Experiment | Formula | What exists | What to build |
|---|---|---|---|
| Persistence | `x_next = x` | trivial | eval only; no params |
| Action-conditioned MLP | `x_next = x + g(x, action)` | `build_mlp` exists; no training loop | `dynamics/models.py: ActionConditionedMLP`; `dynamics/train.py` |
| Action ablation | eval trained model with aligned vs shuffled actions; optional separate shuffled-control | nothing | `dynamics/eval.py` with action-sensitivity + shuffle harness |

Hyperparameters from the brief, all configurable (no hardcoding):
`hidden_dim=256`, `2 hidden layers`, `AdamW`, `lr=0.001`, `batch_size=256`,
`max_epochs=100`, `early_stopping_patience=10`. Pusher-v1 constants:
`observation_dim=23`, `action_dim=7`, `action_range=[-2,2]`,
`dataset_seed=20261110`, `model_seed=42`.

**Critical honesty constraint (from brief + user):** report failed experiments
honestly. If Action-MLP does **not** beat Persistence, do **not** proceed to JEPA
("If P2-02's Action-conditioned MLP shows no clear advantage over Persistence,
don't rush into JEPA").

---

## 4. Evaluation — what's missing vs core

Core `evaluate.py` reports **latent** MSE/cosine over horizons — useless for
raw-state dynamics. P2-02 needs a new `dynamics/eval.py`:
- one-step MSE/RMSE (`x̂_1 vs x_1`, `x̂_t vs x_t`)
- multi-step open-loop rollout error (`x̂_{t+k}` unrolled, vs ground truth
  `x_{t+k}`) — note Pusher episodes are 100 steps; open-loop drift will be large
  and must be reported, not hidden
- per-state-dimension error (23 dims) — plot/table
- per-episode error distribution (box/violin)
- action sensitivity (perturb `u` → measure `∂x_next/∂u` or shuffle ablation)
- validation vs frozen_test comparison (keep `ood_test` quarantined; do not call
  it OOD)

Use **training-only** normalization statistics (`dynamics/normalize.py`): fit
mean/std on `train` split, apply to all splits. Pattern exists in
`analyze.py:216-218` (train-stats normalization) — copy the discipline, apply to
raw obs/actions.

---

## 5. OOD 分割 — 明确结论 (OOD split — explicit conclusion)

**Not genuinely OOD.** See `ARCHITECTURE_AUDIT.md` §6 for the empirical proof.
Summary: `ood_test` is IID with the other splits (same policies, same env, only
reset-seed differs). Feature means are within ~6% across splits; the 0.94
"separability" is a 15-sample overfitting artifact. **Do not make OOD claims.**
Use `ood_test` only as a second held-out estimate; keep `frozen_test` quarantined
as immutable evidence; never select models on it.

If genuine OOD is later required, the fix is a **real** distribution shift
mechanism in `robotics/data/splits.py` plan_collection (e.g. altered object/goal
placement ranges, a held-out policy, perturbed physics), not a rename.

---

## 6. 关键数据限制 (Key dataset limitation — must be in every P2-02 report)

Pusher-v1 has **30,000 transitions but 0 fingertip↔object contacts**
(`ARCHITECTURE_AUDIT.md` §6, plus P2-01 report: contact is incidental
arm-link touching, fingertip-object = 0/30,000). Therefore:
- It can support **arm-kinematics** experiments (predicting joint/qpos evolution
  under actions).
- It **cannot** establish reliable **contact-rich object-pushing** dynamics.
- P2-02 must never claim "successful object pushing" without genuine contact +
  object-displacement evidence. Report arm-dynamics results as the real deliverable.

---

## 7. 验收门映射 (Gate mapping)

| Gate | Unlocked by | Evidence in this/next phase |
|---|---|---|
| **A0** architecture audit complete | this doc set | 4 audit docs delivered |
| **A1** data-contract tests pass | Task 2 (Phase 1) | `test_dynamics_batch.py` green; no Pusher dim hardcoded |
| **A2** old interface + old checkpoint compatible | Task 2 (Phase 2) | `test_conditional_predictor.py` loads legacy `WorldModel` ckpt unchanged |
| **A3** Pusher baseline experiments complete | Task 3 (Phase 3-4) | Persistence vs Action-MLP vs ablation metrics + curves |
| **A4** validation + frozen-test results complete | Task 3 (Phase 4) | eval report (val vs frozen); `ood_test` quarantined |
| **A5** contact-data problem has a plan | Task 3 + user | explicit statement that contact-rich dynamics are out of scope for v1 |

**Stop-after-each-phase rule (from brief):** Phase 0 done → report → wait for A0
review before any production code change. Subsequent phases follow the same
stop-and-report cadence.

---

## 8. 建议的 P2-02 文件布局 (Proposed additive layout — no core edits)

```
dynamics/                        # NEW package (generic, no robotics import)
  __init__.py
  batch.py                      # DynamicsBatch + builder + invariant checks
  loaders.py                    # pusher_to_dynamics() using robotics data path
  normalize.py                  # train-only Normalizer
  models.py                     # ActionConditionedMLP (Persistence is eval-only)
  train.py                      # 3-baseline training loop (config-driven)
  eval.py                       # 1-step / rollout / per-dim / per-episode / action-sens
  manifest.py                   # experiment reproducibility manifest (SHA-256 pattern)
models/
  conditional_predictor.py      # NEW, separate from predictor.py
scripts/
  run_pusher_dynamics.py        # CLI: train + eval the 3 baselines
tests/robotics/
  test_dynamics_batch.py        # contract tests (A1)
  test_conditional_predictor.py # + legacy checkpoint load (A2)
  test_pusher_training.py       # 3 baselines + eval (A3/A4)
configs/robotics/
  pusher_dynamics.yaml          # P2-02 config (obs_dim=23, action_dim=7, seeds, hparams)
```

This layout adds everything P2-02 needs while leaving `models/predictor.py`,
`data/dataset.py`, `train.py`, `evaluate.py`, `research_api/*` **untouched**,
preserving the boundary in `CORE_ADAPTER_BOUNDARY.md`.
