# P2-03A — Phase 0.1 Review Amendments + P0-fix Report

> **Stop condition honoured.** This is the terminal deliverable of the
> "Phase 0.1 + P0-fix" task. No controlled-pendulum simulator, no dataset
> generation (not even the 20 pilot episodes), no JEPA model, no training, no
> MPC, no robotics changes, no merge to `main`, and no PR. Work stays on
> `feature/p2-03a-action-jepa`; `origin/main` remains `e4eccc0`.
>
> All numbers below come from commands actually executed in this session. The
> baseline was **measured**, not assumed.

## 1. Documentation corrections (Commit A)

Four existing Phase 0 audit documents were amended **without deleting historical
audit evidence**; corrections are marked `AMENDED (Phase 0.1)` inline, and the
full item-by-item change log (with the original incorrect text quoted verbatim)
is `P2_03A_REVIEW_AMENDMENT.md`.

| # | Review item | Correction |
|---|---|---|
| 1 | multi-step window K+1 frames | `obs[s+k : s+k+K+1]` → `obs[s+k : s+k+K]`, always `[B, K*D]`; added `[B,K*D]` assertions for k∈{1,2,5,10,20}, the h=1 identity with the single-step target, and a numerically verified table |
| 2 | E0 undefined | locked to `x_hat[t+h]=x[t]` in raw state, no encoder/predictor/probe/training, MSE+MAE on identical episodes, `Skill=1-MSE_model/MSE_persistence` with explicit denominator + near-zero guard |
| 3 | E1/E2 capacity | E1 = instantiated legacy `Predictor(z)` (never `condition=None`); identical Encoder/hidden/protocol; exact parameter counts recorded; E2/E3/E4 architecturally identical and equal parameter count |
| 4 | OOD changes 3 params at once | 4 disjoint labelled groups × 75 episodes (`mass`/`length`/`damping`/`joint`), ranges + counts in manifest, per-group G6 |
| 5 | G2/G3 on latent loss | primary = train-only Linear-Probe **raw-state RMSE/MAE** + validation-only probe tuning + paired episode-level bootstrap CI; latent loss demoted to diagnostic |
| 6 | collapse criteria | pre-registered in config (min latent std 0.1, collapsed fraction 0.25, cov offdiag 0.1), frozen before test inspection |
| 7 | physics config | completed (g, dt, RK4, unwrapped theta, initial ranges, torque bounds, hold duration, profiles, distributions) + mandatory 20-pilot-episode stability gates |
| 8 | incomplete `isinstance(...)` | replaced by explicit `predictor_mode` ∈ {`conditional`, `unconditioned`} with validated dispatch |
| 9 | Q7 stale | closed; Q1/Q2/Q5 closed with explicit decisions, Q3/Q4/Q6 remain open |

**Additional specification defect found during the amendment (not in the
review):** a valid multi-step window needs `s <= L - K - k_max`. With
`K=10, L=60, k_max=20` the bound is 30, not the single-step 49; otherwise
horizon-20 windows are silently short. Verified by real NumPy slicing
(`(1,20)` correct vs `(1,22)` erroneous for every k), now a required P2 test.

Commit A diff: **5 files, 575 insertions, 38 deletions** — documentation only,
zero production lines.

## 2. Production changes (Commit B)

Exactly two production files, validation-only:

| File | Change |
|---|---|
| `models/state_dynamics.py` | `_check_hidden_dims` now uses stdlib `numbers.Integral` + explicit `bool` exclusion. Accepts Python `int` and numpy integer scalars (`np.int64/int32/uint8`, verified). Rejects floats **including `3.0`**, Python `bool`, `np.bool_`, strings, `None`, nested/invalid entries, and non-positive values. **Zero new dependencies.** Docstring records why `int(h)` was unsafe (`3.7→3`, `True→1`). |
| `dynamics/normalization.py` | `NormalizationStats.from_dict` gained full schema validation: required 7 fields, non-mapping rejection, 1-D vectors, equal state/cond lengths, finite values only, strictly positive `state_std` and `cond_std`, `cond_mean`/`cond_std` paired (both `None` or both present), positive finite non-bool `eps`, `source_split` restricted to the train-only contract (`train`), non-negative non-bool integer `n_valid_transitions`. A `_validate_vector` helper rejects bool/nested elements **before** `torch.as_tensor` silently casts them. `to_dict`/`save`/`load`/`fit_normalization` semantics unchanged. |

**Explicitly NOT changed** (§3.D ban, verified via `git diff main`):
`models/world_model.py`, `models/predictor.py`, `models/conditional_predictor.py`,
`train.py`, `utils/losses.py`, `data/pendulum.py`, `data/dataset.py`,
`dynamics/contracts.py`, `dynamics/metrics.py`, `dynamics/evaluation.py`,
`configs/pendulum.yaml`, `datasets/pusher_v1/**`, and all historical
checkpoints. All 11 checked files reported `UNCHANGED`.

**Action persistence (review item C):** intentionally NOT applied. `pusher_v1`
and `RobotTransition` stay byte-identical; the exact float64 action requirement
belongs to the NEW controlled-pendulum pipeline in P1, as the task book states.

## 3. Tests actually executed

New file `tests/dynamics/test_p0_fix_validation.py` (29 tests, 74 subtests)
covering every §4 requirement:

- integer validation (accept int/np.integer; reject float incl. 3.0, bool,
  `np.bool_`, str, None, invalid sequence entries, non-positive, mixed entries,
  constructor routing, architecture-unchanged regression)
- normalization deserialization (valid minimal/conditioned, non-mapping,
  all 7 missing-field cases, NaN/Inf, zero/negative std, 2-D/scalar/empty
  vectors, bool elements, state length mismatch, cond half-pair, cond length
  mismatch, bad eps, invalid source_split, bad `n_valid_transitions`, numpy
  integer count, in-memory + disk roundtrip, unconditioned roundtrip)
- **existing behaviour unchanged**: the already-persisted P2-02
  `normalization.json` loads through both `from_dict` and `load()`; train-only
  fitting roundtrips; **3 real 2026-09 historical checkpoints load with
  `strict=True`, `missing=[] unexpected=[]`**; AST-level no-domain-import guard
  for the two hardened modules.

Commands (all executed, verbatim):

```bash
# baseline, measured BEFORE any change, at 144a300
python -m pytest -q --ignore=test_pysr.py
# new P0-fix tests
python -m pytest -q tests/dynamics/test_p0_fix_validation.py
# full regression AFTER the fix
python -m pytest -q --ignore=test_pysr.py
# grouped regression
python -m pytest -q tests/dynamics/test_contracts.py tests/dynamics/test_validation.py tests/dynamics/test_episode_isolation.py
python -m pytest -q tests/models/test_conditional_predictor.py
python -m pytest -q tests/dynamics/test_normalization.py tests/dynamics/test_metrics.py tests/models/test_state_dynamics.py
python test_data.py
python -m pytest -q tests/robotics
python -m pytest -q tests/api_tests
python -m pytest -q test_pysr.py    # confirms the pre-existing env failure
```

## 4. Tests passed and failed

| Suite | Before (measured baseline) | After |
|---|---|---|
| **Full pytest** (`--ignore=test_pysr.py`) | **210 passed, 22 subtests** (119.23s) | **239 passed, 96 subtests** (110.24s) |
| Delta | — | **+29 passed, +74 subtests** — exactly the new test file |
| A1 generic contract | 41 passed | 41 passed |
| A2 ConditionalPredictor | 29 passed | 29 passed |
| P2-02 existing tests | 60 passed | 60 passed |
| P0-fix new tests | n/a | **29 passed, 74 subtests** |
| Physics smoke (Pendulum/Double/Newton) | passed | `All data smoke tests passed` |
| Robotics | 28 passed | 28 passed |
| JEPA / API | 52 passed, 22 subtests | 52 passed, 22 subtests |

Grouped sum `41+29+60+29+28+52 = 239` and `74+22 = 96` — consistent with the
full-suite numbers. **0 failures, 0 errors in the full regression.**

**Honest exception:** `test_pysr.py` fails at collection with
`ValueError: PYTHON_JULIACALL_PROJECT=D:\ProgramData\anaconda3\julia_env: path
does not exist`. `git diff main -- test_pysr.py` is empty, so this is a
**pre-existing environment defect**, unrelated to this phase; it was excluded
from the full suite, not hidden. During this phase, no test needed a fix — the
29 new tests passed on the first run.

**Compatibility evidence beyond tests:** the persisted P2-02
`normalization.json` still loads (23 state dims, 7 cond dims, min `state_std`
`9.99999993922529e-09` > 0, `source_split=train`, `n=24000`); applying
`denormalize_state(normalize_state(x))` to a real frozen-test npz gives max
error `4.3546606320887804e-08` (float32 round-off) and all finite values.
P2-02 evidence hashes unchanged: manifest
`de69a06753bdb4d7954a0e40aa106337236905582cba905033d9f51020248d33`,
normalization.json
`d5629595d90e745e6edde398980a722553edc486ba55fc561c4998e1a515d7fc`,
metrics.json `49abc173aba1e308354cb997f557448b0018f0d3806040b40118cd0c29b0e003`;
300 npz files; 0 historical-checkpoint changes.

## 5. Remaining risks / unresolved items

1. **Residual risk (unchanged, review item D):** `models/conditional_predictor.py:91`
   still uses `int(h)` and therefore still silently truncates float/bool
   hidden_dims. It is on the §3.D do-not-modify list; Q5 was closed as "no
   change in P0-fix". Fixing it needs a separate explicit approval.
2. **`_check_positive_int`** in `state_dynamics.py` still uses plain
   `isinstance(value, int)` (it already rejects float/bool, but does not accept
   numpy integer scalars). Review item A targeted only `hidden_dims`; left as-is
   deliberately to keep the change narrow. Recorded for a future decision.
3. **Q3 open (scientific):** physics pilot values (`g=9.81`, `dt=0.02`, 1000
   steps, torque `[-1,1]`, parameter ranges) are unverified starting points;
   P1-A must pass 20 pilot episodes and stability checks before collection.
4. **Q4 open:** `latent_dim` 8/16, `block_size` 10, horizon sets.
5. **Q6 open:** new-model child naming requires `model_class` + two-way
   checkpoint load tests.
6. **Action dtype:** exact float64 action persistence is deferred to the P1
   new dataset pipeline; `pusher_v1` remains float32 (P2-02 exact replay relied
   on deterministic policy reconstruction).
7. **Pusher-v1 scientific limit unchanged:** 0/30000 fingertip-object contacts;
   no contact-rich pushing claim. `test_pysr.py` environment still broken.
8. **No CI record:** GitHub Check Runs are not configured; all results above
   are local executions, independently reproducible with the listed commands.

## 6. Exact Git commit SHAs

| Commit | SHA | Message | Content |
|---|---|---|---|
| base | `e4eccc0b1da2c8a365706f97ded95f3723e0f5e9` | `docs(dynamics): add P2-02 final report` | `origin/main` (unchanged) |
| Phase 0 | `144a300546638d54e6a08e0142a027af1812a720` | `docs(architecture): P2-03A Phase 0 architecture audit (read-only)` | 4 audit docs |
| **Commit A** | `d3382bb78e228c807d13ac425387a4469b46b04a` | `docs(p2-03a): clarify experimental protocol` | 5 docs, 575+/38-, zero production code |
| **Commit B** | see the final delivery message (this report is inside Commit B, so its own hash cannot be self-referenced) | `fix(core): harden dynamics validation` | 2 production files + 1 test file + this report |

Both commits are pushed to `origin/feature/p2-03a-action-jepa`; `origin/main`
is **not** merged and no PR was created. Working tree is clean after Commit B.

---

**STOP.** Awaiting explicit review approval before P1 (20 pilot episodes →
full controlled-pendulum dataset).
