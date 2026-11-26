# P2-03A — Gap Analysis (requirement-by-requirement)

> Phase 0 read-only document. Status legend:
> **DONE** = exists and verified in source; **PARTIAL** = exists but semantically
> incompatible; **MISSING** = must be newly built.
> Every DONE/PARTIAL row carries a `file:line` reference.

## 1. Phase 0 audit requirements (task §1)

| Requirement | Status | Evidence / gap | Landing point |
|---|---|---|---|
| Inspect Encoder, ConditionalPredictor, WorldModel, EMA, VICReg | **DONE** | `encoder.py:76-150`; `conditional_predictor.py:57-203`; `world_model.py:31-159`; `utils/losses.py:27-85`; `train.py:162-187` | this audit |
| Inspect DynamicsBatch and P2-02 pipelines | **DONE** | `contracts.py:49-312`; `normalization.py`; `metrics.py`; `evaluation.py`; `scripts/{train,evaluate}_dynamics.py` | this audit |
| Identify reusable components | **DONE** | §3 of audit / map §3 | — |
| Identify incompatible interfaces | **DONE** | I1–I12 in audit §4 | — |
| Document all new modules and dependencies | **DONE** | dependency map §2 | — |
| Verify legacy checkpoint compatibility | **DONE** | 3 ckpts, `strict=True`, `missing=[] unexpected=[]` (audit §5) | — |
| Identify + fix 3 P2-02 validation issues | **IDENTIFIED, NOT FIXED** | audit §6 / §7 — blocked by "review before implementation" + §8 ban on `conditional_predictor.py` | P0-fix after approval |
| Do not modify production modules before review | **DONE** | `git status` clean; only new docs | — |

## 2. Controlled Pendulum dataset (task §2)

| Requirement | Status | Evidence / gap | Landing point |
|---|---|---|---|
| State `x=[theta, omega]`, control `u=[torque]` | **MISSING** | existing `PendulumGenerator` is `state_dim=1`, analytic, no control (`data/pendulum.py:26-57`) | `data/controlled_pendulum.py` NEW |
| ODE `theta_dot=omega`, `omega_dot=-(g/L)sinθ - b·ω + u/(mL²)` | **MISSING** | no RK4 anywhere in `data/` | NEW simulator |
| Deterministic RK4, fixed dt | **MISSING** | — | NEW simulator + config |
| Piecewise-constant diverse action profiles | **MISSING** | — | NEW action generator |
| train 2000 / validation 300 / frozen_test 300 / parameter_ood 300 | **MISSING** | legacy `SPLITS=("train","validation","adapt","test")` (`data/dataset.py:66`) | NEW dataset pipeline |
| Train params m 0.8–1.2, L 0.8–1.2, b 0.05–0.15 | **MISSING** (and explicitly flagged as *initial design values* requiring numerical-stability verification) | — | config + stability check in P1 |
| Parameter OOD m/L 1.4–1.6, b 0.20–0.30 | **MISSING** | Pusher-v1's `ood_test` was seed-only; A0 proved it is NOT OOD | config + manifest |
| Separate episode ids / seeds / manifest hashes; OOD params in manifest | **PARTIAL (pattern reusable)** | `data/dataset.py:99-155`, `:162-217` content-addressed + SHA-256 + per-split RNG | NEW manifest |
| No test data for normalisation / model selection | **PARTIAL** | `fit_normalization(train_batch)` (`normalization.py:118+`) is train-only by signature | reuse; assert `source_split=="train"` |
| Adjacent shifted windows, not random windows | **PARTIAL** | `dynamics_batch_from_windows` gives in-episode windows (`contracts.py:241-312`) but no `(context, action, target)` triple / flattening | `dynamics/trajectory_windows.py` NEW |

## 3. Architecture (task §3)

| Requirement | Status | Evidence / gap | Landing point |
|---|---|---|---|
| Independent `ActionConditionedWorldModel` | **MISSING** | `WorldModel` is unconditional (`world_model.py:94`) and §8 forbids editing it | `models/action_conditioned_world_model.py` NEW |
| Online Encoder | **DONE (reuse)** | `Encoder` (`encoder.py:76-150`) | instantiate |
| ConditionalPredictor | **DONE (reuse)** | `conditional_predictor.py:57-203`, incl. `rollout` | instantiate, no edits |
| EMA target encoder | **PARTIAL (pattern)** | `world_model.py:67-69`, `:76-79`, `:128-134` | re-implement inside new class |
| VICReg regularisation | **DONE (reuse)** | `utils/losses.py:53-85` | call `vicreg_loss` |
| `sg(target)`, no grad through target | **PARTIAL** | `@torch.no_grad()` in `world_model.py:76` | reproduce in new class |
| Not inserted into legacy `WorldModel` | **DONE (design constraint)** | §8 | enforced by tests |
| Adjacent context/target windows; exact index documented + tested | **MISSING** | `BlockSequenceDataset` uses non-overlapping blocks (`data/dataset.py:286`) | `trajectory_windows.py` + test |

## 4. Training (task §4)

| Requirement | Status | Evidence / gap | Landing point |
|---|---|---|---|
| Single-step JEPA **first** | **MISSING** | legacy `train.py` is multi-step over non-overlapping blocks (`:116-125`) | `scripts/train_action_jepa.py` NEW |
| `loss = prediction_loss + λ·vicreg_loss` | **DONE (reuse)** | `train.py:173-179`; `losses.py:27-85` | reuse losses |
| EMA update after optimizer step | **DONE (pattern)** | `train.py:181-187` | new loop |
| Record momentum / VICReg coeffs / latent var / cov / pred loss / grad norms / seeds | **MISSING** | `train.py:190-218` logs some but no grad norms, no latent variance diagnostics in the new pipeline | NEW logger |
| Multi-step JEPA only after single-step passes; horizons `[1,2,5,10]` | **PARTIAL** | `ConditionalPredictor.rollout` already provides autoregressive `[B,H,L]` (`:148-197`) | use directly (do NOT use `dynamics.evaluation.open_loop_rollout`, which is raw-state) |
| Never use ground-truth latents in open-loop rollout | **DONE (reusable)** | `conditional_predictor.py:192-196` feeds `z` back | enforced by test |
| 5 seeds | **DONE (pattern)** | P2-02 config `seeds: [11,22,33,44,55]` | NEW config |

## 5. Required ablations E0–E5 (task §5)

| Exp | Status | Design decision needed |
|---|---|---|
| E0 Persistence | **MISSING** | Task says "no Encoder / no Action / lowest baseline". Needs an unambiguous definition: recommended **raw-state Persistence evaluated through the SAME Linear-Probe protocol** (so G4 has a comparable denominator), NOT a latent-persistence model with a random encoder. ⚠️ **Needs review confirmation.** |
| E1 Unconditioned JEPA | **PARTIAL** | A2's `ConditionalPredictor` **rejects** `condition=None` (`conditional_predictor.py:102-108`). Recommended: instantiate legacy `Predictor(z)` (allowed — instantiation ≠ modification). ⚠️ **Needs review confirmation.** |
| E2 Action-conditioned JEPA | **MISSING** | target model |
| E3 Shuffled-action | **DONE (pattern)** | `shuffle_conditions` in `models/state_dynamics.py` — generic, no domain constants; reuse with a fixed seed |
| E4 Zero-action | **MISSING** | distinct from E1: same conditional architecture, `condition = 0` tensor (NOT `None`) — preserves A1 semantics |
| E5 Multi-step | **PARTIAL** | reuse `ConditionalPredictor.rollout` |
| Comparable capacity / protocol across E1–E5 | **MISSING** | must be enforced in config (same latent_dim, hidden, epochs, batch, seeds) |

## 6. Evaluation (task §6)

| Requirement | Status | Evidence / gap | Landing point |
|---|---|---|---|
| Latent variance / covariance / collapse check | **MISSING** | `variance_loss` / `covariance_loss` exist (`losses.py:53-80`) but no diagnostic reporter | NEW evaluator |
| Latent prediction MSE | **DONE (reuse)** | `prediction_loss` (`losses.py:27-50`) | — |
| Horizon 1/2/5/10/20 | **PARTIAL** | `metrics.py` + `evaluation.py::_valid_rollout_windows` are raw-state; JEPA needs latent rollout | NEW evaluator (reuse metrics + CI) |
| Raw-state Linear Probe, fit on TRAIN only | **MISSING** | `models/probe.py` exists for concept labels but is bound to the legacy pipeline; a **new** train-only linear probe mapping `z → x` is required | NEW probe in evaluator |
| Raw-state RMSE / MAE / Persistence Skill | **DONE (reuse)** | `dynamics/metrics.py:51`, `:55`, `:67`, `:83` | reuse |
| Action-conditioned vs unconditioned difference | **MISSING** | — | NEW comparison |
| Physical generalisation by mass / length / damping | **MISSING** | — | NEW grouped reporting |
| Parameter OOD explicitly reported | **MISSING** | — | NEW grouped reporting |
| 95% episode-level paired bootstrap CI | **DONE (reuse)** | `metrics.py:134+` `paired_bootstrap_ci` | reuse |
| Report by physical parameter group | **MISSING** | — | NEW |

## 7. Acceptance gates (task §7)

| Gate | Status | Proposed operational criterion (to be reviewed) |
|---|---|---|
| G0 dataset integrity | **MISSING** | RK4 replay with exact persisted action reproduces `x[t+1]` (tolerance), manifest SHA-256 + parameter-range audit |
| G1 no latent collapse | **MISSING** | per-dim latent std above a declared threshold on frozen test; covariance off-diagonal bounded |
| G2 conditional > unconditioned JEPA | **MISSING** | E2 latent/raw-state error < E1 with paired CI excluding 0 |
| G3 correct > shuffled action | **MISSING** | E2 < E3 with paired CI excluding 0 |
| G4 Linear Probe > Persistence | **MISSING** | E2 probe RMSE < E0, skill > 0 |
| G5 multi-step rollout stable | **MISSING** | finite errors at horizons 1/2/5/10/20, no divergence |
| G6 Parameter OOD measured honestly | **MISSING** | report degradation magnitude; **NOT** a requirement for high OOD skill |

## 8. Architecture restrictions (task §8) — all currently satisfied

| Forbidden | Phase 0 state |
|---|---|
| `models/world_model.py`, `models/predictor.py`, `models/conditional_predictor.py` | unmodified |
| existing JEPA loss behaviour / `train.py` | unmodified |
| `data/pendulum.py` | unmodified |
| existing datasets (`pusher_v1`) / historical checkpoints | unmodified, read-only verified |
| robotics imports in generic Core | zero (AST-verified in P2-02; must re-verify in implementation) |
| Pusher-specific constants in generic models | zero |

## 9. Test requirements (task §9)

| Test | Status | Landing point |
|---|---|---|
| Context/target window alignment | MISSING | `tests/dynamics/test_trajectory_windows.py` |
| Action alignment `(x_t,u_t)→x_{t+1}` | MISSING | same |
| Single-step forward | MISSING | `tests/models/test_action_conditioned_world_model.py` |
| Multi-step rollout | MISSING | same |
| EMA update | MISSING | same |
| Target encoder gradient isolation | MISSING | same |
| VICReg anti-collapse | MISSING | same |
| Gradient propagation | MISSING | same |
| Checkpoint round-trip | MISSING | same |
| Deterministic reproducibility | MISSING | same |
| Legacy checkpoint compatibility | **PARTIAL** | A2 already has `test_legacy_world_model_checkpoint_loading`; add a new test for the new model's checkpoint |
| Train/test isolation | **PARTIAL** | A1 `test_episode_isolation.py` + P2-02 tests; add OOD-range disjointness test |
| Parameter OOD isolation | MISSING | new test |
| Full existing regression | **DONE (baseline)** | 210 passed + 22 subtests at `e4eccc0` |

## 10. Deliverables (task §10)

| Deliverable | Status |
|---|---|
| Controlled pendulum dataset + manifest | MISSING (P1) |
| Independent Action-conditioned JEPA model | MISSING (P2) |
| Training / evaluation pipelines | MISSING (P2/P4) |
| Baseline comparisons E0–E5 | MISSING |
| Multi-step rollout results | MISSING (P4) |
| Parameter OOD results | MISSING (P5) |
| Five-seed reproducibility records | MISSING |
| Tests | MISSING |
| Architecture documentation | **PARTIAL** — this audit + dependency map + design doc (`P2_03_ACTION_JEPA.md`) delivered; `P2_03A_FINAL_REPORT.md` later |
| Commit to feature branch first, **do NOT auto-merge main** | **DONE for Phase 0** — branch `feature/p2-03a-action-jepa` |

---

## 11. Open decisions requiring review BEFORE implementation

| # | Question | Why it matters |
|---|---|---|
| Q1 | Exact definition of **E0 Persistence** (raw-state baseline through the Linear-Probe protocol vs latent persistence) | Determines the denominator of G4 |
| Q2 | **E1** implemented by instantiating legacy `Predictor(z)` vs a new unconditioned predictor | Affects capacity comparability and §8 interpretation |
| Q3 | Physical parameter ranges (m/L/b, dt, episode length 1000) must pass a **numerical-stability check** before full collection | Task explicitly says these are initial design values, not validated results |
| Q4 | `latent_dim` 8 vs 16; `block_size` 10; first-round multi-step horizons `[1,2,5,10]` vs evaluation `[1,2,5,10,20]` | Determines config and compute budget |
| Q5 | Whether P0-fix should also touch `models/conditional_predictor.py:91` | §8 forbids it; currently recorded as a residual risk |
| Q6 | Should the new model reuse `predictor` as its state_dict child name (natural but collides with legacy naming) | Requires `model_class` field + two-way load tests (map §4) |
| Q7 | Push the feature branch to GitHub for review (without merging main)? | Task says commit to branch first; user historically reviews GitHub `main` |

**Phase 0 stops here.** No implementation, no training, no merge to `main`.

---

## 12. AMENDED (Phase 0.1) — decision log after review

The Phase 0 table above is preserved as historical evidence. Review decisions:

| # | Resolution | Evidence |
|---|---|---|
| **Q1** | **CLOSED.** E0 = `x_hat[t+h] = x[t]` in raw state, no Encoder / Predictor / Linear Probe / training; MSE+MAE on the identical evaluation episodes; `Skill = 1 - MSE_model/MSE_persistence` with the denominator reported and a near-zero-denominator guard. This is the G4 denominator. | `P2_03_ACTION_JEPA.md` §5.0 |
| **Q2** | **CLOSED.** E1 = instantiated legacy `Predictor(z)` (no edits). `condition=None` must NEVER be passed to A2 `ConditionalPredictor`. Encoder/hidden config identical to E2; **exact parameter counts and the capacity difference recorded**; E2/E3/E4 share an identical architecture and parameter count. | `P2_03_ACTION_JEPA.md` §3, §5.2 |
| **Q3** | **OPEN (scientific).** Pilot values `g=9.81, dt=0.02, episode_steps=1000, torque_bounds=[-1,1]` are recorded as unverified starting points; P1-A must run 20 pilot episodes and pass stability checks before full collection. | §4.4 of spec |
| **Q4** | **OPEN (scientific).** latent_dim 8/16, block_size 10, horizons remain first-round design values. | §4.4 / §7 of spec |
| **Q5** | **CLOSED for P0-fix.** `conditional_predictor.py:91` is on the explicit do-not-modify list; its `int(h)` truncation remains a **documented residual risk**, to be decided by a later explicit approval. | `P2_03A_P0_FIX_REPORT.md` |
| **Q6** | **OPEN (engineering).** `model_class` field + two-way checkpoint load tests remain required. | map §4 |
| **Q7** | **CLOSED.** The feature branch was pushed at Phase 0 (`144a300`), read back from `origin/feature/p2-03a-action-jepa` and reviewed; per the current task book, `main` is still NOT merged and no PR is created. | review §1 |

### 12.1 Requirement-status deltas caused by the review

| Item | Phase 0 status | Phase 0.1 status |
|---|---|---|
| Multi-step target window indexing | MISSING / incorrect `K+1` formula | **SPEC CORRECTED** to `[s+k:s+k+K]`, `[B,K*D]` asserted for k∈{1,2,5,10,20}, h=1 identity + `s<=L-K-k_max` bound added (numerically verified) |
| Parameter OOD design | single joint OOD group | **SPEC CORRECTED** to 4 labelled groups × 75 episodes, ranges + counts in manifest, per-group G6 reporting |
| G2 / G3 evaluation | latent loss based | **SPEC CORRECTED** to train-only Linear-Probe raw-state RMSE/MAE + validation-only probe tuning + paired episode-level bootstrap CI; latent loss demoted to diagnostic |
| Collapse criteria | "declared threshold" (unspecified) | **PRE-REGISTERED** in config (min std / collapsed fraction / cov offdiag), frozen before any test inspection |
| Physics configuration | partial | **COMPLETED** (g, dt, RK4, unwrapped theta, initial ranges, torque bounds, hold duration, profiles, distributions, stability gates) |
| Model dispatch | incomplete `isinstance(...)` | **REPLACED** by explicit `predictor_mode` with validated dispatch |
| P0-fix validation gaps | IDENTIFIED, NOT FIXED | **APPLIED (2 of 3)** in Commit B; 3rd (exact action dtype) belongs to the P1 new pipeline, intentionally not applied |

**Phase 0.1 / P0-fix stop here.** No dataset, simulator, model or training code
is implemented. Await explicit approval before P1.
