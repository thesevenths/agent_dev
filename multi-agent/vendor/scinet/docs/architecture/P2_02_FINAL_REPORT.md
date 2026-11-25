# P2-02 — Raw-State Dynamics Baselines: Final Report

> Generated from real executions on 2026-11-11. Every number below is read from
> `outputs/dynamics/pusher_v1/` produced by the commands in §2. No result is
> estimated, interpolated or fabricated.

## 1. Conclusion first

**G0–G4 all PASS on `frozen_test`, but the scientific claim is deliberately narrow.**

| Finding | Verdict |
|---|---|
| Action-conditioned MLP (B2) beats Persistence (B0) | ✅ MSE **0.024401** vs **0.195003**, skill **0.8749** |
| B2 beats Unconditioned MLP (B1) | ✅ **6.53×** lower MSE (0.024401 vs 0.159410) |
| B2 beats Linear/Ridge (B3) | ✅ 1.91× lower MSE (0.024401 vs 0.046535) |
| B2 beats Shuffled-Action control (B4) | ✅ 6.62× lower MSE (0.024401 vs 0.161415); B4 collapses back to B1 level |
| Open-loop rollout horizon 1→20 | ✅ stable, B2 RMSE 0.1561 → 0.2578, skill stays > 0.87 |
| All B2 comparisons statistically significant | ✅ episode-level paired bootstrap 95% CIs exclude zero |
| **Object pushing / contact dynamics learned** | ❌ **NO.** 0/30000 fingertip-object contacts; `object_com` skill is **negative** |

**What may be claimed:** SciNet++ now has a validated **action-conditioned arm-kinematics dynamics model** on Pusher-v1. Action information provides large, statistically significant, reproducible predictive value.

**What may NOT be claimed:** a pushing world model, contact-rich manipulation dynamics, OOD generalisation, or planning/control ability. See §10.

---

## 2. Exact commands executed

```bash
# Phase 0 (STOP GATE)
python scripts/audit_pusher_dynamics.py --dataset datasets/pusher_v1 \
    --output outputs/dynamics/pusher_v1
#   -> integrity_pass=True, verdict=PASS - proceed to P2-02 Phase 1

# Phase 3 (20 experiments: 4 baselines x 5 seeds)
python scripts/train_dynamics.py --config configs/dynamics/pusher_v1_baselines.yaml
#   -> TRAIN_EXIT=0, duration 8m20s, config_hash=c791729e1c5738f8...

# Phase 4/5
python scripts/evaluate_dynamics.py --config configs/dynamics/pusher_v1_baselines.yaml
#   -> EVAL_EXIT=0, duration 1m03s, gates G0..G4 = PASS

# Regression
python -m pytest -q --ignore=test_pysr.py     # 210 passed, 22 subtests passed
python test_data.py                            # All data smoke tests passed
```

Interpreter: `/d/python313/python` (Python 3.13.0). Logs: `training.log`, `evaluation.log`.

---

## 3. Reproducibility manifest

| Item | Value |
|---|---|
| Python / torch / CUDA | 3.13.0 / 2.14.1+cu126 / 12.6 |
| Actual device | **cpu** (small MLP, fully deterministic; recorded in `repro.json`) |
| numpy / gymnasium / mujoco | 2.2.6 / 1.2.1 / 3.3.7 |
| Dataset version | `pusher-8af70418c4fb` |
| manifest SHA-256 | `de69a06753bdb4d7954a0e40aa106337236905582cba905033d9f51020248d33` |
| Config hash | `c791729e1c5738f8fc9a969dcf2a2982bc84be8a1e298d07f376ace28d197ddf` |
| Git commit at training | `bafb8282063bd24caf7e7dcf1b798950fcfdc751` |
| dataset_seed / model seeds | `20261110` (data identity) / `[11, 22, 33, 44, 55]` — strictly separated |
| Hyperparameters | hidden `[256,256]`, AdamW lr `1e-3`, wd `1e-4`, batch `256`, epochs `200`, patience `20`, horizons `[1,2,5,10,20]` |
| Normalisation | fitted on **train only** (`source_split=train`, `n_valid_transitions=24000`), persisted to `normalization.json` |
| Splits | train 240 ep / validation 30 / frozen_test 15 / ood_test 15 (episode-level) |
| Bootstrap | 10,000 resamples, α=0.05, seed 20261110, unit = **episode** (paired) |

20 `repro.json` files (one per experiment) record seed, duration, best epoch, best validation MSE and the full environment/config identity.

---

## 4. Phase 0 — Dataset audit (G0)

| Check | Result | Evidence |
|---|---|---|
| Manifest + 300 per-file SHA-256 | PASS | 300 declared / 300 on disk, 0 mismatches, 0 undeclared, 0 missing |
| obs/action dimensions | PASS | 300 episodes, 30,000 transitions, D=23, U=7 |
| Action bounds | PASS | observed `[-2.0, 2.0]` = declared `[-2.0, 2.0]` |
| Episode ID uniqueness | PASS | 300 ids / 300 unique / 0 filename-meta mismatches |
| Split isolation | PASS | 0 episode-id overlap, 0 reset-seed overlap across splits |
| Temporal continuity `obs[t+1]==next_obs[t]` | PASS | **683,100** elements bitwise-identical, 0 mismatches |
| Action alignment (exact MuJoCo replay) | PASS | 300 episodes / 30,000 steps, **max_abs_err = 0.0** |

**Observation semantics are verified from source, not guessed**: `gymnasium 1.2.1 / pusher_v5.PusherEnv._get_obs` → `[0:7]` qpos, `[7:14]` qvel, `[14:17]` tips_arm COM, `[17:20]` object COM, `[20:23]` goal COM. The full source is embedded in `data_audit.md`.

**An honest methodological finding (recorded, not hidden):** replaying the stored **float32** actions directly diverges on 146/30,000 steps (max 0.580, worst `train_random_00019`). Root cause was proved empirically: the collector generated float64 actions but persisted float32 (quantisation ≈ 5.94e-08), amplified by 100-step nonlinear joint dynamics. Rebuilding the original float64 actions from the deterministic collection policies gives **0.0** error across all 300 episodes. This is a replay artefact, **not** a data defect: the supervised triple `(x_t, u_t_float32, x_{t+1})` comes from one `env.step` and is self-consistent. Both replay modes are stored in `data_audit.json`.

**Contact decomposition:** fingertip↔object **0 / 30,000**; other arm-link↔object 2,819 / 30,000 (116/300 episodes). State-change magnitudes confirm `object_com ≈ 3e-05`, `goal_com = 0.0`.

---

## 5. Five baselines — Frozen Test (primary results)

Mean ± std over 5 seeds. All metrics in **original physical units** (denormalised before scoring). `Skill = 1 − MSE_model / MSE_persistence`.

| ID | Model | MSE | RMSE | MAE | Skill |
|---|---|---|---|---|---|
| B0 | Persistence `x_next=x` | 0.195003 ± 0.000000 | 0.441592 | 0.143530 | 0.0000 |
| B1 | Unconditioned MLP | 0.159410 ± 0.001250 | 0.399258 | 0.130809 | 0.1825 |
| **B2** | **Action-conditioned MLP** | **0.024401 ± 0.001431** | **0.156141** | **0.030024** | **0.8749** |
| B3 | Linear / Ridge | 0.046535 ± 0.000000 | 0.215720 | 0.047327 | 0.7614 |
| B4 | Shuffled-action control | 0.161415 ± 0.001151 | 0.401763 | 0.131611 | 0.1722 |

- B0 is constant and B3 is a deterministic closed-form fit, so their 5-seed std is exactly 0 — expected, not a bug.
- On the primary group `robot_kinematics [0:17]`: B0 MSE 0.263828, B1 0.215672, **B2 0.033014 (skill 0.8749)**, B3 0.062960, B4 0.218385.
- Validation and `ood_test` give the same ordering (B2 MSE 0.019141–0.024401), so there is no split-specific overfitting.

### Cross-split consistency

| Split | B0 MSE | B1 | **B2** | B3 | B4 | B2 skill |
|---|---|---|---|---|---|---|
| validation | 0.192042 | 0.154554 | **0.019141** | 0.039477 | 0.157978 | 0.9003 |
| frozen_test | 0.195003 | 0.159410 | **0.024401** | 0.046535 | 0.161415 | 0.8749 |
| ood_test | 0.197815 | 0.158406 | **0.022025** | 0.047852 | 0.161842 | 0.8887 |

`ood_test` differs from train only by reset seed (A0 finding). Its near-identical numbers **confirm it is an ordinary held-out split, not evidence of OOD generalisation**.

---

## 6. Semantic feature groups (B2, frozen_test)

| Group | dims | MSE | Persistence MSE | Skill |
|---|---|---|---|---|
| joint_qpos | 0:7 | 0.00014099 | 0.00316010 | **0.9554** |
| joint_qvel | 7:14 | 0.08001798 | 0.63750821 | **0.8745** |
| fingertip_com | 14:17 | 0.00003896 | 0.00013268 | **0.7063** |
| **object_com** | 17:20 | 0.00000005 | 0.00000001 | **−2.24** ⚠️ |
| goal_com | 20:23 | 0.0 | 0.0 | `None` (constant) |
| robot_kinematics | 0:17 | 0.03301351 | 0.26382804 | **0.8749** |

⚠️ **Negative object skill is a real, reported result**: the object barely moves, so Persistence is already near-perfect on that group; the learned model is very slightly worse there. The overall win comes **entirely from arm kinematics**, not from modelling the object. `goal_com` is exactly constant → skill undefined (`None`), not infinity.

Goal-distance MAE = 0.000183 against a mean true object→goal distance of 0.2384 — i.e. essentially no object displacement to predict.

---

## 7. Multi-step open-loop rollout (frozen_test)

Predicted states are fed back autoregressively; only the **recorded** action sequence is used, never ground-truth states. Rollouts start from every valid in-episode position (1,215–1,500 windows depending on horizon), then collapse to episode-level aggregates.

| Horizon | B0 RMSE | B1 | **B2 RMSE** | B3 | B4 | **B2 skill** | windows |
|---|---|---|---|---|---|---|---|
| 1 | 0.4416 | 0.3993 | **0.1561** | 0.2157 | 0.4018 | **0.875** | 1500 |
| 2 | 0.5229 | 0.4528 | **0.1677** | 0.2499 | 0.4551 | **0.897** | 1485 |
| 5 | 0.6695 | 0.5361 | **0.1908** | 0.3010 | 0.5346 | **0.919** | 1440 |
| 10 | 0.8006 | 0.5957 | **0.2179** | 0.3374 | 0.5889 | **0.926** | 1365 |
| 20 | 0.9226 | 0.6481 | **0.2578** | 0.3649 | 0.6364 | **0.922** | 1215 |

**Interpretation caution**: B2 skill rising from 0.875 → 0.926 does **not** mean the model improves with longer rollouts. Its absolute RMSE grows **+65%** (0.1561 → 0.2578), i.e. error does accumulate; Persistence simply degrades faster (+109%, 0.4416 → 0.9226), so the *relative* ratio improves. Horizon-1 open-loop Persistence equals teacher-forced Persistence exactly (verified in the pipeline smoke test), confirming the metric is self-consistent.

---

## 8. Statistical significance (episode-level paired bootstrap)

`n_pairs = 15` frozen-test episodes, 10,000 resamples, 95% CI, seed 20261110. `mean_diff = MSE_a − MSE_b`; negative = `a` better.

| Comparison | mean_diff | median_diff | 95% CI | Significant? |
|---|---|---|---|---|
| B2 vs Persistence | −0.170602 | −0.195467 | **[−0.235467, −0.104204]** | ✅ B2 better |
| B2 vs B1 | −0.135008 | −0.155349 | **[−0.185407, −0.083120]** | ✅ B2 better |
| B2 vs B3 Ridge | −0.022134 | −0.010958 | **[−0.034918, −0.010430]** | ✅ B2 better |
| B2 vs B4 shuffled | −0.137014 | −0.160780 | **[−0.187964, −0.084210]** | ✅ B2 better |
| B1 vs Persistence | −0.035594 | −0.037672 | [−0.051225, −0.020814] | ✅ B1 better |
| B4 vs B1 | +0.002006 | +0.000176 | [+0.000212, +0.003968] | B1 slightly better |

All B2 comparisons are also significant at horizons 5 and 10. The B4-vs-B1 line is a **tiny** effect: one-step B1 is slightly better (CI just above zero), h5 is not significant, h10 slightly favours B4 — no strong claim is made from this secondary comparison. Correlated timesteps were never treated as independent samples.

---

## 9. Acceptance gates

| Gate | Criterion | Evidence | Verdict |
|---|---|---|---|
| G0 | Data integrity + temporal alignment | Phase 0 `integrity_pass=True`, 683,100/683,100 exact, exact replay max err 0 | **PASS** |
| G1 | B2 beats Persistence on `robot_kinematics` | 0.033014 vs 0.263828 (skill 0.8749) | **PASS** |
| G2 | B2 beats B1 (action value) | 0.024401 vs 0.159410; CI [−0.1854, −0.0831] | **PASS** |
| G3 | B2 beats B4 shuffled control | 0.024401 vs 0.161415; CI [−0.1880, −0.0842] | **PASS** |
| G4 | h5/h10 finite and skill > 0 | h5 skill 0.919, h10 0.926, RMSE 0.1908 / 0.2179 | **PASS** |

`overall_pass = True`. Gates use mean improvements on `frozen_test`; bootstrap CIs are reported as separate statistical support, never substituted for a claim.

---

## 10. Scientific interpretation and limitations (must be read with §1)

**Why B2 is so much better**: in Pusher-v5 the 7-D action is joint torque; the next joint-velocity change is largely determined by that torque plus damping. `Persistence` and `B1` have no access to `u_t`, so they can only predict a mean trend. `B2` recovers the torque→Δqvel mapping. The shuffled-action negative control collapsing to the B1 level confirms the gain comes from **correctly aligned action information**, not from extra capacity or state-only regularities. Ridge (B3) captures a large part of this (skill 0.761), so the nonlinear MLP adds a real but smaller increment (0.761 → 0.875).

**Hard limits — nothing here proves object pushing:**
1. **0 / 30,000 fingertip-object contacts.** Pusher-v1 cannot establish contact-rich pushing dynamics.
2. `object_com` skill is **negative**; the object essentially does not move (mean object→goal distance 0.2384, goal-distance MAE 0.000183). The model wins on arm kinematics only.
3. Results describe an **action-conditioned arm-kinematics dynamics model**, not a general pushing World Model.
4. `ood_test` is not a verified distribution shift; no OOD generalisation claim.
5. A low one-step MSE (and even a stable 20-step rollout) does **not** demonstrate planning or control ability; no MPC, closed-loop control or task-success metric was evaluated.
6. Rollouts follow recorded action sequences only; no counterfactual action sequences were tested.
7. `frozen_test` was quarantined: never used for training, early stopping or ridge-α selection.
8. `test_pysr.py` remains uncollectable due to a **pre-existing** `PYTHON_JULIACALL_PROJECT` path issue, unrelated to P2-02. It was excluded explicitly, not hidden.

---

## 11. Regression and architecture invariants

| Suite | Result |
|---|---|
| Full pytest (excluding pre-existing broken `test_pysr.py`) | **210 passed, 22 subtests passed** |
| A1 DynamicsBatch contract | 41 passed |
| A2 ConditionalPredictor | 29 passed |
| P2-02 new tests (state_dynamics / normalization / metrics) | 60 passed |
| Physics smoke (Pendulum / Double-Pendulum / Newton) | `All data smoke tests passed` |
| Robotics | 28 passed |
| JEPA / API | 52 passed, 22 subtests |

Invariants verified against `main` (`5f4b278`):
- 14 production files (`models/predictor.py`, `world_model.py`, `encoder.py`, `conditional_predictor.py`, `data/*`, `train.py`, `evaluate.py`, `analyze.py`, `tta.py`, `dynamics/contracts.py`, `validation.py`) — **all UNCHANGED**.
- AST check (docstrings stripped): **zero** `robotics` imports and zero Pusher/field constants in all 13 `models/` + `dynamics/` modules. Domain semantics enter only through `configs/dynamics/` and `scripts/`.
- `datasets/pusher_v1/`: 300 npz files, manifest SHA-256 unchanged (`de69a067…`), git status clean.
- Three 2026-09 legacy checkpoints unchanged.
- Training script loads only `train` and `validation` splits (verified by source inspection).

---

## 12. Deliverables

| Category | Files |
|---|---|
| Phase 0 audit | `scripts/audit_pusher_dynamics.py`, `outputs/dynamics/pusher_v1/data_audit.{json,md}` |
| Model | `models/state_dynamics.py`, `tests/models/test_state_dynamics.py` |
| Generic infra | `dynamics/{normalization,metrics,evaluation}.py`, `tests/dynamics/test_{normalization,metrics}.py`, `dynamics/__init__.py` (additive re-exports) |
| Experiment layer | `configs/dynamics/pusher_v1_baselines.yaml`, `scripts/{train_dynamics,evaluate_dynamics}.py` |
| Documentation | `docs/architecture/P2_02_BASELINES.md`, `docs/architecture/P2_02_FINAL_REPORT.md` (this file) |
| Training evidence | 20 × `checkpoint.pt` + `history.json` + `repro.json`, `normalization.json`, `training_manifest.json`, `training.log` |
| Evaluation evidence | `metrics.json`, `runs.json`, `bootstrap_results.json`, `gates.json`, `per_dimension_metrics.csv` (345 rows), `horizon_metrics.csv` (75), `episode_metrics.csv` (1500), 5 PNG plots, `evaluation.log` |

---

## 13. Commits and GitHub visibility

Dedicated branch `feature/p2-02-raw-state-dynamics`, small independently reviewable commits:

| # | SHA | Subject |
|---|---|---|
| 1 | `5716e21` | `feat(dynamics): add P2-02 Phase 0 Pusher-v1 dataset audit (STOP GATE)` |
| 2 | `9590011` | `feat(models): add raw-state residual dynamics baselines` |
| 3 | `08bb09a` | `feat(dynamics): add train-only normalisation, metrics and open-loop evaluation` |
| 4 | `bafb828` | `feat(dynamics): add P2-02 training and evaluation pipelines` |
| 5 | `fe55f80` | `experiment(dynamics): P2-02 five-seed baseline results and evidence` |
| 6 | (final report commit — see delivery notes) | `docs(dynamics): add P2-02 final report` |

> A commit cannot contain its own hash. The exact commit-6 SHA and the merged
> `origin/main` HEAD are reported in the delivery message and are verifiable
> with `git log --oneline origin/main -8`.

Verify on GitHub `main` after merge/push:

```bash
git ls-tree -r --name-only origin/main -- \
  SciNet++/outputs/dynamics/pusher_v1/evaluation/metrics.json \
  SciNet++/docs/architecture/P2_02_FINAL_REPORT.md
git log --oneline origin/main -8
```

---

## 14. Next experiments (proposed, NOT started)

Per the A5 gate, **P2-03 is not started automatically**. The action-conditioning question is answered positively on arm kinematics, but the contact question is not.

1. **P2-02 data augmentation (highest priority)**: build a separate `datasets/pusher_v2/` with a redesigned guided policy that produces genuine fingertip contact, multi-direction pushes and measurable object displacement. Keep `pusher_v1` untouched; never mix old and new data before splitting.
2. Re-run the same five baselines on contact-rich data with `object_com` / object-displacement / goal-distance as primary metrics. Only then can contact dynamics claims be evaluated.
3. Only after that is it meaningful to connect to A2 `ConditionalPredictor` / Action-conditioned JEPA (P2-03), and later MPC (P2-04).
4. Keep tracking `Semigroup-JEPA`-style generalisation across physical conditions, but only after basic dynamics are validated on the target regime.
