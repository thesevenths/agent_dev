# ARR_EMB_RSI_01 — Pilot Report

Experiment ID: `ARR_EMB_RSI_01` · Branch: `feature/arr-embodied-rsi-01`
Stage: **PILOT ONLY** — Full experiment remains NOT AUTHORIZED.
Artifact: `ARR_EMB_RSI_01_PILOT4/pilot_results.json` (SHA-256 recorded in that file)
Date: 2026-11-22

## 1. Executive Summary

The embodied world-model adapter executed **15/15 real SciNet++ runs** across three verified failure mechanisms (E01 capacity, E02 coverage, E03 dynamics shift) with five matched arms (Base, Candidate, Random, Shuffled, Oracle). Infrastructure failures: 0. Applied config rate: 100%. Evidence artifacts: all claims resolved with matching SHA-256 and claim hash.

Five independent verdicts:

| Layer | Verdict | Reason |
|---|---|---|
| Engineering | **PASS** | 15/15 real engine runs, config applied, artifacts hashed, budget and reproducibility recorded |
| Benchmark Integrity | **PASS** | GT sealed outside run root, zero leakage, evidence fully verified, benchmark frozen |
| Diagnosis | **FAIL** | Root-cause and intervention-key accuracy 1.00, precision/recall 1.00, but confidence ECE=0.30 > frozen 0.15 |
| Intervention | **PASS** | Registry verified, 100% applied, Oracle measurable effect confirmed |
| Causal Improvement | **INCONCLUSIVE** | Only 3 independent units; Candidate-vs-Random/Shuffled 95% CI includes 0; below preregistered power |

**Overall: not scientific PASS.** Full experiment must not start from this pilot.

## 2. Source Audit / Reused vs New Modules

Source audit: `ARR_EMB_RSI_01_SOURCE_AUDIT.md`.

Reused without modification: `SciNetAdapter`, `ExperimentSpec/Result`, engine settings, `evolution/statistics.py`, `harness_rsi/evaluation.py` primitives, P0/P0.1 as immutable reference only.

New modules: `research_runtime/arr_emb_rsi_01/` — `adapter.py`, `benchmark.py`, `integrity.py`, `diagnosis.py`, `evaluation.py`, `prereg.py`, `runner.py`, `cli.py`, `smoke_adapter.py`, `setup_benchmark.py`, tests.

No P0/P0.1 file was modified; no Runtime refactor.

## 3. Benchmark / Failure Injection

Four mechanisms were defined. Oracle sensitivity was tested with the real engine before pilot:

| ID | Mechanism | Injection | Oracle | Engine-verified effect |
|---|---|---|---|---|
| E01 | Insufficient model capacity | `model.predictor_hidden=[8,8]` | `[16,16]` | mse@1 0.0553→0.0239, cos@1 0.023→0.643 |
| E02 | Insufficient state coverage | `train_samples=16`, fixed hidden test | `train_samples=64` | mse@1 0.0285→0.0240 |
| E03 | Dynamics distribution shift | ID train, OOD `t_max=14` eval | `hidden=[16,16]` | OOD mse@1 0.0502→0.0142 |
| E04 | Long-horizon error accumulation | `rollout_steps=[1,2]` | `[1,2,3]` | **no measurable Oracle effect** → BLOCKED |

Seeds are repeated measurements, not independent units. E04 was excluded from the pilot under the STOP rule rather than lowering the criterion.

## 4. Ground Truth / Preregistration / Design

- GT sealed in `GT_SEALED/ground_truth.sealed.json`, outside agent-visible run directory.
- Preregistration frozen before pilot in `ARR_EMB_RSI_01_PILOT4/preregistration.json` (`full_experiment_authorized=false`), MME=0.02, alpha=0.05, min 6 independent units for causal PASS.
- Pilot reduction from 40 to 15 runs was justified in the preregistration: E04 blocked, engine cost, seeds non-independent.

## 5. Pilot Results

15 runs: 3 mechanisms × 5 arms × 1 seed. Every arm `ok=True`; every applied config verified; 0 infrastructure failures.

Diagnosis (n=3 cases per controller):

| Controller | Root cause | Top-2 | Intervention key | Evidence precision/recall | Confidence ECE | Abstention |
|---|---|---|---|---|---|---|
| Evidence | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 | **0.30** | 0 |
| Rule | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 | 0.30 | 0 |
| Random | 0.33 | 0.33 | 0.33 | 0 / 0 | 0.167 | 0 |
| Shuffled | 0.00 | 0.00 | 0.00 | 0 / 0 | 0.50 | 0 |
| Oracle | 1.00 | 1.00 | 1.00 | 0 / 0 | 0.10 | 0 |

Evidence precision/recall for Oracle is 0 because Oracle is intentionally evaluator-side and does not cite agent evidence.

Causal statistics (unit-level, metric = validation or OOD `latent_mse@1`, negative = candidate better):

| Contrast | n | Mean Δ | 95% CI |
|---|---|---|---|
| Candidate vs Base | 3 | −0.023949 | [−0.043249, −0.004650] |
| Candidate vs Oracle | 3 | 0.000000 | [0, 0] |
| Candidate vs Random | 3 | −0.003603 | [−0.010517, 0.003311] |
| Candidate vs Shuffled | 3 | −0.003603 | [−0.010517, 0.003311] |
| Oracle vs Base | 3 | −0.023949 | [−0.043249, −0.004650] |

Candidate matches Oracle exactly because the evidence-based intervention was correct for all three cases. Candidate improves over Base, but is **not distinguishable from Random/Shuffled** at n=3; the causal gate is therefore INCONCLUSIVE.

## 6. Evidence Audit / Failures / Limitations

- All 3 claims per case (result JSON, checkpoint, actual config) resolved and hash-verified: 9/9.
- Three orchestration defects were found and fixed during the pilot (vacuous PASS on zero cases, shuffle pool missing for first case, controller-to-arm mapping mismatch). Failed attempts `ARR_EMB_RSI_01_PILOT`, `_PILOT2`, `_PILOT3` are preserved, not deleted; final valid artifact is `ARR_EMB_RSI_01_PILOT4`.
- E04 is BLOCKED: Oracle cannot produce a measurable effect.
- Only 3 independent units: underpowered for scientific PASS; no full experiment authorized.
- Confidence is overconfident (ECE 0.30); Diagnosis FAIL despite 100% accuracy.
- Action-conditioned dynamics are not validated in this phase: pendulum data has no action channel.

## 7. Verdict and Recommendation

Do **not** claim ARR is an Autonomous Scientific Research System from this pilot. The observed result is:

- The benchmark and engineering plumbing are valid for E01–E03.
- Evidence-based diagnosis found the correct cause and intervention in 3/3 verified cases, but its confidence calibration failed the frozen gate.
- The controlled improvement over Random/Shuffled is not established with only 3 independent units.

Recommended next step after review: fix confidence calibration and add at least 3 more genuinely independent, Oracle-sensitive mechanisms before any full experiment. Do not proceed to Reacher or Visual JEPA.

## 8. Reproducibility / Environment

```bash
# audit and benchmark setup (already executed)
python -m research_runtime.arr_emb_rsi_01.setup_benchmark
# preregistration
python -m research_runtime.arr_emb_rsi_01.cli --stage prereg --out ARR_EMB_RSI_01_PILOT4
# pilot
python -m research_runtime.arr_emb_rsi_01.cli --stage pilot --out ARR_EMB_RSI_01_PILOT4 --seed 42
# offline re-evaluation (no engine rerun)
python -m research_runtime.arr_emb_rsi_01.cli --stage finalize --out ARR_EMB_RSI_01_PILOT4
```

Environment: agent runtime Python 3.13, SciNet++ branch `feature/p2-03a-action-jepa` HEAD `c6f66c3`, engine Python `D:\conda_envs\scinet_gpu\python.exe` (torch 2.5.1+cu121), concurrency=1.
Unit tests: 5/5 PASS.
Report SHA-256: recorded in `ARR_EMB_RSI_01_PILOT4/pilot_results.json`.
