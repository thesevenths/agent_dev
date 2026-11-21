# ARR-RSI-01 P0.1 — Experimental Integrity Repair: Final Report

Experiment ID: `ARR-RSI-01-P0.1` · Branch: `feature/arr-0.1-domain-registry` · Date: 2026-11-21

## 1. Original P0 verdict (preserved, untouched)

- Original experiment: `ARR-RSI-01-P0`, 20 tasks × 4 arms, concurrency 6.
- Original verdict: **BLOCKED**.
- Original artifacts remain in `ARR_RSI_01_RUN_FULL/` and the root P0 artifacts; no result was overwritten or retroactively upgraded.
- P0 exposed: benchmark information leakage, label-driven diagnosis, unverified intervention application, invalid shuffled control, seed-replicate overcounting, weak evidence checks, and infrastructure-failure bias.

## 2. P0.1 changes (minimum, inside `research_runtime/arr_rsi_01/`)

| File | Repair |
|---|---|
| `tasks.py` | Agent view reduced to opaque id, neutral objective, train/validation refs, metric/budget, allowed config schema. Removed family, failure mode, candidate/placebo labels and values, hidden-test ref, engine config, seed. 20 executions declared as 5 independent units × 4 seed replicates. |
| `diagnosis.py` | Removed task-label→root-cause mapping. Causes inferred from horizon ratio, cosine similarity, validation/OOD gap, metric availability and failure rules. Insufficient evidence → explicit `insufficient_evidence`, no intervention. Added `diagnosis_id` and evidence basis. Shuffled control uses another task's real diagnosis or a synthetic distinct knob. |
| `arms.py` | Records diagnosis id, evidence refs, hypothesis, expected effect, falsification, before/after/applied config, diff hash, and status (`INERT`/`APPLIED`/`NOT_APPLIED`). Placebo selected generically from the schema with range-safe values. |
| `runner.py` | Forced concurrency=1; run-scoped IDs; 180 s uniform CLI timeout; resume reuses only complete arms and archives failed attempts; applied config read from the actual engine campaign directory; full traceability bound into evaluation. |
| `evaluation.py` | Ground truth only here; evidence refs resolved against real files; per-arm infra records; complete-case explicitly flagged; seed replicates clustered by family×dataset; all pre-registered gates enforced; BLOCKED cannot become PASS. |
| `cli.py`, `tests/test_arr_rsi_01.py` | Staged pilot/reliability/full commands; 11 isolation, diagnosis, traceability and integrity tests. |

File-level diff vs P0 commit `44decf93`: 7 files, +1509/−1326 lines. No new agents, no new orchestration, no P1.

## 3. Test and staged execution results

| Stage | Result |
|---|---|
| Unit tests | **11/11 passed** |
| One-task pilot (1×4 arms) | All 4 arms ok; 0 infra; candidate `APPLIED` (`train.lr 0.001→0.0008`); shuffled distinct; evidence 8/8; verdict **BLOCKED** only because n=1 < 5 and units=1 < 5 |
| Five-task reliability (5×4 arms) | First run hit 1 engine CLI health-check timeout and 4 out-of-range placebo values; fixed by uniform timeout and range-safe placebo, then **3 explicit resume retries** (failed evidence archived, not deleted). Final: **0 infra failures, 5/5 complete-case, all APPLIED, shuffled 5/5 distinct**; verdict FAIL from statistics, not reliability |
| Full 20-task run (20×4 arms) | **80/80 real arms ok**, 0 infra, 20/20 complete-case, 20/20 APPLIED, 20/20 shuffled distinct, evidence 160/160. One random-shuffle arm initially applied an invalid `ema_momentum=1.245`; fixed range-aware and explicitly retried. Verdict **INCONCLUSIVE** |

All staged runs used `concurrency=1`. Original P0 evidence was never modified.

## 4. Full experiment results

**Verdict: INCONCLUSIVE.** Reason: pre-registered integrity/statistical gates not all met.

| Dimension | Mean |
|---|---|
| Scientific correctness | 0.60 (12/20 diagnoses matched sealed root cause) |
| OOD performance | 0.20 |
| Evidence completeness | 1.00 |
| Experimental validity | 1.00 |
| Reproducibility | 1.00 |
| Budget efficiency | 0.043842 |

**Failure rate by arm:** base 0/20, candidate 0/20, placebo 0/20, shuffled 0/20 (final state; all early failures archived, retried and reported).

**Intervention validity:** 20/20 candidate APPLIED against actual engine config; 0 INERT, 0 NOT_APPLIED; 20/20 shuffled diagnoses distinct from the candidate knob.

### Paired statistics (20 pairs, 5 independent units × 4 seed replicates)

| Contrast | Mean Δ | 95% CI | Holm p | Bootstrap CI | Clustered CI |
|---|---|---|---|---|---|
| Candidate vs Base | +0.016885 | [0.003884, 0.029886] | **0.013641** | [0.005855, 0.028876] | [0.010466, 0.028685] |
| Candidate vs Placebo | +0.004517 | [−0.002503, 0.011537] | 0.193932 | [−0.000998, 0.011727] | [−0.001882, 0.017311] |
| Candidate vs Shuffled | +0.000384 | [−0.010255, 0.011023] | 0.940619 | [−0.009114, 0.010123] | [−0.015505, 0.018464] |

- **MME gate:** pre-registered 0.02. Candidate-vs-Base Δ=+0.016885 is significant but **below MME**. MME was not tuned using P0 results.
- Because Candidate is not significantly better than Placebo or Shuffled, the improvement cannot be attributed to diagnosis quality.
- `BLOCKED` was not converted to `PASS`: the final INCONCLUSIVE verdict stands despite 0 infra failures.

**Evidence integrity:** all 160 diagnosis evidence refs resolved to recorded artifacts; GT verified by SHA from `ground_truth.sealed.json`, read only by the evaluator; no GT field detected in any agent-facing trajectory, diagnosis or intervention artifact; protocol SHA `2ccbd43020dead0ff712b1733503ef3580b23f08507ae80e287f46237f51acc5`.

## 5. Limitations and the real research question

1. Seed replicates are not independent tasks; inference is clustered to 5 independent units. This is now explicit in the protocol and manifest.
2. Only two already-verified frozen SciNet smoke configs were used; family labels represent research questions, not distinct data generators. Cross-family scientific generalization claims remain unsupported.
3. Candidate intervention values are derived from generic type-aware perturbation of the real baseline config, not from sealed GT values; diagnosis accuracy is measured by mapped root cause (12/20).
4. The shuffled diagnosis control was distinct but not superior to Candidate; the P0.1 result therefore **does not show that ARR's diagnosis adds information beyond a random diagnosis**.
5. This is engineering evidence with a pre-registered protocol, not scientific validation.

The next autonomous research question is therefore concrete: **why does evidence-grounded diagnosis fail to outperform a shuffled diagnosis under matched budgets?** That question is explicitly deferred; P1 is not started.

## 6. Artifacts

- Full run: `ARR_RSI_01_P0_1_FULL/` (protocol, sealed GT, task manifest, trajectories, diagnoses, interventions, results, statistics, report, 80 real arms).
- Pilot: `ARR_RSI_01_P0_1_PILOT/`
- Reliability: `ARR_RSI_01_P0_1_RELIABILITY/` (includes archived failed attempts).
- Root-level copy of this report and `ARR_RSI_01_P0_1_REPORT.md`.
