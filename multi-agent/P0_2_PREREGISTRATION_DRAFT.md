# ARR-RSI-01 P0.2 — Preregistration DRAFT (NOT FROZEN)

Experiment ID: `ARR-RSI-01-P0.2` · Branch: `feature/arr-rsi-01-p0-2-causal-validation`
Status: **DRAFT FOR REVIEW**. Nothing here is frozen. No experiment has been run.

## 0. Scope and prohibition

This draft is limited to the staged plan requested for P0.2-01. It does **not** authorize code changes to P0/P0.1, engine runs, or the full experiment. P0/P0.1 code, data, reports and verdicts are immutable (see `P0_2_BENCHMARK_AUDIT.md` §1).

The question P0.2 must answer, and nothing broader:

> Can ARR infer a root cause from real experimental evidence, choose a justified intervention, and show a reproducible, controlled improvement that is better than Random and Shuffled controls?

Five independent verdicts are mandatory: Engineering, Benchmark Integrity, Diagnosis, Intervention, Causal Improvement. An overall scientific claim is permitted only if all required gates pass. No automatic upgrade from INCONCLUSIVE to PASS.

## 1. Gates before any experiment (G0)

| ID | Gate | Fail action |
|---|---|---|
| G0.1 | P0/P0.1 hashes unchanged | STOP / BLOCKED |
| G0.2 | Independent Scientific Units defined by generator × data identity, not family name or seed | STOP / BLOCKED |
| G0.3 | At least 6 verified independent regimes available (for large-effect power) | STOP / BLOCKED |
| G0.4 | Ground-truth seal outside agent-visible root; capability-token access only | STOP / BLOCKED |
| G0.5 | Artifact resolver verifies path → SHA-256 → claim hash → run id | STOP / BLOCKED |
| G0.6 | This preregistration reviewed and frozen before first engine run | STOP / BLOCKED |

Current status of G0 from the audit: G0.1 PASS; G0.2 FAIL (P0.1 pseudo-independence); G0.3 FAIL (only pendulum verified); G0.4 PARTIAL; G0.5 FAIL; G0.6 pending. **Therefore no engine run is authorized now.**

## 2. Independent Scientific Units (proposed)

Unit = `(generator identity, frozen dataset version, regime parameter set)`. Seeds are repeated measurements nested within a unit and are aggregated to the unit level before inference.

Candidate units:
- U1 pendulum ID (`pendulum_smoke_v1`, verified)
- U2 pendulum OOD (`pendulum_smoke_ood_v1`, verified)
- U3 Newton generator (`configs/newton.yaml`, requires smoke verification)
- U4 double-pendulum generator (`configs/double_pendulum.yaml`, requires smoke verification)
- U5+ only from genuinely distinct verified generators or verified frozen regime changes; **not** from renaming a family or changing a seed.

Minimum for a PASS-capable full experiment: **6 verified units** (dz≥1.5) or **10 units** (dz≈1.0). If fewer, the experiment is scientifically underpowered and must be BLOCKED.

## 3. Stage A — Benchmark freeze + Diagnosis-Only Benchmark

This stage does **not** require full causal experiments and is the first thing to build after G0.

5 diagnosis controllers on held-out evidence cases:

| Controller | Access | Purpose |
|---|---|---|
| Evidence-Based | evidence package only (artifacts, hashes, metrics, budget) | ARR's real diagnosis |
| Random | same evidence package, random root cause from schema | chance baseline |
| Shuffled | another case's diagnosis, evidence refs from own case | attribution control |
| Rule-Based | deterministic rules from recorded metrics, no ground truth | non-LLM mechanistic baseline |
| Oracle | sealed GT (evaluator side only) | upper bound / benchmark sensitivity |

Diagnosis output schema (all required): `diagnosis_id`, `root_cause`, `evidence_refs`, `confidence`, `alternative_explanations`, `intervention_hypothesis`, `expected_effect`, `falsification_condition`, `abstain`.

Pre-registered diagnosis thresholds (DRAFT, to be frozen after review):
- Root-cause accuracy (Evidence-Based): ≥ 0.75.
- Evidence-Based accuracy advantage over Random and Shuffled: ≥ 0.20 absolute, Holm-adjusted p ≤ 0.05 at the **unit** level.
- Evidence precision: ≥ 0.80 of cited refs pass artifact-level hash + claim verification.
- Confidence calibration: ECE ≤ 0.15; abstention rate reported and high-confidence subset accuracy ≥ 0.85.
- Oracle accuracy must be 1.00; otherwise the benchmark is invalid → BLOCKED.

## 4. Stage B — Intervention Registry and Oracle sensitivity

No generic parameter multipliers are allowed. Every candidate intervention must be registered with:

- mechanism hypothesis tied to a diagnosed cause;
- allowed parameter key, legal value/range, and source (real baseline config);
- before-config, after-config, expected direction;
- falsification condition;
- applied config and diff hash verified against the engine artifact;
- provenance: diagnosis_id + evidence_refs.

The Oracle arm uses a preregistered intervention known to improve the metric on that independent unit. If Oracle cannot produce a measurable improvement under matched budgets, the benchmark lacks sensitivity → **BLOCKED**, not FAIL of the agent.

Stage B gates (DRAFT): intervention applied rate = 100%; config-vs-result consistency = 100%; Oracle improvement ≥ MME with paired evidence; Random and Shuffled controls matched on knob count, budget and runtime.

## 5. Stage C — Controlled experiment (only after A and B gates pass)

Arms: Base, Candidate (evidence-based), Random, Shuffled, Oracle.

Freeze before execution: engine version, runtime version, configs, frozen datasets, seeds, tool budget, metric, evaluator, preregistration SHA, ground-truth SHA.

Execution: `concurrency=1`; completed arms never silently rerun; failed arms explicitly retried with archived evidence; per-arm infrastructure failures retained.

Statistical protocol (DRAFT):
- Primary endpoint: paired improvement on the unit-level primary metric (normalised direction so positive = better).
- Unit of analysis: independent scientific unit; seed replicates aggregated within unit.
- Paired t / CI, permutation p, clustered bootstrap by unit, Holm correction across the three agent-vs-control contrasts (Candidate–Random, Candidate–Shuffled, Candidate–Base).
- Effect size (dz), 95% CI, full contrast table; no post-hoc metric or subgroup switching.
- MME (DRAFT): 0.02 on the normalised primary metric, inherited from P0.1 and **not** tuned on results.
- Power: computed before freezing; underpowered designs are BLOCKED.

## 6. Five verdict rules (DRAFT)

| Layer | PASS requires | Otherwise |
|---|---|---|
| Engineering | all unit/integration tests pass; real engine; 100% applied; artifact integrity; budget compliance; reproducible | FAIL or BLOCKED if infra prevents judgment |
| Benchmark Integrity | units real; generator independence; GT isolation; leakage=0; benchmark+prereg frozen | BLOCKED |
| Diagnosis | schema, accuracy, evidence precision, calibration, abstention gates; beats Random and Shuffled | FAIL or INCONCLUSIVE |
| Intervention | registry verified, applied rate 100%, Oracle sensitive, controls matched | FAIL or BLOCKED if Oracle invalid |
| Causal Improvement | Candidate beats Random, Shuffled and Base with MME, corrected significance, adequate power | FAIL / INCONCLUSIVE / BLOCKED |

Overall scientific claim requires all five PASS. No layer may be inferred from engineering success.

## 7. Planned reproducible commands (NOT TO RUN YET)

After G0 and frozen preregistration:

```bash
# benchmark audit + freeze
python -m research_runtime.arr_rsi_02.cli audit --out ARR_RSI_01_P0_2_AUDIT
python -m research_runtime.arr_rsi_02.cli freeze-prereg --draft P0_2_PREREGISTRATION_DRAFT.md

# staged gates
python -m research_runtime.arr_rsi_02.cli diagnose-benchmark --stage pilot
python -m research_runtime.arr_rsi_02.cli diagnose-benchmark --stage reliability
python -m research_runtime.arr_rsi_02.cli diagnose-benchmark --stage full
# only if all Diagnosis gates PASS:
python -m research_runtime.arr_rsi_02.cli controlled-experiment --stage pilot
python -m research_runtime.arr_rsi_02.cli controlled-experiment --stage reliability
python -m research_runtime.arr_rsi_02.cli controlled-experiment --stage full
```

Module names and flags are placeholders pending implementation; no code is authorized by this draft.

## 8. Stop rules

Stop immediately and report BLOCKED if:
- any real independence assumption fails;
- GT is reachable from any agent-facing path;
- any evidence claim cannot be hash-verified;
- Oracle fails to validate benchmark sensitivity;
- power is insufficient at the frozen design;
- an infrastructure failure prevents a complete matched arm set.

Report FAIL if the experiment is valid and powered but ARR's Diagnosis or Intervention does not meet the frozen thresholds.

Report INCONCLUSIVE only when the experiment is valid but the pre-registered evidence is insufficient to decide; never upgrade it to PASS.

## 9. What this draft deliberately does not claim

ARR is not described here as an Autonomous Scientific Research System. P0.2 is a falsifiable capability test. Until Evidence → Diagnosis → Intervention → Controlled Validation → Measurable Improvement is demonstrated under the gates above, the correct description is "automated experiment runner with a diagnosis component under evaluation".
