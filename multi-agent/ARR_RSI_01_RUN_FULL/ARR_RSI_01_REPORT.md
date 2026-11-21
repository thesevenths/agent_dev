# ARR-RSI-01 P0 — Experimental Report

## Verdict

**BLOCKED**

- blocked tasks/arms: ['arr_rsi_01::pendulum_dynamics::s42', 'arr_rsi_01::pendulum_dynamics::s44', 'arr_rsi_01::newtonian_dynamics::s43', 'arr_rsi_01::newtonian_dynamics::s45', 'arr_rsi_01::symbolic_regression::s43']

## Dimension means

| Dimension | Mean |
|---|---|
| scientific_correctness | 0.75 |
| ood_performance | 0.35 |
| evidence_completeness | 0.98889 |
| experimental_validity | 0.55 |
| reproducibility | 1.0 |
| budget_efficiency | 0.057599 |

## Gates

- `primary_significant`: True
- `primary_holm_p`: 0.02640034791030386
- `primary_mean_difference`: 0.01653110052631579
- `candidate_vs_placebo_delta`: 0.004742320526315789
- `candidate_vs_shuffled_delta`: -0.0016008454999999998
- `minimum_meaningful_effect`: 0.02
- `min_n_tasks`: 5

## Task outcomes

| task_id | cause correct | intervention correct | validity |
|---|---|---|---|
| arr_rsi_01::pendulum_dynamics::s42 | False | False | False |
| arr_rsi_01::pendulum_dynamics::s43 | True | True | True |
| arr_rsi_01::pendulum_dynamics::s44 | True | True | False |
| arr_rsi_01::pendulum_dynamics::s45 | True | True | True |
| arr_rsi_01::newtonian_dynamics::s42 | False | True | True |
| arr_rsi_01::newtonian_dynamics::s43 | False | True | False |
| arr_rsi_01::newtonian_dynamics::s44 | False | True | True |
| arr_rsi_01::newtonian_dynamics::s45 | False | True | False |
| arr_rsi_01::symbolic_regression::s42 | False | True | True |
| arr_rsi_01::symbolic_regression::s43 | False | True | False |
| arr_rsi_01::symbolic_regression::s44 | False | True | True |
| arr_rsi_01::symbolic_regression::s45 | False | True | False |
| arr_rsi_01::robot_dynamics::s42 | True | True | True |
| arr_rsi_01::robot_dynamics::s43 | True | True | True |
| arr_rsi_01::robot_dynamics::s44 | True | True | True |
| arr_rsi_01::robot_dynamics::s45 | True | True | False |
| arr_rsi_01::scientific_ml::s42 | True | True | False |
| arr_rsi_01::scientific_ml::s43 | True | True | False |
| arr_rsi_01::scientific_ml::s44 | True | True | True |
| arr_rsi_01::scientific_ml::s45 | True | True | True |

## Evidence boundary

- Ground truth was sealed separately and read only by the evaluator.
- Research/Diagnosis agents received an agent view with no GT and no hidden-test reference.
- Missing evidence was never converted into PASS.

## Scope statement

This verdict reports engineering evidence from the executed pilot only. It is NOT a scientific validation of the method.

Preregistration SHA: `ef972fd0be42b2594ae9a6b8585fcb89a6492c48f73cec69da6bcbe41eddc0e5`
Manifest SHA: `20b97f9d5c8b5e77b3606c056896cc196c96c1eda6102d431b1861865c513972`
Statistical analysis SHA: `09f776615483355e4b5f46573e3cd9b680acd40c57cd3a1dcebce09533d0abf0`

## Infrastructure note

- Full 20-task pilot executed all 80 real arms; 13 arms failed (engine CLI timeout / worker interruption / deterministic SciNet job-ID collision with the earlier single-task pilot).
- Root cause of the collision: SciNet++ derives job IDs from the research task ID; the fix adds a run-scoped nonce to the research task ID in `runner.build_research_task` for subsequent runs. The recorded pilot was NOT altered or re-run.
- Therefore the verdict is BLOCKED by pre-registered experimental-validity gates. The observed candidate-vs-base paired delta (+0.0165, Holm p=0.0264) is below the pre-registered minimum meaningful effect (0.02) and is NOT claimed as an improvement.
