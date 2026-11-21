# ARR-RSI-01 P0 — Experimental Report

## Verdict

**INCONCLUSIVE**

- n_tasks=1 < preregistered 5
- pre-registered evidence gates not all satisfied

## Dimension means

| Dimension | Mean |
|---|---|
| scientific_correctness | 1.0 |
| ood_performance | 0.0 |
| evidence_completeness | 0.5 |
| experimental_validity | 1.0 |
| reproducibility | 1.0 |
| budget_efficiency | n/a |

## Gates

- `primary_significant`: False
- `primary_holm_p`: None
- `primary_mean_difference`: None
- `candidate_vs_placebo_delta`: None
- `candidate_vs_shuffled_delta`: None
- `minimum_meaningful_effect`: 0.02
- `min_n_tasks`: 5

## Task outcomes

| task_id | cause correct | intervention correct | validity |
|---|---|---|---|
| arr_rsi_01::pendulum_dynamics::s42 | True | True | True |

## Evidence boundary

- Ground truth was sealed separately and read only by the evaluator.
- Research/Diagnosis agents received an agent view with no GT and no hidden-test reference.
- Missing evidence was never converted into PASS.

## Scope statement

This verdict reports engineering evidence from the executed pilot only. It is NOT a scientific validation of the method.

Preregistration SHA: `b1da84b682cf924d63c9378445b5dce6bd6d8b49a980299e99b563b8c603a64b`
Manifest SHA: `bf5805e2af23a58a898187f0179e29df95c44c70533ff5167d1a852da8630a49`
Statistical analysis SHA: `f9fe2bbc48b3069f46e08f56d8354e297b2328663ed4c31c3697c1d4526cb065`
