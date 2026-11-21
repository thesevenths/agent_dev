# ARR-RSI-01-P0.1 — pilot Report

## Verdict

**BLOCKED**

- n_tasks=1 < preregistered 5
- n_independent_units=1 < 5

## Dimension means

| Dimension | Mean |
|---|---|
| scientific_correctness | 1.0 |
| ood_performance | 0.0 |
| evidence_completeness | 1.0 |
| experimental_validity | 1.0 |
| reproducibility | 1.0 |
| budget_efficiency | 0.04283 |

## Gates

- `primary_significant`: False
- `primary_holm_p`: None
- `primary_mean_difference`: None
- `candidate_vs_placebo_delta`: None
- `candidate_vs_shuffled_delta`: None
- `minimum_meaningful_effect`: 0.02
- `n_independent_units`: 1
- `complete_case`: True
- `min_n_tasks`: 5

## Per-task integrity

| task | cause correct | intervention valid | applied | infra | complete-case |
|---|---|---|---|---|---|
| arr_rsi_01::pendulum_dynamics::s42 | True | True | APPLIED | 4 | True |

## Scope

Original P0 verdict remains BLOCKED and is untouched. This report covers only the P0.1 integrity repair. Engineering evidence only; NOT scientific validation. Original P0 BLOCKED verdict is preserved unchanged.

Protocol SHA: `87c5a3224e26491f89e4a41e315cda699725473310d17f1b91c117e14e61c34a`
Statistical SHA: `f9fe2bbc48b3069f46e08f56d8354e297b2328663ed4c31c3697c1d4526cb065`
