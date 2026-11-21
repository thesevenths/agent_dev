# ARR-RSI-01-P0.1 — full Report

## Verdict

**INCONCLUSIVE**

- pre-registered integrity/statistical gates not all met

## Dimension means

| Dimension | Mean |
|---|---|
| scientific_correctness | 0.6 |
| ood_performance | 0.2 |
| evidence_completeness | 1.0 |
| experimental_validity | 1.0 |
| reproducibility | 1.0 |
| budget_efficiency | 0.043842 |

## Gates

- `primary_significant`: True
- `primary_holm_p`: 0.013641318727203575
- `primary_mean_difference`: 0.0168849545
- `candidate_vs_placebo_delta`: 0.0045168435
- `candidate_vs_shuffled_delta`: 0.0003836984999999998
- `minimum_meaningful_effect`: 0.02
- `n_independent_units`: 5
- `complete_case`: True
- `min_n_tasks`: 5

## Per-task integrity

| task | cause correct | intervention valid | applied | infra | complete-case |
|---|---|---|---|---|---|
| arr_rsi_01::pendulum_dynamics::s42 | True | True | APPLIED | 4 | True |
| arr_rsi_01::pendulum_dynamics::s43 | True | True | APPLIED | 4 | True |
| arr_rsi_01::pendulum_dynamics::s44 | True | True | APPLIED | 4 | True |
| arr_rsi_01::pendulum_dynamics::s45 | True | True | APPLIED | 4 | True |
| arr_rsi_01::newtonian_dynamics::s42 | False | True | APPLIED | 4 | True |
| arr_rsi_01::newtonian_dynamics::s43 | False | True | APPLIED | 4 | True |
| arr_rsi_01::newtonian_dynamics::s44 | False | True | APPLIED | 4 | True |
| arr_rsi_01::newtonian_dynamics::s45 | False | True | APPLIED | 4 | True |
| arr_rsi_01::symbolic_regression::s42 | False | True | APPLIED | 4 | True |
| arr_rsi_01::symbolic_regression::s43 | False | True | APPLIED | 4 | True |
| arr_rsi_01::symbolic_regression::s44 | False | True | APPLIED | 4 | True |
| arr_rsi_01::symbolic_regression::s45 | False | True | APPLIED | 4 | True |
| arr_rsi_01::robot_dynamics::s42 | True | True | APPLIED | 4 | True |
| arr_rsi_01::robot_dynamics::s43 | True | True | APPLIED | 4 | True |
| arr_rsi_01::robot_dynamics::s44 | True | True | APPLIED | 4 | True |
| arr_rsi_01::robot_dynamics::s45 | True | True | APPLIED | 4 | True |
| arr_rsi_01::scientific_ml::s42 | True | True | APPLIED | 4 | True |
| arr_rsi_01::scientific_ml::s43 | True | True | APPLIED | 4 | True |
| arr_rsi_01::scientific_ml::s44 | True | True | APPLIED | 4 | True |
| arr_rsi_01::scientific_ml::s45 | True | True | APPLIED | 4 | True |

## Scope

Original P0 verdict remains BLOCKED and is untouched. This report covers only the P0.1 integrity repair. Engineering evidence only; NOT scientific validation. Original P0 BLOCKED verdict is preserved unchanged.

Protocol SHA: `2ccbd43020dead0ff712b1733503ef3580b23f08507ae80e287f46237f51acc5`
Statistical SHA: `b25b5bc80e55979f104fe2d3f05446cce1a93aa3913717da37ed85261bbb42d1`
